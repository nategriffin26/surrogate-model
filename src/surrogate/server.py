"""Serve the browser tool in `web/` on this machine: `uv run surrogate app`.

Everything the tool computes runs in the visitor's browser, so this is a static file server with one route of
its own: `/api/assistant`, which forwards the assistant's conversation to the model and streams the reply back,
adding the key, which never reaches the browser. The public site serves the same route from a Cloudflare Worker
(`deploy/assistant-worker/`); the two share this contract:

- `GET /api/assistant` answers `{"enabled": true, "model": "<id>"}`, or 503 with `enabled` false.
- `POST /api/assistant` takes `{"messages": [...], "tools": [...]}` (OpenAI chat format) and answers with the
  model's server-sent events, passed through unchanged. The model, reasoning effort, output limit and prompt
  caching are set here; anything else in the body is ignored.
- Every refusal or failure is `{"error": {"message": "..."}}` with its status: 400 a malformed body, 413 too
  large, 503 switched off, and the model service's own status and body when it refuses.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = ROOT / "web"
SECRETS = ROOT / "local" / "secrets" / "assistant.env"

ASSISTANT_ROUTE = "/api/assistant"
UPSTREAM = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-opus-5.5"
REASONING_EFFORT = "high"
MAX_OUTPUT_TOKENS = 16_000
MAX_REQUEST_BYTES = 512 * 1024
UNAVAILABLE = "The assistant is not available in this copy."
# Server-sent events are UTF-8; saying so keeps every client from guessing.
STREAM_TYPE = "text/event-stream; charset=utf-8"

# The kernel's worker may load Pyodide from its CDN and nothing else: no request to any other address, this
# server included, and no scripts or workers from anywhere else. The browser enforces it, whatever code runs.
KERNEL_WORKER = "/assistant/kernel-worker.js"
PYODIDE_CDN = "https://cdn.jsdelivr.net/pyodide/"
KERNEL_POLICY = f"default-src 'none'; script-src {PYODIDE_CDN} 'wasm-unsafe-eval'; connect-src {PYODIDE_CDN}"


@dataclass(frozen=True)
class AssistantConfig:
    """How the assistant route reaches the model. Off without a key, or with ASSISTANT_ENABLED=0."""

    key: str | None
    model: str
    enabled: bool
    upstream: str = UPSTREAM


def assistant_config(environ: Mapping[str, str] = os.environ, secrets: Path = SECRETS) -> AssistantConfig:
    """Read OPENROUTER_API_KEY, ASSISTANT_MODEL and ASSISTANT_ENABLED from the environment, then the secrets file."""
    values = dict(_read_env_file(secrets)) | dict(environ)
    key = values.get("OPENROUTER_API_KEY") or None
    return AssistantConfig(
        key=key,
        model=values.get("ASSISTANT_MODEL") or DEFAULT_MODEL,
        enabled=key is not None and values.get("ASSISTANT_ENABLED") != "0",
    )


def _read_env_file(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        return []
    pairs = (line.split("=", 1) for line in path.read_text().splitlines() if "=" in line and not line.startswith("#"))
    return [(name.strip(), value.strip()) for name, value in pairs]


class ToolRequestHandler(SimpleHTTPRequestHandler):
    """Static files with the module types browsers insist on, never cached stale, and the assistant route."""

    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".json": "application/json",
        ".bin": "application/octet-stream",
        ".py": "text/x-python",
    }

    def __init__(self, *args: object, assistant: AssistantConfig, **kwargs: object) -> None:
        self.assistant = assistant
        super().__init__(*args, **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-cache")
        if self.path.split("?")[0] == KERNEL_WORKER:
            self.send_header("Content-Security-Policy", KERNEL_POLICY)
        super().end_headers()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - the base class names it
        pass

    def do_GET(self) -> None:
        if self.path.split("?")[0] != ASSISTANT_ROUTE:
            return super().do_GET()
        if not self.assistant.enabled:
            return self._send_json(503, {"enabled": False, "model": None, "error": {"message": UNAVAILABLE}})
        return self._send_json(200, {"enabled": True, "model": self.assistant.model})

    def do_POST(self) -> None:
        if self.path.split("?")[0] != ASSISTANT_ROUTE:
            return self.send_error(404)
        if not self.assistant.enabled:
            return self._send_error(503, UNAVAILABLE)
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_REQUEST_BYTES:
            return self._send_error(413, "This conversation is too long to send. Start a new one.")
        try:
            body = json.loads(self.rfile.read(length))
        except ValueError:
            return self._send_error(400, "The request body is not JSON.")
        messages, tools = body.get("messages"), body.get("tools")
        if not isinstance(messages, list) or not messages or not isinstance(tools, list):
            return self._send_error(400, "Send {messages, tools}: messages a non-empty list, tools a list.")
        self._forward(messages, tools)

    def _forward(self, messages: list, tools: list) -> None:
        upstream_body = {
            "model": self.assistant.model,
            "messages": messages,
            "tools": tools,
            "stream": True,
            "reasoning": {"effort": REASONING_EFFORT},
            "max_tokens": MAX_OUTPUT_TOKENS,
            # The conversation's prefix is cached and moves forward each turn, so a turn pays full price only for its new part.
            "cache_control": {"type": "ephemeral"},
        }
        request = urllib.request.Request(
            self.assistant.upstream,
            data=json.dumps(upstream_body).encode(),
            headers={"Authorization": f"Bearer {self.assistant.key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            upstream = urllib.request.urlopen(request)
        except urllib.error.HTTPError as error:
            return self._send(error.code, error.read(), error.headers.get("Content-Type") or "application/json")
        except urllib.error.URLError as error:
            return self._send_error(502, f"The model service could not be reached ({error.reason}).")
        with upstream:
            self.send_response(200)
            self.send_header("Content-Type", STREAM_TYPE)
            self.end_headers()
            try:
                while chunk := upstream.read1(65536):
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # The engineer stopped the run; closing the upstream connection stops the model.

    def _send_error(self, status: int, message: str) -> None:
        self._send_json(status, {"error": {"message": message}})

    def _send_json(self, status: int, value: object) -> None:
        self._send(status, json.dumps(value).encode(), "application/json")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ToolServer(ThreadingHTTPServer):
    """The page asks for about forty modules at once. The standard library lets five connections wait and
    drops the rest, which a proxy in front of the server turns into failed modules and a blank page."""

    request_queue_size = 128


def make_server(port: int, root: Path = WEB_ROOT, assistant: AssistantConfig | None = None) -> ToolServer:
    """Bind the tool's server to 127.0.0.1; port 0 picks a free one."""
    handler = partial(ToolRequestHandler, directory=str(root), assistant=assistant or assistant_config())
    return ToolServer(("127.0.0.1", port), handler)
