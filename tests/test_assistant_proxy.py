"""The assistant route of the local server (`surrogate.server`), against a fake model service on this machine that
replays a stream recorded from OpenRouter (`fixtures/assistant_tool_call.sse`: anthropic/claude-opus-5.5 streaming
its thinking as text, then calling a tool); the public site's Worker for the same route (`deploy/assistant-worker/`)
and the panel's Markdown, whose tests run under Deno; and the contract the two proxies share."""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from surrogate import server as local_proxy
from surrogate.server import (
    KERNEL_POLICY,
    KERNEL_WORKER,
    MAX_OUTPUT_TOKENS,
    MAX_REQUEST_BYTES,
    STREAM_TYPE,
    WEB_ROOT,
    AssistantConfig,
    assistant_config,
    make_server,
)

DENO = shutil.which("deno")
TESTS = Path(__file__).parent
WORKER = TESTS.parent / "deploy" / "assistant-worker" / "assistant.js"
RECORDED = (TESTS / "fixtures" / "assistant_tool_call.sse").read_bytes()
KEY = "sk-test-key"
MODEL = "meta/muse-spark-1.3-contributor"
CONVERSATION = {
    "messages": [{"role": "user", "content": "What is the lowest Cp on case 1?"}],
    "tools": [{"type": "function", "function": {"name": "read_section", "parameters": {"type": "object"}}}],
}


class FakeModelService:
    """Records each request and answers with `reply`: (status, body bytes, content type). With `hold` set, the
    stream stops after its first event until `hold` is released, so a test can see what arrived before the end."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.reply = (200, RECORDED, "text/event-stream")
        self.hold: threading.Event | None = None
        service = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                service.requests.append({"headers": dict(self.headers), "body": body})
                status, payload, content_type = service.reply
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.end_headers()
                if service.hold is None:
                    self.wfile.write(payload)
                    return
                first_event_end = payload.index(b"\n\n", payload.index(b"data:")) + 2
                self.wfile.write(payload[:first_event_end])
                self.wfile.flush()
                service.hold.wait(10)
                self.wfile.write(payload[first_event_end:])

            def log_message(self, *args: object) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/api/v1/chat/completions"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


@pytest.fixture
def service() -> Iterator[FakeModelService]:
    fake = FakeModelService()
    yield fake
    fake.server.shutdown()


def serve(config: AssistantConfig, root: Path = WEB_ROOT) -> tuple[str, ThreadingHTTPServer]:
    server = make_server(0, root=root, assistant=config)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}", server


@pytest.fixture
def tool(service: FakeModelService) -> Iterator[str]:
    base, server = serve(AssistantConfig(key=KEY, model=MODEL, enabled=True, upstream=service.url))
    yield base
    server.shutdown()


def post(base: str, body: bytes) -> tuple[int, dict, bytes]:
    request = urllib.request.Request(f"{base}/api/assistant", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()


def get(base: str, path: str) -> tuple[int, dict, bytes]:
    try:
        with urllib.request.urlopen(f"{base}{path}") as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()


def test_the_reply_stream_passes_through_unchanged(tool: str, service: FakeModelService) -> None:
    """Catches the proxy re-encoding, dropping or reordering the model's events, its streamed thinking included."""
    assert b'"type":"reasoning.text"' in RECORDED and b'"signature"' in RECORDED
    status, headers, body = post(tool, json.dumps(CONVERSATION).encode())
    assert status == 200
    assert headers["Content-Type"] == STREAM_TYPE
    assert body == RECORDED


def test_the_server_sets_model_key_and_limits_and_the_page_cannot(tool: str, service: FakeModelService) -> None:
    """Catches the page choosing the model, the output limit or reasoning, and the key not being added."""
    page_body = {
        **CONVERSATION,
        "model": "openai/some-expensive-model",
        "max_tokens": 1_000_000,
        "reasoning": {"effort": "max"},
    }
    post(tool, json.dumps(page_body).encode())
    sent = service.requests[0]
    assert sent["headers"]["Authorization"] == f"Bearer {KEY}"
    assert sent["body"] == {
        "model": MODEL,
        "messages": CONVERSATION["messages"],
        "tools": CONVERSATION["tools"],
        "stream": True,
        "reasoning": {"effort": "high"},
        "max_tokens": MAX_OUTPUT_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }


def test_events_reach_the_page_as_they_arrive(tool: str, service: FakeModelService) -> None:
    """Catches the proxy buffering the whole reply before sending any of it."""
    service.hold = threading.Event()
    request = urllib.request.Request(f"{tool}/api/assistant", data=json.dumps(CONVERSATION).encode())
    with urllib.request.urlopen(request) as response:
        early = b""
        while b"data:" not in early:
            early += response.read1(65536)
        assert b"[DONE]" not in early
        service.hold.set()
        rest = response.read()
    assert early + rest == RECORDED


def test_a_refusal_by_the_model_service_reaches_the_page_as_its_error(tool: str, service: FakeModelService) -> None:
    """Catches an upstream refusal (spent key, bad request) turning into a stream or a generic failure."""
    refusal = {"error": {"code": 402, "message": "This key has used its credit limit."}}
    service.reply = (402, json.dumps(refusal).encode(), "application/json")
    status, headers, body = post(tool, json.dumps(CONVERSATION).encode())
    assert status == 402
    assert headers["Content-Type"] == "application/json"
    assert json.loads(body) == refusal


def test_an_unreachable_model_service_is_a_502_with_a_message() -> None:
    base, server = serve(AssistantConfig(key=KEY, model=MODEL, enabled=True, upstream="http://127.0.0.1:9/nothing"))
    status, _, body = post(base, json.dumps(CONVERSATION).encode())
    server.shutdown()
    assert status == 502
    assert "could not be reached" in json.loads(body)["error"]["message"]


def test_status_names_the_model_and_never_the_key(tool: str) -> None:
    status, _, body = get(tool, "/api/assistant")
    assert status == 200
    assert json.loads(body) == {"enabled": True, "model": MODEL}
    assert KEY.encode() not in body


@pytest.mark.parametrize(
    ("environ", "secrets_text"),
    [
        ({}, None),
        ({"OPENROUTER_API_KEY": KEY, "ASSISTANT_ENABLED": "0"}, None),
        ({"ASSISTANT_ENABLED": "0"}, f"OPENROUTER_API_KEY={KEY}\n"),
    ],
    ids=["no key", "switched off", "switched off with the key in the secrets file"],
)
def test_without_a_key_or_switched_off_the_route_answers_503(
    environ: dict, secrets_text: str | None, service: FakeModelService, tmp_path: Path
) -> None:
    secrets = tmp_path / "assistant.env"
    if secrets_text:
        secrets.write_text(secrets_text)
    config = assistant_config(environ, secrets)
    base, server = serve(dataclasses.replace(config, upstream=service.url))
    get_status, _, get_body = get(base, "/api/assistant")
    post_status, _, post_body = post(base, json.dumps(CONVERSATION).encode())
    server.shutdown()
    assert get_status == post_status == 503
    assert json.loads(get_body)["enabled"] is False
    assert json.loads(post_body)["error"]["message"] == "The assistant is not available in this copy."
    assert service.requests == []


def test_the_key_and_model_come_from_the_environment_before_the_secrets_file(tmp_path: Path) -> None:
    secrets = tmp_path / "assistant.env"
    secrets.write_text(f"# the test key\nOPENROUTER_API_KEY={KEY}\nASSISTANT_MODEL=from/file\n")
    assert assistant_config({}, secrets) == AssistantConfig(key=KEY, model="from/file", enabled=True)
    assert assistant_config({"ASSISTANT_MODEL": MODEL}, secrets).model == MODEL
    assert assistant_config({}, tmp_path / "missing.env").model == "anthropic/claude-opus-5.5"


@pytest.mark.parametrize(
    ("body", "status"),
    [
        (b"not json", 400),
        (b"[]", 400),
        (json.dumps({"messages": [], "tools": []}).encode(), 400),
        (json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode(), 400),
        (json.dumps({**CONVERSATION, "padding": "x" * MAX_REQUEST_BYTES}).encode(), 413),
    ],
    ids=["not json", "not an object", "no messages", "no tools", "too large"],
)
def test_malformed_or_oversized_requests_never_reach_the_model(
    tool: str, service: FakeModelService, body: bytes, status: int
) -> None:
    got, _, reply = post(tool, body)
    assert got == status
    assert json.loads(reply)["error"]["message"]
    assert service.requests == []


def test_the_kernel_worker_is_served_with_its_network_policy(tmp_path: Path) -> None:
    """Catches the header that keeps the Python kernel off the network going missing."""
    worker = tmp_path / KERNEL_WORKER.lstrip("/")
    worker.parent.mkdir(parents=True)
    worker.write_text("// kernel")
    (tmp_path / "index.html").write_text("<!doctype html>")
    base, server = serve(AssistantConfig(key=None, model=MODEL, enabled=False), root=tmp_path)
    _, worker_headers, _ = get(base, KERNEL_WORKER)
    _, page_headers, _ = get(base, "/index.html")
    server.shutdown()
    assert worker_headers["Content-Security-Policy"] == KERNEL_POLICY
    assert "connect-src https://cdn.jsdelivr.net/pyodide/" in KERNEL_POLICY
    assert "Content-Security-Policy" not in page_headers


def test_a_page_load_arriving_all_at_once_is_served_whole(tmp_path: Path) -> None:
    """Catches the server dropping connections when the tool's modules are requested together, as a proxy in
    front of it does: with the standard library's queue of five waiting connections the page came up blank."""
    (tmp_path / "index.html").write_text("<!doctype html>")
    base, server = serve(AssistantConfig(key=None, model=MODEL, enabled=False), root=tmp_path)
    with ThreadPoolExecutor(max_workers=60) as pool:
        statuses = list(pool.map(lambda _: get_or_error(base, "/index.html"), range(60)))
    server.shutdown()
    assert statuses == [200] * 60


def get_or_error(base: str, path: str) -> int | str:
    try:
        return get(base, path)[0]
    except OSError as error:
        return type(error).__name__


def run_deno_tests(script: str) -> None:
    command = [DENO, "test", "--allow-read", str(TESTS / script)]
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.skipif(DENO is None, reason="the panel's Markdown tests run under Deno")
def test_the_panel_reads_reply_tables() -> None:
    """Runs tests/assistant_markdown.test.js: an escaped pipe inside a table cell stays in its cell."""
    run_deno_tests("assistant_markdown.test.js")


@pytest.mark.skipif(DENO is None, reason="the Worker's tests run under Deno")
def test_the_public_worker() -> None:
    """Runs tests/assistant_worker.test.js: requests over the Worker's limits get 429, and it keeps this contract."""
    run_deno_tests("assistant_worker.test.js")


SHARED = [
    "UPSTREAM",
    "DEFAULT_MODEL",
    "REASONING_EFFORT",
    "MAX_OUTPUT_TOKENS",
    "MAX_REQUEST_BYTES",
    "UNAVAILABLE",
    "STREAM_TYPE",
    "PYODIDE_CDN",
    "KERNEL_POLICY",
]


@pytest.mark.skipif(DENO is None, reason="the Worker's module is read under Deno")
def test_the_two_proxies_share_their_contract() -> None:
    """Catches the model, a limit, a message or the kernel's policy changed in one proxy and not the other."""
    names = json.dumps(SHARED)
    script = (
        f"const worker = await import({json.dumps(WORKER.as_uri())});"
        f"console.log(JSON.stringify(Object.fromEntries({names}.map((name) => [name, worker[name] ?? null]))));"
    )
    done = subprocess.run([DENO, "eval", script], capture_output=True, text=True, check=True)
    assert json.loads(done.stdout) == {name: getattr(local_proxy, name) for name in SHARED}
