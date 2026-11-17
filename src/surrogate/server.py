"""Serve the browser tool in `web/` on this machine: `uv run surrogate-app`.

Everything the tool computes runs in the visitor's browser, so this is a plain static file server.
"""

from __future__ import annotations

import argparse
import errno
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB_ROOT = Path(__file__).resolve().parents[2] / "web"


class ToolRequestHandler(SimpleHTTPRequestHandler):
    """Static files with the module types browsers insist on, never cached stale."""

    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".json": "application/json",
        ".bin": "application/octet-stream",
    }

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - the base class names it
        pass


def make_server(port: int, root: Path = WEB_ROOT) -> ThreadingHTTPServer:
    """Bind the tool's server to 127.0.0.1; port 0 picks a free one."""
    return ThreadingHTTPServer(("127.0.0.1", port), partial(ToolRequestHandler, directory=str(root)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the surface-pressure tool on this machine.")
    parser.add_argument("--port", type=int, default=8740, help="port to listen on (default 8740)")
    args = parser.parse_args()
    try:
        server = make_server(args.port)
    except OSError as error:
        if error.errno != errno.EADDRINUSE:
            raise
        parser.exit(1, f"Port {args.port} is already in use. Choose another with --port, for example --port {args.port + 1}.\n")
    print(f"Surface-pressure tool: http://127.0.0.1:{server.server_address[1]}/  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
