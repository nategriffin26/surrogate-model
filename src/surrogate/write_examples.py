"""Write the tool's bundled examples with the browser engine itself.

    uv run surrogate examples                # web/examples/run_1, run_500
    uv run surrogate examples --development  # also local/dev/examples/run_1_no_cfd, run_1_stretched

Each surface file is analysed by the tool's engine in headless Chrome, exactly as a visitor's browser
does it, and its example is written once every polygon is predicted, with the numbers summed over all
of them (the engine's `exact` event), the working mesh with its fields, and the lookups `explain` needs.

Needs Google Chrome with WebGPU (set CHROME to its executable when it is not at the macOS default
path), the DrivAerML files `data/raw/holdout/run_1/boundary_1.vtp` and `run_500/boundary_500.vtp`, and
`web/model/` exported from the current checkpoint (`uv run surrogate export`). Run it again
after every retraining: the tests fail while an example was computed with other weights.

`--development` also writes the surfaces behind the two development examples, run 1 without its CFD
pressure and run 1 stretched 25% in x (both with every solver array removed), to
`local/dev/run_1_no_cfd.vtp` and `local/dev/run_1_stretched.vtp`, ready to drop into the tool.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

import pyvista as pv

from surrogate.server import DEFAULT_MODEL, WEB_ROOT, AssistantConfig, ToolRequestHandler

REPO = Path(__file__).resolve().parents[2]
CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
SECONDS_PER_FILE = 300

BUNDLED = {
    "run_1": REPO / "data/raw/holdout/run_1/boundary_1.vtp",
    "run_500": REPO / "data/raw/holdout/run_500/boundary_500.vtp",
}
DEVELOPMENT = {
    "run_1_no_cfd": REPO / "local/dev/run_1_no_cfd.vtp",
    "run_1_stretched": REPO / "local/dev/run_1_stretched.vtp",
}

# The page Chrome opens: it analyses each source in turn and posts the example's two files back.
WRITER_PAGE = """<!doctype html><meta charset="utf-8"><script type="module">
const report = (path, body) => fetch(path, { method: 'POST', body });
try {
  const { createEngine } = await import('/engine/engine.js');
  const { packResult } = await import('/engine/result.js');
  const engine = createEngine();
  const device = await engine.device;
  if (!device.startsWith('WebGPU')) throw new Error(`Chrome offers no WebGPU here (${device}), and an example needs every polygon predicted.`);
  for (const { name, file } of JOBS) {
    const blob = await fetch(`/sources/${name}`).then((response) => response.blob());
    let exactArrived;
    const exact = new Promise((resolve) => { exactArrived = resolve; });
    const result = await engine.analyze(new File([blob], file), { exact: exactArrived });
    const { json, block } = await packResult({ ...result, ...(await exact) });
    await report(`/written/${name}.json`, json);
    await report(`/written/${name}.bin`, block);
  }
  await report('/finished', '');
} catch (error) {
  await report('/failed', error.message);
}
</script>
"""


class ExampleWriter(ToolRequestHandler):
    """The tool's files, the writer page, the source surfaces, and a place to post the examples to."""

    def __init__(self, *args, jobs: dict[str, tuple[Path, Path]], outcome: dict, done: threading.Event, **kwargs):
        self.jobs, self.outcome, self.done = jobs, outcome, done
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        if self.path == "/writer.html":
            page = [{"name": name, "file": source.name} for name, (source, _) in self.jobs.items()]
            self.send_body(WRITER_PAGE.replace("JOBS", json.dumps(page)).encode(), "text/html")
        elif self.path.startswith("/sources/"):
            source, _ = self.jobs[self.path.removeprefix("/sources/")]
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(source.stat().st_size))
            self.end_headers()
            with source.open("rb") as stream:
                shutil.copyfileobj(stream, self.wfile, 1 << 22)
        else:
            super().do_GET()

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers["Content-Length"]))
        if self.path.startswith("/written/"):
            filename = self.path.removeprefix("/written/")
            _, out_dir = self.jobs[filename.rsplit(".", 1)[0]]
            (out_dir / filename).write_bytes(body)
            print(f"Wrote {out_dir / filename} ({len(body) / 1e6:.1f} MB)", flush=True)
        else:
            self.outcome[self.path.removeprefix("/")] = body.decode()
            self.done.set()
        self.send_body(b"", "text/plain")

    def send_body(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def write_development_surfaces() -> None:
    """Run 1 without CFD, and stretched 25% in x, each with every solver array removed."""
    for name, path in DEVELOPMENT.items():
        surface = pv.read(BUNDLED["run_1"])
        surface.clear_data()
        if name == "run_1_stretched":
            surface.points[:, 0] *= 1.25
        path.parent.mkdir(parents=True, exist_ok=True)
        surface.save(path)
        print(f"Wrote {path}", flush=True)


def write_examples(jobs: dict[str, tuple[Path, Path]]) -> None:
    """Analyse each source in headless Chrome and write its example, `name` -> (source, out_dir)."""
    for _, out_dir in jobs.values():
        out_dir.mkdir(parents=True, exist_ok=True)
    outcome: dict[str, str] = {}
    done = threading.Event()
    # Writing examples needs no assistant, so its route stays off whatever key this machine holds.
    no_assistant = AssistantConfig(key=None, model=DEFAULT_MODEL, enabled=False)
    handler = partial(ExampleWriter, jobs=jobs, outcome=outcome, done=done, directory=str(WEB_ROOT), assistant=no_assistant)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/writer.html"
    with tempfile.TemporaryDirectory() as profile:
        chrome = subprocess.Popen(
            [CHROME, "--headless", "--enable-unsafe-webgpu", f"--user-data-dir={profile}", "--no-first-run", url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            finished = done.wait(SECONDS_PER_FILE * len(jobs))
        finally:
            chrome.terminate()
            chrome.wait()
            server.shutdown()
    if not finished:
        raise SystemExit(f"Chrome did not finish the examples within {SECONDS_PER_FILE * len(jobs)} s.")
    if "failed" in outcome:
        raise SystemExit(f"The engine failed: {outcome['failed']}")


def write_all_examples(development: bool) -> None:
    """The bundled examples, and with `development` the two development examples too."""
    jobs = {name: (source, WEB_ROOT / "examples") for name, source in BUNDLED.items()}
    if development:
        write_development_surfaces()
        jobs |= {name: (source, REPO / "local/dev/examples") for name, source in DEVELOPMENT.items()}
    write_examples(jobs)
