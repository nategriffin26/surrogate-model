"""Write the tool's bundled examples, `web/examples/run_1` and `run_500`, with the browser engine itself:
`uv run surrogate examples`.

Each surface file is analysed by the tool's engine in headless Chrome, exactly as a visitor's browser
does it, and its example is written once every polygon is predicted, with the numbers summed over all
of them (the engine's `exact` event), the working mesh with its fields, and the lookups `explain` needs.

Needs Google Chrome with WebGPU (set CHROME to its executable when it is not at the macOS default
path), the DrivAerML files `data/raw/holdout/run_1/boundary_1.vtp` and `run_500/boundary_500.vtp`, and
`web/model/` exported from the current checkpoint (`uv run surrogate export`). Run it again
after every retraining: the tests fail while an example was computed with other weights.
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

from surrogate.dataset import HOLDOUT_RUNS, surface_path
from surrogate.progress import Progress, Step
from surrogate.server import DEFAULT_MODEL, REPO, WEB_ROOT, AssistantConfig, ToolRequestHandler

CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
SECONDS_PER_FILE = 300
EXAMPLES = WEB_ROOT / "examples"
BUNDLED = {f"run_{run}": surface_path(REPO / "data/raw", run) for run in HOLDOUT_RUNS}
# Both examples, analysed in Chrome in about 40 s on MEASURED_ON.
EXAMPLES_STEPS = [Step("Analysing in headless Chrome", len(BUNDLED), "car", 40)]

# The page Chrome opens: it analyses each source in turn and posts the example's two files back.
WRITER_PAGE = """<!doctype html><meta charset="utf-8"><script type="module">
const report = (path, body) => fetch(path, { method: 'POST', body });
try {
  const { createEngine } = await import('/engine/engine.js');
  const { packResult } = await import('/engine/result.js');
  const engine = createEngine();
  const device = await engine.device;
  if (!device.startsWith('WebGPU')) {
    throw new Error(`Chrome offers no WebGPU here (${device}), and an example needs every polygon predicted.`);
  }
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

    def __init__(
        self, *args: object, outcome: dict[str, str], done: threading.Event, progress: Progress, **kwargs: object
    ) -> None:
        self.outcome, self.done, self.progress = outcome, done, progress
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        if self.path == "/writer.html":
            page = [{"name": name, "file": source.name} for name, source in BUNDLED.items()]
            self._send(200, WRITER_PAGE.replace("JOBS", json.dumps(page)).encode(), "text/html")
        elif self.path.startswith("/sources/"):
            source = BUNDLED[self.path.removeprefix("/sources/")]
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
            path = EXAMPLES / self.path.removeprefix("/written/")
            path.write_bytes(body)
            self.progress.note(f"Wrote {path.relative_to(REPO)} ({len(body) / 1e6:.1f} MB)")
            if path.suffix == ".bin":  # an example's second and last file
                self.progress.advance()
        else:
            self.outcome[self.path.removeprefix("/")] = body.decode()
            self.done.set()
        self._send(200, b"", "text/plain")


def write_all_examples(progress: Progress) -> None:
    """Analyse each bundled source in headless Chrome and write its example to `web/examples/`, reporting
    EXAMPLES_STEPS on `progress`."""
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    outcome: dict[str, str] = {}
    done = threading.Event()
    # Writing examples needs no assistant, so its route stays off whatever key this machine holds.
    no_assistant = AssistantConfig(key=None, model=DEFAULT_MODEL, enabled=False)
    handler = partial(
        ExampleWriter, outcome=outcome, done=done, progress=progress, directory=str(WEB_ROOT), assistant=no_assistant
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/writer.html"
    timeout = SECONDS_PER_FILE * len(BUNDLED)
    progress.step("Analysing in headless Chrome")
    with tempfile.TemporaryDirectory() as profile:
        chrome = subprocess.Popen(
            [CHROME, "--headless", "--enable-unsafe-webgpu", f"--user-data-dir={profile}", "--no-first-run", url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            finished = done.wait(timeout)
        finally:
            chrome.terminate()
            chrome.wait()
            server.shutdown()
    if not finished:
        raise SystemExit(f"Chrome did not finish the examples within {timeout} s.")
    if "failed" in outcome:
        raise SystemExit(f"The engine failed: {outcome['failed']}")
