# Reproducing the model, tests and layout

## Reproducing the model

Retraining needs the twelve DrivAerML surface files (7.8 GB) and, for the bundled examples, Google Chrome:

```bash
uv run surrogate fetch       # the ten training and two holdout cars -> data/raw/
uv run surrogate reproduce   # each step below whose output is not current, then the tool
```

| Step | Writes | Time on an Apple M5 Pro |
| :--- | :--- | :--- |
| `uv run surrogate preprocess` | `data/processed/`: 120,000 cells of each car, drawn by area, and their inputs | 8 min |
| `uv run surrogate train` | `artifacts/`: the five networks and their spread factor (`ensemble.pt`, 3.2 MB), and `metrics.json`, which holds the numbers in the results table, the ablations and the learning curve | 16 min on the GPU, 87 min on the CPU |
| `uv run surrogate export` | `web/model/`: the networks for the browser, and the training cars the Model page draws | seconds |
| `uv run surrogate examples` | `web/examples/`: runs 1 and 500, analysed by the browser engine in headless Chrome | 40 s |

`reproduce` skips a step whose output was made from the current output of the step before, and stops before it starts when a surface file or Chrome is missing. After a change to the inputs' code, run `preprocess` first.

`train` runs on the GPU when there is one. `--device cpu` trains without it: one network takes 58 s there against 9 s on the GPU, and cross-validation $R^2$ is `0.9458` on the GPU and `0.9454` on the CPU. `--skip-ablations` trains only the shipped networks (2.5 min on the GPU, 18 min on the CPU).

## Tests

`uv run pytest` runs the whole suite in about 10 minutes on an Apple M5 Pro, most of it the browser-engine tests. `pytest` and `ruff` are in the `dev` dependency group, which `uv run` installs by itself.

- The browser-engine tests need Bun or Deno and skip without one.
- The Worker's tests and the assistant's Markdown test need Deno.
- The tests that read a full car skip without its file in `data/raw/`.
- The engine's mesh, file-format and sampling tests run under Deno: `deno test --allow-read tests/meshes.test.js tests/result.test.js tests/nearest.test.js`.
- The lint is `uv run ruff check` and `uv run ruff format --check`; the JavaScript formatting check is `bunx prettier --check web tests deploy`.

## Repository layout

```
src/surrogate/       the model in Python; the reference for the browser engine
  cli.py               the surrogate command: one subcommand per step, and the status of this copy
  dataset.py           the DrivAerML runs, where their files go under data/raw/, and fetching them
  surface.py           reads a .vtp: polygons only, outward normals, areas, force integrals
  features.py          the 107 inputs
  preprocess.py        samples the ten training and two holdout cars and computes their inputs
  model.py             network, five-fold ensemble, spread, coverage check
  train.py             training, cross-validation and ablations
  baselines.py         LightGBM baseline
  metrics.py           pressure and force metrics
  inference.py         any .vtp -> Cp at every cell, and the file written back with it
  export_web.py        trained ensemble -> web/model/
  export_training_cars.py  the ten training cars' CFD pressure, seen from the side -> web/model/training-cars.json
  write_examples.py    bundled examples, written by the browser engine itself
  server.py            the tool's files and the assistant's route to its model
web/                 Surrogate Model, the browser tool
  engine/              .vtp reader and writer, inputs (their surface grids on WebGPU), networks on WebGPU;
                       result.js documents what it returns
  viewer/              3D view of the file's mesh, and the marks and labels drawn over it
  app/                 interface; operations.js (every action, for controls and assistant) and workspace.js
                       (every file) are its two contracts; section.js is the Section tool and mesh-cut.js
                       cuts the mesh; annotations.js holds the layers drawn on a car and expression.js
                       parses their field expressions; app/model/ is the Model page
  assistant/           the assistant: its conversation loop, panel and prompt, and the Pyodide kernel with
                       pressure.py, the module that gives Python the cases, the operations and the model
  model/               exported weights and model description; the training cars the Model page draws
  examples/            runs 1 and 500, analysed
  vendor/              three.js, meshoptimizer, and the Hanken Grotesk and TASA Orbiter typefaces
artifacts/           ensemble.pt, metrics.json
tests/               the Python pipeline and its command, the browser engine against it, and the assistant's two proxies
deploy/              assistant-worker/assistant.js: the assistant's route for a public site's Cloudflare Worker
docs/                the model (approach, inputs, results, limitations), the web app, the assistant, the demo outline, this file
data/                the dataset's description and license; the surface files are fetched into raw/ (data/README.md)
```
