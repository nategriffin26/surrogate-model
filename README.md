# Surface pressure surrogate for DrivAerML

A model that predicts the mean surface pressure coefficient (`CpMeanTrim`) at every cell of a car body from its geometry alone, and a browser tool for working with the prediction on a dropped `.vtp` file. It is trained on the ten assigned [DrivAerML](https://huggingface.co/datasets/neashton/drivaerml) cases (runs 80, 98, 102, 208, 213, 281, 397, 425, 445, 474). On the two cars kept out of training (runs 1 and 500) it explains 94% of the variance of the CFD pressure (area-weighted $R^2$ `0.9425`).

## Setup

Needs [uv](https://docs.astral.sh/uv/) and Python 3.11 to 3.13.

```bash
uv sync --extra dev
```

The web app runs from the repository as cloned. Retraining, and the tests that read a full car (skipped when the file is absent), need the DrivAerML surface files; [`data/README.md`](data/README.md) says where they go and how to fetch them. The browser-engine tests need Bun or Deno and are skipped without one.

## Web app

```bash
uv run surrogate-app
```

Open `http://127.0.0.1:8740/` in a browser with WebGPU (current Chrome, Edge or Safari), then drop a `boundary_i.vtp` file on the page or open one of the two bundled examples (runs 1 and 500). `--port` picks another port.

The app is for an engineer who has a geometry and no CFD run for it. The Result view treats the prediction as a solution to work with: the pressure field on the car, pressure drag with the range across the five networks and its split between forward-facing and rearward-facing surfaces, the highest and lowest pressure and where they are, section cuts, and point probes that read the file's own cell under the cursor. It exports pictures, CSV, and the prediction as a `.vtp` on the file's own mesh (`CpPredicted` and `CpSpread` added as cell arrays) for ParaView or a loads mapping, to download or to keep in the workspace. The workspace, in a panel on the left, holds every file of the session: geometry dropped or chosen, exports, and what the assistant writes; a geometry runs the model from its row and opens its result, and the bundled examples are listed beside the files. Every action, from a control or from the assistant, is one operation in one registry, so both act on the car and the files alike. A line appears when the geometry lies outside the training cars' dimensions. The Model view explains the model on the car itself: what it is told about a point that can be dragged across the surface, what each group of inputs contributes, the networks, and the prediction beside CFD when the file carries it.

The whole analysis runs in the browser, so the 650 MB file never leaves the machine; `surrogate-app` serves the files and the assistant's route to its model:

- `web/engine/` reads the `.vtp`, builds the same lookups and 191 inputs as `src/surrogate/features.py`, and runs the five networks on the GPU at every cell of the file. A first result comes from about a million cells spaced evenly by area; the numbers summed over every cell replace it a few seconds later. On an Apple M5 Pro, run 1 (8.8M cells) shows its result about 5 s after the drop and its all-cell numbers at about 12 s. Without WebGPU the networks run on CPU threads (34 s to the result on the same machine). The engine also analyses a surface given as arrays, such as a car reshaped on the page, exactly as it would the same surface saved to a file; predicts every cell of a 600,000-triangle mesh at once for studies that compare variants; writes any surface with cell arrays as a `.vtp`; and reads one car's pressure at another car's surface for difference maps.
- `web/viewer/` draws the file's own mesh, every cell (17.7M triangles for run 1), where the GPU keeps up. A 600,000-triangle simplified copy, made by edge collapse onto the file's own points, serves section cuts, slower GPUs and the bundled examples.
- The Python pipeline is the reference. `tests/test_web_engine.py` runs the browser engine's modules under Bun or Deno and holds its cell geometry, inputs, each network's $C_p$ and the spread to the Python values. On the two holdout files the all-cell drag from the browser and from `run_vtp_inference` agree to `0.0007`, which is the difference between their two frontal-area estimates.

There is no build step: the app is plain ES modules, with three.js and meshoptimizer vendored in `web/vendor/`.

### Assistant

The Assistant panel holds one agent the engineer talks to. Its tools are the operations the app's own controls run (`web/app/operations.js`), so what it does shows on screen as if the engineer had done it, and a Python kernel. Each tool call is a step in the conversation that opens to show what was asked and what came back. Operations that open, write, download or delete ask the engineer first, and so does Python code that removes, replaces or moves the engineer's files. The stop button ends a run at once.

The kernel is Pyodide (Python 3.14 with numpy, scipy, pandas and matplotlib) in a worker of its own, loaded from the jsDelivr CDN on first use and kept for the conversation. Its working directory is the app's workspace, the same files the Files panel shows. It has no network: its script is served with a Content-Security-Policy that allows only the Pyodide CDN, and the request functions are removed once Pyodide has loaded. Its module `pressure` gives the open cases as numpy arrays, runs any operation, predicts a changed geometry with the browser engine and writes `.vtp` files, so a study is a loop of deform, predict and measure. A run computing for more than 60 s is stopped by restarting the kernel.

The page sends the conversation to `/api/assistant`, and the server adds the key, the model, the reasoning effort and the output limit before forwarding it to OpenRouter and streaming the reply back. To enable it, set `OPENROUTER_API_KEY` in the environment or in `local/secrets/assistant.env`; `ASSISTANT_MODEL` picks the model (default `anthropic/claude-opus-5.5`) and `ASSISTANT_ENABLED=0` switches it off. Without a key the panel says the assistant is not available and the rest of the app works as before. `deploy/assistant-worker/` serves the same route on a public site as a Cloudflare Worker, with limits of 120 requests a minute overall, 30 a minute per visitor and 5,000 a day.

## Model

The model predicts one cell at a time. Ten cars are ten examples for a model that reads a whole car, and 1.2 million for a model that reads one cell and a description of the shape around it. Each cell is described by 191 numbers computed from the polygon geometry only; solver fields are removed when the file is read.

| Input group | Inputs | What it tells the cell |
| :--- | :---: | :--- |
| Position and facing | 10 | Where the cell is on the car (from the nose, the tail, the centre plane and the lowest point, and as fractions of the car's size) and its outward normal. |
| Nearby surface | 60 | In boxes of 3, 10, 25 and 75 cm around the cell: the mean normal, how much surface the box holds, and how the cell's own normal differs from the mean. The same mean and amount one box upstream and downstream. |
| The car ahead and behind | 121 | The car's upper and lower outline in the cell's own lengthwise slice at six distances ahead and six behind, from 5 cm to 1.6 m, and whether there is body there; the same in slices 4, 12 and 36 cm to either side; and the plan-view outline at the cell's height. |

The car's two sides are kept apart in every lookup, so a part that exists on one side only is read on that side only. The lookups are built from every cell of the file.

A network of three hidden layers (`191 -> 256 -> 256 -> 256 -> 1`, GELU) is trained on 120,000 cells per car, drawn in proportion to area. Five such networks are trained, each with two whole cars held out; no fold splits a car's cells. The prediction is their mean. Their standard deviation, multiplied by a factor fitted on the training cars (`2.39`), is reported as the spread.

## Reproducing the model

```bash
uv run python -m surrogate.preprocess       # ten training cars -> data/processed/ (samples and inputs)
uv run python -m surrogate.train            # five-fold ensemble and ablations -> artifacts/
uv run python -m surrogate.export_web       # artifacts/ensemble.pt -> web/model/
uv run python -m surrogate.write_examples   # runs 1 and 500, analysed in headless Chrome -> web/examples/
uv run pytest
```

`artifacts/ensemble.pt` (3.7 MB) is the trained ensemble and `artifacts/metrics.json` holds the numbers in the results table. The engine's mesh, file-format and sampling tests run under Deno: `deno test --allow-read tests/meshes.test.js tests/result.test.js tests/nearest.test.js`.

Python inference on any `.vtp`, with or without `CpMeanTrim`, asks the model at every cell of the file (about 25 s for 8.8M cells):

```python
from surrogate import SurrogateEnsemble, run_vtp_inference

ensemble = SurrogateEnsemble.load("artifacts/ensemble.pt")
result = run_vtp_inference("data/raw/holdout/run_1/boundary_1.vtp", ensemble)
```

`result` holds $C_p$ and its spread per cell, pressure drag for the mean field and for each network, the car's coverage by the training cars, and $C_p$ and cumulative drag at stations along the car.

## Results

Evaluated on 120,000 area-weighted cells per car. Cross-validation holds out two whole cars per fold across the ten training cars; holdout is the five-network ensemble on runs 1 and 500. $C_{d,p}$ is the pressure drag coefficient integrated from the predicted field.

| Model and inputs | CV $R^2$ | Worst-car CV $R^2$ | CV MAE ($C_p$) | CV $C_{d,p}$ MAE | Holdout $R^2$ | Holdout $C_{d,p}$ MAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Constant (fold mean $C_p$ and $C_{d,p}$) | 0.0000 | 0.0000 | 0.1661 | 0.0173 | 0.0000 | 0.0332 |
| PointNet whole-car vector (64) + position and facing | 0.6188 | 0.5622 | 0.0899 | 0.0403 | 0.6322 | 0.0309 |
| Position and facing only (10) | 0.7667 | 0.4805 | 0.0682 | 0.0481 | 0.8003 | 0.0808 |
| Position and facing + nearby surface (70) | 0.9100 | 0.8728 | 0.0453 | 0.0295 | 0.9304 | 0.0122 |
| Position and facing + the car ahead and behind (131) | 0.8948 | 0.8247 | 0.0471 | **0.0159** | 0.9239 | 0.0169 |
| LightGBM, 400 trees, all inputs (191) | 0.9139 | 0.8813 | 0.0448 | 0.0284 | 0.9350 | 0.0206 |
| **Shipped: five networks, all inputs (191)** | **0.9233** | **0.8900** | **0.0415** | 0.0185 | **0.9425** | **0.0097** |

- **A learned whole-car vector does worse than no shape description at all.** With eight training cars per fold, the PointNet vector scores `0.6188` on unseen cars against `0.7667` for position and facing alone.
- **The two shape groups do different jobs.** Nearby surface gives the higher pointwise $R^2$ (`0.9100` against `0.8948`) but a biased drag integral (`0.0295`, worse than the constant's `0.0173`); the car ahead and behind removes most of that bias (`0.0159`). Together they reach `0.9233`.
- **Distances are ladders.** Nearby shape matters in detail and distant shape only in bulk, so box sizes grow about threefold per step and lengthwise distances double. 3 cm is the smallest box that still holds ten or more mesh cells on 98% of the surface.
- **Differences under the seed-to-seed spread are not findings.** Retraining with three seeds moves CV $R^2$ by about `0.003` and CV $C_{d,p}$ MAE by about `0.004`.
- **Integrated drag is a screening estimate.** Summed over every cell, run 1 gives `0.266` against CFD `0.273` and run 500 gives `0.259` against `0.273`. In cross-validation the drag error (`0.0185`) is no better than predicting the training cars' mean drag (`0.0173`), so ten cars do not support ranking designs by drag.
- **The spread tracks the error.** The five networks train on overlapping cars and share blind spots, so their raw standard deviation ranks the error well but understates it; the factor is fitted on the training cars only, by repeating the fold structure one level down. On the holdout cars `71%` of the surface is within one spread of CFD and `92%` within two, and across the spread's quintiles the realized RMSE rises `0.024`, `0.032`, `0.039`, `0.051`, `0.117`. The five networks' drag values are reported individually and not as an error bar, because their range is several times smaller than the drag error.

## Repository layout

```
src/surrogate/       the model in Python; the reference for the browser engine
  surface.py           reads a .vtp: polygons only, outward normals, areas, force integrals
  features.py          the 191 inputs
  preprocess.py        samples the ten training cars and computes their inputs
  model.py             network, five-fold ensemble, spread, coverage check
  train.py             training, cross-validation and ablations
  baselines.py         LightGBM baseline
  metrics.py           pressure and force metrics
  inference.py         any .vtp -> Cp at every cell
  export_web.py        trained ensemble -> web/model/
  write_examples.py    bundled examples, written by the browser engine itself
  server.py            uv run surrogate-app: the tool's files and the assistant's route to its model
web/                 the browser tool
  engine/              .vtp reader and writer, inputs, networks on WebGPU; result.js documents what it returns
  viewer/              3D view of the file's mesh
  app/                 interface; operations.js (every action, for controls and assistant) and workspace.js
                       (every file) are its two contracts; app/model/ is the Model view
  assistant/           the assistant: its conversation loop, panel and prompt, and the Pyodide kernel with
                       pressure.py, the module that gives Python the cases, the operations and the model
  model/               exported weights and model description
  examples/            runs 1 and 500, analysed
  vendor/              three.js, meshoptimizer
artifacts/           ensemble.pt, metrics.json
tests/               the Python pipeline, the browser engine against it, and the assistant's two proxies
deploy/              assistant-worker/: the assistant's route as a Cloudflare Worker for the public site
data/                DrivAerML tables; the surface files are fetched (data/README.md)
```

## Data

DrivAerML (Ashton et al.) is licensed CC BY-SA 4.0; see `data/metadata/LICENSE.txt`. Only surface files are used, and only the ten assigned runs train the model.
