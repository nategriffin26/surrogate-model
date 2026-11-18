# Surface pressure surrogate for DrivAerML

A model that predicts the mean surface pressure coefficient (`CpMeanTrim`) at every cell of a car body from its geometry alone, and a browser tool for working with the prediction on a dropped `.vtp` file. It is trained on the ten assigned [DrivAerML](https://huggingface.co/datasets/neashton/drivaerml) cases (runs 80, 98, 102, 208, 213, 281, 397, 425, 445, 474). On the two cars kept out of training (runs 1 and 500) it explains 94% of the variance of the CFD pressure (area-weighted $R^2$ `0.9425`).

## Setup

Needs [uv](https://docs.astral.sh/uv/), which installs Python (3.11 to 3.13) and the dependencies on first use. Everything is one command, `surrogate`: `uv run surrogate` lists its commands and what this copy holds, with the command that fills each gap.

The repository as cloned holds the trained model and two analysed examples, so the tool runs at once:

```bash
uv run surrogate app
```

To run the model on a DrivAerML file from the command line, fetch the two holdout cars and predict one:

```bash
uv run surrogate fetch holdout     # runs 1 and 500, 1.3 GB -> data/raw/holdout/
uv run surrogate predict data/raw/holdout/run_1/boundary_1.vtp --output run_1_prediction.vtp
```

`predict` takes any `.vtp` surface, with or without `CpMeanTrim`, and asks the model at every cell (38 s for run 1's 8.8M cells on an Apple M5 Pro). It prints the car's size against the training cars', its coverage, the pressure drag and lift with each network's drag, and the spread; when the file carries CFD pressure, also the CFD drag and lift and the area-weighted $R^2$ against them. `--output` writes the file back with the cell arrays `CpPredicted` and `CpSpread`, as the tool's export does, and `--json` writes the summary for scripts. The same from Python:

```python
from surrogate import SurrogateEnsemble, run_vtp_inference

ensemble = SurrogateEnsemble.load("artifacts/ensemble.pt")
result = run_vtp_inference("data/raw/holdout/run_1/boundary_1.vtp", ensemble)
```

`result` holds $C_p$ and its spread per cell, pressure drag for the mean field and for each network, the car's coverage by the training cars, and $C_p$ and cumulative drag at stations along the car. Retraining is under [Reproducing the model](#reproducing-the-model).

## Web app

`uv run surrogate app` serves the tool. Open `http://127.0.0.1:8740/` in a browser with WebGPU (current Chrome, Edge or Safari), then drop a `boundary_i.vtp` file on the page or open one of the two bundled examples (runs 1 and 500). `--port` picks another port.

The app is for an engineer who has a geometry and no CFD run for it. The Result view treats the prediction as a solution to work with: the pressure field on the car; the pressure force coefficients $C_x$, $C_y$ and $C_z$, with the range of $C_x$ across the five networks; the highest and lowest pressure and where they are; regions of the car, picked by the way their faces face or painted or lassoed on it, each read out with its area, mean pressure, forces and share of the whole car's, and saved to measure on any car; section cuts, several kept at once and plotted against position or as a fraction of the car's length over every open car; probes, in named sets placed by clicking, at stations along a section or from a CSV of points, each read on the file's own cell and on every other car at its own nearest cell; and a histogram of the field shown, the area of the car or of the selection in each band of values, where a range brushed across the bars becomes a selection. Over the car go layers of annotation, the engineer's or the assistant's: a field of their own (an expression of the case's arrays, or values from Python) with its legend and probe readout, colour painted on faces, and labels, points, lines, arrows, region outlines, boxes and planes. Each layer belongs to its car and is listed in the Layers tab, where it is shown, hidden, renamed, recoloured or removed; the Label tool there puts a note on the car with a click. Labels keep clear of each other, and a mark the car hides is drawn faint. The result card reads out the selected region, or the whole car when nothing is selected. It exports pictures, CSV, and the prediction as a `.vtp` on the file's own mesh (`CpPredicted` and `CpSpread` added as cell arrays) for ParaView or a loads mapping, to download or to keep in the workspace. The workspace, in a panel on the left, holds every file of the session: geometry dropped or chosen, exports, and what the assistant writes; a geometry runs the model from its row and opens its result, and the bundled examples are listed beside the files. Every action, from a control or from the assistant, is one operation in one registry, so both act on the car and the files alike. A line appears when the geometry lies outside the training cars' dimensions. The Model page, from the toggle in the top bar, draws the model as one diagram for a reader new to it: the whole pipeline in one line, then each step on the open car, with a point the reader drags to see what the networks are told there.

The whole analysis runs in the browser, so the 650 MB file never leaves the machine; `uv run surrogate app` serves the files and the assistant's route to its model:

- `web/engine/` reads the `.vtp`, builds the same lookups and 99 inputs as `src/surrogate/features.py`, and runs the five networks on the GPU at every cell of the file. The lookups' flow and surface grids are built on the GPU too (under a second for run 1, where Python takes 17 s), or on the CPU without WebGPU. A first result comes from about a million cells spaced evenly by area; the numbers summed over every cell replace it a few seconds later. On an Apple M5 Pro, run 1 (8.8M cells) shows its result about 5 s after the drop and its all-cell numbers at about 12 s. Without WebGPU the grids and the networks run on the CPU (about 40 s to the result on the same machine). The engine also analyses a surface given as arrays, such as a car reshaped on the page, exactly as it would the same surface saved to a file; predicts every cell of a 600,000-triangle mesh at once for studies that compare variants; writes any surface with cell arrays as a `.vtp`; reads one car's pressure at another car's surface for difference maps; and takes the CFD pressure from a second file when the analysed one has none, cell for cell when the two files share their polygons and by nearest cell otherwise.
- `web/viewer/` draws the file's own mesh, every cell (17.7M triangles for run 1), where the GPU keeps up. A 600,000-triangle simplified copy, made by edge collapse onto the file's own points, serves section cuts, slower GPUs and the bundled examples.
- The Python pipeline is the reference. `tests/test_web_engine.py` runs the browser engine's modules under Bun or Deno and holds its cell geometry, inputs, each network's $C_p$ and the spread to the Python values. On the two holdout files the all-cell drag from the browser and from `run_vtp_inference` agree to `0.0007`, which is the difference between their two frontal-area estimates.

There is no build step: the app is plain ES modules, with three.js and meshoptimizer vendored in `web/vendor/`.

### Assistant

The Assistant panel holds one agent the engineer talks to. Its tools are the operations the app's own controls run (`web/app/operations.js`), so what it does shows on screen as if the engineer had done it, and a Python kernel. Each tool call is a step in the conversation that opens to show what was asked and what came back. Operations that open, write, download or delete ask the engineer first, and so do removing a layer the engineer made and Python code that removes, replaces or moves the engineer's files. The stop button ends a run at once.

The kernel is Pyodide (Python 3.14 with numpy, scipy, pandas and matplotlib) in a worker of its own, loaded from the jsDelivr CDN on first use and kept for the conversation. Its working directory is the app's workspace, the same files the Files panel shows. It has no network: its script is served with a Content-Security-Policy that allows only the Pyodide CDN, and the request functions are removed once Pyodide has loaded. Its module `pressure` gives the open cases as numpy arrays, runs any operation, predicts a changed geometry with the browser engine and writes `.vtp` files, so a study is a loop of deform, predict and measure. It draws on the car as a plotting library would: `case.show(values, name)` colours the car by an array, `case.paint(mask, 'orange')` paints faces, and `view.label`, `view.arrows`, `view.line` and the rest put marks on it, thousands in one call. A run computing for more than 60 s is stopped by restarting the kernel.

The page sends the conversation to `/api/assistant`, and the server adds the key, the model, the reasoning effort and the output limit before forwarding it to OpenRouter and streaming the reply back. To enable it, set `OPENROUTER_API_KEY` in the environment or in `local/secrets/assistant.env`; `ASSISTANT_MODEL` picks the model (default `anthropic/claude-opus-5.5`) and `ASSISTANT_ENABLED=0` switches it off. Without a key the panel says the assistant is not available and the rest of the app works as before. `deploy/assistant-worker/` serves the same route on a public site as a Cloudflare Worker, with limits of 120 requests a minute overall, 30 a minute per visitor and 5,000 a day.

## Model

The model predicts one cell at a time. Ten cars are ten examples for a model that reads a whole car, and 1.2 million for a model that reads one cell and a description of the shape around it. Each cell is described by 99 numbers computed from the polygon geometry only; solver fields are removed when the file is read.

| Input group | Inputs | What it tells the cell |
| :--- | :---: | :--- |
| Position and facing | 10 | Where the cell is on the car (from the nose, the tail, the centre plane and the lowest point, and as fractions of the car's size) and its outward normal. |
| Potential flow | 4 | The flow around the car standing on the ground in a uniform stream, without viscosity: the velocity the car adds at the cell and the squared speed along the surface, which such a flow turns into pressure as $C_p = 1 - V_t^2$. It places stagnation and the speed-up over curves and under the floor; it knows nothing of separation. |
| Nearby surface | 60 | The surface around the cell blurred at 3, 10, 25 and 75 cm: its mean normal, how much of it there is, and how the cell's own normal differs from the mean; the same mean and amount for the part upstream and the part downstream. |
| Plan-view outline | 25 | How far the car reaches sideways beyond the cell at the cell's height: at the cell, at six distances ahead and six behind from 5 cm to 1.6 m, and 4 and 12 cm above and below. |

The flow and the nearby surface are each a quantity carried by every cell (a source strength, an area) spread onto a grid in metres, convolved with a kernel and read back at the cell, so they do not depend on how finely the file is meshed. The flow is solved on a 4 cm grid with the car's image under the ground, in four passes toward no flow through the surface. The car's two sides are kept apart in every lookup, so a part that exists on one side only is read on that side only. The lookups are built from every cell of the file.

A network of three hidden layers (`99 -> 256 -> 256 -> 256 -> 1`, GELU) is trained on 120,000 cells per car, drawn in proportion to area. Five such networks are trained, each with two whole cars held out; no fold splits a car's cells. The prediction is their mean. Their standard deviation, multiplied by a factor fitted on the training cars (`2.59`), is reported as the spread.

## Reproducing the model

Retraining needs the twelve DrivAerML surface files (7.8 GB) and, for the bundled examples, Google Chrome:

```bash
uv run surrogate fetch       # the ten training and two holdout cars -> data/raw/
uv run surrogate reproduce   # each step below whose output is not current, then the tool
```

| Step | Writes | Time on an Apple M5 Pro |
| :--- | :--- | :--- |
| `uv run surrogate preprocess` | `data/processed/`: 120,000 cells of each car, drawn by area, and their inputs | 4 min |
| `uv run surrogate train` | `artifacts/`: the five networks and their spread factor (`ensemble.pt`, 3.2 MB), the ablations, and `metrics.json`, which holds the numbers in the results table | 8 min on the GPU |
| `uv run surrogate export` | `web/model/`: the networks for the browser, and the training cars the Model page draws | seconds |
| `uv run surrogate examples` | `web/examples/`: runs 1 and 500, analysed by the browser engine in headless Chrome | 40 s |

`reproduce` skips a step whose output was made from the current output of the step before, and stops before it starts when a surface file or Chrome is missing; after a change to the inputs' code, run `preprocess` first. `train` runs on the GPU when there is one. `--device cpu` trains without it: one network takes 57 s there against 10 s on the GPU, and the result is the same (cross-validation $R^2$ `0.9518` on both). `--skip-ablations` trains only the shipped networks (2.5 min on the GPU, 18 min on the CPU).

The tests:

```bash
uv sync --extra dev
uv run pytest
```

The browser-engine tests need Bun or Deno and skip without one; the tests that read a full car skip without its file in `data/raw/`. The engine's mesh, file-format and sampling tests run under Deno: `deno test --allow-read tests/meshes.test.js tests/result.test.js tests/nearest.test.js`.

## Results

Evaluated on 120,000 area-weighted cells per car. Cross-validation holds out two whole cars per fold across the ten training cars; holdout is the five-network ensemble on runs 1 and 500. $C_{d,p}$ is the pressure drag coefficient integrated from the predicted field.

| Model and inputs | CV $R^2$ | Worst-car CV $R^2$ | CV MAE ($C_p$) | CV $C_{d,p}$ MAE | Holdout $R^2$ | Holdout $C_{d,p}$ MAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Constant (fold mean $C_p$ and $C_{d,p}$) | 0.0000 | 0.0000 | 0.1661 | 0.0173 | 0.0000 | 0.0332 |
| PointNet whole-car vector (64) + position and facing | 0.6183 | 0.5583 | 0.0900 | 0.0407 | 0.6301 | 0.0287 |
| Position and facing only (10) | 0.7661 | 0.4804 | 0.0683 | 0.0495 | 0.7999 | 0.0800 |
| + potential flow (14) | 0.9033 | 0.8396 | 0.0456 | 0.0212 | 0.9209 | 0.0330 |
| + nearby surface (74) | 0.9466 | 0.8742 | 0.0340 | 0.0132 | 0.9626 | **0.0071** |
| LightGBM, 400 trees, all inputs (99) | 0.9408 | **0.9225** | 0.0366 | 0.0179 | 0.9475 | 0.0161 |
| **Shipped: five networks, all inputs (99)** | **0.9518** | 0.9166 | **0.0325** | **0.0110** | **0.9631** | 0.0086 |

- **A learned whole-car vector does worse than no shape description at all.** With eight training cars per fold, the PointNet vector scores `0.6183` on unseen cars against `0.7661` for position and facing alone.
- **Potential flow carries the most.** Added to position and facing, its four inputs raise CV $R^2$ from `0.7661` to `0.9033` and cut the drag error from `0.0495` to `0.0212`. The nearby surface takes $R^2$ to `0.9466`; the plan-view outline adds least on average (`0.9518`) and most to the worst car (`0.8742` to `0.9166`).
- **Distances are ladders.** Nearby shape matters in detail and distant shape only in bulk, so the surface scales grow about threefold per step and lengthwise distances double.
- **Differences under the seed-to-seed spread are not findings.** Retraining with three seeds moves CV $R^2$ by about `0.003` and CV $C_{d,p}$ MAE by about `0.004`.
- **Integrated drag is a screening estimate.** On the holdout cars' sampled cells the ensemble gives `0.261` for run 1 against CFD `0.268`, and `0.269` for run 500 against `0.279`. In cross-validation the drag error (`0.0110`) is two thirds of the error of predicting the training cars' mean drag (`0.0173`), so designs whose drag differs by less than about `0.01` are not told apart.
- **The spread tracks the error.** The five networks train on overlapping cars and share blind spots, so their raw standard deviation ranks the error well but understates it; the factor is fitted on the training cars only, by repeating the fold structure one level down. On the holdout cars `68%` of the surface is within one spread of CFD and `90%` within two, and across the spread's quintiles the realized RMSE rises `0.019`, `0.024`, `0.032`, `0.044`, `0.093`. The five networks' drag values are reported individually; on the holdout cars they span `0.018` (run 1) and `0.015` (run 500), about twice the drag error of their mean.

## Repository layout

```
src/surrogate/       the model in Python; the reference for the browser engine
  cli.py               the surrogate command: one subcommand per step, and the status of this copy
  dataset.py           the DrivAerML runs, where their files go under data/raw/, and fetching them
  surface.py           reads a .vtp: polygons only, outward normals, areas, force integrals
  features.py          the 99 inputs
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
web/                 the browser tool
  engine/              .vtp reader and writer, inputs (their flow and surface grids on WebGPU), networks on
                       WebGPU; result.js documents what it returns
  viewer/              3D view of the file's mesh; annotations.js draws marks and labels over it
  app/                 interface; operations.js (every action, for controls and assistant) and workspace.js
                       (every file) are its two contracts; annotations.js holds the layers drawn on a car;
                       app/model/ is the Model page
  assistant/           the assistant: its conversation loop, panel and prompt, and the Pyodide kernel with
                       pressure.py, the module that gives Python the cases, the operations and the model
  model/               exported weights and model description; the training cars the Model page draws
  examples/            runs 1 and 500, analysed
  vendor/              three.js, meshoptimizer
artifacts/           ensemble.pt, metrics.json
tests/               the Python pipeline and its command, the browser engine against it, and the assistant's two proxies
deploy/              assistant-worker/: the assistant's route as a Cloudflare Worker for the public site
data/                DrivAerML tables; the surface files are fetched into raw/ (data/README.md)
```

## Data

DrivAerML (Ashton et al.) is licensed CC BY-SA 4.0; see `data/metadata/LICENSE.txt`. Only surface files are used, and only the ten assigned runs train the model.
