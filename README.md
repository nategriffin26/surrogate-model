# Surface pressure surrogate for DrivAerML

A model that predicts the mean surface pressure coefficient (`CpMeanTrim`) at every cell of a car body from its geometry alone, and a browser tool for working with the prediction on a dropped `.vtp` file. It is trained on the ten assigned [DrivAerML](https://huggingface.co/datasets/neashton/drivaerml) cases (runs 80, 98, 102, 208, 213, 281, 397, 425, 445, 474). On the two cars kept out of training (runs 1 and 500) it explains 96% of the variance of the CFD pressure (area-weighted $R^2$ `0.9558`).

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

`predict` takes any `.vtp` surface, with or without `CpMeanTrim`, and asks the model at every cell (95 s for run 1's 8.8M cells on an Apple M5 Pro). It prints the car's size against the training cars', its coverage, the pressure drag and lift with each network's drag, and the spread; when the file carries CFD pressure, also the CFD drag and lift and the area-weighted $R^2$ against them. `--output` writes the file back with the cell arrays `CpPredicted` and `CpSpread`, as the tool's export does, and `--json` writes the summary for scripts. The same from Python:

```python
from surrogate import SurrogateEnsemble, run_vtp_inference

ensemble = SurrogateEnsemble.load("artifacts/ensemble.pt")
result = run_vtp_inference("data/raw/holdout/run_1/boundary_1.vtp", ensemble)
```

`result` holds $C_p$ and its spread per cell, pressure drag for the mean field and for each network, the car's coverage by the training cars, and $C_p$ and cumulative drag at stations along the car. Retraining is under [Reproducing the model](#reproducing-the-model).

## Web app

`uv run surrogate app` serves the tool. Open `http://127.0.0.1:8740/` in a browser with WebGPU (current Chrome, Edge or Safari), then drop a `boundary_i.vtp` file on the page or open one of the two bundled examples (runs 1 and 500). `--port` picks another port.

The app is for an engineer who has a geometry and no CFD run for it. The Result view treats the prediction as a solution to work with:

- **Field and forces.** The pressure field on the car; the pressure force coefficients $C_x$, $C_y$ and $C_z$, with the range of $C_x$ across the five networks; the highest and lowest pressure and where they are. A line appears when the geometry lies outside the training cars' dimensions.
- **Regions.** Part of the surface, selected by the way it faces, by painting, by lasso or by a range of values, and read out on the result card with its area, mean pressure, forces and share of the whole car's. A histogram in the same tab shows the area of the car, or of the selection, in each band of the field shown; a range brushed across its bars replaces, adds to, subtracts from or intersects the selection. A region is saved by name and measured on any car. Sums run over every polygon of a dropped file; a bundled example carries only the simplified mesh, and its table says so.
- **Sections and probes.** Up to six cuts kept at once, plotted against position or as a fraction of the car's length, over every open car. Probes in named sets, placed by clicking, at stations along a section or from a CSV of points, each read on the file's own cell and on every other car at its nearest cell.
- **Comparison.** The car against the CFD its file carries, against the CFD another file of the same surface carries (attached from the Files panel, which refuses another car's), or against another open car sampled onto its surface: the other's pressure and the difference as fields, the two side by side, both lines on the section plot, and a table for the whole car and every region.
- **Variants.** The car stretched along an axis, raised or lowered with the wheels left on the ground, or with a region moved and the surface around it following. A variant is predicted on the simplified mesh in about a second, opens beside its car and is compared with that car predicted unchanged on the same mesh, so the difference is the change alone. A car and its variants make one study table.
- **Layers.** A field of one's own (an expression of the case's arrays, or values from Python), colour painted on faces, and labels, points, lines, arrows, outlines, boxes and planes, each in a named layer that is hidden, recoloured or removed from the Layers tab.
- **Files and export.** A panel on the left holds every file of the session: geometry, exports and what the assistant writes. Exports are pictures, CSV, the prediction as a `.vtp` on the file's own mesh (`CpPredicted` and `CpSpread` added as cell arrays) for ParaView or a loads mapping, and a report: one HTML page of the analysis on screen that loads nothing.

Every action, from a control or from the assistant, is one operation in one registry (`web/app/operations.js`), so both act on the car and the files alike. The Model page, from the toggle in the top bar, draws the model as one diagram for a reader new to it: the whole pipeline in one line, then each step on the open car, with a point the reader drags to see what the networks are told there.

The whole analysis runs in the browser, so the 650 MB file never leaves the machine; `uv run surrogate app` serves the files and the assistant's route to its model:

- `web/engine/` reads the `.vtp`, builds the same lookups and inputs as `src/surrogate/features.py`, and runs the five networks on the GPU at every cell of the file. The lookups' surface grids are built on the GPU too (under a second for run 1, where Python takes 12 s), or on the CPU without WebGPU. A first result comes from about a million cells spaced evenly by area; the numbers summed over every cell replace it a few seconds later. On an Apple M5 Pro, run 1 (8.8M cells) shows its result about 5 s after the drop and its all-cell numbers at about 12.5 s. Without WebGPU the grids and the networks run on the CPU (about 40 s to the result on the same machine). The engine also analyses a surface given as arrays, such as a car reshaped on the page, exactly as it would the same surface saved to a file; predicts every cell of a 600,000-triangle mesh at once for studies that compare variants; writes any surface with cell arrays as a `.vtp`; reads one car's pressure at another car's surface for difference maps; and takes the CFD pressure from a second file when the analysed one has none, cell for cell when the two files share their polygons and by nearest cell otherwise.
- `web/viewer/` draws the file's own mesh, every cell (17.7M triangles for run 1), where the GPU keeps up. A 600,000-triangle simplified copy, made by edge collapse onto the file's own points, serves section cuts, slower GPUs and the bundled examples.
- The Python pipeline is the reference. `tests/test_web_engine.py` runs the browser engine's modules under Bun or Deno and holds its cell geometry, inputs, each network's $C_p$ and the spread to the Python values. On the two holdout files the all-cell drag from the browser and from `run_vtp_inference` agree to `0.0007`, which is the difference between their two frontal-area estimates.

There is no build step: the app is plain ES modules, with three.js, meshoptimizer and the two typefaces (Hanken Grotesk and TASA Orbiter, under the SIL Open Font License) vendored in `web/vendor/`.

### Assistant

The Assistant panel holds one agent the engineer talks to. Its tools are the operations the app's own controls run (`web/app/operations.js`), so what it does shows on screen as if the engineer had done it, and a Python kernel. A reply streams in as it is written: the model's thinking (Claude's summary of it) opens while it arrives and folds to how long it took, each tool call is a step that opens to show what was asked and what came back, and whatever is in progress has its name shimmering. Operations that open, write, download or delete ask the engineer first, and so do removing a layer the engineer made and Python code that removes, replaces or moves the engineer's files. The stop button ends a run at once and keeps what arrived.

The kernel is Pyodide (Python 3.14 with numpy, scipy, pandas and matplotlib) in a worker of its own, loaded from the jsDelivr CDN on first use and kept for the conversation. Its working directory is the app's workspace, the same files the Files panel shows. It has no network: its script is served with a Content-Security-Policy that allows only the Pyodide CDN, and the request functions are removed once Pyodide has loaded. Its module `pressure` gives the open cases as numpy arrays, runs any operation, predicts a changed geometry with the browser engine and writes `.vtp` files, so a study is a loop of deform, predict and measure. It draws on the car as a plotting library would: `case.show(values, name)` colours the car by an array, `case.paint(mask, 'orange')` paints faces, and `view.label`, `view.arrows`, `view.line` and the rest put marks on it, thousands in one call. A run computing for more than 60 s is stopped by restarting the kernel.

The page sends the conversation to `/api/assistant`, and the server adds the key, the model, the reasoning effort and the output limit before forwarding it to OpenRouter and streaming the reply back. To enable it, set `OPENROUTER_API_KEY` in the environment or in `local/secrets/assistant.env`; `ASSISTANT_MODEL` picks the model (default `anthropic/claude-opus-5.5`) and `ASSISTANT_ENABLED=0` switches it off. Without a key the panel says the assistant is not available and the rest of the app works as before. `deploy/assistant-worker/` serves the same route on a public site as a Cloudflare Worker, with limits of 120 requests a minute overall, 30 a minute per visitor and 5,000 a day.

## Model

The model predicts one cell at a time. Ten cars are ten examples for a model that reads a whole car, and 1.2 million for a model that reads one cell and a description of the shape around it. Each cell is described by 107 numbers computed from the polygon geometry only; solver fields are removed when the file is read. None of them solves for the flow: the network learns it from the ten cars.

| Input group | Inputs | What it tells the cell |
| :--- | :---: | :--- |
| Position and facing | 7 | Where the cell is on the car, in metres from the nose, the tail, the centre plane and the lowest point, and its outward normal. |
| Nearby surface | 48 | The surface around the cell blurred at 3, 10, 25 and 75 cm: its mean normal and how much of it there is, and the same for the part upstream of the cell and the part downstream. |
| Slices across the car | 52 | Thirteen slices across the car, at the cell and 5, 10, 20, 40, 80 and 160 cm ahead and behind. From each: its area, how far it reaches outboard of the cell at the cell's height, and how far it reaches above and below the cell at the cell's distance from the centre plane. |

The lookups are built from the whole surface of the file, cut where its polygons are large into pieces no more than 1 cm across, so a coarse mesh fills them as a fine one does. The nearby surface is the area and the area times the normal of the surface, spread onto a grid whose step is a third of the size (1 cm for the 3 cm size), blurred, and read back at the cell, so it does not depend on how finely the file is meshed. The slices are read from a 2 cm raster of the car. Every lookup is placed in the car's own frame, from its nose, centre plane and lowest point, so moving the whole car changes no input. Every read interpolates smoothly between grid nodes and raster bins, so the inputs do not step as a cell moves across them, which would draw contour rings on the map of a change. The car's two sides are kept apart in every lookup, so a part that exists on one side only is read on that side only.

A network of three hidden layers (`107 -> 256 -> 256 -> 256 -> 1`, GELU) is trained on 120,000 cells per car, drawn in proportion to area. Five such networks are trained, each with two whole cars held out; no fold splits a car's cells. The prediction is their mean. Their standard deviation, multiplied by a factor fitted on the training cars (`2.13`), is reported as the spread.

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

`reproduce` skips a step whose output was made from the current output of the step before, and stops before it starts when a surface file or Chrome is missing; after a change to the inputs' code, run `preprocess` first. `train` runs on the GPU when there is one. `--device cpu` trains without it: one network takes 58 s there against 9 s on the GPU, and the result agrees within the seed-to-seed spread (cross-validation $R^2$ `0.9458` on the GPU, `0.9454` on the CPU). `--skip-ablations` trains only the shipped networks (2.5 min on the GPU, 16 min on the CPU).

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
| Constant (fold mean $C_p$ and $C_{d,p}$) | 0.0000 | 0.0000 | 0.1661 | **0.0173** | 0.0000 | 0.0332 |
| PointNet whole-car vector (64) + position and facing | 0.5634 | 0.5021 | 0.0976 | 0.0321 | 0.5798 | 0.0311 |
| Position and facing only (7) | 0.6486 | 0.2510 | 0.0820 | 0.0822 | 0.7375 | 0.1143 |
| + nearby surface (55) | 0.9364 | 0.8159 | 0.0381 | 0.0213 | **0.9579** | 0.0165 |
| LightGBM, 400 trees, all inputs (107) | 0.9273 | 0.9115 | 0.0435 | **0.0173** | 0.9473 | **0.0095** |
| **Shipped: five networks, all inputs (107)** | **0.9458** | **0.9293** | **0.0356** | 0.0196 | 0.9558 | 0.0128 |

- **A learned whole-car vector does worse than no shape description at all.** With eight training cars per fold, the PointNet vector scores `0.5634` on unseen cars against `0.6486` for position and facing alone.
- **The nearby surface carries the most.** Added to position and facing, its 48 inputs raise CV $R^2$ from `0.6486` to `0.9364` and cut the drag error from `0.0822` to `0.0213`. The slices add least on average (`0.9458`) and most to the worst car (`0.8159` to `0.9293`).
- **Distances are ladders.** Nearby shape matters in detail and distant shape only in bulk, so the surface scales grow about threefold per step and the slices' distances double.
- **Differences under the seed-to-seed spread are not findings.** Retraining with three seeds moves CV $R^2$ by about `0.003` and CV $C_{d,p}$ MAE by up to `0.007`.
- **Integrated drag does not tell these cars apart.** On the holdout cars' sampled cells the ensemble gives `0.254` for run 1 against CFD `0.268`, and `0.268` for run 500 against `0.279`. In cross-validation the drag error (`0.0196`) is no smaller than the error of predicting every car at the training cars' mean drag (`0.0173`).
- **Integrated lift is closer.** In cross-validation the lift error is `0.0397` against `0.0613` for the training cars' mean lift, and on the holdout cars the ensemble gives `0.064` for run 1 against CFD `0.063`, and `0.118` for run 500 against `0.092`.
- **A variant's map carries some discretisation.** Moving the whole car changes no input. A variant that moves the body against what holds the car's nose and ground changes the prediction with no change in the flow: on run 1, moved 7 mm with every lookup held where the unmoved car put it, by under `0.02` in $C_p$ (99th percentile) for a ride-height change and under `0.035` for a shift along the car, as a stretch does (`test_a_variant_moved_against_the_lookups_changes_the_prediction_by_less_than_the_stated_bound`). Most of the shift's share is the slices' 2 cm columns along the car. Differences smaller than that on a variant's map are not findings.
- **The spread ranks the error.** The five networks train on overlapping cars and share blind spots, so their raw standard deviation ranks the error well but understates it; the factor is fitted on the training cars only, by repeating the fold structure one level down. On the holdout cars `57%` of the surface is within one spread of CFD and `84%` within two, and across the spread's quintiles the realized RMSE rises `0.020`, `0.028`, `0.035`, `0.048`, `0.101`. The five networks' drag values are reported individually; on the holdout cars they span `0.051` (run 1) and `0.034` (run 500), about three times the drag error of their mean.

## Limitations

- **Ten cars.** Accuracy is still rising with the number of training cars. Single networks trained on 2, 4, 6 and 8 cars score `0.838`, `0.897`, `0.927` and `0.946` on cars they did not see, with drag errors of 39, 39, 23 and 19 counts (`learning_curve` in `artifacts/metrics.json`). More CFD runs are the first improvement, and DrivAerML holds several hundred more.
- **Pressure only.** The model predicts mean $C_p$. Drag and lift are integrated from pressure alone and leave out wall shear stress, so they are not the car's total coefficients.
- **One flow condition.** The DrivAerML runs share one speed at zero yaw, and no input describes the flow. The model cannot answer another speed, yaw angle or ground condition.
- **One family of shapes.** The training cars are variants of one notchback body, in DrivAerML's axes and in metres. `predict` reports when a car's size, slice-area curve or spread falls outside the training cars'; a prediction there is unchecked.
- **Variants are not validated.** No CFD run exists for a transformed car, so a variant's map has the discretisation bound under [Results](#results) and no measured error.
- **Coarse meshes are little tested.** The tests and examples use DrivAerML's full files. Run 1 decimated to 150,000 triangles scores the same $R^2$ in Python as the full file, and the browser's quick prediction on that mesh scored about `0.04` lower; both were measured once and neither is covered by the tests.
- **One machine.** The tool was run in Chrome on an Apple M5 Pro. A full DrivAerML file peaks near 4.9 GB of memory there, and without WebGPU the result takes about 48 s against about 5 s with it. Safari, Firefox, Linux and Windows are untested.

With a few hundred cars two things become worth trying that lose or do not exist here: a learned encoder of the whole surface in place of the hand-built inputs, and a model of the wall shear stress, which would give total drag.

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
web/                 the browser tool
  engine/              .vtp reader and writer, inputs (their surface grids on WebGPU), networks on WebGPU;
                       result.js documents what it returns
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

## Demo outline

How the tool is presented to a customer: an aerodynamics team that has a geometry and no CFD run for it.

1. **The question.** How many design variants did the last programme put through CFD, and how many did the team want to look at? The surrogate is for the ones in between.
2. **Their file, in seconds.** The customer's `.vtp` is dropped on the page. The pressure field is on the car about five seconds later, computed in the browser; the file is not uploaded.
3. **Worked as an engineer would.** A region is selected and its force, area and mean pressure read off; a section is cut and kept; a probe is placed.
4. **A study, handed to the assistant.** "Raise the car 20 mm and show me where the pressure changes most." The assistant makes the variant with the same operations the engineer has, predicts it in about a second, colours the car by the change and marks where it is largest.
5. **Checked.** On a car that has CFD: prediction and CFD side by side, the difference map, and the spread where the five networks disagree. On cars outside training the model explains 96% of the variance of the CFD pressure. Integrated drag is not yet a ranking number: its error in cross-validation is 20 counts, against 17 for guessing the average car.
6. **What it takes.** Everything the model is told is a description of the car's shape; it learns the flow from ten CFD runs, and that is enough to carry the pressure field to a new car. Telling close designs apart by drag needs more runs, so the loop is: look at every variant with the surrogate, confirm the ones that matter with CFD, retrain on the new runs.

## Data

DrivAerML (Ashton et al.) is licensed CC BY-SA 4.0; see `data/metadata/LICENSE.txt`. Only surface files are used, and only the ten assigned runs train the model.
