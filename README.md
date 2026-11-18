# Surface pressure surrogate for DrivAerML

A model that predicts the mean surface pressure coefficient (`CpMeanTrim`) at every cell of a car body from its geometry alone, and Surrogate Model, a browser tool for working with the prediction on a dropped `.vtp` file. The model is trained on the ten assigned [DrivAerML](https://huggingface.co/datasets/neashton/drivaerml) cases (runs 80, 98, 102, 208, 213, 281, 397, 425, 445, 474). On the two cars kept out of training (runs 1 and 500) it explains 96% of the variance of the CFD pressure (area-weighted $R^2$ `0.9558`).

![Surrogate Model showing the predicted pressure on DrivAerML run 1](docs/images/surrogate-model.png)

## What the submission holds

- The model and its pipeline: `src/surrogate/`, run through one command, `surrogate` ([Model](#model), [Reproducing the model](#reproducing-the-model)).
- The trained networks and their measured results: `artifacts/ensemble.pt` and `artifacts/metrics.json` ([Results](#results)).
- The web app: `web/`, served by `uv run surrogate app` ([Web app](#web-app)).
- The outline of a customer demo: [docs/demo-outline.md](docs/demo-outline.md).

## Setup

Needs [uv](https://docs.astral.sh/uv/), which installs Python (3.11 to 3.13) and the dependencies on first use. `uv run surrogate` lists its commands and what this copy holds, with the command that fills each gap.

The repository as cloned holds the trained model and two analysed examples, so the tool runs at once:

```bash
uv run surrogate app
```

Open `http://127.0.0.1:8740/` in a browser with WebGPU (current Chrome, Edge or Safari), then drop a `boundary_i.vtp` file on the page or open one of the two bundled examples (runs 1 and 500). `--port` picks another port.

To run the model on a DrivAerML file from the command line, fetch the two holdout cars and predict one:

```bash
uv run surrogate fetch holdout     # runs 1 and 500, 1.3 GB -> data/raw/holdout/
uv run surrogate predict data/raw/holdout/run_1/boundary_1.vtp --output run_1_prediction.vtp
```

`predict` takes any `.vtp` surface, with or without `CpMeanTrim`, and asks the model at every cell (95 s for run 1's 8.8M cells on an Apple M5 Pro). It prints:

- the car's size against the training cars', and its coverage;
- the pressure drag and lift, each network's drag, and the spread;
- when the file carries CFD pressure, the CFD drag and lift and the area-weighted $R^2$ against them.

`--output` writes the file back with the cell arrays `CpPredicted` and `CpSpread`, as the tool's export does, and `--json` writes the summary for scripts. The same from Python:

```python
from surrogate import SurrogateEnsemble, run_vtp_inference

ensemble = SurrogateEnsemble.load("artifacts/ensemble.pt")
result = run_vtp_inference("data/raw/holdout/run_1/boundary_1.vtp", ensemble)
```

`result` holds $C_p$ and its spread per cell; pressure drag and lift for the mean field and for each network; the car's size, frontal area and coverage by the training cars; cumulative drag from nose to tail; and, when the file carries CFD, the CFD forces and the agreement with them.

## Web app

Surrogate Model is for an engineer who has a geometry and no CFD run for it. With it the engineer can:

- read the pressure field and the whole car's force coefficients, beside the CFD's when the file carries it;
- cut a section normal to x, y or z and plot the pressure along it;
- place probes by clicking or at stations along the section, and read them on every open car;
- compare the car with its own CFD, the CFD of another file of the same surface, or another open car;
- stretch the car or change its ride height, predicted in about a second and compared with the car;
- add layers: fields of their own, painted faces, labels, arrows and other marks;
- export pictures, CSV tables, the prediction as a `.vtp` for ParaView, and a one-page HTML report;
- read how the model works on the Model page.

The whole analysis runs in the browser, so the 650 MB file never leaves the machine. [docs/web-app.md](docs/web-app.md) describes each view, how the browser reproduces the Python model, and how long it takes.

The Assistant panel holds one agent the engineer talks to. Its tools are the operations the tool's own controls run, so what it does shows on screen as if the engineer had done it, and a Python kernel in the browser. To enable it, set `OPENROUTER_API_KEY` in the environment or in `local/secrets/assistant.env`; without a key the panel says the assistant is not available and the rest of the tool works. [docs/assistant.md](docs/assistant.md) lists what it asks before doing and describes its kernel and its route to its model.

## Model

The model predicts one cell at a time. Ten cars are ten examples for a model that reads a whole car, and 1.2 million for a model that reads one cell and a description of the shape around it. Each cell is described by 107 numbers computed from the polygon geometry only; solver fields are removed when the file is read. None of them solves for the flow: the network learns it from the ten cars.

| Input group | Inputs | What it tells the cell |
| :--- | :---: | :--- |
| Position and facing | 7 | Where the cell is on the car, in metres from the nose, the tail, the centre plane and the lowest point, and its outward normal. |
| Nearby surface | 48 | The surface around the cell blurred at 3, 10, 25 and 75 cm: its mean normal and how much of it there is, and the same for the part upstream of the cell and the part downstream. |
| Slices across the car | 52 | Thirteen slices across the car, at the cell and 5, 10, 20, 40, 80 and 160 cm ahead and behind. From each: its area, how far it reaches outboard of the cell at the cell's height, and how far it reaches above and below the cell at the cell's distance from the centre plane. |

[docs/model.md](docs/model.md) describes how the lookups behind these inputs are built from the surface.

A network of three hidden layers (`107 -> 256 -> 256 -> 256 -> 1`, GELU) is trained on 120,000 cells per car, drawn in proportion to area. Five such networks are trained, each with two whole cars held out; no fold splits a car's cells. The prediction is their mean. Their standard deviation, multiplied by a factor fitted on the training cars (`2.13`), is reported as the spread.

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
- **Integrated drag does not tell these cars apart.** On the holdout cars' sampled cells the ensemble gives `0.254` for run 1 against CFD `0.268`, and `0.268` for run 500 against `0.279`. In cross-validation the drag error (`0.0196`) is no smaller than the error of predicting every car at the training cars' mean drag (`0.0173`).
- **The spread ranks the error.** On the holdout cars `57%` of the surface is within one spread of CFD and `84%` within two, and across the spread's quintiles the realized RMSE rises `0.020`, `0.028`, `0.035`, `0.048`, `0.101`.
- **The upper surfaces and the middle of the car are predicted best; the underbody, the sides and the outer part worst.** Cross-validation $R^2$ by region, where a surface faces a direction when its outward normal's component along it exceeds 0.7, and the regions overlap (`regional_breakdown_cv` in `artifacts/metrics.json`, from `regional_r2_breakdown` in `src/surrogate/metrics.py`):
  - facing up `0.9745`, facing forward `0.9468`, facing rearward `0.9262`, facing down `0.9219`, facing sideways `0.9203`;
  - within 0.3 m of the centre plane `0.9737`, beyond 0.6 m from it `0.9128`.

How the spread is scaled is under [Training](docs/model.md#training); each network's drag, integrated lift, the distances the inputs read at, the seed-to-seed spread and the discretisation of a variant's map are under [Further results](docs/model.md#further-results).

## Limitations

- **Ten cars.** Accuracy is still rising with the number of training cars: single networks trained on 2, 4, 6 and 8 cars score `0.838`, `0.897`, `0.927` and `0.946` on cars they did not see, with drag errors of 39, 39, 23 and 19 counts (a count is 0.001 of a drag coefficient; `learning_curve` in `artifacts/metrics.json`). More CFD runs are the first improvement, and DrivAerML holds several hundred more.
- **Pressure only.** The model predicts mean $C_p$. Drag and lift are integrated from pressure alone and leave out wall shear stress, so they are not the car's total coefficients.
- **One flow condition.** The DrivAerML runs share one speed at zero yaw, and no input describes the flow, so the model cannot answer another speed, yaw angle or ground condition.
- **One family of shapes.** The training cars are variants of one notchback body, in DrivAerML's axes and in metres. `predict` reports when a car's size, slice-area curve or spread falls outside the training cars'; a prediction there is unchecked.
- **Variants are not validated.** No CFD run exists for a transformed car, so a variant's map has the discretisation bound under [Further results](docs/model.md#further-results) and no measured error.
- **Coarse meshes are little tested.** The bundled examples are DrivAerML's full files, and the tests use those files or a small synthetic car. Run 1 decimated to 150,000 triangles scores the same $R^2$ in Python as the full file and about `0.04` lower in the browser's quick prediction; both were measured once and neither is covered by the tests.
- **One machine.** The tool was run in Chrome on an Apple M5 Pro, where a full DrivAerML file peaks near 4.9 GB of memory and the result takes about 5 s with WebGPU and about 48 s without. Safari, Firefox, Linux and Windows are untested.

With a few hundred cars two things become worth trying that lose or do not exist here: a learned encoder of the whole surface in place of the hand-built inputs, and a model of the wall shear stress, which would give total drag.

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

```bash
uv run pytest
```

`pytest` and `ruff` are in the `dev` dependency group, which `uv run` installs by itself. The suite takes about 10 minutes on an Apple M5 Pro, most of it the browser-engine tests.

- The browser-engine tests need Bun or Deno and skip without one.
- The Worker's tests and the assistant's Markdown test need Deno.
- The tests that read a full car skip without its file in `data/raw/`.
- The engine's mesh, file-format and sampling tests run under Deno: `deno test --allow-read tests/meshes.test.js tests/result.test.js tests/nearest.test.js`.
- The lint is `uv run ruff check` and `uv run ruff format --check`.

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
docs/                the web app, the assistant, the model's inputs and further results, the demo outline
data/                the dataset's description and license; the surface files are fetched into raw/ (data/README.md)
```

## Data and license

DrivAerML (Ashton et al., 2024) is licensed CC BY-SA 4.0; see [data/README.md](data/README.md) and `data/metadata/LICENSE.txt`. Only surface files are used, and only the ten assigned runs train the model.
