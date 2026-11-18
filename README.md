# Surface-Only Aerodynamics Surrogate (`DrivAerML`)

Surface pressure surrogate (`CpMeanTrim`) trained on the assigned 10-case subset of [DrivAerML](https://huggingface.co/datasets/neashton/drivaerml) (`run_80, 98, 102, 208, 213, 281, 397, 425, 445, 474`). All model inputs, outward normal orientation, and frontal-area normalization derive from surface geometry in `boundary_i.vtp`; solver fields are stripped on load.

## Setup

```bash
uv sync --extra dev
```

## Web app

```bash
uv run surrogate-app
```

Open `http://127.0.0.1:8740/` in a browser with WebGPU (current Chrome, Edge or Safari), then drop a `boundary_i.vtp` file on the page or open one of the two bundled examples (`run_1` and `run_500`, the cars outside the training set). `--port` picks another port.

The app is for an engineer who has a geometry and no CFD run for it. The Result view treats the prediction as a solution to work with: the pressure field on the car, pressure drag with the range across the five networks and its split between forward-facing and rearward-facing surfaces, the highest and lowest pressure and where they are, section cuts, and point probes that read the file's own cell under the cursor. It exports pictures, CSV, and the prediction as a `.vtp` on the file's own mesh (`CpPredicted` and `CpSpread` added as cell arrays) for ParaView or a loads mapping. A line appears when the geometry lies outside the training cars' dimensions. The Model view explains the model on the car itself: what it is told about a point that can be dragged across the surface, what each group of inputs contributes, the networks, and the prediction beside CFD when the file carries it.

The whole analysis runs in the browser, so the 650 MB file never leaves the machine and `surrogate-app` only serves static files:

- `web/engine/` reads the `.vtp`, builds the same lookup grids and 191 features as `src/surrogate/features.py`, and runs the five networks on the GPU at every cell of the file. A first result comes from about a million cells spaced evenly by area; the numbers summed over every cell replace it a few seconds later. On an Apple M5 Pro, `run_1` (8.8M cells) shows its result about 5 s after the drop and its all-cell numbers at about 12 s. Without WebGPU the networks run on CPU threads (34 s to the result on the same machine).
- `web/viewer/` draws the file's own mesh, every cell (17.7M triangles for `run_1`), where the GPU keeps up. A 600,000-triangle simplified copy, made by edge collapse onto the file's own points, serves section cuts, slower GPUs and the bundled examples.
- The Python pipeline below is the reference. `tests/test_web_engine.py` runs the browser engine's modules under Bun or Deno and holds its cell geometry, features, each network's $C_p$ and the spread to the Python values (the tests skip when neither runtime is installed). On the two holdout files the all-cell drag from the browser and from `run_vtp_inference` agree to `0.0003`, which is the difference between their two frontal-area estimates.

After retraining, regenerate the model files and the examples the app ships:

```bash
uv run python -m surrogate.export_web       # artifacts/ensemble.pt -> web/model/
uv run python -m surrogate.write_examples   # analyses run_1 and run_500 in headless Chrome -> web/examples/
```

No build step: the app is plain ES modules, with three.js and meshoptimizer vendored in `web/vendor/`.

## Pipeline

### 1. Preprocess & Feature Extraction

Reads raw `.vtp` files, verifies outward normal orientation via Gauss's divergence theorem, draws `120,000` area-weighted cell samples per vehicle, and extracts 191 deterministic multi-scale geometric features (`10` local coordinates/normals + `60` regional voxel normal/area-density features at 3/10/25/75 cm + `121` streamwise `XZ` buttock envelopes, $\pm 4/12/36\text{ cm}$ spanwise neighbor hops, `XY` waterlines, and body-presence masks). The car's two sides are kept apart in every lookup, and each cell is told which side it is on, so a part that exists on one side only is read on that side only. The lookup grids behind the features are built from every cell of the file; only the training set is a sample (60,000 cells per car score the same as 120,000):

```bash
uv run python -m surrogate.preprocess
```

### 2. Train 5-Fold Ensemble & Ablation Suite

Fits a 5-fold cross-validated per-cell MLP ensemble (`191 -> 256 -> 256 -> 256 -> 1`, GELU, mean squared error on area-weighted cell samples) strictly on the 10 assigned training vehicles, measures the factor that turns the five models' disagreement into an expected error, saves `artifacts/ensemble.pt` (3.7 MB), evaluates unseen holdout runs (`run_1`, `run_500`), and records controlled ablations in `artifacts/metrics.json`:

```bash
uv run python -m surrogate.train
```

### 3. Cold `.vtp` Inference

Runs geometry-only inference on any `.vtp` file (with or without `CpMeanTrim`). The model is asked at every cell of the file, and every output is summed or averaged over all of them with their areas: $C_p$ and its expected error per cell, pressure drag for the mean field and for each of the five models, geometric domain coverage, spanwise $C_p(x) \pm \sigma$ station slices, and cumulative drag buildup $C_{d,p}(x)$ (~25 s end-to-end on an 8.8M-cell mesh):

```python
from surrogate import SurrogateEnsemble, run_vtp_inference

ensemble = SurrogateEnsemble.load("artifacts/ensemble.pt")
result = run_vtp_inference("data/raw/holdout/run_1/boundary_1.vtp", ensemble)
```

### 4. Tests

```bash
uv run pytest
```

## Measured Ablations & Unseen Holdout Performance

Evaluated on `120,000` area-weighted native cells per car. Five-fold CV holds out 2 complete vehicles per fold across the 10 training cases; holdout evaluates the 5-fold ensemble on unseen `run_1` and `run_500`.

| Model / Feature Set | 5-Fold CV $R^2$ | Worst-Car CV $R^2$ | CV MAE ($C_p$) | CV $C_{d,p}$ MAE | Holdout $R^2$ | Holdout $C_{d,p}$ MAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Constant baseline (fold mean $C_p$ / $C_{d,p}$) | 0.0000 | 0.0000 | 0.1661 | 0.0173 | 0.0000 | 0.0332 |
| PointNet global 64D $\mathbf{g}$ + local $(x,y,z,\mathbf{n})$ | 0.6188 | 0.5622 | 0.0899 | 0.0403 | 0.6322 | 0.0309 |
| Local MLP only (`10` cols: position + normal) | 0.7667 | 0.4805 | 0.0682 | 0.0481 | 0.8003 | 0.0808 |
| Local + 3D regional voxels (`70` cols: 3/10/25/75 cm) | 0.9100 | 0.8728 | 0.0453 | 0.0295 | 0.9304 | 0.0122 |
| Local + streamwise `XZ`/`XY` slices & Y-hops (`131` cols) | 0.8948 | 0.8247 | 0.0471 | **0.0159** | 0.9239 | 0.0169 |
| LightGBM 400 trees (`191` cols) | 0.9139 | 0.8813 | 0.0448 | 0.0284 | 0.9350 | 0.0206 |
| **Shipped 5-Fold MLP Ensemble (`191` cols)** | **0.9233** | **0.8900** | **0.0415** | 0.0185 | **0.9425** | **0.0097** |

### Key Findings & Scope Boundaries

- **Learned global vectors overfit 8 training cars**: `PointNet` 64D global shape codes memorize training car IDs across DrivAerML's 16 morph parameters and drop unseen-car $R^2$ (`0.6188`) below a purely local MLP (`0.7667`).
- **Regional voxels and streamwise slices do different jobs**: regional voxels give the higher pointwise $R^2$ of the two (`0.9100` against `0.8948`) but a biased drag integral (`0.0295` CV $C_{d,p}$ MAE, worse than the constant baseline's `0.0173`); streamwise `XZ`/`XY` slice envelopes and spanwise neighbor hops remove most of that bias (`0.0159`). Together they reach `0.9233`.
- **Feature distances are ladders, because nearby shape matters in detail and distant shape only in bulk**: voxel sizes grow about threefold per step (3, 10, 25, 75 cm); 3 cm is the smallest box that still holds ten or more mesh cells on 98% of the surface, and it resolves the sharp edges where suction peaks. Streamwise offsets double from 5 cm to 1.6 m; spanwise hops are 4, 12, 36 cm.
- **Seed-to-seed spread sets what a comparison can show**: retraining with three seeds moves CV $R^2$ by about `0.003` and CV $C_{d,p}$ MAE by about `0.004`. A variant that folds the car's two sides together scores the same $R^2$ within that spread.
- **Integrated drag is a screening estimate**: summed over every cell, `run_1` gives `0.266` against CFD `0.273` and `run_500` gives `0.259` against `0.273`. In cross-validation the drag error (`0.0185`) is no better than predicting the training cars' mean drag (`0.0173`), so ten cars do not support ranking designs by drag; that still requires CFD.
- **Uncertainty is the five models' disagreement, scaled**: the models train on overlapping cars and share blind spots, so their raw standard deviation ranks the error well but understates it. The scale (`2.39`) is fitted on the training cars only, by repeating the fold structure one level down. On the holdout cars `71%` of the surface is within one scaled standard deviation of CFD and `92%` within two, and across its quintiles realized RMSE rises `0.024 -> 0.032 -> 0.039 -> 0.051 -> 0.117`. The five models' drag values are reported individually, not as an error bar: their spread is several times smaller than the drag error.
