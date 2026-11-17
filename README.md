# Surface-Only Aerodynamics Surrogate (`DrivAerML`)

Surface pressure surrogate (`CpMeanTrim`) trained on the assigned 10-case subset of [DrivAerML](https://huggingface.co/datasets/neashton/drivaerml) (`run_80, 98, 102, 208, 213, 281, 397, 425, 445, 474`). All model inputs, outward normal orientation, frontal-area normalization, and display decimation derive from surface geometry in `boundary_i.vtp`; solver fields are stripped on load.

## Setup

```bash
uv sync --extra dev
```

## Pipeline

### 1. Preprocess & Feature Extraction

Reads raw `.vtp` files, verifies outward normal orientation via Gauss's divergence theorem, draws `120,000` area-weighted cell samples per vehicle, and extracts 176 deterministic multi-scale geometric features (`10` local coordinates/normals + `45` midplane-symmetrized 8/25/75 cm regional voxel occupancy/clearance features + `121` streamwise `XZ` buttock envelopes, $\pm 4/12/30\text{ cm}$ spanwise neighbor hops, `XY` waterlines, and body-presence masks):

```bash
uv run python -m surrogate.preprocess
```

### 2. Train 5-Fold Ensemble & Ablation Suite

Fits a 5-fold cross-validated per-cell MLP ensemble (`176 -> 256 -> 256 -> 256 -> 1`, GELU, area-weighted Huber loss) strictly on the 10 assigned training vehicles, saves `artifacts/ensemble.pt` (3.5 MB), evaluates unseen holdout runs (`run_1`, `run_500`), and records controlled ablations in `artifacts/metrics.json`:

```bash
uv run python -m surrogate.train
```

### 3. Cold `.vtp` Inference

Runs geometry-only inference on any `.vtp` file (with or without `CpMeanTrim`), returning area-weighted force coefficients with ensemble spread, geometric domain coverage, spanwise $C_p(x) \pm \sigma$ station slices, cumulative drag buildup $C_{d,p}(x)$, and a decimated triangle mesh for interactive WebGL inspection (~10.6 s end-to-end on an 8.8M-cell mesh):

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
| PointNet global 64D $\mathbf{g}$ + local $(x,y,z,\mathbf{n})$ | 0.6468 | 0.5717 | 0.0879 | 0.0603 | 0.6697 | 0.0506 |
| Local MLP only (`10` cols: position + normal) | 0.7691 | 0.6283 | 0.0710 | 0.0465 | 0.7857 | 0.0760 |
| Local + 3D regional voxels (`55` cols: 8/25/75 cm) | 0.8854 | 0.8262 | 0.0505 | 0.0287 | 0.8979 | 0.0362 |
| Local + streamwise `XZ`/`XY` slices & Y-hops (`131` cols) | 0.8815 | 0.8292 | 0.0511 | **0.0147** | 0.9044 | 0.0173 |
| LightGBM 400 trees (`176` cols) | **0.9112** | **0.8785** | **0.0458** | 0.0273 | **0.9336** | 0.0241 |
| **Shipped 5-Fold MLP Ensemble (`176` cols)** | **0.9015** | **0.8577** | **0.0463** | **0.0157** | **0.9153** | **0.0119** |

### Key Findings & Scope Boundaries

- **Learned global vectors overfit 8 training cars**: `PointNet` 64D global shape codes memorize training car IDs across DrivAerML's 16 morph parameters and drop unseen-car $R^2$ (`0.6468`) below a purely local MLP (`0.7691`).
- **Streamwise profiles anchor stagnation and wake recovery**: 3D regional voxels capture local curvature well (`0.8854` CV $R^2$) but leave systematic front-face stagnation bias (`0.0362` holdout $C_{d,p}$ MAE). Adding streamwise `XZ`/`XY` slice envelopes and spanwise neighbor hops cuts holdout $C_{d,p}$ MAE to `0.0119` (`run_1`: predicted `0.261 ± 0.012` vs. CFD `0.274`; `run_500`: predicted `0.262 ± 0.005` vs. CFD `0.273`).
- **Integrated drag remains an approximate screening check**: `~0.012–0.016` $C_{d,p}$ MAE (~120–160 drag counts) separates high-drag shapes from baseline cars and flags out-of-family geometries, while 5-count final design ranking still requires full CFD.
- **Monotonic ensemble uncertainty**: Across holdout surface cells binned into 5 quintiles of fold disagreement $\sigma(C_p)$, realized RMSE rises monotonically (`0.0316 -> 0.0385 -> 0.0469 -> 0.0599 -> 0.1429`).
