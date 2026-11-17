# Surface-Only Aerodynamics Surrogate (`DrivAerML`)

End-to-end surface pressure surrogate (`CpMeanTrim`) and interactive 3D inspection application trained on the 10-case subset of [DrivAerML](https://huggingface.co/datasets/neashton/drivaerml).

## Setup

```bash
uv sync --extra dev
```

## Gate 1 — Preprocessing & Feature Extraction

Extracts native cell geometry, verifies outward normal orientation via Gauss's divergence theorem, draws area-weighted cell samples (`120,000` per case), and computes the 176-column multi-scale geometric feature matrix (`local` + `8/25/75 cm` regional voxels + streamwise `XZ`/`XY` slice envelopes with spanwise neighbor hops):

```bash
uv run python -m surrogate.preprocess
```

Run physics and geometry-boundary unit tests:

```bash
uv run pytest
```
