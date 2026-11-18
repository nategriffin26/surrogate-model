"""Write the browser tool's model files, `web/model/`, from the trained ensemble checkpoint and the
metrics file written next to it.

    uv run surrogate export

`ensemble.json` holds everything the browser needs to know about the model, so nothing about its shape
is fixed in the browser's code:
- `features`: the feature definition (the constants of `features.py`) and its column count;
- `zClip`, `activation` (what the networks compute between layers, read from `PerCellMLP`), and
  `folds`: per network its output scaling, layer sizes, and where its arrays sit in `weights.bin`;
- `spreadScale`: the factor fitted in training that turns the networks' standard deviation into the
  reported spread σ(Cp);
- `card`: the facts the Model page and the result show, from `metrics.json` and the checkpoint: `dataset`
  (the training data, `DATASET` below); `trainRuns`; `samplesPerCar` (the area-weighted cells each
  training car gave); `dimensions` (the training cars' range of length, width and height, which a car's
  dimensions are measured against); `spreadCoverage` (the share of the holdout cars' surface whose error
  is within the spread σ(Cp)); `networks` (per network `trainRuns`, `validationRuns` and `layers`, its
  widths from input to output); `crossValidation` (`meanR2`); `outsideRuns` (per holdout run `run` and
  `r2`); `comparison` (per model compared, `learner` and its cross-validation `dragMae`);
- `weightsSha256`, the digest of `weights.bin`, which each analysis records so a stale example shows.
`weights.bin` holds every float32 array back to back, little-endian: per network the input mean and
standard deviation, then each layer's weight (outputs x inputs, row-major) and bias. The same
checkpoint and metrics always give the same bytes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from surrogate import features
from surrogate.model import DIMENSIONS, Z_CLIP, PerCellMLP

# The data the ensemble was trained on (Ashton et al., DrivAerML, 2024).
DATASET = {
    "name": "DrivAerML",
    "family": "notchback",
    "freestreamSpeed": 38.889,
    "yawDegrees": 0,
    "wheels": "static",
}


def feature_definition() -> dict:
    """The feature definition of `features.py` and the number of columns it produces."""
    return {
        "count": features.FEATURE_COLUMNS.total_dim,
        "rasterStep": features.RASTER_STEP,
        "domainXFromNose": list(features.DOMAIN_X_FROM_NOSE),
        "domainYHalf": features.DOMAIN_Y_HALF,
        "domainZFromGround": list(features.DOMAIN_Z_FROM_GROUND),
        "heightStep": features.HEIGHT_STEP,
        "heightWindow": features.HEIGHT_WINDOW,
        "sliceDx": list(features.SLICE_DX),
        "surfaceScales": list(features.SURFACE_SCALES),
        "surfaceSteps": list(features.SURFACE_STEPS),
        "surfaceSigma": features.SURFACE_SIGMA,
        "gaussianTruncate": features.GAUSSIAN_TRUNCATE,
        "surfaceReads": [name for name, _ in features.SURFACE_READS],
    }


def activation(payload: dict) -> str:
    """What the networks compute between their linear layers, as `PerCellMLP` builds them."""
    size = {name: int(payload[name]) for name in ("in_dim", "hidden_dim", "depth")}
    net = PerCellMLP(**size).net
    names = {
        type(module).__name__ + ("" if getattr(module, "approximate", "none") == "none" else f" ({module.approximate})")
        for module in net
        if not isinstance(module, torch.nn.Linear)
    }
    if len(names) != 1:
        raise ValueError(f"PerCellMLP mixes activations {sorted(names)}; the browser computes one between every layer")
    return names.pop()


def model_card(payload: dict, metrics: dict, layers: list[list[dict]]) -> dict:
    """What the Model page and the result show: the training data, the networks, and the accuracy measured
    in training. `layers` holds each network's layers as `ensemble.json`'s `folds` lists them."""
    bounds = payload["training_envelope"]["descriptor_bounds"]
    return {
        "dataset": DATASET,
        "trainRuns": [int(run) for run in metrics["allowed_train_runs"]],
        "samplesPerCar": int(metrics["samples_per_car"]),
        "dimensions": {name: {"min": bounds[name]["min"], "max": bounds[name]["max"]} for name in DIMENSIONS},
        "spreadCoverage": metrics["uncertainty_coverage_holdout"]["within_1_std"],
        "networks": [
            {
                "trainRuns": [int(run) for run in fold["train_runs"]],
                "validationRuns": [int(run) for run in fold["val_runs"]],
                "layers": [network[0]["inputs"], *(layer["outputs"] for layer in network)],
            }
            for fold, network in zip(payload["folds"], layers, strict=True)
        ],
        "crossValidation": {"meanR2": metrics["shipped_cv"]["mean_r2"]},
        "outsideRuns": [
            {"run": int(run), "r2": car["r2"]} for run, car in metrics["shipped_holdout"]["per_car"].items()
        ],
        "comparison": [
            {"learner": ablation["learner"], "dragMae": ablation["cv"]["cd_p_mae"]}
            for ablation in metrics["ablations"].values()
        ],
    }


def export_web_model(checkpoint: Path, out_dir: Path) -> None:
    """Write `ensemble.json` and `weights.bin` to `out_dir` from `checkpoint` and the `metrics.json` beside it."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metrics = json.loads((checkpoint.parent / "metrics.json").read_text())
    definition = feature_definition()
    if definition["count"] != int(payload["in_dim"]):
        raise ValueError(
            f"features.py defines {definition['count']} columns but the checkpoint's networks take "
            f"{payload['in_dim']}: export with the features.py the ensemble was trained with"
        )
    depth = int(payload["depth"])
    blocks: list[np.ndarray] = []
    offset = 0

    def place(array: np.ndarray) -> int:
        nonlocal offset
        data = np.ascontiguousarray(array, dtype="<f4").ravel()
        blocks.append(data)
        at = offset
        offset += data.size
        return at

    folds = []
    for fold in payload["folds"]:
        state = fold["state_dict"]
        x_mean = place(fold["x_mean"])
        x_std = place(fold["x_std"])
        layers = []
        # PerCellMLP.net is Linear, GELU, ..., Linear: the linear layers sit at even indices.
        for index in range(0, 2 * depth + 1, 2):
            weight = state[f"net.{index}.weight"].numpy()
            layers.append(
                {
                    "inputs": int(weight.shape[1]),
                    "outputs": int(weight.shape[0]),
                    "weight": place(weight),
                    "bias": place(state[f"net.{index}.bias"].numpy()),
                }
            )
        folds.append(
            {
                "xMean": x_mean,
                "xStd": x_std,
                "yMean": float(fold["y_mean"]),
                "yStd": float(fold["y_std"]),
                "layers": layers,
            }
        )
    weights = np.concatenate(blocks).tobytes()

    model = {
        "features": definition,
        "zClip": Z_CLIP,
        "activation": activation(payload),
        "spreadScale": float(payload["spread_scale"]),
        "folds": folds,
        "card": model_card(payload, metrics, [fold["layers"] for fold in folds]),
        "weightsSha256": hashlib.sha256(weights).hexdigest(),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ensemble.json").write_text(json.dumps(model, indent=1) + "\n")
    (out_dir / "weights.bin").write_bytes(weights)
