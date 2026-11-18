"""Write the browser tool's model files, `web/model/`, from the trained ensemble checkpoint and the
metrics file written next to it.

    uv run python -m surrogate.export_web [artifacts/ensemble.pt] [web/model]

`ensemble.json` holds everything the browser needs to know about the model, so nothing about its shape
is fixed in the browser's code:
- `features`: the feature definition (the constants of `features.py`) and its column count;
- `zClip`, `activation` (what the networks compute between layers, read from `PerCellMLP`), and
  `folds`: per network its output scaling, layer sizes, and where its arrays sit in `weights.bin`;
- `spreadScale`: the factor fitted in training that turns the networks' standard deviation into the
  reported spread σ(Cp);
- `card`: the facts the model view shows, from `metrics.json` and the checkpoint: `dataset` (the
  training data, `DATASET` below); `trainRuns`; `samplesPerCar` (the area-weighted cells each
  training car gave); `dimensions` (the training cars' range of length, width and height, which a
  car's dimensions are also measured against); `networks` (per network `trainRuns`, `validationRuns`
  and `layers`, its widths from input to output); `crossValidation` (`meanR2`, `dragMae`);
  `outsideRuns` (per run `run`, `r2`, `dragMae`); `spreadCoverage` (the share of the outside runs'
  surface whose error is within the spread σ(Cp)); `comparison` (per model compared `learner`,
  `columnRanges`, the [start, stop) column ranges it was given, `crossValidationR2`, `dragMae`);
- `weightsSha256`, the digest of `weights.bin`, which each analysis records so a stale example shows.
`weights.bin` holds every float32 array back to back, little-endian: per network the input mean and
standard deviation, then each layer's weight (outputs x inputs, row-major) and bias. The same
checkpoint and metrics always give the same bytes.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from surrogate import features
from surrogate.model import Z_CLIP, PerCellMLP

REPO = Path(__file__).resolve().parents[2]
DIMENSIONS = ("length", "width", "height")
# The data the ensemble was trained on (Ashton et al., DrivAerML, 2024).
DATASET = {
    "name": "DrivAerML",
    "family": "notchback",
    "freestreamSpeed": 38.889,
    "yawDegrees": 0,
    "wheels": "static",
    "source": "https://arxiv.org/abs/2408.11969",
}


def feature_definition() -> dict:
    """The feature definition of `features.py` and the number of columns it produces."""
    return {
        "count": features.FEATURE_COLUMNS.total_dim,
        "rasterStep": features.RASTER_STEP,
        "domainX": list(features.DOMAIN_X),
        "domainYHalf": features.DOMAIN_Y_HALF,
        "domainZ": list(features.DOMAIN_Z),
        "dxOwn": list(features.DX_OWN),
        "dxNeighbor": list(features.DX_NEIGHBOR),
        "dyNeighbor": list(features.DY_NEIGHBOR),
        "dzNeighbor": list(features.DZ_NEIGHBOR),
        "voxelScales": list(features.VOXEL_SCALES),
    }


def layer_widths(state_dict: dict, depth: int) -> list[int]:
    """A network's widths from input to output; `PerCellMLP.net` holds its linear layers at even indices."""
    weights = [state_dict[f"net.{index}.weight"] for index in range(0, 2 * depth + 1, 2)]
    return [int(weights[0].shape[1])] + [int(weight.shape[0]) for weight in weights]


def activation(payload: dict) -> str:
    """What the networks compute between their linear layers, as `PerCellMLP` builds them."""
    net = PerCellMLP(in_dim=int(payload["in_dim"]), hidden_dim=int(payload["hidden_dim"]), depth=int(payload["depth"])).net
    names = {
        type(module).__name__ + ("" if getattr(module, "approximate", "none") == "none" else f" ({module.approximate})")
        for module in net
        if not isinstance(module, torch.nn.Linear)
    }
    if len(names) != 1:
        raise ValueError(f"PerCellMLP mixes activations {sorted(names)}; the browser computes one between every layer")
    return names.pop()


def model_card(payload: dict, metrics: dict) -> dict:
    """What the model view shows: the training data, the networks, and the accuracy measured in training."""
    cv = metrics["shipped_cv"]
    outside = metrics["shipped_holdout"]["per_car"]
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
                "layers": layer_widths(fold["state_dict"], int(payload["depth"])),
            }
            for fold in payload["folds"]
        ],
        "crossValidation": {"meanR2": cv["mean_r2"], "dragMae": cv["cd_p_mae"]},
        "outsideRuns": [{"run": int(run), "r2": car["r2"], "dragMae": car["cd_p_abs_err"]} for run, car in outside.items()],
        "comparison": [
            {
                "learner": ablation["learner"],
                "columnRanges": [
                    [block.start, block.stop] for block in (getattr(features.FEATURE_COLUMNS, name) for name in ablation["blocks"])
                ],
                "crossValidationR2": ablation["cv"]["mean_r2"],
                "dragMae": ablation["cv"]["cd_p_mae"],
            }
            for ablation in metrics["ablations"].values()
        ],
    }


def export_web_model(checkpoint: Path, out_dir: Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metrics = json.loads((checkpoint.parent / "metrics.json").read_text())
    definition = feature_definition()
    if definition["count"] != int(payload["in_dim"]):
        raise ValueError(
            f"features.py defines {definition['count']} columns but the checkpoint's networks take {payload['in_dim']}: "
            "export with the features.py the ensemble was trained with"
        )
    if "spread_scale" not in payload:
        raise ValueError(f"{checkpoint} has no spread_scale, which the reported spread needs: retrain with the current train.py")
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
        "card": model_card(payload, metrics),
        "weightsSha256": hashlib.sha256(weights).hexdigest(),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ensemble.json").write_text(json.dumps(model, indent=1) + "\n")
    (out_dir / "weights.bin").write_bytes(weights)


def main() -> None:
    checkpoint = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "artifacts/ensemble.pt"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "web/model"
    export_web_model(checkpoint, out_dir)
    print(f"Wrote {out_dir}/ensemble.json and {out_dir}/weights.bin")


if __name__ == "__main__":
    main()
