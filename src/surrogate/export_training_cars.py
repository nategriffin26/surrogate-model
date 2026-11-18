"""Write `web/model/training-cars.json`, the ten training cars as the model view draws them: each car seen
from its left (−y) side as a coarse raster of its CFD pressure, with its polygon count and CFD pressure drag.

    uv run python -m surrogate.export_training_cars [data/processed] [web/model/training-cars.json]

Reads the preprocessed cases (`surrogate.preprocess`): the manifest and each training run's area-weighted
sample of cell centres and Cp. A raster cell holds the Cp of the sample nearest the viewer, the one with the
lowest y, as an orthographic view from −y draws it; a cell no sample falls in is null. Cp is stored in
hundredths, rounded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
# Raster cell size in metres: a 4.7 m car is about 120 cells long, more than the thumbnails drawn of it.
STEP = 0.04


def side_raster(centers: np.ndarray, cp: np.ndarray) -> dict[str, object]:
    """Cp seen from −y on a STEP grid in x and z, rows from the lowest z up."""
    x0 = float(np.floor(centers[:, 0].min() / STEP) * STEP)
    z0 = float(np.floor(centers[:, 2].min() / STEP) * STEP)
    col = ((centers[:, 0] - x0) / STEP).astype(int)
    row = ((centers[:, 2] - z0) / STEP).astype(int)
    nx, nz = int(col.max()) + 1, int(row.max()) + 1
    cell = row * nx + col
    # Nearest to the viewer last, so its value is the one that stays.
    order = np.argsort(-centers[:, 1])
    values = np.full(nx * nz, np.nan)
    values[cell[order]] = cp[order]
    hundredths = [None if np.isnan(v) else int(round(v * 100)) for v in values]
    return {"x0": round(x0, 4), "z0": round(z0, 4), "nx": nx, "nz": nz, "cp": hundredths}


def export_training_cars(processed: Path, out: Path) -> None:
    manifest = json.loads((processed / "manifest.json").read_text())
    cases = {case["run"]: case for case in manifest["cases"]}
    cars = []
    for run in manifest["allowed_train_runs"]:
        with np.load(processed / f"run_{run}.npz") as arrays:
            raster = side_raster(arrays["centers"], arrays["cp"])
        cars.append({"run": run, "cells": cases[run]["n_cells"], "cdP": cases[run]["exact_cd_p"], **raster})
    out.write_text(json.dumps({"step": STEP, "cars": cars}, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    processed = Path(args[0]) if args else REPO / "data" / "processed"
    out = Path(args[1]) if len(args) > 1 else REPO / "web" / "model" / "training-cars.json"
    export_training_cars(processed, out)
