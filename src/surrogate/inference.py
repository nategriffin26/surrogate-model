"""Prediction on a surface file: Cp at every cell, the pressure drag and lift, the coverage check, the
comparison with the file's CFD when it carries it, and the file written back with the prediction."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

from surrogate.features import SurfaceContext
from surrogate.metrics import regional_r2_breakdown, surface_field_metrics
from surrogate.model import SurrogateEnsemble
from surrogate.progress import QUIET, Progress, Step
from surrogate.surface import compute_pressure_forces, read_vtp_surface

# The steps of `run_vtp_inference`, with their seconds for run 1 (8.8M cells) on MEASURED_ON's GPU.
INFERENCE_STEPS = [
    Step("Reading the file", 1, "stage", 6),
    Step("Building lookups", 1, "stage", 34),
    Step("Predicting", 8_828_095, "cells", 21),
    Step("Integrating forces", 1, "stage", 1),
]


@dataclass(frozen=True)
class InferenceResult:
    """Inference on one surface file.

    The model is asked at every cell of the file, and every number here is summed or averaged
    over all of them with their areas: nothing is estimated from a sample of cells.
    """

    has_ground_truth: bool
    n_cells: int
    total_area: float
    frontal_area: float
    bounds: tuple[float, float, float, float, float, float]
    dimensions: tuple[float, float, float]
    cp_mean: np.ndarray  # (N,) float32, in the file's cell order
    cp_std: np.ndarray  # (N,) float32, the spread σ(Cp) (see SurrogateEnsemble)
    forces: dict[str, Any]
    ground_truth_forces: dict[str, float] | None
    ground_truth_metrics: dict[str, Any] | None
    coverage: dict[str, Any]
    drag_buildup: list[dict[str, float]]
    timings_sec: dict[str, float]


def _compute_drag_buildup(
    centers: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    cp_mean: np.ndarray,
    cp_true: np.ndarray | None,
    *,
    frontal_area: float,
    bounds: tuple[float, float, float, float, float, float],
    n_stations: int = 60,
) -> list[dict[str, float]]:
    """Cumulative pressure drag Cd,p(x) from nose to tail; the last station is the car's Cd,p."""
    xmin, xmax = bounds[0], bounds[1]
    station = ((centers[:, 0] - xmin) / max(xmax - xmin, 1e-9) * n_stations).astype(np.int64)
    station = np.minimum(station, n_stations - 1)
    drag_per_cp = -normals[:, 0].astype(np.float64) * areas / frontal_area

    def cumulative(cp: np.ndarray) -> np.ndarray:
        return np.cumsum(np.bincount(station, weights=cp * drag_per_cp, minlength=n_stations))

    cum_pred = cumulative(cp_mean)
    cum_true = cumulative(cp_true) if cp_true is not None else None
    curve: list[dict[str, float]] = []
    for i in range(n_stations):
        entry = {
            "x": round(float(xmin + (xmax - xmin) * (i + 1) / n_stations), 3),
            "cd_p_cum_pred": round(float(cum_pred[i]), 5),
        }
        if cum_true is not None:
            entry["cd_p_cum_true"] = round(float(cum_true[i]), 5)
        curve.append(entry)
    return curve


def run_vtp_inference(vtp_path: str | Path, ensemble: SurrogateEnsemble, progress: Progress = QUIET) -> InferenceResult:
    """Predict every cell of the `.vtp` surface file at `vtp_path`, with the numbers `InferenceResult` holds,
    reporting INFERENCE_STEPS on `progress`."""
    t_start = time.perf_counter()
    path = Path(vtp_path)

    # Solver arrays are stripped inside the reader; `surf.cp` is kept aside for comparison only.
    progress.step("Reading the file", detail=path.name)
    surf = read_vtp_surface(path)
    t_read = time.perf_counter() - t_start

    progress.step("Building lookups", detail=f"{surf.n_cells:,} cells")
    t0 = time.perf_counter()
    context = SurfaceContext.from_surface(surf)
    t_context = time.perf_counter() - t0

    progress.step("Predicting", count=surf.n_cells)
    t0 = time.perf_counter()
    pred = ensemble.predict_query_points(context, surf.centers, surf.normals, progress=progress)
    t_predict = time.perf_counter() - t0

    progress.step("Integrating forces")

    forces = ensemble.integrated_forces(
        pred, surf.normals, surf.areas, total_area=surf.total_area, frontal_area=surf.frontal_area
    )
    coverage = ensemble.assess_coverage(context, pred.cp_std, surf.areas)

    gt_forces: dict[str, float] | None = None
    gt_metrics: dict[str, Any] | None = None
    if surf.cp is not None:
        gt_forces = compute_pressure_forces(
            surf.cp, surf.normals, surf.areas, total_area=surf.total_area, ref_area=surf.frontal_area
        )
        gt_metrics = {
            **{k: round(v, 4) for k, v in surface_field_metrics(surf.cp, pred.cp_mean, surf.areas).items()},
            "regions": regional_r2_breakdown(surf.cp, pred.cp_mean, surf.centers, surf.normals, surf.areas),
        }

    drag_buildup = _compute_drag_buildup(
        surf.centers,
        surf.normals,
        surf.areas,
        pred.cp_mean,
        surf.cp,
        frontal_area=surf.frontal_area,
        bounds=surf.bounds,
    )

    return InferenceResult(
        has_ground_truth=surf.cp is not None,
        n_cells=surf.n_cells,
        total_area=round(surf.total_area, 4),
        frontal_area=round(surf.frontal_area, 4),
        bounds=surf.bounds,
        dimensions=surf.dimensions,
        cp_mean=pred.cp_mean,
        cp_std=pred.cp_std,
        forces={k: [round(x, 5) for x in v] if isinstance(v, list) else round(v, 5) for k, v in forces.items()},
        ground_truth_forces={k: round(v, 5) for k, v in gt_forces.items()} if gt_forces else None,
        ground_truth_metrics=gt_metrics,
        coverage=coverage,
        drag_buildup=drag_buildup,
        timings_sec={
            "read": round(t_read, 2),
            "context": round(t_context, 2),
            "prediction": round(t_predict, 2),
            "total": round(time.perf_counter() - t_start, 2),
        },
    )


def write_prediction_vtp(source: str | Path, result: InferenceResult, out: str | Path) -> None:
    """Write the surface file `source` to `out` with `result` added as two float32 cell arrays, as the
    browser tool's export does: `CpPredicted`, the networks' mean, and `CpSpread`, the spread σ(Cp).
    The file's own arrays stay; an earlier `CpPredicted` or `CpSpread` is replaced."""
    mesh = pv.read(source)
    mesh.cell_data["CpPredicted"] = result.cp_mean
    mesh.cell_data["CpSpread"] = result.cp_std
    mesh.save(out)
