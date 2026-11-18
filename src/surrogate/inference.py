"""Cold end-to-end VTP surface inference: Cp at every cell of the file, aero diagnostics, and the file
written back with the prediction."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

from surrogate.features import DOMAIN_X_FROM_NOSE, RASTER_STEP, SurfaceContext
from surrogate.metrics import regional_r2_breakdown, surface_field_metrics
from surrogate.model import SurrogateEnsemble
from surrogate.surface import compute_pressure_forces, read_vtp_surface


@dataclass(frozen=True)
class InferenceResult:
    """Inference on one surface file.

    The model is asked at every cell of the file, and every number here is summed or averaged
    over all of them with their areas: nothing is estimated from a sample of cells.
    """

    source_name: str
    has_ground_truth: bool
    n_cells: int
    total_area: float
    frontal_area: float
    bounds: tuple[float, float, float, float, float, float]
    dimensions: tuple[float, float, float]
    cp_mean: np.ndarray             # (N,) float32, in the file's cell order
    cp_std: np.ndarray              # (N,) float32, expected error (see SurrogateEnsemble)
    forces: dict[str, Any]
    ground_truth_forces: dict[str, float] | None
    ground_truth_metrics: dict[str, Any] | None
    coverage: dict[str, Any]
    sections: dict[str, Any]
    drag_buildup: list[dict[str, float]]
    slice_area_profile: list[dict[str, float]]
    timings_sec: dict[str, float]


def _extract_spanwise_sections(
    centers: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    cp_mean: np.ndarray,
    cp_std: np.ndarray,
    cp_true: np.ndarray | None,
    bounds: tuple[float, float, float, float, float, float],
) -> dict[str, Any]:
    """Build upper/lower Cp(x) ± σ curves at selectable spanwise stations y."""
    xmin, xmax, ymin, ymax, _, _ = bounds
    half_w = 0.5 * (ymax - ymin)
    y_abs = np.abs(centers[:, 1] - 0.5 * (ymin + ymax))
    x_edges = np.linspace(xmin, xmax, 73)
    x_mid = 0.5 * (x_edges[:-1] + x_edges[1:])

    stations = [
        ("centerline", "Centerline (y = 0.00 m)", 0.02, 0.045),
        ("midbody", f"Cabin / A-Pillar (y = {0.42 * half_w:.2f} m)", 0.42 * half_w, 0.045),
        ("outboard", f"Outboard / Wheel (y = {0.78 * half_w:.2f} m)", 0.78 * half_w, 0.045),
    ]
    out_stations: list[dict[str, Any]] = []
    for st_id, label, y0, tol in stations:
        slab = np.flatnonzero(np.abs(y_abs - y0) <= tol)
        x, z, w = centers[slab, 0], centers[slab, 2], areas[slab]
        faces_up = normals[slab, 2] >= -0.15
        fields = {"cp_mean": cp_mean[slab], "cp_std": cp_std[slab]}
        if cp_true is not None:
            fields["cp_true"] = cp_true[slab]

        def skin_point(cells: np.ndarray, x_station: float, z_percentile: float) -> dict[str, float]:
            """Area-weighted values on the outer skin: the cells within 8 cm of the top (or bottom) height."""
            z_skin = np.percentile(z[cells], z_percentile)
            near = cells[np.abs(z[cells] - z_skin) <= 0.08]
            if len(near) == 0:  # the percentile fell in a gap between two surfaces
                near = cells
            point = {"x": round(float(x_station), 3), "z": round(float(np.average(z[near], weights=w[near])), 3)}
            for name, values in fields.items():
                point[name] = round(float(np.average(values[near], weights=w[near])), 4)
            return point

        upper_pts: list[dict[str, float]] = []
        lower_pts: list[dict[str, float]] = []
        for i in range(len(x_mid)):
            in_bin = (x >= x_edges[i]) & (x < x_edges[i + 1])
            if int(in_bin.sum()) < 2:
                continue
            upper, lower = np.flatnonzero(in_bin & faces_up), np.flatnonzero(in_bin & ~faces_up)
            if len(upper):
                upper_pts.append(skin_point(upper, x_mid[i], 85))
            if len(lower):
                lower_pts.append(skin_point(lower, x_mid[i], 15))
        out_stations.append({"id": st_id, "label": label, "y": round(float(y0), 3), "upper": upper_pts, "lower": lower_pts})

    return {"stations": out_stations}


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


def run_vtp_inference(vtp_path: str | Path, ensemble: SurrogateEnsemble) -> InferenceResult:
    """Run full geometry-only inference + diagnostics on any `.vtp` surface file."""
    t_start = time.perf_counter()
    path = Path(vtp_path)

    # Solver arrays are stripped inside the reader; `surf.cp` is kept aside for comparison only.
    surf = read_vtp_surface(path)
    t_read = time.perf_counter() - t_start

    t0 = time.perf_counter()
    context = SurfaceContext.from_surface(surf)
    t_context = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = ensemble.predict_query_points(context, surf.centers, surf.normals)
    t_predict = time.perf_counter() - t0

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

    sections = _extract_spanwise_sections(
        surf.centers, surf.normals, surf.areas, pred.cp_mean, pred.cp_std, surf.cp, surf.bounds
    )
    drag_buildup = _compute_drag_buildup(
        surf.centers,
        surf.normals,
        surf.areas,
        pred.cp_mean,
        surf.cp,
        frontal_area=surf.frontal_area,
        bounds=surf.bounds,
    )

    # The area of the car's slice across x, S(x), along the car against the 10-car training band
    xs_grid = surf.bounds[0] + DOMAIN_X_FROM_NOSE[0] + RASTER_STEP * np.arange(context.nx, dtype=np.float32)
    in_body = (xs_grid >= surf.bounds[0] - 0.05) & (xs_grid <= surf.bounds[1] + 0.05)
    s_min_env = np.asarray(ensemble.training_envelope["slice_area_min"], dtype=np.float32)
    s_max_env = np.asarray(ensemble.training_envelope["slice_area_max"], dtype=np.float32)
    slice_area_profile = [
        {
            "x": round(float(xs_grid[i]), 3),
            "area": round(float(context.slice_area[i]), 4),
            "train_min": round(float(s_min_env[i]), 4),
            "train_max": round(float(s_max_env[i]), 4),
        }
        for i in np.where(in_body)[0][::2]
    ]

    return InferenceResult(
        source_name=path.name,
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
        sections=sections,
        drag_buildup=drag_buildup,
        slice_area_profile=slice_area_profile,
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
