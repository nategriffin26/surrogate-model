"""Cold end-to-end VTP surface inference, display decimation, and aero diagnostics."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import vtk
from scipy.spatial import KDTree

from surrogate.features import RASTER_STEP, DOMAIN_X, SurfaceContext
from surrogate.metrics import regional_r2_breakdown, surface_field_metrics
from surrogate.model import EnsemblePrediction, SurrogateEnsemble
from surrogate.surface import TARGET_FIELD, compute_pressure_forces, estimate_frontal_area, sample_cells_by_area


@dataclass(frozen=True)
class DisplayMesh:
    """Decimated triangle mesh for 60 fps WebGL inspection with outward normals."""

    vertices: np.ndarray            # (V, 3) float32
    faces: np.ndarray               # (T, 3) int32
    centers: np.ndarray             # (T, 3) float32
    normals: np.ndarray             # (T, 3) float32, outward-oriented
    areas: np.ndarray               # (T,) float64
    cp_mean: np.ndarray             # (T,) float32
    cp_std: np.ndarray              # (T,) float32
    drag_density: np.ndarray        # (T,) float32, -Cp * n_x (positive = drag, negative = thrust)
    cp_true: np.ndarray | None      # (T,) float32 when VTP includes CpMeanTrim, else None


@dataclass(frozen=True)
class InferenceResult:
    """Complete cold-file inference output for CLI and FastAPI web endpoints."""

    source_name: str
    has_ground_truth: bool
    n_raw_cells: int
    n_display_faces: int
    total_area: float
    frontal_area: float
    bounds: tuple[float, float, float, float, float, float]
    dimensions: tuple[float, float, float]
    forces: dict[str, float]
    ground_truth_forces: dict[str, float] | None
    ground_truth_metrics: dict[str, Any] | None
    coverage: dict[str, Any]
    slices: dict[str, Any]
    drag_buildup: list[dict[str, float]]
    section_profile: list[dict[str, float]]
    timings_sec: dict[str, float]
    display: DisplayMesh


def _decimate_for_display(
    raw_poly: pv.PolyData,
    *,
    inward_raw: bool,
    divisions: tuple[int, int, int] = (175, 90, 75),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reduce ~8.5M polygons to ~150k–200k clean triangles in <1 s via quadric clustering."""
    tri = raw_poly if raw_poly.is_all_triangles else raw_poly.triangulate()
    if tri.n_cells <= 220_000:
        dec = tri
    else:
        qc = vtk.vtkQuadricClustering()
        qc.SetInputData(tri)
        qc.SetNumberOfDivisions(*divisions)
        qc.UseInternalTrianglesOff()
        qc.CopyCellDataOff()
        qc.Update()
        dec = pv.wrap(qc.GetOutput())
        if not dec.is_all_triangles:
            dec = dec.triangulate()

    vertices = np.asarray(dec.points, dtype=np.float32)
    faces = np.asarray(dec.regular_faces, dtype=np.int32)
    centers = np.asarray(dec.cell_centers().points, dtype=np.float32)
    dec_nm = dec.compute_normals(
        cell_normals=True,
        point_normals=False,
        consistent_normals=False,
        auto_orient_normals=False,
        split_vertices=False,
    )
    normals = np.asarray(dec_nm.cell_data["Normals"], dtype=np.float32)
    areas = np.maximum(
        np.asarray(dec.compute_cell_sizes(length=False, volume=False).cell_data["Area"], dtype=np.float64),
        0.0,
    )
    if inward_raw:
        normals = -normals
    return vertices, faces, centers, normals, areas


def _extract_spanwise_slices(
    centers: np.ndarray,
    normals: np.ndarray,
    cp_mean: np.ndarray,
    cp_std: np.ndarray,
    cp_true: np.ndarray | None,
    bounds: tuple[float, float, float, float, float, float],
) -> dict[str, Any]:
    """Build upper/lower Cp(x) ± σ curves at selectable spanwise stations y."""
    xmin, xmax, ymin, ymax, _, _ = bounds
    yc = 0.5 * (ymin + ymax)
    half_w = 0.5 * (ymax - ymin)
    x = centers[:, 0]
    y_abs = np.abs(centers[:, 1] - yc)
    z = centers[:, 2]

    x_edges = np.linspace(xmin, xmax, 73)
    x_mid = 0.5 * (x_edges[:-1] + x_edges[1:])

    stations = [
        ("centerline", "Centerline (y = 0.00 m)", 0.02, 0.045),
        ("midbody", f"Cabin / A-Pillar (y = {0.42 * half_w:.2f} m)", 0.42 * half_w, 0.045),
        ("outboard", f"Outboard / Wheel (y = {0.78 * half_w:.2f} m)", 0.78 * half_w, 0.045),
    ]
    out_stations: list[dict[str, Any]] = []

    for st_id, label, y0, tol in stations:
        in_slab = np.abs(y_abs - y0) <= tol
        upper_pts: list[dict[str, float]] = []
        lower_pts: list[dict[str, float]] = []

        for i in range(len(x_mid)):
            in_bin = in_slab & (x >= x_edges[i]) & (x < x_edges[i + 1])
            if int(in_bin.sum()) < 2:
                continue

            up_mask = in_bin & (normals[:, 2] >= -0.15)
            lo_mask = in_bin & (normals[:, 2] < -0.15)

            if int(up_mask.sum()) >= 1:
                idx_u = np.where(up_mask)[0]
                z_hi = float(np.percentile(z[idx_u], 85))
                top = idx_u[np.abs(z[idx_u] - z_hi) <= 0.08]
                if len(top) == 0:
                    top = idx_u
                pt_u: dict[str, float] = {
                    "x": round(float(x_mid[i]), 3),
                    "z": round(float(z[top].mean()), 3),
                    "cp_mean": round(float(cp_mean[top].mean()), 4),
                    "cp_std": round(float(cp_std[top].mean()), 4),
                }
                if cp_true is not None:
                    pt_u["cp_true"] = round(float(cp_true[top].mean()), 4)
                upper_pts.append(pt_u)

            if int(lo_mask.sum()) >= 1:
                idx_l = np.where(lo_mask)[0]
                z_lo = float(np.percentile(z[idx_l], 15))
                bot = idx_l[np.abs(z[idx_l] - z_lo) <= 0.08]
                if len(bot) == 0:
                    bot = idx_l
                pt_l: dict[str, float] = {
                    "x": round(float(x_mid[i]), 3),
                    "z": round(float(z[bot].mean()), 3),
                    "cp_mean": round(float(cp_mean[bot].mean()), 4),
                    "cp_std": round(float(cp_std[bot].mean()), 4),
                }
                if cp_true is not None:
                    pt_l["cp_true"] = round(float(cp_true[bot].mean()), 4)
                lower_pts.append(pt_l)

        out_stations.append({"id": st_id, "label": label, "y": round(float(y0), 3), "upper": upper_pts, "lower": lower_pts})

    return {"stations": out_stations}


def _compute_drag_buildup(
    centers: np.ndarray,
    normals: np.ndarray,
    pred: EnsemblePrediction,
    cp_true: np.ndarray | None,
    *,
    total_area: float,
    frontal_area: float,
    bounds: tuple[float, float, float, float, float, float],
    n_stations: int = 60,
) -> list[dict[str, float]]:
    """Compute cumulative pressure drag Cd,p(x) from nose to tail on the area-weighted sample."""
    xmin, xmax = bounds[0], bounds[1]
    x_edges = np.linspace(xmin, xmax, n_stations + 1)
    n_samples = len(centers)
    scale = total_area / (frontal_area * max(n_samples, 1))
    nx = normals[:, 0]
    contrib_pred = -pred.cp_mean * nx * scale
    contrib_true = (-cp_true * nx * scale) if cp_true is not None else None

    order = np.argsort(centers[:, 0])
    x_sorted = centers[order, 0]
    cum_pred = np.cumsum(contrib_pred[order])
    cum_true = np.cumsum(contrib_true[order]) if contrib_true is not None else None

    curve: list[dict[str, float]] = []
    indices = np.searchsorted(x_sorted, x_edges[1:], side="right") - 1
    for x_val, idx in zip(x_edges[1:], indices, strict=True):
        if idx < 0:
            val_p = 0.0
            val_t = 0.0
        else:
            val_p = float(cum_pred[idx])
            val_t = float(cum_true[idx]) if cum_true is not None else 0.0
        entry: dict[str, float] = {"x": round(float(x_val), 3), "cd_p_cum_pred": round(val_p, 5)}
        if cum_true is not None:
            entry["cd_p_cum_true"] = round(val_t, 5)
        curve.append(entry)
    return curve


def run_vtp_inference(
    vtp_path: str | Path,
    ensemble: SurrogateEnsemble,
    *,
    eval_samples: int = 120_000,
) -> InferenceResult:
    """Run full geometry-only inference + diagnostics on any `.vtp` surface file."""
    t_start = time.perf_counter()
    path = Path(vtp_path)
    if not path.is_file():
        raise FileNotFoundError(f"Surface VTP file not found: {path}")

    t0 = time.perf_counter()
    loaded = pv.read(path)
    if not isinstance(loaded, pv.PolyData):
        raise TypeError(f"Expected PolyData surface mesh in {path}, got {type(loaded).__name__}")
    raw_poly = loaded
    if raw_poly.n_cells == 0 or raw_poly.n_points == 0:
        raise ValueError(f"Empty surface mesh in {path}")

    raw_cp: np.ndarray | None = None
    if TARGET_FIELD in raw_poly.cell_data:
        arr = np.asarray(raw_poly.cell_data[TARGET_FIELD], dtype=np.float32).copy()
        if arr.shape == (raw_poly.n_cells,) and np.isfinite(arr).all():
            raw_cp = arr

    # Strip all solver arrays before any geometry or display processing.
    raw_poly.cell_data.clear()
    raw_poly.point_data.clear()
    raw_poly.field_data.clear()
    t_read = time.perf_counter() - t0

    t1 = time.perf_counter()
    points = np.asarray(raw_poly.points, dtype=np.float32)
    centers = np.asarray(raw_poly.cell_centers().points, dtype=np.float32)
    nm = raw_poly.compute_normals(
        cell_normals=True,
        point_normals=False,
        consistent_normals=False,
        auto_orient_normals=False,
        split_vertices=False,
    )
    normals = np.asarray(nm.cell_data["Normals"], dtype=np.float32)
    areas = np.maximum(
        np.asarray(raw_poly.compute_cell_sizes(length=False, volume=False).cell_data["Area"], dtype=np.float64),
        0.0,
    )
    signed_vol = float(np.sum(np.einsum("ij,ij->i", centers, normals) * areas) / 3.0)
    inward_raw = signed_vol < 0.0
    if inward_raw:
        normals = -normals

    bounds = (
        float(centers[:, 0].min()),
        float(centers[:, 0].max()),
        float(centers[:, 1].min()),
        float(centers[:, 1].max()),
        float(centers[:, 2].min()),
        float(centers[:, 2].max()),
    )
    frontal_area = estimate_frontal_area(raw_poly)
    total_area = float(areas.sum())

    from surrogate.surface import SurfaceMesh

    surf = SurfaceMesh(
        points=points,
        centers=centers,
        normals=normals,
        areas=areas,
        bounds=bounds,
        frontal_area=frontal_area,
        enclosed_volume=abs(signed_vol),
        normals_flipped=inward_raw,
        cp=raw_cp,
    )
    context = SurfaceContext.from_surface(surf)
    t_context = time.perf_counter() - t1

    # 1. Area-weighted sample for unbiased force integration, slice curves, and OOD check
    t2 = time.perf_counter()
    sample_idx = sample_cells_by_area(areas, eval_samples, seed=0)
    s_centers = centers[sample_idx]
    s_normals = normals[sample_idx]
    s_pred = ensemble.predict_query_points(context, s_centers, s_normals)
    forces = ensemble.integrated_forces_with_spread(
        s_pred,
        s_normals,
        None,
        total_area=total_area,
        frontal_area=frontal_area,
    )
    coverage = ensemble.assess_coverage(context, s_pred)

    # 2. Fast display mesh (~170k faces) + direct ensemble prediction on display triangles
    d_verts, d_faces, d_centers, d_normals, d_areas = _decimate_for_display(
        raw_poly, inward_raw=inward_raw
    )
    d_pred = ensemble.predict_query_points(context, d_centers, d_normals)
    t_infer = time.perf_counter() - t2

    # Map ground-truth Cp onto display triangles via nearest cell center when available
    d_cp_true: np.ndarray | None = None
    gt_forces: dict[str, float] | None = None
    gt_metrics: dict[str, Any] | None = None

    if raw_cp is not None:
        s_cp_true = raw_cp[sample_idx]
        gt_forces = compute_pressure_forces(
            raw_cp, normals, areas, total_area=total_area, ref_area=frontal_area
        )
        gt_metrics = {
            **{k: round(v, 4) for k, v in surface_field_metrics(s_cp_true, s_pred.cp_mean).items()},
            "regions": regional_r2_breakdown(s_cp_true, s_pred.cp_mean, s_centers, s_normals),
        }
        # Subsample KD-tree over raw centers (every 2nd cell = ~4M pts, builds in ~0.4s)
        stride = 2 if len(centers) > 1_000_000 else 1
        tree = KDTree(centers[::stride])
        _, nn_idx = tree.query(d_centers, workers=4)
        d_cp_true = raw_cp[::stride][nn_idx].astype(np.float32)
    else:
        s_cp_true = None

    slices = _extract_spanwise_slices(
        s_centers, s_normals, s_pred.cp_mean, s_pred.cp_std, s_cp_true, bounds
    )
    drag_buildup = _compute_drag_buildup(
        s_centers,
        s_normals,
        s_pred,
        s_cp_true,
        total_area=total_area,
        frontal_area=frontal_area,
        bounds=bounds,
    )

    # Cross-sectional area S(x) along the vehicle length vs. 10-car training band
    xs_grid = DOMAIN_X[0] + RASTER_STEP * np.arange(context.nx, dtype=np.float32)
    in_body = (xs_grid >= bounds[0] - 0.05) & (xs_grid <= bounds[1] + 0.05)
    s_min_env = np.asarray(ensemble.training_envelope["section_area_min"], dtype=np.float32)
    s_max_env = np.asarray(ensemble.training_envelope["section_area_max"], dtype=np.float32)
    section_profile = [
        {
            "x": round(float(xs_grid[i]), 3),
            "area": round(float(context.section_area_x[i]), 4),
            "train_min": round(float(s_min_env[i]), 4),
            "train_max": round(float(s_max_env[i]), 4),
        }
        for i in np.where(in_body)[0][::2]
    ]

    display = DisplayMesh(
        vertices=d_verts,
        faces=d_faces,
        centers=d_centers,
        normals=d_normals,
        areas=d_areas,
        cp_mean=d_pred.cp_mean,
        cp_std=d_pred.cp_std,
        drag_density=(-d_pred.cp_mean * d_normals[:, 0]).astype(np.float32),
        cp_true=d_cp_true,
    )

    t_total = time.perf_counter() - t_start
    return InferenceResult(
        source_name=path.name,
        has_ground_truth=raw_cp is not None,
        n_raw_cells=int(len(centers)),
        n_display_faces=int(len(d_faces)),
        total_area=round(total_area, 4),
        frontal_area=round(frontal_area, 4),
        bounds=(
            round(bounds[0], 4),
            round(bounds[1], 4),
            round(bounds[2], 4),
            round(bounds[3], 4),
            round(bounds[4], 4),
            round(bounds[5], 4),
        ),
        dimensions=(
            round(surf.dimensions[0], 4),
            round(surf.dimensions[1], 4),
            round(surf.dimensions[2], 4),
        ),
        forces={k: round(v, 5) for k, v in forces.items()},
        ground_truth_forces={k: round(v, 5) for k, v in gt_forces.items()} if gt_forces else None,
        ground_truth_metrics=gt_metrics,
        coverage=coverage,
        slices=slices,
        drag_buildup=drag_buildup,
        section_profile=section_profile,
        timings_sec={
            "read": round(t_read, 2),
            "context": round(t_context, 2),
            "inference_and_mesh": round(t_infer, 2),
            "total": round(t_total, 2),
        },
        display=display,
    )
