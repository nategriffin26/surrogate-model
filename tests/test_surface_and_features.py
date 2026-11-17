"""Physics, symmetry, and geometry-only boundary tests for Gate 1 preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

from surrogate.features import FEATURE_COLUMNS, SurfaceContext, extract_query_features
from surrogate.surface import compute_pressure_forces, read_vtp_surface, sample_cells_by_area


def _make_closed_box_vtp(
    path: Path,
    *,
    inward_normals: bool = False,
    include_cp: bool = True,
    poison_solver_fields: bool = False,
) -> Path:
    """Create a synthetic car-sized closed box surface mesh saved as `.vtp`."""
    box = pv.Box(bounds=(-0.8, 3.8, -0.9, 0.9, -0.3, 1.1)).triangulate().subdivide(3)
    if inward_normals:
        box = box.flip_faces()

    centers = np.asarray(box.cell_centers().points, dtype=np.float32)
    if include_cp:
        # Higher pressure on front face (x near -0.8), lower on rear face (x near +3.8)
        cp = np.where(centers[:, 0] < -0.79, 0.8, np.where(centers[:, 0] > 3.79, -0.25, -0.35)).astype(np.float32)
        box.cell_data["CpMeanTrim"] = cp

    if poison_solver_fields:
        box.cell_data["pMeanTrim"] = np.full(box.n_cells, np.nan, dtype=np.float32)
        box.cell_data["wallShearStressMeanTrim"] = np.full((box.n_cells, 3), 1e12, dtype=np.float32)

    box.save(path)
    return path


def test_inward_vtp_normals_flip_outward_and_preserve_positive_drag(tmp_path: Path) -> None:
    """Catches backwards drag sign when CAD/solver VTP polygon winding points inward."""
    out_vtp = _make_closed_box_vtp(tmp_path / "outward.vtp", inward_normals=False)
    in_vtp = _make_closed_box_vtp(tmp_path / "inward.vtp", inward_normals=True)

    surf_out = read_vtp_surface(out_vtp)
    surf_in = read_vtp_surface(in_vtp)

    assert surf_out.normals_flipped is False
    assert surf_in.normals_flipped is True
    np.testing.assert_allclose(surf_in.normals, surf_out.normals, atol=1e-5)

    assert surf_in.cp is not None
    forces = compute_pressure_forces(
        surf_in.cp,
        surf_in.normals,
        surf_in.areas,
        total_area=surf_in.total_area,
        ref_area=surf_in.frontal_area,
    )
    # Front stagnation (+0.8 on nx=-1) + rear base suction (-0.25 on nx=+1) => Cd,p = +1.05
    assert forces["cd_p"] == pytest.approx(1.05, rel=0.03)


def test_geometry_only_vtp_without_cp_and_poisoned_solver_arrays_produce_identical_features(
    tmp_path: Path,
) -> None:
    """Catches hidden solver-field leakage or crashes when unseen `.vtp` lacks CpMeanTrim."""
    clean_geom_vtp = _make_closed_box_vtp(
        tmp_path / "geom_only.vtp",
        include_cp=False,
        poison_solver_fields=False,
    )
    poisoned_vtp = _make_closed_box_vtp(
        tmp_path / "with_solver.vtp",
        include_cp=True,
        poison_solver_fields=True,
    )

    surf_geom = read_vtp_surface(clean_geom_vtp)
    surf_full = read_vtp_surface(poisoned_vtp)

    assert surf_geom.cp is None
    assert surf_full.cp is not None

    ctx_geom = SurfaceContext.from_surface(surf_geom)
    ctx_full = SurfaceContext.from_surface(surf_full)

    f_geom = extract_query_features(ctx_geom, surf_geom.centers[:256], surf_geom.normals[:256])
    f_full = extract_query_features(ctx_full, surf_full.centers[:256], surf_full.normals[:256])

    assert f_geom.shape == (256, FEATURE_COLUMNS.total_dim)
    assert np.isfinite(f_geom).all()
    np.testing.assert_allclose(f_geom, f_full, atol=1e-6)


def test_mirror_symmetric_points_receive_identical_feature_vectors(tmp_path: Path) -> None:
    """Catches broken lateral symmetry folding on y and n_y across left/right car halves."""
    vtp = _make_closed_box_vtp(tmp_path / "sym.vtp", include_cp=False)
    surf = read_vtp_surface(vtp)
    ctx = SurfaceContext.from_surface(surf)

    left_centers = np.array([[0.5, 0.45, 0.35], [2.1, 0.85, 0.15]], dtype=np.float32)
    left_normals = np.array([[-0.3, 0.8, 0.52], [0.1, 0.99, 0.1]], dtype=np.float32)

    right_centers = left_centers.copy()
    right_centers[:, 1] *= -1.0
    right_normals = left_normals.copy()
    right_normals[:, 1] *= -1.0

    f_left = extract_query_features(ctx, left_centers, left_normals)
    f_right = extract_query_features(ctx, right_centers, right_normals)

    np.testing.assert_allclose(f_left, f_right, atol=1e-5)


def test_area_weighted_sampling_matches_physical_surface_share() -> None:
    """Catches uniform-over-cells bias when half the polygons are tiny mirror refinements."""
    # 1,000 tiny triangles of area 1e-4 (total 0.1) vs. 100 roof panels of area 1.0 (total 100.0)
    areas = np.concatenate([np.full(1000, 1e-4), np.full(100, 1.0)])
    idx = sample_cells_by_area(areas, 50_000, seed=7)
    roof_share = float(np.mean(idx >= 1000))
    expected_share = 100.0 / 100.1
    assert roof_share == pytest.approx(expected_share, abs=0.005)
