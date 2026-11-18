"""Physics and geometry-only boundary tests for the surface reader and the model's inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

from surrogate import features
from surrogate.features import FEATURE_COLUMNS, OUTLINE_DX, SurfaceContext, extract_query_features
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


def test_potential_flow_around_a_sphere_matches_the_exact_solution(tmp_path: Path) -> None:
    """Catches a wrong source kernel, strength, velocity scale or wall condition: around a sphere,
    potential flow has Cp = 1 - 9/4 sin^2 of the angle from the stream (the first pass alone,
    sources of -2 n_x, gives 1 - 25/9 sin^2)."""
    sphere = pv.Icosphere(radius=0.6, nsub=5, center=(0.3, 0.0, 0.0))
    # A speck 1.2 m below is the ground: the sphere's image is 4 radii away and barely acts on it.
    ground = pv.Triangle([[0.3, 0.0, -1.2], [0.301, 0.0, -1.2], [0.3, 0.001, -1.2]])
    sphere.merge(ground).save(tmp_path / "sphere.vtp")
    surface = read_vtp_surface(tmp_path / "sphere.vtp")
    on_sphere = surface.centers[:, 2] > -1.0
    f = extract_query_features(SurfaceContext.from_surface(surface), surface.centers[on_sphere], surface.normals[on_sphere])
    cp = 1.0 - f[:, FEATURE_COLUMNS.flow.start + 3]
    sin2 = 1.0 - surface.normals[on_sphere, 0].astype(np.float64) ** 2
    rms = np.sqrt(np.average((cp - (1.0 - 9.0 / 4.0 * sin2)) ** 2, weights=surface.areas[on_sphere]))
    assert rms < 0.05  # 0.025 on the 4 cm grid; one pass instead of four gives 0.11


def test_the_flow_does_not_pass_through_the_ground(tmp_path: Path) -> None:
    """Catches a missing or misplaced ground image: under a car near the road, the flow the car
    induces must run along the road, with no vertical component at the road's surface."""
    body = pv.Box(bounds=(0.0, 3.0, -0.8, 0.8, -0.15, 1.0)).triangulate().subdivide(3)
    wheel = pv.Box(bounds=(0.5, 0.8, 0.5, 0.8, -0.3, -0.15)).triangulate().subdivide(1)  # sets the ground at z = -0.3
    body.merge(wheel).save(tmp_path / "car.vtp")
    surface = read_vtp_surface(tmp_path / "car.vtp")
    context = SurfaceContext.from_surface(surface)
    road = np.column_stack([np.linspace(0.2, 2.8, 50), np.zeros(50), np.full(50, surface.bounds[4])])
    ux, _, uz = (context.flow_grid.read(c, road) for c in context.induced_velocity)
    assert np.abs(ux).max() > 0.05
    assert np.abs(uz).max() < 1e-3 * np.abs(ux).max()


def test_the_surface_around_a_cell_does_not_depend_on_the_mesh_density(tmp_path: Path) -> None:
    """Catches surroundings averaged over cells rather than area: the same box meshed with cells of
    14 cm and of 3.6 cm must read the same mean normal and amount, and about 1 on a flat face."""
    box = pv.Box(bounds=(-0.8, 3.8, -0.9, 0.9, -0.3, 1.1)).triangulate()
    box.subdivide(5).save(tmp_path / "coarse.vtp")
    box.subdivide(7).save(tmp_path / "fine.vtp")
    coarse, fine = read_vtp_surface(tmp_path / "coarse.vtp"), read_vtp_surface(tmp_path / "fine.vtp")
    # Points on the roof at least 60 cm from its edges
    roof = coarse.centers[(coarse.centers[:, 2] > 1.09) & (np.abs(coarse.centers[:, 1]) < 0.3) & (coarse.centers[:, 0] > 0.0) & (coarse.centers[:, 0] < 3.0)]
    up = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (len(roof), 1))
    f_coarse = extract_query_features(SurfaceContext.from_surface(coarse), roof, up)
    f_fine = extract_query_features(SurfaceContext.from_surface(fine), roof, up)
    for k, scale in enumerate(features.SURFACE_SCALES):
        if scale < 0.25:  # the coarse cells (14 cm) cannot resolve the smaller scales
            continue
        cols = slice(FEATURE_COLUMNS.surface.start + 15 * k, FEATURE_COLUMNS.surface.start + 15 * k + 4)
        np.testing.assert_allclose(f_coarse[:, cols], f_fine[:, cols], atol=0.03)
        if scale < 0.6:  # the roof is flat for 60 cm around every point
            np.testing.assert_allclose(f_fine[:, cols][:, :3], np.tile([0.0, 0.0, 1.0], (len(roof), 1)), atol=0.03)
            # Where the face falls between grid nodes moves the amount by up to 10%; 1 is its mean
            np.testing.assert_allclose(f_fine[:, cols][:, 3], 1.0, atol=0.1)


def test_a_part_on_one_side_of_the_car_is_not_seen_from_the_other_side(tmp_path: Path) -> None:
    """Catches lookups that blend the two halves of the car: a part sticking out of the +y side must
    show beyond the cells behind it on that side, and not behind the same cells on the -y side. A part
    as wide on the -y side, 1.4 m further back, keeps the centre plane at y = 0."""
    body = pv.Box(bounds=(-0.8, 3.8, -0.9, 0.9, -0.3, 1.1)).triangulate().subdivide(6)
    part = pv.Box(bounds=(1.2, 1.4, 0.9, 1.15, 0.8, 0.95)).triangulate().subdivide(4)
    other_part = pv.Box(bounds=(2.6, 2.8, -1.15, -0.9, 0.8, 0.95)).triangulate().subdivide(4)
    body.merge(part).merge(other_part).save(tmp_path / "one_sided.vtp")
    surface = read_vtp_surface(tmp_path / "one_sided.vtp")
    ctx = SurfaceContext.from_surface(surface)

    def cell_near(point: tuple[float, float, float], facing: tuple[float, float, float]) -> int:
        on_face = np.flatnonzero(surface.normals @ np.array(facing, dtype=np.float32) > 0.99)
        return int(on_face[np.argmin(np.linalg.norm(surface.centers[on_face] - np.array(point), axis=1))])

    # Cells on either side, 20 cm behind the middle of the part
    behind = cell_near((1.5, 0.9, 0.87), (0.0, 1.0, 0.0))
    mirrored = cell_near((1.5, -0.9, 0.87), (0.0, -1.0, 0.0))
    f = extract_query_features(ctx, surface.centers[[behind, mirrored]], surface.normals[[behind, mirrored]])
    ahead_by_20cm = FEATURE_COLUMNS.outline.start + 1 + OUTLINE_DX.index(-0.2)
    assert f[0, ahead_by_20cm] == pytest.approx(0.25, abs=0.02)
    assert f[1, ahead_by_20cm] < 0.02


def test_area_weighted_sampling_matches_physical_surface_share() -> None:
    """Catches uniform-over-cells bias when half the polygons are tiny mirror refinements."""
    # 1,000 tiny triangles of area 1e-4 (total 0.1) vs. 100 roof panels of area 1.0 (total 100.0)
    areas = np.concatenate([np.full(1000, 1e-4), np.full(100, 1.0)])
    idx = sample_cells_by_area(areas, 50_000, seed=7)
    roof_share = float(np.mean(idx >= 1000))
    expected_share = 100.0 / 100.1
    assert roof_share == pytest.approx(expected_share, abs=0.005)
