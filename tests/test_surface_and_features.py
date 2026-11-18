"""Physics and geometry-only boundary tests for the surface reader and the model's inputs."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

from surrogate import features
from surrogate.features import (
    FEATURE_COLUMNS,
    SLICE_DX,
    SLICE_READS,
    SURFACE_WIDTH,
    SurfaceContext,
    extract_query_features,
    surface_samples,
)
from surrogate.model import SurrogateEnsemble
from surrogate.surface import SurfaceMesh, compute_pressure_forces, read_vtp_surface, sample_cells_by_area

REPO = Path(__file__).resolve().parents[1]
RUN_1 = REPO / "data/raw/holdout/run_1/boundary_1.vtp"
# A car-sized box.
BOX = (-0.8, 3.8, -0.9, 0.9, -0.3, 1.1)


def _make_closed_box_vtp(
    path: Path,
    *,
    inward_normals: bool = False,
    include_cp: bool = True,
    poison_solver_fields: bool = False,
) -> Path:
    """Create a synthetic car-sized closed box surface mesh saved as `.vtp`."""
    box = pv.Box(bounds=BOX).triangulate().subdivide(3)
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


def test_the_surface_around_a_cell_does_not_depend_on_the_mesh_density(tmp_path: Path) -> None:
    """Catches surroundings averaged over cells rather than area: the same box meshed with cells of
    14 cm and of 3.6 cm must read the same mean normal and amount, and about 1 on a flat face."""
    box = pv.Box(bounds=BOX).triangulate()
    box.subdivide(5).save(tmp_path / "coarse.vtp")
    box.subdivide(7).save(tmp_path / "fine.vtp")
    coarse, fine = read_vtp_surface(tmp_path / "coarse.vtp"), read_vtp_surface(tmp_path / "fine.vtp")
    # Points on the roof at least 60 cm from its edges
    on_roof = (coarse.centers[:, 2] > 1.09) & (np.abs(coarse.centers[:, 1]) < 0.3)
    roof = coarse.centers[on_roof & (coarse.centers[:, 0] > 0.0) & (coarse.centers[:, 0] < 3.0)]
    up = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (len(roof), 1))
    f_coarse = extract_query_features(SurfaceContext.from_surface(coarse), roof, up)
    f_fine = extract_query_features(SurfaceContext.from_surface(fine), roof, up)
    for k, scale in enumerate(features.SURFACE_SCALES):
        if scale < 0.25:  # the coarse cells (14 cm) cannot resolve the smaller scales
            continue
        start = FEATURE_COLUMNS.surface.start + SURFACE_WIDTH * k
        cols = slice(start, start + 4)
        np.testing.assert_allclose(f_coarse[:, cols], f_fine[:, cols], atol=0.03)
        if scale < 0.6:  # the roof is flat for 60 cm around every point
            np.testing.assert_allclose(f_fine[:, cols][:, :3], np.tile([0.0, 0.0, 1.0], (len(roof), 1)), atol=0.03)
            # Where the face falls between grid nodes moves the amount by up to 10%; 1 is its mean
            np.testing.assert_allclose(f_fine[:, cols][:, 3], 1.0, atol=0.1)


def _cell_near(surface: SurfaceMesh, point: tuple[float, float, float], facing: tuple[float, float, float]) -> int:
    """The cell facing `facing` whose centre is nearest `point`."""
    on_face = np.flatnonzero(surface.normals @ np.array(facing, dtype=np.float32) > 0.99)
    return int(on_face[np.argmin(np.linalg.norm(surface.centers[on_face] - np.array(point), axis=1))])


def _slice_read(read: str, dx: float) -> int:
    """The column of one of a slice's reads, for the slice `dx` from the cell's station."""
    return FEATURE_COLUMNS.slices.start + len(SLICE_READS) * (1 + SLICE_DX.index(dx)) + SLICE_READS.index(read)


def test_a_part_on_one_side_of_the_car_is_not_seen_from_the_other_side(tmp_path: Path) -> None:
    """Catches lookups that blend the two halves of the car: a part sticking out of the +y side must
    show beyond the cells behind it on that side, and not behind the same cells on the -y side. A part
    as wide on the -y side, 1.4 m further back, keeps the centre plane at y = 0."""
    body = pv.Box(bounds=BOX).triangulate().subdivide(6)
    part = pv.Box(bounds=(1.2, 1.4, 0.9, 1.15, 0.8, 0.95)).triangulate().subdivide(4)
    other_part = pv.Box(bounds=(2.6, 2.8, -1.15, -0.9, 0.8, 0.95)).triangulate().subdivide(4)
    body.merge(part).merge(other_part).save(tmp_path / "one_sided.vtp")
    surface = read_vtp_surface(tmp_path / "one_sided.vtp")
    ctx = SurfaceContext.from_surface(surface)

    # Cells on either side, 20 cm behind the middle of the part
    behind = _cell_near(surface, (1.5, 0.9, 0.87), (0.0, 1.0, 0.0))
    mirrored = _cell_near(surface, (1.5, -0.9, 0.87), (0.0, -1.0, 0.0))
    f = extract_query_features(ctx, surface.centers[[behind, mirrored]], surface.normals[[behind, mirrored]])
    assert f[0, _slice_read("outboard", -0.2)] == pytest.approx(0.25, abs=0.02)
    assert f[1, _slice_read("outboard", -0.2)] < 0.02


def test_a_roof_block_on_one_side_shows_in_the_slices_through_it(tmp_path: Path) -> None:
    """Catches a slice's reach above or below read from the wrong spanwise column (the other side,
    or the centre plane) or from the body's underside, and its area summed along the wrong axis or
    unscaled: a 20 cm block on the roof's +y half must stand 20 cm above the roof cells 20 cm behind
    it on that half and not on the -y half, and add its 0.3 x 0.2 m to the area of its slice."""
    body = pv.Box(bounds=BOX).triangulate().subdivide(6)
    block = pv.Box(bounds=(1.2, 1.4, 0.3, 0.6, 1.1, 1.3)).triangulate().subdivide(5)
    body.merge(block).save(tmp_path / "roof_block.vtp")
    surface = read_vtp_surface(tmp_path / "roof_block.vtp")
    ctx = SurfaceContext.from_surface(surface)

    behind = _cell_near(surface, (1.5, 0.45, 1.1), (0.0, 0.0, 1.0))
    mirrored = _cell_near(surface, (1.5, -0.45, 1.1), (0.0, 0.0, 1.0))
    f = extract_query_features(ctx, surface.centers[[behind, mirrored]], surface.normals[[behind, mirrored]])
    assert f[0, _slice_read("above", -0.2)] == pytest.approx(0.2, abs=0.01)
    assert abs(f[1, _slice_read("above", -0.2)]) < 0.01
    assert f[1, _slice_read("below", -0.2)] == pytest.approx(1.4, abs=0.01)  # the body down to its floor
    own_slice = f[0, FEATURE_COLUMNS.slices.start]
    assert own_slice == pytest.approx(1.8 * 1.4, abs=0.05)
    assert f[0, _slice_read("area", -0.2)] - own_slice == pytest.approx(0.3 * 0.2, abs=0.01)


def test_a_coarse_mesh_fills_the_slices_as_a_fine_one_does(tmp_path: Path) -> None:
    """Catches lookups filled at polygon centres alone: a box meshed with triangles up to 60 cm long
    leaves most of the 2 cm raster empty, and its slices then have gaps in them. Meshed coarsely and
    finely, the box must fill the rasters alike, and its samples must carry its whole area. The side
    view holds the reach within 1 cm of each height, so where the box's edge lies just at a window's
    edge the two meshes' samples, a few millimetres apart, can fall either side of it."""
    box = pv.Box(bounds=BOX).triangulate()
    box.subdivide(3).save(tmp_path / "coarse.vtp")
    box.subdivide(7).save(tmp_path / "fine.vtp")
    coarse, fine = read_vtp_surface(tmp_path / "coarse.vtp"), read_vtp_surface(tmp_path / "fine.vtp")
    assert surface_samples(coarse).areas.sum() == pytest.approx(coarse.total_area, rel=1e-9)

    from_coarse, from_fine = SurfaceContext.from_surface(coarse), SurfaceContext.from_surface(fine)
    np.testing.assert_allclose(from_coarse.slice_area, from_fine.slice_area, atol=0.03)
    for raster in ("z_top", "z_bot"):
        np.testing.assert_allclose(getattr(from_coarse, raster), getattr(from_fine, raster), atol=0.02)
    assert np.mean(np.abs(from_coarse.y_out - from_fine.y_out) > 0.02) < 1e-4


def test_moving_the_whole_car_changes_none_of_its_inputs(tmp_path: Path) -> None:
    """Catches lookups placed in the file's coordinates rather than the car's own frame: slice rasters
    binned from a fixed x and z moved the prediction of a car moved whole by 0.03 in Cp, which no flow
    does. A tilted body with a block on its roof, moved by a few millimetres in x, y and z, must give
    the same inputs at every cell, up to the float32 rounding of its moved coordinates."""
    def car(offset: tuple[float, float, float]) -> SurfaceMesh:
        body = pv.Box(bounds=(-0.3, 3.3, -0.9, 0.9, 0.0, 0.6)).triangulate().subdivide(5)
        block = pv.Box(bounds=(1.2, 1.4, 0.3, 0.6, 0.6, 0.8)).triangulate().subdivide(3)
        moved = body.merge(block).rotate_y(15, point=(1.5, 0.0, 0.3)).translate(offset)
        moved.save(tmp_path / f"car_{offset[0]}.vtp")
        return read_vtp_surface(tmp_path / f"car_{offset[0]}.vtp")

    before, after = car((0.0, 0.0, 0.0)), car((0.0137, 0.0043, 0.0071))
    f_before = extract_query_features(SurfaceContext.from_surface(before), before.centers, before.normals)
    f_after = extract_query_features(SurfaceContext.from_surface(after), after.centers, after.normals)
    np.testing.assert_allclose(f_after, f_before, atol=1e-4)


def test_lifting_the_body_against_the_lookups_grids_hardly_moves_its_inputs(tmp_path: Path) -> None:
    """Catches lookups that step with where the surface falls between their nodes and bins, which drew
    contour rings on the difference map of a raised car: the surface spread and read trilinearly,
    whose amount at 3 cm swings by a third between nodes, and slices read from the nearest bin or
    from fixed height bands. A body tilted 15 degrees is lifted 7 mm and moved 7 mm back past a stand
    that stays put ahead of and below it, and so holds the car's nose and ground and with them every
    grid, as a car's wheels hold its ground when its ride height changes; the inputs of cells on its
    sloping top, which crosses the grids at every phase, must hardly change."""
    def car(lift: float) -> SurfaceMesh:
        body = pv.Box(bounds=(-0.3, 3.3, -0.9, 0.9, 0.0, 0.6)).triangulate().subdivide(6)
        body = body.rotate_y(15, point=(1.5, 0.0, 0.3)).translate((lift, 0.0, lift))
        stand = pv.Box(bounds=(-0.6, -0.5, -0.2, 0.2, -0.52, -0.5)).triangulate()
        body.merge(stand).save(tmp_path / f"car_{lift}.vtp")
        return read_vtp_surface(tmp_path / f"car_{lift}.vtp")

    before, after = car(0.0), car(0.007)
    assert before.bounds[0] == after.bounds[0] and before.bounds[4] == after.bounds[4]
    on_top = (before.normals[:, 2] > 0.95) & (np.abs(before.centers[:, 1]) < 0.6)
    top = np.flatnonzero(on_top & (before.centers[:, 0] > 0.2) & (before.centers[:, 0] < 1.5))
    f_before = extract_query_features(SurfaceContext.from_surface(before), before.centers[top], before.normals[top])
    f_after = extract_query_features(SurfaceContext.from_surface(after), after.centers[top], after.normals[top])
    change = np.abs(f_after - f_before)

    amount = FEATURE_COLUMNS.surface.start + 3
    assert change[:, amount].max() < 0.08  # 3 cm
    assert change[:, amount + SURFACE_WIDTH].max() < 0.02  # 10 cm
    own_slice = change[:, FEATURE_COLUMNS.slices.start : FEATURE_COLUMNS.slices.start + len(SLICE_READS)].max(axis=0)
    assert own_slice[0] < 0.01  # area, m^2
    assert own_slice[1:].max() < 0.003  # outboard, above and below, m


@pytest.mark.skipif(not RUN_1.is_file(), reason="needs run 1's surface: uv run surrogate fetch holdout")
def test_a_variant_moved_against_the_lookups_changes_the_prediction_by_less_than_the_stated_bound() -> None:
    """Catches lookups that step with where a variant's body falls against their grids, which draws
    contour rings on a variant's difference map; the README states this bound. Run 1's body is moved
    7 mm up, as by a ride-height change whose wheels hold the ground, and 7 mm along the car, as by a
    stretch, while every lookup stays where the unmoved car put it. The shipped networks' prediction at
    20,000 cells drawn by area, from the moved surface and slices inputs, must change by p99 under 0.02
    and 0.035 in Cp."""
    surface = read_vtp_surface(RUN_1, include_target=False)
    ensemble = SurrogateEnsemble.load(REPO / "artifacts/ensemble.pt")
    cells = sample_cells_by_area(surface.areas, 20_000, seed=0)
    centers, normals = surface.centers[cells], surface.normals[cells]
    inputs = extract_query_features(SurfaceContext.from_surface(surface), centers, normals)
    before = ensemble.predict_features(inputs).cp_mean

    looked_up = slice(FEATURE_COLUMNS.surface.start, FEATURE_COLUMNS.slices.stop)
    for axis, bound in ((2, 0.02), (0, 0.035)):
        offset = np.zeros(3, dtype=np.float32)
        offset[axis] = 0.007
        # The bounds are kept, and with them the nose, ground and grids of the unmoved car
        moved = dataclasses.replace(surface, points=surface.points + offset, centers=surface.centers + offset)
        moved_inputs = inputs.copy()
        moved_context = SurfaceContext.from_surface(moved)
        moved_inputs[:, looked_up] = extract_query_features(moved_context, centers + offset, normals)[:, looked_up]
        change = np.abs(ensemble.predict_features(moved_inputs).cp_mean - before)
        assert np.percentile(change, 99) < bound, f"axis {'xyz'[axis]}"


def test_area_weighted_sampling_matches_physical_surface_share() -> None:
    """Catches uniform-over-cells bias when half the polygons are tiny mirror refinements."""
    # 1,000 tiny triangles of area 1e-4 (total 0.1) vs. 100 roof panels of area 1.0 (total 100.0)
    areas = np.concatenate([np.full(1000, 1e-4), np.full(100, 1.0)])
    idx = sample_cells_by_area(areas, 50_000, seed=7)
    roof_share = float(np.mean(idx >= 1000))
    expected_share = 100.0 / 100.1
    assert roof_share == pytest.approx(expected_share, abs=0.005)
