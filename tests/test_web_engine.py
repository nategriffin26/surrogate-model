"""The browser engine (`web/engine`) against the Python pipeline it reimplements, on a small synthetic
car: the reader across the ways VTK writes a .vtp and the files it must refuse, the polygon geometry
and frontal area, the features at every polygon and each network's Cp there, analysis of a surface
given as arrays, the quick prediction, the export and the .vtp writer; and the exporter of `web/model/`.

The engine's modules run under Bun or Deno through `tests/web_engine_outputs.mjs`, as the browser
runs them; without either runtime these tests skip.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv
import torch
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkFiltersCore import vtkAppendPolyData, vtkStripper
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader, vtkXMLPolyDataWriter

from surrogate import features as feature_module
from surrogate.export_web import export_web_model, feature_definition
from surrogate.features import FeatureColumns, SurfaceContext, extract_query_features
from surrogate.inference import run_vtp_inference
from surrogate.model import Z_CLIP, SurrogateEnsemble
from surrogate.surface import SurfaceMesh, estimate_frontal_area, read_vtp_surface

REPO = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO / "artifacts/ensemble.pt"
RUNNER = REPO / "tests/web_engine_outputs.mjs"


def _javascript_runtime() -> list[str] | None:
    # Node cannot run the engine: it lacks Uint8Array.setFromBase64 (before 25) and fetch of file:// URLs.
    if bun := shutil.which("bun"):
        return [bun]
    if deno := shutil.which("deno"):
        return [deno, "run", "--allow-read", "--allow-write"]
    return None


RUNTIME = _javascript_runtime()
requires_runtime = pytest.mark.skipif(RUNTIME is None, reason="needs Bun or Deno to run the browser engine's modules")


def _synthetic_car() -> pv.PolyData:
    """A car-sized closed body of quads with a tapered, cambered roof, a triangulated wheel, and a
    pentagon and a hexagon fin; wound inward as DrivAerML's files are, with CFD pressure and a
    solver field the engine must ignore."""
    body = pv.Box(bounds=(-0.75, 3.9, -0.95, 0.95, -0.3, 1.2), level=12, quads=True)
    points = np.asarray(body.points, dtype=np.float64)
    along = (points[:, 0] + 0.75) / 4.65
    points[:, 2] = -0.3 + (points[:, 2] + 0.3) * (1.0 - 0.3 * along**2) + 0.08 * np.sin(np.pi * along) * (points[:, 2] > 0.5)
    points[:, 1] *= 1.0 - 0.12 * np.sin(np.pi * along)
    body.points = points
    wheel = pv.Sphere(radius=0.3, center=(0.6, 0.8, 0.0), theta_resolution=16, phi_resolution=10)
    angles = np.linspace(0.0, 2.0 * np.pi, 6)[:-1]
    pentagon = pv.PolyData(np.column_stack([3.4 + 0.2 * np.cos(angles), np.full(5, 0.3), 1.0 + 0.2 * np.sin(angles)]), faces=[5, 0, 1, 2, 3, 4])
    angles = np.linspace(0.0, 2.0 * np.pi, 7)[:-1]
    hexagon = pv.PolyData(np.column_stack([3.4 + 0.2 * np.cos(angles), np.full(6, -0.3), 1.0 + 0.2 * np.sin(angles)]), faces=[6, 0, 1, 2, 3, 4, 5])

    append = vtkAppendPolyData()
    for part in (body, wheel, pentagon, hexagon):
        append.AddInputData(part)
    append.Update()
    car = pv.wrap(append.GetOutput()).flip_faces()
    car.points = np.asarray(car.points, dtype=np.float32)
    centers = np.asarray(car.cell_centers().points)
    car.cell_data["CpMeanTrim"] = (np.sin(3.0 * centers[:, 0]) * np.cos(2.0 * centers[:, 2])).astype(np.float32)
    car.cell_data["pMeanTrim"] = np.full(car.n_cells, np.nan, dtype=np.float32)
    return car


def _write_vtp(surface: pv.PolyData, path: Path, *, mode: str = "binary", encode: bool = True, compressor: str | None = None,
               header: str = "UInt64", big_endian: bool = False, pieces: int = 1) -> Path:
    writer = vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(surface)
    {"binary": writer.SetDataModeToBinary, "appended": writer.SetDataModeToAppended, "ascii": writer.SetDataModeToAscii}[mode]()
    writer.SetEncodeAppendedData(encode)
    {None: writer.SetCompressorTypeToNone, "zlib": writer.SetCompressorTypeToZLib, "lz4": writer.SetCompressorTypeToLZ4}[compressor]()
    {"UInt32": writer.SetHeaderTypeToUInt32, "UInt64": writer.SetHeaderTypeToUInt64}[header]()
    if big_endian:
        writer.SetByteOrderToBigEndian()
    writer.SetNumberOfPieces(pieces)
    assert writer.Write() == 1
    return path


def _run_engine(mode: str, path: Path, out_dir: Path, argument: Path | str | None = None) -> dict:
    """Run the engine's modules on `path`; `argument` is the model directory (engine, pick), a feature
    definition (features) or the stretch factor (stretch)."""
    out_dir.mkdir(exist_ok=True)
    command = [*RUNTIME, str(RUNNER), mode, str(path), str(out_dir)] + ([str(argument)] if argument else [])
    subprocess.run(command, check=True, cwd=REPO, timeout=300)
    return json.loads((out_dir / "outcome.json").read_text())


def _array(out_dir: Path, name: str) -> np.ndarray:
    dtype = {"f32": "<f4", "f64": "<f8", "u32": "<u4"}[name.rsplit(".", 1)[1]]
    return np.fromfile(out_dir / name, dtype=dtype)


@pytest.fixture(scope="module")
def car() -> pv.PolyData:
    return _synthetic_car()


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("model")
    export_web_model(CHECKPOINT, out)
    return out


# Every way VTK writes the binary data the reader accepts: inline base64, appended raw and appended
# base64, each plain and zlib-compressed, with UInt32 and UInt64 headers; and ASCII.
LAYOUTS = {"inline": {"mode": "binary"}, "appended-raw": {"mode": "appended", "encode": False}, "appended-base64": {"mode": "appended"}}
ENCODINGS = {
    **{
        f"{layout}-{compression}-{header.lower()}": {**settings, "compressor": compressor, "header": header}
        for layout, settings in LAYOUTS.items()
        for compression, compressor in {"plain": None, "zlib": "zlib"}.items()
        for header in ("UInt32", "UInt64")
    },
    "ascii": {"mode": "ascii"},
}


@requires_runtime
@pytest.mark.parametrize("triangles_only", [False, True], ids=["mixed-polygons", "triangles-only"])
@pytest.mark.parametrize("encoding", ENCODINGS, ids=list(ENCODINGS))
def test_reader_gives_vtk_surface_for_every_encoding(car: pv.PolyData, tmp_path: Path, encoding: str, triangles_only: bool) -> None:
    """Catches a reader that mis-decodes one of the ways VTK and ParaView write a .vtp."""
    surface = car.triangulate() if triangles_only else car
    path = _write_vtp(surface, tmp_path / "car.vtp", **ENCODINGS[encoding])
    outcome = _run_engine("parse", path, tmp_path / "out")
    assert "error" not in outcome, outcome

    reader = vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    expected = reader.GetOutput()
    polys = expected.GetPolys()
    out = tmp_path / "out"
    np.testing.assert_array_equal(_array(out, "points.f32"), vtk_to_numpy(expected.GetPoints().GetData()).astype(np.float32).ravel())
    np.testing.assert_array_equal(_array(out, "connectivity.u32"), vtk_to_numpy(polys.GetConnectivityArray()))
    np.testing.assert_array_equal(_array(out, "offsets.u32"), vtk_to_numpy(polys.GetOffsetsArray()))
    np.testing.assert_array_equal(_array(out, "cfd.f32"), vtk_to_numpy(expected.GetCellData().GetArray("CpMeanTrim")))


def _truncated(car: pv.PolyData, path: Path) -> Path:
    whole = _write_vtp(car, path.with_suffix(".whole.vtp")).read_bytes()
    path.write_bytes(whole[: len(whole) // 2])
    return path


def _points_only(car: pv.PolyData, path: Path) -> Path:
    return _write_vtp(pv.PolyData(np.asarray(car.points)), path)


def _strips(car: pv.PolyData, path: Path) -> Path:
    stripper = vtkStripper()
    stripper.SetInputData(car.triangulate())
    stripper.Update()
    return _write_vtp(pv.wrap(stripper.GetOutput()), path)


def _volume(car: pv.PolyData, path: Path) -> Path:
    path = path.with_suffix(".vtu")
    pv.UnstructuredGrid(car.cast_to_unstructured_grid()).save(path)
    return path


def _legacy(car: pv.PolyData, path: Path) -> Path:
    path = path.with_suffix(".vtk")
    car.save(path)
    return path


def _empty(car: pv.PolyData, path: Path) -> Path:
    path.write_bytes(b"")
    return path


REFUSALS = {
    "empty": (_empty, "The file is empty"),
    "truncated": (_truncated, "The file ends early"),
    "lz4": (lambda car, path: _write_vtp(car, path, compressor="lz4"), "compressed with LZ4"),
    "big-endian": (lambda car, path: _write_vtp(car, path, big_endian=True), "big-endian"),
    "two-pieces": (lambda car, path: _write_vtp(car, path, pieces=2), "several pieces"),
    "points-only": (_points_only, "holds points but no polygons"),
    "triangle-strips": (_strips, "triangle strips"),
    "unstructured-grid": (_volume, "UnstructuredGrid file, not a surface"),
    "legacy": (_legacy, "legacy VTK file"),
}


@requires_runtime
@pytest.mark.parametrize("case", REFUSALS, ids=list(REFUSALS))
def test_reader_names_what_is_wrong_with_a_file_it_refuses(car: pv.PolyData, tmp_path: Path, case: str) -> None:
    """Catches a refusal that crashes or gives another problem's message."""
    make, message = REFUSALS[case]
    outcome = _run_engine("parse", make(car, tmp_path / "car.vtp"), tmp_path / "out")
    assert message in outcome.get("error", ""), outcome


@requires_runtime
def test_frontal_area_does_not_depend_on_how_finely_the_surface_is_meshed(tmp_path: Path) -> None:
    """Catches a frontal area drawn from the surface's points alone, which a coarse mesh's outline
    leaks through: a box of 12 triangles must give the box's frontal area, as the reference does."""
    box = pv.Box(bounds=(0.0, 4.0, -0.9, 0.9, 0.0, 1.4)).triangulate()
    box.points = np.asarray(box.points, dtype=np.float32)
    area = _run_engine("parse", _write_vtp(box, tmp_path / "box.vtp"), tmp_path / "out")["frontalArea"]
    # Within the one pixel row and column the 5 mm grid adds along the outline.
    assert area == pytest.approx(1.8 * 1.4, rel=0.01)
    assert area == pytest.approx(estimate_frontal_area(box), rel=0.01)


@requires_runtime
def test_engine_geometry_features_and_networks_match_python(car: pv.PolyData, model_dir: Path, tmp_path: Path) -> None:
    """Catches a browser feature column, network weight, spread scale, layer shown by `explain` or
    polygon measure that drifts from the Python pipeline the networks were trained with."""
    path = _write_vtp(car, tmp_path / "car.vtp")
    out = tmp_path / "out"
    outcome = _run_engine("engine", path, out, model_dir)

    surface = read_vtp_surface(path)
    assert surface.normals_flipped
    assert outcome["flip"] == surface.normals_flipped
    np.testing.assert_allclose(_array(out, "centers.f32").reshape(-1, 3), surface.centers, atol=1e-6)
    np.testing.assert_allclose(_array(out, "normals.f32").reshape(-1, 3), surface.normals, atol=1e-6)
    np.testing.assert_allclose(_array(out, "areas.f64"), surface.areas, rtol=1e-6)

    features = extract_query_features(SurfaceContext.from_surface(surface), surface.centers, surface.normals)
    browser_features = _array(out, "features.f32").reshape(features.shape)
    worst = np.abs(browser_features - features).max(axis=0)
    assert worst.max() < 1e-4, f"feature columns {np.flatnonzero(worst >= 1e-4).tolist()} differ by up to {worst.max():.3g}"

    ensemble = SurrogateEnsemble.load(CHECKPOINT, device=torch.device("cpu"))
    expected = ensemble.predict_features(features)
    np.testing.assert_allclose(_array(out, "models.f32").reshape(expected.cp_per_fold.shape), expected.cp_per_fold, atol=1e-4)
    np.testing.assert_allclose(_array(out, "sigma.f32"), expected.cp_std, atol=1e-4)
    # An example pack keeps the lookups exactly, so `explain` on an example sees the same inputs.
    np.testing.assert_array_equal(_array(out, "features_from_example.f32"), _array(out, "features.f32"))

    # `explain` gives the same Cp to the bit, and each network's layers as PyTorch computes them.
    browser_models = _array(out, "models.f32").reshape(expected.cp_per_fold.shape)
    np.testing.assert_array_equal(_array(out, "explained_models.f32"), browser_models[:, 0])
    np.testing.assert_allclose(_array(out, "activations.f32"), _layers(ensemble, browser_features[0]), atol=1e-5)


def _layers(ensemble: SurrogateEnsemble, features: np.ndarray) -> np.ndarray:
    """Each network's standardized, clipped input and its hidden layers after GELU, back to back."""
    layers = []
    with torch.no_grad():
        for fold in ensemble.folds:
            x = torch.from_numpy((features - fold.x_mean) / fold.x_std).clamp(-Z_CLIP, Z_CLIP)
            layers.append(x)
            for module in fold.model.net:
                x = module(x)
                if isinstance(module, torch.nn.GELU):
                    layers.append(x)
    return torch.cat(layers).numpy()


@requires_runtime
def test_a_second_analysis_waits_for_the_first_to_end(car: pv.PolyData, tmp_path: Path) -> None:
    """Catches an engine that abandons a car when the next file is dropped: the first car must still
    end with its exact numbers and complete full-mesh fields, and the second start only after that."""
    events = _run_engine("sequence", _write_vtp(car, tmp_path / "car.vtp"), tmp_path / "out")["events"]
    for name in ("first", "second"):
        own = [event for event in events if event["car"] == name]
        kinds = [event["event"] for event in own]
        assert "rejected" not in kinds, f"the {name} analysis was abandoned: {own}"
        assert "result" in kinds and "exact" in kinds, f"the {name} analysis did not finish: {kinds}"
        assert own[-1] == {"car": name, "event": "fullFields", "fraction": 1}, f"the {name} car's last event: {own[-1]}"
    cars = [event["car"] for event in events]
    assert cars == sorted(cars), "the two analyses' events interleave: one ran before the other had ended"


def _drag_by_facing(surface: SurfaceMesh, cp: np.ndarray, frontal_area: float) -> dict[str, float]:
    """`forces.byFacing` as result.js defines it, from Cp at every polygon."""
    nx = surface.normals[:, 0].astype(np.float64)
    drag = -np.asarray(cp, dtype=np.float64) * nx * surface.areas / frontal_area
    return {"forward": drag[nx < 0].sum(), "rearward": drag[nx >= 0].sum()}


def _assert_drag_by_facing(forces: dict, expected: dict[str, float]) -> None:
    # Exact against exact: the engine's Cp at each polygon is within about 2e-6 of Python's.
    assert forces["byFacing"]["forward"] == pytest.approx(expected["forward"], abs=2e-5)
    assert forces["byFacing"]["rearward"] == pytest.approx(expected["rearward"], abs=2e-5)
    assert forces["byFacing"]["forward"] + forces["byFacing"]["rearward"] == pytest.approx(forces["cdP"], abs=1e-12)


def _assert_totals(numbers: dict, surface: SurfaceMesh, cp: np.ndarray, frontal_area: float) -> None:
    """`totals` as result.js defines it, against Cp at every polygon."""
    totals, cp = numbers["totals"], np.asarray(cp, dtype=np.float64)
    assert totals["area"] == pytest.approx(surface.areas.sum(), rel=1e-9)
    assert totals["meanCp"] == pytest.approx((cp * surface.areas).sum() / surface.areas.sum(), abs=1e-5)
    force = -(cp[:, None] * surface.normals * surface.areas[:, None]).sum(axis=0) / frontal_area
    np.testing.assert_allclose(totals["coefficients"], force, atol=2e-5)
    assert totals["coefficients"][0] == numbers["forces"]["cdP"]
    for name, index in (("cpMin", cp.argmin()), ("cpMax", cp.argmax())):
        assert totals[name]["value"] == pytest.approx(cp[index], abs=1e-4)
        np.testing.assert_allclose(totals[name]["center"], surface.centers[index], atol=1e-6)
        np.testing.assert_allclose(totals[name]["normal"], surface.normals[index], atol=1e-5)


@requires_runtime
def test_drag_split_and_totals_follow_their_definitions(car: pv.PolyData, tmp_path: Path) -> None:
    """Catches a drag split or whole-car totals that sum the wrong polygons, weights, normals or field,
    or extremes placed at another polygon than the one where Cp is lowest or highest."""
    path = _write_vtp(car, tmp_path / "car.vtp")
    outcome = _run_engine("analysis", path, tmp_path / "out")
    reference = run_vtp_inference(path, SurrogateEnsemble.load(CHECKPOINT, device=torch.device("cpu")))
    surface = read_vtp_surface(path)
    expected = _drag_by_facing(surface, reference.cp_mean, outcome["geometry"]["frontalArea"])
    _assert_drag_by_facing(outcome["exact"]["forces"], expected)
    _assert_totals(outcome["exact"], surface, reference.cp_mean, outcome["geometry"]["frontalArea"])


def _assert_polygon(polygon: dict, c: int, mesh: pv.PolyData, surface: SurfaceMesh, cp: np.ndarray, sigma: np.ndarray) -> None:
    assert polygon["index"] == c
    np.testing.assert_array_equal(np.reshape(polygon["corners"], (-1, 3)), mesh.get_cell(c).points.astype(np.float32))
    np.testing.assert_allclose(polygon["center"], surface.centers[c], atol=1e-6)
    np.testing.assert_allclose(polygon["normal"], surface.normals[c], atol=1e-5)
    assert polygon["cp"] == pytest.approx(cp[c], abs=1e-4)
    assert polygon["sigma"] == pytest.approx(sigma[c], abs=1e-4)


@requires_runtime
def test_polygon_at_gives_the_polygon_under_a_pick(car: pv.PolyData, tmp_path: Path) -> None:
    """Catches `polygonAt` or `explain` on the file analysed last choosing another polygon than the
    one under the point, outlining it with other corners than the file's, giving other values than
    the prediction there (before or after the background prediction reaches it), or `explain`
    showing CFD from elsewhere than that polygon."""
    path = _write_vtp(car, tmp_path / "car.vtp")
    probes = _run_engine("analysis", path, tmp_path / "out")["probes"]
    reference = run_vtp_inference(path, SurrogateEnsemble.load(CHECKPOINT, device=torch.device("cpu")))
    mesh, surface = pv.read(path), read_vtp_surface(path)
    for probe in probes:
        assert probe["explained"] == {"index": probe["target"], "cfd": float(mesh.cell_data["CpMeanTrim"][probe["target"]])}
        for polygon in (probe["early"], probe["late"]):
            _assert_polygon(polygon, probe["target"], mesh, surface, reference.cp_mean, reference.cp_std)


@requires_runtime
def test_polygons_asked_on_demand_match_the_reference(car: pv.PolyData, model_dir: Path, tmp_path: Path) -> None:
    """Catches the networks asked on demand, at a polygon the background prediction has not reached,
    giving other values than the prediction there."""
    path = _write_vtp(car, tmp_path / "car.vtp")
    picks = _run_engine("pick", path, tmp_path / "out", model_dir)["picks"]
    reference = run_vtp_inference(path, SurrogateEnsemble.load(CHECKPOINT, device=torch.device("cpu")))
    mesh, surface = pv.read(path), read_vtp_surface(path)
    for pick in picks:
        _assert_polygon(pick["polygon"], pick["target"], mesh, surface, reference.cp_mean, reference.cp_std)


STRETCH = 1.1


def _stretched(car: pv.PolyData, factor: float) -> pv.PolyData:
    """The car stretched lengthwise as the runner's `stretch` mode does it: each float32 x times the
    factor in float64, stored as float32."""
    stretched = car.copy()
    points = np.array(stretched.points)
    points[:, 0] = (points[:, 0].astype(np.float64) * factor).astype(np.float32)
    stretched.points = points
    return stretched


@requires_runtime
def test_a_surface_given_as_arrays_is_analysed_as_its_file(car: pv.PolyData, tmp_path: Path) -> None:
    """Catches `analyzeSurface` analysing a car made on the page otherwise than `analyze` analyses the
    same car written to a file; its export, `fileCells` or `workingCells` going wrong for such a car;
    or arrays it must refuse getting through."""
    stretched = _write_vtp(_stretched(car, STRETCH), tmp_path / "stretched_car.vtp")
    from_arrays = _run_engine("stretch", _write_vtp(car, tmp_path / "car.vtp"), tmp_path / "arrays", str(STRETCH))
    from_file = _run_engine("analysis", stretched, tmp_path / "file")
    assert from_arrays["form"] == "full"
    for part in ("geometry", "sampled", "exact"):
        assert from_arrays[part] == from_file[part], part
    assert from_arrays["moved"], "the arrays were copied to the engine instead of moved"
    refusals = from_arrays["refusals"]
    assert "whole buffer of its own" in refusals["viewOfLargerBuffer"]
    assert "offsets end at" in refusals["offsetsShort"]
    assert "refers to point" in refusals["pointMissing"]

    out = tmp_path / "arrays"
    reference = run_vtp_inference(stretched, SurrogateEnsemble.load(CHECKPOINT, device=torch.device("cpu")))
    surface = read_vtp_surface(stretched)
    np.testing.assert_allclose(_array(out, "file_cp.f32"), reference.cp_mean, atol=1e-4)
    np.testing.assert_allclose(_array(out, "file_sigma.f32"), reference.cp_std, atol=1e-4)
    np.testing.assert_allclose(_array(out, "file_centers.f32").reshape(-1, 3), surface.centers, atol=1e-6)
    np.testing.assert_allclose(_array(out, "file_normals.f32").reshape(-1, 3), surface.normals, atol=1e-5)
    np.testing.assert_allclose(_array(out, "file_areas.f64"), surface.areas, rtol=1e-5)

    assert from_arrays["exportName"] == "stretched_car_prediction.vtp"
    expected, exported = pv.read(stretched), _read_without_complaint(out / "exported.vtp")
    np.testing.assert_array_equal(exported.points, expected.points)
    np.testing.assert_array_equal(exported.faces, expected.faces)
    np.testing.assert_array_equal(exported.cell_data["CpPredicted"], _array(out, "file_cp.f32"))
    np.testing.assert_array_equal(exported.cell_data["CpSpread"], _array(out, "file_sigma.f32"))
    np.testing.assert_array_equal(exported.cell_data["CpMeanTrim"], expected.cell_data["CpMeanTrim"])

    # The working mesh's triangles as VTK measures them, each with the mean Cp of its corners.
    triangles = _array(out, "working_indices.u32").reshape(-1, 3).astype(np.int64)
    working = pv.PolyData(_array(out, "working_positions.f32").reshape(-1, 3), np.column_stack([np.full(len(triangles), 3), triangles]).ravel())
    np.testing.assert_allclose(_array(out, "working_areas.f64"), working.compute_cell_sizes(length=False, volume=False).cell_data["Area"], rtol=1e-5)
    np.testing.assert_allclose(_array(out, "working_centers.f32").reshape(-1, 3), working.cell_centers().points, atol=1e-6)
    normals = working.compute_normals(cell_normals=True, point_normals=False, consistent_normals=False, auto_orient_normals=False).cell_data["Normals"]
    np.testing.assert_allclose(_array(out, "working_normals.f32").reshape(-1, 3), normals, atol=1e-5)
    np.testing.assert_allclose(_array(out, "working_cp.f32"), _array(out, "working_vertex_cp.f32")[triangles].mean(axis=1), atol=1e-6)


@requires_runtime
def test_quick_prediction_matches_the_reference_at_every_polygon(car: pv.PolyData, tmp_path: Path) -> None:
    """Catches `predictQuick` giving other values than the reference at the polygons it is given,
    summing its forces or totals over other polygons, weights or field, drawing other than the whole
    surface, or leaving the car held as if it were an analysis."""
    path = _write_vtp(car, tmp_path / "car.vtp")
    outcome = _run_engine("quick", path, tmp_path / "out")
    reference = run_vtp_inference(path, SurrogateEnsemble.load(CHECKPOINT, device=torch.device("cpu")))
    out, surface = tmp_path / "out", read_vtp_surface(path)
    np.testing.assert_allclose(_array(out, "cp.f32"), reference.cp_mean, atol=1e-4)
    np.testing.assert_allclose(_array(out, "sigma.f32"), reference.cp_std, atol=1e-4)
    _assert_drag_by_facing(outcome["forces"], _drag_by_facing(surface, reference.cp_mean, outcome["geometry"]["frontalArea"]))
    _assert_totals(outcome, surface, reference.cp_mean, outcome["geometry"]["frontalArea"])
    corners = np.diff(vtk_to_numpy(car.GetPolys().GetOffsetsArray()))
    assert outcome["form"] == "quick"
    assert outcome["reference"] == {"file": "car.vtp", "array": "CpMeanTrim", "mapping": "own"}
    assert (outcome["triangles"], outcome["vertices"]) == (int((corners - 2).sum()), car.n_points)
    assert not outcome["exportable"] and outcome["sameCells"]


@requires_runtime
def test_an_attached_reference_answers_as_the_file_own_cfd(car: pv.PolyData, tmp_path: Path) -> None:
    """Catches a CFD field attached from a second file giving other numbers or fields than the same
    field read with the car's own file (agreement, CFD drag, working-mesh field, file polygons,
    export), mapping a file of other polygons onto the wrong polygons or onto the far side of a thin
    panel, taking another car's field, a refusal that does not say why, or a detach that leaves the
    field behind."""
    case = car.copy()
    case.cell_data.clear()
    bare = case.copy()
    bare.cell_data["Velocity"] = np.zeros((bare.n_cells, 3), dtype=np.float32)
    # The car triangulated, with the pentagon fin's back face added on top of it: the nearest polygon
    # to the fin, at no distance, but facing away from it.
    fin = car.get_cell(int(np.flatnonzero(np.diff(vtk_to_numpy(car.GetPolys().GetOffsetsArray())) == 5)[0]))
    back = pv.PolyData(fin.points, faces=[5, 4, 3, 2, 1, 0])
    back.cell_data["CpMeanTrim"] = np.full(1, 99.0, dtype=np.float32)
    append = vtkAppendPolyData()
    for part in (car.triangulate(), back):
        append.AddInputData(part)
    append.Update()
    remeshed = pv.wrap(append.GetOutput())
    # Another car: this one 2% larger, every face 15 to 45 mm out from the car's.
    other = car.copy()
    other.points = np.asarray(car.center + 1.02 * (car.points - car.center), dtype=np.float32)
    surfaces = ((car, "car.vtp"), (remeshed, "remeshed.vtp"), (bare, "bare.vtp"), (other, "other.vtp"))
    files = [_write_vtp(surface, tmp_path / name) for surface, name in surfaces]
    outcome = _run_engine("reference", _write_vtp(case, tmp_path / "case.vtp"), tmp_path / "out", ",".join(map(str, files)))
    out, cfd = tmp_path / "out", np.asarray(car.cell_data["CpMeanTrim"])

    alone = outcome["alone"]
    assert alone["reference"] == {"file": "car.vtp", "array": "CpMeanTrim", "mapping": "own"}
    assert outcome["before"] == {"reference": None, "cdPCfd": None, "accuracy": None, "meshHasCfd": False}
    assert outcome["names"] == {"withCfd": ["CpMeanTrim", "pMeanTrim"], "bare": []}
    assert "no cell array of one number per polygon" in outcome["refusals"]["bare"]
    assert "other.vtp is not this car's surface: its polygons lie a median" in outcome["refusals"]["otherCar"]
    assert "CpMissing" in outcome["refusals"]["missing"] and "CpMeanTrim, pMeanTrim" in outcome["refusals"]["missing"]
    assert outcome["progressed"]

    same = outcome["same"]
    assert same == {**alone, "reference": {"file": "car.vtp", "array": "CpMeanTrim", "mapping": "polygons"}}
    np.testing.assert_array_equal(_array(out, "same_mesh_cfd.f32"), _array(out, "alone_mesh_cfd.f32"))
    np.testing.assert_array_equal(_array(out, "same_returned_cfd.f32"), _array(out, "alone_mesh_cfd.f32"))
    np.testing.assert_array_equal(_array(out, "same_file_cfd.f32"), cfd)
    np.testing.assert_array_equal(_read_without_complaint(out / "exported.vtp").cell_data["CpMeanTrim"], cfd)

    # Each polygon takes the value of the remeshed polygon whose centre is nearest among those not facing away.
    polygons, triangles = read_vtp_surface(tmp_path / "case.vtp"), read_vtp_surface(files[1])
    distances = np.linalg.norm(polygons.centers[:, None, :].astype(np.float64) - triangles.centers[None, :, :], axis=2)
    distances[polygons.normals @ triangles.normals.T < 0] = np.inf
    expected = np.asarray(remeshed.cell_data["CpMeanTrim"])[distances.argmin(axis=1)]
    np.testing.assert_array_equal(_array(out, "nearest_file_cfd.f32"), expected)
    nearest = outcome["nearest"]
    assert nearest["reference"] == {"file": "remeshed.vtp", "array": "CpMeanTrim", "mapping": "nearest"}
    drag = -(expected.astype(np.float64) * polygons.normals[:, 0] * polygons.areas).sum() / outcome["geometry"]["frontalArea"]
    assert nearest["cdPCfd"] == pytest.approx(drag, abs=1e-6)

    assert outcome["detached"] == outcome["before"]
    assert outcome["fileCellsAfterDetach"] is None


@requires_runtime
def test_a_bundled_example_is_explained_at_a_vertex(tmp_path: Path) -> None:
    """Catches `explain` failing, or `polygonAt` answering, for a bundled example opened before any
    file was analysed, when the engine holds no polygons."""
    outcome = _run_engine("example", REPO / "web/examples/run_1", tmp_path / "out")
    assert outcome["explained"]["at"] == "vertex"
    assert outcome["explained"]["cfd"] == outcome["vertexCfd"]
    assert outcome["polygon"] is None


RUN_1 = REPO / "data/raw/holdout/run_1/boundary_1.vtp"


@requires_runtime
@pytest.mark.skipif(not RUN_1.is_file(), reason="needs DrivAerML run 1 in data/raw")
def test_a_pick_on_run_1_takes_a_few_milliseconds(tmp_path: Path) -> None:
    """Catches the polygon lookup on a real 8.8M-polygon file slowing back down (a scan of every
    polygon took 75 ms a pick; asking the networks a block at a time, 10 ms) or missing the polygon
    under the point."""
    picks = _run_engine("pick", RUN_1, tmp_path / "out", REPO / "web/model")["picks"]
    assert [pick["polygon"]["index"] for pick in picks] == [pick["target"] for pick in picks]
    assert np.percentile([pick["seconds"] for pick in picks], 95) < 0.010


@pytest.mark.skipif(not RUN_1.is_file(), reason="needs DrivAerML run 1 in data/raw")
def test_bundled_run_1_drag_split_matches_the_reference() -> None:
    """Catches the bundled run 1 example's drag split drifting from the Python reference prediction
    on every polygon of the real file."""
    summary = json.loads((REPO / "web/examples/run_1.json").read_text())
    reference = run_vtp_inference(RUN_1, SurrogateEnsemble.load(CHECKPOINT))
    expected = _drag_by_facing(read_vtp_surface(RUN_1), reference.cp_mean, summary["geometry"]["frontalArea"])
    _assert_drag_by_facing(summary["forces"], expected)


def _read_without_complaint(path: Path) -> pv.PolyData:
    with pv.VtkErrorCatcher() as catcher:
        mesh = pv.read(path)
    assert not catcher.events, f"VTK complained reading {path.name}: {catcher.events}"
    return mesh


@requires_runtime
@pytest.mark.parametrize("encoding", ENCODINGS, ids=list(ENCODINGS))
def test_exported_vtp_is_the_file_with_the_prediction_added(car: pv.PolyData, tmp_path: Path, encoding: str) -> None:
    """Catches an export that changes the file's own geometry or arrays, stores the prediction in
    another polygon order than the file's, or writes a .vtp that VTK reads with complaints."""
    path = _write_vtp(car, tmp_path / "car.vtp", **ENCODINGS[encoding])
    outcome = _run_engine("export", path, tmp_path / "out")
    assert outcome == {"name": "car_prediction.vtp", "exportableEarly": False, "exportable": True}

    original = _read_without_complaint(path)
    exported = _read_without_complaint(tmp_path / "out" / "exported.vtp")
    np.testing.assert_array_equal(exported.points, original.points)
    np.testing.assert_array_equal(exported.faces, original.faces)
    assert set(exported.cell_data) == set(original.cell_data) | {"CpPredicted", "CpSpread"}
    for name in original.cell_data:
        np.testing.assert_array_equal(exported.cell_data[name], original.cell_data[name])

    surface = read_vtp_surface(path)
    features = extract_query_features(SurfaceContext.from_surface(surface), surface.centers, surface.normals)
    expected = SurrogateEnsemble.load(CHECKPOINT, device=torch.device("cpu")).predict_features(features)
    np.testing.assert_allclose(exported.cell_data["CpPredicted"], expected.cp_mean, atol=1e-4)
    np.testing.assert_allclose(exported.cell_data["CpSpread"], expected.cp_std, atol=1e-4)


@requires_runtime
@pytest.mark.parametrize("encoding", ["inline-plain-uint64", "appended-base64-zlib-uint32"])
def test_exporting_an_exported_file_replaces_its_prediction(car: pv.PolyData, tmp_path: Path, encoding: str) -> None:
    """Catches a second export, of an exported file re-saved as ParaView would (its prediction inline
    or appended), that adds another `CpPredicted` beside the first, which VTK would read in its place."""
    first = tmp_path / "first"
    _run_engine("export", _write_vtp(car, tmp_path / "car.vtp", **ENCODINGS[encoding]), first)
    once = _read_without_complaint(first / "exported.vtp")
    second = tmp_path / "second"
    _run_engine("export", _write_vtp(once, tmp_path / "resaved.vtp", **ENCODINGS[encoding]), second)

    text = (second / "exported.vtp").read_bytes()
    for name in ("CpPredicted", "CpSpread"):
        assert text.count(f'Name="{name}"'.encode()) == 1, f"{name} is in the file more than once"
    again = _read_without_complaint(second / "exported.vtp")
    np.testing.assert_allclose(again.cell_data["CpPredicted"], once.cell_data["CpPredicted"], atol=1e-6)
    np.testing.assert_array_equal(again.cell_data["CpMeanTrim"], once.cell_data["CpMeanTrim"])


@requires_runtime
def test_written_vtp_reads_back_in_vtk_with_the_same_cells_and_arrays(car: pv.PolyData, tmp_path: Path) -> None:
    """Catches `writeVtp` writing a file VTK reads otherwise than it was given (other points or
    polygons, arrays of another type, width or order) or reads with complaints."""
    path = _write_vtp(car, tmp_path / "car.vtp")
    outcome = _run_engine("write", path, tmp_path / "out")
    assert outcome == {"name": "written.vtp", "readBack": True}
    original, written = pv.read(path), _read_without_complaint(tmp_path / "out" / "written.vtp")
    np.testing.assert_array_equal(written.points, original.points)
    np.testing.assert_array_equal(written.faces, original.faces)
    assert written.cell_data["CpMeanTrim"].dtype == np.float32
    np.testing.assert_array_equal(written.cell_data["CpMeanTrim"], original.cell_data["CpMeanTrim"])
    assert written.cell_data["centers"].dtype == np.float64
    np.testing.assert_allclose(written.cell_data["centers"], read_vtp_surface(path).centers, atol=1e-6)
    assert written.cell_data["corners"].dtype == np.uint8
    np.testing.assert_array_equal(written.cell_data["corners"], np.diff(vtk_to_numpy(original.GetPolys().GetOffsetsArray())))
    np.testing.assert_array_equal(written.point_data["height"], original.points[:, 2])


@requires_runtime
def test_features_follow_the_model_files_definition(car: pv.PolyData, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches browser features that assume the shipped definition: with two voxel scales and two
    neighbour slices instead (125 columns) they must still match Python's column for column."""
    monkeypatch.setattr(feature_module, "VOXEL_SCALES", (0.05, 0.25))
    monkeypatch.setattr(feature_module, "DY_NEIGHBOR", (-0.2, 0.2))
    monkeypatch.setattr(feature_module, "FEATURE_COLUMNS", FeatureColumns(regional=slice(10, 40), slices=slice(40, 125), total_dim=125))
    definition = tmp_path / "definition.json"
    definition.write_text(json.dumps(feature_definition()))
    path = _write_vtp(car, tmp_path / "car.vtp")
    out = tmp_path / "out"
    outcome = _run_engine("features", path, out, definition)

    surface = read_vtp_surface(path)
    expected = extract_query_features(SurfaceContext.from_surface(surface), surface.centers, surface.normals)
    assert outcome["featureCount"] == expected.shape[1] == 125
    worst = np.abs(_array(out, "features.f32").reshape(expected.shape) - expected).max(axis=0)
    assert worst.max() < 1e-4, f"feature columns {np.flatnonzero(worst >= 1e-4).tolist()} differ by up to {worst.max():.3g}"


def test_exported_model_files_are_reproducible_and_current(tmp_path: Path) -> None:
    """Catches an exporter whose output varies between runs, and `web/model/` left stale after retraining."""
    first, second = tmp_path / "first", tmp_path / "second"
    export_web_model(CHECKPOINT, first)
    export_web_model(CHECKPOINT, second)
    for name in ("ensemble.json", "weights.bin"):
        assert (first / name).read_bytes() == (second / name).read_bytes(), f"{name} differs between two exports"
        assert (first / name).read_bytes() == (REPO / "web/model" / name).read_bytes(), (
            f"web/model/{name} is not the export of {CHECKPOINT.name}: run `uv run python -m surrogate.export_web`"
        )


def test_exporter_refuses_a_checkpoint_without_a_spread_scale(tmp_path: Path) -> None:
    """Catches an exporter that would ship the networks' raw standard deviation as the spread when
    the checkpoint predates the fitted spread scale."""
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    del payload["spread_scale"]
    (tmp_path / "metrics.json").write_bytes((CHECKPOINT.parent / "metrics.json").read_bytes())
    torch.save(payload, tmp_path / "ensemble.pt")
    with pytest.raises(ValueError, match="retrain"):
        export_web_model(tmp_path / "ensemble.pt", tmp_path / "model")


def test_bundled_examples_match_the_model_and_fit_the_host() -> None:
    """Catches bundled examples left from an earlier model or feature definition (their numbers and
    `explain` would disagree with a fresh analysis), and any file of the site over the host's 25 MB limit."""
    weights_sha256 = hashlib.sha256((REPO / "web/model/weights.bin").read_bytes()).hexdigest()
    examples = sorted((REPO / "web/examples").glob("*.json"))
    assert examples
    for example in examples:
        summary = json.loads(example.read_text())
        stale = f"web/examples/{example.stem} was computed with another model: run `uv run python -m surrogate.write_examples`"
        assert summary["model"] == weights_sha256, stale
        assert summary["lookups"]["definition"] == feature_definition(), stale
    oversized = [str(path.relative_to(REPO)) for path in (REPO / "web").rglob("*") if path.is_file() and path.stat().st_size > 25_000_000]
    assert not oversized, f"over 25 MB: {oversized}"
