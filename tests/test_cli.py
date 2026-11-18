"""The `surrogate` command: `predict` on the small synthetic car, `fetch` against a local stand-in for the
dataset's server, and the messages a copy without data gets from `status` and `reproduce`."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv
from test_web_engine import _synthetic_car
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader

from surrogate import cli
from surrogate.model import SurrogateEnsemble


@pytest.fixture
def checkpoint(tmp_path: Path, dummy_ensemble: SurrogateEnsemble, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "ensemble.pt"
    dummy_ensemble.save(path)
    monkeypatch.setattr(cli, "CHECKPOINT", path)
    return path


def _cell_arrays(path: Path) -> dict[str, np.ndarray]:
    """Every cell array of a .vtp as VTK itself reads it."""
    reader = vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    cells = reader.GetOutput().GetCellData()
    return {cells.GetArrayName(i): vtk_to_numpy(cells.GetArray(i)) for i in range(cells.GetNumberOfArrays())}


def test_predict_writes_the_prediction_on_the_files_own_cells(tmp_path: Path, checkpoint: Path) -> None:
    """Catches a written file whose arrays sit on other cells than the file's own (a reordered or
    re-meshed surface), arrays under other names or types than the browser's export, and the file's own
    arrays dropped: the same car with its cells shuffled must get the same values, shuffled the same way."""
    car = _synthetic_car()
    order = np.random.default_rng(7).permutation(car.n_cells)
    faces = np.concatenate([[len(ids), *ids] for ids in (car.get_cell(i).point_ids for i in order)])
    shuffled = pv.PolyData(np.asarray(car.points), faces)
    for name in car.cell_data:
        shuffled.cell_data[name] = np.asarray(car.cell_data[name])[order]
    car.save(tmp_path / "car.vtp")
    shuffled.save(tmp_path / "shuffled.vtp")

    for name in ("car", "shuffled"):
        prediction = tmp_path / f"{name}_prediction.vtp"
        cli.main(["predict", str(tmp_path / f"{name}.vtp"), "--output", str(prediction), "--device", "cpu"])
    written = _cell_arrays(tmp_path / "car_prediction.vtp")
    reshuffled = _cell_arrays(tmp_path / "shuffled_prediction.vtp")

    assert set(written) == {"CpMeanTrim", "pMeanTrim", "CpPredicted", "CpSpread"}
    np.testing.assert_array_equal(written["CpMeanTrim"], np.asarray(car.cell_data["CpMeanTrim"]))
    for array in ("CpPredicted", "CpSpread"):
        assert written[array].dtype == np.float32 and written[array].shape == (car.n_cells,)
        np.testing.assert_allclose(reshuffled[array], written[array][order], atol=1e-4)
    assert float(written["CpSpread"].min()) > 0.0


def test_predict_compares_with_cfd_only_when_the_file_carries_it(
    tmp_path: Path, checkpoint: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches a summary that reports agreement with CFD for a file without `CpMeanTrim`, or omits it for
    one with, and a JSON summary that cannot be written (NumPy values json cannot encode)."""
    car = _synthetic_car()
    car.save(tmp_path / "with_cfd.vtp")
    car.clear_data()
    car.save(tmp_path / "geometry_only.vtp")

    cli.main(["predict", str(tmp_path / "with_cfd.vtp"), "--json", str(tmp_path / "with_cfd.json"), "--device", "cpu"])
    with_cfd = capsys.readouterr().out
    geometry_only_json = tmp_path / "geometry_only.json"
    cli.main(["predict", str(tmp_path / "geometry_only.vtp"), "--json", str(geometry_only_json), "--device", "cpu"])
    geometry_only = capsys.readouterr().out

    assert "CFD agreement" in with_cfd and "no CpMeanTrim" not in with_cfd
    assert "CFD agreement" not in geometry_only and "no CpMeanTrim" in geometry_only
    summary = json.loads((tmp_path / "with_cfd.json").read_text())
    assert summary["has_ground_truth"] is True and "r2" in summary["ground_truth_metrics"]
    assert len(summary["forces"]["cd_p_per_fold"]) == 3
    assert "cp_mean" not in summary
    assert json.loads((tmp_path / "geometry_only.json").read_text())["ground_truth_metrics"] is None


def test_predict_refuses_an_output_that_is_not_a_vtp(
    tmp_path: Path, checkpoint: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches `--output car.vtk` written silently in another format than the browser's export."""
    _synthetic_car().save(tmp_path / "car.vtp")
    with pytest.raises(SystemExit) as stopped:
        cli.main(["predict", str(tmp_path / "car.vtp"), "--output", str(tmp_path / "car.vtk")])
    assert stopped.value.code == 1
    assert "car.vtp" in capsys.readouterr().err
    assert not (tmp_path / "car.vtk").exists()


class StandIn:
    """A local stand-in for the dataset's file server: HEAD gives the size, GET honours a byte range
    unless told to ignore it, and can drop the connection part way through a file."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.ignore_range = False
        self.drop_after: int | None = None
        self.gets: list[tuple[str, str | None]] = []
        stand_in = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                pass

            def do_HEAD(self) -> None:
                self.send_response(200)
                self.send_header("Content-Length", str(len(stand_in.files[self.path])))
                self.end_headers()

            def do_GET(self) -> None:
                body = stand_in.files[self.path]
                requested = self.headers.get("Range")
                stand_in.gets.append((self.path, requested))
                if requested and not stand_in.ignore_range:
                    start = int(requested.removeprefix("bytes=").rstrip("-"))
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
                    body = body[start:]
                else:
                    self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body if stand_in.drop_after is None else body[: stand_in.drop_after])

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"


@pytest.fixture
def stand_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[StandIn]:
    rng = np.random.default_rng(3)
    files = {f"/run_{run}/boundary_{run}.vtp": rng.bytes(3_000_000 + run) for run in (1, 500)}
    server = StandIn(files)
    threading.Thread(target=server.server.serve_forever, daemon=True).start()
    monkeypatch.setattr(cli, "DATASET_URL", server.url)
    monkeypatch.setattr(cli, "REPO", tmp_path)
    monkeypatch.setattr(cli, "RAW", tmp_path / "data/raw")
    yield server
    server.server.shutdown()
    server.server.server_close()


def _fetched(tmp_path: Path, run: int) -> Path:
    return tmp_path / f"data/raw/holdout/run_{run}/boundary_{run}.vtp"


def test_fetch_writes_each_file_whole_and_skips_files_already_here(tmp_path: Path, stand_in: StandIn) -> None:
    """Catches a file left under its own name before it is complete, and a second fetch that downloads
    again what is already here."""
    cli.main(["fetch", "holdout"])
    for run in (1, 500):
        assert _fetched(tmp_path, run).read_bytes() == stand_in.files[f"/run_{run}/boundary_{run}.vtp"]
    assert not list(tmp_path.rglob("*.part"))
    stand_in.gets.clear()
    cli.main(["fetch", "holdout"])
    assert stand_in.gets == []


def test_fetch_continues_an_interrupted_download(
    tmp_path: Path, stand_in: StandIn, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catches a dropped connection reported as success or as a traceback, a part thrown away instead of
    continued, and a continuation that appends at the wrong offset."""
    stand_in.drop_after = 1_000_000
    with pytest.raises(SystemExit) as stopped:
        cli.main(["fetch", "holdout"])
    assert stopped.value.code == 1
    assert "Run the same command again" in capsys.readouterr().err
    part = _fetched(tmp_path, 1).with_name("boundary_1.vtp.part")
    assert part.stat().st_size == 1_000_000 and not _fetched(tmp_path, 1).exists()

    stand_in.drop_after = None
    stand_in.gets.clear()
    cli.main(["fetch", "holdout"])
    assert stand_in.gets[0] == ("/run_1/boundary_1.vtp", "bytes=1000000-")
    for run in (1, 500):
        assert _fetched(tmp_path, run).read_bytes() == stand_in.files[f"/run_{run}/boundary_{run}.vtp"]


def test_fetch_starts_over_when_the_server_ignores_the_range(tmp_path: Path, stand_in: StandIn) -> None:
    """Catches the whole file appended after a part when the server answers a range with all of it."""
    whole = stand_in.files["/run_1/boundary_1.vtp"]
    part = _fetched(tmp_path, 1).with_name("boundary_1.vtp.part")
    part.parent.mkdir(parents=True)
    part.write_bytes(whole[:500_000])
    stand_in.ignore_range = True
    cli.main(["fetch", "holdout"])
    assert _fetched(tmp_path, 1).read_bytes() == whole


@pytest.fixture
def empty_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """This repository as a fresh clone has it: the model and the tool, no DrivAerML files, nothing preprocessed."""
    monkeypatch.setattr(cli, "RAW", tmp_path / "data/raw")
    monkeypatch.setattr(cli, "PROCESSED", tmp_path / "data/processed")
    monkeypatch.setattr(cli, "MANIFEST", tmp_path / "data/processed/manifest.json")


def test_status_without_data_names_the_command_for_each_gap(empty_copy: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches status failing on a copy without data, or leaving a gap without the command that fills it."""
    monkeypatch.setattr(cli, "REPO", cli.RAW.parents[1])
    checks = {check.label: check for check in cli.status_checks()}
    assert checks["Training surfaces"].state == "0 of 10 in data/raw/train"
    assert checks["Training surfaces"].fix == "uv run surrogate fetch training"
    assert checks["Holdout surfaces"].fix == "uv run surrogate fetch holdout"
    assert checks["Preprocessed cars"].fix == "uv run surrogate preprocess"
    assert checks["Training cars"].fix is None


def test_reproduce_without_data_stops_before_any_step(empty_copy: None, capsys: pytest.CaptureFixture[str]) -> None:
    """Catches reproduce starting a step it cannot finish when the DrivAerML files are missing."""
    with pytest.raises(SystemExit) as stopped:
        cli.main(["reproduce"])
    assert stopped.value.code == 1
    captured = capsys.readouterr()
    assert "uv run surrogate fetch" in captured.err
    assert captured.out == ""
