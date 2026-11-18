"""The ensemble's checkpoint, inference on a surface file without CFD and with it, and the coverage check."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import pyvista as pv
import torch

from surrogate.features import FEATURE_COLUMNS
from surrogate.inference import run_vtp_inference
from surrogate.model import SurrogateEnsemble


def test_ensemble_checkpoint_roundtrip_preserves_mean_and_spread(
    tmp_path: Path, dummy_ensemble: SurrogateEnsemble
) -> None:
    """Catches state_dict or scaler drift when saving/loading `ensemble.pt`."""
    rng = np.random.default_rng(123)
    x = rng.normal(size=(512, FEATURE_COLUMNS.total_dim)).astype(np.float32)

    before = dummy_ensemble.predict_features(x, batch_size=128)
    ckpt = tmp_path / "ensemble.pt"
    dummy_ensemble.save(ckpt)

    reloaded = SurrogateEnsemble.load(ckpt, device=torch.device("cpu"))
    after = reloaded.predict_features(x, batch_size=128)

    np.testing.assert_allclose(after.cp_mean, before.cp_mean, atol=1e-6)
    np.testing.assert_allclose(after.cp_std, before.cp_std, atol=1e-6)
    assert float(after.cp_std.mean()) > 0.0


def test_inference_without_cfd_reports_a_car_longer_than_the_training_cars_by_its_length(
    tmp_path: Path, dummy_ensemble: SurrogateEnsemble
) -> None:
    """Catches crashes on a file without Cp or any solver array, and a 7 m car not reported outside the
    training cars, or reported with a reason that does not name the dimension that is out of range."""
    long_box = pv.Box(bounds=(-1.0, 6.0, -1.0, 1.0, -0.318, 1.10)).triangulate().subdivide(3)
    vtp_path = tmp_path / "long.vtp"
    long_box.save(vtp_path)

    res = run_vtp_inference(vtp_path, dummy_ensemble)

    assert res.has_ground_truth is False
    assert res.ground_truth_forces is None
    assert res.ground_truth_metrics is None
    assert res.cp_mean.shape == (res.n_cells,)
    assert res.cp_std.shape == (res.n_cells,)
    assert np.isfinite(res.cp_mean).all()
    assert res.coverage["status"] == "outside"
    assert "length outside the training cars' range" in res.coverage["reasons"]
    assert [d["dimension"] for d in res.coverage["out_of_range_dimensions"]] == ["length"]


def test_moving_a_car_off_the_training_cars_ground_height_leaves_its_coverage_unchanged(
    tmp_path: Path, dummy_ensemble: SurrogateEnsemble
) -> None:
    """Catches the coverage check judging where a car sits rather than its shape: the inputs are taken in
    the car's own frame, so a car of the training cars' size raised 0.5 m must get the same coverage, with
    no dimension out of range."""
    coverage = []
    for z_ground in (-0.318, 0.182):
        box = pv.Box(bounds=(-0.8, 3.8, -0.98, 0.98, z_ground, z_ground + 1.4)).triangulate().subdivide(3)
        box.save(tmp_path / f"car_{z_ground}.vtp")
        coverage.append(run_vtp_inference(tmp_path / f"car_{z_ground}.vtp", dummy_ensemble).coverage)

    unmoved, moved = coverage
    assert moved["out_of_range_dimensions"] == []
    assert (moved["status"], moved["reasons"]) == (unmoved["status"], unmoved["reasons"])


def test_uniform_pressure_on_a_closed_body_gives_zero_drag(
    tmp_path: Path, dummy_ensemble: SurrogateEnsemble
) -> None:
    """Catches drag estimated from a sample of cells: uniform pressure on a closed surface has no
    net force, which only the sum over every cell reproduces (120,000 random cells give 0.002 here)."""
    for fold in dummy_ensemble.folds:
        torch.nn.init.zeros_(fold.model.net[-1].weight)  # every network now predicts its bias everywhere
    body = pv.Sphere(radius=0.7, center=(1.5, 0.0, 0.4), theta_resolution=60, phi_resolution=60)
    body.points[:, 0] *= 3.0  # car-length ellipsoid with cells of very unequal size
    vtp_path = tmp_path / "closed_body.vtp"
    body.save(vtp_path)

    res = run_vtp_inference(vtp_path, dummy_ensemble)

    assert float(np.ptp(res.cp_mean)) < 1e-6
    assert abs(float(res.cp_mean[0])) > 0.1
    assert abs(res.forces["cd_p"]) < 1e-4
    assert res.drag_buildup[-1]["cd_p_cum_pred"] == pytest.approx(res.forces["cd_p"], abs=1e-4)
