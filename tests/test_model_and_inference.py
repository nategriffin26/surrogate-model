"""Model, ensemble save/load, coverage diagnostic, and cold geometry-only inference tests."""

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
    ens = dummy_ensemble
    rng = np.random.default_rng(123)
    x = rng.normal(size=(512, FEATURE_COLUMNS.total_dim)).astype(np.float32)

    before = ens.predict_features(x, batch_size=128)
    ckpt = tmp_path / "ensemble.pt"
    ens.save(ckpt)

    reloaded = SurrogateEnsemble.load(ckpt, device=torch.device("cpu"))
    after = reloaded.predict_features(x, batch_size=128)

    np.testing.assert_allclose(after.cp_mean, before.cp_mean, atol=1e-6)
    np.testing.assert_allclose(after.cp_std, before.cp_std, atol=1e-6)
    assert float(after.cp_std.mean()) > 0.0


def test_cold_vtp_inference_succeeds_without_cp_and_flags_out_of_domain_geometry(
    tmp_path: Path, dummy_ensemble: SurrogateEnsemble
) -> None:
    """Catches crashes on files without Cp, and an oversized body not being reported as outside the training cars."""
    ens = dummy_ensemble

    # Create an oversized 7.0 m long geometry-only VTP with NO solver arrays at all
    long_box = pv.Box(bounds=(-1.0, 6.0, -1.0, 1.0, -0.318, 1.10)).triangulate().subdivide(3)
    vtp_path = tmp_path / "unseen_oversized.vtp"
    long_box.save(vtp_path)

    res = run_vtp_inference(vtp_path, ens)

    assert res.has_ground_truth is False
    assert res.ground_truth_forces is None
    assert res.ground_truth_metrics is None
    assert res.cp_mean.shape == (res.n_cells,)
    assert res.cp_std.shape == (res.n_cells,)
    assert np.isfinite(res.cp_mean).all()
    assert res.coverage["status"] in ("near", "outside")
    oob_features = {d["feature"] for d in res.coverage["out_of_range_dimensions"]}
    assert "length" in oob_features


def test_cold_vtp_inference_populates_ground_truth_when_cp_present(
    tmp_path: Path, dummy_ensemble: SurrogateEnsemble
) -> None:
    """Verifies the CFD comparison activates when CpMeanTrim exists."""
    ens = dummy_ensemble
    in_domain_box = pv.Box(bounds=(-0.8, 3.85, -0.98, 0.98, -0.318, 1.10)).triangulate().subdivide(3)
    centers = np.asarray(in_domain_box.cell_centers().points, dtype=np.float32)
    in_domain_box.cell_data["CpMeanTrim"] = (-0.25 + 0.1 * centers[:, 0]).astype(np.float32)
    vtp_path = tmp_path / "with_gt.vtp"
    in_domain_box.save(vtp_path)

    res = run_vtp_inference(vtp_path, ens)

    assert res.has_ground_truth is True
    assert res.ground_truth_forces is not None
    assert res.ground_truth_metrics is not None
    assert "r2" in res.ground_truth_metrics
    assert len(res.forces["cd_p_per_fold"]) == len(ens.folds)


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
