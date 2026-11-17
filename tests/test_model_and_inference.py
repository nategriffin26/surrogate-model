"""Model, ensemble save/load, coverage diagnostic, and cold geometry-only inference tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import torch

from surrogate.features import FEATURE_COLUMNS
from surrogate.inference import run_vtp_inference
from surrogate.model import FoldModel, PerCellMLP, SurrogateEnsemble


def _make_dummy_ensemble(device: torch.device | None = None) -> SurrogateEnsemble:
    device = device or torch.device("cpu")
    folds: list[FoldModel] = []
    for k in range(3):
        torch.manual_seed(10 + k)
        net = PerCellMLP(in_dim=FEATURE_COLUMNS.total_dim, hidden_dim=32, depth=2)
        folds.append(
            FoldModel(
                fold_index=k,
                train_runs=(80, 98, 102, 208),
                val_runs=(213, 281),
                x_mean=np.zeros(FEATURE_COLUMNS.total_dim, dtype=np.float32),
                x_std=np.ones(FEATURE_COLUMNS.total_dim, dtype=np.float32),
                y_mean=-0.30,
                y_std=0.35,
                model=net,
            )
        )
    envelope = {
        "train_runs": [80, 98, 102, 208, 213, 281, 397, 425, 445, 474],
        "descriptor_bounds": {
            "length": {"min": 4.30, "max": 4.85, "mean": 4.65},
            "width": {"min": 1.84, "max": 2.18, "mean": 2.02},
            "height": {"min": 1.29, "max": 1.53, "mean": 1.42},
            "z_min": {"min": -0.32, "max": -0.31, "mean": -0.318},
            "max_section_area": {"min": 2.0, "max": 3.0, "mean": 2.5},
            "plan_volume_proxy": {"min": 6.5, "max": 9.5, "mean": 8.0},
        },
        "section_area_min": [0.0] * 300,
        "section_area_max": [3.5] * 300,
        "cv_mean_cp_std": 0.05,
    }
    return SurrogateEnsemble(
        folds,
        in_dim=FEATURE_COLUMNS.total_dim,
        hidden_dim=32,
        depth=2,
        training_envelope=envelope,
        device=device,
    )


def test_ensemble_checkpoint_roundtrip_preserves_mean_and_spread(tmp_path: Path) -> None:
    """Catches state_dict or scaler drift when saving/loading `ensemble.pt`."""
    ens = _make_dummy_ensemble()
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
    tmp_path: Path,
) -> None:
    """Catches crashes on target-free VTP uploads and verifies OOD escalation on oversized bodies."""
    ens = _make_dummy_ensemble()

    # Create an oversized 7.0 m long geometry-only VTP with NO solver arrays at all
    long_box = pv.Box(bounds=(-1.0, 6.0, -1.0, 1.0, -0.318, 1.10)).triangulate().subdivide(3)
    vtp_path = tmp_path / "unseen_oversized.vtp"
    long_box.save(vtp_path)

    res = run_vtp_inference(vtp_path, ens, eval_samples=4096)

    assert res.has_ground_truth is False
    assert res.ground_truth_forces is None
    assert res.ground_truth_metrics is None
    assert res.display.cp_true is None
    assert res.display.cp_mean.shape == (res.n_display_faces,)
    assert res.display.cp_std.shape == (res.n_display_faces,)
    assert np.isfinite(res.display.cp_mean).all()
    assert res.coverage["status"] in ("caution", "escalate")
    oob_features = {d["feature"] for d in res.coverage["out_of_range_dimensions"]}
    assert "length" in oob_features


def test_cold_vtp_inference_populates_ground_truth_when_cp_present(tmp_path: Path) -> None:
    """Verifies reference metrics and display ground-truth mapping activate when CpMeanTrim exists."""
    ens = _make_dummy_ensemble()
    in_domain_box = pv.Box(bounds=(-0.8, 3.85, -0.98, 0.98, -0.318, 1.10)).triangulate().subdivide(3)
    centers = np.asarray(in_domain_box.cell_centers().points, dtype=np.float32)
    in_domain_box.cell_data["CpMeanTrim"] = (-0.25 + 0.1 * centers[:, 0]).astype(np.float32)
    vtp_path = tmp_path / "with_gt.vtp"
    in_domain_box.save(vtp_path)

    res = run_vtp_inference(vtp_path, ens, eval_samples=4096)

    assert res.has_ground_truth is True
    assert res.ground_truth_forces is not None
    assert res.ground_truth_metrics is not None
    assert res.display.cp_true is not None
    assert res.display.cp_true.shape == (res.n_display_faces,)
    assert "r2" in res.ground_truth_metrics
    assert res.forces["cd_p_std"] >= 0.0
