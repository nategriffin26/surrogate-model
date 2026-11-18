"""Fixtures shared by the test modules."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from surrogate.features import FEATURE_COLUMNS
from surrogate.model import FoldModel, PerCellMLP, SurrogateEnsemble


@pytest.fixture
def dummy_ensemble() -> SurrogateEnsemble:
    """Three small untrained fold models on the CPU, with a training envelope shaped like the real one."""
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
            "max_slice_area": {"min": 2.0, "max": 3.0, "mean": 2.5},
            "plan_volume_proxy": {"min": 6.5, "max": 9.5, "mean": 8.0},
        },
        "slice_area_min": [0.0] * 300,
        "slice_area_max": [3.5] * 300,
        "cv_mean_cp_std": 0.05,
    }
    return SurrogateEnsemble(
        folds,
        in_dim=FEATURE_COLUMNS.total_dim,
        hidden_dim=32,
        depth=2,
        training_envelope=envelope,
        spread_scale=2.5,
        device=torch.device("cpu"),
    )
