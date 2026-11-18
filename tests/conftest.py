"""Fixtures shared by the test modules."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import pyvista as pv
import torch
from vtkmodules.vtkFiltersCore import vtkAppendPolyData

from surrogate.features import FEATURE_COLUMNS
from surrogate.model import FoldModel, PerCellMLP, SurrogateEnsemble


def _synthetic_car() -> pv.PolyData:
    """A car-sized closed body of quads with a tapered, cambered roof, a triangulated wheel, and a
    pentagon and a hexagon fin; wound inward as DrivAerML's files are, with CFD pressure and a
    solver field the engine must ignore."""
    body = pv.Box(bounds=(-0.75, 3.9, -0.95, 0.95, -0.3, 1.2), level=12, quads=True)
    points = np.asarray(body.points, dtype=np.float64)
    along = (points[:, 0] + 0.75) / 4.65
    points[:, 2] = (
        -0.3 + (points[:, 2] + 0.3) * (1.0 - 0.3 * along**2) + 0.08 * np.sin(np.pi * along) * (points[:, 2] > 0.5)
    )
    points[:, 1] *= 1.0 - 0.12 * np.sin(np.pi * along)
    body.points = points
    wheel = pv.Sphere(radius=0.3, center=(0.6, 0.8, 0.0), theta_resolution=16, phi_resolution=10)
    angles = np.linspace(0.0, 2.0 * np.pi, 6)[:-1]
    pentagon = pv.PolyData(
        np.column_stack([3.4 + 0.2 * np.cos(angles), np.full(5, 0.3), 1.0 + 0.2 * np.sin(angles)]),
        faces=[5, 0, 1, 2, 3, 4],
    )
    angles = np.linspace(0.0, 2.0 * np.pi, 7)[:-1]
    hexagon = pv.PolyData(
        np.column_stack([3.4 + 0.2 * np.cos(angles), np.full(6, -0.3), 1.0 + 0.2 * np.sin(angles)]),
        faces=[6, 0, 1, 2, 3, 4, 5],
    )

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


@pytest.fixture(scope="session")
def synthetic_car() -> Callable[[], pv.PolyData]:
    """The builder of the small synthetic car; each call gives a fresh surface a test may change."""
    return _synthetic_car


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
