"""The per-cell network, the five-network ensemble with its spread, and the check of a car against the
training cars."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from surrogate.features import SurfaceContext, extract_query_features
from surrogate.surface import compute_pressure_forces

# Each input is standardised by the training cells' mean and standard deviation, then clipped to this many
# standard deviations.
Z_CLIP = 6.0
# The shipped network: three hidden layers of 256.
HIDDEN_DIM = 256
DEPTH = 3
# Threads computing features for a whole file. For 8.8M cells: 36 s on one, 13 s on four,
# and no faster on eight, where the networks on the GPU set the pace.
FEATURE_THREADS = 4
# The coverage check. A car is outside the training cars when one of its dimensions lies beyond their range
# by more than OUTSIDE_EXCURSION of that range, or all three lie beyond it; when more than OUTSIDE_SLICE_SHARE
# of its slice-area curve S(x) lies outside their band; or when its spread is more than OUTSIDE_SPREAD_RATIO
# times the ensemble's mean spread on the training cars. It is near them when any dimension lies beyond the
# range, or S(x) or the spread passes the NEAR_ limits.
DIMENSIONS = ("length", "width", "height")
OUTSIDE_EXCURSION = 0.25
OUTSIDE_SLICE_SHARE = 0.25
OUTSIDE_SPREAD_RATIO = 1.6
NEAR_SLICE_SHARE = 0.08
NEAR_SPREAD_RATIO = 1.25
# S(x) is compared at the stations where the car or a training car has more than SLICE_COMPARED_AREA (m^2),
# and counts as outside the band when it is more than SLICE_BAND_MARGIN (m^2) beyond it.
SLICE_COMPARED_AREA = 0.05
SLICE_BAND_MARGIN = 0.03


def select_device() -> torch.device:
    """Choose MPS on Apple Silicon, CUDA when available, otherwise CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class PerCellMLP(nn.Module):
    """The network asked at each cell: the cell's inputs in, its Cp out, GELU between linear layers."""

    def __init__(self, in_dim: int, hidden_dim: int = HIDDEN_DIM, depth: int = DEPTH) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(depth):
            layers.extend([nn.Linear(dim, hidden_dim), nn.GELU()])
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PointNetGlobalMLP(nn.Module):
    """The PointNet ablation: a whole-car vector learned from the car's cells, pooled by mean and maximum,
    given to each cell beside its position and facing. Not part of the shipped model."""

    def __init__(self, in_dim: int, global_dim: int = 64, hidden_dim: int = 256) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Linear(64, global_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(in_dim + 2 * global_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pool_points: torch.Tensor, query_points: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(pool_points)
        g_vec = torch.cat([encoded.mean(dim=1), encoded.amax(dim=1)], dim=-1)
        g_expanded = g_vec.unsqueeze(1).expand(-1, query_points.shape[1], -1)
        return self.head(torch.cat([query_points, g_expanded], dim=-1)).squeeze(-1)


@dataclass(frozen=True)
class FoldModel:
    """One of the five networks, with the input and output scaling of the cars it was trained on."""

    fold_index: int
    train_runs: tuple[int, ...]
    val_runs: tuple[int, ...]
    x_mean: np.ndarray     # (F,) float32
    x_std: np.ndarray      # (F,) float32
    y_mean: float
    y_std: float
    model: PerCellMLP


@dataclass(frozen=True)
class EnsemblePrediction:
    """The ensemble's mean Cp, its spread σ(Cp), and each network's Cp."""

    cp_mean: np.ndarray      # (M,) float32
    cp_std: np.ndarray       # (M,) float32, the networks' standard deviation times the ensemble's spread_scale
    cp_per_fold: np.ndarray  # (K, M) float32


class SurrogateEnsemble:
    """The five networks, their mean and spread, and the training cars' envelope the coverage check reads.

    The networks train on overlapping cars and share blind spots, so their raw standard
    deviation ranks where the error is large but understates its size. `spread_scale` is the
    factor, measured in training on cars no network saw, that makes one `cp_std` cover the
    error about two times in three; 1.0 leaves the raw standard deviation.
    """

    def __init__(
        self,
        folds: list[FoldModel],
        *,
        in_dim: int,
        hidden_dim: int,
        depth: int,
        training_envelope: dict[str, Any],
        spread_scale: float = 1.0,
        device: torch.device | None = None,
    ) -> None:
        if not folds:
            raise ValueError("SurrogateEnsemble requires at least one FoldModel")
        self.device = device or select_device()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.training_envelope = training_envelope
        self.spread_scale = spread_scale
        self.folds: list[FoldModel] = []
        self._scalers: list[tuple[torch.Tensor, torch.Tensor]] = []
        for f in folds:
            f.model.to(self.device)
            f.model.eval()
            self.folds.append(f)
            self._scalers.append(
                (torch.as_tensor(f.x_mean, device=self.device), torch.as_tensor(f.x_std, device=self.device))
            )

    def _predict_per_fold(self, features: np.ndarray, batch_size: int) -> np.ndarray:
        """Each fold model's Cp over `[M, F]` features, `(K, M)`, in bounded-memory chunks."""
        if features.ndim != 2 or features.shape[1] != self.in_dim:
            raise ValueError(f"Expected features shape (M, {self.in_dim}), got {features.shape}")
        per_fold = np.empty((len(self.folds), features.shape[0]), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, features.shape[0], batch_size):
                x = torch.as_tensor(features[start : start + batch_size], dtype=torch.float32, device=self.device)
                for k, (fold, (x_mean, x_std)) in enumerate(zip(self.folds, self._scalers, strict=True)):
                    z = ((x - x_mean) / x_std).clamp(-Z_CLIP, Z_CLIP)
                    per_fold[k, start : start + batch_size] = (fold.model(z) * fold.y_std + fold.y_mean).cpu().numpy()
        return per_fold

    def _summarize(self, per_fold: np.ndarray) -> EnsemblePrediction:
        return EnsemblePrediction(
            cp_mean=per_fold.mean(axis=0, dtype=np.float32),
            cp_std=per_fold.std(axis=0, dtype=np.float32) * np.float32(self.spread_scale),
            cp_per_fold=per_fold,
        )

    def predict_features(self, features: np.ndarray, *, batch_size: int = 65_536) -> EnsemblePrediction:
        """Run all fold models over `[M, F]` features."""
        return self._summarize(self._predict_per_fold(features, batch_size))

    def predict_query_points(
        self,
        context: SurfaceContext,
        query_centers: np.ndarray,
        query_normals: np.ndarray,
        *,
        chunk: int = 250_000,
    ) -> EnsemblePrediction:
        """Extract the geometric features at `query_centers` and run the ensemble, chunk by chunk.

        A whole file is millions of cells, so the feature matrices of at most `FEATURE_THREADS`
        chunks exist at a time.
        """

        def features_at(start: int) -> np.ndarray:
            return extract_query_features(
                context, query_centers[start : start + chunk], query_normals[start : start + chunk]
            )

        starts = list(range(0, len(query_centers), chunk))
        per_fold = np.empty((len(self.folds), len(query_centers)), dtype=np.float32)
        with ThreadPoolExecutor(FEATURE_THREADS) as pool:
            for first in range(0, len(starts), FEATURE_THREADS):
                group = starts[first : first + FEATURE_THREADS]
                for start, features in zip(group, pool.map(features_at, group), strict=True):
                    per_fold[:, start : start + chunk] = self._predict_per_fold(features, chunk)
        return self._summarize(per_fold)

    def integrated_forces(
        self,
        prediction: EnsemblePrediction,
        normals: np.ndarray,
        areas: np.ndarray | None,
        *,
        total_area: float,
        frontal_area: float,
    ) -> dict[str, Any]:
        """Integrate pressure drag and lift of the mean field and of each fold's field.

        The folds are given one by one, not as a standard deviation: across cars no fold saw,
        the drag error is often several times the folds' spread, so it is no error bar.
        """
        per_fold = [
            compute_pressure_forces(cp, normals, areas, total_area=total_area, ref_area=frontal_area)
            for cp in prediction.cp_per_fold
        ]
        mean = compute_pressure_forces(
            prediction.cp_mean, normals, areas, total_area=total_area, ref_area=frontal_area
        )
        return {
            "cd_p": mean["cd_p"],
            "cd_p_per_fold": [f["cd_p"] for f in per_fold],
            "cl_p": mean["cl_p"],
            "cl_p_per_fold": [f["cl_p"] for f in per_fold],
        }

    def assess_coverage(
        self,
        context: SurfaceContext,
        cp_std: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """The car against the training cars: its length, width and height against their range, its
        slice-area curve S(x) against their band, and its spread against the ensemble's mean spread on them
        (see the limits above). `weights` are the cells' areas when `cp_std` covers every cell; omit them
        for an area-weighted sample."""
        dimensions = context.shape_descriptor()
        bounds = self.training_envelope["descriptor_bounds"]
        out_of_range: list[dict[str, Any]] = []
        excursion = 0.0
        for name in DIMENSIONS:
            value, low, high = dimensions[name], float(bounds[name]["min"]), float(bounds[name]["max"])
            if low <= value <= high:
                continue
            excursion = max(excursion, (low - value if value < low else value - high) / (high - low))
            side = "below" if value < low else "above"
            out_of_range.append({"dimension": name, "value": round(value, 4), "min": low, "max": high, "side": side})

        s_curve = context.slice_area
        s_min = np.asarray(self.training_envelope["slice_area_min"], dtype=np.float32)
        s_max = np.asarray(self.training_envelope["slice_area_max"], dtype=np.float32)
        compared = (s_curve > SLICE_COMPARED_AREA) | (s_max > SLICE_COMPARED_AREA)
        s_curve, s_min, s_max = s_curve[compared], s_min[compared], s_max[compared]
        slice_outside = float(np.mean((s_curve < s_min - SLICE_BAND_MARGIN) | (s_curve > s_max + SLICE_BAND_MARGIN)))

        w = np.ones(len(cp_std)) if weights is None else np.asarray(weights, dtype=np.float64)
        mean_cp_std = float(np.average(cp_std, weights=w))
        order = np.argsort(cp_std)
        cumulative = np.cumsum(w[order])
        p95_cp_std = float(cp_std[order][np.searchsorted(cumulative, 0.95 * cumulative[-1])])
        spread_ratio = mean_cp_std / float(self.training_envelope["cv_mean_cp_std"])

        reasons: list[str] = []
        if out_of_range:
            names = [d["dimension"] for d in out_of_range]
            listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
            reasons.append(f"{listed} outside the training cars' range")
        if slice_outside > NEAR_SLICE_SHARE:
            reasons.append(f"{slice_outside * 100:.0f}% of the slice-area curve outside the training cars' band")
        if spread_ratio > NEAR_SPREAD_RATIO:
            reasons.append(f"spread {spread_ratio:.2f} times its mean on the training cars")

        if (
            excursion > OUTSIDE_EXCURSION
            or len(out_of_range) == len(DIMENSIONS)
            or slice_outside > OUTSIDE_SLICE_SHARE
            or spread_ratio > OUTSIDE_SPREAD_RATIO
        ):
            status, summary = "outside", "outside the training cars"
        elif reasons:
            status, summary = "near", "at the edge of the training cars"
        else:
            status, summary = "inside", "size, slice-area curve and spread within the training cars'"

        return {
            "status": status,
            "summary": summary,
            "reasons": reasons,
            "out_of_range_dimensions": out_of_range,
            "slice_area_outside_fraction": round(slice_outside, 4),
            "mean_cp_std": round(mean_cp_std, 4),
            "p95_cp_std": round(p95_cp_std, 4),
            "spread_vs_training_ratio": round(spread_ratio, 3),
        }

    def save(self, path: str | Path) -> None:
        """Write the networks, their scaling, the spread factor and the training envelope to `path`."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "in_dim": self.in_dim,
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "training_envelope": self.training_envelope,
            "spread_scale": self.spread_scale,
            "folds": [
                {
                    "fold_index": f.fold_index,
                    "train_runs": list(f.train_runs),
                    "val_runs": list(f.val_runs),
                    "x_mean": f.x_mean,
                    "x_std": f.x_std,
                    "y_mean": f.y_mean,
                    "y_std": f.y_std,
                    "state_dict": {k: v.cpu() for k, v in f.model.state_dict().items()},
                }
                for f in self.folds
            ],
        }
        torch.save(payload, out_path)

    @classmethod
    def load(cls, path: str | Path, *, device: torch.device | None = None) -> SurrogateEnsemble:
        """Read an ensemble written by `save`, onto `device` (the best one here when omitted)."""
        ckpt_path = Path(path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Ensemble checkpoint not found: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        in_dim = int(payload["in_dim"])
        hidden_dim = int(payload["hidden_dim"])
        depth = int(payload["depth"])
        folds: list[FoldModel] = []
        for item in payload["folds"]:
            model = PerCellMLP(in_dim=in_dim, hidden_dim=hidden_dim, depth=depth)
            model.load_state_dict(item["state_dict"])
            folds.append(
                FoldModel(
                    fold_index=int(item["fold_index"]),
                    train_runs=tuple(int(r) for r in item["train_runs"]),
                    val_runs=tuple(int(r) for r in item["val_runs"]),
                    x_mean=np.asarray(item["x_mean"], dtype=np.float32),
                    x_std=np.asarray(item["x_std"], dtype=np.float32),
                    y_mean=float(item["y_mean"]),
                    y_std=float(item["y_std"]),
                    model=model,
                )
            )
        return cls(
            folds=folds,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            training_envelope=payload["training_envelope"],
            spread_scale=float(payload["spread_scale"]),
            device=device,
        )
