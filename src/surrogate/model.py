"""Per-cell neural network, 5-fold ensemble, and geometric coverage diagnostics."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from surrogate.features import FEATURE_COLUMNS, SurfaceContext, extract_query_features
from surrogate.surface import compute_pressure_forces

Z_CLIP = 6.0
# Threads computing features for a whole file. For 8.8M cells: 36 s on one, 13 s on four,
# and no faster on eight, where the networks on the GPU set the pace.
FEATURE_THREADS = 4


def select_device() -> torch.device:
    """Choose MPS on Apple Silicon, CUDA when available, otherwise CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class PerCellMLP(nn.Module):
    """Shared per-cell multilayer perceptron mapping deterministic surface features to Cp."""

    def __init__(self, in_dim: int = FEATURE_COLUMNS.total_dim, hidden_dim: int = 256, depth: int = 3) -> None:
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
    """Ablation model: appends a pooled whole-car PointNet vector `g` to local cell features.

    Included strictly for the controlled N=10 ablation demonstrating that learned
    whole-car latent vectors act as car-ID lookup keys when only 8 training vehicles
    exist per fold. Not shipped in the inference ensemble.
    """

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
    """One cross-validation fold checkpoint with training-only feature/target scalers."""

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
    """Mean prediction, expected error, and per-fold outputs."""

    cp_mean: np.ndarray      # (M,) float32
    cp_std: np.ndarray       # (M,) float32, fold standard deviation x the ensemble's spread_scale
    cp_per_fold: np.ndarray  # (K, M) float32


class SurrogateEnsemble:
    """5-fold Per-Cell MLP ensemble with training-domain geometric coverage checks.

    The fold models train on overlapping cars and share blind spots, so their raw standard
    deviation ranks where the error is large but understates its size. `spread_scale` is the
    factor, measured in training on cars no model saw, that makes one `cp_std` cover the
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
        """Evaluate geometric coverage and ensemble spread against the 10-car domain.

        Separates input-space extrapolation (vehicle dimensions & cross-section envelope S(x))
        from epistemic disagreement across the 5 cross-validation fold models. `weights` are
        the cells' areas when `cp_std` covers every cell; omit them for an area-weighted sample.
        """
        desc = context.shape_descriptor()
        bounds_env = self.training_envelope["descriptor_bounds"]
        out_of_range_dims: list[dict[str, Any]] = []
        max_norm_excursion = 0.0

        for key, stats in bounds_env.items():
            val = desc[key]
            lo, hi = float(stats["min"]), float(stats["max"])
            span = max(hi - lo, 1e-6)
            if val < lo:
                exc = (lo - val) / span
                max_norm_excursion = max(max_norm_excursion, exc)
                out_of_range_dims.append({"feature": key, "value": round(val, 4), "min": lo, "max": hi, "side": "below"})
            elif val > hi:
                exc = (val - hi) / span
                max_norm_excursion = max(max_norm_excursion, exc)
                out_of_range_dims.append({"feature": key, "value": round(val, 4), "min": lo, "max": hi, "side": "above"})

        # Compare streamwise cross-sectional area curve S(x) against 10-car min/max band
        s_curve = context.section_area_x
        s_min = np.asarray(self.training_envelope["section_area_min"], dtype=np.float32)
        s_max = np.asarray(self.training_envelope["section_area_max"], dtype=np.float32)
        active = (s_curve > 0.05) | (s_max > 0.05)
        if int(active.sum()) > 0:
            outside_band = (s_curve[active] < s_min[active] - 0.03) | (s_curve[active] > s_max[active] + 0.03)
            silhouette_out_frac = float(np.mean(outside_band))
        else:
            silhouette_out_frac = 1.0

        w = np.ones(len(cp_std)) if weights is None else np.asarray(weights, dtype=np.float64)
        mean_cp_std = float(np.average(cp_std, weights=w))
        order = np.argsort(cp_std)
        cumulative = np.cumsum(w[order])
        p95_cp_std = float(cp_std[order][np.searchsorted(cumulative, 0.95 * cumulative[-1])])
        ref_mean_std = float(self.training_envelope.get("cv_mean_cp_std", 0.035))
        spread_ratio = mean_cp_std / max(ref_mean_std, 1e-6)

        reasons: list[str] = []
        if max_norm_excursion > 0.25 or len(out_of_range_dims) >= 3:
            reasons.append(f"{len(out_of_range_dims)} vehicle dimension(s) outside 10-car training range")
        if silhouette_out_frac > 0.25:
            reasons.append(f"{silhouette_out_frac * 100:.0f}% of streamwise S(x) profile outside 10-car envelope")
        if spread_ratio > 1.6:
            reasons.append(f"Ensemble spread {spread_ratio:.2f}x above 10-car CV baseline")

        if reasons:
            status = "escalate"
            recommendation = "Outside 10-car training coverage — run full CFD verification before design sign-off."
        elif out_of_range_dims or silhouette_out_frac > 0.08 or spread_ratio > 1.25:
            status = "caution"
            if out_of_range_dims:
                names = ", ".join(d["feature"] for d in out_of_range_dims)
                reasons.append(f"Near/outside 10-car range on: {names}")
            if silhouette_out_frac > 0.08:
                reasons.append(f"{silhouette_out_frac * 100:.0f}% of S(x) stations outside 10-car band")
            if spread_ratio > 1.25:
                reasons.append(f"Moderate fold disagreement ({spread_ratio:.2f}x CV mean)")
            recommendation = "Directional pressure trends usable for screening; confirm top candidates in CFD."
        else:
            status = "pass"
            recommendation = "Within 10-car geometric range and ensemble spread baseline — suitable for rapid screening."

        return {
            "status": status,
            "recommendation": recommendation,
            "reasons": reasons,
            "out_of_range_dimensions": out_of_range_dims,
            "silhouette_outside_fraction": round(silhouette_out_frac, 4),
            "mean_cp_std": round(mean_cp_std, 4),
            "p95_cp_std": round(p95_cp_std, 4),
            "spread_vs_cv_ratio": round(spread_ratio, 3),
        }

    def save(self, path: str | Path) -> None:
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
