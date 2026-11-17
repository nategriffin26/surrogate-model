"""Surface pressure and integrated aerodynamic force evaluation metrics."""

from __future__ import annotations

import numpy as np

from surrogate.surface import compute_pressure_forces


def surface_field_metrics(cp_true: np.ndarray, cp_pred: np.ndarray) -> dict[str, float]:
    """Compute area-weighted surface pressure metrics on an area-weighted cell sample."""
    t = np.asarray(cp_true, dtype=np.float64)
    p = np.asarray(cp_pred, dtype=np.float64)
    err = p - t
    mse = float(np.mean(err * err))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    denom_var = float(np.sum((t - t.mean()) ** 2))
    r2 = float(1.0 - np.sum(err * err) / max(denom_var, 1e-12))
    rel_l2 = float(np.sqrt(np.sum(err * err) / max(float(np.sum(t * t)), 1e-12)))
    return {
        "r2": r2,
        "rel_l2": rel_l2,
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
    }


def case_evaluation_summary(
    cp_true: np.ndarray,
    cp_pred: np.ndarray,
    normals: np.ndarray,
    *,
    total_area: float,
    frontal_area: float,
    exact_cd_p: float | None = None,
) -> dict[str, float]:
    """Evaluate surface field metrics and integrated pressure drag/lift on one vehicle."""
    field = surface_field_metrics(cp_true, cp_pred)
    sample_true_forces = compute_pressure_forces(
        cp_true, normals, None, total_area=total_area, ref_area=frontal_area
    )
    pred_forces = compute_pressure_forces(
        cp_pred, normals, None, total_area=total_area, ref_area=frontal_area
    )
    ref_cd_p = sample_true_forces["cd_p"] if exact_cd_p is None else exact_cd_p
    cd_err = pred_forces["cd_p"] - sample_true_forces["cd_p"]
    return {
        **field,
        "cd_p_true_sample": sample_true_forces["cd_p"],
        "cd_p_true_exact": ref_cd_p,
        "cd_p_pred": pred_forces["cd_p"],
        "cd_p_err": cd_err,
        "cd_p_abs_err": abs(cd_err),
        "cl_p_true_sample": sample_true_forces["cl_p"],
        "cl_p_pred": pred_forces["cl_p"],
    }


def regional_r2_breakdown(
    cp_true: np.ndarray,
    cp_pred: np.ndarray,
    centers: np.ndarray,
    normals: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Break down surface metrics across aerodynamic facing zones and spanwise bands."""
    y_abs = np.abs(centers[:, 1])
    masks = {
        "forebody (nx < -0.7)": normals[:, 0] < -0.7,
        "base_wake (nx > 0.7)": normals[:, 0] > 0.7,
        "upper_body (nz > 0.7)": normals[:, 2] > 0.7,
        "underbody (nz < -0.7)": normals[:, 2] < -0.7,
        "side_panels (|ny| > 0.7)": np.abs(normals[:, 1]) > 0.7,
        "inboard (|y| < 0.3 m)": y_abs < 0.3,
        "outboard (|y| > 0.6 m)": y_abs > 0.6,
    }
    out: dict[str, dict[str, float]] = {}
    for name, mask in masks.items():
        if int(mask.sum()) == 0:
            continue
        m = surface_field_metrics(cp_true[mask], cp_pred[mask])
        out[name] = {
            "area_share": round(float(np.mean(mask)), 4),
            "r2": round(m["r2"], 4),
            "mae": round(m["mae"], 4),
            "bias": round(m["bias"], 4),
        }
    return out


def uncertainty_calibration_bins(
    cp_true: np.ndarray,
    cp_pred: np.ndarray,
    cp_std: np.ndarray,
    *,
    n_bins: int = 5,
) -> list[dict[str, float]]:
    """Compare ensemble spread quantiles against realized absolute Cp error."""
    abs_err = np.abs(np.asarray(cp_pred, dtype=np.float64) - np.asarray(cp_true, dtype=np.float64))
    spread = np.asarray(cp_std, dtype=np.float64)
    edges = np.quantile(spread, np.linspace(0.0, 1.0, n_bins + 1))
    bins: list[dict[str, float]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (spread >= lo) & (spread <= hi if i == n_bins - 1 else spread < hi)
        if int(mask.sum()) == 0:
            continue
        bins.append(
            {
                "bin": i + 1,
                "mean_std": round(float(spread[mask].mean()), 4),
                "realized_mae": round(float(abs_err[mask].mean()), 4),
                "realized_rmse": round(float(np.sqrt(np.mean(abs_err[mask] ** 2))), 4),
            }
        )
    return bins
