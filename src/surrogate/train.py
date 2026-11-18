"""Training on the ten DrivAerML cars: the shipped five-network ensemble and its spread, evaluated in
cross-validation and on the two holdout cars, and the ablations and learning curve recorded beside it.

The ablations are a staircase that adds one block of inputs at a time (`features.BLOCKS`, starting from a
constant), closed by the shipped ensemble on every block, and two other learners: a learned whole-car
vector, and boosted trees on every input. Run by `uv run surrogate train`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from surrogate.baselines import BOOST_ROUNDS, lightgbm_predictions
from surrogate.dataset import ALLOWED_TRAIN_RUNS, HOLDOUT_RUNS
from surrogate.features import BLOCKS, FEATURE_COLUMNS
from surrogate.metrics import (
    case_evaluation_summary,
    regional_r2_breakdown,
    spread_coverage,
    surface_field_metrics,
    uncertainty_calibration_bins,
)
from surrogate.model import (
    DEPTH,
    HIDDEN_DIM,
    Z_CLIP,
    FoldModel,
    PerCellMLP,
    PointNetGlobalMLP,
    SurrogateEnsemble,
)
from surrogate.surface import compute_pressure_forces

# Five folds of two validation cars each; no fold splits a car's cells.
CAR_FOLDS: tuple[tuple[int, int], ...] = (
    (80, 98),
    (102, 208),
    (213, 281),
    (397, 425),
    (445, 474),
)
BATCH_SIZE = 8192
LEARNING_RATE = 5e-3
# The PointNet ablation: the width of its whole-car vector and its number of gradient steps.
POINTNET_DIM = 64
POINTNET_STEPS = 600
# Each block of inputs as the README's results table names it.
BLOCK_NAMES = {"local": "position and facing", "surface": "nearby surface", "slices": "slices across the car"}


def _load_processed_dataset(
    processed_dir: Path,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, dict[str, Any]]]:
    manifest_path = processed_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing {manifest_path}. Run `uv run surrogate preprocess` first.")
    manifest = json.loads(manifest_path.read_text())
    meta_by_run: dict[int, dict[str, Any]] = {int(c["run"]): c for c in manifest["cases"]}
    arrays_by_run: dict[int, dict[str, np.ndarray]] = {}
    for run_id in meta_by_run:
        npz_path = processed_dir / f"run_{run_id}.npz"
        with np.load(npz_path) as data:
            arrays_by_run[run_id] = {k: data[k] for k in data.files}
    return arrays_by_run, meta_by_run


def _others(held_out: tuple[int, ...]) -> tuple[int, ...]:
    """The training cars other than `held_out`."""
    return tuple(r for r in ALLOWED_TRAIN_RUNS if r not in held_out)


def _select_columns(features: np.ndarray, blocks: tuple[str, ...]) -> np.ndarray:
    return np.concatenate([features[:, getattr(FEATURE_COLUMNS, b)] for b in blocks], axis=1)


def _width(blocks: tuple[str, ...]) -> int:
    """The number of input columns in `blocks`."""
    return sum(getattr(FEATURE_COLUMNS, b).stop - getattr(FEATURE_COLUMNS, b).start for b in blocks)


def _fit_mlp_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    fold_index: int,
    train_runs: tuple[int, ...],
    val_runs: tuple[int, ...],
    epochs: int,
    seed: int,
    device: torch.device,
) -> FoldModel:
    torch.manual_seed(seed + fold_index)
    x_mean = x_train.mean(axis=0).astype(np.float32)
    x_std = (x_train.std(axis=0) + 1e-6).astype(np.float32)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std() + 1e-6)

    x_norm = np.clip((x_train - x_mean) / x_std, -Z_CLIP, Z_CLIP)
    y_norm = (y_train - y_mean) / y_std

    xt = torch.as_tensor(x_norm, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y_norm, dtype=torch.float32, device=device)

    model = PerCellMLP(in_dim=x_train.shape[1]).to(device)
    n_batches = max(1, len(xt) // BATCH_SIZE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, total_steps=epochs * n_batches)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(xt), device=device)
        for step in range(n_batches):
            batch_idx = perm[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]
            pred = model(xt[batch_idx])
            loss = F.mse_loss(pred, yt[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

    model.eval()
    return FoldModel(
        fold_index=fold_index,
        train_runs=train_runs,
        val_runs=val_runs,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        model=model,
    )


def _predict_single_fold(fold: FoldModel, x_eval: np.ndarray, device: torch.device) -> np.ndarray:
    n_pts = x_eval.shape[0]
    out = np.empty(n_pts, dtype=np.float32)
    batch_size = 65_536
    fold.model.eval()
    with torch.no_grad():
        for start in range(0, n_pts, batch_size):
            end = min(start + batch_size, n_pts)
            x_chunk = np.clip((x_eval[start:end] - fold.x_mean) / fold.x_std, -Z_CLIP, Z_CLIP)
            xt = torch.as_tensor(x_chunk, dtype=torch.float32, device=device)
            out[start:end] = fold.model(xt).cpu().numpy() * fold.y_std + fold.y_mean
    return out


def _train_folds(
    arrays_by_run: dict[int, dict[str, np.ndarray]],
    blocks: tuple[str, ...],
    *,
    epochs: int,
    seed: int,
    device: torch.device,
) -> tuple[list[FoldModel], dict[int, np.ndarray]]:
    """One network per fold of CAR_FOLDS on the inputs of `blocks`, and each training car's prediction by
    the network that did not see it."""
    folds: list[FoldModel] = []
    out_of_fold: dict[int, np.ndarray] = {}
    for fold_index, val_pair in enumerate(CAR_FOLDS):
        train_runs = _others(val_pair)
        fold = _fit_mlp_fold(
            np.concatenate([_select_columns(arrays_by_run[r]["features"], blocks) for r in train_runs], axis=0),
            np.concatenate([arrays_by_run[r]["cp"] for r in train_runs], axis=0),
            fold_index=fold_index,
            train_runs=train_runs,
            val_runs=val_pair,
            epochs=epochs,
            seed=seed,
            device=device,
        )
        folds.append(fold)
        for r in val_pair:
            out_of_fold[r] = _predict_single_fold(fold, _select_columns(arrays_by_run[r]["features"], blocks), device)
    return folds, out_of_fold


def _ensemble(
    folds: list[FoldModel], envelope: dict[str, Any], device: torch.device, spread_scale: float = 1.0
) -> SurrogateEnsemble:
    return SurrogateEnsemble(
        folds,
        in_dim=folds[0].x_mean.shape[0],
        hidden_dim=HIDDEN_DIM,
        depth=DEPTH,
        training_envelope=envelope,
        spread_scale=spread_scale,
        device=device,
    )


def _summarize_predictions(
    runs: tuple[int, ...],
    preds_by_run: dict[int, np.ndarray],
    arrays_by_run: dict[int, dict[str, np.ndarray]],
    meta_by_run: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    per_car: dict[str, dict[str, float]] = {}
    for r in runs:
        meta = meta_by_run[r]
        arr = arrays_by_run[r]
        summary = case_evaluation_summary(
            arr["cp"],
            preds_by_run[r],
            arr["normals"],
            total_area=float(meta["total_area"]),
            frontal_area=float(meta["frontal_area"]),
        )
        per_car[str(r)] = {k: round(v, 5) for k, v in summary.items()}

    r2_vals = [per_car[str(r)]["r2"] for r in runs]
    rel_l2_vals = [per_car[str(r)]["rel_l2"] for r in runs]
    mae_vals = [per_car[str(r)]["mae"] for r in runs]
    cd_abs_errs = [per_car[str(r)]["cd_p_abs_err"] for r in runs]
    cd_signed_errs = [per_car[str(r)]["cd_p_err"] for r in runs]
    cl_signed_errs = [per_car[str(r)]["cl_p_pred"] - per_car[str(r)]["cl_p_true_sample"] for r in runs]

    return {
        "mean_r2": round(float(np.mean(r2_vals)), 4),
        "worst_car_r2": round(float(np.min(r2_vals)), 4),
        "best_car_r2": round(float(np.max(r2_vals)), 4),
        "mean_rel_l2": round(float(np.mean(rel_l2_vals)), 4),
        "mean_mae": round(float(np.mean(mae_vals)), 4),
        "cd_p_mae": round(float(np.mean(cd_abs_errs)), 4),
        "cd_p_bias": round(float(np.mean(cd_signed_errs)), 4),
        "cl_p_mae": round(float(np.mean(np.abs(cl_signed_errs))), 4),
        "cl_p_bias": round(float(np.mean(cl_signed_errs)), 4),
        "per_car": per_car,
    }


def _print_scores(label: str, cv: dict[str, Any]) -> None:
    print(f"{label}: CV R² {cv['mean_r2']:.4f} (worst car {cv['worst_car_r2']:.4f}), Cd,p MAE {cv['cd_p_mae']:.4f}")


def _run_pointnet_global_ablation(
    arrays_by_run: dict[int, dict[str, np.ndarray]],
    meta_by_run: dict[int, dict[str, Any]],
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """The PointNet ablation on the same car folds, and trained on all ten cars for the holdout cars: each
    cell's position and facing, with a whole-car vector of POINTNET_DIM learned from the car's cells."""

    def _fit_and_eval(tr_runs: tuple[int, ...], te_runs: tuple[int, ...]) -> dict[int, np.ndarray]:
        torch.manual_seed(seed)
        x_cat = np.concatenate([arrays_by_run[r]["features"][:, FEATURE_COLUMNS.local] for r in tr_runs])
        y_cat = np.concatenate([arrays_by_run[r]["cp"] for r in tr_runs])
        mu = x_cat.mean(axis=0).astype(np.float32)
        sd = (x_cat.std(axis=0) + 1e-6).astype(np.float32)
        ym, ys = float(y_cat.mean()), float(y_cat.std() + 1e-6)

        car_x = {
            r: torch.as_tensor(
                np.clip((arrays_by_run[r]["features"][:, FEATURE_COLUMNS.local] - mu) / sd, -Z_CLIP, Z_CLIP),
                dtype=torch.float32,
                device=device,
            )
            for r in tr_runs + te_runs
        }
        car_y = {
            r: torch.as_tensor((arrays_by_run[r]["cp"] - ym) / ys, dtype=torch.float32, device=device) for r in tr_runs
        }

        net = PointNetGlobalMLP(in_dim=x_cat.shape[1], global_dim=POINTNET_DIM).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=POINTNET_STEPS)

        net.train()
        for _ in range(POINTNET_STEPS):
            xp_batch, xq_batch, yq_batch = [], [], []
            for r in tr_runs:
                n_cells = len(car_x[r])
                ip = torch.randint(0, n_cells, (4096,), device=device)
                iq = torch.randint(0, n_cells, (2048,), device=device)
                xp_batch.append(car_x[r][ip])
                xq_batch.append(car_x[r][iq])
                yq_batch.append(car_y[r][iq])
            pred = net(torch.stack(xp_batch), torch.stack(xq_batch))
            loss = F.mse_loss(pred, torch.stack(yq_batch))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

        net.eval()
        preds: dict[int, np.ndarray] = {}
        with torch.no_grad():
            for r in te_runs:
                xp = car_x[r][:32_768].unsqueeze(0)
                xq = car_x[r].unsqueeze(0)
                preds[r] = net(xp, xq).squeeze(0).cpu().numpy() * ys + ym
        return preds

    cv_preds: dict[int, np.ndarray] = {}
    for val_pair in CAR_FOLDS:
        cv_preds.update(_fit_and_eval(_others(val_pair), val_pair))
    holdout_preds = _fit_and_eval(ALLOWED_TRAIN_RUNS, HOLDOUT_RUNS)
    return {
        "cv": _summarize_predictions(ALLOWED_TRAIN_RUNS, cv_preds, arrays_by_run, meta_by_run),
        "holdout": _summarize_predictions(HOLDOUT_RUNS, holdout_preds, arrays_by_run, meta_by_run),
    }


def _build_training_envelope(
    arrays_by_run: dict[int, dict[str, np.ndarray]],
    meta_by_run: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """What the 10 allowed training runs span: their shape descriptors, their slice-area curves S(x),
    and each input's 2nd percentile, median and 98th percentile over their sampled cells."""
    desc_keys = list(meta_by_run[ALLOWED_TRAIN_RUNS[0]]["shape_descriptor"].keys())
    descriptor_bounds: dict[str, dict[str, float]] = {}
    for key in desc_keys:
        vals = [float(meta_by_run[r]["shape_descriptor"][key]) for r in ALLOWED_TRAIN_RUNS]
        descriptor_bounds[key] = {
            "min": round(float(min(vals)), 4),
            "max": round(float(max(vals)), 4),
            "mean": round(float(np.mean(vals)), 4),
        }

    slice_areas = np.stack([arrays_by_run[r]["slice_area"] for r in ALLOWED_TRAIN_RUNS], axis=0)
    training_inputs = np.concatenate([arrays_by_run[r]["features"] for r in ALLOWED_TRAIN_RUNS], axis=0)
    return {
        "train_runs": list(ALLOWED_TRAIN_RUNS),
        "descriptor_bounds": descriptor_bounds,
        "slice_area_min": np.round(slice_areas.min(axis=0), 4).tolist(),
        "slice_area_max": np.round(slice_areas.max(axis=0), 4).tolist(),
        "input_ranges": np.round(np.percentile(training_inputs, [2, 50, 98], axis=0).T, 4).tolist(),
    }


def _fit_spread_scale(
    arrays_by_run: dict[int, dict[str, np.ndarray]],
    *,
    epochs: int,
    seed: int,
    device: torch.device,
) -> float:
    """Factor that turns the fold models' standard deviation into a one-sigma error.

    Measuring it needs cars that none of the averaged models saw, and every shipped model saw
    eight of the ten. So the shipped structure is repeated one level down: for each held-out
    pair, four models are trained on six of the other eight cars each, and their standard
    deviation on the pair is compared with the error of their mean. The factor returned puts
    68% of those errors within one scaled standard deviation.
    """
    ratios: list[np.ndarray] = []
    for outer_idx, held_pair in enumerate(CAR_FOLDS):
        models: list[FoldModel] = []
        for inner_idx, inner_pair in enumerate(p for p in CAR_FOLDS if p != held_pair):
            tr_runs = _others(held_pair + inner_pair)
            models.append(
                _fit_mlp_fold(
                    np.concatenate([arrays_by_run[r]["features"] for r in tr_runs], axis=0),
                    np.concatenate([arrays_by_run[r]["cp"] for r in tr_runs], axis=0),
                    fold_index=inner_idx,
                    train_runs=tr_runs,
                    val_runs=inner_pair,
                    epochs=epochs,
                    seed=seed + 100 * (outer_idx + 1),
                    device=device,
                )
            )
        for r in held_pair:
            per_model = np.stack([_predict_single_fold(m, arrays_by_run[r]["features"], device) for m in models])
            abs_err = np.abs(per_model.mean(axis=0) - arrays_by_run[r]["cp"])
            ratios.append(abs_err / np.maximum(per_model.std(axis=0), 1e-9))
    return float(np.quantile(np.concatenate(ratios), 0.68))


def _training_mean_spread(ensemble: SurrogateEnsemble, arrays_by_run: dict[int, dict[str, np.ndarray]]) -> float:
    """The ensemble's mean spread on its own training cars, which `assess_coverage` compares a new car's
    spread with. Four of the five networks saw each car, so it is scaled by √(5/4)."""
    spreads = [
        float(ensemble.predict_features(arrays_by_run[r]["features"]).cp_std.mean() * np.sqrt(5.0 / 4.0))
        for r in ALLOWED_TRAIN_RUNS
    ]
    return round(float(np.mean(spreads)), 4)


def _evaluate_holdout(
    ensemble: SurrogateEnsemble,
    arrays_by_run: dict[int, dict[str, np.ndarray]],
    meta_by_run: dict[int, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, float]], dict[str, float]]:
    """The shipped ensemble on the holdout cars: its scores with each car's drag and lift by network, its
    spread against the realised error by spread quintile, and the share of cells within one and two spreads."""
    preds: dict[int, np.ndarray] = {}
    stds: dict[int, np.ndarray] = {}
    forces: dict[str, dict[str, Any]] = {}
    for r in HOLDOUT_RUNS:
        prediction = ensemble.predict_features(arrays_by_run[r]["features"])
        preds[r], stds[r] = prediction.cp_mean, prediction.cp_std
        car_forces = ensemble.integrated_forces(
            prediction,
            arrays_by_run[r]["normals"],
            None,
            total_area=float(meta_by_run[r]["total_area"]),
            frontal_area=float(meta_by_run[r]["frontal_area"]),
        )
        forces[str(r)] = {
            k: [round(x, 5) for x in v] if isinstance(v, list) else round(v, 5) for k, v in car_forces.items()
        }

    summary = _summarize_predictions(HOLDOUT_RUNS, preds, arrays_by_run, meta_by_run)
    summary["ensemble_forces"] = forces
    true = np.concatenate([arrays_by_run[r]["cp"] for r in HOLDOUT_RUNS])
    pred = np.concatenate([preds[r] for r in HOLDOUT_RUNS])
    std = np.concatenate([stds[r] for r in HOLDOUT_RUNS])
    return summary, uncertainty_calibration_bins(true, pred, std), spread_coverage(true, pred, std)


def _constant_baseline(
    arrays_by_run: dict[int, dict[str, np.ndarray]],
    meta_by_run: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Every car predicted at the mean Cp of the cars it was held out from, its drag and lift at their mean
    drag and lift; for the holdout cars, the means of all ten."""
    sampled = {
        r: compute_pressure_forces(
            arrays_by_run[r]["cp"],
            arrays_by_run[r]["normals"],
            None,
            total_area=float(meta_by_run[r]["total_area"]),
            ref_area=float(meta_by_run[r]["frontal_area"]),
        )
        for r in ALLOWED_TRAIN_RUNS + HOLDOUT_RUNS
    }

    def scores(train_runs: tuple[int, ...], runs: tuple[int, ...]) -> list[tuple[float, float, float, float]]:
        """(R², MAE, drag error, lift error) of each car in `runs`."""
        mean_cp = float(np.mean([arrays_by_run[r]["cp"].mean() for r in train_runs]))
        mean_cd = float(np.mean([sampled[r]["cd_p"] for r in train_runs]))
        mean_cl = float(np.mean([sampled[r]["cl_p"] for r in train_runs]))
        rows = []
        for r in runs:
            cp = arrays_by_run[r]["cp"]
            field = surface_field_metrics(cp, np.full_like(cp, mean_cp))
            rows.append((field["r2"], field["mae"], mean_cd - sampled[r]["cd_p"], mean_cl - sampled[r]["cl_p"]))
        return rows

    def summary(rows: list[tuple[float, float, float, float]]) -> dict[str, float]:
        r2, mae, cd_err, cl_err = (np.array(column) for column in zip(*rows, strict=True))
        return {
            "mean_r2": round(float(r2.mean()), 4),
            "worst_car_r2": round(float(r2.min()), 4),
            "mean_mae": round(float(mae.mean()), 4),
            "cd_p_mae": round(float(np.abs(cd_err).mean()), 4),
            "cd_p_bias": round(float(cd_err.mean()), 4),
            "cl_p_mae": round(float(np.abs(cl_err).mean()), 4),
            "cl_p_bias": round(float(cl_err.mean()), 4),
        }

    cv_rows = [row for val_pair in CAR_FOLDS for row in scores(_others(val_pair), val_pair)]
    return {
        "label": "Constant (fold mean Cp and Cd,p)",
        "blocks": [],
        "learner": "constant",
        "cv": summary(cv_rows),
        "holdout": summary(scores(ALLOWED_TRAIN_RUNS, HOLDOUT_RUNS)),
    }


def _input_staircase(
    arrays_by_run: dict[int, dict[str, np.ndarray]],
    meta_by_run: dict[int, dict[str, Any]],
    envelope: dict[str, Any],
    *,
    epochs: int,
    seed: int,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    """Five networks on the first block of inputs, then on the first two; the shipped ensemble on all three
    closes the staircase."""
    ablations: dict[str, dict[str, Any]] = {}
    for n_blocks in range(1, len(BLOCKS)):
        blocks = BLOCKS[:n_blocks]
        folds, out_of_fold = _train_folds(arrays_by_run, blocks, epochs=epochs, seed=seed, device=device)
        cv = _summarize_predictions(ALLOWED_TRAIN_RUNS, out_of_fold, arrays_by_run, meta_by_run)
        ensemble = _ensemble(folds, envelope, device)
        holdout_preds = {
            r: ensemble.predict_features(_select_columns(arrays_by_run[r]["features"], blocks)).cp_mean
            for r in HOLDOUT_RUNS
        }
        name = BLOCK_NAMES[blocks[-1]]
        label = f"{name.capitalize()} only" if n_blocks == 1 else f"+ {name}"
        label = f"{label} ({_width(blocks)})"
        ablations[f"{'_'.join(blocks)}_mlp"] = {
            "label": label,
            "blocks": list(blocks),
            "learner": "mlp",
            "cv": cv,
            "holdout": _summarize_predictions(HOLDOUT_RUNS, holdout_preds, arrays_by_run, meta_by_run),
        }
        _print_scores(label, cv)
    return ablations


def _other_learners(
    processed_dir: Path,
    arrays_by_run: dict[int, dict[str, np.ndarray]],
    meta_by_run: dict[int, dict[str, Any]],
    *,
    seed: int,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    """A learned whole-car vector beside position and facing, and boosted trees on every input."""
    pointnet = _run_pointnet_global_ablation(arrays_by_run, meta_by_run, seed=seed, device=device)
    pointnet_label = f"PointNet whole-car vector ({POINTNET_DIM}) + {BLOCK_NAMES['local']}"
    _print_scores(pointnet_label, pointnet["cv"])

    cv_preds, holdout_preds = lightgbm_predictions(processed_dir, ALLOWED_TRAIN_RUNS, CAR_FOLDS, HOLDOUT_RUNS)
    lightgbm = {
        "cv": _summarize_predictions(ALLOWED_TRAIN_RUNS, cv_preds, arrays_by_run, meta_by_run),
        "holdout": _summarize_predictions(HOLDOUT_RUNS, holdout_preds, arrays_by_run, meta_by_run),
    }
    lightgbm_label = f"LightGBM, {BOOST_ROUNDS} trees, all inputs ({FEATURE_COLUMNS.total_dim})"
    _print_scores(lightgbm_label, lightgbm["cv"])

    return {
        "pointnet_global_64d": {"label": pointnet_label, "blocks": ["local"], "learner": "pointnet", **pointnet},
        "lightgbm_full": {"label": lightgbm_label, "blocks": list(BLOCKS), "learner": "trees", **lightgbm},
    }


def _learning_curve(
    arrays_by_run: dict[int, dict[str, np.ndarray]],
    meta_by_run: dict[int, dict[str, Any]],
    *,
    epochs: int,
    seed: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Accuracy on unseen cars against the number of cars trained on, which says whether more CFD
    runs would still help. For each fold's held-out pair, single networks are trained on 2, 4, 6 and
    all 8 of the other cars, with as many gradient steps whatever the number of cars, and scored on
    the held-out pair only."""
    scores: dict[int, list[dict[str, float]]] = {2: [], 4: [], 6: [], 8: []}
    for fold_idx, held_pair in enumerate(CAR_FOLDS):
        others = _others(held_pair)
        pairs = [others[i : i + 2] for i in range(0, 8, 2)]
        subsets = {
            2: pairs,
            4: [pairs[0] + pairs[1], pairs[2] + pairs[3]],
            6: [tuple(r for r in others if r not in pair) for pair in pairs],
            8: [others],
        }
        for n_cars, groups in subsets.items():
            for group_idx, cars in enumerate(groups):
                model = _fit_mlp_fold(
                    np.concatenate([arrays_by_run[r]["features"] for r in cars], axis=0),
                    np.concatenate([arrays_by_run[r]["cp"] for r in cars], axis=0),
                    fold_index=0,
                    train_runs=cars,
                    val_runs=held_pair,
                    epochs=round(epochs * 8 / n_cars),
                    seed=seed + 10 * fold_idx + group_idx,
                    device=device,
                )
                preds = {r: _predict_single_fold(model, arrays_by_run[r]["features"], device) for r in held_pair}
                per_car = _summarize_predictions(held_pair, preds, arrays_by_run, meta_by_run)["per_car"]
                scores[n_cars].extend(per_car.values())
    return {
        str(n_cars): {
            "r2": round(float(np.mean([s["r2"] for s in rows])), 4),
            "worst_car_r2": round(float(np.min([s["r2"] for s in rows])), 4),
            "cd_p_mae": round(float(np.mean([s["cd_p_abs_err"] for s in rows])), 4),
            "networks": len(rows) // 2,
        }
        for n_cars, rows in scores.items()
    }


def train_and_evaluate(
    processed_dir: Path,
    artifacts_dir: Path,
    *,
    epochs: int,
    seed: int,
    run_ablations: bool,
    device: torch.device,
) -> dict[str, Any]:
    """Train the shipped ensemble on `processed_dir`, fit its spread, evaluate it in cross-validation and on
    the holdout cars, run the ablations and the learning curve when `run_ablations`, and write
    `ensemble.pt` and `metrics.json` to `artifacts_dir`."""
    arrays_by_run, meta_by_run = _load_processed_dataset(processed_dir)

    start = time.perf_counter()
    folds, out_of_fold = _train_folds(arrays_by_run, BLOCKS, epochs=epochs, seed=seed, device=device)
    train_sec = time.perf_counter() - start
    cv_summary = _summarize_predictions(ALLOWED_TRAIN_RUNS, out_of_fold, arrays_by_run, meta_by_run)
    regions = regional_r2_breakdown(
        np.concatenate([arrays_by_run[r]["cp"] for r in ALLOWED_TRAIN_RUNS]),
        np.concatenate([out_of_fold[r] for r in ALLOWED_TRAIN_RUNS]),
        np.concatenate([arrays_by_run[r]["centers"] for r in ALLOWED_TRAIN_RUNS]),
        np.concatenate([arrays_by_run[r]["normals"] for r in ALLOWED_TRAIN_RUNS]),
    )

    spread_scale = _fit_spread_scale(arrays_by_run, epochs=epochs, seed=seed, device=device)
    envelope = _build_training_envelope(arrays_by_run, meta_by_run)
    ensemble = _ensemble(folds, envelope, device, spread_scale)
    envelope["cv_mean_cp_std"] = _training_mean_spread(ensemble, arrays_by_run)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = artifacts_dir / "ensemble.pt"
    ensemble.save(checkpoint)
    shipped_label = f"Shipped: five networks, all inputs ({FEATURE_COLUMNS.total_dim})"
    _print_scores(shipped_label, cv_summary)
    print(f"Spread factor {spread_scale:.2f}; wrote {checkpoint} ({train_sec:.0f} s for the five networks)")

    holdout_summary, calibration_bins, coverage = _evaluate_holdout(ensemble, arrays_by_run, meta_by_run)
    print(f"Holdout: R² {holdout_summary['mean_r2']:.4f}, Cd,p MAE {holdout_summary['cd_p_mae']:.4f}")

    shipped = {
        "label": shipped_label,
        "blocks": list(BLOCKS),
        "learner": "ensemble",
        "cv": cv_summary,
        "holdout": holdout_summary,
    }
    ablations: dict[str, dict[str, Any]] = {"shipped_mlp_ensemble": shipped}
    if run_ablations:
        ablations = {
            "constant_baseline": _constant_baseline(arrays_by_run, meta_by_run),
            **_input_staircase(arrays_by_run, meta_by_run, envelope, epochs=epochs, seed=seed, device=device),
            "shipped_mlp_ensemble": shipped,
            **_other_learners(processed_dir, arrays_by_run, meta_by_run, seed=seed, device=device),
        }

    report: dict[str, Any] = {
        "device": str(device),
        "allowed_train_runs": list(ALLOWED_TRAIN_RUNS),
        "samples_per_car": int(meta_by_run[ALLOWED_TRAIN_RUNS[0]]["n_samples"]),
        "holdout_runs": list(HOLDOUT_RUNS),
        "epochs": epochs,
        "feature_dim": FEATURE_COLUMNS.total_dim,
        "train_seconds": round(train_sec, 2),
        "shipped_cv": cv_summary,
        "shipped_holdout": holdout_summary,
        "regional_breakdown_cv": regions,
        "spread_scale": round(spread_scale, 3),
        "uncertainty_calibration_holdout": calibration_bins,
        "uncertainty_coverage_holdout": coverage,
        "ablations": ablations,
    }
    if run_ablations:
        report["learning_curve"] = _learning_curve(arrays_by_run, meta_by_run, epochs=epochs, seed=seed, device=device)
        curve = ", ".join(f"{n} cars {c['r2']:.4f}" for n, c in report["learning_curve"].items())
        print(f"Learning curve, R² on cars not trained on: {curve}")
    metrics_path = artifacts_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {metrics_path}")
    return report
