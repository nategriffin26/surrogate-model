"""Five-fold training on the ten DrivAerML cases, and the ablations recorded beside it.

The ablations are a staircase that adds one block of inputs at a time (`features.BLOCKS`, starting
from a constant), closed by the shipped ensemble on every block, and two other learners: a learned
whole-car vector, and boosted trees on every input.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from surrogate.baselines import run_lightgbm_baseline_isolated
import numpy as np
import torch
import torch.nn.functional as F

from surrogate.dataset import ALLOWED_TRAIN_RUNS, DEFAULT_HOLDOUT_RUNS
from surrogate.features import BLOCKS, FEATURE_COLUMNS
from surrogate.metrics import (
    case_evaluation_summary,
    regional_r2_breakdown,
    spread_coverage,
    uncertainty_calibration_bins,
)
from surrogate.model import (
    Z_CLIP,
    FoldModel,
    PerCellMLP,
    PointNetGlobalMLP,
    SurrogateEnsemble,
)

CAR_FOLDS: tuple[tuple[int, int], ...] = (
    (80, 98),
    (102, 208),
    (213, 281),
    (397, 425),
    (445, 474),
)


def _load_processed_dataset(
    processed_dir: Path,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, dict[str, Any]]]:
    manifest_path = processed_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run `uv run surrogate preprocess` first."
        )
    manifest = json.loads(manifest_path.read_text())
    meta_by_run: dict[int, dict[str, Any]] = {int(c["run"]): c for c in manifest["cases"]}
    arrays_by_run: dict[int, dict[str, np.ndarray]] = {}
    for run_id in meta_by_run:
        npz_path = processed_dir / f"run_{run_id}.npz"
        with np.load(npz_path) as data:
            arrays_by_run[run_id] = {k: data[k] for k in data.files}
    return arrays_by_run, meta_by_run


def _select_columns(features: np.ndarray, blocks: tuple[str, ...]) -> np.ndarray:
    return np.concatenate([features[:, getattr(FEATURE_COLUMNS, b)] for b in blocks], axis=1)


def _fit_mlp_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    fold_index: int,
    train_runs: tuple[int, ...],
    val_runs: tuple[int, ...],
    hidden_dim: int = 256,
    depth: int = 3,
    epochs: int = 30,
    batch_size: int = 8192,
    lr: float = 5e-3,
    seed: int = 42,
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

    model = PerCellMLP(in_dim=x_train.shape[1], hidden_dim=hidden_dim, depth=depth).to(device)
    n_batches = max(1, len(xt) // batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs * n_batches
    )

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(xt), device=device)
        for step in range(n_batches):
            batch_idx = perm[step * batch_size : (step + 1) * batch_size]
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
            exact_cd_p=float(meta["exact_cd_p"]),
        )
        per_car[str(r)] = {k: round(v, 5) for k, v in summary.items()}

    r2_vals = [per_car[str(r)]["r2"] for r in runs]
    rel_l2_vals = [per_car[str(r)]["rel_l2"] for r in runs]
    mae_vals = [per_car[str(r)]["mae"] for r in runs]
    cd_abs_errs = [per_car[str(r)]["cd_p_abs_err"] for r in runs]
    cd_signed_errs = [per_car[str(r)]["cd_p_err"] for r in runs]

    return {
        "mean_r2": round(float(np.mean(r2_vals)), 4),
        "worst_car_r2": round(float(np.min(r2_vals)), 4),
        "best_car_r2": round(float(np.max(r2_vals)), 4),
        "mean_rel_l2": round(float(np.mean(rel_l2_vals)), 4),
        "mean_mae": round(float(np.mean(mae_vals)), 4),
        "cd_p_mae": round(float(np.mean(cd_abs_errs)), 4),
        "cd_p_bias": round(float(np.mean(cd_signed_errs)), 4),
        "per_car": per_car,
    }


def _run_pointnet_global_ablation(
    arrays_by_run: dict[int, dict[str, np.ndarray]],
    meta_by_run: dict[int, dict[str, Any]],
    holdout_runs: tuple[int, ...],
    *,
    device: torch.device,
    steps: int = 600,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate PointNet global-vector conditioning (`g_dim=64`) on identical car folds."""

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
            r: torch.as_tensor((arrays_by_run[r]["cp"] - ym) / ys, dtype=torch.float32, device=device)
            for r in tr_runs
        }

        net = PointNetGlobalMLP(in_dim=x_cat.shape[1], global_dim=64).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)

        net.train()
        for _ in range(steps):
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
        tr = tuple(r for r in ALLOWED_TRAIN_RUNS if r not in val_pair)
        cv_preds.update(_fit_and_eval(tr, val_pair))

    out: dict[str, Any] = {
        "cv": _summarize_predictions(ALLOWED_TRAIN_RUNS, cv_preds, arrays_by_run, meta_by_run)
    }
    if holdout_runs:
        ho_preds = _fit_and_eval(ALLOWED_TRAIN_RUNS, holdout_runs)
        out["holdout"] = _summarize_predictions(holdout_runs, ho_preds, arrays_by_run, meta_by_run)
    return out



def _build_training_envelope(
    arrays_by_run: dict[int, dict[str, np.ndarray]],
    meta_by_run: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Record min/max geometric descriptors strictly across the 10 allowed training runs."""
    desc_keys = list(meta_by_run[ALLOWED_TRAIN_RUNS[0]]["shape_descriptor"].keys())
    descriptor_bounds: dict[str, dict[str, float]] = {}
    for key in desc_keys:
        vals = [float(meta_by_run[r]["shape_descriptor"][key]) for r in ALLOWED_TRAIN_RUNS]
        descriptor_bounds[key] = {
            "min": round(float(min(vals)), 4),
            "max": round(float(max(vals)), 4),
            "mean": round(float(np.mean(vals)), 4),
        }

    section_curves = np.stack([arrays_by_run[r]["section_area_x"] for r in ALLOWED_TRAIN_RUNS], axis=0)
    return {
        "train_runs": list(ALLOWED_TRAIN_RUNS),
        "descriptor_bounds": descriptor_bounds,
        "section_area_min": np.round(section_curves.min(axis=0), 4).tolist(),
        "section_area_max": np.round(section_curves.max(axis=0), 4).tolist(),
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
            tr_runs = tuple(r for r in ALLOWED_TRAIN_RUNS if r not in held_pair + inner_pair)
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


def train_and_evaluate(
    processed_dir: Path = Path("data/processed"),
    artifacts_dir: Path = Path("artifacts"),
    *,
    epochs: int = 30,
    seed: int = 42,
    run_ablations: bool = True,
    device: torch.device,
) -> dict[str, Any]:
    arrays_by_run, meta_by_run = _load_processed_dataset(processed_dir)
    holdout_runs = tuple(r for r in DEFAULT_HOLDOUT_RUNS if r in arrays_by_run)

    print(f"Device: {device} | Train runs: {len(ALLOWED_TRAIN_RUNS)} | Holdout runs: {holdout_runs}")

    # 1. Train the shipped 5-fold Per-Cell MLP Ensemble on every feature column
    t0 = time.perf_counter()
    shipped_folds: list[FoldModel] = []
    oof_preds: dict[int, np.ndarray] = {}

    for fold_idx, val_pair in enumerate(CAR_FOLDS):
        tr_runs = tuple(r for r in ALLOWED_TRAIN_RUNS if r not in val_pair)
        x_tr = np.concatenate([arrays_by_run[r]["features"] for r in tr_runs], axis=0)
        y_tr = np.concatenate([arrays_by_run[r]["cp"] for r in tr_runs], axis=0)

        fold_model = _fit_mlp_fold(
            x_tr,
            y_tr,
            fold_index=fold_idx,
            train_runs=tr_runs,
            val_runs=val_pair,
            epochs=epochs,
            seed=seed,
            device=device,
        )
        shipped_folds.append(fold_model)
        for val_run in val_pair:
            oof_preds[val_run] = _predict_single_fold(
                fold_model, arrays_by_run[val_run]["features"], device
            )

    train_sec = time.perf_counter() - t0
    cv_summary = _summarize_predictions(ALLOWED_TRAIN_RUNS, oof_preds, arrays_by_run, meta_by_run)

    # Regional breakdown across all 10 out-of-fold vehicles
    oof_true_cat = np.concatenate([arrays_by_run[r]["cp"] for r in ALLOWED_TRAIN_RUNS])
    oof_pred_cat = np.concatenate([oof_preds[r] for r in ALLOWED_TRAIN_RUNS])
    oof_centers_cat = np.concatenate([arrays_by_run[r]["centers"] for r in ALLOWED_TRAIN_RUNS])
    oof_normals_cat = np.concatenate([arrays_by_run[r]["normals"] for r in ALLOWED_TRAIN_RUNS])
    regions = regional_r2_breakdown(oof_true_cat, oof_pred_cat, oof_centers_cat, oof_normals_cat)

    spread_scale = _fit_spread_scale(arrays_by_run, epochs=epochs, seed=seed, device=device)

    # Calibrate reference ensemble spread strictly on the 10 training cars (inflated by sqrt(5/4)
    # because 4 of 5 folds saw each training car).
    envelope = _build_training_envelope(arrays_by_run, meta_by_run)
    ensemble = SurrogateEnsemble(
        shipped_folds,
        in_dim=FEATURE_COLUMNS.total_dim,
        hidden_dim=256,
        depth=3,
        training_envelope=envelope,
        spread_scale=spread_scale,
        device=device,
    )
    in_domain_stds = [
        float(ensemble.predict_features(arrays_by_run[r]["features"]).cp_std.mean() * np.sqrt(5.0 / 4.0))
        for r in ALLOWED_TRAIN_RUNS
    ]
    envelope["cv_mean_cp_std"] = round(float(np.mean(in_domain_stds)), 4)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = artifacts_dir / "ensemble.pt"
    ensemble.save(ckpt_path)
    print(
        f"[Shipped 5-Fold MLP] CV R²={cv_summary['mean_r2']:.4f} "
        f"(worst {cv_summary['worst_car_r2']:.4f}) | "
        f"Cd,p MAE={cv_summary['cd_p_mae']:.4f} | spread scale {spread_scale:.2f} | "
        f"saved {ckpt_path} ({train_sec:.1f}s)"
    )

    # Evaluate the saved 5-fold ensemble directly on untouched holdout runs (run_1, run_500)
    holdout_summary: dict[str, Any] = {}
    calibration_bins: list[dict[str, float]] = []
    coverage: dict[str, float] = {}
    if holdout_runs:
        ho_preds: dict[int, np.ndarray] = {}
        ho_stds: dict[int, np.ndarray] = {}
        ho_forces: dict[str, dict[str, Any]] = {}
        for r in holdout_runs:
            ens_pred = ensemble.predict_features(arrays_by_run[r]["features"])
            ho_preds[r] = ens_pred.cp_mean
            ho_stds[r] = ens_pred.cp_std
            forces = ensemble.integrated_forces(
                ens_pred,
                arrays_by_run[r]["normals"],
                None,
                total_area=float(meta_by_run[r]["total_area"]),
                frontal_area=float(meta_by_run[r]["frontal_area"]),
            )
            ho_forces[str(r)] = {
                k: [round(x, 5) for x in v] if isinstance(v, list) else round(v, 5) for k, v in forces.items()
            }

        holdout_summary = _summarize_predictions(holdout_runs, ho_preds, arrays_by_run, meta_by_run)
        holdout_summary["ensemble_forces"] = ho_forces
        ho_true = np.concatenate([arrays_by_run[r]["cp"] for r in holdout_runs])
        ho_pred = np.concatenate([ho_preds[r] for r in holdout_runs])
        ho_std = np.concatenate([ho_stds[r] for r in holdout_runs])
        calibration_bins = uncertainty_calibration_bins(ho_true, ho_pred, ho_std)
        coverage = spread_coverage(ho_true, ho_pred, ho_std)
        print(
            f"[5-Fold Ensemble Holdout] R²={holdout_summary['mean_r2']:.4f} | "
            f"Cd,p MAE={holdout_summary['cd_p_mae']:.4f} (bias {holdout_summary['cd_p_bias']:+.4f})"
        )

    # 2. Ablations
    ablations: dict[str, Any] = {}
    if run_ablations:
        # 2a. Constant fold-mean baseline (zero learning)
        const_cv_preds: dict[int, np.ndarray] = {}
        const_cd_cv_errs: list[float] = []
        true_cd_sample = {
            r: case_evaluation_summary(
                arrays_by_run[r]["cp"],
                arrays_by_run[r]["cp"],
                arrays_by_run[r]["normals"],
                total_area=float(meta_by_run[r]["total_area"]),
                frontal_area=float(meta_by_run[r]["frontal_area"]),
            )["cd_p_true_sample"]
            for r in ALLOWED_TRAIN_RUNS + holdout_runs
        }
        for val_pair in CAR_FOLDS:
            tr_runs = tuple(r for r in ALLOWED_TRAIN_RUNS if r not in val_pair)
            mean_cp = float(np.mean([arrays_by_run[r]["cp"].mean() for r in tr_runs]))
            mean_cd = float(np.mean([true_cd_sample[r] for r in tr_runs]))
            for r in val_pair:
                const_cv_preds[r] = np.full_like(arrays_by_run[r]["cp"], mean_cp)
                const_cd_cv_errs.append(mean_cd - true_cd_sample[r])

        const_entry: dict[str, Any] = {
            "label": "Constant fold-mean baseline (no model)",
            "blocks": [],
            "learner": "constant",
            "cv": {
                "mean_r2": 0.0,
                "worst_car_r2": 0.0,
                "mean_mae": round(
                    float(
                        np.mean(
                            [np.abs(const_cv_preds[r] - arrays_by_run[r]["cp"]).mean() for r in ALLOWED_TRAIN_RUNS]
                        )
                    ),
                    4,
                ),
                "cd_p_mae": round(float(np.mean(np.abs(const_cd_cv_errs))), 4),
                "cd_p_bias": round(float(np.mean(const_cd_cv_errs)), 4),
            },
        }
        if holdout_runs:
            all_mean_cp = float(np.mean([arrays_by_run[r]["cp"].mean() for r in ALLOWED_TRAIN_RUNS]))
            all_mean_cd = float(np.mean([true_cd_sample[r] for r in ALLOWED_TRAIN_RUNS]))
            ho_cd_errs = [all_mean_cd - true_cd_sample[r] for r in holdout_runs]
            const_entry["holdout"] = {
                "mean_r2": 0.0,
                "mean_mae": round(
                    float(np.mean([np.abs(all_mean_cp - arrays_by_run[r]["cp"]).mean() for r in holdout_runs])),
                    4,
                ),
                "cd_p_mae": round(float(np.mean(np.abs(ho_cd_errs))), 4),
                "cd_p_bias": round(float(np.mean(ho_cd_errs)), 4),
            }
        ablations["constant_baseline"] = const_entry

        # 2b. The staircase: one network per fold on the first one, two and three blocks of inputs; the
        # shipped ensemble on all of them closes it
        for n_blocks in range(1, len(BLOCKS)):
            blocks = BLOCKS[:n_blocks]
            key = f"{'_'.join(blocks)}_mlp"
            label = f"Five networks on {' + '.join(blocks)}"
            sub_folds: list[FoldModel] = []
            sub_oof: dict[int, np.ndarray] = {}
            for fold_idx, val_pair in enumerate(CAR_FOLDS):
                tr_runs = tuple(r for r in ALLOWED_TRAIN_RUNS if r not in val_pair)
                x_tr = np.concatenate(
                    [_select_columns(arrays_by_run[r]["features"], blocks) for r in tr_runs], axis=0
                )
                y_tr = np.concatenate([arrays_by_run[r]["cp"] for r in tr_runs], axis=0)
                fm = _fit_mlp_fold(
                    x_tr,
                    y_tr,
                    fold_index=fold_idx,
                    train_runs=tr_runs,
                    val_runs=val_pair,
                    epochs=epochs,
                    seed=seed,
                    device=device,
                )
                sub_folds.append(fm)
                for val_run in val_pair:
                    x_val = _select_columns(arrays_by_run[val_run]["features"], blocks)
                    sub_oof[val_run] = _predict_single_fold(fm, x_val, device)

            sub_cv = _summarize_predictions(ALLOWED_TRAIN_RUNS, sub_oof, arrays_by_run, meta_by_run)
            entry: dict[str, Any] = {"label": label, "blocks": list(blocks), "learner": "mlp", "cv": sub_cv}
            if holdout_runs:
                sub_ens = SurrogateEnsemble(
                    sub_folds,
                    in_dim=sub_folds[0].x_mean.shape[0],
                    hidden_dim=256,
                    depth=3,
                    training_envelope=envelope,
                    device=device,
                )
                sub_ho_preds = {
                    r: sub_ens.predict_features(_select_columns(arrays_by_run[r]["features"], blocks)).cp_mean
                    for r in holdout_runs
                }
                entry["holdout"] = _summarize_predictions(holdout_runs, sub_ho_preds, arrays_by_run, meta_by_run)
            ablations[key] = entry
            print(f"[Ablation: {label}] CV R²={sub_cv['mean_r2']:.4f} | Cd,p MAE={sub_cv['cd_p_mae']:.4f}")

    ablations["shipped_mlp_ensemble"] = {
        "label": f"Shipped: five networks on all {FEATURE_COLUMNS.total_dim} inputs",
        "blocks": list(BLOCKS),
        "learner": "ensemble",
        "cv": cv_summary,
        "holdout": holdout_summary,
    }

    if run_ablations:
        # 2c. Two other learners: a learned whole-car vector beside position and facing, and boosted
        # trees on every input
        pnet = _run_pointnet_global_ablation(arrays_by_run, meta_by_run, holdout_runs, device=device, seed=seed)
        ablations["pointnet_global_64d"] = {
            "label": "PointNet learned global vector g (64D)",
            "blocks": ["local"],
            "learner": "pointnet",
            **pnet,
        }
        print(f"[Ablation: PointNet 64D g] CV R²={pnet['cv']['mean_r2']:.4f} | Cd,p MAE={pnet['cv']['cd_p_mae']:.4f}")

        lgb_res = run_lightgbm_baseline_isolated(
            processed_dir,
            ALLOWED_TRAIN_RUNS,
            CAR_FOLDS,
            holdout_runs,
            arrays_by_run,
            meta_by_run,
        )
        ablations["lightgbm_full"] = {
            "label": f"LightGBM boosted trees (all {FEATURE_COLUMNS.total_dim} inputs)",
            "blocks": list(BLOCKS),
            "learner": "trees",
            **lgb_res,
        }
        print(f"[Baseline: LightGBM] CV R²={lgb_res['cv']['mean_r2']:.4f} | Cd,p MAE={lgb_res['cv']['cd_p_mae']:.4f}")

    report = {
        "device": str(device),
        "allowed_train_runs": list(ALLOWED_TRAIN_RUNS),
        "samples_per_car": int(meta_by_run[ALLOWED_TRAIN_RUNS[0]]["n_samples"]),
        "holdout_runs": list(holdout_runs),
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
    metrics_path = artifacts_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {metrics_path}")
    return report
