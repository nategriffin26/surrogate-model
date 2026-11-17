"""Torch-free tabular gradient-boosted tree baseline (runs in an isolated subprocess on macOS)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from surrogate.metrics import case_evaluation_summary

_WORKER_CODE = """\
import json, sys
from pathlib import Path
import lightgbm as lgb
import numpy as np

cfg = json.loads(sys.argv[1])
pdir = Path(cfg["processed_dir"])
out_path = Path(cfg["out_path"])
jobs = cfg["jobs"]

saved = {}
for idx, (tr_runs, te_runs) in enumerate(jobs):
    x_tr = np.concatenate([np.load(pdir / f"run_{r}.npz")["features"] for r in tr_runs], axis=0)
    y_tr = np.concatenate([np.load(pdir / f"run_{r}.npz")["cp"] for r in tr_runs], axis=0)
    dtrain = lgb.Dataset(x_tr, label=y_tr)
    booster = lgb.train(
        {
            "objective": "regression",
            "learning_rate": 0.1,
            "num_leaves": 127,
            "min_data_in_leaf": 100,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "feature_fraction": 0.8,
            "seed": 42,
            "num_threads": 4,
            "verbose": -1,
        },
        dtrain,
        num_boost_round=400,
    )
    for r in te_runs:
        feats = np.load(pdir / f"run_{r}.npz")["features"]
        saved[f"job{idx}_run{r}"] = np.asarray(booster.predict(feats), dtype=np.float32)

np.savez(out_path, **saved)
"""


def run_lightgbm_baseline_isolated(
    processed_dir: Path,
    train_runs: tuple[int, ...],
    car_folds: tuple[tuple[int, int], ...],
    holdout_runs: tuple[int, ...],
    arrays_by_run: dict[int, dict[str, np.ndarray]],
    meta_by_run: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Run LightGBM baseline in a torch-free Python subprocess so OpenMP runtimes never collide."""
    jobs: list[tuple[list[int], list[int]]] = [
        ([r for r in train_runs if r not in val_pair], list(val_pair))
        for val_pair in car_folds
    ]
    if holdout_runs:
        jobs.append((list(train_runs), list(holdout_runs)))

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "lgb_preds.npz"
        payload = json.dumps(
            {
                "processed_dir": str(processed_dir),
                "out_path": str(out_path),
                "jobs": jobs,
            }
        )
        subprocess.run([sys.executable, "-c", _WORKER_CODE, payload], check=True)
        with np.load(out_path) as data:
            raw_preds = {k: np.asarray(data[k], dtype=np.float32) for k in data.files}

    cv_preds: dict[int, np.ndarray] = {}
    for idx, val_pair in enumerate(car_folds):
        for r in val_pair:
            cv_preds[r] = raw_preds[f"job{idx}_run{r}"]

    def _summarize(runs: tuple[int, ...], preds_map: dict[int, np.ndarray]) -> dict[str, Any]:
        per_car: dict[str, dict[str, float]] = {}
        for r in runs:
            meta = meta_by_run[r]
            arr = arrays_by_run[r]
            s = case_evaluation_summary(
                arr["cp"],
                preds_map[r],
                arr["normals"],
                total_area=float(meta["total_area"]),
                frontal_area=float(meta["frontal_area"]),
                exact_cd_p=float(meta["exact_cd_p"]),
            )
            per_car[str(r)] = {k: round(v, 5) for k, v in s.items()}

        r2_vals = [per_car[str(r)]["r2"] for r in runs]
        rel_l2_vals = [per_car[str(r)]["rel_l2"] for r in runs]
        mae_vals = [per_car[str(r)]["mae"] for r in runs]
        cd_abs = [per_car[str(r)]["cd_p_abs_err"] for r in runs]
        cd_err = [per_car[str(r)]["cd_p_err"] for r in runs]
        return {
            "mean_r2": round(float(np.mean(r2_vals)), 4),
            "worst_car_r2": round(float(np.min(r2_vals)), 4),
            "best_car_r2": round(float(np.max(r2_vals)), 4),
            "mean_rel_l2": round(float(np.mean(rel_l2_vals)), 4),
            "mean_mae": round(float(np.mean(mae_vals)), 4),
            "cd_p_mae": round(float(np.mean(cd_abs)), 4),
            "cd_p_bias": round(float(np.mean(cd_err)), 4),
            "per_car": per_car,
        }

    out: dict[str, Any] = {"cv": _summarize(train_runs, cv_preds)}
    if holdout_runs:
        ho_idx = len(car_folds)
        ho_preds = {r: raw_preds[f"job{ho_idx}_run{r}"] for r in holdout_runs}
        out["holdout"] = _summarize(holdout_runs, ho_preds)
    return out
