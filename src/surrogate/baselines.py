"""LightGBM on every input, the boosted-tree comparison for the networks. It runs in a separate Python process
so that LightGBM's OpenMP runtime and PyTorch's are never loaded into one process."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

BOOST_ROUNDS = 400

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
        num_boost_round=cfg["rounds"],
    )
    for r in te_runs:
        feats = np.load(pdir / f"run_{r}.npz")["features"]
        saved[f"job{idx}_run{r}"] = np.asarray(booster.predict(feats), dtype=np.float32)
    print("MODEL_DONE", flush=True)

np.savez(out_path, **saved)
"""


def lightgbm_predictions(
    processed_dir: Path,
    train_runs: tuple[int, ...],
    car_folds: tuple[tuple[int, int], ...],
    holdout_runs: tuple[int, ...],
    on_model: Callable[[], object],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Each training car's prediction by the model of the fold that held it out, and the holdout cars'
    prediction by a model trained on every car of `train_runs`. `on_model` is called as each model is done."""
    jobs: list[tuple[list[int], list[int]]] = [
        ([r for r in train_runs if r not in val_pair], list(val_pair)) for val_pair in car_folds
    ]
    jobs.append((list(train_runs), list(holdout_runs)))

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "lgb_preds.npz"
        payload = json.dumps(
            {"processed_dir": str(processed_dir), "out_path": str(out_path), "jobs": jobs, "rounds": BOOST_ROUNDS}
        )
        command = [sys.executable, "-c", _WORKER_CODE, payload]
        with subprocess.Popen(command, stdout=subprocess.PIPE, text=True) as worker:
            assert worker.stdout is not None
            for line in worker.stdout:
                if line.strip() == "MODEL_DONE":
                    on_model()
                else:
                    sys.stderr.write(line)
        if worker.returncode != 0:
            raise subprocess.CalledProcessError(worker.returncode, command)
        with np.load(out_path) as data:
            raw_preds = {k: np.asarray(data[k], dtype=np.float32) for k in data.files}

    cv_preds = {r: raw_preds[f"job{idx}_run{r}"] for idx, val_pair in enumerate(car_folds) for r in val_pair}
    holdout_preds = {r: raw_preds[f"job{len(car_folds)}_run{r}"] for r in holdout_runs}
    return cv_preds, holdout_preds
