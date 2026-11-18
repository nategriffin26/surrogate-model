"""The preprocessing pipeline for the ten DrivAerML training cars and the two holdout cars.

Reads each `boundary_i.vtp` once, verifies outward normal orientation and pressure
force integrals, draws an area-weighted cell sample (`n_samples` per vehicle),
extracts the deterministic surface feature matrix, and saves compact
`.npz` archives + `manifest.json` under `data/processed/`. Run by `uv run surrogate preprocess`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
import numpy as np

from surrogate.dataset import ALLOWED_TRAIN_RUNS, DEFAULT_HOLDOUT_RUNS, surface_path
from surrogate.features import FEATURE_COLUMNS, SurfaceContext, extract_query_features
from surrogate.surface import compute_pressure_forces, read_vtp_surface, sample_cells_by_area


def preprocess_case(
    vtp_path: Path,
    out_dir: Path,
    *,
    run_id: int,
    split: str,
    n_samples: int = 120_000,
) -> dict[str, Any]:
    """Preprocess one `boundary_<run>.vtp` into `out_dir/run_<run>.npz` and return summary."""
    t0 = time.perf_counter()
    surface = read_vtp_surface(vtp_path, include_target=True)
    if surface.cp is None:
        raise ValueError(f"Expected {vtp_path} to contain CpMeanTrim for {split} split")
    t_read = time.perf_counter() - t0

    context = SurfaceContext.from_surface(surface)
    idx = sample_cells_by_area(surface.areas, n_samples, seed=run_id)

    sampled_centers = surface.centers[idx]
    sampled_normals = surface.normals[idx]
    sampled_cp = surface.cp[idx]
    features = extract_query_features(context, sampled_centers, sampled_normals)

    exact_forces = compute_pressure_forces(
        surface.cp,
        surface.normals,
        surface.areas,
        total_area=surface.total_area,
        ref_area=surface.frontal_area,
    )
    sample_forces = compute_pressure_forces(
        sampled_cp,
        sampled_normals,
        None,
        total_area=surface.total_area,
        ref_area=surface.frontal_area,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"run_{run_id}.npz"
    np.savez_compressed(
        npz_path,
        centers=sampled_centers,
        normals=sampled_normals,
        cp=sampled_cp,
        features=features,
        slice_area=context.slice_area,
    )

    elapsed = time.perf_counter() - t0
    return {
        "run": run_id,
        "split": split,
        "n_cells": surface.n_cells,
        "n_samples": n_samples,
        "total_area": round(surface.total_area, 4),
        "frontal_area": round(surface.frontal_area, 4),
        "enclosed_volume": round(surface.enclosed_volume, 4),
        "normals_flipped": surface.normals_flipped,
        "bounds": [round(b, 4) for b in surface.bounds],
        "shape_descriptor": {k: round(v, 4) for k, v in context.shape_descriptor().items()},
        "exact_cd_p": round(exact_forces["cd_p"], 5),
        "sample_cd_p": round(sample_forces["cd_p"], 5),
        "exact_cl_p": round(exact_forces["cl_p"], 5),
        "cp_min": round(float(surface.cp.min()), 4),
        "cp_max": round(float(surface.cp.max()), 4),
        "read_sec": round(t_read, 2),
        "total_sec": round(elapsed, 2),
    }


def preprocess_cases(raw_dir: Path, out_dir: Path, *, samples: int = 120_000) -> dict[str, Any]:
    """Preprocess the ten training cars and the two holdout cars under `raw_dir` into `out_dir`, write
    its `manifest.json` and return the manifest. The holdout cars are only reported, never trained on."""
    cases: list[tuple[str, int, Path]] = []
    for split, runs in (("train", ALLOWED_TRAIN_RUNS), ("holdout", DEFAULT_HOLDOUT_RUNS)):
        for run_id in runs:
            vtp = surface_path(raw_dir, run_id)
            if not vtp.is_file():
                raise FileNotFoundError(f"Missing {vtp}. Fetch it with `uv run surrogate fetch`.")
            cases.append((split, run_id, vtp))

    manifest_cases: list[dict[str, Any]] = []
    for split, run_id, vtp in cases:
        summary = preprocess_case(
            vtp,
            out_dir,
            run_id=run_id,
            split=split,
            n_samples=samples,
        )
        manifest_cases.append(summary)
        print(
            f"[{split:7s}] run_{run_id:3d} | cells={summary['n_cells']:8d} | "
            f"A_ref={summary['frontal_area']:.3f} m² | "
            f"Cd,p={summary['exact_cd_p']:.4f} (MC {summary['sample_cd_p']:.4f}) | "
            f"{summary['total_sec']:.1f}s",
            flush=True,
        )

    manifest = {
        "allowed_train_runs": list(ALLOWED_TRAIN_RUNS),
        "samples_per_case": samples,
        "feature_dim": FEATURE_COLUMNS.total_dim,
        "cases": manifest_cases,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {manifest_path} ({len(manifest_cases)} cases)")
    return manifest
