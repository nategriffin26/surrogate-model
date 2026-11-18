# Data

Subset of DrivAerML (Ashton et al., CC BY-SA 4.0; see `metadata/LICENSE.txt`), downloaded from
https://huggingface.co/datasets/neashton/drivaerml. Only surface data is kept, per the assignment.

```
metadata/                 dataset README, license, and per-run tables for all ~500 runs
  force_mom_all.csv         integrated Cd/Cl etc. (per-run frontal area reference)
  force_mom_constref_all.csv  same, constant reference area
  geo_parameters_all.csv    the 16 geometric morph parameters per run
raw/train/run_N/          the 10 cases the assignment allows for training
raw/holdout/run_N/        unseen cases (run_1, run_500) for testing inference and the web app
```

Each `run_N/` holds `boundary_N.vtp`, the surface mesh: 570 to 710 MB per file (run 1 has 8.8 million cells). All fields are cell data (no point data): `CpMeanTrim` (the target; run_80 spans −4.5 to 1.05), `pMeanTrim`, `pPrime2MeanTrim`, `wallShearStressMeanTrim`.

Training set: runs 80, 98, 102, 208, 213, 281, 397, 425, 445, 474.

`raw/` is git-ignored. `uv run surrogate fetch` downloads the twelve files (7.8 GB); `fetch training` or `fetch holdout` downloads one set. Files already here are skipped, and an interrupted download continues from where it stopped.
