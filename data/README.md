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

Each `run_N/` holds:

- `boundary_N.vtp`: surface mesh, ~8.2M cells and ~600 MB per file. All fields are cell data (no point data): `CpMeanTrim` (the target; run_80 spans −4.5 to 1.05), `pMeanTrim`, `pPrime2MeanTrim`, `wallShearStressMeanTrim`
- `force_mom_N.csv`, `force_mom_constref_N.csv`: integrated forces and moments for the run
- `geo_parameters_N.csv`: the run's morph parameters
- `geo_ref_N.csv`: reference area and length

Training set: runs 80, 98, 102, 208, 213, 281, 397, 425, 445, 474.

`raw/` is git-ignored. To fetch one file again:

```
curl -L -o data/raw/train/run_80/boundary_80.vtp \
  https://huggingface.co/datasets/neashton/drivaerml/resolve/main/run_80/boundary_80.vtp
```
