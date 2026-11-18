# Data

The cars are from DrivAerML: N. Ashton, C. Mockett, M. Fuchs, L. Fliessbach, H. Hetmann, T. Knacke, N. Schonwald, V. Skaperdas, G. Fotiadis, A. Walle, B. Hupertz and D. Maddix, "DrivAerML: High-Fidelity Computational Fluid Dynamics Dataset for Road-Car External Aerodynamics", 2024, [arXiv:2408.11969](https://arxiv.org/abs/2408.11969). The dataset is at https://huggingface.co/datasets/neashton/drivaerml and is licensed CC BY-SA 4.0; the license text is `metadata/LICENSE.txt`. Only surface data is used, per the assignment.

```
metadata/LICENSE.txt      the dataset's license
raw/train/run_N/          the 10 cases the assignment allows for training
raw/holdout/run_N/        unseen cases (run_1, run_500) for testing inference and the web app
```

Each `run_N/` holds `boundary_N.vtp`, the surface mesh: 570 to 710 MB per file (run 1 has 8.8 million cells). All fields are cell data (no point data): `CpMeanTrim` (the target; run_80 spans −4.5 to 1.05), `pMeanTrim`, `pPrime2MeanTrim`, `wallShearStressMeanTrim`.

Training set: runs 80, 98, 102, 208, 213, 281, 397, 425, 445, 474.

`raw/` is git-ignored. `uv run surrogate fetch` downloads the twelve files (7.8 GB); `fetch training` or `fetch holdout` downloads one set. Files already here are skipped, and an interrupted download continues from where it stopped.
