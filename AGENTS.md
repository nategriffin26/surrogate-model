# AGENTS.md

## Scope & Rules

DrivAerML surface-only aerodynamics surrogate (`CpMeanTrim`) and interactive 3D web application.

1. **Strict geometry-only input boundary**: Model inputs, normalization, normal orientation, and frontal-area estimation derive exclusively from polygon geometry in `boundary_i.vtp`. Solver fields (`CpMeanTrim`, `pMeanTrim`, `pPrime2MeanTrim`, `wallShearStressMeanTrim`) are never read by feature extraction, mesh decimation, or reference-area calculation. Missing `CpMeanTrim` is valid at inference (disables ground-truth comparison views only).
2. **Training set constraint**: Train exclusively on the 10 assigned DrivAerML cases (`80, 98, 102, 208, 213, 281, 397, 425, 445, 474`). Holdout files (`data/raw/holdout/`) are never used for training, feature scaling, or hyperparameter selection.
3. **Physics invariants**:
   - Preserve native `CellData` association (no `cell_data_to_point_data` smoothing of targets).
   - Orient face normals outward from signed divergence volume ($\frac{1}{3}\sum (\mathbf{c}_i \cdot \mathbf{n}_i)\,dA_i > 0$).
   - Weight cell sampling and evaluation metrics by polygon area $dA$.
   - Split cross-validation folds strictly by whole vehicle (5 folds, 8 train / 2 val), never by random cells.
4. **Repository hygiene**: Keep `local/`, `data/raw/`, and `data/processed/` out of git. Commit only clean, self-contained source, tests, model artifacts, and concise documentation.
