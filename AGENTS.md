# AGENTS.md

A surrogate for the surface pressure coefficient (`CpMeanTrim`) on DrivAerML cars and a browser tool for working with its predictions. The repository is a take-home submission: its reviewers read the code, the tests and the documents, and the author has to be able to explain every part. Read `README.md` first; it holds the commands, the repository layout, the model's inputs and the measured results.

## Rules

1. **Geometry-only inputs.** Model inputs, normalization, normal orientation and frontal-area estimation derive only from the polygon geometry in `boundary_i.vtp`. Solver fields (`CpMeanTrim`, `pMeanTrim`, `pPrime2MeanTrim`, `wallShearStressMeanTrim`) are never read by feature extraction, mesh simplification or reference-area calculation. A file without `CpMeanTrim` is valid; it only disables comparison with CFD.
2. **Training set.** Train only on the ten assigned cases (`80, 98, 102, 208, 213, 281, 397, 425, 445, 474`). The holdout files (`data/raw/holdout/`) are never used for training, feature scaling or choosing between designs; they are reported.
3. **Physics invariants.**
   - Keep the file's `CellData` association (no `cell_data_to_point_data` smoothing of targets).
   - Orient face normals outward by the signed divergence volume ($\frac{1}{3}\sum (\mathbf{c}_i \cdot \mathbf{n}_i)\,dA_i > 0$).
   - Weight cell sampling and every metric by polygon area $dA$.
   - Split cross-validation by whole car (5 folds, 8 train and 2 validation), never by cells.
4. **Python is the reference.** `src/surrogate/` defines the model and `web/engine/` reproduces it in the browser. `tests/test_web_engine.py` holds the two together on inputs, each network's $C_p$, the spread and the drag. A change to either side is finished when the other matches and those tests pass.
5. **The tool computes in the browser.** Plain ES modules with no build step and no CDN; libraries are vendored in `web/vendor/`. WebGPU with a CPU fallback. Nothing may depend on `SharedArrayBuffer`, so the tool can be hosted as static files without special headers.
6. **Repository hygiene.** `local/`, `data/raw/` and `data/processed/` stay out of git; `local/` is scratch. Commit only source, tests, model artifacts and documents that describe the code as it is.

## Contracts

- `web/engine/result.js` documents everything the engine returns. The interface reads the engine through it and nothing else.
- `web/model/ensemble.json`, written by `surrogate.export_web`, is everything the browser knows about the model: the input definition, the scales, the weights' hash and the numbers shown about the model. Nothing about the model is typed into `web/`.

## Changing the model

After a change to the inputs or a retraining, in this order: `surrogate.preprocess` (inputs changed), `surrogate.train`, the same change in `web/engine/features.js` (inputs changed), `surrogate.export_web`, `surrogate.write_examples`, `uv run pytest`, then the README's input table and results from `artifacts/metrics.json`. The tests fail while a bundled example was computed with other weights.

Choose between designs on cross-validation over the ten training cars. Two single runs that differ by less than `0.003` in CV $R^2$ or `0.004` in CV $C_{d,p}$ MAE are within seed-to-seed spread; compare on three seeds before concluding anything.

## Checks

- `uv run pytest`: the Python pipeline and the browser engine against it (needs Bun or Deno; full-car tests need the files in `data/raw/`).
- `deno test --allow-read tests/meshes.test.js tests/result.test.js`: the engine's meshes and file format.
- An interface change is checked in a real browser with WebGPU, on a bundled example and on a full DrivAerML file. A browser without WebGPU exercises only the CPU path.
- Analysing a full file takes about 4.5 GB in the browser; a training run or a full-surface job in Python is heavy too. Run at most three such jobs at once on one machine; where `local/bin/heavy` exists, start them through it.

## Documentation

- A change is finished when the documents that describe it are true: `README.md` (commands, layout, input table, results), this file, `data/README.md`, and the docstring at the top of each module touched.
- Describe what the code does now. Delete a statement that is no longer true; do not mark it as old, and do not record history, plans or the discussion behind a change. The git log is the history.
- Every number is copied from `artifacts/metrics.json`, `web/examples/*.json` or a script named beside it, in the same change that moves it.
- Keep them short. A sentence no reader would act on is removed, and a stale statement found while working elsewhere is fixed or deleted in that change.
