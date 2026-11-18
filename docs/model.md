# Model: inputs, training and further results

The model and its 107 inputs are summarised in the [README](../README.md#model). `src/surrogate/features.py` defines the inputs, and `web/engine/features.js` and `web/engine/lookups.js` reproduce them in the browser.

## How the inputs are built

Each input reads a lookup built from the whole surface of the file:

- The surface is cut where its polygons are large into pieces no more than 1 cm across, so a coarse mesh fills the lookups as a fine one does.
- The nearby surface is the area and the area times the normal of the surface, spread onto a grid whose step is a third of the size (1 cm for the 3 cm size), blurred, and read back at the cell, so it does not depend on how finely the file is meshed.
- The slices are read from a 2 cm raster of the car.
- Every lookup is placed in the car's own frame, from its nose, centre plane and lowest point, so moving the whole car changes no input.
- Every read interpolates smoothly between grid nodes and raster bins, so the inputs do not step as a cell moves across them, which would draw contour rings on the map of a change.
- The car's two sides are kept apart in every lookup, so a part that exists on one side only is read on that side only.

## Training

`uv run surrogate preprocess` draws 120,000 cells from each car in proportion to area and computes their inputs; `uv run surrogate train` trains each network for 30 epochs. The five networks hold out these pairs of cars:

| Network | 1 | 2 | 3 | 4 | 5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Held-out runs | 80, 98 | 102, 208 | 213, 281 | 397, 425 | 445, 474 |

Cross-validation scores each car with the network that did not see it. The holdout cars (runs 1 and 500) are scored with the five networks' mean and are never used for training, feature scaling or choosing between designs.

The five networks train on overlapping cars and share blind spots, so their raw standard deviation ranks the error well but understates it. The factor that scales it into the spread is fitted on the training cars only, by repeating the fold structure one level down.

## Further results

- **Distances are ladders.** Nearby shape matters in detail and distant shape only in bulk, so the surface scales grow about threefold per step and the slices' distances double.
- **Differences under the seed-to-seed spread are not findings.** Retraining the shipped networks with seeds 42, 43 and 44 gives CV $R^2$ `0.9458`, `0.9461` and `0.9460`, and CV $C_{d,p}$ MAE `0.0196`, `0.0145` and `0.0178`.
- **Integrated lift is closer.** In cross-validation the lift error is `0.0397` against `0.0613` for the training cars' mean lift, and on the holdout cars the ensemble gives `0.064` for run 1 against CFD `0.063`, and `0.118` for run 500 against `0.092`.
- **Each network's drag is reported.** On the holdout cars the five networks' drag values span `0.051` (run 1) and `0.034` (run 500), about three times the drag error of their mean.
- **A variant's map carries some discretisation.** Moving the whole car changes no input, but a variant that moves the body against what holds the car's nose and ground changes the prediction with no change in the flow.
  - On run 1, moved 7 mm with every lookup held where the unmoved car put it, the prediction changes by under `0.02` in $C_p$ (99th percentile) for a ride-height change and under `0.035` for a shift along the car, as a stretch does (`test_a_variant_moved_against_the_lookups_changes_the_prediction_by_less_than_the_stated_bound`).
  - Most of the shift's share is the slices' 2 cm columns along the car.
  - Differences smaller than that on a variant's map are not findings.
