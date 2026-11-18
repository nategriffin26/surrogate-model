# Model

## Approach

The model predicts one cell at a time. Ten cars are ten examples for a model that reads a whole car, and 1.2 million for a model that reads one cell and a description of the shape around it. Each cell is described by 107 numbers computed from the polygon geometry only; solver fields are removed when the file is read. None of them solves for the flow: the network learns it from the ten cars.

## Inputs

| Input group | Inputs | What it tells the cell |
| :--- | :---: | :--- |
| Position and facing | 7 | Where the cell is on the car, in metres from the nose, the tail, the centre plane and the lowest point, and its outward normal. |
| Nearby surface | 48 | The surface around the cell blurred at 3, 10, 25 and 75 cm: its mean normal and how much of it there is, and the same for the part upstream of the cell and the part downstream. |
| Slices across the car | 52 | Thirteen slices across the car, at the cell and 5, 10, 20, 40, 80 and 160 cm ahead and behind. From each: its area, how far it reaches outboard of the cell at the cell's height, and how far it reaches above and below the cell at the cell's distance from the centre plane. |

`src/surrogate/features.py` defines the inputs, and `web/engine/features.js` and `web/engine/lookups.js` reproduce them in the browser. Each input reads a lookup built from the whole surface of the file:

- The surface is cut where its polygons are large into pieces no more than 1 cm across, so a coarse mesh fills the lookups as a fine one does.
- The nearby surface is the area and the area times the normal of the surface, spread onto a grid whose step is a third of the size (1 cm for the 3 cm size), blurred, and read back at the cell, so it does not depend on how finely the file is meshed.
- The slices are read from a 2 cm raster of the car.
- Every lookup is placed in the car's own frame, from its nose, centre plane and lowest point, so moving the whole car changes no input.
- Every read interpolates smoothly between grid nodes and raster bins, so the inputs do not step as a cell moves across them, which would draw contour rings on the map of a change.
- The car's two sides are kept apart in every lookup, so a part that exists on one side only is read on that side only.

## Network, ensemble and training

A network of three hidden layers (`107 -> 256 -> 256 -> 256 -> 1`, GELU) is trained on 120,000 cells per car, drawn in proportion to area, for 30 epochs. Five such networks are trained, each with two whole cars held out; no fold splits a car's cells. The prediction is their mean. Their standard deviation, multiplied by a factor fitted on the training cars (`2.13`), is reported as the spread.

| Network | 1 | 2 | 3 | 4 | 5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Held-out runs | 80, 98 | 102, 208 | 213, 281 | 397, 425 | 445, 474 |

Cross-validation scores each car with the network that did not see it. The holdout cars (runs 1 and 500) are scored with the five networks' mean and are never used for training, feature scaling or choosing between designs.

## Results

Evaluated on 120,000 area-weighted cells per car. Cross-validation holds out two whole cars per fold across the ten training cars; holdout is the five-network ensemble on runs 1 and 500. $C_{d,p}$ is the pressure drag coefficient integrated from the predicted field.

| Model and inputs | CV $R^2$ | Worst-car CV $R^2$ | CV MAE ($C_p$) | CV $C_{d,p}$ MAE | Holdout $R^2$ | Holdout $C_{d,p}$ MAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Constant (fold mean $C_p$ and $C_{d,p}$) | −0.0031 | −0.0085 | 0.1661 | **0.0173** | −0.0006 | 0.0332 |
| PointNet whole-car vector (64) + position and facing | 0.5634 | 0.5021 | 0.0976 | 0.0321 | 0.5798 | 0.0311 |
| Position and facing only (7) | 0.6486 | 0.2510 | 0.0820 | 0.0822 | 0.7375 | 0.1143 |
| + nearby surface (55) | 0.9364 | 0.8159 | 0.0381 | 0.0213 | **0.9579** | 0.0165 |
| LightGBM, 400 trees, all inputs (107) | 0.9273 | 0.9115 | 0.0435 | **0.0173** | 0.9473 | **0.0095** |
| **Shipped: five networks, all inputs (107)** | **0.9458** | **0.9293** | **0.0356** | 0.0196 | 0.9558 | 0.0128 |

- **A learned whole-car vector does worse than no shape description at all.** With eight training cars per fold, the PointNet vector scores `0.5634` on unseen cars against `0.6486` for position and facing alone.
- **The nearby surface carries the most.** Added to position and facing, its 48 inputs raise CV $R^2$ from `0.6486` to `0.9364` and cut the drag error from `0.0822` to `0.0213`. The slices add least on average (`0.9458`) and most to the worst car (`0.8159` to `0.9293`).
- **Distances are ladders.** Nearby shape matters in detail and distant shape only in bulk, so the surface scales grow about threefold per step and the slices' distances double.
- **Differences under the seed-to-seed spread are not findings.** Retraining the shipped networks with seeds 42, 43 and 44 gives CV $R^2$ `0.9458`, `0.9461` and `0.9460`, and CV $C_{d,p}$ MAE `0.0196`, `0.0145` and `0.0178`.
- **Integrated drag does not tell these cars apart.** On the holdout cars' sampled cells the ensemble gives `0.254` for run 1 against CFD `0.268`, and `0.268` for run 500 against `0.279`. In cross-validation the drag error (`0.0196`) is no smaller than the error of predicting every car at the training cars' mean drag (`0.0173`).
- **Integrated lift is closer.** In cross-validation the lift error is `0.0397` against `0.0613` for the training cars' mean lift, and on the holdout cars the ensemble gives `0.064` for run 1 against CFD `0.063`, and `0.118` for run 500 against `0.092`.
- **A variant's map carries some discretisation.** Moving the whole car changes no input, but a variant that moves the body against what holds the car's nose and ground changes the prediction with no change in the flow.
  - On run 1, moved 7 mm with every lookup held where the unmoved car put it, the prediction changes by under `0.02` in $C_p$ (99th percentile) for a ride-height change and under `0.035` for a shift along the car, as a stretch does (`test_a_variant_moved_against_the_lookups_changes_the_prediction_by_less_than_the_stated_bound`).
  - Most of the shift's share is the slices' 2 cm columns along the car.
  - Differences smaller than that on a variant's map are not findings.
- **The spread ranks the error.** The five networks train on overlapping cars and share blind spots, so their raw standard deviation ranks the error well but understates it; the factor is fitted on the training cars only, by repeating the fold structure one level down.
  - On the holdout cars `57%` of the surface is within one spread of CFD and `84%` within two, and across the spread's quintiles the realized RMSE rises `0.020`, `0.028`, `0.035`, `0.048`, `0.101`.
  - The five networks' drag values are reported individually; on the holdout cars they span `0.051` (run 1) and `0.034` (run 500), about three times the drag error of their mean.
- **The upper surfaces and the middle of the car are predicted best; the underbody, the sides and the outer part worst.** Cross-validation $R^2$ by region (`regional_breakdown_cv` in `artifacts/metrics.json`), where a surface faces a direction when its outward normal's component along it exceeds 0.7 (`regional_r2_breakdown` in `src/surrogate/metrics.py`) and the regions overlap:
  - by facing: up `0.9745`, forward `0.9468`, rearward `0.9262`, down `0.9219`, sideways `0.9203`;
  - by distance from the centre plane: within 0.3 m `0.9737`, beyond 0.6 m `0.9128`.

## Limitations

- **Ten cars.** Accuracy is still rising with the number of training cars. Single networks trained on 2, 4, 6 and 8 cars score `0.838`, `0.897`, `0.927` and `0.946` on cars they did not see, with drag errors of 39, 39, 23 and 19 counts, a count being 0.001 of a drag coefficient (`learning_curve` in `artifacts/metrics.json`). More CFD runs are the first improvement, and DrivAerML holds several hundred more.
- **Pressure only.** The model predicts mean $C_p$. Drag and lift are integrated from pressure alone and leave out wall shear stress, so they are not the car's total coefficients.
- **One flow condition.** The DrivAerML runs share one speed at zero yaw, and no input describes the flow. The model cannot answer another speed, yaw angle or ground condition.
- **One family of shapes.** The training cars are variants of one notchback body, in DrivAerML's axes and in metres. `predict` reports when a car's size, slice-area curve or spread falls outside the training cars'; a prediction there is unchecked.
- **Variants are not validated.** No CFD run exists for a transformed car, so a variant's map has the discretisation bound under [Results](#results) and no measured error.
- **Coarse meshes are little tested.** The bundled examples are DrivAerML's full files, and the tests use those files or a small synthetic car. Run 1 decimated to 150,000 triangles scores the same $R^2$ in Python as the full file, and the browser's quick prediction on that mesh scored about `0.04` lower; both were measured once and neither is covered by the tests.
- **One machine.** The tool was run in Chrome on an Apple M5 Pro. A full DrivAerML file peaks near 4.9 GB of memory there, and without WebGPU the result takes about 48 s against about 5 s with it. Safari, Firefox, Linux and Windows are untested.

With a few hundred cars two things become worth trying that lose or do not exist here: a learned encoder of the whole surface in place of the hand-built inputs, and a model of the wall shear stress, which would give total drag.
