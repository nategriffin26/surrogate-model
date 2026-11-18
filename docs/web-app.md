# Web app

`uv run surrogate app` serves Surrogate Model at `http://127.0.0.1:8740/` (setup is in the [README](../README.md#run-it)). The tool is for an engineer who has a geometry and no CFD run for it.

## Result view

The Result view treats the prediction as a solution to work with:

- **Fields and forces.** The pressure field on the car, its drag contribution or the spread between the networks, the colour range typed or dragged on the legend. A car that carries CFD, its file's own or one attached from another file of the same surface in the Files panel (which refuses another car's), also offers the CFD pressure, on the pressure scale, and the error, the predicted pressure minus the CFD's; a variant offers its change, its pressure minus its car's predicted unchanged. The whole car's pressure force coefficients $C_x$, $C_y$ and $C_z$, beside the CFD's when the file carries CFD, with the range of $C_x$ across the five networks; the highest and lowest pressure and where they are, flown to from the result card; the frontal area and size. A line appears when the geometry lies outside the training cars' dimensions.
- **Sections and probes.** A cut normal to x, y or z, placed by its slider, by typing or by dragging its line on the car, with the pressure along it plotted beside the CFD's on a car that carries CFD, or on a variant beside its car's predicted unchanged; a cursor on the plot, moved by the pointer or a finger, marks the place on the car. Probes in named sets, placed by clicking or at stations along the section, each read on the file's own cell and on every other car at its nearest cell.
- **Variants.** The car stretched along an axis, or raised or lowered with the wheels left on the ground. A variant is predicted on the simplified mesh in about a second and opens beside its car, which is predicted unchanged on the same mesh, so the variant's change and its differences in the study are the change alone. A car and its variants make one study table.
- **Layers.** A field of one's own (an expression of the case's arrays, or values from Python), colour painted on faces, and labels, points, lines, arrows, boxes and planes, each in a named layer that is hidden or removed from the Layers menu among the view tools.
- **Files and export.** A panel on the left holds every file of the session: geometry, exports and what the assistant writes. Exports are pictures; the section, the probes and the study as CSV; the prediction as a `.vtp` on the file's own mesh (`CpPredicted` and `CpSpread` added as cell arrays) for ParaView or a loads mapping; and a report: one HTML page of the analysis on screen that loads nothing.

Every action, from a control or from the assistant, is one operation in one registry (`web/app/operations.js`), so both act on the car and the files alike.

The Model page, from the toggle in the top bar, draws the model as one diagram for a reader new to it: the whole pipeline in one line, then each step on a fixed reference car, run 1, with a point the reader drags to see what the networks are told there.

## How the analysis runs in the browser

The whole analysis runs in the browser, so the 650 MB file never leaves the machine; `uv run surrogate app` serves the files and the assistant's route to its model.

`web/engine/` reads the `.vtp`, builds the same lookups and inputs as `src/surrogate/features.py`, and runs the five networks on the GPU at every cell of the file. `web/engine/result.js` documents everything it returns.

- The lookups' surface grids are built on the GPU too (about a second for run 1, where Python takes about 43 s), or on the CPU without WebGPU.
- A first result comes from about a million cells spaced evenly by area; the numbers summed over every cell replace it a few seconds later.
- On an Apple M5 Pro, run 1 (8.8M cells) shows its result about 5 s after the drop and its all-cell numbers at about 12.5 s. Without WebGPU the grids and the networks run on the CPU (about 48 s to the result on the same machine).
- Beside the analysis of a file, the engine predicts every cell of a 600,000-triangle mesh at once, for variants and the assistant's studies; writes any surface with cell arrays as a `.vtp`; and takes the CFD pressure from a second file when the analysed one has none, cell for cell when the two files share their polygons and by nearest cell otherwise.

`web/viewer/` draws the file's own mesh, every cell (17.7M triangles for run 1), where the GPU keeps up. A 600,000-triangle simplified copy, made by edge collapse onto the file's own points, serves section cuts, slower GPUs and the bundled examples.

The Python pipeline is the reference. `tests/test_web_engine.py` runs the browser engine's modules under Bun or Deno and holds its cell geometry, inputs, each network's $C_p$ and the spread to the Python values. On the two holdout files the all-cell drag from the browser and from `run_vtp_inference` agree to `0.0007`, which is the difference between their two frontal-area estimates.

## Vendored libraries

There is no build step: the app is plain ES modules, with three.js, meshoptimizer and the two typefaces (Hanken Grotesk and TASA Orbiter, under the SIL Open Font License) vendored in `web/vendor/`.
