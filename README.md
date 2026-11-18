# Surface pressure surrogate for DrivAerML

A model that predicts the mean pressure coefficient $C_p$ on every cell of a DrivAerML car from its geometry alone, and Surrogate Model, a browser tool for working with what it predicts. I trained it on the ten assigned cases (runs 80, 98, 102, 208, 213, 281, 397, 425, 445 and 474). On the two cars I kept out of training entirely, runs 1 and 500, it scores an area-weighted $R^2$ of 0.956 against the CFD.

![Surrogate Model showing the predicted pressure on DrivAerML run 1](docs/images/surrogate-model.png)

## Quick start

The one prerequisite is [uv](https://docs.astral.sh/uv/getting-started/installation/). The first command you run installs the dependencies, about 300 MB of packages that are mostly PyTorch and VTK (20 s on my machine), plus Python itself if you don't have 3.11 to 3.13.

```bash
git clone https://github.com/nategriffin26/surrogate-model.git
cd surrogate-model
uv run surrogate
```

That last command lists the commands and what your copy has. The clone already holds the trained model and the two held-out cars analysed, so the web app works straight away. The DrivAerML files themselves are about 0.6 GB a car and download only when a command needs them.

| To | Run | Time on my Apple M5 Pro | Download |
| :--- | :--- | :--- | :--- |
| Use the web app | `uv run surrogate app` | starts at once | none |
| Check it on a car it never saw | `uv run surrogate predict 1` | about 75 s | 0.66 GB |
| Train it yourself | `fetch`, `preprocess`, `train` | about 25 min plus the download | 7.8 GB |

## Run the web app

```bash
uv run surrogate app
```

Open http://127.0.0.1:8740/ in Chrome, Edge or Safari. The first screen has runs 1 and 500 ready to open; click one to see the predicted pressure on the car with the CFD forces beside it. From there you can switch the field to drag, the spread between the networks, the CFD pressure or the error; cut a section and plot $C_p$ along it against the CFD; or stretch, raise or lower the car and see what changes. You can also drop any `boundary_<run>.vtp` on the page, like the one `predict` downloads below, and the model runs on it in the browser in a few seconds.

If port 8740 is taken, add `--port 8741`. The Assistant panel needs an OpenRouter key; [docs/assistant.md](docs/assistant.md) covers it, and the rest of the tool works without one.

## Check it on a car it never saw

```bash
uv run surrogate predict 1
```

This downloads run 1 into `data/raw/holdout/` the first time, predicts all 8.8 million cells, and since the file carries the CFD answer, finishes with the two side by side:

```
                   Predicted      CFD    Error
Drag Cd,p             0.2592   0.2736  -0.0144
Lift Cl,p             0.0659   0.0652  +0.0007
Cp, area-weighted  R² 0.9503, MAE 0.0337 against CFD over all 8,828,095 cells
```

Above that it prints the car's size against the training cars' and how much the five networks disagree. Runs 1 and 500 are the two cars I held out from training, feature scaling and every design decision. Any other run from 1 to 500 that isn't one of the ten training cars is just as unseen, so `uv run surrogate predict 250` works too; that car is wider than any of the training cars, and the output says so. `--output run_1.vtp` writes the prediction back into the file as `CpPredicted` and `CpSpread` for ParaView, and `--json run_1.json` saves the summary.

## Train it yourself

```bash
uv run surrogate fetch        # the ten training cars and the two holdout cars, 7.8 GB
uv run surrogate preprocess   # about 9 min
uv run surrogate train        # about 16 min
```

`preprocess` samples 120,000 cells per car and computes their inputs; `train` fits the five networks and their spread, then the ablations and the learning curve behind the results in [docs/model.md](docs/model.md). Each command shows what it's working on, how far along it is and the time left, and ends with the command to run next. `train` finishes by putting your run's scores beside the model that was in `artifacts/` before it, which in a fresh clone is mine, and checks whether the networks came out the same:

```
                    CV R²  CV Cd,p MAE  Holdout R²  Holdout Cd,p MAE
artifacts/ before  0.9458       0.0196      0.9558            0.0128
This run           0.9458       0.0196      0.9558            0.0128
```

On my M5 Pro every score and every network weight came out identical to mine, starting from a fresh `preprocess`. For a quicker check, `uv run surrogate train --check` (about 5 min) trains only the five shipped networks and their spread, prints the same comparison and writes nothing.

Without a GPU it trains on the CPU, several times slower: `train --check` took 18 min there, and cross-validation $R^2$ came out 0.9454 instead of 0.9458.

## More

- [docs/model.md](docs/model.md): the approach, the inputs, the results table and the limitations.
- [docs/web-app.md](docs/web-app.md): what the tool does and how it runs in the browser.
- [docs/assistant.md](docs/assistant.md): the assistant, its Python kernel and its key.
- [docs/development.md](docs/development.md): every pipeline step, predicting from Python, the tests and lint, and the repository layout.
- [docs/demo-outline.md](docs/demo-outline.md): how I'd demo it to a customer.

## Data and license

DrivAerML (Ashton et al., 2024) is licensed CC BY-SA 4.0; see [data/README.md](data/README.md) and `data/metadata/LICENSE.txt`. Only the surface files are used, and only the ten assigned runs train the model.
