"""The `surrogate` command: everything a user does with the model, one subcommand per step.

    uv run surrogate                   the commands, and what this copy holds with the command for each gap
    uv run surrogate app               serve Surrogate Model, the browser tool
    uv run surrogate predict 1         the model at every cell of DrivAerML run 1, fetched when missing, or of a .vtp
    uv run surrogate fetch             the twelve DrivAerML surface files -> data/raw/
    uv run surrogate preprocess        twelve cars -> data/processed/ (samples and inputs)
    uv run surrogate train             five-fold ensemble and ablations -> artifacts/; --check writes nothing
    uv run surrogate export            artifacts/ and data/processed/ -> web/model/
    uv run surrogate examples          runs 1 and 500, analysed in headless Chrome -> web/examples/
    uv run surrogate reproduce         each step from preprocess on whose output is not current, then the tool
    uv run surrogate status            what this copy holds, and the command for each gap

Each step reads and writes fixed places in the repository. Its output is current, and `reproduce` skips it,
when it was made from the current output of the step before: preprocessed inputs of the width `features.py`
defines, a checkpoint newer than them, `web/model/` byte for byte the export of the checkpoint and of
`data/processed/`, and examples computed with `web/model/`'s weights.

Every long step shows a status line (`progress.py`) and ends with the time it took and the command to run next.
"""

from __future__ import annotations

import argparse
import dataclasses
import errno
import hashlib
import http.client
import io
import json
import re
import shutil
import sys
import tempfile
import urllib.error
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from surrogate.dataset import (
    ALLOWED_TRAIN_RUNS,
    DATASET_URL,
    HOLDOUT_RUNS,
    RemoteSurface,
    fetch_surface,
    remote_surfaces,
    surface_path,
)
from surrogate.export_training_cars import export_training_cars
from surrogate.export_web import export_web_model, feature_definition
from surrogate.features import FEATURE_COLUMNS
from surrogate.inference import INFERENCE_STEPS, InferenceResult, run_vtp_inference, write_prediction_vtp
from surrogate.model import DIMENSIONS, SurrogateEnsemble, select_device
from surrogate.preprocess import PREPROCESS_STEPS, preprocess_cases
from surrogate.progress import MEASURED_ON, Progress, Step, clock, duration
from surrogate.server import REPO, WEB_ROOT, make_server
from surrogate.train import train_and_evaluate, training_steps
from surrogate.write_examples import BUNDLED, CHROME, EXAMPLES, EXAMPLES_STEPS, write_all_examples

RAW = REPO / "data/raw"
PROCESSED = REPO / "data/processed"
MANIFEST = PROCESSED / "manifest.json"
ARTIFACTS = REPO / "artifacts"
CHECKPOINT = ARTIFACTS / "ensemble.pt"
METRICS = ARTIFACTS / "metrics.json"
WEB_MODEL = WEB_ROOT / "model"
TRAINING_CARS = WEB_MODEL / "training-cars.json"

RUN_SETS = {
    "all": ALLOWED_TRAIN_RUNS + HOLDOUT_RUNS,
    "training": ALLOWED_TRAIN_RUNS,
    "holdout": HOLDOUT_RUNS,
}
DEVICES = ("cpu", "mps", "cuda")
DEFAULT_PORT = 8740
DEFAULT_SAMPLES = 120_000
DEFAULT_EPOCHS = 30
DEFAULT_SEED = 42

# A training run on MEASURED_ON's CPU, with and without the ablations: its networks being the five folds and
# the twenty that fit the spread, and with the ablations also the learning curve's 55, whose CPU time is its
# networks' count times one network's measured 58 s. The GPU's times are the sum of `training_steps`.
CPU_TRAIN_SECONDS = {True: 5242, False: 1066}


class StepError(Exception):
    """A step cannot run; the message says what is missing and the command that provides it."""


# Status ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    """One part of this copy: what is there, and the command that makes it current (None when it is)."""

    label: str
    state: str
    fix: str | None = None


def surfaces_check(label: str, runs: tuple[int, ...], run_set: str) -> Check:
    present = sum(surface_path(RAW, run).is_file() for run in runs)
    where = surface_path(RAW, runs[0]).parent.parent.relative_to(REPO)
    state = f"{present} of {len(runs)} in {where}"
    return Check(label, state, None if present == len(runs) else f"uv run surrogate fetch {run_set}")


def processed_check() -> Check:
    label, fix = "Preprocessed cars", "uv run surrogate preprocess"
    if not MANIFEST.is_file():
        return Check(label, "none (needed to retrain)", fix)
    manifest = json.loads(MANIFEST.read_text())
    if manifest["feature_dim"] != FEATURE_COLUMNS.total_dim:
        defined = FEATURE_COLUMNS.total_dim
        return Check(label, f"{manifest['feature_dim']} inputs, but features.py defines {defined}", fix)
    runs = {case["run"] for case in manifest["cases"]}
    if missing := [run for run in RUN_SETS["all"] if run not in runs]:
        return Check(label, f"runs {', '.join(map(str, missing))} missing", fix)
    cells = manifest["samples_per_case"]
    return Check(label, f"{len(runs)} cars, {cells:,} cells each, {manifest['feature_dim']} inputs")


def model_check() -> Check:
    label, fix = "Trained model", "uv run surrogate train"
    if not (CHECKPOINT.is_file() and METRICS.is_file()):
        return Check(label, "none", fix)
    in_dim = int(torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["in_dim"])
    if in_dim != FEATURE_COLUMNS.total_dim:
        return Check(label, f"takes {in_dim} inputs, but features.py defines {FEATURE_COLUMNS.total_dim}", fix)
    if MANIFEST.is_file() and MANIFEST.stat().st_mtime > CHECKPOINT.stat().st_mtime:
        return Check(label, "older than data/processed", fix)
    metrics = json.loads(METRICS.read_text())
    cv, holdout = metrics["shipped_cv"], metrics["shipped_holdout"]
    scores = f"CV R² {cv['mean_r2']:.4f}, holdout R² {holdout['mean_r2']:.4f}"
    return Check(label, f"{scores}, holdout Cd,p MAE {holdout['cd_p_mae']:.4f}")


def web_model_check() -> Check:
    label = "Browser model"
    if not CHECKPOINT.is_file():
        return Check(label, "not checked without artifacts/ensemble.pt")
    with tempfile.TemporaryDirectory() as scratch:
        try:
            export_web_model(CHECKPOINT, Path(scratch))
        except ValueError as error:  # the checkpoint does not fit features.py; the model's line says so
            return Check(label, f"cannot be exported: {error}")
        current = all(
            (Path(scratch) / name).read_bytes() == (WEB_MODEL / name).read_bytes()
            for name in ("ensemble.json", "weights.bin")
        )
    if current:
        return Check(label, "web/model is the export of artifacts/ensemble.pt")
    return Check(label, "web/model is not the export of artifacts/ensemble.pt", "uv run surrogate export")


def training_cars_check() -> Check:
    label = "Training cars"
    if not MANIFEST.is_file():
        return Check(label, "not checked without data/processed")
    with tempfile.TemporaryDirectory() as scratch:
        export_training_cars(PROCESSED, Path(scratch) / TRAINING_CARS.name)
        current = (Path(scratch) / TRAINING_CARS.name).read_bytes() == TRAINING_CARS.read_bytes()
    if current:
        return Check(label, f"{TRAINING_CARS.relative_to(REPO)} is made from data/processed")
    return Check(label, f"{TRAINING_CARS.relative_to(REPO)} is not made from data/processed", "uv run surrogate export")


def examples_check() -> Check:
    weights = hashlib.sha256((WEB_MODEL / "weights.bin").read_bytes()).hexdigest()
    definition = feature_definition()
    names, stale = [], []
    for summary_path in sorted(EXAMPLES.glob("*.json")):
        summary = json.loads(summary_path.read_text())
        names.append(summary_path.stem)
        if summary["model"] != weights or summary["lookups"]["definition"] != definition:
            stale.append(summary_path.stem)
    if not names:
        return Check("Examples", "none in web/examples", "uv run surrogate examples")
    if stale:
        return Check("Examples", f"{', '.join(stale)} computed with another model", "uv run surrogate examples")
    return Check("Examples", f"{', '.join(names)} computed with web/model")


def chrome_check() -> Check:
    label = "Chrome for examples"
    if Path(CHROME).is_file():
        return Check(label, CHROME)
    return Check(label, f"not found at {CHROME}", "install Google Chrome, or set CHROME to its executable")


def javascript_check() -> Check:
    label = "Bun or Deno for tests"
    runtimes = [name for name in ("bun", "deno") if shutil.which(name)]
    if runtimes:
        return Check(label, ", ".join(runtimes))
    return Check(label, "neither; the engine tests skip", "install Bun: https://bun.sh")


def device_check() -> Check:
    return Check("PyTorch device", describe_device(select_device()))


def status_checks() -> list[Check]:
    return [
        surfaces_check("Training surfaces", ALLOWED_TRAIN_RUNS, "training"),
        surfaces_check("Holdout surfaces", HOLDOUT_RUNS, "holdout"),
        processed_check(),
        model_check(),
        web_model_check(),
        training_cars_check(),
        examples_check(),
        chrome_check(),
        javascript_check(),
        device_check(),
    ]


def print_status() -> None:
    checks = status_checks()
    width = max(len(check.label) for check in checks)
    for check in checks:
        fix = f"  → {check.fix}" if check.fix else ""
        print(f"{check.label:<{width}}  {check.state}{fix}")


# Steps -----------------------------------------------------------------------------------------------


def describe_device(device: torch.device) -> str:
    return {"mps": "mps (Apple GPU)", "cuda": "cuda (NVIDIA GPU)", "cpu": "cpu"}[device.type]


def chosen_device(name: str | None) -> torch.device:
    """The device asked for, or the best one here when none was."""
    if name is None:
        return select_device()
    available = {"cpu": True, "mps": torch.backends.mps.is_available(), "cuda": torch.cuda.is_available()}
    if not available[name]:
        raise StepError(
            f"This machine has no {name} device. Use --device cpu, or leave --device out for the best one here."
        )
    return torch.device(name)


def measured(seconds: float) -> str:
    return f"about {duration(seconds)} on {MEASURED_ON}"


def finish(progress: Progress, next_command: str) -> None:
    """The closing lines every long step prints: how long it took and what to run next."""
    print(f"Done in {clock(progress.elapsed)}.")
    print(f"Next: {next_command}")


def remote(runs: tuple[int, ...]) -> list[RemoteSurface]:
    """The dataset's surface files for `runs`, with their sizes."""
    try:
        return remote_surfaces(runs, RAW, DATASET_URL)
    except urllib.error.HTTPError as error:
        if error.code == 404 and len(runs) == 1:
            raise StepError(
                f"DrivAerML has no run {runs[0]}. Its runs are numbered 1 to 500 with a few missing; try another."
            ) from error
        raise StepError(f"The DrivAerML dataset at {DATASET_URL} answered {error}. Run the command again.") from error
    except OSError as error:
        raise StepError(
            f"Could not reach the DrivAerML dataset at {DATASET_URL} ({error}). "
            "Check the connection and run the command again."
        ) from error


def download(surfaces: list[RemoteSurface]) -> Progress:
    """Fetch each of `surfaces` not yet here, under one status line for all of them."""
    needed = [surface for surface in surfaces if not surface.present]
    total = sum(surface.size for surface in needed)
    print(f"Downloading {len(needed)} DrivAerML surface file{'s' * (len(needed) > 1)}, {total / 1e9:.1f} GB.")
    # One step, so its seconds only set the share done, which is all of it.
    with Progress([Step("Downloading", total, "bytes", 1)], sys.stdout) as progress:
        progress.step("Downloading")
        for surface in needed:
            progress.describe(f"run_{surface.run}")
            held = 0

            def report(now_held: int) -> None:
                nonlocal held
                progress.advance(now_held - held)
                held = now_held

            try:
                fetch_surface(surface, report)
            except (OSError, http.client.HTTPException) as error:
                raise StepError(
                    f"The download of run_{surface.run} stopped ({error}). "
                    "Run the same command again to continue where it stopped."
                ) from error
            progress.note(f"Wrote {surface.path.relative_to(REPO)} ({surface.size / 1e6:.0f} MB)")
    return progress


def fetch(run_set: str) -> None:
    surfaces = remote(RUN_SETS[run_set])
    next_command = "uv run surrogate preprocess" if run_set == "all" else "uv run surrogate predict 1"
    if all(surface.present for surface in surfaces):
        print(f"All {len(surfaces)} surface files are already in data/raw.")
        print(f"Next: {next_command}")
    else:
        finish(download(surfaces), next_command)


def require_surfaces() -> None:
    """The pipeline reads all twelve surface files: ten to train on, two to report."""
    if missing := [run for run in RUN_SETS["all"] if not surface_path(RAW, run).is_file()]:
        raise StepError(
            f"{len(missing)} of the twelve DrivAerML surface files are not in data/raw "
            f"(runs {', '.join(map(str, missing))}). "
            "Fetch them with: uv run surrogate fetch  (7.8 GB for all twelve)"
        )


def preprocess(samples: int) -> None:
    require_surfaces()
    cars = len(RUN_SETS["all"])
    estimate = measured(sum(step.seconds for step in PREPROCESS_STEPS) * samples / DEFAULT_SAMPLES)
    print(f"Preprocessing {cars} cars, {samples:,} cells each: {estimate}.")
    with Progress(PREPROCESS_STEPS, sys.stdout) as progress:
        preprocess_cases(RAW, PROCESSED, samples=samples, progress=progress)
    print(f"Wrote data/processed/ ({cars} cars).")
    finish(progress, "uv run surrogate train")


def train(epochs: int, seed: int, check: bool, device: torch.device) -> None:
    """Train into artifacts/, or with `check` into a scratch directory, and compare the result with the
    model in artifacts/ before the run."""
    processed = processed_check()
    if processed.fix:
        if any(not surface_path(RAW, run).is_file() for run in RUN_SETS["all"]):
            raise StepError(
                "Training reads the preprocessed cars, and data/raw does not hold the twelve DrivAerML surface "
                "files they come from. Fetch them (7.8 GB) and preprocess them:\n"
                "  uv run surrogate fetch\n"
                "  uv run surrogate preprocess"
            )
        raise StepError(f"The preprocessed cars are not current ({processed.state}). Run: {processed.fix}")
    samples = json.loads(MANIFEST.read_text())["samples_per_case"]
    steps = training_steps(not check)
    scale = epochs / DEFAULT_EPOCHS * samples / DEFAULT_SAMPLES
    match device.type:
        case "mps":
            estimate = measured(sum(step.seconds for step in steps) * scale)
        case "cpu":
            gpu = duration(sum(step.seconds for step in steps) * scale)
            estimate = f"{measured(CPU_TRAIN_SECONDS[not check] * scale)}'s CPU, against {gpu} on its GPU"
        case _:
            estimate = f"no time measured on {device.type}; the estimate follows once training starts"
    what = "the shipped networks only, compared with artifacts/" if check else "with the ablations"
    print(f"Training on {describe_device(device)}, {epochs} epochs, seed {seed}, {what}: {estimate}.")

    before = json.loads(METRICS.read_text()) if METRICS.is_file() else None
    before_weights = CHECKPOINT.read_bytes() if CHECKPOINT.is_file() else None
    with tempfile.TemporaryDirectory() as scratch:
        out_dir = Path(scratch) if check else ARTIFACTS
        with Progress(steps, sys.stdout) as progress:
            after = train_and_evaluate(
                PROCESSED, out_dir, epochs=epochs, seed=seed, run_ablations=not check, device=device, progress=progress
            )
        weights = (out_dir / "ensemble.pt").read_bytes()

    print()
    print_scores_against(before, after)
    if before_weights is not None:
        print(compare_checkpoints(before_weights, weights))
    if check:
        print("Nothing was written: artifacts/ is as it was.")
        finish(progress, "uv run surrogate train  (the full run, written to artifacts/)")
        return
    print("Wrote artifacts/ensemble.pt and artifacts/metrics.json.")
    finish(progress, web_model_check().fix or "uv run surrogate predict 1")


def compare_checkpoints(before: bytes, after: bytes) -> str:
    """Whether a run's checkpoint holds the networks of the one in artifacts/ before it: the same file, the same
    weights and scaling with other bytes around them, or other networks."""
    if after == before:
        return "This run's ensemble.pt is the same, byte for byte, as artifacts/ensemble.pt before it."
    old, new = (torch.load(io.BytesIO(data), map_location="cpu", weights_only=False) for data in (before, after))
    if (
        old["spread_scale"] == new["spread_scale"]
        and len(old["folds"]) == len(new["folds"])
        and all(same_network(a, b) for a, b in zip(old["folds"], new["folds"], strict=True))
    ):
        return (
            "This run's five networks are the same, weight for weight, as those in artifacts/ensemble.pt "
            "before it; the rest of the file differs."
        )
    return "This run's five networks differ from those in artifacts/ensemble.pt before it."


def same_network(a: dict, b: dict) -> bool:
    """Two checkpoint folds with the same input and output scaling and the same weights."""
    scaling = all(np.array_equal(a[key], b[key]) for key in ("x_mean", "x_std", "y_mean", "y_std"))
    weights_a, weights_b = a["state_dict"], b["state_dict"]
    return (
        scaling
        and weights_a.keys() == weights_b.keys()
        and all(torch.equal(weights_a[key], weights_b[key]) for key in weights_a)
    )


def print_scores_against(before: dict | None, after: dict) -> None:
    """The run's cross-validation and holdout scores beside those of the model it is compared with."""
    columns = [
        ("CV R²", "shipped_cv", "mean_r2"),
        ("CV Cd,p MAE", "shipped_cv", "cd_p_mae"),
        ("Holdout R²", "shipped_holdout", "mean_r2"),
        ("Holdout Cd,p MAE", "shipped_holdout", "cd_p_mae"),
    ]
    rows = ([("artifacts/ before", before)] if before is not None else []) + [("This run", after)]
    width = max(len(label) for label, _ in rows)
    widths = [max(len(title), len("0.0000")) for title, _, _ in columns]
    print(" " * width + "".join(f"  {title:>{w}}" for (title, _, _), w in zip(columns, widths, strict=True)))
    for label, metrics in rows:
        cells = zip(columns, widths, strict=True)
        values = "".join(f"  {metrics[part][key]:>{w}.4f}" for (_, part, key), w in cells)
        print(f"{label:<{width}}{values}")


def export() -> None:
    if not (CHECKPOINT.is_file() and METRICS.is_file()):
        raise StepError("There is no trained model in artifacts/ to export. Train one with: uv run surrogate train")
    export_web_model(CHECKPOINT, WEB_MODEL)
    print("Wrote web/model/ensemble.json and web/model/weights.bin")
    if MANIFEST.is_file():
        export_training_cars(PROCESSED, TRAINING_CARS)
        print(f"Wrote {TRAINING_CARS.relative_to(REPO)}")
    else:
        print(
            f"Kept {TRAINING_CARS.relative_to(REPO)}: it is made from data/processed, "
            "which this copy does not have (uv run surrogate preprocess)."
        )
    print("Next: uv run surrogate examples")


def examples() -> None:
    runs = " and ".join(name.removeprefix("run_") for name in BUNDLED)
    if missing := [run for run in HOLDOUT_RUNS if not surface_path(RAW, run).is_file()]:
        raise StepError(
            f"The examples are runs {runs}, and data/raw lacks {', '.join(f'run_{run}' for run in missing)}. "
            "Fetch them with: uv run surrogate fetch holdout"
        )
    require_chrome()
    if (model := web_model_check()).fix:
        raise StepError(f"{model.state}, so the examples would show another model. Run first: {model.fix}")
    estimate = measured(sum(step.seconds for step in EXAMPLES_STEPS))
    print(f"Analysing runs {runs} in headless Chrome: {estimate}.")
    with Progress(EXAMPLES_STEPS, sys.stdout) as progress:
        write_all_examples(progress)
    finish(progress, "uv run surrogate app")


def app(port: int) -> None:
    try:
        server = make_server(port)
    except OSError as error:
        if error.errno != errno.EADDRINUSE:
            raise
        raise StepError(
            f"Port {port} is already in use. Choose another with --port, for example --port {port + 1}."
        ) from error
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Surrogate Model is at {url}")
    print("Open it in Chrome, Edge or Safari. Ctrl+C stops the server.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def require_chrome() -> None:
    if (chrome := chrome_check()).fix:
        raise StepError(f"Writing the examples needs Google Chrome with WebGPU, {chrome.state}: {chrome.fix}.")


def reproduce(port: int) -> None:
    """Every step whose output is not current, in order, then the tool."""
    require_surfaces()
    require_chrome()
    steps: list[tuple[list[Callable[[], Check]], Callable[[], None]]] = [
        ([processed_check], lambda: preprocess(DEFAULT_SAMPLES)),
        ([model_check], lambda: train(DEFAULT_EPOCHS, DEFAULT_SEED, False, select_device())),
        ([web_model_check, training_cars_check], export),
        ([examples_check], examples),
    ]
    for checks, run in steps:
        outputs = [check() for check in checks]
        for output in outputs:
            print(f"{output.label}: {output.state}", flush=True)
        if any(output.fix for output in outputs):
            run()
    app(port)


# Predict ---------------------------------------------------------------------------------------------


def surface_file(target: str) -> tuple[Path, int | None]:
    """The file `predict` reads, and its DrivAerML run when it is one: a run number names the run's file in
    data/raw, fetched first when it is not there; anything else is a path."""
    if target.isdigit():
        run = int(target)
        path = surface_path(RAW, run)
        if not path.is_file():
            download(remote((run,)))
        return path, run
    path = Path(target)
    if not path.is_file():
        raise StepError(f"There is no file at {path}. Give a .vtp file, or a DrivAerML run number such as 1.")
    named = re.fullmatch(r"boundary_(\d+)\.vtp", path.name)
    return path, int(named.group(1)) if named else None


def predict(target: str, output: Path | None, json_path: Path | None, device: torch.device) -> None:
    if output is not None and output.suffix.lower() != ".vtp":
        suggestion = output.with_suffix(".vtp").name
        raise StepError(f"--output writes a .vtp file; give a name ending in .vtp, such as {suggestion}.")
    if not CHECKPOINT.is_file():
        raise StepError("There is no trained model at artifacts/ensemble.pt. Train one with: uv run surrogate train")
    file, run = surface_file(target)
    ensemble = SurrogateEnsemble.load(CHECKPOINT, device=device)
    if run is None:
        what = str(file)
    elif run in ALLOWED_TRAIN_RUNS:
        what = f"DrivAerML run {run}, one of the ten training cars,"
    else:
        what = f"DrivAerML run {run}, which the model was not trained on,"
    print(f"Predicting {what} on {describe_device(device)}.")
    with Progress(INFERENCE_STEPS, sys.stdout) as progress:
        result = run_vtp_inference(file, ensemble, progress=progress)
    if output is not None:
        write_prediction_vtp(file, result, output)
        print(f"Wrote {output} with cell arrays CpPredicted and CpSpread")
    if json_path is not None:
        json_path.write_text(json.dumps(prediction_summary(result), indent=1) + "\n")
        print(f"Wrote {json_path}")
    print_prediction(result, ensemble)
    finish(progress, f"uv run surrogate app, then drop {shown(file)} on the page to see the prediction on the car")


def shown(path: Path) -> Path:
    """`path` as the user would type it from where they are."""
    return path.resolve().relative_to(Path.cwd()) if path.resolve().is_relative_to(Path.cwd()) else path


def print_prediction(result: InferenceResult, ensemble: SurrogateEnsemble) -> None:
    """What the model says about the car, then its drag and lift (beside the CFD's when the file has it)."""
    bounds = ensemble.training_envelope["descriptor_bounds"]
    sizes = []
    for name, value in zip(DIMENSIONS, result.dimensions, strict=True):
        low, high = bounds[name]["min"], bounds[name]["max"]
        where = "inside" if low <= value <= high else "outside"
        sizes.append(f"{name} {value:.2f} m ({where} {low:.2f}–{high:.2f})")
    forces, coverage, cfd = result.forces, result.coverage, result.ground_truth_forces
    facts = [
        ("Cells", f"{result.n_cells:,}, {result.total_area:.2f} m², frontal area {result.frontal_area:.3f} m²"),
        ("Size", ", ".join(sizes)),
        ("Coverage", coverage["summary"]),
        *[("", reason) for reason in coverage["reasons"]],
        ("Spread", f"mean σ(Cp) {coverage['mean_cp_std']:.3f}, 95th percentile {coverage['p95_cp_std']:.3f}"),
        ("Each network", "Cd,p " + "  ".join(f"{cd:.4f}" for cd in forces["cd_p_per_fold"])),
    ]
    width = max(len(label) for label, _ in facts)
    for label, text in facts:
        print(f"{label:<{width}}  {text}")

    print()
    rows = [("Drag Cd,p", "cd_p"), ("Lift Cl,p", "cl_p")]
    label_width = len("Cp, area-weighted")
    if cfd is None or (agreement := result.ground_truth_metrics) is None:
        print(f"{'':<{label_width}}  {'Predicted':>9}")
        for label, key in rows:
            print(f"{label:<{label_width}}  {forces[key]:>9.4f}")
        print("The file has no CpMeanTrim, so there is no CFD to compare with.")
        return
    print(f"{'':<{label_width}}  {'Predicted':>9}  {'CFD':>7}  {'Error':>7}")
    for label, key in rows:
        print(f"{label:<{label_width}}  {forces[key]:>9.4f}  {cfd[key]:>7.4f}  {forces[key] - cfd[key]:>+7.4f}")
    scores = f"R² {agreement['r2']:.4f}, MAE {agreement['mae']:.4f}"
    print(f"{'Cp, area-weighted':<{label_width}}  {scores} against CFD over all {result.n_cells:,} cells")


def prediction_summary(result: InferenceResult) -> dict:
    """Everything in `result` but the per-cell arrays, which the .vtp carries."""
    arrays = ("cp_mean", "cp_std")
    return {field.name: getattr(result, field.name) for field in dataclasses.fields(result) if field.name not in arrays}


# Command line ----------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="surrogate",
        description="Run, check and retrain the DrivAerML surface-pressure model. "
        "Run a command with -h for its options.",
    )
    commands = parser.add_subparsers(title="commands", dest="command", metavar="COMMAND")

    def command(name: str, summary: str) -> argparse.ArgumentParser:
        return commands.add_parser(name, help=summary, description=summary[0].upper() + summary[1:] + ".")

    sub = command("app", f"serve Surrogate Model, the browser tool, at http://127.0.0.1:{DEFAULT_PORT}/")
    sub.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port to listen on (default {DEFAULT_PORT})")
    sub.set_defaults(run=lambda args: app(args.port))

    sub = command("predict", "predict Cp at every cell of a DrivAerML run or a .vtp surface file")
    sub.add_argument(
        "target",
        metavar="RUN_OR_FILE",
        help="a DrivAerML run number (1 to 500; its 0.6 GB file is fetched into data/raw when missing), "
        "or a .vtp file of a closed car surface",
    )
    sub.add_argument(
        "--output", type=Path, help="write the file with cell arrays CpPredicted and CpSpread to this .vtp"
    )
    sub.add_argument("--json", type=Path, help="write the summary as JSON to this file")
    add_device(sub)
    sub.set_defaults(run=lambda args: predict(args.target, args.output, args.json, chosen_device(args.device)))

    sub = command("fetch", "download DrivAerML surface files into data/raw")
    sub.add_argument(
        "runs",
        nargs="?",
        choices=RUN_SETS,
        default="all",
        help="all twelve (default, 7.8 GB), the ten training cars (6.5 GB) or the two holdout cars (1.3 GB); "
        "files already here are skipped and an interrupted download continues",
    )
    sub.set_defaults(run=lambda args: fetch(args.runs))

    sub = command("preprocess", "sample the twelve cars and compute their inputs into data/processed")
    sub.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help=f"cells drawn per car, by area (default {DEFAULT_SAMPLES:,})",
    )
    sub.set_defaults(run=lambda args: preprocess(args.samples))

    sub = command("train", "train the five networks, the ablations and the learning curve into artifacts")
    sub.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS, help=f"passes over the training cells (default {DEFAULT_EPOCHS})"
    )
    sub.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"random seed (default {DEFAULT_SEED})")
    sub.add_argument(
        "--check",
        action="store_true",
        help="train only the five networks and their spread, compare them with artifacts/, and write nothing",
    )
    add_device(sub)
    sub.set_defaults(run=lambda args: train(args.epochs, args.seed, args.check, chosen_device(args.device)))

    command("export", "write web/model from the trained model").set_defaults(run=lambda args: export())

    sub = command("examples", "write web/examples with the browser engine in headless Chrome")
    sub.set_defaults(run=lambda args: examples())

    sub = command("reproduce", "run each step that is not current, then serve the tool")
    sub.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"port to serve the tool on (default {DEFAULT_PORT})"
    )
    sub.set_defaults(run=lambda args: reproduce(args.port))

    sub = command("status", "what this copy holds, and the command for each gap")
    sub.set_defaults(run=lambda args: print_status())
    return parser


def add_device(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device", choices=DEVICES, help="where the networks run (default: the GPU when there is one, else cpu)"
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        print("\nThis copy:")
        print_status()
        return
    try:
        args.run(args)
    except StepError as error:
        parser.exit(1, f"{error}\n")
