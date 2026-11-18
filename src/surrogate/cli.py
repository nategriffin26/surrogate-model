"""The `surrogate` command: everything a user does with the model, one subcommand per step.

    uv run surrogate                   the commands, and what this copy holds with the command for each gap
    uv run surrogate app               serve the tool
    uv run surrogate predict FILE.vtp  the model at every cell of a surface file
    uv run surrogate fetch             the twelve DrivAerML surface files -> data/raw/
    uv run surrogate preprocess        twelve cars -> data/processed/ (samples and inputs)
    uv run surrogate train             five-fold ensemble and ablations -> artifacts/
    uv run surrogate export            artifacts/ and data/processed/ -> web/model/
    uv run surrogate examples          runs 1 and 500, analysed in headless Chrome -> web/examples/
    uv run surrogate reproduce         each step from preprocess on whose output is not current, then the tool

Each step reads and writes fixed places in the repository. Its output is current, and `reproduce` skips it,
when it was made from the current output of the step before: preprocessed inputs of the width `features.py`
defines, a checkpoint newer than them, `web/model/` byte for byte the export of the checkpoint and of
`data/processed/`, and examples computed with `web/model/`'s weights.
"""

from __future__ import annotations

import argparse
import dataclasses
import errno
import hashlib
import http.client
import json
import shutil
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from surrogate.dataset import ALLOWED_TRAIN_RUNS, DATASET_URL, DEFAULT_HOLDOUT_RUNS, fetch_surface, remote_surfaces, surface_path
from surrogate.export_training_cars import export_training_cars
from surrogate.export_web import export_web_model, feature_definition
from surrogate.features import FEATURE_COLUMNS
from surrogate.inference import InferenceResult, run_vtp_inference, write_prediction_vtp
from surrogate.model import SurrogateEnsemble, select_device
from surrogate.preprocess import preprocess_cases
from surrogate.server import make_server
from surrogate.train import train_and_evaluate
from surrogate.write_examples import CHROME, write_all_examples

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data/raw"
PROCESSED = REPO / "data/processed"
MANIFEST = PROCESSED / "manifest.json"
ARTIFACTS = REPO / "artifacts"
CHECKPOINT = ARTIFACTS / "ensemble.pt"
METRICS = ARTIFACTS / "metrics.json"
WEB_MODEL = REPO / "web/model"
TRAINING_CARS = WEB_MODEL / "training-cars.json"
EXAMPLES = REPO / "web/examples"

RUN_SETS = {
    "all": ALLOWED_TRAIN_RUNS + DEFAULT_HOLDOUT_RUNS,
    "training": ALLOWED_TRAIN_RUNS,
    "holdout": DEFAULT_HOLDOUT_RUNS,
}
DEVICES = ("cpu", "mps", "cuda")
DEFAULT_PORT = 8740
DEFAULT_SAMPLES = 120_000
DEFAULT_EPOCHS = 30
DEFAULT_SEED = 42

# Measured on an Apple M5 Pro (18 cores, 48 GB) with 107 inputs, 120,000 cells a car and 30 epochs: one car
# preprocessed; a training run by (device, with the ablations), its networks being the five folds and the
# twenty that fit the spread, and with the ablations also the learning curve's 55, whose CPU time is its
# networks' count times one network's measured 58 s; the two examples written in Chrome.
MEASURED_ON = "an Apple M5 Pro"
PREPROCESS_SECONDS_PER_CAR = 40
TRAIN_SECONDS = {("mps", True): 957, ("mps", False): 150, ("cpu", True): 5242, ("cpu", False): 1066}
EXAMPLES_SECONDS = 40


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
        return Check(label, f"{manifest['feature_dim']} inputs, but features.py defines {FEATURE_COLUMNS.total_dim}", fix)
    runs = {case["run"] for case in manifest["cases"]}
    if missing := [run for run in RUN_SETS["all"] if run not in runs]:
        return Check(label, f"runs {', '.join(map(str, missing))} missing", fix)
    return Check(label, f"{len(runs)} cars, {manifest['samples_per_case']:,} cells each, {manifest['feature_dim']} inputs")


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
    return Check(label, f"CV R² {cv['mean_r2']:.4f}, holdout R² {holdout['mean_r2']:.4f}, holdout Cd,p MAE {holdout['cd_p_mae']:.4f}")


def web_model_check() -> Check:
    label = "Browser model"
    if not CHECKPOINT.is_file():
        return Check(label, "not checked without artifacts/ensemble.pt")
    with tempfile.TemporaryDirectory() as scratch:
        try:
            export_web_model(CHECKPOINT, Path(scratch))
        except ValueError as error:  # the checkpoint does not fit features.py; the model's line says so
            return Check(label, f"cannot be exported: {error}")
        current = all((Path(scratch) / name).read_bytes() == (WEB_MODEL / name).read_bytes() for name in ("ensemble.json", "weights.bin"))
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
        surfaces_check("Holdout surfaces", DEFAULT_HOLDOUT_RUNS, "holdout"),
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
        raise StepError(f"This machine has no {name} device. Use --device cpu, or leave --device out for the best one here.")
    return torch.device(name)


def minutes(seconds: float) -> str:
    return f"about {max(1, round(seconds / 60))} min"


def fetch(run_set: str) -> None:
    runs = RUN_SETS[run_set]
    try:
        surfaces = remote_surfaces(runs, RAW, DATASET_URL)
    except OSError as error:
        raise StepError(f"Could not reach the DrivAerML dataset at {DATASET_URL} ({error}). Check the connection and run the command again.") from error
    needed = [surface for surface in surfaces if not surface.present]
    total = sum(surface.size for surface in needed)
    if not needed:
        print(f"All {len(surfaces)} surface files are in data/raw.")
        return
    print(f"Fetching {len(needed)} of {len(surfaces)} surface files into data/raw, {total / 1e9:.1f} GB.", flush=True)
    for surface in needed:
        label = f"run_{surface.run:<4} {surface.size / 1e6:6.0f} MB"
        report = progress_line(label, surface.size)
        try:
            fetch_surface(surface, report)
        except (OSError, http.client.HTTPException) as error:
            print(flush=True)
            raise StepError(f"The download of run_{surface.run} stopped ({error}). Run the same command again to continue where it stopped.") from error
        print(f"\r{label}  {surface.path.relative_to(REPO)}", flush=True)


def progress_line(label: str, size: int) -> Callable[[int], None]:
    """Shows the share downloaded in place on a terminal; elsewhere only the finished line is printed."""
    shown = -1

    def report(held: int) -> None:
        nonlocal shown
        percent = 100 * held // size
        if sys.stdout.isatty() and percent != shown:
            shown = percent
            print(f"\r{label}  {percent:3d}%", end="", flush=True)

    return report


def require_surfaces() -> None:
    """The pipeline reads all twelve surface files: ten to train on, two to report."""
    if missing := [run for run in RUN_SETS["all"] if not surface_path(RAW, run).is_file()]:
        raise StepError(
            f"{len(missing)} of the twelve DrivAerML surface files are not in data/raw (runs {', '.join(map(str, missing))}). "
            "Fetch them with: uv run surrogate fetch"
        )


def preprocess(samples: int) -> None:
    require_surfaces()
    cars = len(RUN_SETS["all"])
    print(f"Preprocessing {cars} cars, {samples:,} cells each: {minutes(PREPROCESS_SECONDS_PER_CAR * cars)} on {MEASURED_ON}.", flush=True)
    preprocess_cases(RAW, PROCESSED, samples=samples)
    print("Next: uv run surrogate train")


def train(epochs: int, seed: int, ablations: bool, device: torch.device) -> None:
    processed = processed_check()
    if processed.fix:
        raise StepError(f"The preprocessed cars are not current ({processed.state}). Run: {processed.fix}")
    samples = json.loads(MANIFEST.read_text())["samples_per_case"]
    print(f"Training on {describe_device(device)}, {epochs} epochs, seed {seed}{'' if ablations else ', without the ablations'}: {train_estimate(device, epochs, samples, ablations)}.", flush=True)
    train_and_evaluate(PROCESSED, ARTIFACTS, epochs=epochs, seed=seed, run_ablations=ablations, device=device)
    print("Next: uv run surrogate export")


def train_estimate(device: torch.device, epochs: int, samples: int, ablations: bool) -> str:
    """Scaled from the measured run by epochs and cells per car, which training time grows with."""
    seconds = TRAIN_SECONDS.get((device.type, ablations))
    if seconds is None:
        return f"no time measured for this on {device.type}"
    return f"{minutes(seconds * epochs / DEFAULT_EPOCHS * samples / DEFAULT_SAMPLES)} on {MEASURED_ON}"


def export() -> None:
    if not (CHECKPOINT.is_file() and METRICS.is_file()):
        raise StepError("There is no trained model in artifacts/ to export. Train one with: uv run surrogate train")
    export_web_model(CHECKPOINT, WEB_MODEL)
    print("Wrote web/model/ensemble.json and web/model/weights.bin")
    if MANIFEST.is_file():
        export_training_cars(PROCESSED, TRAINING_CARS)
        print(f"Wrote {TRAINING_CARS.relative_to(REPO)}")
    else:
        print(f"Kept {TRAINING_CARS.relative_to(REPO)}: it is made from data/processed, which this copy does not have (uv run surrogate preprocess).")
    print("Next: uv run surrogate examples")


def examples(development: bool) -> None:
    if missing := [run for run in DEFAULT_HOLDOUT_RUNS if not surface_path(RAW, run).is_file()]:
        raise StepError(f"The examples are runs {', '.join(map(str, missing))}, which are not in data/raw. Fetch them with: uv run surrogate fetch holdout")
    require_chrome()
    if (model := web_model_check()).fix:
        raise StepError(f"{model.state}, so the examples would show another model. Run first: {model.fix}")
    print(f"Analysing runs 1 and 500 in headless Chrome: {minutes(EXAMPLES_SECONDS)} on {MEASURED_ON}.", flush=True)
    write_all_examples(development)


def app(port: int) -> None:
    try:
        server = make_server(port)
    except OSError as error:
        if error.errno != errno.EADDRINUSE:
            raise
        raise StepError(f"Port {port} is already in use. Choose another with --port, for example --port {port + 1}.") from error
    print(f"Surface-pressure tool: http://127.0.0.1:{server.server_address[1]}/  (Ctrl+C to stop)", flush=True)
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
        ([model_check], lambda: train(DEFAULT_EPOCHS, DEFAULT_SEED, True, select_device())),
        ([web_model_check, training_cars_check], export),
        ([examples_check], lambda: examples(False)),
    ]
    for checks, run in steps:
        outputs = [check() for check in checks]
        for output in outputs:
            print(f"{output.label}: {output.state}", flush=True)
        if any(output.fix for output in outputs):
            run()
    app(port)


# Predict ---------------------------------------------------------------------------------------------


def predict(file: Path, output: Path | None, json_path: Path | None, device: torch.device) -> None:
    if not file.is_file():
        raise StepError(f"There is no file at {file}.")
    if output is not None and output.suffix.lower() != ".vtp":
        raise StepError(f"--output writes a .vtp file; give a name ending in .vtp, such as {output.with_suffix('.vtp').name}.")
    if not CHECKPOINT.is_file():
        raise StepError("There is no trained model at artifacts/ensemble.pt. Train one with: uv run surrogate train")
    ensemble = SurrogateEnsemble.load(CHECKPOINT, device=device)
    print(f"Predicting {file} on {describe_device(device)}", flush=True)
    result = run_vtp_inference(file, ensemble)
    print_prediction(result, ensemble)
    if output is not None:
        write_prediction_vtp(file, result, output)
        print(f"Wrote {output} with cell arrays CpPredicted and CpSpread")
    if json_path is not None:
        json_path.write_text(json.dumps(prediction_summary(result), indent=1) + "\n")
        print(f"Wrote {json_path}")


def print_prediction(result: InferenceResult, ensemble: SurrogateEnsemble) -> None:
    bounds = ensemble.training_envelope["descriptor_bounds"]
    sizes = []
    for name, value in zip(("length", "width", "height"), result.dimensions, strict=True):
        low, high = bounds[name]["min"], bounds[name]["max"]
        where = "inside" if low <= value <= high else "outside"
        sizes.append(f"{name} {value:.2f} m ({where} {low:.2f}–{high:.2f})")
    forces, coverage, cfd = result.forces, result.coverage, result.ground_truth_forces

    def force(name: str) -> str:
        predicted = f"{forces[name]:.4f}"
        return predicted if cfd is None else f"{predicted}   CFD {cfd[name]:.4f}, error {forces[name] - cfd[name]:+.4f}"

    lines = [
        ("Cells", f"{result.n_cells:,}, {result.total_area:.2f} m², frontal area {result.frontal_area:.3f} m²"),
        ("Size", ", ".join(sizes)),
        ("Coverage", coverage["summary"]),
        *[("", reason) for reason in coverage["reasons"]],
        ("Drag Cd,p", force("cd_p")),
        ("Lift Cl,p", force("cl_p")),
        ("Each network", "Cd,p " + "  ".join(f"{cd:.4f}" for cd in forces["cd_p_per_fold"])),
        ("Spread", f"mean σ(Cp) {coverage['mean_cp_std']:.3f}, 95th percentile {coverage['p95_cp_std']:.3f}"),
    ]
    if (agreement := result.ground_truth_metrics) is not None:
        lines.append(("CFD agreement", f"R² {agreement['r2']:.4f}, MAE {agreement['mae']:.4f}, RMSE {agreement['rmse']:.4f} over every cell, by area"))
    else:
        lines.append(("CFD", "the file has no CpMeanTrim to compare with"))
    lines.append(("Time", f"{result.timings_sec['total']:.0f} s"))
    width = max(len(label) for label, _ in lines)
    for label, text in lines:
        print(f"{label:<{width}}  {text}")


def prediction_summary(result: InferenceResult) -> dict:
    """Everything in `result` but the per-cell arrays, which the .vtp carries."""
    return {field.name: getattr(result, field.name) for field in dataclasses.fields(result) if field.name not in ("cp_mean", "cp_std")}


# Command line ----------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="surrogate",
        description="Set up, train, check and use the DrivAerML surface-pressure model. Run a command with -h for its options.",
    )
    commands = parser.add_subparsers(title="commands", dest="command", metavar="COMMAND")

    def command(name: str, summary: str) -> argparse.ArgumentParser:
        return commands.add_parser(name, help=summary, description=summary[0].upper() + summary[1:] + ".")

    command("status", "what this copy holds, and the command for each gap").set_defaults(run=lambda args: print_status())

    sub = command("app", f"serve the tool at http://127.0.0.1:{DEFAULT_PORT}/")
    sub.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port to listen on (default {DEFAULT_PORT})")
    sub.set_defaults(run=lambda args: app(args.port))

    sub = command("predict", "predict Cp at every cell of a .vtp surface file")
    sub.add_argument("file", type=Path, help="a DrivAerML boundary_<run>.vtp or any closed car surface")
    sub.add_argument("--output", type=Path, help="write the file with cell arrays CpPredicted and CpSpread to this .vtp")
    sub.add_argument("--json", type=Path, help="write the summary as JSON to this file")
    add_device(sub)
    sub.set_defaults(run=lambda args: predict(args.file, args.output, args.json, chosen_device(args.device)))

    sub = command("fetch", "download DrivAerML surface files into data/raw")
    sub.add_argument(
        "runs", nargs="?", choices=RUN_SETS, default="all",
        help="all twelve (default, 7.8 GB), the ten training cars (6.5 GB) or the two holdout cars (1.3 GB); "
        "files already here are skipped and an interrupted download continues",
    )
    sub.set_defaults(run=lambda args: fetch(args.runs))

    sub = command("preprocess", "sample the twelve cars and compute their inputs into data/processed")
    sub.add_argument("--samples", type=int, default=DEFAULT_SAMPLES, help=f"cells drawn per car, by area (default {DEFAULT_SAMPLES:,})")
    sub.set_defaults(run=lambda args: preprocess(args.samples))

    sub = command("train", "train the five networks and the ablations into artifacts")
    sub.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help=f"passes over the training cells (default {DEFAULT_EPOCHS})")
    sub.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"random seed (default {DEFAULT_SEED})")
    sub.add_argument("--skip-ablations", action="store_true", help="train only the shipped networks")
    add_device(sub)
    sub.set_defaults(run=lambda args: train(args.epochs, args.seed, not args.skip_ablations, chosen_device(args.device)))

    command("export", "write web/model from the trained model").set_defaults(run=lambda args: export())

    sub = command("examples", "write web/examples with the browser engine in headless Chrome")
    sub.add_argument("--development", action="store_true", help="also write the development examples into local/dev/examples")
    sub.set_defaults(run=lambda args: examples(args.development))

    sub = command("reproduce", "run each step that is not current, then serve the tool")
    sub.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port to serve the tool on (default {DEFAULT_PORT})")
    sub.set_defaults(run=lambda args: reproduce(args.port))
    return parser


def add_device(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", choices=DEVICES, help="where the networks run (default: the GPU when there is one, else cpu)")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        print()
        print_status()
        return
    try:
        args.run(args)
    except StepError as error:
        parser.exit(1, f"{error}\n")
