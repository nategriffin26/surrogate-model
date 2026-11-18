"""The DrivAerML surface files the pipeline reads: the runs it uses, where their files live under
`data/raw/`, and fetching them from the dataset on Hugging Face.

Each run is one file, `boundary_<run>.vtp`. A download goes to `boundary_<run>.vtp.part` and is renamed
when complete, so a file under its own name is always whole, and fetching again continues a part from where
it stopped.
"""

from __future__ import annotations

import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

DATASET_URL = "https://huggingface.co/datasets/neashton/drivaerml/resolve/main"
ALLOWED_TRAIN_RUNS: tuple[int, ...] = (80, 98, 102, 208, 213, 281, 397, 425, 445, 474)
HOLDOUT_RUNS: tuple[int, ...] = (1, 500)
CHUNK_BYTES = 1 << 20


def surface_path(raw_dir: Path, run: int) -> Path:
    """Where run `run`'s surface file lives: `train/` for the ten training runs, `holdout/` for any other."""
    split = "train" if run in ALLOWED_TRAIN_RUNS else "holdout"
    return raw_dir / split / f"run_{run}" / f"boundary_{run}.vtp"


@dataclass(frozen=True)
class RemoteSurface:
    """One run's surface file in the dataset, and where it goes on this machine."""

    run: int
    url: str
    path: Path
    size: int  # bytes, as the dataset states it

    @property
    def part(self) -> Path:
        return self.path.with_name(self.path.name + ".part")

    @property
    def present(self) -> bool:
        """Whether the whole file is already here."""
        return self.path.is_file() and self.path.stat().st_size == self.size


def remote_surfaces(runs: tuple[int, ...], raw_dir: Path, base_url: str = DATASET_URL) -> list[RemoteSurface]:
    """Each run's surface file with its size, asked of the dataset without downloading it."""
    surfaces = []
    for run in runs:
        url = f"{base_url}/run_{run}/boundary_{run}.vtp"
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as response:
            size = int(response.headers["Content-Length"])
        surfaces.append(RemoteSurface(run, url, surface_path(raw_dir, run), size))
    return surfaces


def fetch_surface(surface: RemoteSurface, on_progress: Callable[[int], None]) -> None:
    """Download `surface` to its path, continuing a part left by an earlier attempt. `on_progress` gets
    the bytes held so far after each chunk."""
    surface.path.parent.mkdir(parents=True, exist_ok=True)
    held = surface.part.stat().st_size if surface.part.is_file() else 0
    if held == surface.size:
        surface.part.replace(surface.path)
        return
    if held > surface.size:  # not a part of this file
        held = 0
    request = urllib.request.Request(surface.url, headers={"Range": f"bytes={held}-"} if held else {})
    with urllib.request.urlopen(request) as response:
        # 206 continues the part; a server that ignores the range sends the whole file with 200.
        if response.status != 206:
            held = 0
        with surface.part.open("ab" if held else "wb") as out:
            while chunk := response.read(CHUNK_BYTES):
                out.write(chunk)
                held += len(chunk)
                on_progress(held)
    if held != surface.size:
        raise OSError(f"{surface.url} ended after {held:,} of {surface.size:,} bytes")
    surface.part.replace(surface.path)
