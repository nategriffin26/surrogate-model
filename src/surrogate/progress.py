"""What a long command shows while it works: how far along it is, the time since it started, an estimate of
the time left, and the step it is on, in one status line.

     12%   1:58 elapsed,  14 min left  Spread factor: network 9 of 20, runs 80, 98, 102 and 208 held out

On a terminal the status line is rewritten in place twice a second. When the output goes to a file or a
pipe (a log, CI, pytest), the same line is printed whole when a step starts and every `PIPED_SECONDS`
within it, with no carriage return or escape code. Lines printed with `note` (results, files written) go
above the status line in both cases.

A command lists its steps up front, each with the seconds it took on `MEASURED_ON`. The share done is those
seconds summed over the work finished, and the time left is the seconds still to come at this run's pace,
shown from 1% done, before which the pace says little.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from dataclasses import dataclass, replace
from types import TracebackType
from typing import Literal, TextIO, assert_never

MEASURED_ON = "an Apple M5 Pro"
PIPED_SECONDS = 30
REDRAW_SECONDS = 0.5

# What a step counts. A thing is shown as the one in hand ("network 3 of 20"), bytes and cells as the amount
# done ("1.2 of 7.8 GB"), and a stage, one piece of work, by the step's name alone.
Unit = Literal["network", "model", "car", "bytes", "cells", "stage"]
THINGS: tuple[Unit, ...] = ("network", "model", "car")


@dataclass(frozen=True)
class Step:
    """`count` of `unit`, which took `seconds` in all on MEASURED_ON."""

    name: str
    count: float
    unit: Unit
    seconds: float


def clock(seconds: float) -> str:
    """Elapsed time as m:ss, or h:mm:ss from an hour."""
    hours, rest = divmod(int(seconds), 3600)
    return f"{hours}:{rest // 60:02d}:{rest % 60:02d}" if hours else f"{rest // 60}:{rest % 60:02d}"


def duration(seconds: float) -> str:
    """A time to come, rounded as a person would say it."""
    if seconds >= 90:
        return f"{round(seconds / 60)} min"
    if seconds >= 10:
        return f"{round(seconds / 10) * 10} s"
    return "<10 s"


def _whole(done: float) -> int:
    """The things finished, with the float sum of a thing's shares (thirty epochs of 1/30) counting as one."""
    return int(round(done, 6))


def _position(done: float, step: Step) -> str:
    """Where a step is: the thing in hand, or the amount done of the whole."""
    match step.unit:
        case "network" | "model" | "car":
            return f"{step.unit} {min(_whole(done) + 1, int(step.count))} of {step.count:.0f}"
        case "bytes":
            if step.count >= 1e9:
                return f"{done / 1e9:.1f} of {step.count / 1e9:.1f} GB"
            return f"{done / 1e6:.0f} of {step.count / 1e6:.0f} MB"
        case "cells":
            if step.count >= 1e6:
                return f"{done / 1e6:.1f}M of {step.count / 1e6:.1f}M cells"
            return f"{done:,.0f} of {step.count:,.0f} cells"
        case "stage":
            return ""
        case _:
            assert_never(step.unit)


class Progress:
    """The status line of one command, used as a context manager around the work it describes.

    With `out` None it prints nothing, for code that runs the pipeline without a command around it."""

    def __init__(self, steps: list[Step], out: TextIO | None) -> None:
        self.steps = {step.name: step for step in steps}
        self.out = out
        self.live = out is not None and out.isatty() and os.environ.get("TERM") != "dumb"
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ticker: threading.Thread | None = None
        self._started = time.monotonic()
        self._first_step: float | None = None
        self._last_counted = 0.0
        self._finished_seconds = 0.0
        self._step: Step | None = None
        self._done = 0.0
        self._detail = ""
        self._last_printed = 0.0

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._started

    def __enter__(self) -> Progress:
        self._started = time.monotonic()
        if self.out is not None:
            self._ticker = threading.Thread(target=self._tick, daemon=True)
            self._ticker.start()
        return self

    def __exit__(
        self, kind: type[BaseException] | None, error: BaseException | None, trace: TracebackType | None
    ) -> None:
        self._stop.set()
        if self._ticker is not None:
            self._ticker.join()
        with self._lock:
            if self.live and self._step is not None:
                # A finished command's status line is cleared; a stopped one's stays to show where it stopped.
                self._write("\r\x1b[K" if kind is None else "\n")

    def step(self, name: str, *, count: float | None = None, detail: str = "") -> None:
        """Start the step `name`; `count` gives its size when that is known only now (the cells of a file)."""
        if self.out is None:
            return
        with self._lock:
            if self._step is None:
                self._first_step = time.monotonic()
            else:
                self._finished_seconds += self._step.seconds
            self._last_counted = time.monotonic()
            if count is not None:
                self.steps[name] = replace(self.steps[name], count=count)
            self._step, self._done, self._detail = self.steps[name], 0.0, detail
            self._show(force=True)

    def advance(self, units: float = 1.0, detail: str | None = None) -> None:
        """Count `units` more of the current step done, and say what it is on now when `detail` is given. A
        thing finished takes its detail with it, so the next one is not shown with the last one's."""
        if self.out is None:
            return
        with self._lock:
            finished = _whole(self._done)
            self._done += units
            if units > 0:
                self._last_counted = time.monotonic()
            if self._step is not None and self._step.unit in THINGS and _whole(self._done) > finished:
                self._detail = ""
            if detail is not None:
                self._detail = detail
            if not self.live:
                self._show(force=False)

    def describe(self, detail: str) -> None:
        """Say what the current step is on now."""
        self.advance(0.0, detail)

    def note(self, text: str) -> None:
        """Print a line that stays: a result, or a file written."""
        if self.out is None:
            return
        with self._lock:
            if self.live and self._step is not None:
                self._write(f"\r\x1b[K{text}\n")
                self._redraw()
            else:
                self._write(f"{text}\n")

    def status(self) -> str:
        """The status line as it reads now."""
        step = self._step
        if step is None:
            return ""
        total = sum(each.seconds for each in self.steps.values())
        done_in_step = step.seconds * min(self._done / step.count, 1.0)
        done = self._finished_seconds + done_in_step
        now = time.monotonic()
        if done >= 0.01 * total and self._first_step is not None:
            # This run's seconds per measured second, up to the last work counted. The step in hand counts down
            # from there, and when it runs over, what is left is the steps after it.
            pace = (self._last_counted - self._first_step) / done
            rest_of_step = step.seconds - done_in_step
            in_hand = max(rest_of_step * pace - (now - self._last_counted), 0.0)
            estimate = f", {duration((total - done - rest_of_step) * pace + in_hand):>6} left"
        else:
            estimate = " " * len(", 00 min left")
        what = ", ".join(part for part in (_position(self._done, step), self._detail) if part)
        heading = f"{step.name}: {what}" if what else step.name
        return f"{100 * done / total:3.0f}%  {clock(now - self._started):>5} elapsed{estimate}  {heading}"

    def _show(self, *, force: bool) -> None:
        if self.live:
            self._redraw()
        elif force or time.monotonic() - self._last_printed >= PIPED_SECONDS:
            self._write(f"{self.status()}\n")
            self._last_printed = time.monotonic()

    def _redraw(self) -> None:
        width = shutil.get_terminal_size().columns - 1
        self._write(f"\r{self.status()[:width]}\x1b[K")

    def _tick(self) -> None:
        while not self._stop.wait(REDRAW_SECONDS):
            with self._lock:
                if self._step is not None:
                    self._show(force=False)

    def _write(self, text: str) -> None:
        assert self.out is not None
        self.out.write(text)
        self.out.flush()


QUIET = Progress([], None)
