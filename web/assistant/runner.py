"""Runs the assistant's code in the kernel: one namespace kept for the conversation, with what the code prints, the
tables and figures it shows sent to the page as they appear."""

from __future__ import annotations

import io
import traceback

import js
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler
from matplotlib.figure import Figure
from pyodide.code import eval_code_async
from pyodide.ffi import to_js

import pressure
from _page import emit

CELL = "<cell>"
TABLE_ROWS = 200

# Figures in the tool's light theme (web/app/theme.css): hairline axes, quiet grid, its blues and accents.
plt.rcParams.update(
    {
        "figure.figsize": (6.4, 4.0),
        "figure.dpi": 100,
        "savefig.dpi": 144,
        "savefig.bbox": "tight",
        "font.size": 10,
        "text.color": "#1a2b3a",
        "axes.labelcolor": "#1a2b3a",
        "axes.edgecolor": "#cddce4",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.titlesize": 11,
        "axes.titlelocation": "left",
        "axes.prop_cycle": cycler(color=["#0b6fbf", "#a35b1a", "#197b80", "#a83258", "#9a55e8", "#237f50"]),
        "grid.color": "#e3ecf1",
        "xtick.color": "#416473",
        "ytick.color": "#416473",
        "lines.linewidth": 1.6,
        "legend.frameon": False,
    }
)

namespace: dict = {"__name__": "__main__", "pressure": pressure}


def display(*objects) -> None:
    """Show each object in the conversation: a matplotlib figure as a picture, a pandas table as a table, anything
    else as its text."""
    for obj in objects:
        if isinstance(obj, Figure):
            _figure(obj)
        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            _table(obj.to_frame() if isinstance(obj, pd.Series) else obj)
        else:
            _emit({"kind": "text", "text": repr(obj) + "\n"})


def _show(*args, **kwargs) -> None:
    for number in plt.get_fignums():
        _figure(plt.figure(number))


namespace["display"] = display
plt.show = _show


async def run(code: str) -> str | None:
    """Run `code` in the kept namespace; show its last expression's value and the figures it left open. Returns the
    traceback of an error, from the first line of the code on, or None."""
    try:
        value = await eval_code_async(code, globals=namespace, filename=CELL)
        if value is not None:
            display(value)
        return None
    except BaseException as error:  # noqa: BLE001 - every error in the code is the assistant's to read
        return _traceback(error)
    finally:
        _show()


def set_cases(message: str) -> None:
    pressure._set_cases(message)


def _traceback(error: BaseException) -> str:
    frames = traceback.extract_tb(error.__traceback__)
    first = next((i for i, frame in enumerate(frames) if frame.filename == CELL), len(frames))
    return "".join(["Traceback (most recent call last):\n", *traceback.format_list(frames[first:]), *traceback.format_exception_only(error)])


def _figure(figure: Figure) -> None:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    plt.close(figure)
    _emit({"kind": "figure", "png": to_js(buffer.getvalue()), "title": _describe(figure)})


def _describe(figure: Figure) -> str:
    """What the figure shows, in its own words: its titles and axis labels."""
    parts = [figure.get_suptitle()] if figure.get_suptitle() else []
    for axes in figure.axes:
        labels = [axes.get_title(), axes.get_xlabel() and f"x: {axes.get_xlabel()}", axes.get_ylabel() and f"y: {axes.get_ylabel()}"]
        parts.append(", ".join(label for label in labels if label))
    return "; ".join(part for part in parts if part) or "untitled"


def _table(frame: pd.DataFrame) -> None:
    shown = frame.head(TABLE_ROWS)
    index = not isinstance(frame.index, pd.RangeIndex)
    columns = ([str(frame.index.name or "")] if index else []) + [str(column) for column in frame.columns]
    rows = [([_cell(label)] if index else []) + [_cell(value) for value in row] for label, row in zip(shown.index, shown.itertuples(index=False))]
    text = frame.to_string(max_rows=40, max_cols=12, index=index)
    _emit({"kind": "table", "columns": columns, "rows": rows, "total": len(frame), "text": text})


def _cell(value) -> str:
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def _emit(item: dict) -> None:
    emit(to_js(item, dict_converter=js.Object.fromEntries))
