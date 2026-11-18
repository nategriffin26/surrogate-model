"""The tool from Python: the open cases as numpy arrays, every operation of the tool, and the model's prediction on
new geometry, so a study is a loop of deform, predict, measure, collect.

The working directory is the tool's workspace. Files the engineer added are here to read (they cannot be changed in
place: write a new file instead), and every file written here appears in the workspace. Calls that reach the tool are
coroutines; await them.

    import pressure

    car = pressure.active                    # the case on screen; pressure.cases[2] by id
    w = await car.working()                  # its working mesh: one cell per triangle, about 600,000
    up = w.normals[:, 2] > 0.9               # cells facing up
    mean_cp = (w.cp[up] * w.areas[up]).sum() / w.areas[up].sum()

    points = w.points.copy()
    points[:, 0] = points[:, 0].min() + 1.05 * (points[:, 0] - points[:, 0].min())
    longer = await pressure.predict(w.with_points(points), name="5 % longer")
    print(longer.cx, longer.length_m)

    await pressure.run("list_cases")         # any operation of the tool, with the arguments the assistant gives it
    await pressure.write_vtp("longer.vtp", longer.cells)

Frame: metres, x downstream, y to the car's right seen from the driver's seat, z up. Cp is the pressure coefficient,
`spread` its uncertainty σ(Cp) from the model's five networks. Pressure only: there is no wall shear, so forces from
these fields are pressure forces.
"""

from __future__ import annotations

import json
from functools import cached_property

import js
import numpy as np
from pyodide.ffi import JsProxy, to_js

from _page import request

__all__ = ["Case", "Prediction", "Surface", "ToolError", "active", "cases", "predict", "run", "write_vtp"]


class ToolError(Exception):
    """The tool refused a request: the message says what was wrong and what would work."""


class Surface:
    """A surface as polygons: `points` (P, 3) and each cell's corners in `connectivity`, cell c owning
    `connectivity[offsets[c]:offsets[c + 1]]`. Per cell: `centers` (C, 3), outward unit `normals` (C, 3), `areas` (C,)
    in m², and when predicted `cp` (C,) and `spread` (C,); `reference` (C,), the case's reference field (CFD read from the
    file or attached to it; `Case.reference` says which), else None. Arrays are
    numpy, made from the tool's data on first use."""

    def __init__(self, points, connectivity, offsets, *, centers=None, normals=None, areas=None, cp=None, spread=None, reference=None):
        self._sources = dict(
            points=points, connectivity=connectivity, offsets=offsets, centers=centers, normals=normals, areas=areas, cp=cp, spread=spread, reference=reference
        )

    @classmethod
    def from_triangles(cls, points, triangles) -> Surface:
        """A surface of triangles: `points` (P, 3) and `triangles` (T, 3) point indices, wound counter-clockwise seen from outside."""
        triangles = np.asarray(triangles, dtype=np.uint32)
        return cls(np.asarray(points, dtype=np.float32), triangles.ravel(), np.arange(0, 3 * len(triangles) + 1, 3, dtype=np.uint32))

    def with_points(self, points) -> Surface:
        """The same cells on moved points: a deformed copy, without fields until it is predicted."""
        points = np.asarray(points, dtype=np.float32)
        if points.shape != self.points.shape:
            raise ValueError(f"Expected points of shape {self.points.shape}, got {points.shape}.")
        return Surface(points, self.connectivity, self.offsets)

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __repr__(self) -> str:
        fields = [name for name in ("cp", "spread", "reference") if self._sources[name] is not None]
        return f"Surface({len(self):,} cells, {len(self.points):,} points{', ' + ', '.join(fields) if fields else ''})"

    @cached_property
    def points(self) -> np.ndarray:
        return _numpy(self._sources["points"], np.float32, 3)

    @cached_property
    def connectivity(self) -> np.ndarray:
        return _numpy(self._sources["connectivity"], np.uint32)

    @cached_property
    def offsets(self) -> np.ndarray:
        return _numpy(self._sources["offsets"], np.uint32)

    @cached_property
    def triangles(self) -> np.ndarray:
        """(C, 3) corner indices, when every cell is a triangle."""
        if not np.all(np.diff(self.offsets) == 3):
            raise ValueError("Not every cell is a triangle; use connectivity and offsets.")
        return self.connectivity.reshape(-1, 3)

    @cached_property
    def centers(self) -> np.ndarray:
        return _numpy(self._sources["centers"], np.float32, 3) if self._sources["centers"] is not None else self._geometry[0]

    @cached_property
    def normals(self) -> np.ndarray:
        return _numpy(self._sources["normals"], np.float32, 3) if self._sources["normals"] is not None else self._geometry[1]

    @cached_property
    def areas(self) -> np.ndarray:
        return _numpy(self._sources["areas"], np.float64) if self._sources["areas"] is not None else self._geometry[2]

    @cached_property
    def cp(self) -> np.ndarray | None:
        return _numpy(self._sources["cp"], np.float32)

    @cached_property
    def spread(self) -> np.ndarray | None:
        return _numpy(self._sources["spread"], np.float32)

    @cached_property
    def reference(self) -> np.ndarray | None:
        return _numpy(self._sources["reference"], np.float32)

    @cached_property
    def _geometry(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Centres, unit normals and areas of the cells, each polygon split into a fan of triangles from its first corner."""
        counts = np.diff(self.offsets).astype(np.int64)
        cell = np.repeat(np.arange(len(counts)), counts - 2)
        first = np.repeat(self.offsets[:-1].astype(np.int64), counts - 2)
        step = np.arange(len(cell)) - np.repeat(np.cumsum(counts - 2) - (counts - 2), counts - 2)
        p = self.points.astype(np.float64)
        a, b, c = (p[self.connectivity[first + k]] for k in (np.zeros_like(step), step + 1, step + 2))
        cross = np.cross(b - a, c - a)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        areas = np.bincount(cell, area, len(counts))
        vector = np.stack([np.bincount(cell, cross[:, k], len(counts)) for k in range(3)], axis=1)
        normals = vector / np.maximum(np.linalg.norm(vector, axis=1, keepdims=True), 1e-30)
        weighted = np.stack([np.bincount(cell, area * (a + b + c)[:, k] / 3, len(counts)) for k in range(3)], axis=1)
        centers = weighted / np.maximum(areas, 1e-30)[:, None]
        return centers.astype(np.float32), normals.astype(np.float32), areas


class Case:
    """An open case of the tool: `id`, `name`, `kind` ('file', 'example' or 'variant'), `status` ('queued', 'analysing',
    'loading', 'ready' or 'failed'), `path`, its geometry file in the workspace when it has one, and `reference`, what
    its surfaces' `reference` field is: `{"file": ..., "array": ..., "mapping": ...}` (the file it came from, its cell
    array, CFD when that is "CpMeanTrim", and how it reached these cells), or None."""

    def __init__(self, info: dict) -> None:
        self._update(info)
        self._working: tuple[int, Surface] | None = None

    def _update(self, info: dict) -> None:
        self.id: int = info["id"]
        self.name: str = info["name"]
        self.kind: str = info["kind"]
        self.status: str = info["status"]
        self.path: str | None = info.get("path")
        self.reference: dict | None = info.get("reference")
        self._version: int = info["version"]

    def __repr__(self) -> str:
        return f"Case({self.id}, {self.name!r}, {self.kind}, {self.status})"

    async def working(self) -> Surface:
        """The working mesh the tool's own measurements run on: a faithful simplification of the file (bundled examples
        have only this), one cell per triangle, with Cp and spread per cell."""
        if self._working is None or self._working[0] != self._version:
            self._working = (self._version, _surface(await _ask("working", {"case_id": self.id})))
        return self._working[1]

    async def file(self) -> Surface | None:
        """The file's own polygons with the prediction at each, every one of them (8.8 million on a DrivAerML car): what
        the engine summed for the drag. None when the engine no longer holds the file (it holds the last one dropped)
        or has not yet predicted every polygon. Large: ask for it only when the working mesh will not do."""
        cells = await _ask("file", {"case_id": self.id})
        return None if cells is None else _surface(cells)


class Prediction:
    """The model's prediction on a surface: `cells`, the surface with `cp` and `spread` per cell, and the numbers the
    tool reports for an open case (`list_cases`), under the same names: `cx`, `cy` and `cz`, the pressure force
    coefficients on `reference_area_m2` (`cx` is pressure drag); `cx_by_network`; `area_m2` and `mean_cp`; `cp_min` and
    `cp_max`, each `{"value", "center_m", "normal"}`; `length_m`, `width_m`, `height_m` and `outside_training`. `summary`
    holds them all as a dict, for a table row. `seconds` is the time the prediction took."""

    def __init__(self, name: str, summary: dict, cells: JsProxy, seconds: float) -> None:
        self.name = name
        self.summary = summary
        self.cells = _surface(cells)
        self.seconds = seconds

    def __getattr__(self, key: str):
        if key in self.__dict__.get("summary", {}):
            return self.summary[key]
        raise AttributeError(f"A prediction has no {key!r}; it has cells, seconds and {', '.join(self.summary)}.")

    def __dir__(self) -> list[str]:
        return [*super().__dir__(), *self.summary]

    def __repr__(self) -> str:
        return f"Prediction({self.name!r}: cx {self.cx:.4f}, {len(self.cells):,} cells, {self.seconds:.1f} s)"


cases: dict[int, Case] = {}
"""The open cases by id, in the order of the tool's tabs."""

active: Case | None = None
"""The case on screen."""


async def run(name: str, **arguments) -> dict:
    """Run one of the tool's operations, as the engineer's controls and the assistant do, and return its result. What
    it changes shows on screen; operations that open, write, download or delete ask the engineer first."""
    return _json(await _ask("operation", {"name": name, "args": json.loads(json.dumps(arguments, default=_plain))}))


async def predict(surface: Surface, name: str = "prediction") -> Prediction:
    """Predict Cp on every cell of `surface` with the model, as the tool does for a file, on the cells as given (the
    working mesh's resolution is what a study needs, and coarser starves the model's inputs). Nothing opens in the tool."""
    result = await _ask("predict", {"name": name, **_geometry(surface)})
    return Prediction(name, _json(result.summary), result.cells, float(result.seconds))


async def write_vtp(path: str, surface: Surface, cell_data: dict | None = None, point_data: dict | None = None) -> str:
    """Write `surface` as a VTK PolyData file at `path` in the workspace, with arrays per cell and per point (one value
    or one row each). By default the cells carry the surface's prediction as `CpPredicted` and `CpSpread`, the names
    the tool's own export uses. Returns the path."""
    if cell_data is None:
        cell_data = {label: values for label, values in (("CpPredicted", surface.cp), ("CpSpread", surface.spread)) if values is not None}
    arrays = {
        "cellData": {name: _typed(np.asarray(values, dtype=np.float32)) for name, values in cell_data.items()},
        "pointData": {name: _typed(np.asarray(values, dtype=np.float32)) for name, values in (point_data or {}).items()},
    }
    await _ask("write_vtp", {"path": path, **_geometry(surface), **arrays})
    return path


def _set_cases(message: str) -> None:
    """The page's open cases, `{"cases": [...], "active": id}` as JSON, sent with every message to the kernel."""
    global active
    sent = json.loads(message)
    previous = dict(cases)
    cases.clear()
    for info in sent["cases"]:
        case = previous.get(info["id"]) or Case(info)
        case._update(info)
        cases[case.id] = case
    active = cases.get(sent["active"])


async def _ask(kind: str, payload: dict):
    reply = await request(kind, to_js(payload, dict_converter=js.Object.fromEntries))
    if reply.error:
        raise ToolError(reply.error)
    return reply.value


def _geometry(surface: Surface) -> dict:
    return {
        "points": _typed(np.ascontiguousarray(surface.points, dtype=np.float32)),
        "connectivity": _typed(np.ascontiguousarray(surface.connectivity, dtype=np.uint32)),
        "offsets": _typed(np.ascontiguousarray(surface.offsets, dtype=np.uint32)),
    }


_TYPED = {np.dtype(np.float32): "Float32Array", np.dtype(np.float64): "Float64Array", np.dtype(np.uint32): "Uint32Array", np.dtype(np.int32): "Int32Array"}


def _typed(values: np.ndarray) -> JsProxy:
    """A JavaScript typed array holding its own copy of `values`, flattened, ready to be sent to the page."""
    values = np.ascontiguousarray(values)
    array = getattr(js, _TYPED[values.dtype]).new(values.size)
    array.assign(values.ravel())
    return array


def _numpy(source, dtype, width: int | None = None) -> np.ndarray | None:
    if source is None or isinstance(source, np.ndarray):
        return source
    values = np.empty(source.length, dtype=dtype)
    source.assign_to(values)
    return values.reshape(-1, width) if width else values


def _surface(cells: JsProxy) -> Surface:
    return Surface(
        cells.points, cells.connectivity, cells.offsets,
        centers=cells.centers, normals=cells.normals, areas=cells.areas, cp=cells.cp, spread=cells.sigma, reference=cells.cfd,
    )


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} cannot be sent to the tool.")


def _json(value: JsProxy):
    """Plain data from the page (numbers, strings, lists and objects) as Python values."""
    return json.loads(js.JSON.stringify(value))
