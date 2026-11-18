"""The tool from Python: the open cases as numpy arrays, every operation of the tool, and the model's prediction on
new geometry, so a study is a loop of deform, predict, measure and collect; and drawing on the car to show what was
found.

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

Drawing on the car, like plotting: every array goes to the page whole, never as JSON, so thousands of marks are one
call. A case is coloured by a field of its own or painted; marks are drawn through `pressure.view` (the case on
screen) or `case.view`. Each call draws into a named layer (`layer=`, made if new, added to if it exists), which the
engineer can show, hide or remove from its chip in the strip above the car.

    await car.show(w.spread / np.maximum(np.abs(w.cp), 0.05), name="Spread / |Cp|", cmap="viridis", range=(0, 1))
    await car.paint(w.cp < -1, "orange", layer="Cp below -1")          # a mask per cell, or colours per cell
    view = pressure.view
    await view.label(w.centers[w.cp.argmin()], "Lowest Cp", layer="Extremes")
    await view.arrows(w.centers[up], -(w.cp * w.areas)[up, None] * w.normals[up], layer="Force")
    await view.line([[0, 0, 1.6], [4, 0, 1.6]], color="rose")
    await view.box([-1, -1, -0.4], [0, 1, 0.6])
    await view.clear()                       # every layer the assistant drew; view.clear("Force") one of them

Points are (N, 3) arrays, or one [x, y, z], in metres. `color` is a name (ink, blue, orange, teal, rose, violet,
green, grey, white) or "#rrggbb", and sets the layer's colour; for points and arrows it may also be one colour per
item, (N, 3) or (N, 4) floats from 0 to 1 as a matplotlib colormap gives them. `width` is a line width in px.

Frame: metres, x downstream, y to the car's right seen from the driver's seat, z up. Cp is the pressure coefficient,
`spread` its uncertainty σ(Cp) from the model's five networks. Pressure only: there is no wall shear, so forces from
these fields are pressure forces.
"""

from __future__ import annotations

import json
from functools import cached_property

import js
import numpy as np
from _page import request
from pyodide.ffi import JsProxy, to_js

__all__ = [
    "Case",
    "Prediction",
    "Surface",
    "ToolError",
    "View",
    "active",
    "cases",
    "predict",
    "run",
    "view",
    "write_vtp",
]


class ToolError(Exception):
    """The tool refused a request: the message says what was wrong and what would work."""


class Surface:
    """A surface as polygons: `points` (P, 3) and each cell's corners in `connectivity`, cell c owning
    `connectivity[offsets[c]:offsets[c + 1]]`. Per cell: `centers` (C, 3), outward unit `normals` (C, 3), `areas` (C,)
    in m², and when predicted `cp` (C,) and `spread` (C,); `reference` (C,), the case's reference field (CFD read from
    the file or attached to it; `Case.reference` says which), else None. Arrays are numpy, made from the tool's data
    on first use."""

    def __init__(
        self,
        points,
        connectivity,
        offsets,
        *,
        centers=None,
        normals=None,
        areas=None,
        cp=None,
        spread=None,
        reference=None,
    ):
        self._sources = {
            "points": points,
            "connectivity": connectivity,
            "offsets": offsets,
            "centers": centers,
            "normals": normals,
            "areas": areas,
            "cp": cp,
            "spread": spread,
            "reference": reference,
        }

    @classmethod
    def from_triangles(cls, points, triangles) -> Surface:
        """A surface of triangles: `points` (P, 3) and `triangles` (T, 3) point indices, wound counter-clockwise seen
        from outside."""
        triangles = np.asarray(triangles, dtype=np.uint32)
        offsets = np.arange(0, 3 * len(triangles) + 1, 3, dtype=np.uint32)
        return cls(np.asarray(points, dtype=np.float32), triangles.ravel(), offsets)

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
        if self._sources["centers"] is None:
            return self._geometry[0]
        return _numpy(self._sources["centers"], np.float32, 3)

    @cached_property
    def normals(self) -> np.ndarray:
        if self._sources["normals"] is None:
            return self._geometry[1]
        return _numpy(self._sources["normals"], np.float32, 3)

    @cached_property
    def areas(self) -> np.ndarray:
        if self._sources["areas"] is None:
            return self._geometry[2]
        return _numpy(self._sources["areas"], np.float64)

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
        """Centres, unit normals and areas of the cells, each polygon split into a fan of triangles from its first
        corner."""
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
    array, CFD when that is "CpMeanTrim", and how it reached these cells), or None; and for a variant, `variant_of`,
    the id of the car it was made from (else None)."""

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
        self.variant_of: int | None = info.get("variant_of")
        self._version: int = info["version"]

    def __repr__(self) -> str:
        return f"Case({self.id}, {self.name!r}, {self.kind}, {self.status})"

    async def working(self) -> Surface:
        """The working mesh the tool's own measurements run on: a faithful simplification of the file (bundled examples
        have only this), one cell per triangle, with Cp and spread per cell."""
        if self._working is None or self._working[0] != self._version:
            self._working = (self._version, _surface(await _ask("working", {"case_id": self.id})))
        return self._working[1]

    async def unchanged(self) -> Surface | None:
        """For a variant: the car it was made from, predicted unchanged on this variant's own cells, so
        `(await v.working()).cp - (await v.unchanged()).cp` is the change alone, cell by cell. None for a case that is
        not a variant."""
        if self.variant_of is None:
            return None
        cells = await _ask("unchanged", {"case_id": self.id})
        return None if cells is None else _surface(cells)

    async def file(self) -> Surface | None:
        """The file's own polygons with the prediction at each, every one of them (8.8 million on a DrivAerML car): what
        the engine summed for the drag. None when the engine no longer holds the file (it holds the last one dropped)
        or has not yet predicted every polygon. Large: ask for it only when the working mesh will not do."""
        cells = await _ask("file", {"case_id": self.id})
        return None if cells is None else _surface(cells)

    @property
    def view(self) -> View:
        """Marks drawn on this case, whichever case is on screen."""
        return View(self)

    async def show(
        self,
        values,
        name: str,
        *,
        unit: str = "",
        cmap: str | None = None,
        range: tuple[float, float] | None = None,
    ) -> dict:
        """Colour the car by `values`, one per cell (C,) or per point (P,) of the working mesh, as a field called
        `name`, offered beside Pressure with its own legend, colour scale and probe readout. `unit` titles the legend.
        `cmap` is one of the tool's ramps (coolwarm, turbo, viridis, purplegreen, brownteal, purples, inferno,
        tealorange): by default coolwarm for values either side of zero, else viridis. `range` is (min, max), by
        default from the values. Showing a name again replaces its values. Returns the values' min, max and
        area-weighted mean, and the scale."""
        arguments = {
            "name": name,
            "values": _typed(np.asarray(values, dtype=np.float32).ravel()),
            "unit": unit,
            "ramp": cmap,
            "case_id": self.id,
        }
        if range is not None:
            arguments["min"], arguments["max"] = (float(v) for v in range)
        return await _operation("show_field", arguments)

    async def paint(self, where, color="orange", *, layer: str | None = None, opacity: float | None = None) -> dict:
        """Paint cells of the working mesh over the field shown: `where` is a boolean mask per cell (C,), painted in
        `color`; or colours per cell, (C, 3) or (C, 4) floats from 0 to 1, alpha 0 leaving a cell unpainted. `opacity`
        (0 to 1) lets the field show through. Painting adds to the layer's paint (default "Paint"). Returns the cells
        painted, their area and their share of the car's area."""
        where = np.asarray(where)
        arguments = {"layer": layer, "case_id": self.id}
        if where.ndim == 2:
            arguments["colours"] = _colours(where, len(where))
        else:
            arguments |= {"mask": _typed(where.astype(np.uint8)), "colour": color, "opacity": opacity}
        return await _operation("paint_faces", arguments)


class View:
    """Marks on a case, anchored in 3D so they stay put as the camera moves: labels, points, lines, arrows, boxes and
    planes. `pressure.view` draws on the case on screen at each call; `case.view` on that case.
    Every method draws into `layer` (a default per kind when left out) and returns what it drew."""

    def __init__(self, case: Case | None = None) -> None:
        self._case = case

    def __repr__(self) -> str:
        return f"View({self._case!r})" if self._case else "View(the case on screen)"

    def _id(self) -> int:
        case = self._case or active
        if case is None:
            raise ToolError("No case is on screen. Open a file or an example first.")
        return case.id

    async def _draw(self, name: str, color=None, items: int = 0, **arguments) -> dict:
        if color is not None and not isinstance(color, str):
            arguments["colours"] = _colours(color, items)
        else:
            arguments["colour"] = color
        return await _operation(name, {**arguments, "case_id": self._id()})

    async def label(self, points, texts, *, layer: str | None = None, color: str | None = None, offset=None) -> dict:
        """Label points with text: one point and a string, or (N, 3) points and N strings. Labels keep clear of each
        other, with a leader when moved; `offset` [right, down] in px puts them there first."""
        texts = [texts] if isinstance(texts, str) else [str(text) for text in texts]
        return await self._draw(
            "draw_labels", color, points=_points(points), texts=texts, layer=layer, offset_px=_floats(offset)
        )

    async def points(self, points, *, size: float | None = None, layer: str | None = None, color=None) -> dict:
        """Dots at points, `size` px across."""
        points = _points(points)
        return await self._draw("draw_points", color, points.length // 3, points=points, size_px=size, layer=layer)

    async def line(
        self,
        points,
        *,
        closed: bool = False,
        layer: str | None = None,
        color: str | None = None,
        width: float | None = None,
    ) -> dict:
        """One polyline through (N, 3) points."""
        return await self.lines([points], closed=closed, layer=layer, color=color, width=width)

    async def lines(
        self,
        polylines,
        *,
        closed: bool = False,
        layer: str | None = None,
        color: str | None = None,
        width: float | None = None,
    ) -> dict:
        """Many polylines in one call, each (N, 3) points."""
        lines = [_points(line) for line in polylines]
        return await self._draw("draw_lines", color, lines=lines, closed=closed, layer=layer, width_px=width)

    async def arrows(
        self,
        points,
        vectors,
        *,
        scale: float | None = None,
        layer: str | None = None,
        color=None,
        width: float | None = None,
    ) -> dict:
        """Arrows from (N, 3) points along (N, 3) vectors, `scale` metres per unit of vector; by default the longest is
        drawn 6 % of the car's length. Returns the scale used."""
        points = _points(points)
        scale = None if scale is None else float(scale)
        return await self._draw(
            "draw_arrows",
            color,
            points.length // 3,
            points=points,
            vectors=_points(vectors),
            scale=scale,
            layer=layer,
            width_px=width,
        )

    async def box(
        self, lo, hi, *, layer: str | None = None, color: str | None = None, width: float | None = None
    ) -> dict:
        """A box along the axes from corner `lo` to corner `hi`, [x, y, z] each."""
        return await self._draw("draw_box", color, min=_floats(lo), max=_floats(hi), layer=layer, width_px=width)

    async def plane(
        self,
        point,
        normal,
        size=None,
        *,
        layer: str | None = None,
        color: str | None = None,
        width: float | None = None,
    ) -> dict:
        """A rectangle of the plane through `point` square to `normal`, `size` (width, height) in metres."""
        size = None if size is None else _floats(np.broadcast_to(size, 2))
        return await self._draw(
            "draw_plane", color, point=_floats(point), normal=_floats(normal), size_m=size, layer=layer, width_px=width
        )

    async def layers(self) -> list[dict]:
        """The case's layers, in drawing order: name, kind, who made it, whether it shows, and what it holds."""
        return (await _operation("list_layers", {"case_id": self._id()}))["layers"]

    async def hide(self, layer: str) -> dict:
        """Hide a layer; `show` brings it back."""
        return await _operation("set_layer", {"layer": layer, "visible": False, "case_id": self._id()})

    async def show(self, layer: str) -> dict:
        """Show a hidden layer again; a field layer colours the car by its field."""
        return await _operation("set_layer", {"layer": layer, "visible": True, "case_id": self._id()})

    async def clear(self, layer: str | None = None) -> list[str]:
        """Remove one layer, or every layer the assistant drew on the case. Removing the engineer's asks them first.
        Returns the names removed."""
        return (await _operation("remove_layer", {"layer": layer, "case_id": self._id()}))["removed"]


class Prediction:
    """The model's prediction on a surface: `cells`, the surface with `cp` and `spread` per cell, and the numbers the
    tool reports for an open case (`list_cases`), under the same names: `cx`, `cy` and `cz`, the pressure force
    coefficients on `reference_area_m2` (`cx` is pressure drag); `cx_by_network`; `area_m2` and `mean_cp`; `cp_min`
    and `cp_max`, each `{"value", "center_m", "normal"}`; `length_m`, `width_m`, `height_m` and `outside_training`.
    `summary` holds them all as a dict, for a table row. `seconds` is the time the prediction took."""

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

view = View()
"""Marks drawn on the case on screen."""


async def run(name: str, **arguments) -> dict:
    """Run one of the tool's operations, as the engineer's controls and the assistant do, and return its result. What
    it changes shows on screen. An operation the assistant needs the engineer's approval for (closing a case,
    downloading, deleting a file, moving the engineer's file, removing the engineer's layer) asks them here too."""
    plain = json.loads(json.dumps(arguments, default=_plain))
    return _json(await _ask("operation", {"name": name, "args": plain}))


async def predict(surface: Surface, name: str = "prediction") -> Prediction:
    """Predict Cp on every cell of `surface` with the model, as the tool does for a file, on the cells as given (the
    working mesh's resolution is what a study needs, and coarser starves the model's inputs). Nothing opens in the
    tool."""
    result = await _ask("predict", {"name": name, **_polygons(surface)})
    return Prediction(name, _json(result.summary), result.cells, float(result.seconds))


async def write_vtp(path: str, surface: Surface, cell_data: dict | None = None, point_data: dict | None = None) -> str:
    """Write `surface` as a VTK PolyData file at `path` in the workspace, with arrays per cell and per point (one value
    or one row each). By default the cells carry the surface's prediction as `CpPredicted` and `CpSpread`, the names
    the tool's own export uses. Returns the path."""
    if cell_data is None:
        predicted = (("CpPredicted", surface.cp), ("CpSpread", surface.spread))
        cell_data = {label: values for label, values in predicted if values is not None}
    arrays = {
        "cellData": {name: _typed(np.asarray(values, dtype=np.float32)) for name, values in cell_data.items()},
        "pointData": {
            name: _typed(np.asarray(values, dtype=np.float32)) for name, values in (point_data or {}).items()
        },
    }
    await _ask("write_vtp", {"path": path, **_polygons(surface), **arrays})
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


async def _operation(name: str, arguments: dict) -> dict:
    """Run an operation with its arrays sent as typed arrays, not JSON; arguments left as None are left out."""
    given = {key: value for key, value in arguments.items() if value is not None}
    return _json(await _ask("operation", {"name": name, "args": given}))


def _points(points) -> JsProxy:
    """(N, 3) points, or one [x, y, z], as a flat Float32Array."""
    points = np.asarray(points, dtype=np.float32)
    if points.shape[-1:] != (3,) or points.ndim > 2:
        raise ValueError(f"Expected points of shape (N, 3) or (3,), got {points.shape}.")
    return _typed(points.reshape(-1, 3))


def _floats(values) -> list[float] | None:
    """A sequence of numbers as plain floats, for an operation's JSON arguments; None stays None."""
    return None if values is None else [float(v) for v in values]


def _colours(colours, count: int) -> JsProxy:
    """One colour per item, (N, 3) or (N, 4) floats from 0 to 1 (or bytes), as RGBA bytes."""
    colours = np.asarray(colours)
    if colours.ndim != 2 or colours.shape[1] not in (3, 4) or len(colours) != count:
        raise ValueError(f"Expected one colour per item, shape ({count}, 3) or ({count}, 4); got {colours.shape}.")
    if colours.dtype != np.uint8:
        colours = np.clip(np.rint(255 * colours), 0, 255).astype(np.uint8)
    if colours.shape[1] == 3:
        colours = np.concatenate([colours, np.full((count, 1), 255, np.uint8)], axis=1)
    return _typed(colours)


def _polygons(surface: Surface) -> dict:
    """The surface's points, connectivity and offsets as typed arrays, as the page takes a geometry."""
    return {
        "points": _typed(np.ascontiguousarray(surface.points, dtype=np.float32)),
        "connectivity": _typed(np.ascontiguousarray(surface.connectivity, dtype=np.uint32)),
        "offsets": _typed(np.ascontiguousarray(surface.offsets, dtype=np.uint32)),
    }


_TYPED = {
    np.dtype(np.float32): "Float32Array",
    np.dtype(np.uint32): "Uint32Array",
    np.dtype(np.uint8): "Uint8Array",
}


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
        cells.points,
        cells.connectivity,
        cells.offsets,
        centers=cells.centers,
        normals=cells.normals,
        areas=cells.areas,
        cp=cells.cp,
        spread=cells.sigma,
        reference=cells.cfd,
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
