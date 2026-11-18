"""What the model is told about each cell of a car's surface, computed from the polygons alone.

Three blocks of inputs, all descriptions of the car's shape, in the order of `FEATURE_COLUMNS` (107
columns):

- `local` (7): where the cell is on the car, in metres from the nose, the tail, the centre plane and
  the lowest point, and its outward normal.
- `surface` (12 per scale): the surface around the cell at each scale of `SURFACE_SCALES`: its mean
  facing and how much of it there is (1 on a flat panel), and the same for the part of it ahead of
  the cell and the part behind.
- `slices` (4 per station): thirteen slices across the car, at the cell's station and at `SLICE_DX`
  ahead and behind; from each, its area, how far it reaches outboard of the cell at the cell's
  height, and how far it reaches above and below the cell at the cell's distance from the centre
  plane.

Every lookup is filled from samples of the surface (`surface_samples`): a polygon whose edges are all
within half the raster step is one sample at its centre, and a larger one is cut into pieces that
small, so a coarse mesh fills the lookups as a fine one does. The surface around a cell is a
quantity carried by every sample (an area, an area times the normal) spread onto a grid in metres,
blurred by a Gaussian and read back at the cell, so it does not depend on how finely the file is
meshed; its fields are kept only at the nodes a read at a cell touches, a few percent of their
grids. The slices are read from a 2 cm raster. Every lookup keeps the car's two sides apart. Lookups
are placed in the car's own frame, as the position inputs are: x from its nose, y from its centre
plane, z from its lowest cell centre; so moving the whole car changes none of them.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import correlate1d, gaussian_filter1d, maximum_filter1d

from surrogate.surface import SurfaceMesh

# The slices across the car are read from a 2 cm raster of this box, in the car's own frame: x from
# its nose, y from its centre plane, z from its lowest point, as the position inputs are, so that
# moving the whole car moves nothing the model is told.
RASTER_STEP = 0.02
DOMAIN_X_FROM_NOSE = (-0.5, 5.5)
DOMAIN_Y_HALF = 1.2
DOMAIN_Z_FROM_GROUND = (-0.15, 1.65)
# The side view, from which a slice's outboard reach at the cell's height is read, is binned 2 mm in
# height and holds at each height the reach within 1 cm above and below it. Read from 2 cm bands, the
# reach of a cell on a roof or a sill swings by the whole half-width of the car with where the band
# edge falls against the cell.
HEIGHT_STEP = RASTER_STEP / 10
HEIGHT_WINDOW = RASTER_STEP / 2
# Slices ahead of and behind the cell at distances doubling from 5 cm to a third of the car: shape
# close by matters in detail, shape far away only in bulk.
SLICE_DX = (-1.6, -0.8, -0.4, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6)
# What each slice gives, in column order: its area, and the body's reach outboard of, above and below the cell.
SLICE_READS = ("area", "outboard", "above", "below")

# The surface around a cell at four scales R, about three times apart: the surface blurred by a
# Gaussian of standard deviation 0.4 R, cut at three standard deviations, on a grid of step R / 3.
SURFACE_SCALES = (0.03, 0.10, 0.25, 0.75)
SURFACE_STEPS = (0.01, 0.10 / 3.0, 0.25 / 3.0, 0.25)
SURFACE_SIGMA = 0.4
GAUSSIAN_TRUNCATE = 3.0
# What each scale gives, in column order, with each read's number of columns.
SURFACE_READS = (
    ("meanNormal", 3),
    ("amount", 1),
    ("aheadNormal", 3),
    ("aheadAmount", 1),
    ("behindNormal", 3),
    ("behindAmount", 1),
)
LOCAL_WIDTH = 7
SURFACE_WIDTH = sum(width for _, width in SURFACE_READS)  # columns per scale


@dataclass(frozen=True)
class FeatureColumns:
    """Column slices of the feature matrix, one per block, in order."""

    local: slice
    surface: slice
    slices: slice

    @property
    def total_dim(self) -> int:
        return self.slices.stop


_SURFACE_END = LOCAL_WIDTH + SURFACE_WIDTH * len(SURFACE_SCALES)
FEATURE_COLUMNS = FeatureColumns(
    local=slice(0, LOCAL_WIDTH),
    surface=slice(LOCAL_WIDTH, _SURFACE_END),
    slices=slice(_SURFACE_END, _SURFACE_END + len(SLICE_READS) * (1 + len(SLICE_DX))),
)
BLOCKS = ("local", "surface", "slices")


@dataclass(frozen=True)
class SurfaceSamples:
    """The points every lookup is filled from, each carrying a share of the surface's area."""

    positions: np.ndarray  # (S, 3) float32
    areas: np.ndarray  # (S,) float64 m^2
    normals: np.ndarray  # (S, 3) float32, the outward normal of the polygon the sample is on


def _squared_length(vectors: np.ndarray) -> np.ndarray:
    return (vectors[:, 0] * vectors[:, 0] + vectors[:, 1] * vectors[:, 1]) + vectors[:, 2] * vectors[:, 2]


def _piece_centroids(n: int) -> tuple[np.ndarray, np.ndarray]:
    """The centroids, as (u, v) along a triangle's two edges from its first corner, of the n^2 equal
    triangles it is cut into by n steps along each edge: n(n + 1)/2 pointing like it, n(n - 1)/2 not."""
    u: list[float] = []
    v: list[float] = []
    for a in range(n):
        for b in range(n - a):
            u.append((a + 1.0 / 3.0) / n)
            v.append((b + 1.0 / 3.0) / n)
            if a + b <= n - 2:
                u.append((a + 2.0 / 3.0) / n)
                v.append((b + 2.0 / 3.0) / n)
    return np.array(u), np.array(v)


def surface_samples(surface: SurfaceMesh) -> SurfaceSamples:
    """The surface as samples no further apart than half the raster step. Lookups filled at polygon
    centres alone leave holes where polygons are larger than the raster step, and a coarse file then
    reads as a car with gaps in it. A polygon whose edges are all within half a step is one sample at
    its centre carrying its area. A larger one is fanned into triangles from its first corner and each
    triangle cut into n^2 equal pieces, n = ceil(longest edge / half a step), one sample at each
    piece's centroid carrying its area."""
    half = 0.5 * RASTER_STEP
    offsets, corners = surface.polygon_offsets, surface.polygon_corners
    points = surface.points.astype(np.float64)

    # The corner after each corner, the last of a polygon followed by its first
    following = np.arange(1, len(corners) + 1)
    following[offsets[1:] - 1] = offsets[:-1]
    edge_squared = np.zeros(len(corners))
    for axis in range(3):
        d = points[corners[following], axis] - points[corners, axis]
        edge_squared += d * d
    wide = np.maximum.reduceat(edge_squared, offsets[:-1]) > half * half

    # The fan triangles (c0, cj, cj+1), j = 1 .. k - 2, of every wide polygon of k corners
    wide_polygons = np.flatnonzero(wide)
    fans = np.maximum(np.diff(offsets)[wide_polygons] - 2, 0)
    polygon = np.repeat(wide_polygons, fans)
    first = offsets[polygon]
    j = np.arange(len(polygon)) - np.repeat(np.cumsum(fans) - fans, fans) + 1
    p0 = points[corners[first]]
    pj = points[corners[first + j]]
    pk = points[corners[first + j + 1]]
    e1 = pj - p0
    e2 = pk - p0
    longest = np.maximum(np.maximum(_squared_length(e1), _squared_length(e2)), _squared_length(pk - pj))
    cuts = np.ceil(np.sqrt(longest) / half).astype(np.int64)
    cross = np.column_stack(
        [
            e1[:, 1] * e2[:, 2] - e1[:, 2] * e2[:, 1],
            e1[:, 2] * e2[:, 0] - e1[:, 0] * e2[:, 2],
            e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0],
        ]
    )
    area = 0.5 * np.sqrt(_squared_length(cross))

    positions = [surface.centers[~wide]]
    areas = [surface.areas[~wide]]
    normals = [surface.normals[~wide]]
    for n in np.unique(cuts[cuts > 0]):
        cut = cuts == n
        u, v = _piece_centroids(int(n))
        at = (p0[cut, None, :] + u[None, :, None] * e1[cut, None, :]) + v[None, :, None] * e2[cut, None, :]
        positions.append(at.reshape(-1, 3).astype(np.float32))
        areas.append(np.repeat(area[cut] / (n * n), len(u)))
        normals.append(np.repeat(surface.normals[polygon[cut]], len(u), axis=0))
    return SurfaceSamples(np.concatenate(positions), np.concatenate(areas), np.concatenate(normals))


@dataclass(frozen=True)
class Grid:
    """Nodes at origin + step * (i, j, k)."""

    origin: np.ndarray  # (3,) float64
    step: float
    nodes: tuple[int, int, int]

    def neighbours(self, points: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """For each of the 27 nodes around every point: its flat node number and its quadratic
        B-spline weight. A Gaussian of about one node spread and read with trilinear weights reads a
        panel 16 to 33% differently by where the panel falls between nodes, which draws contours
        along a sloping panel; quadratic weights cut that to 1.5 to 5%."""
        q = (np.asarray(points, dtype=np.float64) - self.origin) / self.step
        nodes = np.array(self.nodes)
        # The nearest node, an exact half step rounding up; never the outer node, which the
        # grid's margin keeps clear of the surface
        i = np.clip(np.floor(q + 0.5).astype(np.int64), 1, nodes - 2)
        t = q - i
        weights = (0.5 * (0.5 - t) ** 2, 0.75 - t * t, 0.5 * (0.5 + t) ** 2)
        for dx in range(3):
            for dy in range(3):
                for dz in range(3):
                    flat = ((i[:, 0] + dx - 1) * nodes[1] + i[:, 1] + dy - 1) * nodes[2] + i[:, 2] + dz - 1
                    yield flat, weights[dx][:, 0] * weights[dy][:, 1] * weights[dz][:, 2]

    def spread(self, points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Each point's weight shared among the 27 nodes around it."""
        size = int(np.prod(self.nodes))
        out = np.zeros(size)
        for flat, w in self.neighbours(points):
            out += np.bincount(flat, weights=weights * w, minlength=size)
        return out.reshape(self.nodes)


@dataclass(frozen=True)
class SurfaceScale:
    """The surface around the car's cells at one scale: blurred area and area-weighted normal, kept
    at the nodes around the cells only."""

    scale: float
    grid: Grid
    kept: np.ndarray  # (K,) int32 flat numbers of the nodes a read at a cell touches, ascending
    whole: np.ndarray  # (K, 4) float32 at those nodes: area, then area times each normal component
    ahead: np.ndarray  # (K, 4) float32: the same with the Gaussian cut to its upstream half
    flat_panel: float  # blurred area read on a flat panel, on average over where it falls between nodes

    def read(self, cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """`whole` and `ahead` at the centres of cells of the car, from the 27 nodes around each."""
        whole = np.zeros((len(cells), 4))
        ahead = np.zeros((len(cells), 4))
        for flat, w in self.grid.neighbours(cells):
            i = np.minimum(np.searchsorted(self.kept, flat), len(self.kept) - 1)
            if not np.array_equal(self.kept[i], flat):
                raise ValueError("surface fields are kept only around the cells they were built from")
            whole += w[:, None] * self.whole[i]
            ahead += w[:, None] * self.ahead[i]
        return whole, ahead


def _gaussian_weights(sigma_nodes: float) -> np.ndarray:
    """The normalised weights `gaussian_filter1d` uses, at offsets -r..r."""
    r = int(GAUSSIAN_TRUNCATE * sigma_nodes + 0.5)
    w = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_nodes) ** 2)
    return w / w.sum()


def _upstream_half(weights: np.ndarray) -> np.ndarray:
    """Correlation weights kept upstream: weight j multiplies the node j - r places along x, so nodes
    at smaller x count fully and the node itself by half."""
    r = len(weights) // 2
    return np.concatenate([weights[:r], [0.5 * weights[r]], np.zeros(r)])


def _surface_scale(surface: SurfaceMesh, samples: SurfaceSamples, scale: float, step: float) -> SurfaceScale:
    h = step
    sigma_nodes = SURFACE_SIGMA * scale / h
    weights = _gaussian_weights(sigma_nodes)
    r = len(weights) // 2
    margin = (r + 2) * h
    xmin, xmax, ymin, ymax, zmin, zmax = surface.bounds
    lo = np.array([xmin, ymin, zmin]) - margin
    nodes = np.ceil((np.array([xmax, ymax, zmax]) + margin - lo) / h).astype(int) + 1
    grid = Grid(lo, h, (int(nodes[0]), int(nodes[1]), int(nodes[2])))

    around_cells = np.zeros(int(nodes.prod()), dtype=bool)
    for flat, _ in grid.neighbours(surface.centers):
        around_cells[flat] = True
    kept = np.flatnonzero(around_cells).astype(np.int32)

    # One quantity at a time, so that only one whole grid is held at once
    positions = samples.positions.astype(np.float64)
    whole = np.empty((len(kept), 4), dtype=np.float32)
    ahead = np.empty((len(kept), 4), dtype=np.float32)
    for k, carried in enumerate((samples.areas, *(samples.areas * samples.normals[:, j] for j in range(3)))):
        g = grid.spread(positions, carried).astype(np.float32)
        yz = gaussian_filter1d(g, sigma_nodes, axis=1, truncate=GAUSSIAN_TRUNCATE)
        yz = gaussian_filter1d(yz, sigma_nodes, axis=2, truncate=GAUSSIAN_TRUNCATE)
        whole[:, k] = gaussian_filter1d(yz, sigma_nodes, axis=0, truncate=GAUSSIAN_TRUNCATE).ravel()[kept]
        ahead[:, k] = correlate1d(yz, _upstream_half(weights), axis=0).ravel()[kept]
    # A flat panel lays h^2 of area per node column. Across the panel, the spread and the read each
    # share it among the three nodes around it; averaged over where the panel falls between nodes,
    # the node itself counts 11/20, each neighbour 13/60 and each second neighbour 1/120 (the
    # quintic B-spline at whole steps).
    flat_panel = h * h * (11.0 / 20.0 * weights[r] + 13.0 / 30.0 * weights[r + 1] + 1.0 / 60.0 * weights[r + 2])
    return SurfaceScale(scale, grid, kept, whole, ahead, flat_panel)


@dataclass(frozen=True)
class SurfaceContext:
    """Every lookup of one car, built once from all of its surface and then read at any of its cells:
    the 120,000 sampled cells of a training car, or every cell at inference."""

    bounds: tuple[float, float, float, float, float, float]
    y_center: float
    nx: int  # raster bins along x
    y_out: np.ndarray  # (2, nx, nh) how far the body reaches from the centre plane at (x, height),
    # within HEIGHT_WINDOW of the height: [0] toward -y, [1] toward +y
    z_top: np.ndarray  # (nx, ny) the highest sample over each (x, y) column, the ground where it holds no body
    z_bot: np.ndarray  # (nx, ny) the lowest, the ground where the column holds no body
    slice_area: np.ndarray  # (nx,) the area of the car's slice across x, S(x) in m^2
    surface_scales: tuple[SurfaceScale, ...]

    @classmethod
    def from_surface(cls, surface: SurfaceMesh) -> SurfaceContext:
        nose, _, ymin, ymax, z_ground, _ = surface.bounds
        y_center = 0.5 * (ymin + ymax)
        samples = surface_samples(surface)
        x_rel = samples.positions[:, 0] - nose
        y_rel = samples.positions[:, 1] - y_center
        z = samples.positions[:, 2]

        h = RASTER_STEP
        nx = round((DOMAIN_X_FROM_NOSE[1] - DOMAIN_X_FROM_NOSE[0]) / h)
        ny = 2 * round(DOMAIN_Y_HALF / h)
        nh = round((DOMAIN_Z_FROM_GROUND[1] - DOMAIN_Z_FROM_GROUND[0]) / HEIGHT_STEP)
        ix = np.clip(((x_rel - DOMAIN_X_FROM_NOSE[0]) / h).astype(np.int64), 0, nx - 1)
        iy = np.clip(((y_rel + DOMAIN_Y_HALF) / h).astype(np.int64), 0, ny - 1)
        ih = np.clip((((z - z_ground) - DOMAIN_Z_FROM_GROUND[0]) / HEIGHT_STEP).astype(np.int64), 0, nh - 1)

        z_top = np.full(nx * ny, -np.inf, dtype=np.float32)
        z_bot = np.full(nx * ny, np.inf, dtype=np.float32)
        np.maximum.at(z_top, ix * ny + iy, z)
        np.minimum.at(z_bot, ix * ny + iy, z)
        empty = ~np.isfinite(z_top)
        z_top[empty] = z_ground
        z_bot[empty] = z_ground
        # S(x): the height of the body over each (x, y) column, summed across y
        column_height = (z_top - z_bot).reshape(nx, ny)

        side = (y_rel >= 0.0).astype(np.int64)
        y_out = np.zeros(2 * nx * nh, dtype=np.float32)
        np.maximum.at(y_out, (side * nx + ix) * nh + ih, np.abs(y_rel))
        reach = round(HEIGHT_WINDOW / HEIGHT_STEP)
        y_out = maximum_filter1d(y_out.reshape(2, nx, nh), size=2 * reach + 1, axis=2, mode="constant", cval=0.0)

        return cls(
            bounds=surface.bounds,
            y_center=y_center,
            nx=nx,
            y_out=y_out,
            z_top=z_top.reshape(nx, ny),
            z_bot=z_bot.reshape(nx, ny),
            slice_area=(column_height.sum(axis=1) * h).astype(np.float32),
            surface_scales=tuple(
                _surface_scale(surface, samples, scale, step)
                for scale, step in zip(SURFACE_SCALES, SURFACE_STEPS, strict=True)
            ),
        )

    def shape_descriptor(self) -> dict[str, float]:
        """The car's length, width and height, which the coverage check compares with the training cars'.
        Never a model input: on ten cars it would let the networks tell the cars apart instead of learning
        pressure."""
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
        return {"length": float(xmax - xmin), "width": float(ymax - ymin), "height": float(zmax - zmin)}


def _between_bins(q: np.ndarray, lo: float, step: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Where `q` falls among the centres lo + (i + 1/2) step of a raster's `n` bins along one axis: the
    bin whose centre is at or below it, and the fraction of the way to the next centre. Beyond the
    outer centres, the outer bin's value."""
    t = (q - lo) / step - 0.5
    i = np.clip(np.floor(t).astype(np.int64), 0, n - 2)
    return i, np.clip(t - i, 0.0, 1.0)


def _bilinear(field: np.ndarray, a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """A raster `field` (na, nb) read between its bin centres; a bin with no body takes part with its
    empty value."""
    (i, fi), (j, fj) = a, b
    return (1.0 - fi) * ((1.0 - fj) * field[i, j] + fj * field[i, j + 1]) + fi * (
        (1.0 - fj) * field[i + 1, j] + fj * field[i + 1, j + 1]
    )


def extract_query_features(context: SurfaceContext, query_centers: np.ndarray, query_normals: np.ndarray) -> np.ndarray:
    """The feature matrix `(M, FEATURE_COLUMNS.total_dim)` at `M` cells of the car `context` was built from."""
    xmin, xmax, _, _, zmin, _ = context.bounds
    centers = np.asarray(query_centers, dtype=np.float64)
    normals = np.asarray(query_normals, dtype=np.float64)
    sx = centers[:, 0]
    sy = centers[:, 1] - context.y_center
    sz = centers[:, 2]

    local = np.column_stack([sx - xmin, xmax - sx, sy, sz - zmin, normals])

    # At each scale, the surface around the cell, the part of it ahead and the part behind: the mean
    # facing of each, and how much there is against what a flat panel holds
    surface_columns: list[np.ndarray] = []
    for s in context.surface_scales:
        whole, ahead = s.read(centers)
        half_panel = 0.5 * s.flat_panel
        for part, flat_panel in ((whole, s.flat_panel), (ahead, half_panel), (whole - ahead, half_panel)):
            surface_columns += [part[:, 1:] / np.maximum(part[:, :1], 1e-12), part[:, :1] / flat_panel]
    surface = np.concatenate(surface_columns, axis=1)

    # Thirteen slices across the car, at the cell's station and ahead and behind it. From each: its
    # area; how far it reaches outboard of the cell on the cell's own side, at the cell's height; and
    # how far it reaches above and below the cell, at the cell's distance from the centre plane. Read
    # between bin centres rather than from the nearest bin, so that a read moves smoothly with the
    # cell instead of stepping where it crosses a bin edge, which drew contours on sloping panels.
    on_plus_side = sy >= 0.0
    height = _between_bins(sz - zmin, DOMAIN_Z_FROM_GROUND[0], HEIGHT_STEP, context.y_out.shape[2])
    row = _between_bins(sy, -DOMAIN_Y_HALF, RASTER_STEP, context.z_top.shape[1])
    slice_columns: list[np.ndarray] = []
    for x_rel in (sx - xmin, *(sx - xmin + dx for dx in SLICE_DX)):
        station = _between_bins(x_rel, DOMAIN_X_FROM_NOSE[0], RASTER_STEP, context.nx)
        i, f = station
        outboard = np.where(
            on_plus_side, _bilinear(context.y_out[1], station, height), _bilinear(context.y_out[0], station, height)
        )
        slice_columns += [
            (1.0 - f) * context.slice_area[i] + f * context.slice_area[i + 1],
            outboard - np.abs(sy),
            _bilinear(context.z_top, station, row) - sz,
            sz - _bilinear(context.z_bot, station, row),
        ]
    slices = np.column_stack(slice_columns)

    features = np.concatenate([local, surface, slices], axis=1).astype(np.float32)
    assert features.shape == (len(centers), FEATURE_COLUMNS.total_dim)
    return features
