"""What the model is told about each cell of a car's surface, computed from the polygons alone.

Three blocks of inputs, all descriptions of the car's shape, in the order of `FEATURE_COLUMNS` (107
columns):

- `local` (7): where the cell is on the car, in metres from the nose, the tail, the centre plane and
  the lowest point, and its outward normal.
- `surface` (12 per scale): the surface around the cell at each scale of `SURFACE_SCALES`: its mean
  facing and how much of it there is (1 on a flat panel), and the same for the part of it ahead of
  the cell and the part behind.
- `sections` (4 per station): thirteen slices across the car, at the cell's station and at
  `SECTION_DX` ahead and behind; from each, its area, how far it reaches outboard of the cell at the
  cell's height, and how far it reaches above and below the cell at the cell's distance from the
  centre plane.

The surface around a cell is a quantity carried by every cell (an area, an area times the normal)
spread onto a grid in metres, blurred by a Gaussian and read back at the cell, so it does not depend
on how finely the file is meshed; its fields are kept only at the nodes a read at a cell touches, a
few percent of their grids. The slices are read from a 2 cm raster filled from the cell centres, so
they need cells smaller than 2 cm (DrivAerML's are 1.1 cm at the 99th percentile). Every lookup keeps
the car's two sides apart. Lookups are placed in the file's own coordinates, which every DrivAerML
run shares, with y measured from the car's centre plane and the ground at its lowest point.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import correlate1d, gaussian_filter1d

from surrogate.surface import SurfaceMesh

# The slices across the car are read from a 2 cm raster of this box.
RASTER_STEP = 0.02
DOMAIN_X = (-1.3, 4.7)
DOMAIN_Y_HALF = 1.2
DOMAIN_Z = (-0.45, 1.35)
# Slices ahead of and behind the cell at distances doubling from 5 cm to a third of the car: shape
# close by matters in detail, shape far away only in bulk.
SECTION_DX = (-1.6, -0.8, -0.4, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6)

# The surface around a cell at four scales R, about three times apart: the surface blurred by a
# Gaussian of standard deviation 0.4 R, cut at three standard deviations, on a grid of step R / 3.
# The 3 cm scale's grid has a step of R / 2: at R / 3 it would hold 17 million nodes on a car, and
# the model is no better for it beyond the spread between training seeds.
SURFACE_SCALES = (0.03, 0.10, 0.25, 0.75)
SURFACE_STEPS = (0.015, 0.10 / 3.0, 0.25 / 3.0, 0.25)
SURFACE_SIGMA = 0.4
GAUSSIAN_TRUNCATE = 3.0
# What each scale gives, in column order, with each read's number of columns.
SURFACE_READS = (("meanNormal", 3), ("amount", 1), ("aheadNormal", 3), ("aheadAmount", 1), ("behindNormal", 3), ("behindAmount", 1))


@dataclass(frozen=True)
class FeatureColumns:
    """Column slices of the feature matrix, one per block, in order."""

    local: slice
    surface: slice
    sections: slice

    @property
    def total_dim(self) -> int:
        return self.sections.stop


_SURFACE_END = 7 + sum(width for _, width in SURFACE_READS) * len(SURFACE_SCALES)
FEATURE_COLUMNS = FeatureColumns(
    local=slice(0, 7),
    surface=slice(7, _SURFACE_END),
    sections=slice(_SURFACE_END, _SURFACE_END + 4 * (1 + len(SECTION_DX))),
)
BLOCKS = ("local", "surface", "sections")


@dataclass(frozen=True)
class Grid:
    """Nodes at origin + step * (i, j, k)."""

    origin: np.ndarray  # (3,) float64
    step: float
    nodes: tuple[int, int, int]

    def corners(self, points: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """For each of the eight nodes around every point: its flat node number and its trilinear weight."""
        q = (np.asarray(points, dtype=np.float64) - self.origin) / self.step
        nodes = np.array(self.nodes)
        i0 = np.clip(np.floor(q).astype(np.int64), 0, nodes - 2)
        f = np.clip(q - i0, 0.0, 1.0)
        for dx in (0, 1):
            wx = f[:, 0] if dx else 1.0 - f[:, 0]
            for dy in (0, 1):
                wy = f[:, 1] if dy else 1.0 - f[:, 1]
                for dz in (0, 1):
                    wz = f[:, 2] if dz else 1.0 - f[:, 2]
                    yield ((i0[:, 0] + dx) * nodes[1] + i0[:, 1] + dy) * nodes[2] + i0[:, 2] + dz, wx * wy * wz

    def spread(self, points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Each point's weight shared among the eight nodes around it by trilinear weights."""
        size = int(np.prod(self.nodes))
        out = np.zeros(size)
        for flat, w in self.corners(points):
            out += np.bincount(flat, weights=weights * w, minlength=size)
        return out.reshape(self.nodes)


@dataclass(frozen=True)
class SurfaceScale:
    """The surface around the car's cells at one scale: blurred area and area-weighted normal, kept
    at the nodes around the cells only."""

    scale: float
    grid: Grid
    kept: np.ndarray   # (K,) int32 flat numbers of the nodes a read at a cell touches, ascending
    whole: np.ndarray  # (K, 4) float32 at those nodes: area, then area times each normal component
    ahead: np.ndarray  # (K, 4) float32: the same with the Gaussian cut to its upstream half
    flat_panel: float  # blurred area read on a flat panel, on average over where it falls between nodes

    def read(self, cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """`whole` and `ahead` at the centres of cells of the car, interpolated trilinearly."""
        whole = np.zeros((len(cells), 4))
        ahead = np.zeros((len(cells), 4))
        for flat, w in self.grid.corners(cells):
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


def _surface_scale(surface: SurfaceMesh, scale: float, step: float) -> SurfaceScale:
    h = step
    sigma_nodes = SURFACE_SIGMA * scale / h
    weights = _gaussian_weights(sigma_nodes)
    r = len(weights) // 2
    margin = (r + 2) * h
    xmin, xmax, ymin, ymax, zmin, zmax = surface.bounds
    lo = np.array([xmin, ymin, zmin]) - margin
    nodes = np.ceil((np.array([xmax, ymax, zmax]) + margin - lo) / h).astype(int) + 1
    grid = Grid(lo, h, (int(nodes[0]), int(nodes[1]), int(nodes[2])))
    centers = surface.centers.astype(np.float64)

    around_cells = np.zeros(int(nodes.prod()), dtype=bool)
    for flat, _ in grid.corners(centers):
        around_cells[flat] = True
    kept = np.flatnonzero(around_cells).astype(np.int32)

    # One quantity at a time, so that only one whole grid is held at once
    whole = np.empty((len(kept), 4), dtype=np.float32)
    ahead = np.empty((len(kept), 4), dtype=np.float32)
    for k, carried in enumerate((surface.areas, *(surface.areas * surface.normals[:, j] for j in range(3)))):
        g = grid.spread(centers, carried).astype(np.float32)
        yz = gaussian_filter1d(g, sigma_nodes, axis=1, truncate=GAUSSIAN_TRUNCATE)
        yz = gaussian_filter1d(yz, sigma_nodes, axis=2, truncate=GAUSSIAN_TRUNCATE)
        whole[:, k] = gaussian_filter1d(yz, sigma_nodes, axis=0, truncate=GAUSSIAN_TRUNCATE).ravel()[kept]
        ahead[:, k] = correlate1d(yz, _upstream_half(weights), axis=0).ravel()[kept]
    # A flat panel lays h^2 of area per node column. Across the panel, the spread and the read each
    # share it between the two nodes either side; averaged over where the panel falls between them,
    # the node itself counts 2/3 and each neighbour 1/6.
    flat_panel = h * h * (2.0 / 3.0 * weights[r] + 1.0 / 3.0 * weights[r + 1])
    return SurfaceScale(scale, grid, kept, whole, ahead, flat_panel)


@dataclass(frozen=True)
class SurfaceContext:
    """Every lookup of one car, built once from all of its cells and then read at any of them: the
    120,000 sampled cells of a training car, or every cell at inference."""

    bounds: tuple[float, float, float, float, float, float]
    y_center: float
    nx: int                     # raster nodes along x
    y_out: np.ndarray           # (2, nx, nz) how far the body reaches from the centre plane at (x, z): [0] toward -y, [1] toward +y
    z_top: np.ndarray           # (nx, ny) the highest cell centre over each (x, y) column, the ground where the column holds no body
    z_bot: np.ndarray           # (nx, ny) the lowest, the ground where the column holds no body
    section_area_x: np.ndarray  # (nx,) cross-section area S(x) in m^2
    surface_scales: tuple[SurfaceScale, ...]

    @classmethod
    def from_surface(cls, surface: SurfaceMesh) -> SurfaceContext:
        _, _, ymin, ymax, z_ground, _ = surface.bounds
        y_center = 0.5 * (ymin + ymax)
        x = surface.centers[:, 0]
        y_rel = surface.centers[:, 1] - y_center
        z = surface.centers[:, 2]

        h = RASTER_STEP
        nx = int(round((DOMAIN_X[1] - DOMAIN_X[0]) / h))
        ny = 2 * int(round(DOMAIN_Y_HALF / h))
        nz = int(round((DOMAIN_Z[1] - DOMAIN_Z[0]) / h))
        ix = np.clip(((x - DOMAIN_X[0]) / h).astype(np.int64), 0, nx - 1)
        iy = np.clip(((y_rel + DOMAIN_Y_HALF) / h).astype(np.int64), 0, ny - 1)
        iz = np.clip(((z - DOMAIN_Z[0]) / h).astype(np.int64), 0, nz - 1)

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
        y_out = np.zeros(2 * nx * nz, dtype=np.float32)
        np.maximum.at(y_out, (side * nx + ix) * nz + iz, np.abs(y_rel))

        return cls(
            bounds=surface.bounds,
            y_center=y_center,
            nx=nx,
            y_out=y_out.reshape(2, nx, nz),
            z_top=z_top.reshape(nx, ny),
            z_bot=z_bot.reshape(nx, ny),
            section_area_x=(column_height.sum(axis=1) * h).astype(np.float32),
            surface_scales=tuple(_surface_scale(surface, scale, step) for scale, step in zip(SURFACE_SCALES, SURFACE_STEPS)),
        )

    def shape_descriptor(self) -> dict[str, float]:
        """Whole-car dimensions for the coverage check against the training cars. Never a model input:
        on ten cars it would let the networks tell the cars apart instead of learning pressure."""
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
        return {
            "length": float(xmax - xmin),
            "width": float(ymax - ymin),
            "height": float(zmax - zmin),
            "z_min": float(zmin),
            "max_section_area": float(self.section_area_x.max()),
            "plan_volume_proxy": float(self.section_area_x.sum() * RASTER_STEP),
        }


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
        for part, flat_panel in ((whole, s.flat_panel), (ahead, 0.5 * s.flat_panel), (whole - ahead, 0.5 * s.flat_panel)):
            surface_columns += [part[:, 1:] / np.maximum(part[:, :1], 1e-12), part[:, :1] / flat_panel]
    surface = np.concatenate(surface_columns, axis=1)

    # Thirteen slices across the car, at the cell's station and ahead and behind it. From each: its
    # area; how far it reaches outboard of the cell on the cell's own side, at the cell's height; and
    # how far it reaches above and below the cell, at the cell's distance from the centre plane.
    side = (sy >= 0.0).astype(np.int64)
    height = np.clip(((sz - DOMAIN_Z[0]) / RASTER_STEP).astype(np.int64), 0, context.y_out.shape[2] - 1)
    row = np.clip(((sy + DOMAIN_Y_HALF) / RASTER_STEP).astype(np.int64), 0, context.z_top.shape[1] - 1)
    section_columns: list[np.ndarray] = []
    for xq in (sx, *(sx + dx for dx in SECTION_DX)):
        i = np.clip(((xq - DOMAIN_X[0]) / RASTER_STEP).astype(np.int64), 0, context.nx - 1)
        section_columns += [
            context.section_area_x[i],
            context.y_out[side, i, height] - np.abs(sy),
            context.z_top[i, row] - sz,
            sz - context.z_bot[i, row],
        ]
    sections = np.column_stack(section_columns)

    features = np.concatenate([local, surface, sections], axis=1).astype(np.float32)
    assert features.shape == (len(centers), FEATURE_COLUMNS.total_dim)
    return features
