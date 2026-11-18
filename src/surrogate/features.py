"""What the model is told about each cell of a car's surface, computed from the polygons alone.

Four blocks of inputs, in the order of `FEATURE_COLUMNS` (99 columns):

- `local` (10): where the cell is on the car (from the nose, the tail, the centre plane and the
  lowest point, and as fractions of the car's size) and its outward normal.
- `flow` (4): the potential flow around the car standing on the ground, in a uniform stream along
  +x: the velocity the car adds to the stream at the cell, and the squared speed of the flow along
  the surface |V_t|^2 (potential flow puts the pressure at Cp = 1 - |V_t|^2). It knows where the
  flow stops against the car and where it speeds up around a curve or under the floor; it knows
  nothing of separation.
- `surface` (15 per scale): the surface around the cell at each scale of `SURFACE_SCALES`: its mean
  normal, how much of it there is (1 on a flat panel), how the cell's own normal differs from that
  mean, and the mean normal and amount of the part ahead of the cell and of the part behind it.
- `outline` (25): how far the car reaches sideways beyond the cell at the cell's height, at the cell,
  at distances ahead of and behind it, and a little above and below.

`flow` and `surface` are computed the same way: a quantity carried by every cell (a source strength,
an area, an area times the normal) is spread onto a grid, convolved with a kernel (the potential of
a source for the flow, a Gaussian for the surface around a cell) and read back at each cell. The
grids are in metres, so the inputs do not depend on how finely the file is meshed, and every lookup
keeps the car's two sides apart. Grids are placed in the file's own coordinates, which every
DrivAerML run shares, with y measured from the car's centre plane and the ground at its lowest point.
The surface fields are kept only at the nodes a read at a cell touches, a few percent of their grids.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from scipy import fft
from scipy.ndimage import correlate1d, gaussian_filter1d, map_coordinates

from surrogate.surface import SurfaceMesh

# The plan-view outline and the cross-section area S(x) are read from a 2 cm raster of this box.
RASTER_STEP = 0.02
DOMAIN_X = (-1.3, 4.7)
DOMAIN_Y_HALF = 1.2
DOMAIN_Z = (-0.45, 1.35)
# Lengthwise distances double from 5 cm to a third of the car: shape close by matters in detail,
# shape far away only in bulk. Above and below the cell, the outline is read at three places.
OUTLINE_DX = (-1.6, -0.8, -0.4, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6)
OUTLINE_DZ = (-0.12, -0.04, 0.04, 0.12)
OUTLINE_DZ_DX = (-0.3, 0.0, 0.3)

# Potential flow on a 4 cm grid of 160 x 64 x 48 nodes above the ground (6.4 x 2.56 x 1.92 m, from
# DOMAIN_X[0]) and its mirror image below. Padded to twice that for the FFT, every size is a product
# of 2, 3 and 5. Four passes bring the wall condition to within a few percent. A 2 cm grid makes the
# flow closer to the exact one and the model no better.
FLOW_STEP = 0.04
FLOW_NODES = (160, 64, 48)
FLOW_PASSES = 4
FLOW_RELAXATION = 0.5

# The surface around a cell at four scales R, about three times apart: the surface blurred by a
# Gaussian of standard deviation 0.4 R, cut at three standard deviations, on a grid of step R / 3.
# The 3 cm scale's grid has a step of R / 2: at R / 3 it would hold 17 million nodes on a car, and
# the model is no better for it beyond the spread between training seeds.
SURFACE_SCALES = (0.03, 0.10, 0.25, 0.75)
SURFACE_STEPS = (0.015, 0.10 / 3.0, 0.25 / 3.0, 0.25)
SURFACE_SIGMA = 0.4
GAUSSIAN_TRUNCATE = 3.0

# Mean of 1/r over a cube of unit side about its centre: the potential of a source spread over its
# own grid node, so the kernel stays finite at r = 0.
CUBE_MEAN_INVERSE_DISTANCE = 2.3800772


@dataclass(frozen=True)
class FeatureColumns:
    """Column slices of the feature matrix, one per block, in order."""

    local: slice
    flow: slice
    surface: slice
    outline: slice

    @property
    def total_dim(self) -> int:
        return self.outline.stop


_SURFACE_END = 14 + 15 * len(SURFACE_SCALES)
FEATURE_COLUMNS = FeatureColumns(
    local=slice(0, 10),
    flow=slice(10, 14),
    surface=slice(14, _SURFACE_END),
    outline=slice(_SURFACE_END, _SURFACE_END + 1 + len(OUTLINE_DX) + len(OUTLINE_DZ) * len(OUTLINE_DZ_DX)),
)
BLOCKS = ("local", "flow", "surface", "outline")


@dataclass(frozen=True)
class Grid:
    """Nodes at origin + step * (i, j, k)."""

    origin: np.ndarray  # (3,) float64
    step: float
    nodes: tuple[int, int, int]

    def coordinates(self, points: np.ndarray) -> np.ndarray:
        """(3, M) positions of `points` in node units, as `map_coordinates` takes them."""
        return ((np.asarray(points, dtype=np.float64) - self.origin) / self.step).T

    def corners(self, points: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """For each of the eight nodes around every point: its flat node number and its trilinear weight."""
        q = self.coordinates(points).T
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

    def read(self, field: np.ndarray, points: np.ndarray) -> np.ndarray:
        """`field` at `points`, interpolated trilinearly."""
        return map_coordinates(field, self.coordinates(points), order=1, mode="nearest")


def _flow_grid(y_center: float, z_ground: float) -> Grid:
    """Nodes symmetric about the ground in z, so the mirror image is the grid flipped in z."""
    nx, ny, nz_above = FLOW_NODES
    h = FLOW_STEP
    origin = np.array([DOMAIN_X[0], y_center - 0.5 * (ny - 1) * h, z_ground - (nz_above - 0.5) * h])
    return Grid(origin, h, (nx, ny, 2 * nz_above))


def _source_potential_spectrum(grid: Grid) -> np.ndarray:
    """Spectrum of -1/(4 pi r), the potential of a unit source, on the grid padded to twice its size
    so that the FFT's circular convolution does not wrap. The kernel is even, so the spectrum is real."""
    padded = tuple(2 * n for n in grid.nodes)
    axes = [np.minimum(np.arange(p), p - np.arange(p)) * grid.step for p in padded]
    r = np.sqrt(axes[0][:, None, None] ** 2 + axes[1][None, :, None] ** 2 + axes[2][None, None, :] ** 2)
    r[0, 0, 0] = grid.step / CUBE_MEAN_INVERSE_DISTANCE
    return fft.rfftn((-1.0 / (4.0 * np.pi * r)).astype(np.float32), workers=-1).real.astype(np.float32)


def _potential(grid: Grid, spectrum: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Potential at the grid's nodes of `sources` (per node) and of their mirror image under the ground."""
    padded = tuple(2 * n for n in grid.nodes)
    spec = fft.rfftn((sources + sources[:, :, ::-1]).astype(np.float32), s=padded, workers=-1)
    spec *= spectrum
    nx, ny, nz = grid.nodes
    return fft.irfftn(spec, s=padded, workers=-1)[:nx, :ny, :nz]


def _induced_velocity(surface: SurfaceMesh, y_center: float) -> tuple[Grid, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Velocity the car adds to a unit stream along +x, at the flow grid's nodes.

    The car is a sheet of sources of density sigma on its surface. No flow through the wall needs
        sigma / 2 + n . u[sigma] = -n_x,
    u[sigma] being the velocity the rest of the sheet induces there. Starting from sigma = -2 n_x (the
    sheet as if alone), each pass moves sigma part of the way to what the wall condition asks given
    the current u. A source and the velocity read back at its cell share the same trilinear weights,
    so a cell's own source exerts no velocity on it.
    """
    grid = _flow_grid(y_center, surface.bounds[4])
    spectrum = _source_potential_spectrum(grid)
    centers = surface.centers.astype(np.float64)
    normals = surface.normals.astype(np.float64)
    sigma = -2.0 * normals[:, 0]
    for _ in range(FLOW_PASSES):
        phi = _potential(grid, spectrum, grid.spread(centers, sigma * surface.areas))
        u = np.stack([grid.read(d, centers) for d in np.gradient(phi, grid.step)], axis=1)
        sigma += FLOW_RELAXATION * (-2.0 * normals[:, 0] - 2.0 * np.einsum("ij,ij->i", u, normals) - sigma)
    phi = _potential(grid, spectrum, grid.spread(centers, sigma * surface.areas))
    ux, uy, uz = np.gradient(phi, grid.step)
    return grid, (ux, uy, uz)


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
    section_area_x: np.ndarray  # (nx,) cross-section area S(x) in m^2
    flow_grid: Grid
    induced_velocity: tuple[np.ndarray, np.ndarray, np.ndarray]  # x, y, z components at the flow grid's nodes
    surface_scales: tuple[SurfaceScale, ...]

    @classmethod
    def from_surface(cls, surface: SurfaceMesh) -> SurfaceContext:
        _, _, ymin, ymax, _, _ = surface.bounds
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

        # S(x): the height of the body over each (x, y) column of the raster, summed across y
        z_top = np.full(nx * ny, -np.inf, dtype=np.float32)
        z_bot = np.full(nx * ny, np.inf, dtype=np.float32)
        np.maximum.at(z_top, ix * ny + iy, z)
        np.minimum.at(z_bot, ix * ny + iy, z)
        column_height = np.where(np.isfinite(z_top), z_top - z_bot, 0.0).reshape(nx, ny)

        side = (y_rel >= 0.0).astype(np.int64)
        y_out = np.zeros(2 * nx * nz, dtype=np.float32)
        np.maximum.at(y_out, (side * nx + ix) * nz + iz, np.abs(y_rel))

        flow_grid, induced_velocity = _induced_velocity(surface, y_center)
        return cls(
            bounds=surface.bounds,
            y_center=y_center,
            nx=nx,
            y_out=y_out.reshape(2, nx, nz),
            section_area_x=(column_height.sum(axis=1) * h).astype(np.float32),
            flow_grid=flow_grid,
            induced_velocity=induced_velocity,
            surface_scales=tuple(_surface_scale(surface, r, h) for r, h in zip(SURFACE_SCALES, SURFACE_STEPS)),
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
    xmin, xmax, ymin, ymax, zmin, zmax = context.bounds
    centers = np.asarray(query_centers, dtype=np.float64)
    normals = np.asarray(query_normals, dtype=np.float64)
    sx = centers[:, 0]
    sy = centers[:, 1] - context.y_center
    sz = centers[:, 2]

    local = np.column_stack([
        sx - xmin, xmax - sx, sy, sz - zmin, normals,
        (sx - xmin) / (xmax - xmin), sy / (0.5 * (ymax - ymin)), (sz - zmin) / (zmax - zmin),
    ])

    u = np.stack([context.flow_grid.read(c, centers) for c in context.induced_velocity], axis=1)
    v = u + np.array([1.0, 0.0, 0.0])
    v_tangential = v - np.einsum("ij,ij->i", v, normals)[:, None] * normals
    flow = np.column_stack([u, (v_tangential ** 2).sum(axis=1)])

    surface_columns: list[np.ndarray] = []
    for s in context.surface_scales:
        whole, ahead = s.read(centers)
        mean_normal = whole[:, 1:] / np.maximum(whole[:, :1], 1e-12)
        surface_columns += [mean_normal, whole[:, :1] / s.flat_panel, normals - mean_normal]
        for part in (ahead, whole - ahead):
            surface_columns += [part[:, 1:] / np.maximum(part[:, :1], 1e-12), part[:, :1] / (0.5 * s.flat_panel)]
    surface = np.concatenate(surface_columns, axis=1)

    # How far the body reaches outboard of the cell on the cell's own side, at the cell's height
    side = (sy >= 0.0).astype(np.int64)
    y_from_centre = np.abs(sy)

    def outboard(xq: np.ndarray, zq: np.ndarray) -> np.ndarray:
        i = np.clip(((xq - DOMAIN_X[0]) / RASTER_STEP).astype(np.int64), 0, context.nx - 1)
        k = np.clip(((zq - DOMAIN_Z[0]) / RASTER_STEP).astype(np.int64), 0, context.y_out.shape[2] - 1)
        return context.y_out[side, i, k] - y_from_centre

    outline = np.column_stack(
        [outboard(sx, sz)]
        + [outboard(sx + dx, sz) for dx in OUTLINE_DX]
        + [outboard(sx + dx, sz + dz) for dz in OUTLINE_DZ for dx in OUTLINE_DZ_DX]
    )

    features = np.concatenate([local, flow, surface, outline], axis=1).astype(np.float32)
    assert features.shape == (len(centers), FEATURE_COLUMNS.total_dim)
    return features
