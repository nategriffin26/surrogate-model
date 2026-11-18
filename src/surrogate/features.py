"""Deterministic multi-scale geometric feature extraction on surface meshes.

Builds three geometric context blocks strictly from polygon coordinates, outward normals, and
areas:
  1. `local`    (10 cols): car-frame offsets (signed sideways), ground clearance, normal,
                           and normalized bounding-box coordinates.
  2. `regional` (60 cols): voxelized normal & area-density fields at each `VOXEL_SCALES` size
                           plus local curvature (`n_cell - n_voxel`) and ±h streamwise steps.
  3. `slices`  (121 cols): streamwise XZ buttock-line envelopes (±1.6 m stencil + explicit
                           on-body occupancy masks), neighboring Y-slice hops (±4, ±12, ±36 cm),
                           and XY waterline planform offsets for side-facing body panels.

Every lookup is laid out on the car as it is: its two sides are kept apart, so a part that
exists on one side only is read on that side only. The grids are in the file's own coordinates
(`DOMAIN_*`), which every DrivAerML run shares; a car placed elsewhere is caught by the coverage
check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates

from surrogate.surface import SurfaceMesh

RASTER_STEP = 0.02  # 2 cm grid for XZ buttock and XY waterline envelopes
DOMAIN_X = (-1.3, 4.7)
DOMAIN_Y_HALF = 1.2
DOMAIN_Z = (-0.45, 1.35)

# Shape that is close matters in detail and shape that is far only in bulk, so every set of
# distances is a ladder: each step a fixed multiple of the last.
DX_OWN = (-1.6, -0.8, -0.4, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6)  # doubling, 5 cm to a third of the car
DX_NEIGHBOR = (-0.3, 0.0, 0.3)
DY_NEIGHBOR = (-0.36, -0.12, -0.04, 0.04, 0.12, 0.36)  # x3 from two raster cells
DZ_NEIGHBOR = (-0.12, -0.04, 0.04, 0.12)
# About three times larger at each step, up to 75 cm. 3 cm is the smallest box that still holds
# ten or more mesh cells (2 to 10 mm across) on 98% of the surface; at 2 cm that falls to 75%.
VOXEL_SCALES = (0.03, 0.10, 0.25, 0.75)


@dataclass(frozen=True)
class FeatureColumns:
    """Column slices inside the concatenated 191-feature matrix."""

    local: slice = slice(0, 10)
    regional: slice = slice(10, 70)
    slices: slice = slice(70, 191)
    total_dim: int = 191


FEATURE_COLUMNS = FeatureColumns()


@dataclass(frozen=True)
class _VoxelScaleGrid:
    scale: float
    origin: np.ndarray      # (3,) float64
    area_sum: np.ndarray    # (Nx, Ny, Nz) float64
    normal_sum: tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass(frozen=True)
class SurfaceContext:
    """Precomputed whole-surface geometric lookup structures for one vehicle.

    Built once from every polygon of the file (~1.8 s for 8.8M on M5 Pro), then queried at any
    cells: the 120k training sample of a car, or all of them at inference.
    """

    bounds: tuple[float, float, float, float, float, float]
    y_center: float
    nx: int
    ny: int
    nz: int
    z_top: np.ndarray            # (nx, ny) upper envelope z(x, y), y from -DOMAIN_Y_HALF
    z_bot: np.ndarray            # (nx, ny) lower envelope z(x, y)
    xz_occupied: np.ndarray      # (nx, ny) bool
    y_out: np.ndarray            # (2, nx, nz) how far the body reaches from the centre plane at (x, z): [0] toward -y, [1] toward +y
    x_first_by_y: np.ndarray     # (ny,) leading-edge x of each y strip
    x_last_by_y: np.ndarray      # (ny,) trailing-edge x of each y strip
    z_peak_by_y: np.ndarray      # (ny,) max height along each y strip
    section_area_x: np.ndarray   # (nx,) cross-sectional area curve S(x) in m^2
    voxel_grids: tuple[_VoxelScaleGrid, ...]

    @classmethod
    def from_surface(cls, surface: SurfaceMesh) -> SurfaceContext:
        c = surface.centers
        n = surface.normals
        areas = surface.areas
        xmin, xmax, ymin, ymax, zmin, _ = surface.bounds
        y_center = 0.5 * (ymin + ymax)

        x = c[:, 0]
        y_rel = c[:, 1] - y_center
        z = c[:, 2]

        x0, x1 = DOMAIN_X
        z0, z1 = DOMAIN_Z
        h = RASTER_STEP
        nx = int(round((x1 - x0) / h))
        ny = 2 * int(round(DOMAIN_Y_HALF / h))
        nz = int(round((z1 - z0) / h))

        ix = np.clip(((x - x0) / h).astype(np.int64), 0, nx - 1)
        iy = np.clip(((y_rel + DOMAIN_Y_HALF) / h).astype(np.int64), 0, ny - 1)
        iz = np.clip(((z - z0) / h).astype(np.int64), 0, nz - 1)

        z_top_flat = np.full(nx * ny, -np.inf, dtype=np.float32)
        z_bot_flat = np.full(nx * ny, np.inf, dtype=np.float32)
        flat_xy = ix * ny + iy
        np.maximum.at(z_top_flat, flat_xy, z)
        np.minimum.at(z_bot_flat, flat_xy, z)

        occ_flat = np.isfinite(z_top_flat)
        z_top_flat[~occ_flat] = zmin
        z_bot_flat[~occ_flat] = zmin

        z_top = z_top_flat.reshape(nx, ny)
        z_bot = z_bot_flat.reshape(nx, ny)
        xz_occupied = occ_flat.reshape(nx, ny)

        side = (y_rel >= 0.0).astype(np.int64)
        y_out_flat = np.zeros(2 * nx * nz, dtype=np.float32)
        np.maximum.at(y_out_flat, (side * nx + ix) * nz + iz, np.abs(y_rel))
        y_out = y_out_flat.reshape(2, nx, nz)

        xs = x0 + h * np.arange(nx, dtype=np.float32)
        any_occ = xz_occupied.any(axis=0)
        x_first = np.where(any_occ, xs[np.argmax(xz_occupied, axis=0)], xmin).astype(np.float32)
        x_last = np.where(any_occ, xs[nx - 1 - np.argmax(xz_occupied[::-1], axis=0)], xmax).astype(np.float32)
        z_peak = z_top.max(axis=0).astype(np.float32)
        section_area_x = ((z_top - z_bot).sum(axis=1) * h).astype(np.float32)

        # Multi-scale 3D voxel grids, one per entry of VOXEL_SCALES
        voxel_grids: list[_VoxelScaleGrid] = []
        c_centered = c.copy()
        c_centered[:, 1] -= y_center

        for scale in VOXEL_SCALES:
            half_ny = int(np.ceil(DOMAIN_Y_HALF / scale))
            g_min = np.array([x0, -half_ny * scale, z0], dtype=np.float64)
            g_max = np.array([x1, half_ny * scale, z1], dtype=np.float64)
            shp = np.array(
                [
                    int(np.ceil((x1 - x0) / scale)) + 1,
                    2 * half_ny,
                    int(np.ceil((z1 - z0) / scale)) + 1,
                ],
                dtype=np.int64,
            )
            vi = np.clip(((c_centered - g_min) / scale).astype(np.int64), 0, shp - 1)
            flat_v = (vi[:, 0] * shp[1] + vi[:, 1]) * shp[2] + vi[:, 2]
            size = int(shp.prod())
            area_raw = np.bincount(flat_v, weights=areas, minlength=size).reshape(shp)
            norm_raw = [
                np.bincount(flat_v, weights=areas * n[:, k], minlength=size).reshape(shp)
                for k in range(3)
            ]
            voxel_grids.append(
                _VoxelScaleGrid(
                    scale=scale,
                    origin=g_min,
                    area_sum=area_raw,
                    normal_sum=(norm_raw[0], norm_raw[1], norm_raw[2]),
                )
            )

        return cls(
            bounds=surface.bounds,
            y_center=y_center,
            nx=nx,
            ny=ny,
            nz=nz,
            z_top=z_top,
            z_bot=z_bot,
            xz_occupied=xz_occupied,
            y_out=y_out,
            x_first_by_y=x_first,
            x_last_by_y=x_last,
            z_peak_by_y=z_peak,
            section_area_x=section_area_x,
            voxel_grids=tuple(voxel_grids),
        )

    def shape_descriptor(self) -> dict[str, float]:
        """Compact geometry-only vehicle summary for out-of-distribution range checks.

        Never passed into the per-cell neural network (which would memorize car IDs on
        N=10 training vehicles); used strictly by the coverage/OOD diagnostic.
        """
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
        length = xmax - xmin
        width = ymax - ymin
        height = zmax - zmin
        return {
            "length": float(length),
            "width": float(width),
            "height": float(height),
            "z_min": float(zmin),
            "max_section_area": float(self.section_area_x.max()),
            "plan_volume_proxy": float(self.section_area_x.sum() * RASTER_STEP),
        }


def extract_query_features(
    context: SurfaceContext,
    query_centers: np.ndarray,
    query_normals: np.ndarray,
) -> np.ndarray:
    """Extract the deterministic feature matrix `[M, FEATURE_COLUMNS.total_dim]` for `M` query cells."""
    xmin, xmax, ymin, ymax, zmin, zmax = context.bounds
    length = max(xmax - xmin, 1e-6)
    half_width = max(0.5 * (ymax - ymin), 1e-6)
    height = max(zmax - zmin, 1e-6)

    sx = query_centers[:, 0].astype(np.float32, copy=False)
    sy = (query_centers[:, 1] - context.y_center).astype(np.float32)
    sz = query_centers[:, 2].astype(np.float32, copy=False)
    sn = query_normals.astype(np.float32, copy=False)

    # 1. Local cell features (10 cols)
    f_local = np.stack(
        [
            sx - xmin,
            xmax - sx,
            sy,
            sz - zmin,
            sn[:, 0],
            sn[:, 1],
            sn[:, 2],
            (sx - xmin) / length,
            sy / half_width,
            (sz - zmin) / height,
        ],
        axis=1,
    )

    # 2. Isotropic 3D regional voxel features, 15 cols per scale in VOXEL_SCALES
    reg_cols: list[np.ndarray] = []
    for vg in context.voxel_grids:
        h_v = vg.scale

        def _lookup_voxel(dx_shift: float, vg: _VoxelScaleGrid = vg, h_v: float = h_v) -> tuple[np.ndarray, np.ndarray]:
            pts = np.stack([sx + dx_shift, sy, sz], axis=1)
            coords = ((pts - vg.origin) / h_v - 0.5).T
            dens = map_coordinates(vg.area_sum, coords, order=1, mode="nearest")
            vec = np.stack(
                [map_coordinates(ns, coords, order=1, mode="nearest") for ns in vg.normal_sum],
                axis=1,
            )
            vec = vec / np.maximum(dens, 1e-9)[:, None]
            return vec.astype(np.float32), (dens / (h_v * h_v)).astype(np.float32)

        v0, d0 = _lookup_voxel(0.0)
        vu, du = _lookup_voxel(-h_v)
        vd, dd = _lookup_voxel(h_v)
        reg_cols.extend([v0, d0[:, None], sn - v0, vu, du[:, None], vd, dd[:, None]])

    f_regional = np.concatenate(reg_cols, axis=1)

    # 3. Streamwise XZ buttock profiles + neighbor Y-hops + XY waterlines + on-body masks (121 cols)
    x0 = DOMAIN_X[0]
    z0 = DOMAIN_Z[0]
    h = RASTER_STEP

    def _y_row(yq: np.ndarray) -> np.ndarray:
        return np.clip(((yq + DOMAIN_Y_HALF) / h).astype(np.int64), 0, context.ny - 1)

    def _xz_lookup(xq: np.ndarray, yq: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        i = np.clip(((xq - x0) / h).astype(np.int64), 0, context.nx - 1)
        j = _y_row(yq)
        return context.z_top[i, j], context.z_bot[i, j], context.xz_occupied[i, j].astype(np.float32)

    # The plan-view outline on the cell's own side of the centre plane, as how far it lies outboard of the cell.
    side = (sy >= 0.0).astype(np.int64)
    sy_from_centre = np.abs(sy)

    def _xy_lookup(xq: np.ndarray, zq: np.ndarray) -> np.ndarray:
        i = np.clip(((xq - x0) / h).astype(np.int64), 0, context.nx - 1)
        k = np.clip(((zq - z0) / h).astype(np.int64), 0, context.nz - 1)
        return context.y_out[side, i, k] - sy_from_centre

    xz_own_cols: list[np.ndarray] = []
    own_masks: list[np.ndarray] = []

    zt0, zb0, m0 = _xz_lookup(sx, sy)
    xz_own_cols.extend([zt0 - sz, zb0 - sz])
    own_masks.append(m0)

    for dx in DX_OWN:
        zt_d, zb_d, m_d = _xz_lookup(sx + dx, sy)
        xz_own_cols.extend([zt_d - sz, zb_d - sz])
        own_masks.append(m_d)

    j_row = _y_row(sy)
    xz_own_cols.extend(
        [
            sx - context.x_first_by_y[j_row],
            context.x_last_by_y[j_row] - sx,
            context.z_peak_by_y[j_row] - sz,
        ]
    )

    xz_nb_cols: list[np.ndarray] = []
    nb_masks: list[np.ndarray] = []
    for dy in DY_NEIGHBOR:
        for dx in DX_NEIGHBOR:
            zt_n, zb_n, m_n = _xz_lookup(sx + dx, sy + dy)
            xz_nb_cols.extend([zt_n - sz, zb_n - sz])
            nb_masks.append(m_n)

    xy_cols: list[np.ndarray] = [_xy_lookup(sx, sz)]
    for dx in DX_OWN:
        xy_cols.append(_xy_lookup(sx + dx, sz))
    for dz in DZ_NEIGHBOR:
        for dx in DX_NEIGHBOR:
            xy_cols.append(_xy_lookup(sx + dx, sz + dz))

    f_slices = np.column_stack(xz_own_cols + own_masks + xz_nb_cols + nb_masks + xy_cols).astype(np.float32)

    features = np.concatenate([f_local, f_regional, f_slices], axis=1).astype(np.float32, copy=False)
    assert features.shape == (len(query_centers), FEATURE_COLUMNS.total_dim)
    return features
