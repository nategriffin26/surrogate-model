"""Reading a `.vtp` surface from its polygons alone (outward normals, areas, frontal area), and the
pressure drag and lift integrated over it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.ndimage import binary_fill_holes

TARGET_FIELD = "CpMeanTrim"
# The frontal area's silhouette grid step in metres, and the number of points it is drawn from.
FRONTAL_AREA_PIXEL = 0.005
FRONTAL_AREA_SAMPLES = 1_500_000


@dataclass(frozen=True)
class SurfaceMesh:
    """A surface file's polygons and the per-cell geometry the model and the force integrals read.

    `cp` holds the file's CFD `CpMeanTrim` when it carries it and `None` otherwise. Every other field
    depends only on the points and polygons.
    """

    points: np.ndarray        # (N_pts, 3) float32
    polygon_offsets: np.ndarray  # (N_cells + 1,) int64: polygon i's corners are corners[offsets[i]:offsets[i + 1]]
    polygon_corners: np.ndarray  # (N_corners,) int32 point numbers, in each polygon's order
    centers: np.ndarray       # (N_cells, 3) float32
    normals: np.ndarray       # (N_cells, 3) float32, guaranteed outward-pointing
    areas: np.ndarray         # (N_cells,) float64, polygon areas in m^2
    bounds: tuple[float, float, float, float, float, float]  # xmin, xmax, ymin, ymax, zmin, zmax
    frontal_area: float       # projected YZ silhouette area in m^2
    enclosed_volume: float    # |1/3 sum (c . n) dA| in m^3
    normals_flipped: bool     # True when raw VTP normals pointed inward
    cp: np.ndarray | None     # (N_cells,) float32 or None

    @property
    def n_cells(self) -> int:
        return int(self.centers.shape[0])

    @property
    def total_area(self) -> float:
        return float(self.areas.sum())

    @property
    def dimensions(self) -> tuple[float, float, float]:
        xmin, xmax, ymin, ymax, zmin, zmax = self.bounds
        return (xmax - xmin, ymax - ymin, zmax - zmin)


def estimate_frontal_area(mesh: pv.PolyData) -> float:
    """The car's frontal area A_ref: its silhouette seen along x, drawn on a FRONTAL_AREA_PIXEL grid from
    FRONTAL_AREA_SAMPLES points spread over the triangles by their projected area, with enclosed gaps filled."""
    pixel_size = FRONTAL_AREA_PIXEL
    tri = mesh if mesh.is_all_triangles else mesh.triangulate()
    faces = tri.regular_faces
    pts = np.asarray(tri.points, dtype=np.float32)
    v0, v1, v2 = pts[faces[:, 0]], pts[faces[:, 1]], pts[faces[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    proj_yz = 0.5 * np.abs(e1[:, 1] * e2[:, 2] - e1[:, 2] * e2[:, 1]).astype(np.float64)
    tot_proj = float(proj_yz.sum())
    if tot_proj <= 0.0:
        return 0.0

    rng = np.random.default_rng(0)
    n_draws = FRONTAL_AREA_SAMPLES
    idx = rng.choice(len(proj_yz), size=n_draws, replace=True, p=proj_yz / tot_proj)
    r1 = np.sqrt(rng.random(n_draws, dtype=np.float32))
    r2 = rng.random(n_draws, dtype=np.float32)
    w0, w1, w2 = 1.0 - r1, r1 * (1.0 - r2), r1 * r2
    py = w0 * v0[idx, 1] + w1 * v1[idx, 1] + w2 * v2[idx, 1]
    pz = w0 * v0[idx, 2] + w1 * v1[idx, 2] + w2 * v2[idx, 2]

    ymin, ymax = float(pts[:, 1].min()), float(pts[:, 1].max())
    zmin, zmax = float(pts[:, 2].min()), float(pts[:, 2].max())
    ny = int(np.ceil((ymax - ymin) / pixel_size)) + 3
    nz = int(np.ceil((zmax - zmin) / pixel_size)) + 3
    iy = np.clip(((py - ymin) / pixel_size).astype(np.int32) + 1, 0, ny - 1)
    iz = np.clip(((pz - zmin) / pixel_size).astype(np.int32) + 1, 0, nz - 1)
    grid = np.zeros((ny, nz), dtype=bool)
    grid[iy, iz] = True
    filled = binary_fill_holes(grid)
    return float(filled.sum() * (pixel_size * pixel_size))


def read_vtp_surface(path: str | Path, *, include_target: bool = True) -> SurfaceMesh:
    """Read a DrivAerML `boundary_i.vtp` surface file with a strict geometry boundary.

    - Preserves native `CellData` association (never averages cells onto vertices).
    - Orients normals outward via Gauss's divergence theorem (`1/3 sum(c · n dA)`),
      so drag and lift integrals carry consistent physical signs regardless of CAD winding.
    - Ignores all non-target solver fields (`pMeanTrim`, `wallShearStressMeanTrim`, etc.).
    """
    vtp_path = Path(path)
    if not vtp_path.is_file():
        raise FileNotFoundError(f"Surface VTP file not found: {vtp_path}")

    mesh = pv.read(vtp_path)
    if not isinstance(mesh, pv.PolyData):
        raise TypeError(f"Expected PolyData surface mesh in {vtp_path}, got {type(mesh).__name__}")
    if mesh.n_cells == 0 or mesh.n_points == 0:
        raise ValueError(f"Empty surface mesh in {vtp_path}")

    cp: np.ndarray | None = None
    if include_target and TARGET_FIELD in mesh.cell_data:
        cp = np.asarray(mesh.cell_data[TARGET_FIELD], dtype=np.float32).copy()
        if cp.shape != (mesh.n_cells,):
            raise ValueError(f"Expected {mesh.n_cells} cell values for {TARGET_FIELD}, got {cp.shape}")
        if not np.isfinite(cp).all():
            raise ValueError(f"Non-finite values in {TARGET_FIELD} for {vtp_path}")

    # Strip all cell/point/field arrays immediately so downstream geometry operations
    # cannot accidentally touch solver outputs.
    mesh.cell_data.clear()
    mesh.point_data.clear()
    mesh.field_data.clear()

    points = np.asarray(mesh.points, dtype=np.float32)
    polygon_offsets = np.asarray(mesh.face_offsets, dtype=np.int64)
    if len(polygon_offsets) != mesh.n_cells + 1:
        raise ValueError(f"{vtp_path} holds cells that are not polygons (lines or points); a surface is polygons only")
    polygon_corners = np.asarray(mesh.face_connectivity, dtype=np.int32)
    centers = np.asarray(mesh.cell_centers().points, dtype=np.float32)
    mesh_normals = mesh.compute_normals(
        cell_normals=True,
        point_normals=False,
        consistent_normals=False,
        auto_orient_normals=False,
        split_vertices=False,
    )
    normals = np.asarray(mesh_normals.cell_data["Normals"], dtype=np.float32)
    areas = np.asarray(
        mesh.compute_cell_sizes(length=False, volume=False).cell_data["Area"],
        dtype=np.float64,
    )
    areas = np.maximum(areas, 0.0)

    # Divergence theorem: 1/3 ∯ (x · n_out) dA = +Volume > 0.
    signed_volume = float(np.sum(np.einsum("ij,ij->i", centers, normals) * areas) / 3.0)
    normals_flipped = signed_volume < 0.0
    if normals_flipped:
        normals = -normals

    bounds = (
        float(centers[:, 0].min()),
        float(centers[:, 0].max()),
        float(centers[:, 1].min()),
        float(centers[:, 1].max()),
        float(centers[:, 2].min()),
        float(centers[:, 2].max()),
    )
    frontal_area = estimate_frontal_area(mesh)

    return SurfaceMesh(
        points=points,
        polygon_offsets=polygon_offsets,
        polygon_corners=polygon_corners,
        centers=centers,
        normals=normals,
        areas=areas,
        bounds=bounds,
        frontal_area=frontal_area,
        enclosed_volume=abs(signed_volume),
        normals_flipped=normals_flipped,
        cp=cp,
    )


def sample_cells_by_area(
    areas: np.ndarray,
    n_samples: int,
    *,
    seed: int,
) -> np.ndarray:
    """Draw cell indices with probability proportional to cell surface area dA."""
    total = float(areas.sum())
    if total <= 0.0:
        raise ValueError("Total surface area must be positive for area-weighted sampling")
    probs = areas / total
    rng = np.random.default_rng(seed)
    return rng.choice(len(areas), size=n_samples, replace=True, p=probs)


def compute_pressure_forces(
    cp: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray | None,
    *,
    total_area: float,
    ref_area: float,
) -> dict[str, float]:
    """Integrate pressure drag C_{d,p} and lift C_{l,p} with outward surface normals.

    Force from pressure on surface element dA with outward normal `n` is `-p n dA`.
    Dividing by freestream dynamic pressure `q_inf` and reference area `A_ref`:
      C_{d,p} = -(1 / A_ref) ∫ Cp * n_x dA
      C_{l,p} = -(1 / A_ref) ∫ Cp * n_z dA

    When `areas` is `None`, `cp` and `normals` are assumed to be an area-weighted sample
    from a surface of total area `total_area`, so the Monte Carlo estimator applies.
    """
    if ref_area <= 0.0:
        raise ValueError(f"ref_area must be positive, got {ref_area}")
    cp64 = np.asarray(cp, dtype=np.float64)
    nx = np.asarray(normals[:, 0], dtype=np.float64)
    nz = np.asarray(normals[:, 2], dtype=np.float64)

    if areas is not None:
        w = np.asarray(areas, dtype=np.float64)
        cd_p = -float(np.sum(cp64 * nx * w) / ref_area)
        cl_p = -float(np.sum(cp64 * nz * w) / ref_area)
    else:
        scale = total_area / ref_area
        cd_p = -float(np.mean(cp64 * nx) * scale)
        cl_p = -float(np.mean(cp64 * nz) * scale)

    return {"cd_p": cd_p, "cl_p": cl_p}
