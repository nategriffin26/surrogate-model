/**
 * What the reference pipeline (`surface.py`) measures on a surface, computed in typed-array loops:
 * each polygon's centre, outward normal and area, the bounds, the frontal area, and whether the
 * file's polygons are wound inward. Centres, normals and areas follow VTK's definitions, since the
 * networks were trained on features built from VTK's values.
 */

/**
 * @typedef {Object} CellGeometry
 * @property {Float32Array} centers   (C, 3) VTK cell centres
 * @property {Float32Array} normals   (C, 3) unit, pointing out of the car
 * @property {Float64Array} areas     (C)
 * @property {number[]} bounds        [xmin, xmax, ymin, ymax, zmin, zmax] of the centres
 * @property {number} surfaceArea
 * @property {number} enclosedVolume
 * @property {boolean} flip           the file's polygons are wound inward
 */

/**
 * @param {import('./result.js').ParsedSurface} surface
 * @returns {CellGeometry}
 */
export function measureCells(surface) {
  const { points, connectivity, offsets } = surface;
  const cellCount = offsets.length - 1;
  const centers = new Float32Array(3 * cellCount);
  const normals = new Float32Array(3 * cellCount);
  const areas = new Float64Array(cellCount);
  let surfaceArea = 0;
  let signedVolume = 0;

  for (let c = 0; c < cellCount; c++) {
    const begin = offsets[c];
    const corners = offsets[c + 1] - begin;
    const a = 3 * connectivity[begin];
    const ax = points[a], ay = points[a + 1], az = points[a + 2];

    // VTK's definitions: the normal is the polygon's vector area (Newell's method, which the fan of
    // triangles from the first corner adds up to); the area is that vector's length for a quad and the
    // sum of the fan's triangle areas for any other polygon (vtkCellSizeFilter).
    let fanArea = 0;
    let nx = 0, ny = 0, nz = 0;
    let sx = ax, sy = ay, sz = az;
    for (let k = 1; k < corners; k++) {
      const b = 3 * connectivity[begin + k];
      sx += points[b];
      sy += points[b + 1];
      sz += points[b + 2];
      if (k === corners - 1) break;
      const d = 3 * connectivity[begin + k + 1];
      const ux = points[b] - ax, uy = points[b + 1] - ay, uz = points[b + 2] - az;
      const vx = points[d] - ax, vy = points[d + 1] - ay, vz = points[d + 2] - az;
      const cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
      nx += cx;
      ny += cy;
      nz += cz;
      fanArea += 0.5 * Math.sqrt(cx * cx + cy * cy + cz * cz);
    }
    const length = Math.sqrt(nx * nx + ny * ny + nz * nz);
    const area = corners === 4 ? 0.5 * length : fanArea;
    if (length > 0) {
      nx /= length;
      ny /= length;
      nz /= length;
    }

    let centerX, centerY, centerZ;
    if (corners <= 4) {
      centerX = sx / corners;
      centerY = sy / corners;
      centerZ = sz / corners;
    } else {
      // vtkPolygon's parametric centre: the middle of the corners' bounding rectangle in the frame
      // of the first edge and the normal.
      const b = 3 * connectivity[begin + 1];
      const ex = points[b] - ax, ey = points[b + 1] - ay, ez = points[b + 2] - az;
      const fx = ny * ez - nz * ey, fy = nz * ex - nx * ez, fz = nx * ey - ny * ex;
      const ee = ex * ex + ey * ey + ez * ez;
      const ff = fx * fx + fy * fy + fz * fz;
      let sMin = 0, sMax = 0, tMin = 0, tMax = 0;
      for (let k = 1; k < corners; k++) {
        const p = 3 * connectivity[begin + k];
        const px = points[p] - ax, py = points[p + 1] - ay, pz = points[p + 2] - az;
        const s = (px * ex + py * ey + pz * ez) / ee;
        const t = (px * fx + py * fy + pz * fz) / ff;
        if (s < sMin) sMin = s;
        if (s > sMax) sMax = s;
        if (t < tMin) tMin = t;
        if (t > tMax) tMax = t;
      }
      const s = 0.5 * (sMin + sMax);
      const t = 0.5 * (tMin + tMax);
      centerX = ax + s * ex + t * fx;
      centerY = ay + s * ey + t * fy;
      centerZ = az + s * ez + t * fz;
    }

    const i = 3 * c;
    centers[i] = centerX;
    centers[i + 1] = centerY;
    centers[i + 2] = centerZ;
    normals[i] = nx;
    normals[i + 1] = ny;
    normals[i + 2] = nz;
    areas[c] = area;
    surfaceArea += area;
    // Divergence theorem with the stored (float32) values, as the reference sums them.
    signedVolume += (centers[i] * normals[i] + centers[i + 1] * normals[i + 1] + centers[i + 2] * normals[i + 2]) * area;
  }

  const flip = signedVolume < 0;
  if (flip) for (let i = 0; i < normals.length; i++) normals[i] = -normals[i];
  return { centers, normals, areas, bounds: boundsOf(centers), surfaceArea, enclosedVolume: Math.abs(signedVolume) / 3, flip };
}

function boundsOf(xyz) {
  const bounds = [Infinity, -Infinity, Infinity, -Infinity, Infinity, -Infinity];
  for (let i = 0; i < xyz.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      const value = xyz[i + k];
      if (value < bounds[2 * k]) bounds[2 * k] = value;
      if (value > bounds[2 * k + 1]) bounds[2 * k + 1] = value;
    }
  }
  return bounds;
}

const SILHOUETTE_PIXEL = 0.005;

/**
 * Projected frontal area, the reference area for the coefficients: the car's outline seen from ahead,
 * drawn on a 5 mm grid by the pixels its points fall in, with enclosed gaps filled. The reference
 * (`estimate_frontal_area`) marks pixels with 1.5 million random samples instead; on DrivAerML cars
 * the two agree within 0.1%, and this one gives the same answer every time.
 * @param {Float32Array} points (P, 3)
 */
export function frontalArea(points) {
  const f32 = Math.fround;
  let ymin = Infinity, ymax = -Infinity, zmin = Infinity, zmax = -Infinity;
  for (let i = 0; i < points.length; i += 3) {
    const y = points[i + 1];
    const z = points[i + 2];
    if (y < ymin) ymin = y;
    if (y > ymax) ymax = y;
    if (z < zmin) zmin = z;
    if (z > zmax) zmax = z;
  }
  // One empty pixel on every side, so the outside of the outline is connected all round.
  const ny = Math.ceil((ymax - ymin) / SILHOUETTE_PIXEL) + 3;
  const nz = Math.ceil((zmax - zmin) / SILHOUETTE_PIXEL) + 3;
  const pixel = f32(SILHOUETTE_PIXEL);
  const covered = new Uint8Array(ny * nz);
  for (let i = 0; i < points.length; i += 3) {
    const iy = Math.min(Math.trunc(f32(f32(points[i + 1] - ymin) / pixel)) + 1, ny - 1);
    const iz = Math.min(Math.trunc(f32(f32(points[i + 2] - zmin) / pixel)) + 1, nz - 1);
    covered[iy * nz + iz] = 1;
  }

  // Flood the outside from the border through empty pixels (4-connected); everything else is the car.
  const outside = new Uint8Array(ny * nz);
  const queue = new Int32Array(ny * nz);
  let head = 0;
  let tail = 0;
  const visit = (y, z) => {
    const p = y * nz + z;
    if (!covered[p] && !outside[p]) {
      outside[p] = 1;
      queue[tail++] = p;
    }
  };
  for (let y = 0; y < ny; y++) {
    visit(y, 0);
    visit(y, nz - 1);
  }
  for (let z = 0; z < nz; z++) {
    visit(0, z);
    visit(ny - 1, z);
  }
  while (head < tail) {
    const p = queue[head++];
    const y = Math.floor(p / nz);
    const z = p - y * nz;
    if (y > 0) visit(y - 1, z);
    if (y < ny - 1) visit(y + 1, z);
    if (z > 0) visit(y, z - 1);
    if (z < nz - 1) visit(y, z + 1);
  }
  return (ny * nz - tail) * SILHOUETTE_PIXEL * SILHOUETTE_PIXEL;
}
