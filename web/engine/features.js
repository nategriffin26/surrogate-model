/**
 * The numbers the networks are told about a surface point, as `features.py` defines them: the point's
 * position and facing; how the surface bends around it at each voxel scale; the body's outline in the
 * point's own streamwise slice and in the slices beside it; the plan-view outline; and flags for
 * whether the body is there at each place looked at.
 *
 * The definition's constants (voxel scales, stencils, raster, domain) come from the model file, which
 * `export_web.py` writes from `features.py`, so a model trained on another definition brings its own.
 * Arithmetic follows the reference's float32 and float64 steps one for one (`Math.fround` where NumPy
 * rounds to float32), so grid cells are chosen exactly as in training and values agree to float32.
 */

const f32 = Math.fround;

/**
 * @typedef {Object} FeatureDefinition  as `export_web.py` writes it
 * @property {number} count
 * @property {number} rasterStep
 * @property {number[]} domainX
 * @property {number} domainYHalf
 * @property {number[]} domainZ
 * @property {number[]} dxOwn
 * @property {number[]} dxNeighbor
 * @property {number[]} dyNeighbor
 * @property {number[]} dzNeighbor
 * @property {number[]} voxelScales
 *
 * @typedef {FeatureDefinition & { h: number, x0: number, z0: number, nx: number, ny: number, nz: number,
 *   dxOwn32: number[], dxNeighbor32: number[], dyNeighbor32: number[], dzNeighbor32: number[] }} FeatureLayout
 */

/**
 * The definition with its float32 constants and raster sizes worked out once.
 * @param {FeatureDefinition} definition
 * @returns {FeatureLayout}
 */
export function featureLayout(definition) {
  const { rasterStep, domainX, domainYHalf, domainZ } = definition;
  const layout = {
    ...definition,
    h: f32(rasterStep),
    x0: f32(domainX[0]),
    z0: f32(domainZ[0]),
    nx: Math.round((domainX[1] - domainX[0]) / rasterStep),
    ny: Math.round(domainYHalf / rasterStep),
    nz: Math.round((domainZ[1] - domainZ[0]) / rasterStep),
    dxOwn32: definition.dxOwn.map(f32),
    dxNeighbor32: definition.dxNeighbor.map(f32),
    dyNeighbor32: definition.dyNeighbor.map(f32),
    dzNeighbor32: definition.dzNeighbor.map(f32),
  };
  const columns = describeColumns(definition).length;
  if (columns !== definition.count) {
    throw new Error(`The model file's feature definition gives ${columns} columns but says ${definition.count}.`);
  }
  return layout;
}

/**
 * For each column, in the networks' order: `kind`, a stable id of its family; `group`, the family's
 * display label; a short plain name; its unit; `offset`, the place it reads relative to the point,
 * [dx, dy, dz] in metres with dx downstream, dy outboard (away from the centre plane, on the point's
 * side) and dz up; and `along`, for a distance in metres, the unit direction it runs from that place,
 * so place + value × along is where it measured to (a body surface, or for the distances from nose,
 * tail, centre plane and lowest point that plane), null for every other column.
 * @param {FeatureDefinition} definition
 * @returns {{ kind: string, group: string, name: string, unit: string, offset: number[], along: number[] | null }[]}
 */
export function describeColumns({ voxelScales, dxOwn, dxNeighbor, dyNeighbor, dzNeighbor }) {
  const columns = [];
  const add = (kind, group, name, { offset = [0, 0, 0], along = null } = {}) => {
    columns.push({ kind, group, name, unit: along ? 'm' : '', offset, along });
  };
  const where = ([dx, dy, dz]) => {
    const parts = [];
    const say = (value, negative, positive) => {
      if (value) parts.push(`${Math.round(Math.abs(value) * 100)} cm ${value < 0 ? negative : positive}`);
    };
    say(dx, 'forward', 'rearward');
    say(dy, 'inboard', 'outboard');
    say(dz, 'down', 'up');
    return parts.length ? `, ${parts.join(', ')}` : '';
  };
  const forward = [-1, 0, 0], rearward = [1, 0, 0], inboard = [0, -1, 0], outboard = [0, 1, 0], down = [0, 0, -1], up = [0, 0, 1];

  const position = 'Position and facing';
  add('position', position, 'Distance from nose', { along: forward });
  add('position', position, 'Distance to tail', { along: rearward });
  add('position', position, 'Distance from centre plane', { along: inboard });
  add('position', position, 'Height above lowest point', { along: down });
  add('position', position, 'Facing rearward');
  add('position', position, 'Facing outboard');
  add('position', position, 'Facing up');
  add('position', position, 'Share of length from nose');
  add('position', position, 'Share of half-width from centre');
  add('position', position, 'Share of height from bottom');

  for (const scale of voxelScales) {
    const group = `Bend at ${Math.round(scale * 100)} cm`;
    const block = (offset) => {
      const label = where(offset);
      add('bend', group, `Mean facing rearward${label}`, { offset });
      add('bend', group, `Mean facing outboard${label}`, { offset });
      add('bend', group, `Mean facing up${label}`, { offset });
      add('bend', group, `Surface density${label}`, { offset });
    };
    block([0, 0, 0]);
    add('bend', group, 'Bend, rearward');
    add('bend', group, 'Bend, outboard');
    add('bend', group, 'Bend, up');
    block([-scale, 0, 0]);
    block([scale, 0, 0]);
  }

  // Tops and bottoms are signed heights of the body above the place read, so both run upward.
  const own = 'Own slice';
  add('slice', own, 'Body top above', { along: up });
  add('slice', own, 'Body bottom below', { along: up });
  for (const dx of dxOwn) {
    add('slice', own, `Body top above${where([dx, 0, 0])}`, { offset: [dx, 0, 0], along: up });
    add('slice', own, `Body bottom below${where([dx, 0, 0])}`, { offset: [dx, 0, 0], along: up });
  }
  add('slice', own, 'Distance from slice front', { along: forward });
  add('slice', own, 'Distance to slice rear', { along: rearward });
  add('slice', own, 'Slice peak above', { along: up });

  const flags = 'On-body flags';
  add('flag', flags, 'Body here');
  for (const dx of dxOwn) add('flag', flags, `Body${where([dx, 0, 0])}`, { offset: [dx, 0, 0] });

  const neighbours = 'Neighbour slices';
  for (const dy of dyNeighbor) {
    for (const dx of dxNeighbor) {
      add('neighbour', neighbours, `Body top above${where([dx, dy, 0])}`, { offset: [dx, dy, 0], along: up });
      add('neighbour', neighbours, `Body bottom below${where([dx, dy, 0])}`, { offset: [dx, dy, 0], along: up });
    }
  }
  for (const dy of dyNeighbor) {
    for (const dx of dxNeighbor) add('flag', flags, `Body${where([dx, dy, 0])}`, { offset: [dx, dy, 0] });
  }

  const outline = 'Plan-view outline';
  add('outline', outline, 'Outline outboard', { along: outboard });
  for (const dx of dxOwn) add('outline', outline, `Outline outboard${where([dx, 0, 0])}`, { offset: [dx, 0, 0], along: outboard });
  for (const dz of dzNeighbor) {
    for (const dx of dxNeighbor) add('outline', outline, `Outline outboard${where([dx, 0, dz])}`, { offset: [dx, 0, dz], along: outboard });
  }
  return columns;
}

/** Column kinds in the order the model view lists them. */
const COLUMN_KINDS = ['position', 'bend', 'slice', 'neighbour', 'outline', 'flag'];

/**
 * The kinds of the columns inside [start, stop) ranges, in `COLUMN_KINDS` order: what a model given
 * those columns was told about each point.
 * @param {{ kind: string }[]} columns  from `describeColumns`
 * @param {number[][]} ranges
 * @returns {string[]}
 */
export function kindsWithin(columns, ranges) {
  const present = new Set(ranges.flatMap(([start, stop]) => columns.slice(start, stop).map((column) => column.kind)));
  return COLUMN_KINDS.filter((kind) => present.has(kind));
}

/**
 * @typedef {Object} VoxelGrid  Area and area-weighted normal sums on a regular grid, mirrored across y = 0.
 *   Summed in float64 as the reference does, then kept as float32: a fine grid is tens of MB, every
 *   helper worker holds a copy, and the features read from it move by under 1e-6.
 * @property {number} scale
 * @property {number[]} origin   (3) float64
 * @property {number[]} shape    (3)
 * @property {Float32Array} area
 * @property {Float32Array} normalX
 * @property {Float32Array} normalY
 * @property {Float32Array} normalZ
 *
 * @typedef {Object} SurfaceContext  Whole-car lookups, built once from every polygon (`SurfaceContext`).
 * @property {FeatureLayout} layout
 * @property {number[]} bounds
 * @property {number} yCenter
 * @property {Float32Array} zTop        (nx, ny) highest centre in each (x, |y|) column
 * @property {Float32Array} zBottom     (nx, ny) lowest
 * @property {Float32Array} occupied    (nx, ny) 1 where the column holds any polygon
 * @property {Float32Array} yOut        (nx, nz) outermost |y| in each (x, z) column
 * @property {Float32Array} xFirst      (ny) leading x of each |y| strip
 * @property {Float32Array} xLast       (ny) trailing x
 * @property {Float32Array} zPeak       (ny) highest z
 * @property {Float32Array} sectionArea (nx) cross-section area S(x) in m²
 * @property {VoxelGrid[]} voxels
 */

/**
 * @param {import('./geometry.js').CellGeometry} cells
 * @param {FeatureLayout} layout
 * @returns {SurfaceContext}
 */
export function buildContext(cells, layout) {
  const { centers, normals, areas, bounds } = cells;
  const { h, x0, z0, nx, ny, nz } = layout;
  const [xmin, xmax, ymin, ymax, zmin] = bounds;
  const yCenter = 0.5 * (ymin + ymax);
  const yCenter32 = f32(yCenter);
  const cellCount = areas.length;

  const zTop = new Float32Array(nx * ny).fill(-Infinity);
  const zBottom = new Float32Array(nx * ny).fill(Infinity);
  const yOut = new Float32Array(nx * nz);
  const voxels = layout.voxelScales.map((scale) => voxelGrid(scale, layout));

  for (let c = 0; c < cellCount; c++) {
    const x = centers[3 * c];
    const yRelative = f32(centers[3 * c + 1] - yCenter32);
    const yAbs = Math.abs(yRelative);
    const z = centers[3 * c + 2];
    const ix = clamp(Math.trunc(f32(f32(x - x0) / h)), nx);
    const iy = clamp(Math.trunc(f32(yAbs / h)), ny);
    const iz = clamp(Math.trunc(f32(f32(z - z0) / h)), nz);
    const column = ix * ny + iy;
    if (z > zTop[column]) zTop[column] = z;
    if (z < zBottom[column]) zBottom[column] = z;
    if (yAbs > yOut[ix * nz + iz]) yOut[ix * nz + iz] = yAbs;

    const area = areas[c];
    for (const grid of voxels) {
      const [ox, oy, oz] = grid.origin;
      const [sx, sy, sz] = grid.shape;
      const vx = clamp(Math.trunc((x - ox) / grid.scale), sx);
      const vy = clamp(Math.trunc((yRelative - oy) / grid.scale), sy);
      const vz = clamp(Math.trunc((z - oz) / grid.scale), sz);
      const v = (vx * sy + vy) * sz + vz;
      grid.area[v] += area;
      grid.normalX[v] += area * normals[3 * c];
      grid.normalY[v] += area * normals[3 * c + 1];
      grid.normalZ[v] += area * normals[3 * c + 2];
    }
  }

  const occupied = new Float32Array(nx * ny);
  for (let i = 0; i < nx * ny; i++) {
    if (zTop[i] !== -Infinity) {
      occupied[i] = 1;
    } else {
      zTop[i] = zmin;
      zBottom[i] = zmin;
    }
  }

  const xFirst = new Float32Array(ny);
  const xLast = new Float32Array(ny);
  const zPeak = new Float32Array(ny);
  for (let j = 0; j < ny; j++) {
    let first = -1;
    let last = -1;
    let peak = -Infinity;
    for (let i = 0; i < nx; i++) {
      if (occupied[i * ny + j]) {
        if (first < 0) first = i;
        last = i;
      }
      if (zTop[i * ny + j] > peak) peak = zTop[i * ny + j];
    }
    xFirst[j] = first >= 0 ? stationX(layout, first) : xmin;
    xLast[j] = last >= 0 ? stationX(layout, last) : xmax;
    zPeak[j] = peak;
  }

  const sectionArea = new Float32Array(nx);
  for (let i = 0; i < nx; i++) {
    let sum = 0;
    for (let j = 0; j < ny; j++) sum += f32(zTop[i * ny + j] - zBottom[i * ny + j]);
    sectionArea[i] = f32(f32(f32(sum) * h) * 2);
  }

  for (const grid of voxels) {
    mirror(grid);
    for (const field of ['area', 'normalX', 'normalY', 'normalZ']) grid[field] = new Float32Array(grid[field]);
  }
  return { layout, bounds, yCenter, zTop, zBottom, occupied, yOut, xFirst, xLast, zPeak, sectionArea, voxels };
}

/** x of raster station i, as the reference's float32 `x0 + h * arange(nx)`. */
export function stationX({ x0, h }, i) {
  return f32(x0 + f32(h * i));
}

function voxelGrid(scale, { domainX, domainYHalf, domainZ }) {
  const halfY = Math.ceil(domainYHalf / scale);
  const origin = [domainX[0], -halfY * scale, domainZ[0]];
  const shape = [
    Math.ceil((domainX[1] - domainX[0]) / scale) + 1,
    2 * halfY,
    Math.ceil((domainZ[1] - domainZ[0]) / scale) + 1,
  ];
  const size = shape[0] * shape[1] * shape[2];
  return {
    scale, origin, shape,
    area: new Float64Array(size), normalX: new Float64Array(size), normalY: new Float64Array(size), normalZ: new Float64Array(size),
  };
}

/** Average each grid with its mirror image across y = 0 (the lateral normal changes sign). */
function mirror(grid) {
  const [sx, sy, sz] = grid.shape;
  for (let x = 0; x < sx; x++) {
    for (let y = 0; y < sy / 2; y++) {
      for (let z = 0; z < sz; z++) {
        const a = (x * sy + y) * sz + z;
        const b = (x * sy + (sy - 1 - y)) * sz + z;
        for (const [field, sign] of [[grid.area, 1], [grid.normalX, 1], [grid.normalY, -1], [grid.normalZ, 1]]) {
          const first = field[a];
          const second = field[b];
          field[a] = 0.5 * (first + sign * second);
          field[b] = 0.5 * (second + sign * first);
        }
      }
    }
  }
}

function clamp(index, size) {
  return index < 0 ? 0 : index >= size ? size - 1 : index;
}

/**
 * Write the features of points [first, first + count) into `out`, one row of `stride` floats per
 * point (`stride` at least the feature count; the rest of a row is left as it is).
 * @param {SurfaceContext} context
 * @param {Float32Array} positions (P, 3)
 * @param {Float32Array} normals   (P, 3) unit, outward
 * @param {Float32Array} out       (count, stride)
 */
export function computeFeatures(context, positions, normals, first, count, out, stride = context.layout.count) {
  const { layout, zTop, zBottom, occupied, yOut, xFirst, xLast, zPeak, voxels } = context;
  const { h, x0, z0, nx, ny, nz, dxOwn32, dxNeighbor32, dyNeighbor32, dzNeighbor32 } = layout;
  const [xmin, xmax, ymin, ymax, zmin, zmax] = context.bounds;
  const xmin32 = f32(xmin), xmax32 = f32(xmax), zmin32 = f32(zmin);
  const length = f32(Math.max(xmax - xmin, 1e-6));
  const halfWidth = f32(Math.max(0.5 * (ymax - ymin), 1e-6));
  const height = f32(Math.max(zmax - zmin, 1e-6));
  const yCenter32 = f32(context.yCenter);
  const columnX = (x) => clamp(Math.trunc(f32(f32(x - x0) / h)), nx);
  const sampled = new Float64Array(4);

  for (let p = 0; p < count; p++) {
    const q = 3 * (first + p);
    const sx = positions[q];
    const yRelative = f32(positions[q + 1] - yCenter32);
    const sya = Math.abs(yRelative);
    const ySign = yRelative >= 0 ? 1 : -1;
    const sz = positions[q + 2];
    const n0 = normals[q];
    const n1 = f32(normals[q + 1] * ySign);
    const n2 = normals[q + 2];
    let k = p * stride;

    // Position and facing.
    const fromNose = f32(sx - xmin32);
    const fromBottom = f32(sz - zmin32);
    out[k++] = fromNose;
    out[k++] = xmax32 - sx;
    out[k++] = sya;
    out[k++] = fromBottom;
    out[k++] = n0;
    out[k++] = n1;
    out[k++] = n2;
    out[k++] = fromNose / length;
    out[k++] = sya / halfWidth;
    out[k++] = fromBottom / height;

    // Bend at each voxel scale: mean facing and surface density here, upstream and downstream.
    for (const grid of voxels) {
      const voxelFace = grid.scale * grid.scale;
      for (const dx of [0, -grid.scale, grid.scale]) {
        const x = dx === 0 ? sx : f32(sx + f32(dx));
        sampleGrid(grid, x, yRelative, sz, sampled);
        const area = Math.max(sampled[0], 1e-9);
        const vx = f32(sampled[1] / area);
        const vy = f32((sampled[2] / area) * ySign);
        const vz = f32(sampled[3] / area);
        out[k++] = vx;
        out[k++] = vy;
        out[k++] = vz;
        out[k++] = sampled[0] / voxelFace;
        if (dx === 0) {
          out[k++] = n0 - vx;
          out[k++] = n1 - vy;
          out[k++] = n2 - vz;
        }
      }
    }

    // Own slice: the body's top and bottom here and along the stencil, then the slice's extent.
    const ownTop = k;
    const ownFlags = ownTop + 5 + 2 * dxOwn32.length;
    let j = clamp(Math.trunc(f32(sya / h)), ny);
    let i = columnX(sx);
    out[ownTop] = zTop[i * ny + j] - sz;
    out[ownTop + 1] = zBottom[i * ny + j] - sz;
    out[ownFlags] = occupied[i * ny + j];
    for (let d = 0; d < dxOwn32.length; d++) {
      i = columnX(f32(sx + dxOwn32[d]));
      out[ownTop + 2 + 2 * d] = zTop[i * ny + j] - sz;
      out[ownTop + 3 + 2 * d] = zBottom[i * ny + j] - sz;
      out[ownFlags + 1 + d] = occupied[i * ny + j];
    }
    const extent = ownTop + 2 + 2 * dxOwn32.length;
    out[extent] = sx - xFirst[j];
    out[extent + 1] = xLast[j] - sx;
    out[extent + 2] = zPeak[j] - sz;

    // Neighbour slices.
    const neighbourTop = ownFlags + 1 + dxOwn32.length;
    const neighbourFlags = neighbourTop + 2 * dyNeighbor32.length * dxNeighbor32.length;
    let n = 0;
    for (let e = 0; e < dyNeighbor32.length; e++) {
      j = clamp(Math.trunc(f32(Math.abs(f32(sya + dyNeighbor32[e])) / h)), ny);
      for (let d = 0; d < dxNeighbor32.length; d++, n++) {
        i = columnX(f32(sx + dxNeighbor32[d]));
        out[neighbourTop + 2 * n] = zTop[i * ny + j] - sz;
        out[neighbourTop + 2 * n + 1] = zBottom[i * ny + j] - sz;
        out[neighbourFlags + n] = occupied[i * ny + j];
      }
    }

    // Plan-view outline: how far outboard the body reaches at the point's height, along the stencil.
    let o = neighbourFlags + n;
    let kz = clamp(Math.trunc(f32(f32(sz - z0) / h)), nz);
    out[o++] = yOut[columnX(sx) * nz + kz] - sya;
    for (let d = 0; d < dxOwn32.length; d++) out[o++] = yOut[columnX(f32(sx + dxOwn32[d])) * nz + kz] - sya;
    for (let e = 0; e < dzNeighbor32.length; e++) {
      kz = clamp(Math.trunc(f32(f32(f32(sz + dzNeighbor32[e]) - z0) / h)), nz);
      for (let d = 0; d < dxNeighbor32.length; d++) out[o++] = yOut[columnX(f32(sx + dxNeighbor32[d])) * nz + kz] - sya;
    }
  }
}

/**
 * Trilinear interpolation of a voxel grid's four fields at (x, y, z), cell-centred, with coordinates
 * clamped to the grid as SciPy's `map_coordinates(order=1, mode='nearest')` does.
 */
function sampleGrid(grid, x, y, z, out) {
  const [sx, sy, sz] = grid.shape;
  const [ox, oy, oz] = grid.origin;
  const h = grid.scale;
  const cx = Math.min(Math.max((x - ox) / h - 0.5, 0), sx - 1);
  const cy = Math.min(Math.max((y - oy) / h - 0.5, 0), sy - 1);
  const cz = Math.min(Math.max((z - oz) / h - 0.5, 0), sz - 1);
  const x0 = Math.floor(cx), y0 = Math.floor(cy), z0 = Math.floor(cz);
  const tx = cx - x0, ty = cy - y0, tz = cz - z0;
  const x1 = Math.min(x0 + 1, sx - 1), y1 = Math.min(y0 + 1, sy - 1), z1 = Math.min(z0 + 1, sz - 1);
  out[0] = out[1] = out[2] = out[3] = 0;
  for (let corner = 0; corner < 8; corner++) {
    const ix = corner & 4 ? x1 : x0;
    const iy = corner & 2 ? y1 : y0;
    const iz = corner & 1 ? z1 : z0;
    const w = (corner & 4 ? tx : 1 - tx) * (corner & 2 ? ty : 1 - ty) * (corner & 1 ? tz : 1 - tz);
    const v = (ix * sy + iy) * sz + iz;
    out[0] += w * grid.area[v];
    out[1] += w * grid.normalX[v];
    out[2] += w * grid.normalY[v];
    out[3] += w * grid.normalZ[v];
  }
}
