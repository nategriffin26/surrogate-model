/**
 * The numbers the networks are told about a surface point, as `features.py` defines them, read from
 * a car's lookups (`lookups.js`): where the point is on the car and which way it faces; the flow
 * around the car standing on the ground, as a sheet of sources on its surface satisfying the wall
 * condition (the velocity the car adds to the stream, and the squared speed along the surface); the
 * surface around the point at four scales, whole and split into the part ahead and the part behind;
 * and how far the car reaches sideways beyond the point, at its height, ahead, behind, above and
 * below. Nothing is folded across the centre plane: a point is read on its own side.
 *
 * Axes are the file's: x downstream, z up, and so (right-handed) +y toward the car's right as seen
 * from the driver's seat.
 *
 * The definition's constants come from the model file, which `export_web.py` writes from
 * `features.py`, so a model trained on another definition brings its own. Arithmetic follows the
 * reference's float32 and float64 steps (`Math.fround` where NumPy rounds to float32), so raster
 * bins are chosen as in training and values agree to float32.
 */

import { gaussianWeights, gradientAt, locate, upstreamHalf } from './grid.js';

const f32 = Math.fround;

/**
 * @typedef {Object} FeatureDefinition  as `export_web.py` writes it
 * @property {number} count
 * @property {number} rasterStep
 * @property {number[]} domainX
 * @property {number[]} domainZ
 * @property {number[]} outlineDx
 * @property {number[]} outlineDz
 * @property {number[]} outlineDzDx
 * @property {number} flowStep
 * @property {number[]} flowNodes      (nx, ny, nz) above the ground; the grid holds their mirror image below too
 * @property {number} flowPasses
 * @property {number} flowRelaxation
 * @property {number[]} surfaceScales
 * @property {number[]} surfaceSteps
 * @property {number} surfaceSigma     the Gaussian's standard deviation, as a share of the scale
 * @property {number} gaussianTruncate the Gaussian's reach, in standard deviations
 * @property {number} cubeMeanInverseDistance
 *
 * @typedef {Object} ScaleLayout  One surface scale's grid step and Gaussian.
 * @property {number} scale
 * @property {number} step
 * @property {number} radius          the Gaussian's reach in nodes
 * @property {Float64Array} weights   the normalised Gaussian at offsets −radius … radius
 * @property {Float64Array} upstream  the same kept upstream: nodes at smaller x fully, the node itself by half
 * @property {number} flatPanel       the blurred area a flat panel through the point reads, on average over
 *   where it falls between nodes
 *
 * @typedef {FeatureDefinition & { h: number, x0: number, z0: number, nx: number, nz: number,
 *   flow: { x0: number, step: number, nodes: number[], passes: number, relaxation: number, cubeMeanInverseDistance: number },
 *   scales: ScaleLayout[] }} FeatureLayout
 *   The outline raster's origin (x0, z0) and step h in float32, as the raster is built, and its size
 *   (nx, nz); the flow grid with its mirror image (nodes[2] twice the definition's); each scale's grid.
 */

/**
 * The definition with its float32 constants, grid sizes and Gaussians worked out once.
 * @param {FeatureDefinition} definition
 * @returns {FeatureLayout}
 */
export function featureLayout(definition) {
  const { rasterStep, domainX, domainZ, flowNodes, surfaceSigma, gaussianTruncate } = definition;
  const layout = {
    ...definition,
    h: f32(rasterStep),
    x0: f32(domainX[0]),
    z0: f32(domainZ[0]),
    nx: Math.round((domainX[1] - domainX[0]) / rasterStep),
    nz: Math.round((domainZ[1] - domainZ[0]) / rasterStep),
    flow: {
      x0: domainX[0],
      step: definition.flowStep,
      nodes: [flowNodes[0], flowNodes[1], 2 * flowNodes[2]],
      passes: definition.flowPasses,
      relaxation: definition.flowRelaxation,
      cubeMeanInverseDistance: definition.cubeMeanInverseDistance,
    },
    scales: definition.surfaceScales.map((scale, s) => {
      const step = definition.surfaceSteps[s];
      const weights = gaussianWeights((surfaceSigma * scale) / step, gaussianTruncate);
      const radius = (weights.length - 1) / 2;
      // A flat panel lays step² of area per column of nodes. Across the panel the spread and the read
      // each share it between the two nodes either side; averaged over where the panel falls between
      // them, the node itself counts 2/3 and each neighbour 1/6.
      const flatPanel = step * step * ((2.0 / 3.0) * weights[radius] + (1.0 / 3.0) * weights[radius + 1]);
      return { scale, step, radius, weights, upstream: upstreamHalf(weights), flatPanel };
    }),
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
 * [dx, dy, dz] in metres along the file's axes (dx downstream, dy toward +y, dz up); and `along`, for
 * a distance in metres, the unit direction it runs from that place, so place + value × along is where
 * it measured to (a body surface, or for the distances from nose, tail, centre plane and lowest point
 * that plane), null for every other column. The plan-view outline is measured outboard, away from the
 * centre plane on the point's own side: its `along` is [0, 1, 0] for a point on the +y side and is
 * turned to [0, −1, 0] for a point on the −y side. A surface column also has `scale`, the size it is
 * named by, `sigma`, its Gaussian's standard deviation (both in metres), and `part`: the whole surface
 * about the point, the part ahead of it, or the part behind.
 * @param {FeatureDefinition} definition
 * @returns {import('./result.js').ModelColumn[]}
 */
export function describeColumns({ surfaceScales, surfaceSigma, outlineDx, outlineDz, outlineDzDx }) {
  const columns = [];
  const add = (kind, group, name, { offset = [0, 0, 0], along = null, ...more } = {}) => {
    columns.push({ kind, group, name, unit: along ? 'm' : '', offset, along, ...more });
  };
  const where = ([dx, dy, dz]) => {
    const parts = [];
    const say = (value, negative, positive) => {
      if (value) parts.push(`${Math.round(Math.abs(value) * 100)} cm ${value < 0 ? negative : positive}`);
    };
    say(dx, 'forward', 'rearward');
    say(dy, 'toward −y', 'toward +y');
    say(dz, 'down', 'up');
    return parts.length ? `, ${parts.join(', ')}` : '';
  };
  const forward = [-1, 0, 0], rearward = [1, 0, 0], towardMinusY = [0, -1, 0], outboard = [0, 1, 0], down = [0, 0, -1];

  const position = 'Position and facing';
  add('position', position, 'Distance from nose', { along: forward });
  add('position', position, 'Distance to tail', { along: rearward });
  // Signed, so a point at y below the centre plane reads a negative value and still reaches the plane.
  add('position', position, 'y from centre plane', { along: towardMinusY });
  add('position', position, 'Height above lowest point', { along: down });
  add('position', position, 'Facing rearward');
  add('position', position, 'Facing +y');
  add('position', position, 'Facing up');
  add('position', position, 'Share of length from nose');
  add('position', position, 'y from centre plane, share of half-width');
  add('position', position, 'Share of height from bottom');

  const flow = 'Potential flow';
  add('flow', flow, 'Induced velocity, rearward');
  add('flow', flow, 'Induced velocity, +y');
  add('flow', flow, 'Induced velocity, up');
  add('flow', flow, 'Speed along the surface, squared');

  for (const scale of surfaceScales) {
    const group = `Surface at ${Math.round(scale * 100)} cm`;
    const surface = (name, part) => add('surface', group, name, { scale, sigma: surfaceSigma * scale, part });
    const facing = (part, label) => {
      surface(`Mean facing rearward${label}`, part);
      surface(`Mean facing +y${label}`, part);
      surface(`Mean facing up${label}`, part);
      surface(`Surface density${label}`, part);
    };
    facing('whole', '');
    surface('Bend, rearward', 'whole');
    surface('Bend, +y', 'whole');
    surface('Bend, up', 'whole');
    facing('ahead', ', ahead');
    facing('behind', ', behind');
  }

  const outline = 'Plan-view outline';
  add('outline', outline, 'Outline outboard', { along: outboard });
  for (const dx of outlineDx) add('outline', outline, `Outline outboard${where([dx, 0, 0])}`, { offset: [dx, 0, 0], along: outboard });
  for (const dz of outlineDz) {
    for (const dx of outlineDzDx) add('outline', outline, `Outline outboard${where([dx, 0, dz])}`, { offset: [dx, 0, dz], along: outboard });
  }
  return columns;
}

/** Column kinds in the order the model view lists them. */
const COLUMN_KINDS = ['position', 'flow', 'surface', 'outline'];

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
 * @typedef {import('./lookups.js').SurfaceLookups & { layout: FeatureLayout }} SurfaceContext
 *   A car's lookups with the layout of the definition they were built for.
 */

const CUBE = new Float64Array(6);
const VELOCITY = new Float64Array(3);
const WHOLE = new Float64Array(4);
const AHEAD = new Float64Array(4);

/**
 * Write the features of points [first, first + count) into `out`, one row of `stride` floats per
 * point (`stride` at least the feature count; the rest of a row is left as it is).
 * @param {SurfaceContext} context
 * @param {Float32Array} positions (P, 3)
 * @param {Float32Array} normals   (P, 3) unit, outward
 * @param {Float32Array} out       (count, stride)
 */
export function computeFeatures(context, positions, normals, first, count, out, stride = context.layout.count) {
  const { layout, yOut, flow, scales } = context;
  const { rasterStep, domainX, domainZ, nx, nz, outlineDx, outlineDz, outlineDzDx } = layout;
  const [xmin, xmax, ymin, ymax, zmin, zmax] = context.bounds;
  const yCenter = context.yCenter;
  const flowGrid = { origin: flow.origin, step: layout.flow.step, nodes: layout.flow.nodes };
  const scaleGrids = scales.map(({ origin, nodes }, s) => ({ origin, nodes, step: layout.scales[s].step }));
  const bin = (value, low, size) => clamp(Math.trunc((value - low) / rasterStep), size);

  for (let p = 0; p < count; p++) {
    const q = 3 * (first + p);
    const sx = positions[q];
    const sy = positions[q + 1] - yCenter;
    const sz = positions[q + 2];
    const n0 = normals[q], n1 = normals[q + 1], n2 = normals[q + 2];
    let k = p * stride;

    // Position and facing.
    out[k++] = sx - xmin;
    out[k++] = xmax - sx;
    out[k++] = sy;
    out[k++] = sz - zmin;
    out[k++] = n0;
    out[k++] = n1;
    out[k++] = n2;
    out[k++] = (sx - xmin) / (xmax - xmin);
    out[k++] = sy / (0.5 * (ymax - ymin));
    out[k++] = (sz - zmin) / (zmax - zmin);

    // The velocity the car adds to the stream, and the squared speed of the whole flow along the surface.
    flowAtPoint(flowGrid, flow.potential, sx, positions[q + 1], sz, n0, n1, n2, out, k);
    k += 4;

    // The surface around the point at each scale: its mean facing and density, the point's own facing
    // against that mean, then the same for the part ahead and the part behind.
    for (let s = 0; s < scales.length; s++) {
      readScale(scaleGrids[s], scales[s], sx, positions[q + 1], sz);
      const flat = layout.scales[s].flatPanel;
      const whole = Math.max(WHOLE[0], 1e-12);
      const mx = WHOLE[1] / whole, my = WHOLE[2] / whole, mz = WHOLE[3] / whole;
      out[k++] = mx;
      out[k++] = my;
      out[k++] = mz;
      out[k++] = WHOLE[0] / flat;
      out[k++] = n0 - mx;
      out[k++] = n1 - my;
      out[k++] = n2 - mz;
      const ahead = Math.max(AHEAD[0], 1e-12);
      out[k++] = AHEAD[1] / ahead;
      out[k++] = AHEAD[2] / ahead;
      out[k++] = AHEAD[3] / ahead;
      out[k++] = AHEAD[0] / (0.5 * flat);
      const behindArea = WHOLE[0] - AHEAD[0];
      const behind = Math.max(behindArea, 1e-12);
      out[k++] = (WHOLE[1] - AHEAD[1]) / behind;
      out[k++] = (WHOLE[2] - AHEAD[2]) / behind;
      out[k++] = (WHOLE[3] - AHEAD[3]) / behind;
      out[k++] = behindArea / (0.5 * flat);
    }

    // Plan-view outline: how far beyond the point the body reaches away from the centre plane, on the
    // point's own side and at its height, along the stencil. Bins are found in float64, as the
    // reference finds them for a point it asks about.
    const side = sy >= 0 ? nx : 0;
    const fromCentre = Math.abs(sy);
    const reach = (x, z) => yOut[(side + bin(x, domainX[0], nx)) * nz + bin(z, domainZ[0], nz)] - fromCentre;
    out[k++] = reach(sx, sz);
    for (const dx of outlineDx) out[k++] = reach(sx + dx, sz);
    for (const dz of outlineDz) {
      for (const dx of outlineDzDx) out[k++] = reach(sx + dx, sz + dz);
    }
  }
}

/**
 * The flow inputs at points with their normals: the velocity the car adds to the stream along +x,
 * in units of the stream's speed, and the squared speed of the whole flow along the surface, so
 * 1 − |V_t|² is potential flow's pressure coefficient.
 * @param {SurfaceContext} context
 * @param {Float32Array} positions (P, 3)
 * @param {Float32Array} normals   (P, 3) unit, outward
 * @returns {{ induced: Float32Array, speedSquared: Float32Array }}  (P, 3) and (P)
 */
export function flowAt({ layout, flow }, positions, normals) {
  const grid = { origin: flow.origin, step: layout.flow.step, nodes: layout.flow.nodes };
  const count = positions.length / 3;
  const induced = new Float32Array(3 * count);
  const speedSquared = new Float32Array(count);
  const row = new Float32Array(4);
  for (let p = 0; p < count; p++) {
    const q = 3 * p;
    flowAtPoint(grid, flow.potential, positions[q], positions[q + 1], positions[q + 2], normals[q], normals[q + 1], normals[q + 2], row, 0);
    induced.set(row.subarray(0, 3), q);
    speedSquared[p] = row[3];
  }
  return { induced, speedSquared };
}

/** The four flow inputs at (x, y, z) with normal n into out[at .. at + 4]: u, then |V_t|² of V = u + (1, 0, 0). */
function flowAtPoint(grid, potential, x, y, z, n0, n1, n2, out, at) {
  gradientAt(grid, potential, x, y, z, VELOCITY, 0);
  const vx = VELOCITY[0] + 1, vy = VELOCITY[1], vz = VELOCITY[2];
  const across = vx * n0 + vy * n1 + vz * n2;
  const tx = vx - across * n0, ty = vy - across * n1, tz = vz - across * n2;
  out[at] = VELOCITY[0];
  out[at + 1] = VELOCITY[1];
  out[at + 2] = VELOCITY[2];
  out[at + 3] = tx * tx + ty * ty + tz * tz;
}

function clamp(index, size) {
  return index < 0 ? 0 : index >= size ? size - 1 : index;
}

/**
 * A scale's whole and ahead fields at (x, y, z) into `WHOLE` and `AHEAD`: trilinear over the eight
 * nodes around the point, each found among the kept nodes, summed corner by corner as the reference sums.
 */
function readScale(grid, { kept, whole, ahead }, x, y, z) {
  const [, ny, nz] = grid.nodes;
  locate(CUBE, grid, x, y, z);
  const ix = CUBE[0], fx = CUBE[1], iy = CUBE[2], fy = CUBE[3], iz = CUBE[4], fz = CUBE[5];
  WHOLE.fill(0);
  AHEAD.fill(0);
  for (let dx = 0; dx < 2; dx++) {
    const wx = dx ? fx : 1 - fx;
    for (let dy = 0; dy < 2; dy++) {
      const wxy = wx * (dy ? fy : 1 - fy);
      // The two nodes along z are neighbours in the kept list, as they are in the grid.
      const node = ((ix + dx) * ny + iy + dy) * nz + iz;
      const slot = keptSlot(kept, node, x, y, z);
      if (kept[slot + 1] !== node + 1) throw notKept(x, y, z);
      for (let dz = 0; dz < 2; dz++) {
        const w = wxy * (dz ? fz : 1 - fz);
        const row = 4 * (slot + dz);
        for (let c = 0; c < 4; c++) {
          WHOLE[c] += w * whole[row + c];
          AHEAD[c] += w * ahead[row + c];
        }
      }
    }
  }
}

/** Where `node` is in the ascending `kept`. */
function keptSlot(kept, node, x, y, z) {
  let low = 0;
  let high = kept.length - 1;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (kept[middle] < node) low = middle + 1;
    else high = middle;
  }
  if (kept[low] !== node) throw notKept(x, y, z);
  return low;
}

function notKept(x, y, z) {
  return new Error(`The car's surface lookups do not reach (${x}, ${y}, ${z}): they are kept only around its polygons and points.`);
}
