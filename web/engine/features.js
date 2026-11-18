/**
 * The numbers the networks are told about a surface point, as `features.py` defines them, read from
 * a car's lookups (`lookups.js`): where the point is on the car and which way it faces; the surface
 * around the point at four scales, whole and split into the part ahead and the part behind; and
 * slices across the car at the point's station and ahead and behind it, each read for its area,
 * how far it reaches outboard of the point within a centimetre of the point's height, and how far it
 * reaches above and below the point at the point's distance from the centre plane. Nothing is folded
 * across the centre plane: a point is read on its own side.
 *
 * Axes are the file's: x downstream, z up, and so (right-handed) +y toward the car's right as seen
 * from the driver's seat.
 *
 * The definition's constants come from the model file, which `export_web.py` writes from
 * `features.py`, so a model trained on another definition brings its own. Arithmetic follows the
 * reference's float32 and float64 steps (`Math.fround` where NumPy rounds to float32), so raster
 * bins are chosen as in training and values agree to float32.
 */

import { gaussianWeights, splineAround, upstreamHalf } from './grid.js';

const f32 = Math.fround;

/**
 * The reads of a surface scale a model can be given, in the reference's column order, each with its
 * columns' names and the part of the surface they read. A model file names the reads it was trained
 * with (`surfaceReads`), the same for every scale.
 */
const SURFACE_READS = {
  meanNormal: [['Mean facing rearward', 'whole'], ['Mean facing sideways', 'whole'], ['Mean facing up', 'whole']],
  amount: [['Surface density', 'whole']],
  ownMinusMean: [['Bend, rearward', 'whole'], ['Bend, sideways', 'whole'], ['Bend, up', 'whole']],
  aheadNormal: [['Mean facing rearward, ahead', 'ahead'], ['Mean facing sideways, ahead', 'ahead'], ['Mean facing up, ahead', 'ahead']],
  aheadAmount: [['Surface density, ahead', 'ahead']],
  behindNormal: [['Mean facing rearward, behind', 'behind'], ['Mean facing sideways, behind', 'behind'], ['Mean facing up, behind', 'behind']],
  behindAmount: [['Surface density, behind', 'behind']],
};

/**
 * @typedef {Object} FeatureDefinition  as `export_web.py` writes it
 * @property {number} count
 * @property {number} rasterStep
 * @property {number[]} domainXFromNose   the raster spans x − xmin over this range, xmin the nose (the
 *   lowest polygon-centre x)
 * @property {number} domainYHalf      the raster spans y from the centre plane over [−domainYHalf, domainYHalf)
 * @property {number[]} domainZFromGround the side view spans z − zmin over this range, zmin the lowest
 *   polygon-centre z
 * @property {number} heightStep       the side view's height bins, finer than the raster step
 * @property {number} heightWindow     how far from a height the side view's reach at it looks
 * @property {number[]} sliceDx        the stations ahead of (negative) and behind the point's own
 * @property {number[]} surfaceScales
 * @property {number[]} surfaceSteps
 * @property {number} surfaceSigma     the Gaussian's standard deviation, as a share of the scale
 * @property {number} gaussianTruncate the Gaussian's reach, in standard deviations
 * @property {string[]} surfaceReads   the reads of each surface scale given to the model, in order
 *   (`SURFACE_READS` names them)
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
 * @typedef {FeatureDefinition & { h: number, x0: number, yHalf: number, z0: number, heightBin: number,
 *   nx: number, ny: number, nh: number, heightReach: number, scales: ScaleLayout[], surfaceColumns: number[] }} FeatureLayout
 *   The raster's origin in the car's frame (x0 from the nose, −yHalf from the centre plane, z0 from the
 *   ground), step h and the side view's height bin in float32, as the raster is built; its size
 *   (nx, ny) and the side view's height bins nh; `heightWindow` in height bins; each scale's grid; and
 *   which of a scale's fifteen values go to the model, in order.
 */

/**
 * The definition with its float32 constants, grid sizes and Gaussians worked out once.
 * @param {FeatureDefinition} definition
 * @returns {FeatureLayout}
 */
export function featureLayout(definition) {
  const { rasterStep, domainXFromNose, domainYHalf, domainZFromGround, heightStep, heightWindow, surfaceSigma, gaussianTruncate } = definition;
  const layout = {
    ...definition,
    h: f32(rasterStep),
    x0: f32(domainXFromNose[0]),
    yHalf: f32(domainYHalf),
    z0: f32(domainZFromGround[0]),
    heightBin: f32(heightStep),
    nx: Math.round((domainXFromNose[1] - domainXFromNose[0]) / rasterStep),
    ny: 2 * Math.round(domainYHalf / rasterStep),
    nh: Math.round((domainZFromGround[1] - domainZFromGround[0]) / heightStep),
    heightReach: Math.round(heightWindow / heightStep),
    scales: definition.surfaceScales.map((scale, s) => {
      const step = definition.surfaceSteps[s];
      const weights = gaussianWeights((surfaceSigma * scale) / step, gaussianTruncate);
      const radius = (weights.length - 1) / 2;
      // A flat panel lays step² of area per column of nodes. Across the panel the spread and the read
      // each share it among the three nodes around it; averaged over where the panel falls between
      // nodes, the node itself counts 11/20, each neighbour 13/60 and each second neighbour 1/120 (the
      // quintic B-spline at whole steps).
      const flatPanel = step * step * ((11.0 / 20.0) * weights[radius] + (13.0 / 30.0) * weights[radius + 1] + (1.0 / 60.0) * weights[radius + 2]);
      return { scale, step, radius, weights, upstream: upstreamHalf(weights), flatPanel };
    }),
    surfaceColumns: surfaceColumns(definition.surfaceReads),
  };
  const columns = describeColumns(definition).length;
  if (columns !== definition.count) {
    throw new Error(`The model file's feature definition gives ${columns} columns but says ${definition.count}.`);
  }
  return layout;
}

/** Where each read's columns sit among a scale's fifteen values, in the order `reads` lists them. */
function surfaceColumns(reads) {
  if (!Array.isArray(reads)) {
    throw new Error("The model file's feature definition does not say which surface reads its model was given (surfaceReads).");
  }
  const unknown = reads.filter((read) => !Object.hasOwn(SURFACE_READS, read));
  if (unknown.length) throw new Error(`The model file's feature definition names surface reads this page does not know: ${unknown.join(', ')}.`);
  const first = {};
  let at = 0;
  for (const [read, columns] of Object.entries(SURFACE_READS)) {
    first[read] = at;
    at += columns.length;
  }
  return reads.flatMap((read) => SURFACE_READS[read].map((_, i) => first[read] + i));
}

/**
 * For each column, in the networks' order: `kind`, a stable id of its family; `group`, the family's
 * display label; a short plain name; its unit; `offset`, the place it reads relative to the point,
 * [dx, dy, dz] in metres along the file's axes (dx downstream, dy toward +y, dz up); and `along`, for
 * a distance in metres, the unit direction it runs from that place, so place + value × along is where
 * it measured to (a body surface, or for the distances from nose, tail, centre plane and lowest point
 * that plane), null for every other column. A slice's reach outboard is measured away from the
 * centre plane on the point's own side: its `along` is [0, 1, 0] for a point on the +y side and is
 * turned to [0, −1, 0] for a point on the −y side. A surface column also has `scale`, the size it is
 * named by, `sigma`, its Gaussian's standard deviation (both in metres), and `part`: the whole surface
 * about the point, the part ahead of it, or the part behind. A slice column also has `station`, its
 * dx, and `read`: 'area', 'outboard', 'above' or 'below'.
 * @param {FeatureDefinition} definition
 * @returns {import('./result.js').ModelColumn[]}
 */
export function describeColumns({ surfaceScales, surfaceSigma, surfaceReads, sliceDx }) {
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
  const forward = [-1, 0, 0], rearward = [1, 0, 0], towardMinusY = [0, -1, 0], outboard = [0, 1, 0], up = [0, 0, 1], down = [0, 0, -1];

  const position = 'Position and facing';
  add('position', position, 'Distance from nose', { along: forward });
  add('position', position, 'Distance to tail', { along: rearward });
  // Signed, so a point at y below the centre plane reads a negative value and still reaches the plane.
  add('position', position, 'Distance from the centre plane', { along: towardMinusY });
  add('position', position, 'Height above lowest point', { along: down });
  add('position', position, 'Facing rearward');
  add('position', position, 'Facing sideways');
  add('position', position, 'Facing up');

  for (const scale of surfaceScales) {
    const group = `Surface at ${Math.round(scale * 100)} cm`;
    for (const read of surfaceReads) {
      for (const [name, part] of SURFACE_READS[read]) add('surface', group, name, { scale, sigma: surfaceSigma * scale, part });
    }
  }

  const slices = 'Slices across the car';
  for (const station of [0, ...sliceDx]) {
    const place = where([station, 0, 0]);
    const at = { offset: [station, 0, 0], station };
    add('slice', slices, `Slice area${place}`, { ...at, read: 'area', unit: 'm²' });
    add('slice', slices, `Reach outboard${place}`, { ...at, read: 'outboard', along: outboard });
    add('slice', slices, `Reach above${place}`, { ...at, read: 'above', along: up });
    add('slice', slices, `Reach below${place}`, { ...at, read: 'below', along: down });
  }
  return columns;
}

/** Column kinds in the order the model view lists them. */
const COLUMN_KINDS = ['position', 'surface', 'slice'];

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

const SPLINE = new Float64Array(12);
/** Where a point falls among a raster's bin centres, per axis: [bin, fraction] for the station, height and row. */
const STATION = new Float64Array(2);
const HEIGHT = new Float64Array(2);
const ROW = new Float64Array(2);
const WHOLE = new Float64Array(4);
const AHEAD = new Float64Array(4);
/** A scale's fifteen values at a point, in `SURFACE_READS` order. */
const SURFACE = new Float64Array(15);

/**
 * Write the features of points [first, first + count) into `out`, one row of `stride` floats per
 * point (`stride` at least the feature count; the rest of a row is left as it is).
 * @param {SurfaceContext} context
 * @param {Float32Array} positions (P, 3)
 * @param {Float32Array} normals   (P, 3) unit, outward
 * @param {Float32Array} out       (count, stride)
 */
export function computeFeatures(context, positions, normals, first, count, out, stride = context.layout.count) {
  const { layout, yOut, zTop, zBottom, sliceArea, scales } = context;
  const { rasterStep, domainXFromNose, domainYHalf, domainZFromGround, heightStep, nx, ny, nh, sliceDx, surfaceColumns: chosen } = layout;
  const [xmin, xmax, , , zmin] = context.bounds;
  const yCenter = context.yCenter;
  const scaleGrids = scales.map(({ origin, nodes }, s) => ({ origin, nodes, step: layout.scales[s].step }));

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

    // The surface around the point at each scale: its mean facing and density, the point's own facing
    // against that mean, then the same for the part ahead and the part behind; the model's share of them.
    for (let s = 0; s < scales.length; s++) {
      readScale(scaleGrids[s], scales[s], sx, positions[q + 1], sz);
      const flat = layout.scales[s].flatPanel;
      const whole = Math.max(WHOLE[0], 1e-12);
      const mx = WHOLE[1] / whole, my = WHOLE[2] / whole, mz = WHOLE[3] / whole;
      const ahead = Math.max(AHEAD[0], 1e-12);
      const behindArea = WHOLE[0] - AHEAD[0];
      const behind = Math.max(behindArea, 1e-12);
      SURFACE[0] = mx;
      SURFACE[1] = my;
      SURFACE[2] = mz;
      SURFACE[3] = WHOLE[0] / flat;
      SURFACE[4] = n0 - mx;
      SURFACE[5] = n1 - my;
      SURFACE[6] = n2 - mz;
      SURFACE[7] = AHEAD[1] / ahead;
      SURFACE[8] = AHEAD[2] / ahead;
      SURFACE[9] = AHEAD[3] / ahead;
      SURFACE[10] = AHEAD[0] / (0.5 * flat);
      SURFACE[11] = (WHOLE[1] - AHEAD[1]) / behind;
      SURFACE[12] = (WHOLE[2] - AHEAD[2]) / behind;
      SURFACE[13] = (WHOLE[3] - AHEAD[3]) / behind;
      SURFACE[14] = behindArea / (0.5 * flat);
      for (const column of chosen) out[k++] = SURFACE[column];
    }

    // Slices across the car at the point's station and at each of `sliceDx`: the area, how far the
    // body reaches outboard of the point on its own side within `heightWindow` of its height, and how
    // far above and below the point at its distance from the centre plane. Each is read between bin
    // centres, so that it moves smoothly with the point instead of stepping where the point crosses a
    // bin edge, which drew contours along sloping panels; a bin with no body takes part with its empty
    // value. Stations and heights are measured from the nose and the ground, as the raster is binned,
    // so moving the whole car moves nothing it reads. In float64, as the reference reads them.
    const fromNose = sx - xmin;
    const side = sy >= 0 ? nx * nh : 0;
    const fromCentre = Math.abs(sy);
    betweenBins(HEIGHT, sz - zmin, domainZFromGround[0], heightStep, nh);
    betweenBins(ROW, sy, -domainYHalf, rasterStep, ny);
    for (let s = -1; s < sliceDx.length; s++) {
      betweenBins(STATION, s < 0 ? fromNose : fromNose + sliceDx[s], domainXFromNose[0], rasterStep, nx);
      const i = STATION[0], f = STATION[1];
      out[k++] = (1.0 - f) * sliceArea[i] + f * sliceArea[i + 1];
      out[k++] = bilinear(yOut, side, nh, STATION, HEIGHT) - fromCentre;
      out[k++] = bilinear(zTop, 0, ny, STATION, ROW) - sz;
      out[k++] = sz - bilinear(zBottom, 0, ny, STATION, ROW);
    }
  }
}

/**
 * Where `q` falls among the centres lo + (i + ½) step of `n` bins along one axis, into `at` as the
 * bin whose centre is at or below it and the fraction of the way to the next centre. Beyond the outer
 * centres, the outer bin's value.
 */
function betweenBins(at, q, lo, step, n) {
  const t = (q - lo) / step - 0.5;
  const i = Math.min(Math.max(Math.floor(t), 0), n - 2);
  at[0] = i;
  at[1] = Math.min(Math.max(t - i, 0.0), 1.0);
}

/** A raster of rows of `columns` bins, starting at `start` in `field`, read between its bin centres. */
function bilinear(field, start, columns, along, across) {
  const fi = along[1], fj = across[1];
  const a = start + along[0] * columns + across[0], b = a + columns;
  return (1.0 - fi) * ((1.0 - fj) * field[a] + fj * field[a + 1]) + fi * ((1.0 - fj) * field[b] + fj * field[b + 1]);
}

/**
 * A scale's whole and ahead fields at each of `positions`, eight floats a point (whole, then ahead),
 * as `computeFeatures` reads them: what a bundled example keeps of a scale whose kept nodes outnumber
 * its mesh's vertices (`packResult`).
 * @param {{ origin: number[], nodes: number[], step: number }} grid
 * @param {import('./lookups.js').ScaleLookup} lookup
 * @param {Float32Array} positions (P, 3)
 * @returns {Float32Array} (P, 8)
 */
export function scaleReadsAt(grid, lookup, positions) {
  const reads = new Float32Array((8 * positions.length) / 3);
  for (let p = 0; p < positions.length / 3; p++) {
    readScale(grid, lookup, positions[3 * p], positions[3 * p + 1], positions[3 * p + 2]);
    reads.set(WHOLE, 8 * p);
    reads.set(AHEAD, 8 * p + 4);
  }
  return reads;
}

/**
 * A bundled example's lookups as `computeFeatures` takes them at vertex `vertex` of its mesh: each
 * scale stored as its reads at the vertices hands over that vertex's read.
 * @param {import('./lookups.js').SurfaceLookups} lookups
 * @param {number} vertex
 */
export function lookupsAtVertex(lookups, vertex) {
  const at = 8 * vertex;
  const scales = lookups.scales.map((scale) => (scale.readsAtVertices ? { ...scale, read: scale.readsAtVertices.subarray(at, at + 8) } : scale));
  return { ...lookups, scales };
}

/**
 * A scale's whole and ahead fields at (x, y, z) into `WHOLE` and `AHEAD`: quadratic B-spline weights
 * over the 27 nodes around the point, each found among the kept nodes, summed node by node in the
 * reference's order. A scale handed over as its stored `read` at the one point asked about (a bundled
 * example's vertex, in `explain`) gives that read.
 */
function readScale(grid, { kept, whole, ahead, read }, x, y, z) {
  if (read) {
    for (let c = 0; c < 4; c++) {
      WHOLE[c] = read[c];
      AHEAD[c] = read[4 + c];
    }
    return;
  }
  const [, ny, nz] = grid.nodes;
  splineAround(SPLINE, grid, x, y, z);
  WHOLE.fill(0);
  AHEAD.fill(0);
  for (let dx = 0; dx < 3; dx++) {
    for (let dy = 0; dy < 3; dy++) {
      const wxy = SPLINE[3 + dx] * SPLINE[6 + dy];
      // The three nodes along z are neighbours in the kept list, as they are in the grid.
      const node = ((SPLINE[0] + dx - 1) * ny + SPLINE[1] + dy - 1) * nz + SPLINE[2] - 1;
      const slot = keptSlot(kept, node, x, y, z);
      if (kept[slot + 2] !== node + 2) throw notKept(x, y, z);
      for (let dz = 0; dz < 3; dz++) {
        const w = wxy * SPLINE[9 + dz];
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
