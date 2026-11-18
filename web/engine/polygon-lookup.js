/**
 * The file polygon under a point picked on the drawn surface, for the hover outline (`polygonAt`)
 * and `explain`. Polygon centres go into a grid of buckets; a polygon whose surface lies within d of
 * the point has its centre within d + `reach` of it, `reach` being the largest centre-to-corner
 * distance of any polygon, so one search out to that radius finds every candidate.
 */

import { buildPointGrid, nearestPoint, pointsWithin } from './nearest.js';
import { computeFeatures } from './features.js';
import { ensembleStatistics, networkActivations } from './network.js';

/**
 * Built once per held car, after its result: on DrivAerML run 1 (8.8M polygons) about 0.15 s and
 * 68 MiB, the grid's two arrays.
 * @param {import('./result.js').ParsedSurface} surface
 * @param {{ centers: Float32Array }} cells
 */
export function indexPolygons({ points, connectivity, offsets }, { centers }) {
  let reach = 0;
  for (let c = 0; c + 1 < offsets.length; c++) {
    const cx = centers[3 * c], cy = centers[3 * c + 1], cz = centers[3 * c + 2];
    for (let i = offsets[c]; i < offsets[c + 1]; i++) {
      const p = 3 * connectivity[i];
      const distance = (points[p] - cx) ** 2 + (points[p + 1] - cy) ** 2 + (points[p + 2] - cz) ** 2;
      if (distance > reach) reach = distance;
    }
  }
  return { grid: buildPointGrid(centers), reach: Math.sqrt(reach) };
}

/**
 * Of the polygons not facing away from `normal`, the one whose surface is nearest `point`; -1 when
 * every polygon faces away.
 * @param {{ surface: import('./result.js').ParsedSurface, cells: { centers: Float32Array, normals: Float32Array },
 *           index: ReturnType<typeof indexPolygons> }} car
 * @param {number[]} point
 * @param {number[]} normal
 */
export function polygonAt({ surface, cells, index }, point, normal) {
  const { centers, normals } = cells;
  const [x, y, z] = point;
  const facing = (c) => normals[3 * c] * normal[0] + normals[3 * c + 1] * normal[1] + normals[3 * c + 2] * normal[2] >= 0;
  const centreDistance = (c) => Math.hypot(centers[3 * c] - x, centers[3 * c + 1] - y, centers[3 * c + 2] - z);
  let best = nearestPoint(index.grid, x, y, z, facing);
  if (best < 0) return -1;
  // That polygon's surface is no farther from the point than its centre is.
  let bestDistance = Math.sqrt(squaredDistanceToPolygon(surface, best, x, y, z));
  pointsWithin(index.grid, x, y, z, centreDistance(best) + index.reach, (c) => {
    if (c === best || centreDistance(c) - index.reach > bestDistance || !facing(c)) return;
    const distance = Math.sqrt(squaredDistanceToPolygon(surface, c, x, y, z));
    if (distance < bestDistance) {
      bestDistance = distance;
      best = c;
    }
  });
  return best;
}

/**
 * The `FilePolygon` (result.js) at polygon `c`. Its Cp and spread are the analysis's own once the
 * networks have been asked there; before that they are asked now, on this thread.
 * @param {{ surface: import('./result.js').ParsedSurface, cells: { centers: Float32Array, normals: Float32Array },
 *           context: object, predicted: { cp: Float32Array, sigma: Float32Array, done: Uint8Array } }} car
 * @param {import('./network.js').Ensemble} ensemble
 * @param {number} c
 * @returns {import('./result.js').FilePolygon}
 */
export function describePolygon({ surface, cells, context, predicted }, ensemble, c) {
  const { points, connectivity, offsets } = surface;
  const corners = new Float32Array(3 * (offsets[c + 1] - offsets[c]));
  for (let i = offsets[c]; i < offsets[c + 1]; i++) corners.set(points.subarray(3 * connectivity[i], 3 * connectivity[i] + 3), 3 * (i - offsets[c]));
  const center = cells.centers.subarray(3 * c, 3 * c + 3);
  const normal = cells.normals.subarray(3 * c, 3 * c + 3);
  let cp = predicted.cp[c];
  let sigma = predicted.sigma[c];
  if (!predicted.done[c]) {
    const features = new Float32Array(ensemble.features.count);
    computeFeatures(context, center, normal, 0, 1, features);
    // One point alone: `predictOnCpu` would fill a whole block of 32.
    const { cpModels } = networkActivations(ensemble, features);
    const statistics = ensembleStatistics(cpModels, 1, ensemble.spreadScale);
    cp = statistics.cp[0];
    sigma = statistics.sigma[0];
  }
  return { index: c, corners, center: Array.from(center), normal: Array.from(normal), cp, sigma };
}

/** Distance from (x, y, z) to polygon `c`, as the triangle fan from its first corner. */
export function distanceToPolygon(surface, c, x, y, z) {
  return Math.sqrt(squaredDistanceToPolygon(surface, c, x, y, z));
}

/** Squared distance from (x, y, z) to polygon `c`, as the triangle fan from its first corner. */
function squaredDistanceToPolygon({ points, connectivity, offsets }, c, x, y, z) {
  const start = offsets[c];
  let best = Infinity;
  for (let i = start + 1; i + 1 < offsets[c + 1]; i++) {
    const distance = squaredDistanceToTriangle(points, 3 * connectivity[start], 3 * connectivity[i], 3 * connectivity[i + 1], x, y, z);
    if (distance < best) best = distance;
  }
  return best;
}

/**
 * Squared distance from (x, y, z) to the triangle with corners at offsets `a`, `b`, `c` of `points`:
 * the closest point is found by the Voronoi region of the triangle the point projects into
 * (Ericson, Real-Time Collision Detection, 5.1.5).
 */
function squaredDistanceToTriangle(points, a, b, c, x, y, z) {
  const ax = points[a], ay = points[a + 1], az = points[a + 2];
  const abx = points[b] - ax, aby = points[b + 1] - ay, abz = points[b + 2] - az;
  const acx = points[c] - ax, acy = points[c + 1] - ay, acz = points[c + 2] - az;
  const apx = x - ax, apy = y - ay, apz = z - az;
  const d1 = abx * apx + aby * apy + abz * apz;
  const d2 = acx * apx + acy * apy + acz * apz;
  const squared = (u, v, w) => u * u + v * v + w * w;
  if (d1 <= 0 && d2 <= 0) return squared(apx, apy, apz);
  const bpx = x - points[b], bpy = y - points[b + 1], bpz = z - points[b + 2];
  const d3 = abx * bpx + aby * bpy + abz * bpz;
  const d4 = acx * bpx + acy * bpy + acz * bpz;
  if (d3 >= 0 && d4 <= d3) return squared(bpx, bpy, bpz);
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const t = d1 / (d1 - d3);
    return squared(apx - t * abx, apy - t * aby, apz - t * abz);
  }
  const cpx = x - points[c], cpy = y - points[c + 1], cpz = z - points[c + 2];
  const d5 = abx * cpx + aby * cpy + abz * cpz;
  const d6 = acx * cpx + acy * cpy + acz * cpz;
  if (d6 >= 0 && d5 <= d6) return squared(cpx, cpy, cpz);
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const t = d2 / (d2 - d6);
    return squared(apx - t * acx, apy - t * acy, apz - t * acz);
  }
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
    const t = (d4 - d3) / (d4 - d3 + d5 - d6);
    return squared(bpx - t * (points[c] - points[b]), bpy - t * (points[c + 1] - points[b + 1]), bpz - t * (points[c + 2] - points[b + 2]));
  }
  const denominator = 1 / (va + vb + vc);
  const v = vb * denominator, w = vc * denominator;
  return squared(apx - abx * v - acx * w, apy - aby * v - acy * w, apz - abz * v - acz * w);
}
