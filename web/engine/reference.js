/**
 * A reference field for an analysed car from a second file: one of its cell arrays put on the car's
 * polygons, value for value when the two files have the same polygons, and otherwise, once the file
 * is found to describe the same surface, from the file's polygon whose centre is nearest.
 */

import { cellArrayNames, readSurface } from './vtp.js';
import { measureCells } from './geometry.js';
import { distanceToPolygon, indexPolygons, polygonAt } from './polygon-lookup.js';
import { nearestPoint } from './nearest.js';

/**
 * How far, as a median over the car's polygons, a second file's surface may lie from the car's for
 * its field to count as a reference for this car. A remeshed export of the same geometry lies within
 * its tessellation error: run 1's 600,000-triangle working mesh, simplified to 0.9 mm, lies a median
 * 0.06 mm from run 1's polygon centres and at most 0.93 mm. DrivAerML's variants differ by
 * centimetres: run 500 lies a median 25 mm from run 1.
 */
const SAME_SURFACE_DISTANCE = 0.005;
/** Polygons of the car, spread through its file, at which that distance is measured. */
const SURFACE_SAMPLES = 4096;
/** Polygons mapped between two calls of `pause`. */
const POLYGONS_BETWEEN_PAUSES = 262_144;

/**
 * The cell array `array` of `file` on the polygons of `car`, and how it was put there
 * (`ReferenceSource.mapping`): 'polygons' when the two files have the same polygons, else 'nearest',
 * each polygon taking the value at the file's polygon whose centre is nearest among those not facing
 * away (any, where all do). Rejects, saying why, when the file has no such array or its surface is
 * not the car's.
 * @param {{ surface: import('./result.js').ParsedSurface, cells: import('./geometry.js').CellGeometry }} car
 * @param {File} file
 * @param {string} array
 * @param {() => Promise<void>} pause  awaited now and then, so other requests are answered meanwhile
 * @returns {Promise<{ cfd: Float32Array, mapping: 'polygons' | 'nearest' }>}
 */
export async function referenceField(car, file, array, pause) {
  const names = await cellArrayNames(file);
  if (!names.includes(array)) {
    throw new Error(names.length
      ? `${file.name} has no cell array named ${array}. Its cell arrays of one number per polygon are ${names.join(', ')}.`
      : `${file.name} has no cell array of one number per polygon, so it holds no pressure field to compare with.`);
  }
  const surface = await readSurface(file, () => {}, array);
  if (samePolygons(car.surface, surface)) return { cfd: surface.cfd, mapping: 'polygons' };
  const cells = measureCells(surface);
  const reference = { surface, cells, index: indexPolygons(surface, cells) };
  const apart = medianDistance(car, reference);
  if (apart > SAME_SURFACE_DISTANCE) {
    throw new Error(`${file.name} is not this car's surface: its polygons lie a median ${(apart * 1000).toFixed(1)} mm from this car's, `
      + `and a field for this car must lie within ${SAME_SURFACE_DISTANCE * 1000} mm. Attach the CFD computed on this geometry.`);
  }
  return { cfd: await nearestCellValues(car.cells, reference, pause), mapping: 'nearest' };
}

function samePolygons(a, b) {
  const equal = (x, y) => x.length === y.length && x.every((value, i) => value === y[i]);
  return equal(a.offsets, b.offsets) && equal(a.connectivity, b.connectivity) && equal(a.points, b.points);
}

/**
 * The median distance from the car's polygon centres, a sample of them spread through the file, to the
 * surface of the reference polygons nearest them that do not face away.
 */
function medianDistance({ cells }, reference) {
  const count = cells.areas.length;
  const samples = Math.min(SURFACE_SAMPLES, count);
  const distances = new Float64Array(samples);
  for (let k = 0; k < samples; k++) {
    const c = Math.floor(((k + 0.5) / samples) * count);
    const point = Array.from(cells.centers.subarray(3 * c, 3 * c + 3));
    const found = polygonAt(reference, point, Array.from(cells.normals.subarray(3 * c, 3 * c + 3)));
    distances[k] = found < 0 ? Infinity : distanceToPolygon(reference.surface, found, ...point);
  }
  return distances.sort()[Math.floor(samples / 2)];
}

/** For each of the car's polygons, the reference value at the nearest reference polygon centre that does not face away. */
async function nearestCellValues(cells, { surface, cells: { normals }, index: { grid } }, pause) {
  const count = cells.areas.length;
  const values = new Float32Array(count);
  let nx = 0, ny = 0, nz = 0;
  const facing = (i) => normals[3 * i] * nx + normals[3 * i + 1] * ny + normals[3 * i + 2] * nz >= 0;
  for (let c = 0; c < count; c++) {
    nx = cells.normals[3 * c];
    ny = cells.normals[3 * c + 1];
    nz = cells.normals[3 * c + 2];
    const x = cells.centers[3 * c], y = cells.centers[3 * c + 1], z = cells.centers[3 * c + 2];
    let i = nearestPoint(grid, x, y, z, facing);
    if (i < 0) i = nearestPoint(grid, x, y, z);
    values[c] = surface.cfd[i];
    if (c % POLYGONS_BETWEEN_PAUSES === POLYGONS_BETWEEN_PAUSES - 1) await pause();
  }
  return values;
}
