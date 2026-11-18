/**
 * The numbers the engine reports about a car: pressure drag, whole-car totals, agreement with the
 * reference field, and its dimensions against the training cars.
 *
 * The networks were trained on the file's own polygons, at their centres with their normals, so they
 * are only ever asked there, and every surface number is a sum over polygons. The exact numbers sum
 * over every polygon; until all are predicted, the same sums run over evenly spaced picks along the
 * running total of area, each pick standing for an equal share of it (`evenPicks`).
 */

/**
 * Polygon indices at the area marks (i + 0.5) / count of the running total of area, in file order: an
 * area-weighted sample with the noise of random sampling taken out. A polygon larger than the spacing
 * is picked more than once. Indices come out in ascending order.
 * @param {Float64Array} areas
 */
export function evenPicks(areas, count) {
  let total = 0;
  for (let c = 0; c < areas.length; c++) total += areas[c];
  const picks = new Uint32Array(count);
  let cumulative = areas[0];
  let c = 0;
  for (let i = 0; i < count; i++) {
    const mark = ((i + 0.5) / count) * total;
    while (cumulative < mark && c < areas.length - 1) cumulative += areas[++c];
    picks[i] = c;
  }
  return { picks, weight: total / count };
}

/**
 * Area-weighted sums over polygons, added one polygon (or one pick) at a time: each network's drag,
 * split by the way the polygon faces, and the whole-car totals of the networks' mean.
 */
export class SurfaceTally {
  /**
   * @param {number} networks
   * @param {number} frontalArea
   * @param {import('./geometry.js').CellGeometry} cells  the polygons' normals, and the centres of the extremes
   */
  constructor(networks, frontalArea, cells) {
    this.networks = networks;
    this.frontalArea = frontalArea;
    this.cells = cells;
    /** Each network's drag from polygons facing forward (n_x < 0) and from the rest. */
    this.dragForward = new Float64Array(networks);
    this.dragRearward = new Float64Array(networks);
    this.area = 0;
    this.pressure = 0;
    this.forceY = 0;
    this.forceZ = 0;
    this.lowest = { cp: Infinity, cell: -1 };
    this.highest = { cp: -Infinity, cell: -1 };
  }

  /**
   * @param {number} weight  the area the polygon stands for
   * @param {number} cell    its index
   * @param {Float32Array} models  (networks, count) each network's Cp; this polygon is entry `at`
   * @param {number} cp      the networks' mean
   */
  add(weight, cell, models, at, count, cp) {
    const { normals } = this.cells;
    const nx = normals[3 * cell];
    const drag = nx < 0 ? this.dragForward : this.dragRearward;
    for (let f = 0; f < this.networks; f++) drag[f] -= models[f * count + at] * nx * weight;
    this.area += weight;
    this.pressure += cp * weight;
    this.forceY -= cp * normals[3 * cell + 1] * weight;
    this.forceZ -= cp * normals[3 * cell + 2] * weight;
    if (cp < this.lowest.cp) this.lowest = { cp, cell };
    if (cp > this.highest.cp) this.highest = { cp, cell };
  }

  forces() {
    const mean = (drag) => drag.reduce((sum, value) => sum + value, 0) / this.networks / this.frontalArea;
    const cdPModels = Array.from(this.dragForward, (value, f) => (value + this.dragRearward[f]) / this.frontalArea);
    const cdP = cdPModels.reduce((sum, value) => sum + value, 0) / this.networks;
    const cdPStd = Math.sqrt(cdPModels.reduce((sum, value) => sum + (value - cdP) ** 2, 0) / this.networks);
    return { cdP, cdPStd, cdPModels, byFacing: { forward: mean(this.dragForward), rearward: mean(this.dragRearward) } };
  }

  /** @returns {import('./result.js').Totals} */
  totals() {
    const { centers, normals } = this.cells;
    const extreme = ({ cp, cell }) => ({
      value: cp,
      center: Array.from(centers.subarray(3 * cell, 3 * cell + 3)),
      normal: Array.from(normals.subarray(3 * cell, 3 * cell + 3)),
    });
    return {
      area: this.area,
      meanCp: this.pressure / this.area,
      coefficients: [this.forces().cdP, this.forceY / this.frontalArea, this.forceZ / this.frontalArea],
      cpMin: extreme(this.lowest),
      cpMax: extreme(this.highest),
    };
  }
}

/** Sums for the area-weighted agreement of the networks' mean Cp with CFD, one polygon at a time. */
export class Agreement {
  area = 0;
  truth = 0;
  truthSquared = 0;
  errorSquared = 0;

  add(weight, cp, cfd) {
    const error = cp - cfd;
    this.area += weight;
    this.truth += weight * cfd;
    this.truthSquared += weight * cfd * cfd;
    this.errorSquared += weight * error * error;
  }

  /** @returns {{ r2: number }} */
  accuracy() {
    const variance = this.truthSquared - (this.truth * this.truth) / this.area;
    return { r2: 1 - this.errorSquared / Math.max(variance, 1e-12) };
  }
}

/**
 * The agreement summed over every polygon in file order, so a CFD field attached to a car later
 * gives the same number, to the last digit, as the same field read with the car's own file.
 */
export function surfaceAgreement(areas, cp, cfd) {
  const agreement = new Agreement();
  for (let c = 0; c < areas.length; c++) agreement.add(areas[c], cp[c], cfd[c]);
  return agreement.accuracy();
}

/**
 * The reference field's whole-car sums over the file's own polygons, as `SurfaceTally.totals` sums the
 * prediction's: its area-weighted mean Cp and [Cx, Cy, Cz] = −(1/A) Σ Cp n dA, so Cx is its pressure drag.
 * @param {import('./geometry.js').CellGeometry} cells
 * @returns {import('./result.js').ReferenceTotals}
 */
export function referenceTotals({ normals, areas }, cfd, frontalArea) {
  let area = 0, pressure = 0, forceX = 0, forceY = 0, forceZ = 0;
  for (let c = 0; c < areas.length; c++) {
    area += areas[c];
    pressure += cfd[c] * areas[c];
    forceX -= cfd[c] * normals[3 * c] * areas[c];
    forceY -= cfd[c] * normals[3 * c + 1] * areas[c];
    forceZ -= cfd[c] * normals[3 * c + 2] * areas[c];
  }
  return { meanCp: pressure / area, coefficients: [forceX / frontalArea, forceY / frontalArea, forceZ / frontalArea] };
}

/**
 * The car's length, width and height against the training cars' range of each.
 * @param {number[]} bounds  [xmin, xmax, ymin, ymax, zmin, zmax]
 * @param {Record<'length' | 'width' | 'height', { min: number, max: number }>} training
 * @returns {import('./result.js').Coverage}
 */
export function coverage([xmin, xmax, ymin, ymax, zmin, zmax], training) {
  const sizes = { length: xmax - xmin, width: ymax - ymin, height: zmax - zmin };
  return {
    dimensions: ['length', 'width', 'height'].map((name) => ({
      name,
      value: sizes[name],
      trainMin: training[name].min,
      trainMax: training[name].max,
    })),
  };
}
