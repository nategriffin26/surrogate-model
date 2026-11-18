/**
 * The numbers the engine reports about a car: pressure drag, agreement with CFD, and its dimensions
 * against the training cars.
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
 * split by the way the polygon faces, and the agreement with CFD.
 */
export class SurfaceTally {
  constructor(networks, frontalArea) {
    this.networks = networks;
    this.frontalArea = frontalArea;
    /** Each network's drag from polygons facing forward (n_x < 0) and from the rest. */
    this.dragForward = new Float64Array(networks);
    this.dragRearward = new Float64Array(networks);
    this.area = 0;
    this.truth = 0;
    this.truthSquared = 0;
    this.errorSquared = 0;
  }

  /**
   * @param {number} weight  the area the polygon stands for
   * @param {number} nx      its outward normal's x
   * @param {Float32Array} models  (networks, count) each network's Cp; this polygon is entry `at`
   * @param {number} cp      the networks' mean
   * @param {number} [cfd]   the CFD Cp, when the file carried it
   */
  add(weight, nx, models, at, count, cp, cfd) {
    const drag = nx < 0 ? this.dragForward : this.dragRearward;
    for (let f = 0; f < this.networks; f++) drag[f] -= models[f * count + at] * nx * weight;
    this.area += weight;
    if (cfd !== undefined) {
      const error = cp - cfd;
      this.truth += weight * cfd;
      this.truthSquared += weight * cfd * cfd;
      this.errorSquared += weight * error * error;
    }
  }

  forces(cdPCfd) {
    const mean = (drag) => drag.reduce((sum, value) => sum + value, 0) / this.networks / this.frontalArea;
    const cdPModels = Array.from(this.dragForward, (value, f) => (value + this.dragRearward[f]) / this.frontalArea);
    const cdP = cdPModels.reduce((sum, value) => sum + value, 0) / this.networks;
    const cdPStd = Math.sqrt(cdPModels.reduce((sum, value) => sum + (value - cdP) ** 2, 0) / this.networks);
    return { cdP, cdPStd, cdPModels, cdPCfd, byFacing: { forward: mean(this.dragForward), rearward: mean(this.dragRearward) } };
  }

  /** Area-weighted agreement of the networks' mean with CFD. */
  accuracy() {
    const variance = this.truthSquared - (this.truth * this.truth) / this.area;
    return { r2: 1 - this.errorSquared / Math.max(variance, 1e-12) };
  }
}

/**
 * Pressure drag from CFD on the file's own polygons, C_d,p = -(1/A) Σ Cp n_x dA.
 * @param {import('./geometry.js').CellGeometry} cells
 */
export function cfdDrag({ normals, areas }, cfd, frontalArea) {
  let total = 0;
  for (let c = 0; c < areas.length; c++) total -= cfd[c] * normals[3 * c] * areas[c];
  return total / frontalArea;
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
