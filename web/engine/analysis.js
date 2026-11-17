/**
 * The numbers the engine reports about a car: pressure drag and how it builds up from nose to tail,
 * agreement with CFD, and coverage against the training cars.
 *
 * The networks were trained on the file's own polygons, at their centres with their normals, so they
 * are only ever asked there, and every surface number is a sum over polygons. The exact numbers sum
 * over every polygon; until all are predicted, the same sums run over evenly spaced picks along the
 * running total of area, each pick standing for an equal share of it (`evenPicks`).
 */

import { stationX } from './features.js';

const BUILDUP_STATIONS = 60;
const SPREAD_BINS = 65_536;
/** Spreads above this share the histogram's last bin; Cp spreads of the networks stay far below. */
const SPREAD_RANGE = 2;

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
 * the drag build-up, the agreement with CFD and the networks' spread.
 */
export class SurfaceTally {
  constructor(networks, frontalArea, bounds) {
    this.networks = networks;
    this.frontalArea = frontalArea;
    this.bounds = bounds;
    this.drag = new Float64Array(networks);
    this.bins = new Float64Array(BUILDUP_STATIONS);
    this.area = 0;
    this.spreadSum = 0;
    this.spreadHistogram = new Float64Array(SPREAD_BINS);
    this.truth = 0;
    this.truthSquared = 0;
    this.errorSquared = 0;
    this.errorAbsolute = 0;
    this.error = 0;
  }

  /**
   * @param {number} weight  the area the polygon stands for
   * @param {number} x       its centre's x
   * @param {number} nx      its outward normal's x
   * @param {Float32Array} models  (networks, count) each network's Cp; this polygon is entry `at`
   * @param {number} cp      the networks' mean
   * @param {number} sigma   their standard deviation
   * @param {number} [cfd]   the CFD Cp, when the file carried it
   */
  add(weight, x, nx, models, at, count, cp, sigma, cfd) {
    for (let f = 0; f < this.networks; f++) this.drag[f] -= models[f * count + at] * nx * weight;
    this.bins[station(x, this.bounds)] -= cp * nx * weight;
    this.area += weight;
    this.spreadSum += sigma * weight;
    this.spreadHistogram[Math.min(Math.floor((sigma / SPREAD_RANGE) * SPREAD_BINS), SPREAD_BINS - 1)] += weight;
    if (cfd !== undefined) {
      const error = cp - cfd;
      this.truth += weight * cfd;
      this.truthSquared += weight * cfd * cfd;
      this.errorSquared += weight * error * error;
      this.errorAbsolute += weight * Math.abs(error);
      this.error += weight * error;
    }
  }

  forces(cdPCfd) {
    const cdPModels = Array.from(this.drag, (value) => value / this.frontalArea);
    const cdP = cdPModels.reduce((sum, value) => sum + value, 0) / this.networks;
    const cdPStd = Math.sqrt(cdPModels.reduce((sum, value) => sum + (value - cdP) ** 2, 0) / this.networks);
    return { cdP, cdPStd, cdPModels, cdPCfd };
  }

  /** Pressure drag accumulated from the nose to the end of each station. */
  dragBuildup(cfdBins) {
    const [xmin, xmax] = this.bounds;
    const points = [];
    let cdP = 0;
    let cdPCfd = 0;
    for (let s = 0; s < BUILDUP_STATIONS; s++) {
      cdP += this.bins[s] / this.frontalArea;
      const point = { x: xmin + ((s + 1) * (xmax - xmin)) / BUILDUP_STATIONS, cdP };
      if (cfdBins) {
        cdPCfd += cfdBins[s];
        point.cdPCfd = cdPCfd;
      }
      points.push(point);
    }
    return points;
  }

  /** Area-weighted agreement of the networks' mean with CFD. */
  accuracy() {
    const variance = this.truthSquared - (this.truth * this.truth) / this.area;
    return {
      r2: 1 - this.errorSquared / Math.max(variance, 1e-12),
      mae: this.errorAbsolute / this.area,
      rmse: Math.sqrt(this.errorSquared / this.area),
      bias: this.error / this.area,
    };
  }

  /** Area-weighted mean and 95th percentile of the networks' spread. */
  spread() {
    let cumulative = 0;
    let bin = 0;
    while (bin < SPREAD_BINS - 1 && (cumulative += this.spreadHistogram[bin]) < 0.95 * this.area) bin++;
    return { mean: this.spreadSum / this.area, p95: ((bin + 0.5) / SPREAD_BINS) * SPREAD_RANGE };
  }
}

/**
 * Pressure drag from CFD on the file's own polygons, C_d,p = -(1/A) Σ Cp n_x dA, in total and binned
 * by polygon centre into the build-up stations.
 * @param {import('./geometry.js').CellGeometry} cells
 */
export function cfdDrag(cells, cfd, frontalArea) {
  const { normals, areas, centers, bounds } = cells;
  const bins = new Float64Array(BUILDUP_STATIONS);
  let total = 0;
  for (let c = 0; c < areas.length; c++) {
    const contribution = -cfd[c] * normals[3 * c] * areas[c];
    total += contribution;
    bins[station(centers[3 * c], bounds)] += contribution;
  }
  return { cdP: total / frontalArea, bins: bins.map((value) => value / frontalArea) };
}

function station(x, [xmin, xmax]) {
  const s = Math.floor(((x - xmin) / (xmax - xmin)) * BUILDUP_STATIONS);
  return s < 0 ? 0 : s >= BUILDUP_STATIONS ? BUILDUP_STATIONS - 1 : s;
}

/**
 * The car against the training cars: its dimensions and its cross-section area along its length.
 * @param {import('./features.js').SurfaceContext} context
 * @param {import('./network.js').Envelope} envelope
 */
export function shapeCoverage(bounds, context, envelope) {
  const [xmin, xmax, ymin, ymax, zmin, zmax] = bounds;
  const sizes = { length: xmax - xmin, width: ymax - ymin, height: zmax - zmin };
  const dimensions = ['length', 'width', 'height'].map((name) => ({
    name,
    value: sizes[name],
    trainMin: envelope.dimensions[name].min,
    trainMax: envelope.dimensions[name].max,
  }));

  // Every second station along the body, as the reference's section profile.
  const sectionArea = [];
  let inBody = 0;
  let active = 0;
  let outside = 0;
  for (let i = 0; i < context.layout.nx; i++) {
    const x = stationX(context.layout, i);
    const area = context.sectionArea[i];
    const trainMin = envelope.sectionArea.min[i];
    const trainMax = envelope.sectionArea.max[i];
    if (x >= xmin - 0.05 && x <= xmax + 0.05) {
      if (inBody % 2 === 0) sectionArea.push({ x, area, trainMin, trainMax });
      inBody++;
    }
    // A station counts where the car or a training car has a real section; leaving the envelope
    // means falling more than 0.03 m² outside it.
    if (area > 0.05 || trainMax > 0.05) {
      active++;
      if (area < trainMin - 0.03 || area > trainMax + 0.03) outside++;
    }
  }
  return { dimensions, sectionArea, sectionOutsideFraction: active > 0 ? outside / active : 1 };
}
