/**
 * The engine's worker: turns a dropped file into a `Result` (`engine.js` is the page's side).
 *
 * reading     the file, streamed into typed arrays (vtp.js)
 * measuring   every polygon's centre, normal and area; the frontal area; the whole-car lookups the
 *             features read (geometry.js, features.js); CFD drag when the file carries CFD
 * meshing     the full mesh, drawn at once, and the working mesh (meshes.js)
 * predicting  the networks at the file's own polygons, at their centres with their normals, as they
 *             were trained: every polygon around a working-mesh vertex, which gives the working mesh
 *             its fields, and evenly spaced area picks, which give the drag, its build-up, the
 *             agreement with CFD and the spread. Features in the helper workers; the networks on the
 *             GPU (gpu.js) or, without WebGPU, in the helpers too (network.js)
 *
 * Once the result is out, the rest of the polygons are predicted in the background when the device
 * can do all of them in a minute, where the working mesh's pressure changes fastest first. The full
 * mesh's fields fill in as they go, and at the end the surface numbers are summed over every polygon.
 */

import { readSurface } from './vtp.js';
import { measureCells, frontalArea } from './geometry.js';
import { buildContext, computeFeatures, describeColumns, featureLayout, kindsWithin } from './features.js';
import { ensembleStatistics, loadEnsemble, predictOnCpu } from './network.js';
import { createGpuNetworks } from './gpu.js';
import { createPool } from './pool.js';
import { buildMeshes, cellToPoint } from './meshes.js';
import { cfdDrag, evenPicks, shapeCoverage, SurfaceTally } from './analysis.js';

/** Helpers: every core but the one the page needs. */
const HELPERS = Math.max(1, (navigator.hardwareConcurrency || 4) - 1);
/**
 * Area picks behind the first surface numbers. With 2^20 the drag lands within 0.00035 of the sum over
 * every polygon on both holdout cars (RMS 0.0005 over pick offsets).
 */
const DRAG_PICKS = 1 << 20;
/** Every polygon is predicted when the device would finish within this time. */
const FULL_PREDICTION_SECONDS = 60;
/** How often, at most, refined full-mesh fields go to the page. */
const FULL_FIELDS_INTERVAL_MS = 1000;
/** Side of the cubes the rest of the surface is predicted in, one cube at a time. */
const REFINE_CUBE = 0.04;

const engine = prepare();
engine.then(
  ({ device, ensemble }) => {
    self.postMessage({ type: 'device', device });
    self.postMessage({ type: 'model', model: modelCard(ensemble) });
  },
  (error) => self.postMessage({ type: 'unavailable', message: error.message }),
);

async function prepare() {
  const ensemble = await loadEnsemble(new URL('../model/', import.meta.url));
  const layout = featureLayout(ensemble.features);
  const gpu = await createGpuNetworks(ensemble);
  const pool = createPool(HELPERS, ensemble);
  const runners = { gpu, pool, networks: ensemble.folds.length, spreadScale: ensemble.spreadScale };
  return { ensemble, layout, runners, device: deviceName(gpu) };
}

/** @returns {import('./result.js').ModelCard} */
function modelCard({ card, activation, spreadScale, envelope, features }) {
  const columns = describeColumns(features);
  const comparison = card.comparison.map(({ columnRanges, ...entry }) => ({ ...entry, inputs: kindsWithin(columns, columnRanges) }));
  return { ...card, comparison, activation, spreadScale, dimensions: envelope.dimensions, columns };
}

function deviceName(gpu) {
  return gpu ? `WebGPU (${gpu.name})` : `CPU (${HELPERS} threads)`;
}

/** The analysis running, or the last one; a new one marks it aborted. */
let job = null;
/** The file analysed last, with its polygons, for `explain`; let go when the next analysis starts. */
let lastCar = null;
/** The lookups `explain` was last given for a result whose polygons are not held, made ready to use. */
let explainedLookups = null;

class Aborted extends Error {}

self.onmessage = ({ data }) => {
  switch (data.type) {
    case 'analyze':
      if (job) abort(job);
      lastCar = null;
      job = { id: data.id, aborted: false, controller: new AbortController() };
      analyze(job, data.file);
      break;
    case 'abort':
      if (job?.id === data.id) abort(job);
      break;
    case 'explain':
      explain(data);
      break;
    default:
      throw new Error(`Unknown engine request ${data.type}`);
  }
};

function abort(current) {
  current.aborted = true;
  current.controller.abort(new Aborted());
  engine.then(({ runners }) => runners.pool.clear(new Aborted()), () => {});
}

function checkpoint(current) {
  if (current.aborted) throw new Aborted();
}

async function analyze(current, file) {
  const post = (message, transfer = []) => {
    if (!current.aborted) self.postMessage({ ...message, id: current.id }, transfer);
  };
  try {
    const { ensemble, layout, runners } = await engine;
    if (runners.gpu?.lost) runners.gpu = await createGpuNetworks(ensemble);
    const device = deviceName(runners.gpu);
    const stages = stageClock(post);

    stages.begin('reading');
    const surface = await readSurface(file, stages.progress);
    checkpoint(current);

    stages.begin('measuring');
    const car = measure(surface, layout, stages.progress);
    runners.pool.broadcast({ type: 'context', context: car.context });
    checkpoint(current);

    stages.begin('meshing');
    const { cells } = car;
    // The area picks need no mesh, so the networks run on them while the meshes are built. Progress
    // counts toward the predicting stage only once it begins (the total is unknown until then).
    let cellsPredicted = 0;
    let cellsToPredict = Infinity;
    const counted = (size) => stages.progress((cellsPredicted += size) / cellsToPredict);
    const picks = evenPicks(cells.areas, DRAG_PICKS);
    const pickedCells = distinct(picks.picks);
    const pickBatch = predictBatch(current, runners, cells, pickedCells, counted);
    pickBatch.catch(() => {});

    let full = null;
    let working = null;
    await buildMeshes(surface, { flip: car.flip, average: car.average, signal: current.controller.signal }, {
      preview: (mesh) => queueMicrotask(() => {
        if (mesh !== full) post({ type: 'geometry', mesh: displayMesh(mesh) });
      }),
      full: (mesh) => {
        full = mesh;
        // The meshes read positions, normals and CFD until they are done, so the page gets copies,
        // made here off its main thread and handed over whole rather than copied again on arrival.
        const copy = {
          ...displayMesh(mesh),
          positions: mesh.positions.slice(),
          normals: mesh.normals.slice(),
          ...(mesh.cfd && { cfd: mesh.cfd.slice() }),
        };
        const arrays = [copy.indices, copy.positions, copy.normals, copy.cfd].filter(Boolean);
        post({ type: 'fullMesh', mesh: copy, preview: true }, arrays.map((array) => array.buffer));
      },
      working: (mesh) => {
        working = mesh;
      },
      progress: stages.progress,
    });
    checkpoint(current);

    stages.begin('predicting');
    const near = car.average.around(working.sourcePoint);
    const ringCells = difference(near.cells, pickedCells);
    cellsToPredict = pickedCells.length + ringCells.length;
    stages.progress(cellsPredicted / cellsToPredict);
    const ringBatch = await predictBatch(current, runners, cells, ringCells, counted);
    const pickBatchDone = await pickBatch;
    checkpoint(current);
    stages.end();

    const predicted = {
      cp: new Float32Array(cells.areas.length),
      sigma: new Float32Array(cells.areas.length),
      done: new Uint8Array(cells.areas.length),
    };
    for (const batch of [pickBatchDone, ringBatch]) {
      batch.cells.forEach((c, j) => {
        predicted.cp[c] = batch.fields.cp[j];
        predicted.sigma[c] = batch.fields.sigma[j];
        predicted.done[c] = 1;
      });
    }
    const mesh = {
      ...displayMesh(working),
      cp: near.average(Float32Array.from(near.cells, (c) => predicted.cp[c])),
      sigma: near.average(Float32Array.from(near.cells, (c) => predicted.sigma[c])),
    };

    const estimate = new SurfaceTally(runners.networks, car.frontal, car.bounds);
    // The picks and their distinct cells both ascend, so one walk finds each pick's prediction.
    const { models: pickModels, fields: pickFields } = pickBatchDone;
    for (let i = 0, j = 0; i < picks.picks.length; i++) {
      const c = picks.picks[i];
      while (pickedCells[j] < c) j++;
      estimate.add(picks.weight, cells.centers[3 * c], cells.normals[3 * c], pickModels, j, pickedCells.length,
        pickFields.cp[j], pickFields.sigma[j], surface.cfd?.[c]);
    }
    const shape = shapeCoverage(car.bounds, car.context, ensemble.envelope);
    const result = {
      source: {
        name: file.name,
        bytes: file.size,
        cells: cells.areas.length,
        points: surface.points.length / 3,
        hasCfd: Boolean(surface.cfd),
      },
      geometry: {
        bounds: car.bounds,
        length: car.bounds[1] - car.bounds[0],
        width: car.bounds[3] - car.bounds[2],
        height: car.bounds[5] - car.bounds[4],
        surfaceArea: car.surfaceArea,
        frontalArea: car.frontal,
      },
      ...surfaceNumbers(estimate, car, shape, ensemble.envelope),
      timings: stages.timings,
      device,
      model: ensemble.weightsSha256,
      mesh,
      lookups: lookupsOf(car.context, ensemble.features),
    };
    lastCar = { id: current.id, context: car.context, cells, average: car.average, working };
    post({ type: 'result', result });

    const cellsPerSecond = cellsToPredict / (pickBatchDone.seconds + ringBatch.seconds);
    completeSurface(current, runners, post, {
      surface, car, shape, envelope: ensemble.envelope, working: { ...mesh, sourcePoint: working.sourcePoint },
      batches: [pickBatchDone, ringBatch], predicted, showFull: Boolean(full),
      affordable: cells.areas.length / cellsPerSecond <= FULL_PREDICTION_SECONDS,
    }).catch((error) => {
      if (!(error instanceof Aborted)) console.error(error);
    });
  } catch (error) {
    if (error instanceof Aborted || current.aborted) return;
    post({ type: 'failed', message: describeFailure(error, file) });
    // Work still queued for this analysis (the area picks) has no one waiting for it.
    abort(current);
  }
}

/** Polygon geometry, the frontal area, the feature lookups, the polygon-to-point operator and the CFD drag. */
function measure(surface, layout, progress) {
  const cells = measureCells(surface);
  progress(0.5);
  const frontal = frontalArea(surface.points);
  progress(0.6);
  const context = buildContext(cells, layout);
  progress(0.8);
  const average = cellToPoint(surface, cells.areas);
  progress(0.95);
  const cfd = surface.cfd ? cfdDrag(cells, surface.cfd, frontal) : null;
  return { cells, context, average, frontal, flip: cells.flip, bounds: cells.bounds, surfaceArea: cells.surfaceArea, cfd };
}

/** The Result's surface numbers from a tally over polygons. */
function surfaceNumbers(tally, car, shape, envelope) {
  return {
    forces: tally.forces(car.cfd ? car.cfd.cdP : null),
    accuracy: car.cfd ? tally.accuracy() : null,
    coverage: { ...shape, spread: { ...tally.spread(), trainMean: envelope.spreadMean } },
    dragBuildup: tally.dragBuildup(car.cfd?.bins),
  };
}

/** An ascending list without its repeats. */
function distinct(sorted) {
  let n = 0;
  const out = new Uint32Array(sorted.length);
  for (let i = 0; i < sorted.length; i++) if (n === 0 || out[n - 1] !== sorted[i]) out[n++] = sorted[i];
  return out.slice(0, n);
}

/** The entries of ascending list `a` that ascending list `b` lacks. */
function difference(a, b) {
  const out = new Uint32Array(a.length);
  let n = 0;
  for (let i = 0, j = 0; i < a.length; i++) {
    while (j < b.length && b[j] < a[i]) j++;
    if (j >= b.length || b[j] !== a[i]) out[n++] = a[i];
  }
  return out.slice(0, n);
}

/**
 * A batch of predicted polygons: those `order` lists, each network's Cp there as one (networks,
 * count) array, their mean and spread, and the seconds it took. `counted(size)` hears of each chunk.
 */
async function predictBatch(current, runners, cells, order, counted) {
  const started = performance.now();
  const count = order.length;
  const models = new Float32Array(runners.networks * count);
  await predictPoints(current, runners, cells, order, (start, size, chunk) => {
    for (let f = 0; f < runners.networks; f++) models.set(chunk.subarray(f * size, (f + 1) * size), f * count + start);
    counted(size);
  });
  return { cells: order, models, fields: ensembleStatistics(models, count, runners.spreadScale), seconds: (performance.now() - started) / 1000 };
}

/** The contract's mesh fields only (the working mesh also carries `sourcePoint`). */
function displayMesh({ sourceTriangles, positions, normals, indices, cfd }) {
  return { sourceTriangles, positions, normals, indices, ...(cfd && { cfd }) };
}

/**
 * Stage events and timings: each stage reports nondecreasing progress, then 1 when the next begins.
 */
function stageClock(post) {
  const timings = [];
  let current = null;
  let started = 0;
  let reported = 0;
  const end = () => {
    if (!current) return;
    post({ type: 'stage', stage: current, fraction: 1 });
    timings.push({ stage: current, seconds: (performance.now() - started) / 1000 });
    current = null;
  };
  return {
    timings,
    begin(stage) {
      end();
      current = stage;
      started = performance.now();
      reported = 0;
      post({ type: 'stage', stage, fraction: 0 });
    },
    progress(fraction) {
      // At most a hundred updates a stage.
      if (current && fraction >= reported + 0.01 && fraction < 1) {
        reported = fraction;
        post({ type: 'stage', stage: current, fraction });
      }
    },
    end,
  };
}

/**
 * Each network's Cp at the polygons `order` lists, at their centres with their normals, chunk by
 * chunk: features in the helpers and the networks on the GPU, or everything in the helpers.
 * `onChunk(start, count, models)` receives each chunk's (networks, count) Cp in order.
 */
async function predictPoints(current, { gpu, pool, networks }, { centers, normals }, order, onChunk, background = false) {
  const count = order.length;
  const chunkSize = gpu ? (background ? 32_768 : 16_384) : 2_048;
  const inFlight = background ? Math.min(pool.size, 6) : pool.size;
  let next = 0;
  const launch = () => {
    const start = next;
    const size = Math.min(chunkSize, count - start);
    next += size;
    const chunkCenters = new Float32Array(3 * size);
    const chunkNormals = new Float32Array(3 * size);
    for (let i = 0; i < size; i++) {
      const c = order[start + i];
      chunkCenters.set(centers.subarray(3 * c, 3 * c + 3), 3 * i);
      chunkNormals.set(normals.subarray(3 * c, 3 * c + 3), 3 * i);
    }
    const task = pool.run(
      gpu
        ? { type: 'features', positions: chunkCenters, normals: chunkNormals, count: size, stride: gpu.stride }
        : { type: 'models', positions: chunkCenters, normals: chunkNormals, count: size },
      [chunkCenters.buffer, chunkNormals.buffer],
    );
    // A task dropped by an abort rejects; whoever awaits it handles that, the rest are let go.
    task.catch(() => {});
    return { start, size, task };
  };
  const pending = [];
  while (pending.length < inFlight && next < count) pending.push(launch());
  while (pending.length) {
    const { start, size, task } = pending.shift();
    const result = await task;
    checkpoint(current);
    if (next < count) pending.push(launch());
    let models = result;
    if (gpu) {
      models = new Float32Array(networks * size);
      await gpu.predict(result, size, models);
      checkpoint(current);
    }
    onChunk(start, size, models);
  }
}

/**
 * The rest of the surface after the result: the full mesh's fields, first from the polygons already
 * predicted with the nearest working-mesh vertex standing in where none of a vertex's polygons is;
 * then, when affordable, every other polygon, cube by cube, the cubes where the working mesh's
 * pressure varies most first; and at the end the surface numbers summed over every polygon.
 */
async function completeSurface(current, runners, post, job) {
  const { surface, car, shape, envelope, working, batches, predicted, showFull, affordable } = job;
  const { cells, average } = car;
  const cellCount = cells.areas.length;
  let predictedCount = batches.reduce((sum, batch) => sum + batch.cells.length, 0);

  let nearest = null;
  let lastSent = 0;
  const send = () => {
    const done = predictedCount < cellCount ? predicted.done : undefined;
    const cp = average.all(predicted.cp, done);
    const sigma = average.all(predicted.sigma, done);
    for (let p = 0; p < cp.length; p++) {
      if (Number.isNaN(cp[p])) {
        cp[p] = working.cp[nearest[p]];
        sigma[p] = working.sigma[nearest[p]];
      }
    }
    post({ type: 'fullFields', fields: { cp, sigma }, fraction: predictedCount / cellCount }, [cp.buffer, sigma.buffer]);
    lastSent = performance.now();
  };
  if (showFull) {
    nearest = await nearestWorkingVertices(runners.pool, surface.points, working.positions);
    checkpoint(current);
    send();
  }
  if (!affordable) return;

  const exact = new SurfaceTally(runners.networks, car.frontal, car.bounds);
  for (const batch of batches) {
    batch.cells.forEach((c, j) => {
      exact.add(cells.areas[c], cells.centers[3 * c], cells.normals[3 * c], batch.models, j, batch.cells.length,
        batch.fields.cp[j], batch.fields.sigma[j], surface.cfd?.[c]);
    });
  }
  const { connectivity, offsets } = surface;
  let order;
  if (nearest) {
    const variation = edgeVariation(working);
    order = refinementOrder(cells.centers, (c) => variation[nearest[connectivity[offsets[c]]]]);
  } else {
    order = Uint32Array.from({ length: cellCount }, (_, c) => c);
  }
  const remaining = order.filter((c) => !predicted.done[c]);

  await predictPoints(current, runners, cells, remaining, (start, size, chunk) => {
    const fields = ensembleStatistics(chunk, size, runners.spreadScale);
    for (let i = 0; i < size; i++) {
      const c = remaining[start + i];
      predicted.cp[c] = fields.cp[i];
      predicted.sigma[c] = fields.sigma[i];
      predicted.done[c] = 1;
      exact.add(cells.areas[c], cells.centers[3 * c], cells.normals[3 * c], chunk, i, size,
        fields.cp[i], fields.sigma[i], surface.cfd?.[c]);
    }
    predictedCount += size;
    if (showFull && predictedCount < cellCount && performance.now() - lastSent > FULL_FIELDS_INTERVAL_MS) send();
  }, true);
  post({ type: 'exact', numbers: surfaceNumbers(exact, car, shape, envelope) });
  if (showFull) send();
}

/** For every file point, the index of the nearest working-mesh vertex, found in the helpers. */
async function nearestWorkingVertices(pool, points, workingPositions) {
  pool.broadcast({ type: 'working', positions: workingPositions });
  const pointCount = points.length / 3;
  const nearest = new Uint32Array(pointCount);
  const step = 262_144;
  await Promise.all(Array.from({ length: Math.ceil(pointCount / step) }, (_, k) => {
    const start = k * step;
    const size = Math.min(step, pointCount - start);
    const slice = points.slice(3 * start, 3 * (start + size));
    return pool.run({ type: 'nearest', positions: slice, count: size }, [slice.buffer])
      .then((result) => nearest.set(result, start));
  }));
  return nearest;
}

/** For each working-mesh vertex, the largest change of Cp along any of its edges. */
function edgeVariation({ indices, cp }) {
  const variation = new Float32Array(cp.length);
  for (let t = 0; t < indices.length; t += 3) {
    for (let e = 0; e < 3; e++) {
      const a = indices[t + e];
      const b = indices[t + ((e + 1) % 3)];
      const change = Math.abs(cp[a] - cp[b]);
      if (change > variation[a]) variation[a] = change;
      if (change > variation[b]) variation[b] = change;
    }
  }
  return variation;
}

/**
 * Items grouped by the cube their centre falls in, cubes ordered from the most varied to the least
 * (a cube's variation is the largest of its items').
 * @param {Float32Array} centers (N, 3)
 * @param {(item: number) => number} variationOf
 */
function refinementOrder(centers, variationOf) {
  const count = centers.length / 3;
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < centers.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      if (centers[i + k] < min[k]) min[k] = centers[i + k];
      if (centers[i + k] > max[k]) max[k] = centers[i + k];
    }
  }
  const shape = [0, 1, 2].map((k) => Math.floor((max[k] - min[k]) / REFINE_CUBE) + 1);
  const cubeOf = new Uint32Array(count);
  const cubeVariation = new Float32Array(shape[0] * shape[1] * shape[2]).fill(-1);
  for (let i = 0; i < count; i++) {
    const cx = Math.floor((centers[3 * i] - min[0]) / REFINE_CUBE);
    const cy = Math.floor((centers[3 * i + 1] - min[1]) / REFINE_CUBE);
    const cz = Math.floor((centers[3 * i + 2] - min[2]) / REFINE_CUBE);
    const cube = (cx * shape[1] + cy) * shape[2] + cz;
    cubeOf[i] = cube;
    const value = variationOf(i);
    if (value > cubeVariation[cube]) cubeVariation[cube] = value;
  }

  const occupied = [];
  for (let cube = 0; cube < cubeVariation.length; cube++) if (cubeVariation[cube] >= 0) occupied.push(cube);
  occupied.sort((a, b) => cubeVariation[b] - cubeVariation[a]);
  const rank = new Uint32Array(cubeVariation.length);
  occupied.forEach((cube, r) => { rank[cube] = r; });
  const starts = new Uint32Array(occupied.length + 1);
  for (let i = 0; i < count; i++) starts[rank[cubeOf[i]] + 1]++;
  for (let r = 1; r < starts.length; r++) starts[r] += starts[r - 1];
  const order = new Uint32Array(count);
  for (let i = 0; i < count; i++) order[starts[rank[cubeOf[i]]]++] = i;
  return order;
}

/** A result's lookups: the car's feature lookups and the feature definition they were built for. */
function lookupsOf({ layout, ...lookups }, definition) {
  return { definition, ...lookups };
}

/**
 * Everything the networks are told at a point on a result's surface, and their answers. `vertex` is
 * the working-mesh vertex the page found nearest the point. While this worker holds the polygons of
 * the result's file (the one analysed last), the networks are asked at the polygon around that
 * vertex whose centre is nearest the point, with its normal, as in training and in the analysis.
 * Otherwise they are asked at the vertex itself, with its polygon-averaged normal, from the lookups
 * the page sends with the first request about each result.
 */
async function explain({ id, point, analysis, model, lookups, vertex }) {
  try {
    const { ensemble, layout } = await engine;
    if (model !== ensemble.weightsSha256) {
      throw new Error('This result was computed with a different version of the model than this page runs, so it cannot be explained.');
    }
    if (lookups) {
      if (JSON.stringify(lookups.definition) !== JSON.stringify(ensemble.features)) {
        throw new Error('This result was computed with different model inputs than this page uses, so it cannot be explained.');
      }
      explainedLookups = { ...lookups, layout };
    }
    let site;
    let context;
    if (lastCar && lastCar.id === analysis) {
      const { cells } = lastCar;
      const cell = nearestPolygon(lastCar, vertex.index, point);
      site = { at: 'polygon', index: cell, position: cells.centers.slice(3 * cell, 3 * cell + 3), normal: cells.normals.slice(3 * cell, 3 * cell + 3) };
      context = lastCar.context;
    } else {
      site = { at: 'vertex', index: vertex.index, position: Float32Array.from(vertex.position), normal: Float32Array.from(vertex.normal) };
      context = explainedLookups;
    }
    const features = new Float32Array(layout.count);
    computeFeatures(context, site.position, site.normal, 0, 1, features);
    const models = new Float32Array(ensemble.folds.length);
    predictOnCpu(ensemble, features, 1, models);
    const { cp, sigma } = ensembleStatistics(models, 1, ensemble.spreadScale);
    const explanation = {
      ...site,
      position: Array.from(site.position),
      normal: Array.from(site.normal),
      features,
      cpModels: Array.from(models),
      cp: cp[0],
      sigma: sigma[0],
    };
    self.postMessage({ type: 'explanation', id, explanation });
  } catch (error) {
    self.postMessage({ type: 'explanationFailed', id, message: error.message });
  }
}

/** Of the file polygons around the file point working-mesh vertex `vertex` sits on, the one whose centre is nearest `point`. */
function nearestPolygon({ cells, average, working }, vertex, point) {
  let cell = -1;
  let best = Infinity;
  for (const c of average.around(working.sourcePoint.subarray(vertex, vertex + 1)).cells) {
    const distance = (cells.centers[3 * c] - point[0]) ** 2 + (cells.centers[3 * c + 1] - point[1]) ** 2 + (cells.centers[3 * c + 2] - point[2]) ** 2;
    if (distance < best) {
      best = distance;
      cell = c;
    }
  }
  return cell;
}

/** The message for a failed analysis: the file's own problem as the reader words it, or what ran out. */
function describeFailure(error, file) {
  if (error instanceof RangeError && /allocat|memory/i.test(error.message)) {
    return `This device ran out of memory analysing ${file.name} (${(file.size / 1e6).toFixed(0)} MB). Close other tabs or windows and try again, or use a machine with more memory.`;
  }
  return error.message;
}
