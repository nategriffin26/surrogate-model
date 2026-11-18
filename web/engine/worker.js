/**
 * The engine's worker: turns a dropped file, or a surface given as arrays, into a `Result`
 * (`engine.js` is the page's side).
 *
 * reading     the file, streamed into typed arrays (vtp.js), or the arrays given, checked (geometry.js)
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
 * From then on the car can be given a reference field from a second file (`attachReference`), which
 * it holds as if its own file had carried it.
 *
 * Analyses run one at a time, each to its end, in the order asked for: a car's background prediction
 * is part of its analysis, and two files at once would need twice the memory. Only an abort ends one
 * early. A quick prediction (`predictQuick`) waits its turn the same way: it predicts every polygon
 * of the surface it is given at once, with nothing drawn first and nothing left for the background.
 */

import { addCellArrays, CFD_ARRAY, cellArrayNames, readSurface, writeVtp } from './vtp.js';
import { boundsOf, checkSurface, measureCells, frontalArea } from './geometry.js';
import { computeFeatures, describeColumns, featureLayout, kindsWithin, lookupsAtVertex } from './features.js';
import { buildLookups } from './lookups.js';
import { ensembleStatistics, loadEnsemble, networkActivations } from './network.js';
import { createGpuNetworks } from './gpu.js';
import { createGridOperations } from './grid-gpu.js';
import { createPool } from './pool.js';
import { buildMeshes, cellToPoint, surfaceMesh } from './meshes.js';
import { Agreement, coverage, evenPicks, referenceTotals, surfaceAgreement, SurfaceTally } from './analysis.js';
import { describePolygon, distanceToPolygon, indexPolygons, polygonAt } from './polygon-lookup.js';
import { nearestPoint } from './nearest.js';

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

const engine = prepare();
engine.then(
  ({ device, model }) => {
    self.postMessage({ type: 'device', device });
    self.postMessage({ type: 'model', model });
  },
  (error) => self.postMessage({ type: 'unavailable', message: error.message }),
);

async function prepare() {
  const ensemble = await loadEnsemble(new URL('../model/', import.meta.url));
  const layout = featureLayout(ensemble.features);
  const model = modelCard(ensemble);
  const gpu = await createGpuNetworks(ensemble);
  const pool = createPool(HELPERS, ensemble);
  const runners = { gpu, grids: gpu && createGridOperations(gpu.device), pool, networks: ensemble.folds.length, spreadScale: ensemble.spreadScale };
  return { ensemble, layout, runners, model, device: deviceName(gpu) };
}

/** A GPU the browser took away, replaced before the next car, with the grid operations on the new device. */
async function replaceLostGpu(runners, ensemble) {
  if (!runners.gpu?.lost) return;
  runners.gpu = await createGpuNetworks(ensemble);
  runners.grids = runners.gpu && createGridOperations(runners.gpu.device);
}

/**
 * The model file's card with what each input is; each input's range over the training cells moves
 * from the card onto its column.
 * @returns {import('./result.js').ModelCard}
 */
function modelCard({ card, activation, spreadScale, features }) {
  const { inputRanges, ...facts } = card;
  const described = describeColumns(features);
  if (inputRanges?.length !== described.length) {
    throw new Error(`The model file gives training ranges for ${inputRanges?.length ?? 0} inputs; its networks take ${described.length}.`);
  }
  const columns = described.map((column, i) => ({ ...column, range: inputRanges[i] }));
  const comparison = facts.comparison.map(({ columnRanges, ...entry }) => ({ ...entry, inputs: kindsWithin(columns, columnRanges) }));
  return { ...facts, comparison, activation, spreadScale, columns };
}

function deviceName(gpu) {
  return gpu ? `WebGPU (${gpu.name})` : `CPU (${HELPERS} threads)`;
}

/** Analyses asked for and not yet ended, by id: the one running and those waiting their turn. */
const jobs = new Map();
/** Settles when the last analysis asked for has ended. */
let turn = Promise.resolve();
/** The analysis running, if any. */
let running = null;
/**
 * The car analysed last: its file (null when it came as arrays), polygons, lookups and predictions
 * so far, with an index of its polygons for `explain` and `polygonAt`. `complete` once every polygon
 * is predicted, for `exportVtp`, `fileCells` and references. `surface.cfd` is its reference field,
 * `ownCfd` the one its file carried, and `sourcePoint` the file point under each working-mesh vertex,
 * which a new field is put on. Let go when its analysis is aborted or the next starts.
 */
let lastCar = null;
/** The lookups `explain` was last given for a result whose polygons are not held, made ready to use. */
let explainedLookups = null;

class Aborted extends Error {}

self.onmessage = ({ data }) => {
  switch (data.type) {
    case 'analyze':
    case 'predictQuick': {
      const job = { id: data.id, aborted: false, controller: new AbortController() };
      jobs.set(job.id, job);
      turn = turn.then(() => run(job, data));
      break;
    }
    case 'abort': {
      const job = jobs.get(data.id);
      if (job) abort(job);
      if (lastCar?.id === data.id) lastCar = null;
      break;
    }
    case 'explain':
      explain(data);
      break;
    case 'polygonAt':
      answerPolygonAt(data);
      break;
    case 'export':
      exportVtp(data);
      break;
    case 'fileCells':
      answerFileCells(data);
      break;
    case 'cellArrayNames':
      cellArrayNames(data.file).then(
        (names) => self.postMessage({ type: 'answer', id: data.id, answer: names }),
        (error) => self.postMessage({ type: 'answerFailed', id: data.id, message: error.message }),
      );
      break;
    case 'attachReference':
      attachReference(data);
      break;
    case 'detachReference':
      detachReference(data);
      break;
    default:
      throw new Error(`Unknown engine request ${data.type}`);
  }
};

/**
 * One analysis or quick prediction, start to end, once those before it have ended. Starting an
 * analysis lets go of the car analysed last and tells the page (`released`); a quick prediction
 * leaves it held. The page lets go of aborted and failed analyses itself.
 */
async function run(job, request) {
  if (!job.aborted) {
    running = job;
    if (request.type === 'analyze') {
      if (lastCar) self.postMessage({ type: 'released', id: lastCar.id });
      lastCar = null;
      await analyze(job, request);
    } else {
      await predictQuick(job, request.surface);
    }
    running = null;
    engine.then(({ runners }) => runners.pool.broadcast({ type: 'release' }), () => {});
  }
  jobs.delete(job.id);
}

function abort(current) {
  current.aborted = true;
  current.controller.abort(new Aborted());
  if (current === running) engine.then(({ runners }) => runners.pool.clear(new Aborted()), () => {});
}

function checkpoint(current) {
  if (current.aborted) throw new Aborted();
}

/** A full analysis of `file`, or of `surface` (arrays) when there is no file. */
async function analyze(current, input) {
  const post = (message, transfer = []) => {
    if (!current.aborted) self.postMessage({ ...message, id: current.id }, transfer);
  };
  try {
    const { ensemble, layout, runners } = await engine;
    await replaceLostGpu(runners, ensemble);
    const device = deviceName(runners.gpu);
    const stages = stageClock(post);

    stages.begin('reading');
    const surface = input.file ? await readSurface(input.file, stages.progress) : checkSurface(input.surface);
    checkpoint(current);

    stages.begin('measuring');
    const car = await measure(surface, layout, runners.grids, stages.progress);
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

    const estimate = new SurfaceTally(runners.networks, car.frontal, cells);
    const agreement = surface.cfd && new Agreement();
    // The picks and their distinct cells both ascend, so one walk finds each pick's prediction.
    const { models: pickModels, fields: pickFields } = pickBatchDone;
    for (let i = 0, j = 0; i < picks.picks.length; i++) {
      const c = picks.picks[i];
      while (pickedCells[j] < c) j++;
      estimate.add(picks.weight, c, pickModels, j, pickedCells.length, pickFields.cp[j]);
      agreement?.add(picks.weight, pickFields.cp[j], surface.cfd[c]);
    }
    const reference = ownReference(surface);
    const result = {
      form: 'full',
      source: { name: surface.name, cells: cells.areas.length },
      geometry: geometryOf(car),
      ...surfaceNumbers(estimate, agreement ? agreement.accuracy() : null),
      reference,
      referenceTotals: car.referenceTotals,
      coverage: coverage(car.bounds, ensemble.card.dimensions),
      timings: stages.timings,
      device,
      model: ensemble.weightsSha256,
      mesh,
      lookups: lookupsOf(car.context, ensemble.features),
    };
    lastCar = {
      id: current.id, file: input.file ?? null, surface, cells, context: car.context, frontal: car.frontal, predicted,
      index: null, complete: false, sourcePoint: working.sourcePoint, showFull: Boolean(full), ownCfd: surface.cfd, reference,
    };
    post({ type: 'result', result });
    lastCar.index = indexPolygons(surface, cells);

    const cellsPerSecond = cellsToPredict / (pickBatchDone.seconds + ringBatch.seconds);
    // The background prediction is the rest of this analysis: the next one waits for it.
    const complete = await completeSurface(current, runners, post, {
      surface, car, working: { ...mesh, sourcePoint: working.sourcePoint },
      batches: [pickBatchDone, ringBatch], predicted, showFull: Boolean(full),
      affordable: cells.areas.length / cellsPerSecond <= FULL_PREDICTION_SECONDS,
    }).catch((error) => {
      if (!(error instanceof Aborted)) console.error(error);
    });
    if (complete && lastCar?.id === current.id) lastCar.complete = true;
  } catch (error) {
    if (error instanceof Aborted || current.aborted) return;
    post({ type: 'failed', message: describeFailure(error, input) });
    // Work still queued for this analysis (the area picks) has no one waiting for it.
    abort(current);
  }
}

/**
 * Every polygon of a surface given as arrays predicted at once, with the lookups built from those
 * polygons: the quick form of an analysis (`Result.form`), for studies that compare many variants
 * of a working mesh. Its result carries the polygons themselves with their predictions (`cells`).
 */
async function predictQuick(current, given) {
  const post = (message, transfer = []) => {
    if (!current.aborted) self.postMessage({ ...message, id: current.id }, transfer);
  };
  try {
    const { ensemble, layout, runners } = await engine;
    await replaceLostGpu(runners, ensemble);
    const started = performance.now();
    const surface = checkSurface(given);
    const car = await measure(surface, layout, runners.grids, () => {});
    runners.pool.broadcast({ type: 'context', context: car.context });
    const measured = performance.now();

    const { cells } = car;
    const count = cells.areas.length;
    const batch = await predictBatch(current, runners, cells, Uint32Array.from({ length: count }, (_, c) => c), () => {});
    checkpoint(current);
    const { cp, sigma } = batch.fields;
    const tally = new SurfaceTally(runners.networks, car.frontal, cells);
    for (let c = 0; c < count; c++) tally.add(cells.areas[c], c, batch.models, c, count, cp[c]);
    const accuracy = surface.cfd ? surfaceAgreement(cells.areas, cp, surface.cfd) : null;
    const drawn = surfaceMesh(surface, car.flip);
    const mesh = {
      ...displayMesh({ ...drawn, cfd: surface.cfd && car.average.all(surface.cfd) }),
      cp: car.average.all(cp),
      sigma: car.average.all(sigma),
    };
    const result = {
      form: 'quick',
      source: { name: surface.name, cells: count },
      geometry: geometryOf(car),
      ...surfaceNumbers(tally, accuracy),
      reference: ownReference(surface),
      referenceTotals: car.referenceTotals,
      coverage: coverage(car.bounds, ensemble.card.dimensions),
      timings: [
        { stage: 'measuring', seconds: (measured - started) / 1000 },
        { stage: 'predicting', seconds: (performance.now() - measured) / 1000 },
      ],
      device: deviceName(runners.gpu),
      model: ensemble.weightsSha256,
      mesh,
      lookups: lookupsOf(car.context, ensemble.features),
      cells: {
        points: surface.points, connectivity: surface.connectivity, offsets: surface.offsets,
        centers: cells.centers, normals: cells.normals, areas: cells.areas, cp, sigma, cfd: surface.cfd,
      },
    };
    // `mesh.positions` is `cells.points`: the clone keeps them one array, and each buffer moves once.
    const arrays = [...Object.values(result.cells), mesh.normals, mesh.indices, mesh.cp, mesh.sigma, mesh.cfd];
    post({ type: 'result', result }, [...new Set(arrays.filter(Boolean).map((array) => array.buffer))]);
  } catch (error) {
    if (error instanceof Aborted || current.aborted) return;
    post({ type: 'failed', message: describeFailure(error, { surface: given }) });
    abort(current);
  }
}

/** The Result's `geometry` of a measured car. */
function geometryOf({ bounds, frontal }) {
  return {
    bounds,
    length: bounds[1] - bounds[0],
    width: bounds[3] - bounds[2],
    height: bounds[5] - bounds[4],
    frontalArea: frontal,
  };
}

/**
 * Polygon geometry, the frontal area, the feature lookups, the polygon-to-point operator and the
 * reference field's whole-car totals.
 */
async function measure(surface, layout, grids, progress) {
  const cells = measureCells(surface);
  progress(0.5);
  const frontal = frontalArea(surface);
  progress(0.6);
  const context = { layout, ...(await buildLookups(surface, cells, layout, grids)) };
  progress(0.8);
  const average = cellToPoint(surface, cells.areas);
  progress(0.95);
  return {
    cells, context, average, frontal, flip: cells.flip, bounds: cells.bounds,
    referenceTotals: surface.cfd ? referenceTotals(cells, surface.cfd, frontal) : null,
  };
}

/** The Result's surface numbers: the forces and totals of a tally over polygons, and the agreement with CFD. */
function surfaceNumbers(tally, accuracy) {
  return { forces: tally.forces(), accuracy, totals: tally.totals() };
}

/** `Result.reference` for a surface's own CFD, as its file or its arrays carried it. */
function ownReference(surface) {
  return surface.cfd ? { file: surface.name, array: CFD_ARRAY, mapping: 'own' } : null;
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

/** The contract's mesh fields only (the working mesh also carries `sourcePoint`), with its bounds. */
function displayMesh({ positions, normals, indices, cfd }) {
  return { positions, normals, indices, bounds: boundsOf(positions), ...(cfd && { cfd }) };
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
 * @returns {Promise<boolean>} whether every polygon was predicted
 */
async function completeSurface(current, runners, post, job) {
  const { surface, car, working, batches, predicted, showFull, affordable } = job;
  const { cells } = car;
  const { points, connectivity, offsets } = surface;
  const cellCount = cells.areas.length;
  const pointCount = points.length / 3;
  let predictedCount = batches.reduce((sum, batch) => sum + batch.cells.length, 0);

  // The full mesh's fields, kept current as polygons are predicted: each file point holds the
  // area-weighted mean of its polygons predicted so far, or until the first one its nearest
  // working-mesh vertex's value. A `fullFields` posts a copy, so the worker stays free for picks.
  let field = null;
  const accumulate = (c, cp, sigma) => {
    const area = cells.areas[c];
    for (let i = offsets[c]; i < offsets[c + 1]; i++) {
      const p = connectivity[i];
      const share = area / (field.area[p] + area);
      field.cp[p] += (cp - field.cp[p]) * share;
      field.sigma[p] += (sigma - field.sigma[p]) * share;
      field.area[p] += area;
    }
  };
  let lastSent = 0;
  const send = () => {
    const cp = field.cp.slice();
    const sigma = field.sigma.slice();
    post({ type: 'fullFields', fields: { cp, sigma }, fraction: predictedCount / cellCount }, [cp.buffer, sigma.buffer]);
    lastSent = performance.now();
  };
  let nearest = null;
  if (showFull) {
    nearest = await nearestWorkingVertices(runners.pool, points, working.positions);
    checkpoint(current);
    field = { cp: new Float32Array(pointCount), sigma: new Float32Array(pointCount), area: new Float32Array(pointCount) };
    for (let p = 0; p < pointCount; p++) {
      field.cp[p] = working.cp[nearest[p]];
      field.sigma[p] = working.sigma[nearest[p]];
    }
    for (const batch of batches) batch.cells.forEach((c, j) => accumulate(c, batch.fields.cp[j], batch.fields.sigma[j]));
    send();
  }
  if (!affordable) return false;

  const exact = new SurfaceTally(runners.networks, car.frontal, cells);
  for (const batch of batches) {
    batch.cells.forEach((c, j) => exact.add(cells.areas[c], c, batch.models, j, batch.cells.length, batch.fields.cp[j]));
  }
  await requestsTurn();
  let order;
  if (nearest) {
    const variation = edgeVariation(working);
    order = refinementOrder(cells.centers, (c) => variation[nearest[connectivity[offsets[c]]]]);
  } else {
    order = Uint32Array.from({ length: cellCount }, (_, c) => c);
  }
  await requestsTurn();
  const remaining = new Uint32Array(cellCount - predictedCount);
  for (let i = 0, k = 0; i < cellCount; i++) if (!predicted.done[order[i]]) remaining[k++] = order[i];

  await predictPoints(current, runners, cells, remaining, (start, size, chunk) => {
    const fields = ensembleStatistics(chunk, size, runners.spreadScale);
    for (let i = 0; i < size; i++) {
      const c = remaining[start + i];
      predicted.cp[c] = fields.cp[i];
      predicted.sigma[c] = fields.sigma[i];
      predicted.done[c] = 1;
      exact.add(cells.areas[c], c, chunk, i, size, fields.cp[i]);
      if (field) accumulate(c, fields.cp[i], fields.sigma[i]);
    }
    predictedCount += size;
    if (showFull && predictedCount < cellCount && performance.now() - lastSent > FULL_FIELDS_INTERVAL_MS) send();
  }, true);
  const accuracy = surface.cfd ? surfaceAgreement(cells.areas, predicted.cp, surface.cfd) : null;
  post({ type: 'exact', numbers: surfaceNumbers(exact, accuracy) });
  if (showFull) send();
  return true;
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

/** Lets requests already waiting (picks, explanations) run before the next long step. */
function requestsTurn() {
  return new Promise((resolve) => setTimeout(resolve, 0));
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
 * the result's file (the one analysed last), the networks are asked at the polygon `polygonAt`
 * finds, with its normal, as in training and in the analysis. Otherwise they are asked at the vertex
 * itself, with its polygon-averaged normal, from the lookups the page sends with the first request
 * about each result.
 */
async function explain({ id, point, normal, analysis, model, lookups, vertex }) {
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
    const car = heldCar(analysis);
    const cell = car ? polygonAt(car, point, normal) : -1;
    if (cell >= 0) {
      const { cells, surface } = car;
      site = {
        at: 'polygon', index: cell, position: cells.centers.slice(3 * cell, 3 * cell + 3),
        normal: cells.normals.slice(3 * cell, 3 * cell + 3), cfd: surface.cfd ? surface.cfd[cell] : null,
      };
      context = car.context;
    } else {
      site = { at: 'vertex', index: vertex.index, position: Float32Array.from(vertex.position), normal: Float32Array.from(vertex.normal), cfd: vertex.cfd };
      context = lookupsAtVertex(explainedLookups, vertex.index);
    }
    const features = new Float32Array(layout.count);
    computeFeatures(context, site.position, site.normal, 0, 1, features);
    const { cpModels, activations } = networkActivations(ensemble, features);
    const { cp, sigma } = ensembleStatistics(cpModels, 1, ensemble.spreadScale);
    const explanation = {
      ...site,
      position: Array.from(site.position),
      normal: Array.from(site.normal),
      features,
      cpModels: Array.from(cpModels),
      cp: cp[0],
      sigma: sigma[0],
      activations,
    };
    self.postMessage({ type: 'answer', id, answer: explanation });
  } catch (error) {
    self.postMessage({ type: 'answerFailed', id, message: error.message });
  }
}

/** The `FilePolygon` under a point picked on the car analysed last; null once that car is let go. */
async function answerPolygonAt({ id, analysis, point, normal }) {
  try {
    const { ensemble } = await engine;
    const car = heldCar(analysis);
    const cell = car ? polygonAt(car, point, normal) : -1;
    self.postMessage({ type: 'answer', id, answer: cell < 0 ? null : describePolygon(car, ensemble, cell) });
  } catch (error) {
    self.postMessage({ type: 'answerFailed', id, message: error.message });
  }
}

/** The car this worker holds for `analysis`, or null: a bundled example has no analysis. */
function heldCar(analysis) {
  return lastCar && lastCar.id === analysis ? lastCar : null;
}

/**
 * The `CellSet` of the car analysed last, copied here and handed over whole; null unless every
 * polygon is predicted. About 650 MB for a DrivAerML car, so it is made only when asked for.
 */
function answerFileCells({ id, analysis }) {
  const car = heldCar(analysis);
  if (!car?.complete) {
    self.postMessage({ type: 'answer', id, answer: null });
    return;
  }
  const { surface, cells, predicted } = car;
  const set = {
    points: surface.points.slice(),
    connectivity: surface.connectivity.slice(),
    offsets: surface.offsets.slice(),
    centers: cells.centers.slice(),
    normals: cells.normals.slice(),
    areas: cells.areas.slice(),
    cp: predicted.cp.slice(),
    sigma: predicted.sigma.slice(),
    cfd: surface.cfd ? surface.cfd.slice() : null,
  };
  self.postMessage({ type: 'answer', id, answer: set }, Object.values(set).filter(Boolean).map((array) => array.buffer));
}

/**
 * A reference field for the car analysed last: the cell array `array` of `file`, put on the car's
 * polygons cell for cell when the two files have the same polygons, and otherwise, once `file` is
 * found to describe the same surface, from the polygon of `file` whose centre is nearest, among those
 * not facing away (any, where all do). The car then holds it as its own CFD; of `file`, only that
 * array is kept.
 */
async function attachReference({ id, analysis, file, array = CFD_ARRAY }) {
  const progress = (fraction) => self.postMessage({ type: 'answerProgress', id, fraction });
  try {
    const car = completeCar(analysis);
    const names = await cellArrayNames(file);
    if (!names.includes(array)) {
      throw new Error(names.length
        ? `${file.name} has no cell array named ${array}. Its cell arrays of one number per polygon are ${names.join(', ')}.`
        : `${file.name} has no cell array of one number per polygon, so it holds no pressure field to compare with.`);
    }
    const surface = await readSurface(file, (fraction) => progress(0.8 * fraction), array);
    let mapping = 'polygons';
    if (samePolygons(car.surface, surface)) {
      car.surface.cfd = surface.cfd;
    } else {
      const cells = measureCells(surface);
      const reference = { surface, cells, index: indexPolygons(surface, cells) };
      const apart = medianDistance(car, reference);
      if (apart > SAME_SURFACE_DISTANCE) {
        throw new Error(`${file.name} is not this car's surface: its polygons lie a median ${(apart * 1000).toFixed(1)} mm from this car's, `
          + `and a field for this car must lie within ${SAME_SURFACE_DISTANCE * 1000} mm. Attach the CFD computed on this geometry.`);
      }
      car.surface.cfd = await nearestCellValues(car.cells, reference, (fraction) => progress(0.8 + 0.2 * fraction));
      mapping = 'nearest';
    }
    car.reference = { file: file.name, array, mapping };
    answerReference(id, car);
  } catch (error) {
    self.postMessage({ type: 'answerFailed', id, message: error.message });
  }
}

/** The car analysed last back to the CFD its own file or arrays carried, or none. */
function detachReference({ id, analysis }) {
  try {
    const car = completeCar(analysis);
    car.surface.cfd = car.ownCfd;
    car.reference = ownReference(car.surface);
    answerReference(id, car);
  } catch (error) {
    self.postMessage({ type: 'answerFailed', id, message: error.message });
  }
}

/** The car held for `analysis` once every polygon is predicted, which a reference needs. */
function completeCar(analysis) {
  const car = heldCar(analysis);
  if (!car?.complete) {
    throw new Error('A reference can be attached to the car analysed last only, once every polygon is predicted. Analyse it again.');
  }
  return car;
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
async function nearestCellValues(cells, { surface, cells: { normals }, index: { grid } }, progress) {
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
    // Explanations and picks asked meanwhile are answered between slices.
    if (c % 262_144 === 262_143) {
      progress(c / count);
      await requestsTurn();
    }
  }
  return values;
}

/**
 * What a reference changes in the car's Result, as `attachReference` and `detachReference` resolve
 * with it: the reference field's whole-car totals and its agreement over every polygon, computed as the
 * analysis computes them, and the field on the working mesh and the full mesh, put on their vertices as
 * the meshes put a file's own.
 */
function answerReference(id, car) {
  const { surface, cells, predicted } = car;
  let answer = { reference: null, referenceTotals: null, accuracy: null, cfd: null, fullCfd: null };
  if (surface.cfd) {
    const all = cellToPoint(surface, cells.areas).all(surface.cfd);
    answer = {
      reference: car.reference,
      referenceTotals: referenceTotals(cells, surface.cfd, car.frontal),
      accuracy: surfaceAgreement(cells.areas, predicted.cp, surface.cfd),
      cfd: Float32Array.from(car.sourcePoint, (p) => all[p]),
      fullCfd: car.showFull ? all : null,
    };
  }
  self.postMessage({ type: 'answer', id, answer }, [answer.cfd, answer.fullCfd].filter(Boolean).map((array) => array.buffer));
}

/**
 * The analysed car with its prediction added as cell arrays: `CpPredicted`, the networks' mean, and
 * `CpSpread`, σ(Cp). A file gets them beside its own arrays (DrivAerML's CFD pressure coefficient is
 * `CpMeanTrim`), with an attached reference under its array's name; a surface that came as arrays is
 * written as a new file, with its reference too.
 */
async function exportVtp({ id, analysis }) {
  const car = heldCar(analysis);
  try {
    if (!car?.complete) {
      throw new Error('This car\'s prediction is no longer held, so it cannot be exported. Analyse it again.');
    }
    const { reference, surface } = car;
    const written = reference && (!car.file || reference.mapping !== 'own') ? { [reference.array]: surface.cfd } : {};
    const arrays = { ...written, CpPredicted: car.predicted.cp, CpSpread: car.predicted.sigma };
    const name = `${surface.name.replace(/\.vtp$/i, '')}_prediction.vtp`;
    if (!car.file) {
      self.postMessage({ type: 'exported', id, file: writeVtp({ ...surface, name }, arrays) });
      return;
    }
    let reported = 0;
    const vtp = await addCellArrays(car.file, arrays, (fraction) => {
      if (fraction >= reported + 0.01) {
        reported = fraction;
        self.postMessage({ type: 'exportProgress', id, fraction });
      }
    });
    self.postMessage({ type: 'exported', id, file: new File([vtp], name, { type: 'application/octet-stream' }) });
  } catch (error) {
    // Reading the file again fails with a DOMException when it was moved or changed after the drop.
    const message = error instanceof DOMException
      ? `${car.file.name} could not be read again; it may have been moved or changed since it was opened. Open it again to export its prediction.`
      : error.message;
    self.postMessage({ type: 'exportFailed', id, message });
  }
}

/** The message for a failed analysis: the file's own problem as the reader words it, or what ran out. */
function describeFailure(error, { file, surface }) {
  if (error instanceof RangeError && /allocat|memory/i.test(error.message)) {
    const what = file
      ? `${file.name} (${(file.size / 1e6).toFixed(0)} MB)`
      : `${surface.name} (${(surface.offsets.length - 1).toLocaleString('en-US')} polygons)`;
    return `This device ran out of memory analysing ${what}. Close other tabs or windows and try again, or use a machine with more memory.`;
  }
  return error.message;
}
