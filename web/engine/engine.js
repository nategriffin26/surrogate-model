/**
 * The engine as the page sees it: `createEngine()` returns the `Engine` of `result.js`, backed by one
 * module worker (`worker.js`) that reads the file, measures the car, builds the meshes and runs the
 * networks, so the page never stalls on any of it.
 */

/** @returns {import('./result.js').Engine} */
export function createEngine() {
  const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  let nextId = 1;
  /** Analyses not yet ended, by id: the one the worker runs and those waiting their turn there. */
  /** @type {Map<number, { id: number, on: import('./result.js').AnalyzeEvents, resolve: Function, reject: Function, settled: boolean, release: () => void }>} */
  const analyses = new Map();
  /** @type {Map<number, { resolve: Function, reject: Function }>} */
  const explanations = new Map();
  /** Set when the worker cannot run at all: every analysis fails with it. */
  let unavailable = null;
  const device = deferred();
  const model = deferred();
  /** Analysis ids of the results this engine made, by their lookups, which survive the page copying a result. */
  const analysed = new WeakMap();
  /** The lookups the worker was last sent for `explain`. */
  let lookupsSent = null;

  worker.onmessage = ({ data }) => {
    switch (data.type) {
      case 'device':
        device.resolve(data.device);
        return;
      case 'model':
        model.resolve(data.model);
        return;
      case 'unavailable':
        unavailable = new Error(`The analysis engine could not start: ${data.message}`);
        device.reject(unavailable);
        model.reject(unavailable);
        for (const current of analyses.values()) finish(current, unavailable);
        return;
      case 'explanation':
      case 'explanationFailed': {
        const request = explanations.get(data.id);
        explanations.delete(data.id);
        if (data.type === 'explanation') request.resolve(data.explanation);
        else request.reject(new Error(data.message));
        return;
      }
      default:
        break;
    }
    const current = analyses.get(data.id);
    if (!current) return;
    switch (data.type) {
      case 'stage':
        current.on.stage?.(data.stage, data.fraction);
        break;
      case 'geometry':
        current.on.geometry?.(data.mesh);
        break;
      case 'fullMesh':
        // The full mesh is also the first drawable mesh: one copy serves both events.
        if (data.preview) current.on.geometry?.(data.mesh);
        current.on.fullMesh?.(data.mesh);
        break;
      case 'result':
        current.settled = true;
        analysed.set(data.result.lookups, current.id);
        current.resolve(data.result);
        break;
      case 'fullFields':
        current.on.fullFields?.(data.fields, data.fraction);
        break;
      case 'exact':
        current.on.exact?.(data.numbers);
        break;
      case 'failed':
        finish(current, new Error(data.message));
        break;
      // Its last event has gone out: the worker has moved on to the next analysis.
      case 'finished':
        finish(current);
        break;
      default:
        throw new Error(`Unknown engine message ${data.type}`);
    }
  };

  // A failure the worker could not report itself (a script that failed to load, an uncaught error).
  worker.onerror = (event) => {
    event.preventDefault();
    const error = new Error(`The analysis engine stopped: ${event.message}. Reload the page and try again.`);
    device.reject(error);
    model.reject(error);
    for (const current of analyses.values()) finish(current, error);
    for (const request of explanations.values()) request.reject(error);
    explanations.clear();
  };

  /** End an analysis: reject it if still pending, stop its events and detach its abort listener. */
  function finish(current, error) {
    if (!current.settled) current.reject(error);
    current.settled = true;
    current.release();
    analyses.delete(current.id);
  }

  function abortError() {
    return new DOMException('The analysis was cancelled.', 'AbortError');
  }

  function analyze(file, on = {}, { signal } = {}) {
    if (unavailable) return Promise.reject(unavailable);
    if (signal?.aborted) return Promise.reject(abortError());
    const id = nextId++;
    return new Promise((resolve, reject) => {
      const onAbort = () => {
        worker.postMessage({ type: 'abort', id });
        finish(current, abortError());
      };
      const current = { id, on, resolve, reject, settled: false, release: () => signal?.removeEventListener('abort', onAbort) };
      signal?.addEventListener('abort', onAbort, { once: true });
      analyses.set(id, current);
      worker.postMessage({ type: 'analyze', id, file });
    });
  }

  function explain(result, point, normal) {
    const id = nextId++;
    const vertex = nearestVertex(result.mesh, point, normal);
    const message = { type: 'explain', id, point, analysis: analysed.get(result.lookups), model: result.model, vertex };
    if (result.lookups !== lookupsSent) {
      message.lookups = result.lookups;
      lookupsSent = result.lookups;
    }
    return new Promise((resolve, reject) => {
      explanations.set(id, { resolve, reject });
      worker.postMessage(message);
    });
  }

  return { analyze, explain, device: device.promise, model: model.promise };
}

/**
 * The working-mesh vertex nearest `point` among those not facing away from `normal`, so a point on
 * one side of a thin panel stays on that side; with its position and normal.
 */
function nearestVertex({ positions, normals }, point, normal) {
  let index = 0;
  let best = Infinity;
  for (let v = 0; v < positions.length / 3; v++) {
    if (normals[3 * v] * normal[0] + normals[3 * v + 1] * normal[1] + normals[3 * v + 2] * normal[2] < 0) continue;
    const distance = (positions[3 * v] - point[0]) ** 2 + (positions[3 * v + 1] - point[1]) ** 2 + (positions[3 * v + 2] - point[2]) ** 2;
    if (distance < best) {
      best = distance;
      index = v;
    }
  }
  return { index, position: Array.from(positions.subarray(3 * index, 3 * index + 3)), normal: Array.from(normals.subarray(3 * index, 3 * index + 3)) };
}

/** A promise with its settling functions; nobody awaiting its rejection is not an error. */
function deferred() {
  const settle = {};
  settle.promise = new Promise((resolve, reject) => Object.assign(settle, { resolve, reject }));
  settle.promise.catch(() => {});
  return settle;
}
