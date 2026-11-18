/**
 * The engine as the page sees it: `createEngine()` returns the `Engine` of `result.js`, backed by one
 * module worker (`worker.js`) that reads the file or takes the arrays given, measures the car, builds
 * the meshes and runs the networks, so the page never stalls on any of it.
 */

import { workingCells } from './result.js';

/** @returns {import('./result.js').Engine} */
export function createEngine() {
	const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
	let nextId = 1;
	/**
	 * Analyses and quick predictions the page has not let go, by id: those waiting their turn in the
	 * worker, the one it runs, and the last analysis, whose car the worker holds (for `explain` and
	 * `polygonAt`, and once `complete`, `exportVtp`, `fileCells` and `attachReference`).
	 * @type {Map<number, { id: number, on: import('./result.js').AnalyzeEvents, resolve: Function, reject: Function, settled: boolean, complete: boolean, release: () => void }>}
	 */
	const analyses = new Map();
	/**
	 * Requests awaiting the worker's answer (`explain`, `polygonAt`, `fileCells`, `exportVtp`,
	 * `attachReference`), by id.
	 * @type {Map<number, { resolve: Function, reject: Function }>}
	 */
	const requests = new Map();
	/** Set when the worker cannot run at all: every analysis fails with it. */
	let unavailable = null;
	const device = deferred();
	const model = deferred();
	/** Analysis ids of the results this engine made, by their lookups, which survive the page copying a result. */
	const analysisOfLookups = new WeakMap();
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
			case 'answer':
			case 'answerFailed': {
				const request = requests.get(data.id);
				requests.delete(data.id);
				if (data.type === 'answer') request.resolve(data.answer);
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
				current.on.geometry?.(data.mesh);
				current.on.fullMesh?.(data.mesh);
				break;
			case 'result':
				current.settled = true;
				analysisOfLookups.set(data.result.lookups, current.id);
				current.resolve(data.result);
				// A quick prediction ends with its result: the worker keeps nothing of it.
				if (data.result.form === 'quick') finish(current);
				break;
			case 'fullFields':
				current.on.fullFields?.(data.fields, data.fraction);
				break;
			case 'exact':
				current.complete = true;
				current.on.exact?.(data.numbers);
				break;
			case 'failed':
				finish(current, new Error(data.message));
				break;
			// The worker has let go of this car: the next analysis has started.
			case 'released':
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
		for (const request of requests.values()) request.reject(error);
		requests.clear();
	};

	/** Let go of an analysis: reject it if still pending, stop its events and detach its abort listener. */
	function finish(current, error) {
		if (!current.settled) current.reject(error);
		current.settled = true;
		current.release();
		analyses.delete(current.id);
	}

	function abortError() {
		return new DOMException('The analysis was cancelled.', 'AbortError');
	}

	function analyze(file, on = {}, options = {}) {
		return start({ type: 'analyze', file }, on, options);
	}

	function predictQuick(surface, options = {}) {
		let transfer;
		try {
			transfer = surfaceBuffers(surface);
		} catch (error) {
			return Promise.reject(error);
		}
		return start({ type: 'predictQuick', surface }, {}, options, transfer);
	}

	/** Queue a request the worker answers with a `Result`; `transfer` moves to the worker. */
	function start(request, on, { signal } = {}, transfer = []) {
		if (unavailable) return Promise.reject(unavailable);
		if (signal?.aborted) return Promise.reject(abortError());
		const id = nextId++;
		return new Promise((resolve, reject) => {
			const onAbort = () => {
				worker.postMessage({ type: 'abort', id });
				finish(current, abortError());
			};
			const current = {
				id,
				on,
				resolve,
				reject,
				settled: false,
				complete: false,
				release: () => signal?.removeEventListener('abort', onAbort),
			};
			signal?.addEventListener('abort', onAbort, { once: true });
			analyses.set(id, current);
			worker.postMessage({ ...request, id }, transfer);
		});
	}

	function explain(result, point, normal) {
		const vertex = nearestVertex(result.mesh, point, normal);
		const message = {
			type: 'explain',
			point,
			normal,
			analysisId: analysisOfLookups.get(result.lookups),
			model: result.model,
			vertex,
		};
		if (result.lookups !== lookupsSent) {
			message.lookups = result.lookups;
			lookupsSent = result.lookups;
		}
		return ask(message);
	}

	function polygonAt(result, point, normal) {
		const analysisId = analysisOfLookups.get(result.lookups);
		// The worker holds the polygons of the last analysis from its result until the next one starts.
		if (!analyses.get(analysisId)?.settled) return Promise.resolve(null);
		return ask({ type: 'polygonAt', analysisId, point, normal });
	}

	function ask(message) {
		const id = nextId++;
		return new Promise((resolve, reject) => {
			requests.set(id, { resolve, reject });
			worker.postMessage({ ...message, id });
		});
	}

	/** The analysis behind a result, while the worker still holds its car with every polygon predicted. */
	function completeAnalysis(result) {
		const current = analyses.get(analysisOfLookups.get(result.lookups));
		return current?.complete ? current : null;
	}

	function fileCells(result) {
		const current = completeAnalysis(result);
		return current ? ask({ type: 'fileCells', analysisId: current.id }) : Promise.resolve(null);
	}

	function canExportVtp(result) {
		return completeAnalysis(result) !== null;
	}

	function exportVtp(result) {
		const current = completeAnalysis(result);
		if (!current)
			return Promise.reject(
				new Error(
					'This result cannot be exported: only the car analysed last can be, once every polygon is predicted.',
				),
			);
		return ask({ type: 'export', analysisId: current.id });
	}

	/** Ask the worker to give the car a reference field, then write what that changes into `result` itself. */
	async function attachReference(result, file, { array } = {}) {
		const current = completeAnalysis(result);
		if (!current)
			throw new Error(
				'A reference can be attached to the car analysed last only, once every polygon is predicted. Analyse it again.',
			);
		const { reference, referenceTotals, accuracy, cfd, fullCfd } = await ask({
			type: 'attachReference',
			analysisId: current.id,
			file,
			array,
		});
		result.reference = reference;
		result.referenceTotals = referenceTotals;
		result.accuracy = accuracy;
		result.mesh.cfd = cfd;
		return { cfd, fullCfd };
	}

	return {
		analyze,
		predictQuick,
		workingCells,
		fileCells,
		explain,
		polygonAt,
		canExportVtp,
		exportVtp,
		attachReference,
		device: device.promise,
		model: model.promise,
	};
}

/**
 * The buffers of a surface's arrays, which move to the worker rather than being copied: each must
 * be a whole buffer of its own, or moving it would take memory the caller still uses. Arrays that
 * are not typed arrays are left to the worker's check, which says what they should be.
 */
function surfaceBuffers({ points, connectivity, offsets, cfd }) {
	const arrays = [points, connectivity, offsets, cfd].filter((array) => ArrayBuffer.isView(array));
	for (const array of arrays) {
		if (array.byteOffset !== 0 || array.byteLength !== array.buffer.byteLength) {
			throw new Error(
				"The surface's arrays move to the engine rather than being copied, so each must be a whole buffer of its own: pass a copy (array.slice()).",
			);
		}
	}
	return arrays.map((array) => array.buffer);
}

/**
 * The working-mesh vertex nearest `point` among those not facing away from `normal`, so a point on
 * one side of a thin panel stays on that side; with its position, normal and CFD Cp.
 */
function nearestVertex({ positions, normals, cfd }, point, normal) {
	let index = 0;
	let best = Infinity;
	for (let v = 0; v < positions.length / 3; v++) {
		if (normals[3 * v] * normal[0] + normals[3 * v + 1] * normal[1] + normals[3 * v + 2] * normal[2] < 0) continue;
		const distance =
			(positions[3 * v] - point[0]) ** 2 +
			(positions[3 * v + 1] - point[1]) ** 2 +
			(positions[3 * v + 2] - point[2]) ** 2;
		if (distance < best) {
			best = distance;
			index = v;
		}
	}
	return {
		index,
		position: Array.from(positions.subarray(3 * index, 3 * index + 3)),
		normal: Array.from(normals.subarray(3 * index, 3 * index + 3)),
		cfd: cfd ? cfd[index] : null,
	};
}

/** A promise with its settling functions; nobody awaiting its rejection is not an error. */
function deferred() {
	const settle = {};
	settle.promise = new Promise((resolve, reject) => Object.assign(settle, { resolve, reject }));
	settle.promise.catch(() => {});
	return settle;
}
