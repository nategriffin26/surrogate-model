// The fields the car can be coloured by, how each is read from a mesh, and the colour maps that suit it. The Result
// view offers pressure, its drag contribution and the spread between the networks; on a case that carries CFD, the CFD
// pressure, on the pressure colour map so the two read on the same scale, and the prediction's error against it. The
// Model page draws the CFD and the error too.

import { niceTicks } from './format.js';

/**
 * `map` names the colour map the field uses (MAPS). The CFD's fields, like those added, give `values(c)` and
 * `unavailable(c)` as addField describes them, for the readers of a field's values: the probes and the section plot.
 */
export const FIELDS = {
	cp: {
		label: 'Pressure',
		title: 'Pressure Cp',
		symbol: 'Cp',
		map: 'cp',
		tip: 'Pressure coefficient, mean of the networks',
	},
	drag: {
		label: 'Drag',
		title: 'Drag −Cp nₓ',
		symbol: '−Cp nₓ',
		map: 'drag',
		tip: 'Local pressure drag: −Cp times the x component of the outward normal',
	},
	sigma: {
		label: 'Spread',
		title: 'Spread σ(Cp)',
		symbol: 'σ',
		map: 'sigma',
		tip: 'Standard deviation of the networks’ Cp, scaled by the model’s calibration',
	},
	cfd: {
		label: 'CFD',
		title: 'Cp CFD',
		symbol: 'Cp',
		map: 'cp',
		tip: 'Pressure coefficient from the CFD solution',
		values: (c) => c.result.mesh.cfd ?? null,
		unavailable: (c) => (c.result.mesh.cfd ? null : noCfd(c)),
	},
	error: {
		label: 'Error',
		title: 'ΔCp predicted − CFD',
		symbol: 'ΔCp',
		map: 'error',
		tip: 'Predicted Cp minus CFD Cp',
		values: (c) => errorOf(c.result.mesh),
		unavailable: (c) => (c.result.mesh.cfd ? null : noCfd(c)),
	},
	none: { label: 'None', title: null, symbol: null, map: null, tip: 'Surface only' },
};

/** The fields every result offers, in toolbar order. */
export const RESULT_FIELDS = ['cp', 'drag', 'sigma'];
/** The fields a case that carries CFD offers after them. */
export const CFD_FIELDS = ['cfd', 'error'];

const noCfd = (c) =>
	`${c.name} carries no CFD pressure, so it has neither CFD nor Error. A dropped file takes one from a .vtp of the same surface (attach_reference).`;

/** Per CFD field, the predicted Cp minus it at each vertex of the mesh that carries it; worked out once. */
const errors = new WeakMap();
function errorOf(mesh) {
	if (!mesh.cfd) return null;
	if (!errors.has(mesh.cfd))
		errors.set(
			mesh.cfd,
			Float32Array.from(mesh.cp, (cp, v) => cp - mesh.cfd[v]),
		);
	return errors.get(mesh.cfd);
}

/** Fields added at run time, such as an annotation's array, by key, in the order added. */
const ADDED = new Set();

/**
 * Add a field the app supplies: its FIELDS entry under `key` (label, title, symbol, tip, `values(c)`, its value at
 * each working-mesh vertex of case c as a Float32Array, or null where c has none, and optionally `unavailable(c)`, why
 * c does not offer it, or null where it does) and its colour map: either the name of an existing field's map, to draw
 * on that scale, or its own {ramps, diverging, start}: the ramps offered, whether it diverges about zero, and the
 * scale it starts at ({ramp, lo, hi, bands}), which resetting its scale returns to.
 */
export function addField(key, field, map) {
	const own = typeof map !== 'string';
	FIELDS[key] = { ...field, map: own ? key : map };
	if (own) MAPS[key] = map;
	ADDED.add(key);
}

export function removeField(key) {
	if (FIELDS[key]?.map === key) delete MAPS[key];
	delete FIELDS[key];
	ADDED.delete(key);
}

/** Every field the Result view can be coloured by: the model's, the CFD's, then those added. */
export const offeredFields = () => [...RESULT_FIELDS, ...CFD_FIELDS, ...ADDED];

/** Whether a field was added at run time, and so is drawn from the values the app supplies rather than from the mesh. */
export const isAdded = (key) => ADDED.has(key);

export const MAPS = {
	cp: { ramps: ['coolwarm', 'turbo', 'viridis'], diverging: true },
	error: { ramps: ['coolwarm', 'purplegreen', 'brownteal'], diverging: true },
	sigma: { ramps: ['purples', 'viridis', 'inferno'], diverging: false },
	drag: { ramps: ['tealorange', 'coolwarm', 'brownteal'], diverging: true },
};

/** A field's value at vertex v of a mesh carrying cp, sigma, cfd and normals. */
function valueAt(mesh, field, v) {
	switch (field) {
		case 'cp':
			return mesh.cp[v];
		case 'cfd':
			return mesh.cfd[v];
		case 'error':
			return mesh.cp[v] - mesh.cfd[v];
		case 'sigma':
			return mesh.sigma[v];
		case 'drag':
			return -mesh.cp[v] * mesh.normals[3 * v];
		case 'none':
			return null;
		default:
			throw new Error(`Unknown field ${field}`);
	}
}

/** A per-vertex value on face f of a mesh with `indices`: the mean of its three corners, as the viewer draws it at the face's centre. */
export const faceMean = (values, indices, f) =>
	(values[indices[3 * f]] + values[indices[3 * f + 1]] + values[indices[3 * f + 2]]) / 3;

/**
 * The end of a scale spanning 98 % of `count` values (value(i) the i-th): |value| at the 98th percentile of an even
 * sample of about 20,000 of them, rounded up to a value the legend can label cleanly.
 */
export function scaleEnd(count, value) {
	const step = Math.max(1, Math.floor(count / 20000));
	// A typed array sorts numerically and natively, several times faster than an array with a comparator.
	const sample = new Float32Array(Math.ceil(count / step));
	for (let v = 0, i = 0; v < count; v += step) sample[i++] = Math.abs(value(v));
	sample.sort();
	const end = sample[Math.floor(0.98 * (sample.length - 1))];
	const { step: tick } = niceTicks(0, end, 4);
	// Values all zero, as a variant with no change has, take one tick: a scale must span something.
	return Math.ceil(end / tick - 1e-9) * tick || tick;
}

/**
 * Starting colour maps for a mesh. Pressure uses a fixed, symmetric range so cars compare on one scale; error,
 * spread and drag contribution span 98 % of the surface's values.
 */
export function defaultMaps(mesh) {
	const count = mesh.positions.length / 3;
	const end = (field) => scaleEnd(count, (v) => valueAt(mesh, field, v));
	const drag = end('drag');
	const maps = {
		cp: { ramp: 'coolwarm', lo: -0.9, hi: 0.9, bands: 0 },
		sigma: { ramp: 'purples', lo: 0, hi: end('sigma'), bands: 0 },
		drag: { ramp: 'tealorange', lo: -drag, hi: drag, bands: 0 },
	};
	if (mesh.cfd) {
		const error = end('error');
		maps.error = { ramp: 'coolwarm', lo: -error, hi: error, bands: 0 };
	}
	return maps;
}
