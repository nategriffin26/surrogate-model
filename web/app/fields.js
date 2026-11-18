// The fields the car can be coloured by, how each is read from a mesh, and the colour maps that suit it. The Result
// view offers pressure, its drag contribution and the spread between the networks; CFD pressure and the error against
// it are defined here for the Model view, and share the pressure colour map so they read on the same scale.

import { niceTicks } from './format.js';

/** `map` names the colour map the field uses; `ramps` are the ramps offered for it, the first being the default. */
export const FIELDS = {
	cp: { label: 'Pressure', title: 'Pressure Cp', symbol: 'Cp', map: 'cp', tip: 'Pressure coefficient, mean of the networks' },
	drag: { label: 'Drag', title: 'Drag −Cp nₓ', symbol: '−Cp nₓ', map: 'drag', tip: 'Local pressure drag: −Cp times the x component of the outward normal' },
	sigma: { label: 'Spread', title: 'Spread σ(Cp)', symbol: 'σ', map: 'sigma', tip: 'Standard deviation of the networks’ Cp, scaled by the model’s calibration' },
	cfd: { label: 'CFD', title: 'Cp CFD', symbol: 'Cp', map: 'cp', tip: 'Pressure coefficient from the CFD solution in the file' },
	error: { label: 'Error', title: 'ΔCp predicted − CFD', symbol: 'ΔCp', map: 'error', tip: 'Predicted Cp minus CFD Cp' },
	none: { label: 'None', title: null, symbol: null, map: null, tip: 'Surface only' },
};

/** The fields the Result view offers, in toolbar order. */
export const RESULT_FIELDS = ['cp', 'drag', 'sigma'];

/** Fields added at run time, such as an annotation's array, by key, in the order added. */
const ADDED = new Set();

/**
 * Add a field the app supplies: its FIELDS entry under `key` (label, title, symbol, tip) and its colour map
 * {ramps, diverging, start}: the ramps offered, whether it diverges about zero, and the scale it starts at
 * ({ramp, lo, hi, bands}), which resetting its scale returns to.
 */
export function addField(key, field, map) {
	FIELDS[key] = { ...field, map: key };
	MAPS[key] = map;
	ADDED.add(key);
}

export function removeField(key) {
	delete FIELDS[key];
	delete MAPS[key];
	ADDED.delete(key);
}

/** Every field the Result view can be coloured by: the model's, then those added. */
export const offeredFields = () => [...RESULT_FIELDS, ...ADDED];

export const MAPS = {
	cp: { ramps: ['coolwarm', 'turbo', 'viridis'], diverging: true },
	error: { ramps: ['coolwarm', 'purplegreen', 'brownteal'], diverging: true },
	sigma: { ramps: ['purples', 'viridis', 'inferno'], diverging: false },
	drag: { ramps: ['tealorange', 'coolwarm', 'brownteal'], diverging: true },
};

/** A field's value at vertex v of a mesh carrying cp, sigma, cfd and normals. */
export function valueAt(mesh, field, v) {
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

/** The `quantile` of |value| over an even sample of about 20,000 of the mesh's vertices: ample for a 98 % range end. */
function magnitudeQuantile(mesh, field, quantile) {
	const count = mesh.positions.length / 3;
	const step = Math.max(1, Math.floor(count / 20000));
	// A typed array sorts numerically and natively, several times faster than an array with a comparator.
	const sample = new Float32Array(Math.ceil(count / step));
	for (let v = 0, i = 0; v < count; v += step) sample[i++] = Math.abs(valueAt(mesh, field, v));
	sample.sort();
	return sample[Math.floor(quantile * (sample.length - 1))];
}

/** Round a range end up to a value the legend can label cleanly. */
function roundUp(x) {
	const { step } = niceTicks(0, x, 4);
	return Math.ceil(x / step - 1e-9) * step;
}

/**
 * Starting colour maps for a mesh. Pressure uses a fixed, symmetric range so cars compare on one scale; error,
 * spread and drag contribution span 98 % of the surface's values.
 */
export function defaultMaps(mesh) {
	const maps = {
		cp: { ramp: 'coolwarm', lo: -0.9, hi: 0.9, bands: 0 },
		sigma: { ramp: 'purples', lo: 0, hi: roundUp(magnitudeQuantile(mesh, 'sigma', 0.98)), bands: 0 },
		drag: { ramp: 'tealorange', lo: 0, hi: 0, bands: 0 },
	};
	const drag = roundUp(magnitudeQuantile(mesh, 'drag', 0.98));
	Object.assign(maps.drag, { lo: -drag, hi: drag });
	if (mesh.cfd) {
		const error = roundUp(magnitudeQuantile(mesh, 'error', 0.98));
		maps.error = { ramp: 'coolwarm', lo: -error, hi: error, bands: 0 };
	}
	return maps;
}
