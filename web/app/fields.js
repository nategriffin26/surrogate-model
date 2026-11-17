// The fields the car can be coloured by, how each is read from a mesh, and the colour maps that suit it.
// Predicted and CFD pressure share one colour map so they always read on the same scale.

import { niceTicks } from './format.js';

/**
 * `map` names the colour map the field uses; `ramps` are the ramps offered for it, the first being the default.
 * `cfd` marks fields that exist only when the file carried CFD pressure.
 */
export const FIELDS = {
	cp: { label: 'Predicted', title: 'Cp predicted', symbol: 'Cp', map: 'cp', tip: 'Pressure coefficient, mean of the networks' },
	cfd: { label: 'CFD', title: 'Cp CFD', symbol: 'Cp', map: 'cp', cfd: true, tip: 'Pressure coefficient from the CFD solution in the file' },
	error: { label: 'Error', title: 'ΔCp predicted − CFD', symbol: 'ΔCp', map: 'error', cfd: true, tip: 'Predicted Cp minus CFD Cp' },
	sigma: { label: 'Spread', title: 'Model spread σ(Cp)', symbol: 'σ', map: 'sigma', tip: 'Standard deviation of the networks’ Cp' },
	drag: { label: 'Drag', title: 'Drag contribution −Cp nₓ', symbol: '−Cp nₓ', map: 'drag', tip: 'Local pressure drag: −Cp times the x component of the outward normal' },
	none: { label: 'None', title: null, symbol: null, map: null, tip: 'Surface only' },
};

const FIELD_ORDER = ['cp', 'cfd', 'error', 'sigma', 'drag', 'none'];

export const MAPS = {
	cp: { ramps: ['coolwarm', 'turbo', 'viridis'], diverging: true },
	error: { ramps: ['coolwarm', 'purplegreen', 'brownteal'], diverging: true },
	sigma: { ramps: ['purples', 'viridis', 'inferno'], diverging: false },
	drag: { ramps: ['tealorange', 'coolwarm', 'brownteal'], diverging: true },
};

/** Fields a result offers, in toolbar order. */
export function fieldsFor(hasCfd) {
	return FIELD_ORDER.filter((f) => hasCfd || !FIELDS[f].cfd);
}

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

/** The `quantile` of |value| over an even sample of about 50,000 of the mesh's vertices. */
function magnitudeQuantile(mesh, field, quantile) {
	const count = mesh.positions.length / 3;
	const step = Math.max(1, Math.floor(count / 50000));
	const sample = [];
	for (let v = 0; v < count; v += step) sample.push(Math.abs(valueAt(mesh, field, v)));
	sample.sort((a, b) => a - b);
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
