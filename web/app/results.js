// The result card over the view: what an engineer reads off a surface-pressure solution. The pressure drag, where
// it comes from (forward- against rearward-facing surfaces), and the pressure extremes on a side view of the car;
// then the car's reference area and size, the dimensions outside the training range (only when there are any), and
// one line about the run. Hovering a part of the drag picks out its faces on the car; an extreme flies the camera to
// its spot and marks it.

import { h, s } from './dom.js';
import { millions, seconds, signed } from './format.js';

const DIMENSION_LABELS = { length: 'Length', width: 'Width', height: 'Height' };

/** The vertices of highest and lowest Cp on the working mesh, with their positions and normals. */
function extremesOf(mesh) {
	const cp = mesh.cp;
	let max = 0, min = 0, hi = cp[0], lo = cp[0];
	for (let v = 1; v < cp.length; v++) {
		const value = cp[v];
		if (value > hi) {
			hi = value;
			max = v;
		} else if (value < lo) {
			lo = value;
			min = v;
		}
	}
	const at = (v) => ({ cp: mesh.cp[v], point: Array.from(mesh.positions.subarray(3 * v, 3 * v + 3)), normal: Array.from(mesh.normals.subarray(3 * v, 3 * v + 3)) });
	return { max: at(max), min: at(min) };
}
