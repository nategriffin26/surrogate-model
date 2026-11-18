// Tests for `sampleFields` in web/engine/nearest.js: one mesh's fields read at another's vertices.
// Run with: deno test --allow-read tests/nearest.test.js
import { sampleFields } from '../web/engine/nearest.js';

function assert(condition, message) {
	if (!condition) throw new Error(message);
}

/**
 * A plate 2 m x 1 m and 4 mm thick on a 1 cm grid: a top sheet facing up and a bottom sheet facing
 * down, each with its own pressure.
 */
function thinPlate() {
	const columns = 201,
		rows = 101,
		gap = 0.004;
	const count = 2 * columns * rows;
	const positions = new Float32Array(3 * count),
		normals = new Float32Array(3 * count);
	const cp = new Float32Array(count),
		sigma = new Float32Array(count),
		cfd = new Float32Array(count);
	for (let side = 0; side < 2; side++) {
		for (let i = 0; i < rows; i++) {
			for (let j = 0; j < columns; j++) {
				const v = (side * rows + i) * columns + j,
					x = j / 100,
					y = i / 100;
				positions.set([x, y, side ? -gap / 2 : gap / 2], 3 * v);
				normals.set([0, 0, side ? -1 : 1], 3 * v);
				cp[v] = side ? -1 + x : 0.5 - y;
				sigma[v] = 0.1 + 0.05 * Math.sin(3 * x + y);
				cfd[v] = cp[v] + 0.1 * Math.cos(5 * y);
			}
		}
	}
	return { positions, normals, cp, sigma, cfd, vertices: count };
}

Deno.test('a mesh sampled at its own vertices gives back its own fields', () => {
	const plate = thinPlate();
	const sampled = sampleFields(plate, plate);
	for (const name of ['cp', 'sigma', 'cfd']) {
		assert(
			sampled[name].length === plate.vertices && sampled[name].every((value, v) => value === plate[name][v]),
			`${name} changed`,
		);
	}
});

Deno.test('a point near a thin panel takes the pressure of the side it faces out of', () => {
	// The bottom sheet's points moved 1 mm up, nearer the bottom sheet than the top, but facing up.
	const plate = thinPlate();
	const half = plate.vertices / 2;
	const at = { positions: plate.positions.slice(3 * half), normals: plate.normals.slice(0, 3 * half) };
	for (let v = 0; v < half; v++) at.positions[3 * v + 2] += 0.001;
	const { cp } = sampleFields(plate, at);
	const wrong = cp.findIndex((value, v) => value !== plate.cp[v]);
	assert(
		wrong < 0,
		`the point above bottom vertex ${wrong} took Cp ${cp[wrong]}, not the top side's ${plate.cp[wrong]}`,
	);
});

Deno.test('only the fields the sampled mesh has are given', () => {
	const { positions, normals, cp } = thinPlate();
	assert(
		Object.keys(sampleFields({ positions, normals, cp }, { positions, normals })).join() === 'cp',
		'fields it lacks were given',
	);
});
