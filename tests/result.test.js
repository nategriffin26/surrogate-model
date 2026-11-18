// Round trip of web/engine/result.js's pack format on a synthetic mesh, and `loadResult` on a bundled
// example. Run with: deno test --allow-read tests/result.test.js
import { loadResult, packResult, unpackResult } from '../web/engine/result.js';
import { scaleReadsAt } from '../web/engine/features.js';

/** A wavy 3 m x 1 m sheet on a 1 cm grid, with smooth pressure-like fields on its vertices. */
function wavySheet(withCfd) {
	const columns = 301,
		rows = 101;
	const positions = new Float32Array(3 * columns * rows);
	const normals = new Float32Array(3 * columns * rows);
	const cp = new Float32Array(columns * rows),
		sigma = new Float32Array(columns * rows),
		cfd = new Float32Array(columns * rows);
	for (let i = 0; i < rows; i++) {
		for (let j = 0; j < columns; j++) {
			const v = i * columns + j,
				x = j / 100,
				y = i / 100;
			const z = 0.1 * Math.sin(3 * x) * Math.cos(2 * y);
			positions.set([x - 0.7, y - 0.5, z], 3 * v);
			const dx = 0.3 * Math.cos(3 * x) * Math.cos(2 * y),
				dy = -0.2 * Math.sin(3 * x) * Math.sin(2 * y);
			const length = Math.hypot(dx, dy, 1);
			normals.set([-dx / length, -dy / length, 1 / length], 3 * v);
			cp[v] = 1.0 - 3.5 * Math.sin(x) ** 2 * Math.cos(y);
			sigma[v] = 0.2 + 0.2 * Math.sin(5 * x + y);
			cfd[v] = cp[v] + 0.05 * Math.cos(7 * y);
		}
	}
	const indices = [];
	for (let i = 0; i + 1 < rows; i++) {
		for (let j = 0; j + 1 < columns; j++) {
			const a = i * columns + j;
			indices.push(a, a + 1, a + columns + 1, a, a + columns + 1, a + columns);
		}
	}
	const mesh = { positions, normals, indices: Uint32Array.from(indices), cp, sigma };
	if (withCfd) mesh.cfd = cfd;
	return { source: { name: 'sheet.vtp', cells: rows * columns }, forces: { cdP: 0.27 }, mesh, lookups: lookups() };
}

/** `ResultLookups` with small slice rasters and a surface scale of irregular values. */
function lookups() {
	const values = (length, seed) => Float32Array.from({ length }, (_, i) => Math.sin(seed + 0.37 * i) * 10 ** (i % 5));
	return {
		definition: {},
		bounds: [-0.7, 2.3, -0.5, 0.5, -0.1, 0.1],
		yCenter: 0,
		yOut: values(60, 1),
		zTop: values(40, 2),
		zBottom: values(40, 5),
		sliceArea: values(10, 6),
		scales: [
			{
				origin: [-1, -1, -0.5],
				nodes: [3, 4, 5],
				kept: Int32Array.of(0, 1, 7, 8, 59),
				whole: values(20, 3),
				ahead: values(20, 4),
			},
		],
	};
}

/** Every array of `ResultLookups`, by name, for comparing two of them value for value. */
function lookupArrays({ yOut, zTop, zBottom, sliceArea, scales }) {
	const names = (scale) => (scale.readsAtVertices ? ['readsAtVertices'] : ['kept', 'whole', 'ahead']);
	return {
		yOut,
		zTop,
		zBottom,
		sliceArea,
		...Object.fromEntries(scales.flatMap((scale, k) => names(scale).map((name) => [`${name}${k}`, scale[name]]))),
	};
}

function sameValues(a, b) {
	return a.constructor === b.constructor && a.length === b.length && a.every((value, i) => Object.is(value, b[i]));
}

function assert(condition, message) {
	if (!condition) throw new Error(message);
}

/** Packs and unpacks `result`; returns the copy and, for each of its vertices, the original vertex. */
async function roundTrip(result) {
	const { json, block } = await packResult(result);
	const back = await unpackResult(JSON.parse(json), block.buffer);
	// Vertices sit on a 1 cm grid in x and y, so x and y to the nearest millimetre name each one.
	const key = (positions, v) => [0, 1].map((axis) => Math.round(positions[3 * v + axis] * 1000)).join();
	const original = new Map();
	for (let v = 0; v < result.mesh.positions.length / 3; v++) original.set(key(result.mesh.positions, v), v);
	const match = Array.from({ length: back.mesh.positions.length / 3 }, (_, v) =>
		original.get(key(back.mesh.positions, v)),
	);
	return { back, match, bytes: block.length };
}

Deno.test('a packed result comes back within the stated precision', async () => {
	const result = wavySheet(true);
	const { back, match, bytes } = await roundTrip(result);
	const raw = ['positions', 'normals', 'indices', 'cp', 'sigma', 'cfd'].reduce(
		(sum, k) => sum + result.mesh[k].byteLength,
		0,
	);
	assert(bytes < raw / 4, `packed to ${bytes} bytes from ${raw}`);
	assert(back.forces.cdP === 0.27 && back.source.name === 'sheet.vtp', 'summary changed');
	const bounds = [0, 1, 2].flatMap((axis) => {
		const lane = result.mesh.positions.filter((_, i) => i % 3 === axis);
		return [Math.min(...lane), Math.max(...lane)];
	});
	assert(
		back.mesh.bounds.every((value, i) => Math.abs(value - bounds[i]) < 1e-4),
		`bounds ${back.mesh.bounds} are not ${bounds}`,
	);
	for (const [name, array] of Object.entries(lookupArrays(result.lookups)))
		assert(sameValues(lookupArrays(back.lookups)[name], array), `lookup ${name} changed`);
	assert(
		JSON.stringify(back.lookups.scales[0].origin) === JSON.stringify(result.lookups.scales[0].origin),
		'scale origin changed',
	);
	assert(new Set(match).size === match.length && !match.includes(undefined), 'vertices do not map one to one');

	for (let v = 0; v < match.length; v++) {
		const o = match[v];
		for (let axis = 0; axis < 3; axis++) {
			const error = Math.abs(back.mesh.positions[3 * v + axis] - result.mesh.positions[3 * o + axis]);
			assert(error < 1e-4, `vertex ${v} moved ${error} m`);
		}
		const dot = [0, 1, 2].reduce(
			(sum, axis) => sum + back.mesh.normals[3 * v + axis] * result.mesh.normals[3 * o + axis],
			0,
		);
		assert(
			dot > Math.cos((0.1 * Math.PI) / 180),
			`normal ${v} turned ${(Math.acos(Math.min(dot, 1)) * 180) / Math.PI} degrees`,
		);
		for (const field of ['cp', 'sigma', 'cfd']) {
			const error = Math.abs(back.mesh[field][v] - result.mesh[field][o]);
			assert(error < 0.002, `${field} at vertex ${v} is off by ${error}`);
		}
	}

	// The same triangles, each still wound the same way round.
	const canonical = (a, b, c) => {
		const turn = a < b && a < c ? [a, b, c] : b < c ? [b, c, a] : [c, a, b];
		return turn.join();
	};
	const triangles = new Set();
	for (let t = 0; t < result.mesh.indices.length; t += 3)
		triangles.add(canonical(...result.mesh.indices.subarray(t, t + 3)));
	assert(back.mesh.indices.length === result.mesh.indices.length, 'triangle count changed');
	for (let t = 0; t < back.mesh.indices.length; t += 3) {
		const [a, b, c] = back.mesh.indices.subarray(t, t + 3);
		assert(triangles.has(canonical(match[a], match[b], match[c])), `triangle ${t / 3} is not one of the original's`);
	}
});

Deno.test('a result without CFD comes back without CFD', async () => {
	const { back } = await roundTrip(wavySheet(false));
	assert(!('cfd' in back.mesh) && back.mesh.cp && back.mesh.sigma, 'fields changed');
});

Deno.test('a scale kept at more nodes than the mesh has vertices comes back as its reads at the vertices', async () => {
	// Every node of a 2 cm grid around the sheet: 138,000 nodes against 30,401 vertices.
	const nodes = [154, 56, 16];
	const count = nodes[0] * nodes[1] * nodes[2];
	const field = (seed) =>
		Float32Array.from({ length: 4 * count }, (_, i) => Math.sin(seed + 0.001 * i) + 0.1 * Math.cos(0.37 * i));
	const scale = {
		origin: [-0.76, -0.56, -0.16],
		nodes,
		kept: Int32Array.from({ length: count }, (_, i) => i),
		whole: field(1),
		ahead: field(2),
	};
	const result = wavySheet(true);
	result.lookups = { ...result.lookups, definition: { surfaceSteps: [0.02] }, scales: [scale] };
	const { back, match } = await roundTrip(result);
	const stored = back.lookups.scales[0];
	assert(stored.readsAtVertices && !stored.kept, 'the scale came back as its nodes');
	assert(JSON.stringify(stored.origin) === JSON.stringify(scale.origin), 'scale origin changed');
	const expected = scaleReadsAt({ origin: scale.origin, nodes, step: 0.02 }, scale, result.mesh.positions);
	for (let v = 0; v < match.length; v++) {
		const read = stored.readsAtVertices.subarray(8 * v, 8 * v + 8);
		assert(
			sameValues(read, expected.subarray(8 * match[v], 8 * match[v] + 8)),
			`vertex ${v} reads another vertex's values`,
		);
	}
});

Deno.test('loadResult decodes a bundled example in a worker to what unpackResult gives', async () => {
	const base = new URL('../web/examples/run_1', import.meta.url);
	const summary = JSON.parse(await Deno.readTextFile(new URL(`${base}.json`)));
	const direct = await unpackResult(summary, (await Deno.readFile(new URL(`${base}.bin`))).slice().buffer);
	const loaded = await loadResult(base.href);
	for (const name of ['positions', 'normals', 'indices', 'cp', 'sigma', 'cfd']) {
		assert(sameValues(loaded.mesh[name], direct.mesh[name]), `mesh ${name} differs`);
	}
	assert(
		loaded.mesh.bounds.every((value, i) => value === direct.mesh.bounds[i]),
		'mesh bounds differ',
	);
	for (const [name, array] of Object.entries(lookupArrays(direct.lookups)))
		assert(sameValues(lookupArrays(loaded.lookups)[name], array), `lookup ${name} differs`);
	assert(JSON.stringify(loaded.forces) === JSON.stringify(direct.forces), 'forces differ');
});
