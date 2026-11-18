// What the browser engine (web/engine) computes for a surface file, written as raw little-endian
// arrays for tests/test_web_engine.py to compare with the Python pipeline:
//
//   bun tests/web_engine_outputs.mjs parse    <surface.vtp> <out dir>
//   bun tests/web_engine_outputs.mjs engine   <surface.vtp> <out dir> <model dir>
//   bun tests/web_engine_outputs.mjs features <surface.vtp> <out dir> <feature definition.json>
//   deno run --allow-read --allow-write tests/web_engine_outputs.mjs features-gpu <surface.vtp> <out dir> <feature definition.json>
//   bun tests/web_engine_outputs.mjs sequence <surface.vtp> <out dir>
//   bun tests/web_engine_outputs.mjs export   <surface.vtp> <out dir>
//   bun tests/web_engine_outputs.mjs analysis <surface.vtp> <out dir>
//   bun tests/web_engine_outputs.mjs pick     <surface.vtp> <out dir> <model dir>
//   bun tests/web_engine_outputs.mjs quick    <surface.vtp> <out dir>
//   bun tests/web_engine_outputs.mjs write    <surface.vtp> <out dir>
//   bun tests/web_engine_outputs.mjs reference <surface.vtp> <out dir> <cfd.vtp>,<remeshed cfd.vtp>,<bare.vtp>,<other car.vtp>
//   bun tests/web_engine_outputs.mjs example  <example base>  <out dir>
//   bun tests/web_engine_outputs.mjs grid-cpu <inputs dir>    <out dir>
//   deno run --allow-read --allow-write tests/web_engine_outputs.mjs grid-gpu <inputs dir> <out dir>
//
// `parse` writes the parsed surface and its frontal area; `engine` adds the polygon geometry, the
// features at every polygon centre, each network's Cp there, and the features again from the lookups
// as an example pack stores and reads them back; `features` writes the features of the given
// definition at every polygon centre, from lookups built on the CPU, and `features-gpu` from lookups
// built on the GPU (it needs a runtime with WebGPU, Deno, and writes outcome.json { "gpu": false }
// where there is no adapter). A file the reader rejects gives outcome.json { "error": message }
// and nothing else.
// `sequence` runs the whole engine (`createEngine`, with `web/model`) on the file twice, asking for
// the second analysis as soon as the first one starts, and writes each one's events in arrival order.
// `export` runs the whole engine on the file and writes what `exportVtp` gives once every polygon is
// predicted, as exported.vtp. `analysis` runs it and writes the Result's geometry, its sampled
// forces, the `exact` numbers, what `polygonAt` and `explain` give at points just outside two dozen
// polygons, and the polygons `fileCells` and `workingCells` give. `pick` times `polygonAt`'s lookup
// at points outside two hundred polygons, the networks asked at each on demand, without the rest of
// the engine. `quick` writes `predictQuick`'s result for the file's surface, with the messages of the
// arrays it must refuse. `write` writes the file's surface anew with `writeVtp`, with cell and point
// arrays of several types and widths. `reference` analyses `cfd.vtp` alone, then the surface, and
// attaches a reference to it from each of the four other files in turn, writing what each step
// gives. `example` asks a fresh engine, which has analysed nothing, `explain` and `polygonAt` at a
// vertex of a bundled example. `grid-cpu` and `grid-gpu` run the grid operations (`grid.js`,
// `grid-gpu.js`) on synthetic inputs (spec.json with points, weights and field as float32) and write
// each result; `grid-gpu` spreads in blocks of spec.sumsBlock nodes, as a car's finest grid is
// spread, and needs a runtime with WebGPU (Deno); it writes outcome.json { "gpu": false } where there
// is no adapter.

import { readFileSync, writeFileSync } from 'node:fs';
import { basename, join } from 'node:path';
import process from 'node:process';
import { pathToFileURL } from 'node:url';
import { readSurface, writeVtp } from '../web/engine/vtp.js';
import { frontalArea, measureCells } from '../web/engine/geometry.js';
import { computeFeatures, featureLayout, lookupsAtVertex } from '../web/engine/features.js';
import { buildLookups } from '../web/engine/lookups.js';
import { ensembleStatistics, loadEnsemble, networkActivations, predictOnCpu } from '../web/engine/network.js';
import { packResult, unpackResult } from '../web/engine/result.js';
import { createEngine } from '../web/engine/engine.js';
import { describePolygon, indexPolygons, polygonAt } from '../web/engine/polygon-lookup.js';
import { correlateAlong, gaussianWeights, spreadOnGrid, upstreamHalf } from '../web/engine/grid.js';
import { createGridOperations } from '../web/engine/grid-gpu.js';

const [mode, path, outDir, argument] = process.argv.slice(2);
const write = (name, array) =>
	writeFileSync(join(outDir, name), new Uint8Array(array.buffer, array.byteOffset, array.byteLength));

/**
 * `count` polygons spread through the file, each with a point just outside it (10 µm off a point
 * inside its first fan triangle, nearer to it than to any other polygon of a real car) and its
 * outward normal.
 */
function probesOutside({ points, connectivity, offsets }, { normals }, count) {
	const cells = offsets.length - 1;
	return Array.from({ length: count }, (_, k) => {
		const target = Math.floor(((k + 0.5) / count) * cells);
		const corner = (i) =>
			points.subarray(3 * connectivity[offsets[target] + i], 3 * connectivity[offsets[target] + i] + 3);
		const normal = Array.from(normals.subarray(3 * target, 3 * target + 3));
		const point = [0, 1, 2].map(
			(axis) => 0.5 * corner(0)[axis] + 0.3 * corner(1)[axis] + 0.2 * corner(2)[axis] + 1e-5 * normal[axis],
		);
		return { target, point, normal };
	});
}

const plain = (polygon) => polygon && { ...polygon, corners: Array.from(polygon.corners) };

if (mode === 'example') {
	const example = await unpackResult(
		JSON.parse(readFileSync(`${path}.json`, 'utf8')),
		readFileSync(`${path}.bin`).buffer,
	);
	const engine = createEngine();
	const v = Math.floor(example.mesh.positions.length / 6);
	const point = Array.from(example.mesh.positions.subarray(3 * v, 3 * v + 3));
	const normal = Array.from(example.mesh.normals.subarray(3 * v, 3 * v + 3));
	const { at, index, cfd } = await engine.explain(example, point, normal);
	const polygon = await engine.polygonAt(example, point, normal);
	writeFileSync(
		join(outDir, 'outcome.json'),
		JSON.stringify({ explained: { at, index, cfd }, vertexCfd: example.mesh.cfd[index], polygon }),
	);
	process.exit(0);
}

if (mode === 'sequence') {
	const engine = createEngine();
	const file = new File([readFileSync(path)], basename(path));
	const events = [];
	const listen = (car) => ({
		stage: (stage, fraction) => events.push({ car, event: 'stage', stage, fraction }),
		fullFields: (fields, fraction) => events.push({ car, event: 'fullFields', fraction }),
		exact: () => events.push({ car, event: 'exact' }),
	});
	const settle = (car, analysis) =>
		analysis.then(
			() => events.push({ car, event: 'result' }),
			(error) => events.push({ car, event: 'rejected', message: `${error.name}: ${error.message}` }),
		);
	let second;
	const first = settle(
		'first',
		engine.analyze(file, {
			...listen('first'),
			stage: (stage, fraction) => {
				events.push({ car: 'first', event: 'stage', stage, fraction });
				second ??= settle('second', engine.analyze(file, listen('second')));
			},
		}),
	);
	await first;
	await second;
	// Each analysis ends with its last full-mesh fields; allow the second its background prediction.
	const ended = () =>
		events.some(
			(e) => e.car === 'second' && (e.event === 'rejected' || (e.event === 'fullFields' && e.fraction === 1)),
		);
	for (let waited = 0; waited < 60_000 && !ended(); waited += 100)
		await new Promise((resolve) => setTimeout(resolve, 100));
	writeFileSync(join(outDir, 'outcome.json'), JSON.stringify({ events }));
	process.exit(0);
}

if (mode === 'export') {
	const engine = createEngine();
	const file = new File([readFileSync(path)], basename(path));
	let exact;
	const everyPolygon = new Promise((resolve) => {
		exact = resolve;
	});
	const result = await engine.analyze(file, { exact });
	const exportableEarly = engine.canExportVtp(result);
	await everyPolygon;
	const vtp = await engine.exportVtp(result);
	writeFileSync(join(outDir, 'exported.vtp'), new Uint8Array(await vtp.arrayBuffer()));
	writeFileSync(
		join(outDir, 'outcome.json'),
		JSON.stringify({ name: vtp.name, exportableEarly, exportable: engine.canExportVtp(result) }),
	);
	process.exit(0);
}

if (mode === 'analysis') {
	const file = new File([readFileSync(path)], basename(path));
	const engine = createEngine();
	let exact;
	const everyPolygon = new Promise((resolve) => {
		exact = resolve;
	});
	const result = await engine.analyze(file, { exact });
	const surface = await readSurface(file);
	const probes = probesOutside(surface, measureCells(surface), 24);
	const ask = (question) => Promise.all(probes.map(({ point, normal }) => question(result, point, normal)));
	// As soon as the result is out, and again once every polygon is predicted.
	const early = await ask(engine.polygonAt);
	const numbers = await everyPolygon;
	const late = await ask(engine.polygonAt);
	const explained = await ask(engine.explain);
	const cells = await engine.fileCells(result);
	const working = engine.workingCells(result);
	for (const name of ['centers', 'normals', 'areas', 'cp', 'sigma']) {
		write(`file_${name}.${name === 'areas' ? 'f64' : 'f32'}`, cells[name]);
		write(`working_${name}.${name === 'areas' ? 'f64' : 'f32'}`, working[name]);
	}
	write('working_positions.f32', result.mesh.positions);
	write('working_indices.u32', result.mesh.indices);
	write('working_vertex_cp.f32', result.mesh.cp);
	writeFileSync(
		join(outDir, 'outcome.json'),
		JSON.stringify({
			geometry: result.geometry,
			sampled: { forces: result.forces },
			exact: numbers,
			probes: probes.map(({ target }, k) => ({
				target,
				early: plain(early[k]),
				late: plain(late[k]),
				explained: { index: explained[k].index, cfd: explained[k].cfd },
			})),
		}),
	);
	process.exit(0);
}

if (mode === 'pick') {
	const ensemble = await loadEnsemble(pathToFileURL(`${argument}/`));
	const surface = await readSurface(new File([readFileSync(path)], basename(path)));
	const cells = measureCells(surface);
	const layout = featureLayout(ensemble.features);
	const context = { layout, ...(await buildLookups(surface, cells, layout, null)) };
	const started = performance.now();
	const index = indexPolygons(surface, cells);
	const indexSeconds = (performance.now() - started) / 1000;
	const count = cells.areas.length;
	// Nothing predicted yet, so every pick also asks the networks: the slowest case.
	const predicted = { cp: new Float32Array(count), sigma: new Float32Array(count), done: new Uint8Array(count) };
	const car = { surface, cells, context, predicted, index };
	const picks = probesOutside(surface, cells, 200).map(({ target, point, normal }) => {
		const pickStarted = performance.now();
		const found = polygonAt(car, point, normal);
		const polygon = found < 0 ? null : describePolygon(car, ensemble, found);
		return { target, polygon: plain(polygon), seconds: (performance.now() - pickStarted) / 1000 };
	});
	const indexBytes = index.grid.order.byteLength + index.grid.starts.byteLength;
	writeFileSync(join(outDir, 'outcome.json'), JSON.stringify({ indexSeconds, indexBytes, picks }));
	process.exit(0);
}

/** The surface with arrays of its own, as `predictQuick` takes them. */
const ownArrays = ({ name, points, connectivity, offsets, cfd }) => ({
	name,
	points: points.slice(),
	connectivity: connectivity.slice(),
	offsets: offsets.slice(),
	cfd: cfd && cfd.slice(),
});

if (mode === 'quick') {
	const read = await readSurface(new File([readFileSync(path)], basename(path)));
	const engine = createEngine();

	const shared = new Float32Array(read.points.length + 3);
	shared.set(read.points);
	const refuse = (surface) =>
		engine.predictQuick(surface).then(
			() => null,
			(error) => error.message,
		);
	const refusals = {
		viewOfLargerBuffer: await refuse({ ...ownArrays(read), points: shared.subarray(0, read.points.length) }),
		offsetsShort: await refuse({ ...ownArrays(read), offsets: read.offsets.slice(0, -1) }),
		pointMissing: await refuse({ ...ownArrays(read), points: read.points.slice(0, -3) }),
	};

	const surface = ownArrays(read);
	const result = await engine.predictQuick(surface);
	write('cp.f32', result.cells.cp);
	write('sigma.f32', result.cells.sigma);
	write('vertex_cp.f32', result.mesh.cp);
	writeFileSync(
		join(outDir, 'outcome.json'),
		JSON.stringify({
			form: result.form,
			geometry: result.geometry,
			forces: result.forces,
			accuracy: result.accuracy,
			totals: result.totals,
			reference: result.reference,
			triangles: result.mesh.indices.length / 3,
			vertices: result.mesh.positions.length / 3,
			exportable: engine.canExportVtp(result),
			sameCells: engine.workingCells(result) === result.cells,
			moved: surface.points.length === 0,
			refusals,
		}),
	);
	process.exit(0);
}

if (mode === 'reference') {
	const [withCfd, remeshed, bare, otherCar] = argument
		.split(',')
		.map((source) => new File([readFileSync(source)], basename(source)));
	const engine = createEngine();
	const analyse = async (file) => {
		let exact;
		const everyPolygon = new Promise((resolve) => {
			exact = resolve;
		});
		const result = await engine.analyze(file, { exact });
		return Object.assign(result, await everyPolygon);
	};
	const numbers = ({ reference, referenceTotals, accuracy, mesh }) => ({
		reference,
		referenceTotals,
		accuracy,
		meshHasCfd: 'cfd' in mesh,
	});
	const refusal = (promise) =>
		promise.then(
			() => null,
			(error) => error.message,
		);

	const alone = await analyse(withCfd);
	write('alone_mesh_cfd.f32', alone.mesh.cfd);
	const result = await analyse(new File([readFileSync(path)], basename(path)));
	const before = numbers(result);
	const refusals = {
		bare: await refusal(engine.attachReference(result, bare)),
		otherCar: await refusal(engine.attachReference(result, otherCar)),
		missing: await refusal(engine.attachReference(result, withCfd, { array: 'CpMissing' })),
	};

	const attached = await engine.attachReference(result, withCfd);
	const same = numbers(result);
	write('same_mesh_cfd.f32', result.mesh.cfd);
	write('same_returned_cfd.f32', attached.cfd);
	write('same_file_cfd.f32', (await engine.fileCells(result)).cfd);
	const exported = await engine.exportVtp(result);
	writeFileSync(join(outDir, 'exported.vtp'), new Uint8Array(await exported.arrayBuffer()));

	await engine.attachReference(result, remeshed);
	const nearest = numbers(result);
	write('nearest_file_cfd.f32', (await engine.fileCells(result)).cfd);

	writeFileSync(
		join(outDir, 'outcome.json'),
		JSON.stringify({
			geometry: result.geometry,
			alone: numbers(alone),
			before,
			refusals,
			same,
			nearest,
		}),
	);
	process.exit(0);
}

if (mode === 'write') {
	const read = await readSurface(new File([readFileSync(path)], basename(path)));
	const { centers } = measureCells(read);
	const sizes = Uint8Array.from({ length: read.offsets.length - 1 }, (_, c) => read.offsets[c + 1] - read.offsets[c]);
	const heights = Float32Array.from({ length: read.points.length / 3 }, (_, p) => read.points[3 * p + 2]);
	const vtp = writeVtp(
		{ ...read, name: 'written.vtp' },
		{ CpMeanTrim: read.cfd, centers: Float64Array.from(centers), corners: sizes },
		{ height: heights },
	);
	const bytes = new Uint8Array(await vtp.arrayBuffer());
	writeFileSync(join(outDir, 'written.vtp'), bytes);
	// The engine's own reader takes it back as it was given.
	const back = await readSurface(new File([bytes], vtp.name));
	const same = (a, b) => a.length === b.length && a.every((value, i) => value === b[i]);
	writeFileSync(
		join(outDir, 'outcome.json'),
		JSON.stringify({
			name: vtp.name,
			readBack: ['points', 'connectivity', 'offsets', 'cfd'].every((key) => same(back[key], read[key])),
		}),
	);
	process.exit(0);
}

if (mode === 'grid-cpu' || mode === 'grid-gpu') {
	// `path` is a directory of synthetic inputs: spec.json, and as float32 points with weights and a
	// field on the grid.
	const spec = JSON.parse(readFileSync(join(path, 'spec.json'), 'utf8'));
	const input = (name) => new Float32Array(readFileSync(join(path, `${name}.f32`)).buffer.slice(0));
	const [points, weights, field] = ['points', 'weights', 'field'].map(input);
	const grid = { origin: spec.origin, step: spec.step, nodes: spec.nodes };
	const gauss = gaussianWeights(spec.sigmaNodes, 3);
	const ahead = upstreamHalf(gauss);
	if (mode === 'grid-cpu') {
		const sums = new Float64Array(grid.nodes[0] * grid.nodes[1] * grid.nodes[2]);
		spreadOnGrid(grid, points, weights, sums);
		write('spread.f32', Float32Array.from(sums));
		for (const axis of [0, 1, 2]) write(`blur${axis}.f32`, correlateAlong(field, grid.nodes, axis, gauss));
		write('ahead.f32', correlateAlong(field, grid.nodes, 0, ahead));
		process.exit(0);
	}
	const adapter = await navigator.gpu?.requestAdapter();
	if (!adapter) {
		writeFileSync(join(outDir, 'outcome.json'), JSON.stringify({ gpu: false }));
		process.exit(0);
	}
	const device = await adapter.requestDevice();
	device.pushErrorScope('validation');
	const operations = createGridOperations(device);
	const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
	const buffer = (floats) => device.createBuffer({ size: 4 * floats, usage });
	const upload = (array) => {
		const created = buffer(array.length);
		device.queue.writeBuffer(created, 0, Float32Array.from(array));
		return created;
	};
	const nodes = grid.nodes[0] * grid.nodes[1] * grid.nodes[2];
	const count = weights.length;
	const positions = upload(points);
	const fieldBuffer = upload(field);
	const outputs = {
		spread: [buffer(nodes), nodes],
		blur0: [buffer(nodes), nodes],
		blur1: [buffer(nodes), nodes],
		blur2: [buffer(nodes), nodes],
		ahead: [buffer(nodes), nodes],
	};
	const encoder = device.createCommandEncoder();
	const sums = buffer(2 * spec.sumsBlock);
	const weightBuffer = upload(weights);
	for (let first = 0; first < nodes; first += spec.sumsBlock) {
		const length = Math.min(spec.sumsBlock, nodes - first);
		encoder.clearBuffer(sums);
		operations.spread(encoder, grid, { positions, weights: weightBuffer, count, sums, first, length });
		operations.sumsToField(encoder, { sums, field: outputs.spread[0], first, length });
	}
	const radius = (gauss.length - 1) / 2;
	for (const axis of [0, 1, 2])
		operations.correlate(encoder, grid.nodes, axis, {
			weights: upload(gauss),
			radius,
			field: fieldBuffer,
			out: outputs[`blur${axis}`][0],
		});
	operations.correlate(encoder, grid.nodes, 0, {
		weights: upload(ahead),
		radius,
		field: fieldBuffer,
		out: outputs.ahead[0],
	});
	device.queue.submit([encoder.finish()]);
	for (const [name, [source, floats]] of Object.entries(outputs)) {
		const staging = device.createBuffer({ size: 4 * floats, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
		const copy = device.createCommandEncoder();
		copy.copyBufferToBuffer(source, 0, staging, 0, 4 * floats);
		device.queue.submit([copy.finish()]);
		await staging.mapAsync(GPUMapMode.READ);
		write(`${name}.f32`, new Float32Array(staging.getMappedRange()).slice());
	}
	const invalid = await device.popErrorScope();
	writeFileSync(join(outDir, 'outcome.json'), JSON.stringify({ gpu: true, error: invalid?.message ?? null }));
	device.destroy();
	process.exit(0);
}

let surface;
try {
	surface = await readSurface(new File([readFileSync(path)], basename(path)));
} catch (error) {
	writeFileSync(join(outDir, 'outcome.json'), JSON.stringify({ error: error.message }));
	process.exit(0);
}
write('points.f32', surface.points);
write('connectivity.u32', surface.connectivity);
write('offsets.u32', surface.offsets);
if (surface.cfd) write('cfd.f32', surface.cfd);

let outcome = { cells: surface.offsets.length - 1, hasCfd: Boolean(surface.cfd), frontalArea: frontalArea(surface) };
if (mode === 'engine' || mode === 'features' || mode === 'features-gpu') {
	const ensemble = mode === 'engine' ? await loadEnsemble(pathToFileURL(`${argument}/`)) : null;
	const layout = featureLayout(ensemble ? ensemble.features : JSON.parse(readFileSync(argument, 'utf8')));
	const cells = measureCells(surface);
	let grids = null;
	if (mode === 'features-gpu') {
		const adapter = await navigator.gpu?.requestAdapter();
		if (!adapter) {
			writeFileSync(join(outDir, 'outcome.json'), JSON.stringify({ gpu: false }));
			process.exit(0);
		}
		grids = createGridOperations(await adapter.requestDevice());
	}
	const context = { layout, ...(await buildLookups(surface, cells, layout, grids)) };
	outcome.gpu = Boolean(grids);
	const count = cells.areas.length;
	const features = new Float32Array(count * layout.count);
	computeFeatures(context, cells.centers, cells.normals, 0, count, features);
	write('centers.f32', cells.centers);
	write('normals.f32', cells.normals);
	write('areas.f64', cells.areas);
	write('features.f32', features);
	outcome = { ...outcome, flip: cells.flip, featureCount: layout.count };
	if (ensemble) {
		const models = new Float32Array(ensemble.folds.length * count);
		predictOnCpu(ensemble, features, count, models);
		write('models.f32', models);
		write('sigma.f32', ensembleStatistics(models, count, ensemble.spreadScale).sigma);
		outcome.networks = ensemble.folds.length;

		// What `explain` gives at the first polygon: each network's layers, back to back, and its Cp.
		const { cpModels, activations } = networkActivations(ensemble, features.subarray(0, layout.count));
		write('explained_models.f32', cpModels);
		write('activations.f32', Float32Array.from(activations.flat().flatMap((layer) => Array.from(layer))));

		// A bundled example of this car whose mesh's vertices are its polygon centres: `explain` at each
		// vertex, given its polygon's own centre and normal, reads what the analysis read.
		const corners = Uint32Array.from({ length: 3 * Math.ceil(count / 3) }, (_, i) => i % count);
		const mesh = { positions: cells.centers, normals: cells.normals, indices: corners };
		const { json, block } = await packResult({ mesh, lookups: { definition: ensemble.features, ...context } });
		const example = await unpackResult(JSON.parse(json), block.buffer);
		/** The polygon whose centre vertex `v` of the example sits on, moved by the pack's rounding. */
		const polygonOf = (v) => {
			let best = -1,
				closest = Infinity;
			for (let c = 0; c < count; c++) {
				const distance = [0, 1, 2].reduce(
					(sum, axis) => sum + (example.mesh.positions[3 * v + axis] - cells.centers[3 * c + axis]) ** 2,
					0,
				);
				if (distance < closest) [best, closest] = [c, distance];
			}
			return best;
		};
		const fromExample = new Float32Array(count * layout.count);
		for (let v = 0; v < count; v++) {
			const c = polygonOf(v);
			computeFeatures(
				{ ...lookupsAtVertex(example.lookups, v), layout },
				cells.centers,
				cells.normals,
				c,
				1,
				fromExample.subarray(c * layout.count),
			);
		}
		write('features_from_example.f32', fromExample);
	}
}
writeFileSync(join(outDir, 'outcome.json'), JSON.stringify(outcome));
