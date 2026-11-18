// What the browser engine (web/engine) computes for a surface file, written as raw little-endian
// arrays for tests/test_web_engine.py to compare with the Python pipeline:
//
//   bun tests/web_engine_outputs.mjs parse    <surface.vtp> <out dir>
//   bun tests/web_engine_outputs.mjs engine   <surface.vtp> <out dir> <model dir>
//   bun tests/web_engine_outputs.mjs features <surface.vtp> <out dir> <feature definition.json>
//   bun tests/web_engine_outputs.mjs sequence <surface.vtp> <out dir>
//   bun tests/web_engine_outputs.mjs export   <surface.vtp> <out dir>
//   bun tests/web_engine_outputs.mjs analysis <surface.vtp> <out dir>
//   bun tests/web_engine_outputs.mjs pick     <surface.vtp> <out dir> <model dir>
//
// `parse` writes the parsed surface; `engine` adds the polygon geometry, the features at every
// polygon centre, each network's Cp there, and the features again from the lookups as an example
// pack stores and reads them back; `features` writes the features of the given definition at every
// polygon centre. A file the reader rejects gives outcome.json { "error": message } and nothing else.
// `sequence` runs the whole engine (`createEngine`, with `web/model`) on the file twice, asking for
// the second analysis as soon as the first one starts, and writes each one's events in arrival order.
// `export` runs the whole engine on the file and writes what `exportVtp` gives once every polygon is
// predicted, as exported.vtp. `analysis` runs it and writes the Result's geometry, its sampled
// forces, the `exact` numbers, and what `polygonAt` and `explain` give at points just outside two
// dozen polygons. `pick` times `polygonAt`'s lookup at points outside two hundred polygons, the
// networks asked at each on demand, without the rest of the engine.

import { readFileSync, writeFileSync } from 'node:fs';
import { basename, join } from 'node:path';
import process from 'node:process';
import { pathToFileURL } from 'node:url';
import { readSurface } from '../web/engine/vtp.js';
import { measureCells } from '../web/engine/geometry.js';
import { buildContext, computeFeatures, featureLayout } from '../web/engine/features.js';
import { ensembleStatistics, loadEnsemble, networkActivations, predictOnCpu } from '../web/engine/network.js';
import { packResult, unpackResult } from '../web/engine/result.js';
import { createEngine } from '../web/engine/engine.js';
import { describePolygon, indexPolygons, polygonAt } from '../web/engine/polygon-lookup.js';

const [mode, path, outDir, modelSource] = process.argv.slice(2);
const write = (name, array) => writeFileSync(join(outDir, name), new Uint8Array(array.buffer, array.byteOffset, array.byteLength));

/**
 * `count` polygons spread through the file, each with a point just outside it (10 µm off a point
 * inside its first fan triangle, nearer to it than to any other polygon of a real car) and its
 * outward normal.
 */
function probesOutside({ points, connectivity, offsets }, { normals }, count) {
  const cells = offsets.length - 1;
  return Array.from({ length: count }, (_, k) => {
    const target = Math.floor(((k + 0.5) / count) * cells);
    const corner = (i) => points.subarray(3 * connectivity[offsets[target] + i], 3 * connectivity[offsets[target] + i] + 3);
    const normal = Array.from(normals.subarray(3 * target, 3 * target + 3));
    const point = [0, 1, 2].map((axis) => 0.5 * corner(0)[axis] + 0.3 * corner(1)[axis] + 0.2 * corner(2)[axis] + 1e-5 * normal[axis]);
    return { target, point, normal };
  });
}

const plain = (polygon) => polygon && { ...polygon, corners: Array.from(polygon.corners) };

if (mode === 'sequence') {
  const engine = createEngine();
  const file = new File([readFileSync(path)], basename(path));
  const events = [];
  const listen = (car) => ({
    stage: (stage, fraction) => events.push({ car, event: 'stage', stage, fraction }),
    fullFields: (fields, fraction) => events.push({ car, event: 'fullFields', fraction }),
    exact: () => events.push({ car, event: 'exact' }),
  });
  const settle = (car, analysis) => analysis.then(
    () => events.push({ car, event: 'result' }),
    (error) => events.push({ car, event: 'rejected', message: `${error.name}: ${error.message}` }),
  );
  let second;
  const first = settle('first', engine.analyze(file, {
    ...listen('first'),
    stage: (stage, fraction) => {
      events.push({ car: 'first', event: 'stage', stage, fraction });
      second ??= settle('second', engine.analyze(file, listen('second')));
    },
  }));
  await first;
  await second;
  // Each analysis ends with its last full-mesh fields; allow the second its background prediction.
  const ended = () => events.some((e) => e.car === 'second' && (e.event === 'rejected' || (e.event === 'fullFields' && e.fraction === 1)));
  for (let waited = 0; waited < 60_000 && !ended(); waited += 100) await new Promise((resolve) => setTimeout(resolve, 100));
  writeFileSync(join(outDir, 'outcome.json'), JSON.stringify({ events }));
  process.exit(0);
}

if (mode === 'export') {
  const engine = createEngine();
  const file = new File([readFileSync(path)], basename(path));
  let exact;
  const everyPolygon = new Promise((resolve) => { exact = resolve; });
  const result = await engine.analyze(file, { exact });
  const exportableEarly = engine.canExportVtp(result);
  await everyPolygon;
  const vtp = await engine.exportVtp(result);
  writeFileSync(join(outDir, 'exported.vtp'), new Uint8Array(await vtp.arrayBuffer()));
  writeFileSync(join(outDir, 'outcome.json'), JSON.stringify({ name: vtp.name, exportableEarly, exportable: engine.canExportVtp(result) }));
  process.exit(0);
}

if (mode === 'analysis') {
  const file = new File([readFileSync(path)], basename(path));
  const engine = createEngine();
  let exact;
  const everyPolygon = new Promise((resolve) => { exact = resolve; });
  const result = await engine.analyze(file, { exact });
  const surface = await readSurface(file);
  const probes = probesOutside(surface, measureCells(surface), 24);
  const ask = (question) => Promise.all(probes.map(({ point, normal }) => question(result, point, normal)));
  // As soon as the result is out, and again once every polygon is predicted.
  const early = await ask(engine.polygonAt);
  const numbers = await everyPolygon;
  const late = await ask(engine.polygonAt);
  const explained = await ask(engine.explain);
  writeFileSync(join(outDir, 'outcome.json'), JSON.stringify({
    geometry: result.geometry,
    sampled: { forces: result.forces },
    exact: numbers,
    probes: probes.map(({ target }, k) => ({ target, early: plain(early[k]), late: plain(late[k]), explained: explained[k].index })),
  }));
  process.exit(0);
}

if (mode === 'pick') {
  const ensemble = await loadEnsemble(pathToFileURL(`${modelSource}/`));
  const surface = await readSurface(new File([readFileSync(path)], basename(path)));
  const cells = measureCells(surface);
  const context = buildContext(cells, featureLayout(ensemble.features));
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

let outcome = { cells: surface.offsets.length - 1, hasCfd: Boolean(surface.cfd) };
if (mode === 'engine' || mode === 'features') {
  const ensemble = mode === 'engine' ? await loadEnsemble(pathToFileURL(`${modelSource}/`)) : null;
  const layout = featureLayout(ensemble ? ensemble.features : JSON.parse(readFileSync(modelSource, 'utf8')));
  const cells = measureCells(surface);
  const context = buildContext(cells, layout);
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

    const triangle = { sourceTriangles: 1, positions: cells.centers.slice(0, 9), normals: cells.normals.slice(0, 9), indices: Uint32Array.of(0, 1, 2) };
    const { json, block } = await packResult({ mesh: triangle, lookups: { definition: ensemble.features, ...context } });
    const example = await unpackResult(JSON.parse(json), block.buffer);
    const fromExample = new Float32Array(count * layout.count);
    computeFeatures({ ...example.lookups, layout }, cells.centers, cells.normals, 0, count, fromExample);
    write('features_from_example.f32', fromExample);
  }
}
writeFileSync(join(outDir, 'outcome.json'), JSON.stringify(outcome));
