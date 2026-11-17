// What the browser engine (web/engine) computes for a surface file, written as raw little-endian
// arrays for tests/test_web_engine.py to compare with the Python pipeline:
//
//   bun tests/web_engine_outputs.mjs parse    <surface.vtp> <out dir>
//   bun tests/web_engine_outputs.mjs engine   <surface.vtp> <out dir> <model dir>
//   bun tests/web_engine_outputs.mjs features <surface.vtp> <out dir> <feature definition.json>
//
// `parse` writes the parsed surface; `engine` adds the polygon geometry, the features at every
// polygon centre, each network's Cp there, and the features again from the lookups as an example
// pack stores and reads them back; `features` writes the features of the given definition at every
// polygon centre. A file the reader rejects gives outcome.json { "error": message } and nothing else.

import { readFileSync, writeFileSync } from 'node:fs';
import { basename, join } from 'node:path';
import process from 'node:process';
import { pathToFileURL } from 'node:url';
import { readSurface } from '../web/engine/vtp.js';
import { measureCells } from '../web/engine/geometry.js';
import { buildContext, computeFeatures, featureLayout } from '../web/engine/features.js';
import { ensembleStatistics, loadEnsemble, predictOnCpu } from '../web/engine/network.js';
import { packResult, unpackResult } from '../web/engine/result.js';

const [mode, path, outDir, modelSource] = process.argv.slice(2);
const write = (name, array) => writeFileSync(join(outDir, name), new Uint8Array(array.buffer, array.byteOffset, array.byteLength));

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

    const triangle = { sourceTriangles: 1, positions: cells.centers.slice(0, 9), normals: cells.normals.slice(0, 9), indices: Uint32Array.of(0, 1, 2) };
    const { json, block } = await packResult({ mesh: triangle, lookups: { definition: ensemble.features, ...context } });
    const example = await unpackResult(JSON.parse(json), block.buffer);
    const fromExample = new Float32Array(count * layout.count);
    computeFeatures({ ...example.lookups, layout }, cells.centers, cells.normals, 0, count, fromExample);
    write('features_from_example.f32', fromExample);
  }
}
writeFileSync(join(outDir, 'outcome.json'), JSON.stringify(outcome));
