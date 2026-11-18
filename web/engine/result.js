/**
 * The result of analysing one surface file, and its on-disk form.
 *
 * The engine returns a `Result` for a dropped file. The bundled examples are the same object
 * stored as `<name>.json` (everything but the arrays) plus `<name>.bin` (gzip over the mesh, in
 * meshoptimizer's index and vertex codecs with quantized positions, fields and normals, and over the
 * lookups as raw arrays; see `packResult`), so the interface never needs to know which of the two
 * it is showing.
 *
 * Frame: metres, x downstream, z up, the car symmetric about y = 0.
 *
 * @typedef {Object} Result
 * @property {{ name: string, bytes: number, cells: number, points: number, hasCfd: boolean }} source
 *   The file as read: its name, size, polygon and point counts, and whether it carried CFD pressure.
 * @property {{ bounds: number[], length: number, width: number, height: number,
 *              surfaceArea: number, frontalArea: number }} geometry
 *   `bounds` is [xmin, xmax, ymin, ymax, zmin, zmax]. `frontalArea` is the reference area for coefficients.
 * @property {{ cdP: number, cdPStd: number, cdPModels: number[], cdPCfd: number | null }} forces
 *   Pressure drag coefficient: mean of the ensemble's networks, their standard deviation, each
 *   network's value, and the CFD value integrated over the file's polygons when it has CFD.
 * @property {{ r2: number, mae: number, rmse: number, bias: number } | null} accuracy
 *   Area-weighted agreement with CFD over the surface. Null when the file has no CFD.
 * @property {Coverage} coverage
 * @property {{ x: number, cdP: number, cdPCfd?: number }[]} dragBuildup
 *   Pressure drag accumulated from the nose to station x.
 *   `forces`, `accuracy`, `dragBuildup` and `coverage.spread` are sums over the file's polygons. In the
 *   Result `analyze` resolves with they run over an evenly spaced area sample of them (drag within
 *   about 0.0005 of the full sum); `exact` brings them summed over every polygon.
 * @property {{ stage: Stage, seconds: number }[]} timings
 * @property {string} device  What ran the networks, as `Engine.device` names it.
 * @property {string} model   SHA-256 of the network weights it was computed with (`web/model/weights.bin`).
 * @property {DisplayMesh} mesh
 * @property {SurfaceLookups} lookups
 *
 * @typedef {Omit<import('./features.js').SurfaceContext, 'layout'> & { definition: import('./features.js').FeatureDefinition }} SurfaceLookups
 *   The whole-car lookups the features read (`SurfaceContext` in `features.js`, a few MB), built from
 *   every polygon of the file, with the feature definition they were built for. They let `explain`
 *   answer for a result without its file.
 *
 * @typedef {Object} Coverage  The car measured against the training cars. Facts only.
 * @property {{ name: 'length' | 'width' | 'height', value: number, trainMin: number, trainMax: number }[]} dimensions
 * @property {{ x: number, area: number, trainMin: number, trainMax: number }[]} sectionArea
 *   Cross-section area S(x) in m², with the training cars' envelope at the same station.
 * @property {number} sectionOutsideFraction  Share of stations where S(x) leaves the envelope.
 * @property {{ mean: number, p95: number, trainMean: number }} spread
 *   Spread σ(Cp) on this car (`DisplayMesh.sigma`), and its mean over the training cars in cross-validation.
 *
 * @typedef {Object} DisplayMesh  A surface the browser draws. Fields are per vertex: the networks are
 *   asked at the file's polygons, as they were trained, and the CFD gives one value per polygon too; a
 *   vertex shows the area-weighted mean of the file polygons around the file point it sits on.
 * @property {number} sourceTriangles  Triangle count of the file's full mesh. Equal to this mesh's
 *   own triangle count when it is the full mesh.
 * @property {Float32Array} positions  (V, 3)
 * @property {Float32Array} normals    (V, 3) unit, pointing out of the car
 * @property {Uint32Array} indices     (T, 3) wound counter-clockwise seen from outside
 * @property {Float32Array} [cp]       (V) mean Cp of the networks; absent until predicted
 * @property {Float32Array} [sigma]    (V) spread σ(Cp): the networks' standard deviation × `ModelCard.spreadScale`
 * @property {Float32Array} [cfd]      (V) CFD Cp, only when the file carried it
 *
 * @typedef {'reading' | 'measuring' | 'meshing' | 'predicting'} Stage
 *
 * @typedef {Object} Engine  Runs in a worker; one instance per page. Create it on page load: it
 *   loads the networks and prepares the GPU up front so the first file starts at once.
 * @property {(file: File, on?: AnalyzeEvents, options?: { signal?: AbortSignal }) => Promise<Result>} analyze
 *   Resolves once `Result.mesh`, the working mesh, carries its fields. The working mesh is a
 *   faithful simplification that the tools (probing, section cuts, plots) run on.
 *   Rejects with an Error whose message says what is wrong with the file and what to do.
 *   Analyses run one at a time, in the order asked for, each to its end: a later one starts once
 *   every earlier one has sent its last event (`exact` and the last `fullFields`, or its failure) or
 *   was aborted, so a car is never left without its exact numbers. Its first `stage` event says it
 *   has started. Aborting `signal` abandons this analysis, whether waiting, running or ended, and
 *   releases what the engine holds of it: its promise rejects with a DOMException named 'AbortError'
 *   if still pending, and its events stop, including those after it resolved.
 * @property {(result: Result, point: number[], normal: number[]) => Promise<Explanation>} explain
 *   For a result this engine analysed or one `loadResult` read: everything the model was told at a
 *   surface point and what each network answered. For the file analysed last, while the engine holds
 *   its polygons, the networks are asked at the file polygon nearest `point` that does not face away
 *   from `normal`, as in the analysis. Otherwise (a bundled example, an earlier file) they are asked at
 *   the working-mesh vertex nearest `point` that does not face away from `normal`, with the vertex's
 *   normal, the area-weighted mean of the polygon normals around it. `Explanation.at` says which.
 *   Rejects for a result computed with other network weights than the engine runs.
 * @property {(result: Result) => boolean} canExportVtp  Whether `exportVtp` can run for `result`: true
 *   from its analysis's `exact` event until the next analysis starts or its `signal` is aborted (the
 *   engine holds the last file's prediction only); never for a bundled example, nor on a device too
 *   slow to predict every polygon. Check it when offering the export.
 * @property {(result: Result, onProgress?: (fraction: number) => void) => Promise<File>} exportVtp
 *   The analysed file with the prediction added as two Float32 cell arrays, one value per polygon:
 *   `CpPredicted`, the networks' mean pressure coefficient, and `CpSpread`, the spread σ(Cp). Every
 *   other byte is the file's own (an earlier `CpPredicted` or `CpSpread` is replaced), and the File is
 *   composed from slices of the original rather than a copy, so saving it streams from the user's
 *   disk. Named `<file>_prediction.vtp`. `onProgress` reports 0 to 1. Rejects with an Error whose
 *   message says what to do when `canExportVtp` is false or the original can no longer be read.
 * @property {Promise<string>} device  What runs the networks, short enough to show as is:
 *   "WebGPU (<adapter>)" when the browser names the adapter, "WebGPU", or "CPU (<n> threads)".
 * @property {Promise<ModelCard>} model  The model, for the model view.
 *
 * @typedef {Object} AnalyzeEvents  Everything is optional; each fires at most as described.
 * @property {(stage: Stage, fraction: number) => void} [stage]  Progress within the current stage, 0 to 1.
 *   Stages run in the order reading, measuring, meshing, predicting; each one's fractions never
 *   decrease, and it reports 1 before the next one starts.
 * @property {(mesh: DisplayMesh) => void} [geometry]
 *   The first drawable mesh, without fields, as early as possible so the car appears before any
 *   pressure is predicted. May fire again with a better mesh.
 * @property {(mesh: DisplayMesh) => void} [fullMesh]
 *   The file's own surface, every triangle, for drawing at full resolution, with `cfd` when the
 *   file carried it. Its `cp` and `sigma` arrive through `fullFields`. Does not fire when the device
 *   cannot hold the full mesh.
 * @property {(fields: { cp: Float32Array, sigma: Float32Array }, fraction: number) => void} [fullFields]
 *   `cp` and `sigma` for every vertex of the full mesh, complete and drawable on every call: a vertex
 *   averages those of its polygons predicted so far, and one with none yet holds the value of the
 *   nearest working-mesh vertex. `fraction` is the share of the file's polygons predicted. Fires after
 *   `analyze` has resolved, then repeatedly as more of the surface is predicted, ending at 1; on a
 *   device too slow to predict the whole surface in a minute it fires once, below 1.
 * @property {(numbers: Pick<Result, 'forces' | 'accuracy' | 'dragBuildup' | 'coverage'>) => void} [exact]
 *   Once every polygon of the file is predicted, just before the last `fullFields`: these parts of the
 *   Result summed over all of them, to replace the sampled ones. Does not fire on a device too slow
 *   to predict the whole surface in a minute.
 *
 * @typedef {Object} Explanation
 * @property {'polygon' | 'vertex'} at  where the networks were asked: a file polygon, or a working-mesh vertex
 * @property {number} index             the polygon's index in the file, or the vertex's in `Result.mesh`
 * @property {number[]} position        the polygon's centre, or the vertex's position
 * @property {number[]} normal          the outward unit normal there
 * @property {Float32Array} features    the model's inputs there, one per entry of `ModelCard.columns`
 * @property {number[]} cpModels        each network's Cp
 * @property {number} cp                their mean
 * @property {number} sigma             spread σ(Cp): their standard deviation × `ModelCard.spreadScale`
 *
 * @typedef {Object} ModelCard  What the model is and how it scored, from training (`export_web.py`).
 * @property {{ name: string, family: string, freestreamSpeed: number, yawDegrees: number, wheels: string,
 *              source: string }} dataset  the training data: its name, car family, freestream speed in
 *   m/s, yaw angle in degrees, the wheels' state, and where it is published
 * @property {number[]} trainRuns  the DrivAerML runs the ensemble was trained on
 * @property {{ trainRuns: number[], validationRuns: number[], parameters: number, layers: number[] }[]} networks
 *   per network: the runs it was fitted on, the runs held out to validate it, its parameter count, and
 *   its widths from input to output, e.g. [191, 256, 256, 256, 1]
 * @property {string} activation  what the networks compute between layers, e.g. 'GELU'
 * @property {number} spreadScale  the factor, fitted on the training cars, that turns the networks'
 *   standard deviation into the spread σ(Cp) the engine reports
 * @property {number} spreadCoverage  the share of the surface of the runs outside training where
 *   |Cp − CFD| ≤ σ(Cp), with σ already scaled (measured in training on an area-weighted cell sample)
 * @property {Record<'length' | 'width' | 'height', { min: number, max: number }>} dimensions  the
 *   training cars' range of each, in metres
 * @property {{ meanR2: number, worstCarR2: number, mae: number, relativeL2: number, dragMae: number }} crossValidation
 *   each training car predicted by the networks that did not see it
 * @property {{ run: number, r2: number, mae: number, dragMae: number }[]} outsideRuns
 *   runs outside the training set
 * @property {{ name: string, areaShare: number, r2: number, mae: number }[]} regions
 *   cross-validated accuracy by surface region
 * @property {{ label: string, learner: 'constant' | 'pointnet' | 'mlp' | 'trees' | 'ensemble', inputs: string[],
 *              crossValidationR2: number, worstCarR2: number, mae: number, dragMae: number,
 *              outsideR2: number, outsideDragMae: number }[]} comparison  the models compared in training:
 *   `learner` a stable id of what was fitted ('ensemble' is the shipped one), `inputs` the column kinds
 *   it was given, from the set of `ModelColumn.kind` in that order ([] for the constant baseline)
 * @property {ModelColumn[]} columns  what each of the model's inputs is, in order
 *
 * @typedef {Object} ModelColumn
 * @property {'position' | 'bend' | 'slice' | 'neighbour' | 'outline' | 'flag'} kind  its family, a stable id
 * @property {string} group     the family's display label
 * @property {string} name      a short plain name
 * @property {string} unit      'm' for distances, '' for the rest
 * @property {number[]} offset  the place it reads relative to the point asked about, [dx, dy, dz] in
 *   metres, dx downstream, dy outboard (away from the centre plane, on the point's side), dz up
 * @property {number[] | null} along  for a distance, the unit direction (same axes) it runs from that
 *   place, so place + value × along is where it measured to: the body surface, or for the distances
 *   from nose, tail, centre plane and lowest point, that plane; null for every other column
 *
 * @typedef {Object} ParsedSurface  The file's surface as the parser hands it to the rest of the engine.
 * @property {string} name
 * @property {Float32Array} points        (P, 3)
 * @property {Uint32Array} connectivity   the point ids of every polygon, back to back
 * @property {Uint32Array} offsets        (C + 1) polygon c owns connectivity[offsets[c] .. offsets[c + 1])
 * @property {Float32Array | null} cfd    (C) CpMeanTrim per polygon when the file carried it
 */

import { MeshoptDecoder } from '../vendor/meshoptimizer/meshopt_decoder.mjs';

/** Fields are stored to within half this step of Cp, far finer than the colour scale can show. */
const FIELD_STEP = 1 / 1024;
/** Bits per normal component in octahedral form. */
const NORMAL_BITS = 12;
const FIELDS = ['cp', 'sigma', 'cfd'];
/** The lookups' arrays, stored as they are: float32 rasters, float64 voxel sums. */
const LOOKUP_RASTERS = ['zTop', 'zBottom', 'occupied', 'yOut', 'xFirst', 'xLast', 'zPeak', 'sectionArea'];
const VOXEL_SUMS = ['area', 'normalX', 'normalY', 'normalZ'];

/**
 * Turn a result into the two files of a bundled example: `json` holds everything but the arrays,
 * `block` the mesh and the lookups, compressed. The mesh's vertices are reordered for compression,
 * and positions and fields are quantized: positions to 1/65535 of the bounding box (0.07 mm on a
 * car), fields to `FIELD_STEP`. The lookups are kept exact.
 *
 * @param {Result} result
 * @returns {Promise<{ json: string, block: Uint8Array }>}
 */
export async function packResult(result) {
  const { MeshoptEncoder } = await import('../vendor/meshoptimizer/meshopt_encoder.js');
  await MeshoptEncoder.ready;
  const { mesh } = result;
  const indices = mesh.indices.slice();
  // Vertices in the order the triangles first use them, which is what the codecs compress best.
  const [remap, vertexCount] = MeshoptEncoder.reorderMesh(indices, true, true);
  const reordered = (values, width) => {
    const out = new Float32Array(vertexCount * width);
    for (let v = 0; v < remap.length; v++) {
      if (remap[v] === 0xffffffff) continue;
      for (let k = 0; k < width; k++) out[remap[v] * width + k] = values[v * width + k];
    }
    return out;
  };

  const streams = [];
  const parts = [];
  const add = (stream, bytes) => {
    streams.push({ ...stream, bytes: bytes.length });
    parts.push(bytes);
  };
  add({ name: 'indices', encoding: 'triangles' },
    MeshoptEncoder.encodeIndexBuffer(new Uint8Array(indices.buffer), indices.length, 4));

  const positions = quantize(reordered(mesh.positions, 3), 3, (range) => range / 65535);
  add({ name: 'positions', encoding: 'quantized', origin: positions.origin, step: positions.step },
    MeshoptEncoder.encodeVertexBufferLevel(positions.bytes, vertexCount, positions.size, 3));

  const normals = reordered(mesh.normals, 3);
  const padded = new Float32Array(vertexCount * 4);
  for (let v = 0; v < vertexCount; v++) padded.set(normals.subarray(3 * v, 3 * v + 3), 4 * v);
  add({ name: 'normals', encoding: 'octahedral' }, MeshoptEncoder.encodeVertexBufferLevel(
    MeshoptEncoder.encodeFilterOct(padded, vertexCount, 8, NORMAL_BITS), vertexCount, 8, 3));

  for (const name of FIELDS) {
    if (!mesh[name]) continue;
    const field = quantize(reordered(mesh[name], 1), 1, (range) => Math.max(FIELD_STEP, range / 65535));
    add({ name, encoding: 'quantized', origin: field.origin, step: field.step },
      MeshoptEncoder.encodeVertexBufferLevel(field.bytes, vertexCount, field.size, 3));
  }

  const { voxels, ...lookups } = result.lookups;
  const lookupStreams = [];
  const addRaw = (name, array) => {
    const bytes = new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
    lookupStreams.push({ name, encoding: array instanceof Float64Array ? 'float64' : 'float32', bytes: bytes.length });
    parts.push(bytes);
  };
  for (const name of LOOKUP_RASTERS) addRaw(name, lookups[name]);
  voxels.forEach((grid, k) => VOXEL_SUMS.forEach((name) => addRaw(`voxels.${k}.${name}`, grid[name])));

  const summary = {
    ...result,
    mesh: { sourceTriangles: mesh.sourceTriangles, vertices: vertexCount, triangles: indices.length / 3, streams },
    lookups: {
      definition: lookups.definition,
      bounds: lookups.bounds,
      yCenter: lookups.yCenter,
      voxels: voxels.map(({ scale, origin, shape }) => ({ scale, origin, shape })),
      streams: lookupStreams,
    },
  };
  const compressed = new Blob(parts).stream().pipeThrough(new CompressionStream('gzip'));
  return { json: JSON.stringify(summary), block: new Uint8Array(await new Response(compressed).arrayBuffer()) };
}

/**
 * Each lane of `values` (`lanes` per vertex) as 16-bit steps up from the lane's minimum, two bytes
 * per lane, each vertex padded to whole 4-byte words as the vertex codec wants.
 */
function quantize(values, lanes, stepFor) {
  const count = values.length / lanes;
  const size = 4 * Math.ceil(lanes / 2);
  const origin = [], step = [];
  for (let lane = 0; lane < lanes; lane++) {
    let low = Infinity, high = -Infinity;
    for (let v = lane; v < values.length; v += lanes) {
      low = Math.min(low, values[v]);
      high = Math.max(high, values[v]);
    }
    origin.push(low);
    step.push(high > low ? stepFor(high - low) : 1);
  }
  const words = new Uint16Array((count * size) / 2);
  for (let v = 0; v < count; v++) {
    for (let lane = 0; lane < lanes; lane++) {
      words[(v * size) / 2 + lane] = Math.round((values[v * lanes + lane] - origin[lane]) / step[lane]);
    }
  }
  return { origin, step, size, bytes: new Uint8Array(words.buffer) };
}

/**
 * Rebuild a result from its JSON object and the block `packResult` wrote.
 *
 * @param {object} summary
 * @param {ArrayBuffer} buffer
 * @returns {Promise<Result>}
 */
export async function unpackResult(summary, buffer) {
  await MeshoptDecoder.ready;
  const decompressed = new Blob([buffer]).stream().pipeThrough(new DecompressionStream('gzip'));
  const block = new Uint8Array(await new Response(decompressed).arrayBuffer());
  const { sourceTriangles, vertices, triangles, streams } = summary.mesh;
  const mesh = { sourceTriangles };
  let offset = 0;
  for (const stream of streams) {
    const source = block.subarray(offset, offset + stream.bytes);
    offset += stream.bytes;
    switch (stream.encoding) {
      case 'triangles': {
        const indices = new Uint32Array(3 * triangles);
        MeshoptDecoder.decodeIndexBuffer(new Uint8Array(indices.buffer), indices.length, 4, source);
        mesh[stream.name] = indices;
        break;
      }
      case 'octahedral': {
        const packed = new Int16Array(4 * vertices);
        MeshoptDecoder.decodeVertexBuffer(new Uint8Array(packed.buffer), vertices, 8, source, 'OCTAHEDRAL');
        const normals = new Float32Array(3 * vertices);
        for (let v = 0; v < vertices; v++) {
          const x = packed[4 * v], y = packed[4 * v + 1], z = packed[4 * v + 2];
          const length = Math.sqrt(x * x + y * y + z * z);
          normals[3 * v] = x / length;
          normals[3 * v + 1] = y / length;
          normals[3 * v + 2] = z / length;
        }
        mesh[stream.name] = normals;
        break;
      }
      case 'quantized': {
        const lanes = stream.origin.length;
        const size = 4 * Math.ceil(lanes / 2);
        const words = new Uint16Array((vertices * size) / 2);
        MeshoptDecoder.decodeVertexBuffer(new Uint8Array(words.buffer), vertices, size, source);
        const values = new Float32Array(vertices * lanes);
        for (let v = 0; v < vertices; v++) {
          for (let lane = 0; lane < lanes; lane++) {
            values[v * lanes + lane] = stream.origin[lane] + stream.step[lane] * words[(v * size) / 2 + lane];
          }
        }
        mesh[stream.name] = values;
        break;
      }
      default:
        throw new Error(`This example stores its mesh as "${stream.encoding}", which this page cannot read.`);
    }
  }

  const { streams: lookupStreams, voxels, ...lookups } = summary.lookups;
  lookups.voxels = voxels.map((grid) => ({ ...grid }));
  for (const stream of lookupStreams) {
    // Copied out of the block, whose offsets need not suit the array's alignment.
    const bytes = block.slice(offset, offset + stream.bytes);
    offset += stream.bytes;
    const values = stream.encoding === 'float64' ? new Float64Array(bytes.buffer) : new Float32Array(bytes.buffer);
    const [first, k, name] = stream.name.split('.');
    if (first === 'voxels') lookups.voxels[Number(k)][name] = values;
    else lookups[first] = values;
  }
  return { ...summary, mesh, lookups };
}

/** Fetch a bundled example: `<base>.json` and `<base>.bin`. */
export async function loadResult(base) {
  const [summary, buffer] = await Promise.all([
    fetch(`${base}.json`).then((response) => response.json()),
    fetch(`${base}.bin`).then((response) => response.arrayBuffer()),
  ]);
  return unpackResult(summary, buffer);
}
