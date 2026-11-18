/**
 * The result of analysing one surface, from a file or from arrays, and its on-disk form.
 *
 * The engine returns a `Result` for a dropped file or a surface given as arrays. The bundled examples
 * are the same object stored as `<name>.json` (everything but the arrays) plus `<name>.bin` (gzip
 * over the mesh, in meshoptimizer's index and vertex codecs with quantized positions, fields and
 * normals, and over the lookups as raw arrays; see `packResult`), so the interface never needs to
 * know which of the two it is showing.
 *
 * Beside the `Engine`, two plain functions run on whatever thread calls them: `writeVtp` (vtp.js)
 * writes any surface with cell and point arrays as a .vtp file, and `boundsOf` (geometry.js) gives
 * the bounds of a set of points.
 *
 * Frame: metres, x downstream, z up, the car's centre plane at y = 0.
 *
 * @typedef {Object} Result
 * @property {'full' | 'quick'} form  How it was made. 'full': by `analyze`, from a working mesh
 *   simplified from the surface, with forces from an area sample of its polygons and then, through
 *   `exact`, from every one. 'quick': by `predictQuick`, every polygon of the surface it was given
 *   predicted at once, with the lookups built from those polygons, for studies.
 * @property {{ name: string, cells: number }} source  The surface analysed: its name (the file's) and polygon count.
 * @property {{ bounds: number[], length: number, width: number, height: number, frontalArea: number }} geometry
 *   `bounds` is [xmin, xmax, ymin, ymax, zmin, zmax]. `frontalArea` is the reference area for coefficients.
 * @property {{ cdP: number, cdPModels: number[] }} forces
 *   Pressure drag coefficient: the mean of the ensemble's networks and each network's value.
 * @property {{ r2: number } | null} accuracy
 *   Area-weighted R² of the networks' mean Cp against the reference field over the surface. Null
 *   when there is none.
 * @property {Totals} totals  The whole car's sums of the networks' mean Cp.
 *   `forces`, `accuracy` and `totals` are sums over the surface's polygons. In a full Result as
 *   `analyze` resolves with it they run over an evenly spaced area sample of them (drag within about
 *   0.0005 of the full sum); `exact` brings them summed over every polygon. A quick Result sums every
 *   polygon.
 * @property {ReferenceSource | null} reference  Where the field the prediction is compared with came
 *   from (shown as CFD, `mesh.cfd`): the surface's own `CpMeanTrim`, or a cell array of another file
 *   attached afterwards (`attachReference`). Null when there is none.
 * @property {ReferenceTotals | null} referenceTotals  The reference field's whole-car sums over every
 *   polygon of the surface, from the start (it needs no prediction), as `totals` sums the prediction's.
 *   Null when there is no reference.
 * @property {Coverage} coverage
 * @property {{ stage: Stage, seconds: number }[]} timings
 * @property {string} device  What ran the networks, as `Engine.device` names it.
 * @property {string} model   SHA-256 of the network weights it was computed with (`web/model/weights.bin`).
 * @property {DisplayMesh} mesh  A full Result's working mesh, or a quick Result's surface itself.
 * @property {ResultLookups} lookups
 * @property {CellSet} [cells]  A quick Result's polygons with the values the networks gave each.
 *
 * @typedef {Object} Totals
 * @property {number} area             m², the surface's
 * @property {number} meanCp           area-weighted
 * @property {number[]} coefficients   [Cx, Cy, Cz] = −Σ Cp n dA / `geometry.frontalArea`, pressure
 *   only; Cx is `forces.cdP`
 * @property {CpExtreme} cpMin
 * @property {CpExtreme} cpMax
 *
 * @typedef {Object} ReferenceTotals
 * @property {number} meanCp          area-weighted
 * @property {number[]} coefficients  [Cx, Cy, Cz] = −Σ Cp n dA / `geometry.frontalArea`, pressure only;
 *   Cx is the reference's pressure drag
 *
 * @typedef {Object} CpExtreme  The polygon where the networks' mean Cp is lowest or highest.
 * @property {number} value
 * @property {number[]} center
 * @property {number[]} normal  outward
 *
 * @typedef {Object} ReferenceSource
 * @property {string} file    the name of the file it came from
 * @property {string} array   the cell array read
 * @property {'own' | 'polygons' | 'nearest'} mapping  'own': the analysed surface's own `CpMeanTrim`;
 *   'polygons': another file with the same polygons, value for value; 'nearest': another file's
 *   polygons, each of the analysed polygons taking the value of the one whose centre is nearest among
 *   those not facing away from it
 *
 * @typedef {Object} CellSet  A surface's polygons with what the analysis knows of each, for the
 *   page's and the assistant's own sums and plots. `centers`, `normals` and `areas` are those the
 *   networks were asked at (VTK's, as in training); `area × cp × n` sums to the forces.
 * @property {Float32Array} points        (P, 3)
 * @property {Uint32Array} connectivity   as in `ParsedSurface`
 * @property {Uint32Array} offsets        (C + 1)
 * @property {Float32Array} centers       (C, 3)
 * @property {Float32Array} normals       (C, 3) unit, pointing out of the car
 * @property {Float64Array} areas         (C)
 * @property {Float32Array} cp            (C) the networks' mean Cp
 * @property {Float32Array} sigma         (C) spread σ(Cp)
 * @property {Float32Array | null} cfd    (C) the reference field, when there is one
 *
 * @typedef {import('./lookups.js').SurfaceLookups & { definition: import('./features.js').FeatureDefinition }} ResultLookups
 *   The whole-car lookups the features read (`lookups.js`, about 55 MB on a DrivAerML car), built
 *   from every polygon of the file, with the feature definition they were built for. They let
 *   `explain` answer for a result without its file; a bundled example keeps of them what `explain`
 *   reads at its vertices (`packResult`).
 *
 * @typedef {Object} Coverage  The car measured against the training cars. Facts only.
 * @property {{ name: 'length' | 'width' | 'height', value: number, trainMin: number, trainMax: number }[]} dimensions
 *
 * @typedef {Object} DisplayMesh  A surface the browser draws. Fields are per vertex: the networks are
 *   asked at the file's polygons, as they were trained, and the reference field gives one value per
 *   polygon too; a vertex shows the area-weighted mean of the file polygons around the file point it
 *   sits on.
 * @property {Float32Array} positions  (V, 3)
 * @property {Float32Array} normals    (V, 3) unit, pointing out of the car
 * @property {Uint32Array} indices     (T, 3) wound counter-clockwise seen from outside
 * @property {number[]} bounds         [xmin, xmax, ymin, ymax, zmin, zmax] of `positions`
 * @property {Float32Array} [cp]       (V) mean Cp of the networks; absent until predicted
 * @property {Float32Array} [sigma]    (V) spread σ(Cp): the networks' standard deviation × the model's spread scale
 * @property {Float32Array} [cfd]      (V) the reference field (`Result.reference`), only when there is one
 *
 * @typedef {'reading' | 'measuring' | 'meshing' | 'predicting'} Stage
 *
 * @typedef {Object} Engine  Runs in a worker; one instance per page. Create it on page load: it
 *   loads the networks and prepares the GPU up front so the first file starts at once.
 * @property {(file: File, on?: AnalyzeEvents, options?: { signal?: AbortSignal }) => Promise<Result>} analyze
 *   A full analysis. Resolves once `Result.mesh`, the working mesh, carries its fields. The working
 *   mesh is a faithful simplification that the tools (probing, section cuts, plots) run on.
 *   Rejects with an Error whose message says what is wrong with the file and what to do.
 *   Analyses run one at a time, in the order asked for, each to its end: a later one starts once
 *   every earlier one has sent its last event (`exact` and the last `fullFields`, or its failure) or
 *   was aborted, so a car is never left without its exact numbers. Its first `stage` event says it
 *   has started. Aborting `signal` abandons this analysis, whether waiting, running or ended, and
 *   releases what the engine holds of it: its promise rejects with a DOMException named 'AbortError'
 *   if still pending, and its events stop, including those after it resolved.
 * @property {(surface: ParsedSurface, options?: { signal?: AbortSignal }) => Promise<Result>} predictQuick
 *   A quick Result (`Result.form`) for a surface given as arrays, typically a variant of a working
 *   mesh (`workingCells`): no events, nothing drawn first, nothing held afterwards. It waits its turn
 *   behind running analyses like they do, but leaves the car analysed last held. The arrays move to
 *   the engine's worker rather than being copied, so the caller's are empty afterwards; each must
 *   therefore be a whole buffer of its own (`byteOffset` 0, its buffer's full length), or the call
 *   rejects saying so. Rejects with an Error naming the array that is wrong. Aborting `signal`
 *   abandons it.
 *   A mesh coarser than a working mesh still gives whole lookups (`lookups.js` takes a wide polygon
 *   in pieces), but the networks are asked at its polygons' centres and normals, which follow the
 *   car less closely, so the prediction loses some accuracy.
 * @property {(result: Result) => CellSet} workingCells  A result's working mesh as polygons, made on
 *   the page from `result.mesh` (positions and indices shared, not copied), with each triangle's Cp,
 *   σ and reference field the mean of its corners', so `area × cp` sums the field the viewer draws
 *   exactly. For a quick Result, its own `cells`. Works for every result, the bundled examples too.
 * @property {(result: Result) => Promise<CellSet | null>} fileCells  The polygons of the car analysed
 *   last, the file's own, copied in the worker and handed over whole (about 650 MB for a DrivAerML
 *   car); null unless `canExportVtp(result)`.
 * @property {(result: Result, point: number[], normal: number[]) => Promise<Explanation>} explain
 *   For a result this engine analysed or one `loadResult` read: everything the model was told at a
 *   surface point and what each network answered. For the file analysed last, while the engine holds
 *   its polygons, the networks are asked at the polygon `polygonAt` finds, as in the analysis.
 *   Otherwise (a bundled example, an earlier file) they are asked at the working-mesh vertex nearest
 *   `point` that does not face away from `normal`, with the vertex's normal, the area-weighted mean
 *   of the polygon normals around it. `Explanation.at` says which.
 *   Rejects for a result computed with other network weights than the engine runs.
 * @property {(result: Result, point: number[], normal: number[]) => Promise<FilePolygon | null>} polygonAt
 *   The file polygon under `point`, a point picked on the drawn surface whose normal there is
 *   `normal`: of the polygons not facing away from `normal`, the one whose surface is nearest `point`.
 *   For the file analysed last, from its result until the next analysis starts or its `signal` is
 *   aborted; null at once for any other result (a bundled example, an earlier file). A few
 *   milliseconds, also while the rest of the surface is being predicted.
 * @property {(result: Result) => boolean} canExportVtp  Whether `exportVtp` can run for `result`: true
 *   from its analysis's `exact` event until the next analysis starts or its `signal` is aborted (the
 *   engine holds the last car's prediction only); never for a bundled example or a quick Result, nor
 *   on a device too slow to predict every polygon. Check it when offering the export.
 * @property {(result: Result) => Promise<File>} exportVtp
 *   The analysed file with the prediction added as two Float32 cell arrays, one value per polygon:
 *   `CpPredicted`, the networks' mean pressure coefficient, and `CpSpread`, the spread σ(Cp). Every
 *   other byte is the file's own (an earlier `CpPredicted` or `CpSpread` is replaced), and the File
 *   is composed from slices of the original rather than a copy, so saving it streams from the user's
 *   disk. An attached reference field is written as a cell array of the name it was read under
 *   (`Result.reference.array`). Named `<name>_prediction.vtp`. Rejects with an Error whose message
 *   says what to do when `canExportVtp` is false or the original can no longer be read.
 * @property {(result: Result, file: File, options?: { array?: string }) => Promise<ReferenceChange>} attachReference
 *   Gives the car a reference field from a second file: its cell array `array` (default `CpMeanTrim`),
 *   put on the car's polygons as `ReferenceSource.mapping` says. The car then answers as if its own
 *   file had carried the field: `result.reference`, `result.referenceTotals`, `result.accuracy` and
 *   `result.mesh.cfd` are written into `result` itself, computed over every polygon exactly as the
 *   analysis computes them, and `fileCells`, `explain` and `exportVtp` use the field. Of the second
 *   file, only that array is kept. Runs while `canExportVtp(result)`; rejects otherwise, with a
 *   message naming the arrays the file has when it lacks `array` or saying it has none, and with one
 *   giving the median distance between the two surfaces when the second file's polygons are not the
 *   car's and its surface lies more than 5 mm from the car's: a field on another car's surface is no
 *   reference for this one.
 * @property {Promise<string>} device  What runs the networks, short enough to show as is:
 *   "WebGPU (<adapter>)" when the browser names the adapter, "WebGPU", or "CPU (<n> threads)".
 * @property {Promise<ModelCard>} model  The model, for the Model page.
 *
 * @typedef {Object} ReferenceChange  The reference field on the meshes the page draws, which it must
 *   update itself.
 * @property {Float32Array} cfd             (V) the new `mesh.cfd`
 * @property {Float32Array | null} fullCfd  (P) the field at every vertex of the full mesh, when
 *   `fullMesh` fired for this car
 *
 * @typedef {Object} AnalyzeEvents  Everything is optional; each fires at most as described.
 * @property {(stage: Stage, fraction: number) => void} [stage]  Progress within the current stage, 0 to 1.
 *   Stages run in the order reading, measuring, meshing, predicting; each one's fractions never
 *   decrease, and it reports 1 before the next one starts.
 * @property {(mesh: DisplayMesh) => void} [geometry]
 *   The first drawable mesh, without fields, as early as possible so the car appears before any
 *   pressure is predicted: the full mesh, or the working mesh when the full one does not fire.
 * @property {(mesh: DisplayMesh) => void} [fullMesh]
 *   The file's own surface, every triangle, for drawing at full resolution, with `cfd` when the
 *   surface carries a reference field. Its `cp` and `sigma` arrive through `fullFields`. Does not
 *   fire when the device cannot hold the full mesh.
 * @property {(fields: { cp: Float32Array, sigma: Float32Array }, fraction: number) => void} [fullFields]
 *   `cp` and `sigma` for every vertex of the full mesh, complete and drawable on every call: a vertex
 *   averages those of its polygons predicted so far, and one with none yet holds the value of the
 *   nearest working-mesh vertex. `fraction` is the share of the file's polygons predicted. Fires after
 *   `analyze` has resolved, then repeatedly as more of the surface is predicted, ending at 1; on a
 *   device too slow to predict the whole surface in a minute it fires once, below 1.
 * @property {(numbers: Pick<Result, 'forces' | 'accuracy' | 'totals'>) => void} [exact]
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
 * @property {number} sigma             spread σ(Cp): their standard deviation × the model's spread scale
 * @property {number | null} cfd        the reference field there: the polygon's own value, or
 *   `Result.mesh.cfd` at the vertex; null when there is no reference
 * @property {Float32Array[][]} activations  per network, what it computed on the way to its Cp:
 *   [input, hidden 1, hidden 2, hidden 3]. `input` is `features` standardized and clipped by that
 *   network's own scaling (what its first layer reads); each hidden layer is its output after GELU.
 *
 * @typedef {Object} FilePolygon  One polygon of the analysed file.
 * @property {number} index          its index among the file's polygons
 * @property {Float32Array} corners  (K, 3) its corners, in the file's order
 * @property {number[]} center       its centre, where the networks are asked
 * @property {number[]} normal       its outward unit normal
 * @property {number} cp             the networks' mean Cp there, the value `exportVtp` writes for it.
 *   Before the background prediction reaches the polygon, the networks are asked there on demand
 *   on the CPU, which agrees with the GPU's value to about 1e-6.
 * @property {number} sigma          spread σ(Cp) there, likewise
 *
 * @typedef {Object} ModelCard  What the model is and how it scored, from training (`export_web.py`).
 * @property {{ name: string, family: string, freestreamSpeed: number, yawDegrees: number, wheels: string }} dataset
 *   the training data: its name, car family, freestream speed in m/s, yaw angle in degrees and the
 *   wheels' state
 * @property {number[]} trainRuns  the DrivAerML runs the ensemble was trained on
 * @property {number} samplesPerCar  the area-weighted surface cells each training car gave the networks
 * @property {{ trainRuns: number[], validationRuns: number[], layers: number[] }[]} networks
 *   per network: the runs it was fitted on, the runs held out to validate it, and its widths from
 *   input to output (the input count first, 1 last)
 * @property {string} activation  what the networks compute between layers, e.g. 'GELU'
 * @property {number} spreadCoverage  the share of the surface of the runs outside training where
 *   |Cp − CFD| ≤ σ(Cp), with σ already scaled (measured in training on an area-weighted cell sample)
 * @property {Record<'length' | 'width' | 'height', { min: number, max: number }>} dimensions  the
 *   training cars' range of each, in metres
 * @property {{ meanR2: number }} crossValidation  each training car predicted by the networks that
 *   did not see it
 * @property {{ run: number, r2: number }[]} outsideRuns  runs outside the training set
 * @property {{ learner: 'constant' | 'pointnet' | 'mlp' | 'trees' | 'ensemble', dragMae: number }[]} comparison
 *   the models compared in training: `learner` a stable id of what was fitted ('ensemble' is the
 *   shipped one), and its drag error in cross-validation
 * @property {ModelColumn[]} columns  what each of the model's inputs is, in order
 *
 * @typedef {Object} ModelColumn
 * @property {'position' | 'surface' | 'slice'} kind  its family, a stable id
 * @property {string} unit      'm' for distances, 'm²' for a slice's area, '' for the rest
 * @property {number[]} offset  the place it reads relative to the point asked about, [dx, dy, dz] in
 *   metres along the file's axes: dx downstream, dy toward +y (the car's right as seen from the
 *   driver's seat), dz up
 * @property {number[] | null} along  for a distance, the unit direction (same axes) it runs from that
 *   place, so place + value × along is where it measured to: the body surface, or for the distances
 *   from nose, tail, centre plane and lowest point, that plane; null for every other column. The
 *   reach outboard of a slice ('slice', read 'outboard') is measured away from the centre plane
 *   on the point's own side: its [0, 1, 0] holds for a point on the +y side and turns to [0, −1, 0]
 *   on the −y side
 * @property {number} [scale]  a surface column's size of nearby surface, in metres
 * @property {number} [sigma]  a surface column's Gaussian standard deviation, in metres
 * @property {'whole' | 'ahead' | 'behind'} [part]  a surface column's part of the nearby surface: all
 *   of it, the part ahead of the point (upstream, at smaller x), or the part behind
 * @property {number} [station]  a slice column's station, its dx from the point in metres (0 for
 *   the point's own slice, negative ahead of it)
 * @property {'area' | 'outboard' | 'above' | 'below'} [read]  what a slice column reads: the
 *   slice's area, or how far the body in it reaches outboard of the point at the point's height,
 *   above it or below it at the point's distance from the centre plane (to the ground where the
 *   slice holds no body there)
 *
 * @typedef {Object} ParsedSurface  A surface as the parser hands it to the rest of the engine, or as
 *   the page gives it to `predictQuick`, where `cfd` may be left out.
 * @property {string} name
 * @property {Float32Array} points        (P, 3)
 * @property {Uint32Array} connectivity   the point ids of every polygon, back to back
 * @property {Uint32Array} offsets        (C + 1) polygon c owns connectivity[offsets[c] .. offsets[c + 1]);
 *   offsets[0] is 0 and offsets[C] the length of `connectivity`
 * @property {Float32Array | null} [cfd]  (C) CpMeanTrim per polygon when the surface carries it
 */

import { MeshoptDecoder } from '../vendor/meshoptimizer/meshopt_decoder.mjs';
import { boundsOf } from './geometry.js';
import { scaleReadsAt } from './features.js';

/** Fields are stored to within half this step of Cp, far finer than the colour scale can show. */
const FIELD_STEP = 1 / 1024;
/** Bits per normal component in octahedral form. */
const NORMAL_BITS = 12;
const FIELDS = ['cp', 'sigma', 'cfd'];

/** The lookups' arrays by their place in `SurfaceLookups`. */
function lookupArrays({ yOut, zTop, zBottom, sliceArea, scales }) {
	return [
		['yOut', yOut],
		['zTop', zTop],
		['zBottom', zBottom],
		['sliceArea', sliceArea],
		...scales.flatMap(({ kept, whole, ahead, readsAtVertices }, k) =>
			readsAtVertices
				? [[`scales.${k}.readsAtVertices`, readsAtVertices]]
				: [
						[`scales.${k}.kept`, kept],
						[`scales.${k}.whole`, whole],
						[`scales.${k}.ahead`, ahead],
					],
		),
	];
}

/**
 * The bytes of an array of 4-byte values grouped by their place in the value: every value's lowest
 * byte, then every value's next byte, and so on. Neighbouring values share their high bytes, so the
 * groups give gzip long runs: a car's lookups pack 17% smaller.
 */
function groupedBytes(array) {
	const bytes = new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
	const count = array.length;
	const grouped = new Uint8Array(bytes.length);
	for (let place = 0; place < 4; place++) {
		for (let i = 0; i < count; i++) grouped[place * count + i] = bytes[4 * i + place];
	}
	return grouped;
}

function ungroupedBytes(grouped) {
	const count = grouped.length / 4;
	const bytes = new Uint8Array(grouped.length);
	for (let place = 0; place < 4; place++) {
		for (let i = 0; i < count; i++) bytes[4 * i + place] = grouped[place * count + i];
	}
	return bytes;
}

/**
 * Turn a result into the two files of a bundled example: `json` holds everything but the arrays,
 * `block` the mesh and the lookups, compressed. The mesh's vertices are reordered for compression,
 * and positions and fields are quantized: positions to 1/65535 of the bounding box (0.07 mm on a
 * car), fields to `FIELD_STEP`. The lookups are kept exact, but a scale kept at more nodes than the
 * mesh has vertices (the 3 cm scale on a DrivAerML car: 1.3M nodes against 0.3M vertices) is stored
 * as its reads at the vertices, all that `explain` asks of it on a bundled example, so the example
 * stays under the host's 25 MB a file.
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
	add(
		{ name: 'indices', encoding: 'triangles' },
		MeshoptEncoder.encodeIndexBuffer(new Uint8Array(indices.buffer), indices.length, 4),
	);

	const positions = quantize(reordered(mesh.positions, 3), 3, (range) => range / 65535);
	add(
		{ name: 'positions', encoding: 'quantized', origin: positions.origin, step: positions.step },
		MeshoptEncoder.encodeVertexBufferLevel(positions.bytes, vertexCount, positions.size, 3),
	);

	const normals = reordered(mesh.normals, 3);
	const padded = new Float32Array(vertexCount * 4);
	for (let v = 0; v < vertexCount; v++) padded.set(normals.subarray(3 * v, 3 * v + 3), 4 * v);
	add(
		{ name: 'normals', encoding: 'octahedral' },
		MeshoptEncoder.encodeVertexBufferLevel(
			MeshoptEncoder.encodeFilterOct(padded, vertexCount, 8, NORMAL_BITS),
			vertexCount,
			8,
			3,
		),
	);

	for (const name of FIELDS) {
		if (!mesh[name]) continue;
		const field = quantize(reordered(mesh[name], 1), 1, (range) => Math.max(FIELD_STEP, range / 65535));
		add(
			{ name, encoding: 'quantized', origin: field.origin, step: field.step },
			MeshoptEncoder.encodeVertexBufferLevel(field.bytes, vertexCount, field.size, 3),
		);
	}

	// A scale with more kept nodes than the mesh has vertices goes as its reads at the vertices: eight
	// floats a vertex against nine numbers a node.
	const { definition, bounds, yCenter, scales } = result.lookups;
	const stored = {
		...result.lookups,
		scales: scales.map((scale, k) => {
			if (8 * vertexCount >= 9 * scale.kept.length) return scale;
			const grid = { origin: scale.origin, nodes: scale.nodes, step: definition.surfaceSteps[k] };
			return { readsAtVertices: reordered(scaleReadsAt(grid, scale, mesh.positions), 8) };
		}),
	};
	// Kept nodes ascend, so their differences are small and pack to almost nothing.
	const lookupStreams = [];
	for (const [name, array] of lookupArrays(stored)) {
		const ascending = array instanceof Int32Array;
		const bytes = groupedBytes(ascending ? array.map((node, i) => (i ? node - array[i - 1] : node)) : array);
		lookupStreams.push({ name, encoding: ascending ? 'int32 differences' : 'float32', bytes: bytes.length });
		parts.push(bytes);
	}

	const summary = {
		...result,
		mesh: { vertices: vertexCount, triangles: indices.length / 3, streams },
		lookups: {
			definition,
			bounds,
			yCenter,
			scales: scales.map(({ origin, nodes }) => ({ origin, nodes })),
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
	const origin = [],
		step = [];
	for (let lane = 0; lane < lanes; lane++) {
		let low = Infinity,
			high = -Infinity;
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
	const { vertices, triangles, streams } = summary.mesh;
	const mesh = {};
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
					const x = packed[4 * v],
						y = packed[4 * v + 1],
						z = packed[4 * v + 2];
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
	mesh.bounds = boundsOf(mesh.positions);

	const { streams: lookupStreams, ...lookups } = summary.lookups;
	lookups.scales = lookups.scales.map((scale) => ({ ...scale }));
	for (const stream of lookupStreams) {
		const { buffer: bytes } = ungroupedBytes(block.subarray(offset, offset + stream.bytes));
		offset += stream.bytes;
		let values;
		if (stream.encoding === 'int32 differences') {
			values = new Int32Array(bytes);
			for (let i = 1; i < values.length; i++) values[i] += values[i - 1];
		} else {
			values = new Float32Array(bytes);
		}
		const path = stream.name.split('.');
		let owner = lookups;
		for (const key of path.slice(0, -1)) owner = owner[key];
		owner[path.at(-1)] = values;
	}
	return { ...summary, mesh, lookups };
}

/**
 * The `CellSet` of a result's working mesh, made on the page: its triangles as polygons, with each
 * triangle's Cp and σ the mean of its corners', so area × Cp sums the field the viewer draws exactly.
 * A quick result's own polygons, with the values the networks gave them, when it has them.
 * @param {Result} result
 * @returns {CellSet}
 */
export function workingCells(result) {
	if (result.cells) return result.cells;
	const { positions, indices, cp, sigma, cfd } = result.mesh;
	const count = indices.length / 3;
	const offsets = new Uint32Array(count + 1);
	const centers = new Float32Array(3 * count);
	const normals = new Float32Array(3 * count);
	const areas = new Float64Array(count);
	const cells = {
		cp: new Float32Array(count),
		sigma: new Float32Array(count),
		cfd: cfd ? new Float32Array(count) : null,
	};
	// One pass with plain loops: the page calls this on its own thread, where it must stay well under a frame budget.
	for (let t = 0; t < count; t++) {
		const i = indices[3 * t],
			j = indices[3 * t + 1],
			k = indices[3 * t + 2];
		const a = 3 * i,
			b = 3 * j,
			c = 3 * k;
		const ux = positions[b] - positions[a],
			uy = positions[b + 1] - positions[a + 1],
			uz = positions[b + 2] - positions[a + 2];
		const vx = positions[c] - positions[a],
			vy = positions[c + 1] - positions[a + 1],
			vz = positions[c + 2] - positions[a + 2];
		const nx = uy * vz - uz * vy,
			ny = uz * vx - ux * vz,
			nz = ux * vy - uy * vx;
		const length = Math.sqrt(nx * nx + ny * ny + nz * nz);
		areas[t] = length / 2;
		if (length > 0) {
			normals[3 * t] = nx / length;
			normals[3 * t + 1] = ny / length;
			normals[3 * t + 2] = nz / length;
		}
		centers[3 * t] = (positions[a] + positions[b] + positions[c]) / 3;
		centers[3 * t + 1] = (positions[a + 1] + positions[b + 1] + positions[c + 1]) / 3;
		centers[3 * t + 2] = (positions[a + 2] + positions[b + 2] + positions[c + 2]) / 3;
		cells.cp[t] = (cp[i] + cp[j] + cp[k]) / 3;
		cells.sigma[t] = (sigma[i] + sigma[j] + sigma[k]) / 3;
		if (cfd) cells.cfd[t] = (cfd[i] + cfd[j] + cfd[k]) / 3;
		offsets[t + 1] = 3 * (t + 1);
	}
	return { points: positions, connectivity: indices, offsets, centers, normals, areas, ...cells };
}

/**
 * Fetch and decode a bundled example, `<base>.json` and `<base>.bin`, in a worker of its own so the
 * page keeps drawing while the mesh decodes (100 ms and more for a car); the arrays come back
 * transferred, not copied.
 * @param {string} base  absolute URL of the example, without its extension
 * @returns {Promise<Result>}
 */
export function loadResult(base) {
	const worker = new Worker(new URL('./example-worker.js', import.meta.url), { type: 'module' });
	return new Promise((resolve, reject) => {
		worker.onmessage = ({ data }) => (data.result ? resolve(data.result) : reject(new Error(data.message)));
		worker.onerror = (event) => {
			event.preventDefault();
			reject(new Error(event.message));
		};
		worker.postMessage(base);
	}).finally(() => worker.terminate());
}
