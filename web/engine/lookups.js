/**
 * A car's lookups: what the features read, built once from all of its polygons as `SurfaceContext`
 * in `features.py` builds them. The raster is a picture of the car in its own frame (x from the nose,
 * y from the centre plane, z from the ground) on `rasterStep` bins that the slices across it are read
 * from: how far the body reaches from the centre plane at each x and height (on `heightStep` height
 * bins, within `heightWindow` of the height), the body's top and bottom over each (x, y) column, and
 * each slice's area S(x). Each surface scale is the area and area-weighted normal spread on a grid
 * by quadratic B-spline weights and blurred, whole and cut to its upstream half; its grids are built
 * on the GPU when there is one (`grid-gpu.js`), else on the CPU (`grid.js`), and the same lookups
 * come out of both.
 *
 * Both take the surface from samples of it, as `features.py` `surface_samples` does: a polygon's
 * centre, or the pieces of a polygon wider than half a raster step (`widePieces`), so a mesh coarser
 * than the raster fills every bin it covers.
 *
 * A surface scale's fields are kept only at the nodes around the polygon centres and the file's
 * points (a few percent of its grid): every read at a polygon, and at a working-mesh vertex, which
 * sits on a file point.
 */

import { correlateAlong, spreadOnGrid } from './grid.js';

/**
 * How far from a polygon centre or a file point a read must still find its nodes kept: beyond the
 * float32 rounding of the GPU's grid coordinates (1 µm), and beyond a bundled example's rounding of
 * its vertex positions (`packResult`, half of 1/65535 of the car: 0.04 mm on a 5 m car).
 */
const KEPT_REACH = 5e-5;

/**
 * The nodes one block of a GPU spread's fixed-point sums holds, at 8 bytes a node: WebGPU's
 * guaranteed binding size of 128 MiB. The finest grid around a DrivAerML car (17M nodes) is spread in
 * two; blocks of this size on every device keep every GPU on the same path.
 */
const SUMS_BLOCK_NODES = (128 * 1024 * 1024) / 8;

/**
 * @typedef {Object} ScaleLookup  The surface around the car at one scale, kept at the nodes around it.
 * @property {number[]} origin
 * @property {number[]} nodes
 * @property {Int32Array} kept       (K) flat numbers of the nodes kept, ascending
 * @property {Float32Array} whole    (K, 4) blurred area, then area × each normal component
 * @property {Float32Array} ahead    (K, 4) the same with the Gaussian cut to its upstream half
 * @property {Float32Array} [readsAtVertices]  (V, 8) in a bundled example, in place of the three
 *   above when they would be larger: whole then ahead read at each vertex of its mesh (`packResult`)
 *
 * @typedef {Object} SurfaceLookups
 * @property {number[]} bounds   [xmin, xmax, ymin, ymax, zmin, zmax] of the polygon centres
 * @property {number} yCenter
 * @property {Float32Array} yOut         (2, nx, nh) how far the body reaches from the centre plane at
 *   each x bin and height bin, within `heightWindow` of the bin's height, [0] toward −y, [1] toward
 *   +y; 0 where it reaches none of those bins
 * @property {Float32Array} zTop         (nx, ny) the highest sample over each (x, y) column, with y
 *   measured from the centre plane; the ground (the lowest polygon centre) where the column holds none
 * @property {Float32Array} zBottom      (nx, ny) the lowest, the ground where the column holds none
 * @property {Float32Array} sliceArea    (nx) the slice across the car at each x, its area S(x) in m²: the
 *   columns' heights summed across y, times the raster step
 *   (x bins are counted from the nose, height bins from the ground; heights themselves are absolute z)
 * @property {ScaleLookup[]} scales  in `FeatureLayout.scales` order
 *
 * @typedef {Object} WidePieces  The pieces the wide polygons are taken in.
 * @property {Uint8Array} wide         (C) 1 for a polygon with an edge longer than half a raster step
 * @property {Float32Array} positions  (S, 3) each piece's centroid
 * @property {Float64Array} areas      (S) each piece's area
 * @property {Uint32Array} polygons    (S) the polygon each piece is cut from
 */

/**
 * @param {import('./result.js').ParsedSurface} surface  its points are the file's points (P, 3)
 * @param {import('./geometry.js').CellGeometry} cells   the surface's polygons, measured
 * @param {import('./features.js').FeatureLayout} layout
 * @param {ReturnType<import('./grid-gpu.js').createGridOperations> | null} grids  the GPU's grid operations, or none
 * @returns {Promise<SurfaceLookups>}
 */
export async function buildLookups(surface, cells, layout, grids) {
	const { bounds } = cells;
	const yCenter = 0.5 * (bounds[2] + bounds[3]);
	const pieces = widePieces(surface, layout.rasterStep / 2);
	const scaleGrids = layout.scales.map((scale) => scaleGrid(scale, bounds));
	const scales = grids
		? await scalesOnGpu(grids, cells, pieces, surface.points, layout, scaleGrids)
		: scalesOnCpu(cells, pieces, surface.points, layout, scaleGrids);
	return {
		bounds,
		yCenter,
		...carRaster(cells, pieces, layout, bounds, yCenter),
		scales: scaleGrids.map(({ origin, nodes }, s) => ({ origin, nodes, ...scales[s] })),
	};
}

/**
 * The wide polygons' pieces, which with the other polygons' centres are where the lookups take the
 * surface from, as `features.py` takes it. A polygon whose edges are all at most `half` (half a raster
 * step) is one sample: its centre, carrying its area. A wider polygon
 * is taken in pieces, so a mesh coarser than the raster still fills every bin it covers and lays its
 * area where it lies: each triangle of its fan from the first corner is cut into n² congruent pieces,
 * n = ⌈longest edge / half⌉, each a sample at its centroid carrying its area and the polygon's
 * normal. Lengths and positions are worked in float64 from the float32 points, and the positions
 * rounded to float32.
 * @returns {WidePieces}
 */
function widePieces({ points, connectivity, offsets }, half) {
	const count = offsets.length - 1;
	const wide = new Uint8Array(count);
	const limit = half * half;
	/** Each fan triangle of polygon `c` as its first corner, its two edges from it and its piece count. */
	const eachTriangle = (c, visit) => {
		const a = 3 * connectivity[offsets[c]];
		for (let i = offsets[c] + 1; i + 1 < offsets[c + 1]; i++) {
			const b = 3 * connectivity[i],
				d = 3 * connectivity[i + 1];
			const ux = points[b] - points[a],
				uy = points[b + 1] - points[a + 1],
				uz = points[b + 2] - points[a + 2];
			const vx = points[d] - points[a],
				vy = points[d + 1] - points[a + 1],
				vz = points[d + 2] - points[a + 2];
			const wx = points[d] - points[b],
				wy = points[d + 1] - points[b + 1],
				wz = points[d + 2] - points[b + 2];
			const longest = Math.max(ux * ux + uy * uy + uz * uz, vx * vx + vy * vy + vz * vz, wx * wx + wy * wy + wz * wz);
			visit(a, ux, uy, uz, vx, vy, vz, Math.ceil(Math.sqrt(longest) / half));
		}
	};

	let pieces = 0;
	for (let c = 0; c < count; c++) {
		const begin = offsets[c],
			end = offsets[c + 1];
		let previous = 3 * connectivity[end - 1];
		for (let i = begin; i < end; i++) {
			const p = 3 * connectivity[i];
			const dx = points[p] - points[previous],
				dy = points[p + 1] - points[previous + 1],
				dz = points[p + 2] - points[previous + 2];
			if (dx * dx + dy * dy + dz * dz > limit) {
				wide[c] = 1;
				eachTriangle(c, (a, ux, uy, uz, vx, vy, vz, n) => (pieces += n * n));
				break;
			}
			previous = p;
		}
	}

	const positions = new Float32Array(3 * pieces);
	const areas = new Float64Array(pieces);
	const polygons = new Uint32Array(pieces);
	let s = 0;
	const put = (c, a, ux, uy, uz, vx, vy, vz, u, v, area) => {
		positions[3 * s] = points[a] + u * ux + v * vx;
		positions[3 * s + 1] = points[a + 1] + u * uy + v * vy;
		positions[3 * s + 2] = points[a + 2] + u * uz + v * vz;
		areas[s] = area;
		polygons[s++] = c;
	};
	for (let c = 0; c < count; c++) {
		if (!wide[c]) continue;
		eachTriangle(c, (a, ux, uy, uz, vx, vy, vz, n) => {
			const cx = uy * vz - uz * vy,
				cy = uz * vx - ux * vz,
				cz = ux * vy - uy * vx;
			const area = (0.5 * Math.sqrt(cx * cx + cy * cy + cz * cz)) / (n * n);
			for (let i = 0; i < n; i++) {
				for (let j = 0; i + j < n; j++) {
					put(c, a, ux, uy, uz, vx, vy, vz, (i + 1.0 / 3.0) / n, (j + 1.0 / 3.0) / n, area);
					if (i + j <= n - 2) put(c, a, ux, uy, uz, vx, vy, vz, (i + 2.0 / 3.0) / n, (j + 2.0 / 3.0) / n, area);
				}
			}
		});
	}
	return { wide, positions, areas, polygons };
}

/** A scale's grid: the polygon centres' bounds and a margin of the Gaussian's reach and two nodes. */
function scaleGrid({ step, radius }, bounds) {
	const margin = (radius + 2) * step;
	const origin = [bounds[0] - margin, bounds[2] - margin, bounds[4] - margin];
	const nodes = [1, 3, 5].map((k, axis) => Math.ceil((bounds[k] + margin - origin[axis]) / step) + 1);
	return { origin, step, nodes };
}

/**
 * The raster, from every sample in the reference's float32 arithmetic, so each lands in the bin it
 * lands in there; S(x) sums each row in NumPy's order and so is the reference's to the bit. Bins are
 * counted from the nose (x − xmin) and the ground (z − zmin), both float32 values, so a car moved
 * whole lands in the same bins.
 */
function carRaster(
	{ centers },
	{ wide, positions },
	{ h, x0, yHalf, z0, heightBin, nx, ny, nh, heightReach },
	bounds,
	yCenter,
) {
	const f32 = Math.fround;
	const [nose, , , , ground] = bounds;
	const yCenter32 = f32(yCenter);
	const reachInBin = new Float32Array(2 * nx * nh);
	const zTop = new Float32Array(nx * ny).fill(-Infinity);
	const zBottom = new Float32Array(nx * ny).fill(Infinity);
	const place = (xyz, i) => {
		const z = xyz[i + 2];
		const yRelative = f32(xyz[i + 1] - yCenter32);
		const ix = clamp(Math.trunc(f32(f32(f32(xyz[i] - nose) - x0) / h)), nx);
		const iy = clamp(Math.trunc(f32(f32(yRelative + yHalf) / h)), ny);
		const ih = clamp(Math.trunc(f32(f32(f32(z - ground) - z0) / heightBin)), nh);
		const bin = ((yRelative >= 0 ? nx : 0) + ix) * nh + ih;
		if (Math.abs(yRelative) > reachInBin[bin]) reachInBin[bin] = Math.abs(yRelative);
		const column = ix * ny + iy;
		if (z > zTop[column]) zTop[column] = z;
		if (z < zBottom[column]) zBottom[column] = z;
	};
	for (let c = 0; c < wide.length; c++) if (!wide[c]) place(centers, 3 * c);
	for (let i = 0; i < positions.length; i += 3) place(positions, i);
	// Each height bin takes the furthest reach within `heightReach` bins of it (SciPy's
	// `maximum_filter1d`, nothing beyond the ends), so that a point's reach does not swing by the car's
	// half-width with where a thin bin's edge falls against a roof or sill.
	const yOut = new Float32Array(reachInBin.length);
	for (let start = 0; start < yOut.length; start += nh) {
		for (let ih = 0; ih < nh; ih++) {
			let furthest = 0;
			for (let k = Math.max(ih - heightReach, 0); k <= Math.min(ih + heightReach, nh - 1); k++)
				furthest = Math.max(furthest, reachInBin[start + k]);
			yOut[start + ih] = furthest;
		}
	}
	const heights = new Float32Array(nx * ny);
	for (let column = 0; column < heights.length; column++) {
		if (zTop[column] === -Infinity) {
			zTop[column] = ground;
			zBottom[column] = ground;
		}
		heights[column] = zTop[column] - zBottom[column];
	}
	const sliceArea = new Float32Array(nx);
	for (let ix = 0; ix < nx; ix++) sliceArea[ix] = f32(numpySum32(heights, ix * ny, ny) * h);
	return { yOut, zTop, zBottom, sliceArea };
}

function clamp(index, size) {
	return index < 0 ? 0 : index >= size ? size - 1 : index;
}

/**
 * values[start, start + n) summed in float32 in NumPy's order (`pairwise_sum`): eight running sums
 * for up to 128 values, joined in pairs, and halves summed apart beyond that.
 */
function numpySum32(values, start, n) {
	const f32 = Math.fround;
	if (n < 8) {
		let sum = 0;
		for (let i = 0; i < n; i++) sum = f32(sum + values[start + i]);
		return sum;
	}
	if (n <= 128) {
		const r = Array.from(values.subarray(start, start + 8));
		let i = 8;
		for (; i < n - (n % 8); i += 8) {
			for (let j = 0; j < 8; j++) r[j] = f32(r[j] + values[start + i + j]);
		}
		let sum = f32(f32(f32(r[0] + r[1]) + f32(r[2] + r[3])) + f32(f32(r[4] + r[5]) + f32(r[6] + r[7])));
		for (; i < n; i++) sum = f32(sum + values[start + i]);
		return sum;
	}
	let half = Math.floor(n / 2);
	half -= half % 8;
	return f32(numpySum32(values, start, half) + numpySum32(values, start + half, n - half));
}

/**
 * One of the surface's four carried quantities (area, then area × each normal component) at each
 * polygon centre into `out`: none at a wide polygon, whose pieces carry it.
 */
function carriedAtCentres({ normals, areas }, { wide }, quantity, out) {
	for (let c = 0; c < areas.length; c++)
		out[c] = wide[c] ? 0 : quantity === 0 ? areas[c] : areas[c] * normals[3 * c + quantity - 1];
	return out;
}

/** The same quantity at each piece of the wide polygons, with its polygon's normal. */
function carriedByPieces({ normals }, { areas, polygons }, quantity, out) {
	for (let s = 0; s < areas.length; s++)
		out[s] = quantity === 0 ? areas[s] : areas[s] * normals[3 * polygons[s] + quantity - 1];
	return out;
}

function scalesOnCpu(cells, pieces, points, layout, scaleGrids) {
	const atCentres = new Float64Array(cells.areas.length);
	const byPieces = new Float64Array(pieces.areas.length);
	return layout.scales.map((scale, s) => {
		const grid = scaleGrids[s];
		const kept = keptNodes(grid, cells.centers, points);
		const whole = new Float32Array(4 * kept.length);
		const ahead = new Float32Array(4 * kept.length);
		// One set of grids for the scale's four quantities: the finest scale's hold millions of nodes,
		// too many to make anew for every quantity and pass.
		const n = grid.nodes[0] * grid.nodes[1] * grid.nodes[2];
		const sums = new Float64Array(n);
		const [spread, blurred, yz] = [new Float32Array(n), new Float32Array(n), new Float32Array(n)];
		for (let q = 0; q < 4; q++) {
			sums.fill(0);
			spreadOnGrid(grid, cells.centers, carriedAtCentres(cells, pieces, q, atCentres), sums);
			spreadOnGrid(grid, pieces.positions, carriedByPieces(cells, pieces, q, byPieces), sums);
			spread.set(sums);
			correlateAlong(correlateAlong(spread, grid.nodes, 1, scale.weights, blurred), grid.nodes, 2, scale.weights, yz);
			gather(correlateAlong(yz, grid.nodes, 0, scale.weights, blurred), kept, whole, q);
			gather(correlateAlong(yz, grid.nodes, 0, scale.upstream, blurred), kept, ahead, q);
		}
		return { kept, whole, ahead };
	});
}

/**
 * The nodes a read within `KEPT_REACH` of any polygon centre or file point touches, ascending: the 27
 * around the node nearest each place in that reach, as `grid.js` `splineAround` picks it.
 */
function keptNodes({ origin, step, nodes }, ...pointSets) {
	const [, ny, nz] = nodes;
	const marked = new Uint8Array(nodes[0] * ny * nz);
	const low = [0, 0, 0],
		high = [0, 0, 0];
	const nearest = (q, axis) => Math.min(Math.max(Math.floor(q + 0.5), 1), nodes[axis] - 2);
	for (const points of pointSets) {
		for (let p = 0; p < points.length; p += 3) {
			for (let axis = 0; axis < 3; axis++) {
				const q = (points[p + axis] - origin[axis]) / step;
				const reach = KEPT_REACH / step;
				low[axis] = nearest(q - reach, axis) - 1;
				high[axis] = nearest(q + reach, axis) + 1;
			}
			for (let i = low[0]; i <= high[0]; i++) {
				for (let j = low[1]; j <= high[1]; j++) {
					for (let k = low[2]; k <= high[2]; k++) marked[(i * ny + j) * nz + k] = 1;
				}
			}
		}
	}
	let count = 0;
	for (let n = 0; n < marked.length; n++) count += marked[n];
	const kept = new Int32Array(count);
	for (let n = 0, i = 0; n < marked.length; n++) if (marked[n]) kept[i++] = n;
	return kept;
}

/** `field` at the kept nodes into lane `lane` of rows of four. */
function gather(field, kept, out, lane) {
	for (let i = 0; i < kept.length; i++) out[4 * i + lane] = field[kept[i]];
}

async function scalesOnGpu(grids, cells, pieces, points, layout, scaleGrids) {
	const { device } = grids;
	const { centers, areas } = cells;
	const count = areas.length;
	// The polygon centres, then the wide polygons' pieces.
	const sampleCount = count + pieces.areas.length;
	const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
	const created = [];
	const buffer = (bytes) => {
		const made = device.createBuffer({ size: Math.max(16, bytes), usage });
		created.push(made);
		return made;
	};
	const upload = (array) => {
		const made = buffer(array.byteLength);
		device.queue.writeBuffer(made, 0, array);
		return made;
	};
	const submit = (record) => {
		const encoder = device.createCommandEncoder();
		record(encoder);
		device.queue.submit([encoder.finish()]);
	};
	const download = async (source, bytes) => {
		const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
		submit((encoder) => encoder.copyBufferToBuffer(source, 0, staging, 0, bytes));
		await staging.mapAsync(GPUMapMode.READ);
		const copy = staging.getMappedRange().slice(0);
		staging.destroy();
		return copy;
	};

	const build = async () => {
		const positions = buffer(12 * sampleCount);
		device.queue.writeBuffer(positions, 0, centers);
		if (pieces.areas.length) device.queue.writeBuffer(positions, centers.byteLength, pieces.positions);
		const nodeCount = ({ nodes }) => nodes[0] * nodes[1] * nodes[2];

		// Which nodes each scale keeps.
		const filePoints = upload(points);
		const bits = scaleGrids.map((grid) => buffer(4 * Math.ceil(nodeCount(grid) / 32)));
		submit((encoder) =>
			scaleGrids.forEach((grid, s) => {
				grids.mark(encoder, grid, { positions, count, reach: KEPT_REACH, bits: bits[s] });
				grids.mark(encoder, grid, {
					positions: filePoints,
					count: points.length / 3,
					reach: KEPT_REACH,
					bits: bits[s],
				});
			}),
		);
		filePoints.destroy();
		const kept = await Promise.all(
			bits.map(async (set, s) => nodesOf(new Uint32Array(await download(set, set.size)), nodeCount(scaleGrids[s]))),
		);

		// Each scale, one carried quantity at a time, so only one quantity's grids are held at once.
		const quantity = new Float32Array(sampleCount);
		const weights = buffer(4 * sampleCount);
		const scales = [];
		for (let s = 0; s < scaleGrids.length; s++) {
			const grid = scaleGrids[s];
			const { radius } = layout.scales[s];
			const n = nodeCount(grid);
			const keptBuffer = upload(kept[s]);
			const [sums, spread, yz, blurred] = [
				buffer(8 * Math.min(n, SUMS_BLOCK_NODES)),
				buffer(4 * n),
				buffer(4 * n),
				buffer(4 * n),
			];
			const whole = buffer(16 * kept[s].length);
			const ahead = buffer(16 * kept[s].length);
			const gaussian = upload(Float32Array.from(layout.scales[s].weights));
			const upstreamHalf = upload(Float32Array.from(layout.scales[s].upstream));
			for (let q = 0; q < 4; q++) {
				carriedAtCentres(cells, pieces, q, quantity.subarray(0, count));
				carriedByPieces(cells, pieces, q, quantity.subarray(count));
				device.queue.writeBuffer(weights, 0, quantity);
				submit((encoder) => {
					for (let first = 0; first < n; first += SUMS_BLOCK_NODES) {
						const length = Math.min(SUMS_BLOCK_NODES, n - first);
						encoder.clearBuffer(sums);
						grids.spread(encoder, grid, { positions, weights, count: sampleCount, sums, first, length });
						grids.sumsToField(encoder, { sums, field: spread, first, length });
					}
					grids.correlate(encoder, grid.nodes, 1, { weights: gaussian, radius, field: spread, out: blurred });
					grids.correlate(encoder, grid.nodes, 2, { weights: gaussian, radius, field: blurred, out: yz });
					grids.correlate(encoder, grid.nodes, 0, { weights: gaussian, radius, field: yz, out: blurred });
					grids.gather(encoder, {
						field: blurred,
						nodes: keptBuffer,
						count: kept[s].length,
						out: whole,
						lane: q,
						lanes: 4,
					});
					grids.correlate(encoder, grid.nodes, 0, { weights: upstreamHalf, radius, field: yz, out: blurred });
					grids.gather(encoder, {
						field: blurred,
						nodes: keptBuffer,
						count: kept[s].length,
						out: ahead,
						lane: q,
						lanes: 4,
					});
				});
			}
			scales.push({
				kept: kept[s],
				whole: new Float32Array(await download(whole, whole.size)),
				ahead: new Float32Array(await download(ahead, ahead.size)),
			});
			for (const made of [keptBuffer, sums, spread, yz, blurred, whole, ahead, gaussian, upstreamHalf]) made.destroy();
		}
		return scales;
	};

	device.pushErrorScope('out-of-memory');
	device.pushErrorScope('validation');
	const built = await build()
		.then(
			(value) => ({ value }),
			(error) => ({ error }),
		)
		.finally(() => {
			for (const made of created) made.destroy();
		});
	const invalid = await device.popErrorScope();
	const outOfMemory = await device.popErrorScope();
	if (built.error) throw built.error;
	if (outOfMemory) throw new Error(`The GPU ran out of memory building the car's lookups: ${outOfMemory.message}`);
	if (invalid) throw new Error(`The GPU could not build the car's lookups: ${invalid.message}`);
	return built.value;
}

/** The set bits of a bitset of `count` nodes, as ascending node numbers. */
function nodesOf(words, count) {
	let total = 0;
	for (let w = 0; w < words.length; w++) {
		let word = words[w];
		while (word) {
			word &= word - 1;
			total++;
		}
	}
	const kept = new Int32Array(total);
	let i = 0;
	for (let n = 0; n < count; n++) if (words[n >>> 5] & (1 << (n & 31))) kept[i++] = n;
	return kept;
}
