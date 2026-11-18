/** Nearest-point queries over a point set, through a uniform grid of buckets, and sampling one mesh's fields at another's vertices. */

import { boundsOf } from './geometry.js';

/**
 * @param {Float32Array} positions (N, 3)
 * @returns {{ positions: Float32Array, origin: number[], cell: number, shape: number[], starts: Uint32Array, order: Uint32Array }}
 */
export function buildPointGrid(positions) {
	const count = positions.length / 3;
	const bounds = boundsOf(positions);
	const min = [bounds[0], bounds[2], bounds[4]];
	const max = [bounds[1], bounds[3], bounds[5]];
	// About one point per bucket of the bounding box; a surface fills few buckets, each with several.
	const volume = Math.max((max[0] - min[0]) * (max[1] - min[1]) * (max[2] - min[2]), 1e-12);
	const cell = Math.max(Math.cbrt(volume / Math.max(count, 1)), 1e-6);
	const shape = [0, 1, 2].map((k) => Math.floor((max[k] - min[k]) / cell) + 1);
	const bucketOf = (i) => {
		const bx = Math.floor((positions[3 * i] - min[0]) / cell);
		const by = Math.floor((positions[3 * i + 1] - min[1]) / cell);
		const bz = Math.floor((positions[3 * i + 2] - min[2]) / cell);
		return (bx * shape[1] + by) * shape[2] + bz;
	};
	const starts = new Uint32Array(shape[0] * shape[1] * shape[2] + 1);
	for (let i = 0; i < count; i++) starts[bucketOf(i) + 1]++;
	for (let b = 1; b < starts.length; b++) starts[b] += starts[b - 1];
	const fill = starts.slice(0, -1);
	const order = new Uint32Array(count);
	for (let i = 0; i < count; i++) order[fill[bucketOf(i)]++] = i;
	return { positions, origin: min, cell, shape, starts, order };
}

/**
 * Index of the grid's point nearest to (x, y, z), searching shells of buckets outward; only points
 * `accepts` admits count when it is given. -1 when it admits none.
 */
export function nearestPoint(grid, x, y, z, accepts) {
	const { positions, origin, cell, shape, starts, order } = grid;
	const cx = clamp(Math.floor((x - origin[0]) / cell), shape[0]);
	const cy = clamp(Math.floor((y - origin[1]) / cell), shape[1]);
	const cz = clamp(Math.floor((z - origin[2]) / cell), shape[2]);
	let best = -1;
	let bestDistance = Infinity;
	const largest = Math.max(...shape);
	for (let ring = 0; ring <= largest; ring++) {
		for (let bx = cx - ring; bx <= cx + ring; bx++) {
			if (bx < 0 || bx >= shape[0]) continue;
			for (let by = cy - ring; by <= cy + ring; by++) {
				if (by < 0 || by >= shape[1]) continue;
				const onShell = bx === cx - ring || bx === cx + ring || by === cy - ring || by === cy + ring;
				for (let bz = cz - ring; bz <= cz + ring; bz += onShell ? 1 : 2 * ring || 1) {
					if (bz < 0 || bz >= shape[2]) continue;
					const bucket = (bx * shape[1] + by) * shape[2] + bz;
					for (let s = starts[bucket]; s < starts[bucket + 1]; s++) {
						const i = order[s];
						if (accepts && !accepts(i)) continue;
						const dx = positions[3 * i] - x,
							dy = positions[3 * i + 1] - y,
							dz = positions[3 * i + 2] - z;
						const distance = dx * dx + dy * dy + dz * dz;
						if (distance < bestDistance) {
							bestDistance = distance;
							best = i;
						}
					}
				}
			}
		}
		// Every bucket beyond this shell is at least `ring` cells away from the query's bucket.
		if (best >= 0 && Math.sqrt(bestDistance) <= ring * cell) break;
	}
	return best;
}

/** Calls `visit(i)` for every grid point within `radius` of (x, y, z), and some a little farther. */
export function pointsWithin(grid, x, y, z, radius, visit) {
	const { origin, cell, shape, starts, order } = grid;
	const [x0, y0, z0] = [x, y, z].map((value, k) => clamp(Math.floor((value - radius - origin[k]) / cell), shape[k]));
	const [x1, y1, z1] = [x, y, z].map((value, k) => clamp(Math.floor((value + radius - origin[k]) / cell), shape[k]));
	for (let bx = x0; bx <= x1; bx++) {
		for (let by = y0; by <= y1; by++) {
			for (let bz = z0; bz <= z1; bz++) {
				const bucket = (bx * shape[1] + by) * shape[2] + bz;
				for (let s = starts[bucket]; s < starts[bucket + 1]; s++) visit(order[s]);
			}
		}
	}
}

/**
 * The fields of mesh `from` at the vertices of `at`, for comparing two cars: each `at` vertex takes
 * the values of the `from` vertex nearest it among those not facing away from its normal, so a point
 * on one side of a thin panel never takes the other side's pressure.
 * @param {{ positions: Float32Array, normals: Float32Array, cp?: Float32Array, sigma?: Float32Array, cfd?: Float32Array }} from
 *   per vertex, as a `DisplayMesh` with its fields
 * @param {{ positions: Float32Array, normals: Float32Array }} at
 * @returns {{ cp?: Float32Array, sigma?: Float32Array, cfd?: Float32Array }} one value per `at` vertex
 *   for each field `from` has; NaN where every `from` vertex faces away
 */
export function sampleFields(from, at) {
	const grid = buildPointGrid(from.positions);
	const names = ['cp', 'sigma', 'cfd'].filter((name) => from[name]);
	const count = at.positions.length / 3;
	const sampled = Object.fromEntries(names.map((name) => [name, new Float32Array(count)]));
	const { normals } = from;
	let nx = 0,
		ny = 0,
		nz = 0;
	const facing = (i) => normals[3 * i] * nx + normals[3 * i + 1] * ny + normals[3 * i + 2] * nz >= 0;
	for (let v = 0; v < count; v++) {
		nx = at.normals[3 * v];
		ny = at.normals[3 * v + 1];
		nz = at.normals[3 * v + 2];
		const i = nearestPoint(grid, at.positions[3 * v], at.positions[3 * v + 1], at.positions[3 * v + 2], facing);
		for (const name of names) sampled[name][v] = i < 0 ? NaN : from[name][i];
	}
	return sampled;
}

function clamp(index, size) {
	return index < 0 ? 0 : index >= size ? size - 1 : index;
}
