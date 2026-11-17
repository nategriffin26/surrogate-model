// Section cuts through the working mesh: where a plane normal to x, y or z meets the surface, as polylines with the
// fields interpolated along the mesh edges they cross, then split into the two branches a plot draws.

const BUCKETS = 512;

/**
 * Triangles grouped by the slab of one axis they span, so a cut only visits the triangles near its plane.
 * Built once per axis, on the first cut along it.
 */
class SlabIndex {
	constructor(mesh, axis) {
		const { positions: P, indices: I } = mesh;
		const triangles = I.length / 3;
		let lo = Infinity, hi = -Infinity;
		for (let v = axis; v < P.length; v += 3) {
			lo = Math.min(lo, P[v]);
			hi = Math.max(hi, P[v]);
		}
		this.lo = lo;
		this.width = (hi - lo) / BUCKETS || 1;
		const first = new Uint16Array(triangles);
		const last = new Uint16Array(triangles);
		const sizes = new Uint32Array(BUCKETS + 1);
		for (let t = 0; t < triangles; t++) {
			const a = P[3 * I[3 * t] + axis], b = P[3 * I[3 * t + 1] + axis], c = P[3 * I[3 * t + 2] + axis];
			first[t] = this.bucket(Math.min(a, b, c));
			last[t] = this.bucket(Math.max(a, b, c));
			for (let k = first[t]; k <= last[t]; k++) sizes[k + 1]++;
		}
		for (let k = 0; k < BUCKETS; k++) sizes[k + 1] += sizes[k];
		const fill = sizes.slice(0, BUCKETS);
		this.start = sizes;
		this.triangles = new Uint32Array(sizes[BUCKETS]);
		for (let t = 0; t < triangles; t++) for (let k = first[t]; k <= last[t]; k++) this.triangles[fill[k]++] = t;
	}

	bucket(value) {
		return Math.min(BUCKETS - 1, Math.max(0, Math.floor((value - this.lo) / this.width)));
	}

	near(offset) {
		const k = this.bucket(offset);
		return this.triangles.subarray(this.start[k], this.start[k + 1]);
	}
}

const indexes = new WeakMap();

function slabIndex(mesh, axis) {
	let perAxis = indexes.get(mesh);
	if (!perAxis) indexes.set(mesh, (perAxis = []));
	perAxis[axis] ??= new SlabIndex(mesh, axis);
	return perAxis[axis];
}

/**
 * The surface cut by the plane coordinate[axis] = offset. Returns polylines, each an array of points
 * {p: [x, y, z], cp, sigma, cfd, nx}, and whether it closes on itself. A point sits on a mesh edge and its fields are
 * interpolated linearly along that edge, as the GPU does when it draws the triangle.
 */
export function cut(mesh, axis, offset) {
	const { positions: P, indices: I } = mesh;
	const vertexCount = P.length / 3;
	const crossings = new Map();
	const crossing = (u, v) => {
		const a = Math.min(u, v), b = Math.max(u, v);
		const key = a * vertexCount + b;
		let point = crossings.get(key);
		if (point) return key;
		const da = P[3 * a + axis] - offset;
		const db = P[3 * b + axis] - offset;
		const t = da / (da - db);
		const lerp = (field) => (field ? field[a] + (field[b] - field[a]) * t : null);
		point = {
			p: [0, 1, 2].map((k) => (k === axis ? offset : P[3 * a + k] + (P[3 * b + k] - P[3 * a + k]) * t)),
			cp: lerp(mesh.cp),
			sigma: lerp(mesh.sigma),
			cfd: lerp(mesh.cfd),
			nx: mesh.normals[3 * a] + (mesh.normals[3 * b] - mesh.normals[3 * a]) * t,
			links: [],
		};
		crossings.set(key, point);
		return key;
	};
	for (const t of slabIndex(mesh, axis).near(offset)) {
		const corners = [I[3 * t], I[3 * t + 1], I[3 * t + 2]];
		const above = corners.map((v) => P[3 * v + axis] >= offset);
		if (above[0] === above[1] && above[1] === above[2]) continue;
		const ends = [];
		for (let e = 0; e < 3; e++) {
			const u = corners[e], v = corners[(e + 1) % 3];
			if (above[e] !== above[(e + 1) % 3]) ends.push(crossing(u, v));
		}
		const [k0, k1] = ends;
		crossings.get(k0).links.push(k1);
		crossings.get(k1).links.push(k0);
	}
	return chain(crossings);
}

/** Join segments that share a crossing into polylines. */
function chain(crossings) {
	const visited = new Set();
	const lines = [];
	const walk = (start) => {
		const keys = [start];
		visited.add(start);
		let current = start;
		for (;;) {
			const next = crossings.get(current).links.find((k) => !visited.has(k));
			if (next === undefined) break;
			visited.add(next);
			keys.push(next);
			current = next;
		}
		return keys;
	};
	// Open chains first, from an end, so each is walked whole; then the closed loops.
	for (const [key, point] of crossings) {
		if (visited.has(key) || point.links.length !== 1) continue;
		lines.push({ points: walk(key).map((k) => crossings.get(k)), closed: false });
	}
	for (const [key] of crossings) {
		if (visited.has(key)) continue;
		lines.push({ points: walk(key).map((k) => crossings.get(k)), closed: true });
	}
	return lines;
}

/** The in-plane axes of a cut: values are plotted against `along`, and branches are told apart by `across`. */
export const CUT_AXES = [
	{ along: 1, across: 2, branches: ['Upper', 'Lower'] },
	{ along: 0, across: 2, branches: ['Upper', 'Lower'] },
	{ along: 0, across: 1, branches: ['y > 0', 'y < 0'] },
];

/**
 * Split each polyline at its two ends along the plot axis into the branch on the high side and the branch on the
 * low side of the other in-plane axis. Each branch runs in increasing `along`. An open chain goes whole to the side
 * its mean lies on.
 */
export function branches(lines, axis) {
	const { along, across } = CUT_AXES[axis];
	const high = [];
	const low = [];
	const mean = (points) => points.reduce((sum, q) => sum + q.p[across], 0) / points.length;
	const all = lines.flatMap((line) => line.points);
	const centre = all.length ? mean(all) : 0;
	for (const { points, closed } of lines) {
		if (points.length < 2) continue;
		if (!closed) {
			const run = points[0].p[along] <= points.at(-1).p[along] ? points : [...points].reverse();
			(mean(points) >= centre ? high : low).push(run);
			continue;
		}
		let iMin = 0, iMax = 0;
		points.forEach((q, i) => {
			if (q.p[along] < points[iMin].p[along]) iMin = i;
			if (q.p[along] > points[iMax].p[along]) iMax = i;
		});
		const n = points.length;
		const path = (from, to) => {
			const out = [];
			for (let i = from; ; i = (i + 1) % n) {
				out.push(points[i]);
				if (i === to) break;
			}
			return out;
		};
		const forward = path(iMin, iMax);
		const backward = path(iMax, iMin).reverse();
		const [top, bottom] = mean(forward) >= mean(backward) ? [forward, backward] : [backward, forward];
		high.push(top);
		low.push(bottom);
	}
	return [high, low];
}
