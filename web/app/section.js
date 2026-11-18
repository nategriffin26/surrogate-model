// Section cuts through the working mesh: where a plane normal to x, y or z meets the surface, as polylines with the
// fields interpolated along the mesh edges they cross, then split into the two branches a plot draws.

const BUCKETS = 512;

/**
 * Triangles grouped by the slab of one axis they span, so a cut only visits the triangles near its plane.
 * Built on the second cut along an axis: one cut alone is cheaper as a scan of every triangle.
 */
class SlabIndex {
	constructor(mesh, axis) {
		const { positions: P, indices: I } = mesh;
		const triangles = I.length / 3;
		let lo = Infinity, hi = -Infinity;
		for (let v = axis; v < P.length; v += 3) {
			if (P[v] < lo) lo = P[v];
			if (P[v] > hi) hi = P[v];
		}
		this.lo = lo;
		this.width = (hi - lo) / BUCKETS || 1;
		// The bucket of each triangle's lowest and highest corner, computed inline: this runs once per triangle.
		const scale = 1 / this.width;
		const first = new Uint16Array(triangles);
		const last = new Uint16Array(triangles);
		const sizes = new Uint32Array(BUCKETS + 1);
		for (let t = 0; t < triangles; t++) {
			const a = P[3 * I[3 * t] + axis], b = P[3 * I[3 * t + 1] + axis], c = P[3 * I[3 * t + 2] + axis];
			const min = a < b ? (a < c ? a : c) : b < c ? b : c;
			const max = a > b ? (a > c ? a : c) : b > c ? b : c;
			const f = Math.min(BUCKETS - 1, ((min - lo) * scale) | 0);
			const l = Math.min(BUCKETS - 1, ((max - lo) * scale) | 0);
			first[t] = f;
			last[t] = l;
			for (let k = f; k <= l; k++) sizes[k + 1]++;
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

/** Per mesh, per axis: undefined before the first cut, null after it, then the index. */
const indexes = new WeakMap();

/** The triangles a cut must visit, or null for every triangle. */
function nearby(mesh, axis, offset) {
	let perAxis = indexes.get(mesh);
	if (!perAxis) indexes.set(mesh, (perAxis = []));
	if (perAxis[axis] === undefined) {
		perAxis[axis] = null;
		return null;
	}
	perAxis[axis] ??= new SlabIndex(mesh, axis);
	return perAxis[axis].near(offset);
}

/**
 * The surface cut by the plane coordinate[axis] = offset. Returns polylines, each an array of points
 * {p: [x, y, z], cp, sigma, nx}, and whether it closes on itself. A point sits on a mesh edge and its fields are
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
		const lerp = (field) => field[a] + (field[b] - field[a]) * t;
		point = {
			p: [0, 1, 2].map((k) => (k === axis ? offset : P[3 * a + k] + (P[3 * b + k] - P[3 * a + k]) * t)),
			cp: lerp(mesh.cp),
			sigma: lerp(mesh.sigma),
			nx: mesh.normals[3 * a] + (mesh.normals[3 * b] - mesh.normals[3 * a]) * t,
			links: [],
		};
		crossings.set(key, point);
		return key;
	};
	const visit = (t) => {
		const a = I[3 * t], b = I[3 * t + 1], c = I[3 * t + 2];
		const aboveA = P[3 * a + axis] >= offset, aboveB = P[3 * b + axis] >= offset, aboveC = P[3 * c + axis] >= offset;
		if (aboveA === aboveB && aboveB === aboveC) return;
		// The plane crosses exactly two of the triangle's edges.
		const ends = [];
		if (aboveA !== aboveB) ends.push(crossing(a, b));
		if (aboveB !== aboveC) ends.push(crossing(b, c));
		if (aboveC !== aboveA) ends.push(crossing(c, a));
		crossings.get(ends[0]).links.push(ends[1]);
		crossings.get(ends[1]).links.push(ends[0]);
	};
	const near = nearby(mesh, axis, offset);
	if (near) for (const t of near) visit(t);
	else for (let t = 0; t < I.length / 3; t++) visit(t);
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
 * The outer surface of a cut, split at its two ends along the plot axis into the branch on the high side and the
 * branch on the low side of the other in-plane axis, each running in increasing `along`. The outer surface is the
 * closed loop enclosing the most area; interior loops (the underbody parts, the cabin) are left out of the plot.
 * When no loop closes, the longest chain goes whole to the side its mean lies on.
 */
export function branches(lines, axis) {
	const { along, across } = CUT_AXES[axis];
	const mean = (points) => points.reduce((sum, q) => sum + q.p[across], 0) / points.length;
	const outer = outerLoop(lines, along, across);
	if (!outer) return [[], []];
	const { points, closed } = outer;
	if (!closed) {
		const all = lines.flatMap((line) => line.points);
		const run = points[0].p[along] <= points.at(-1).p[along] ? points : [...points].reverse();
		return mean(points) >= mean(all) ? [[run], []] : [[], [run]];
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
	return [[top], [bottom]];
}

/** The closed loop enclosing the most area in the cut plane, or the longest chain when none closes. */
function outerLoop(lines, along, across) {
	let best = null;
	let bestArea = -1;
	for (const line of lines) {
		if (!line.closed || line.points.length < 3) continue;
		let twice = 0;
		const pts = line.points;
		for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
			twice += pts[j].p[along] * pts[i].p[across] - pts[i].p[along] * pts[j].p[across];
		}
		if (Math.abs(twice) > bestArea) {
			bestArea = Math.abs(twice);
			best = line;
		}
	}
	if (best) return best;
	return lines.filter((line) => line.points.length >= 2).sort((a, b) => b.points.length - a.points.length)[0] ?? null;
}
