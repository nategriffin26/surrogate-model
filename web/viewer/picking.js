// Where a picked surface point sits on the working mesh. The GPU says where the cursor meets the surface; the vertex
// grid says which vertex sits nearest there, and the face index which face the point lies on.

/** Edge of the vertex grid's cubes, in metres. */
const CELL = 0.02;

export class VertexGrid {
	constructor(positions) {
		const count = positions.length / 3;
		let size = 1;
		while (size < count) size <<= 1;
		this.mask = size - 1;
		this.positions = positions;
		const keys = new Uint32Array(count);
		const start = new Uint32Array(size + 1);
		for (let v = 0; v < count; v++) {
			const key = this.key(Math.floor(positions[3 * v] / CELL), Math.floor(positions[3 * v + 1] / CELL), Math.floor(positions[3 * v + 2] / CELL));
			keys[v] = key;
			start[key + 1]++;
		}
		for (let k = 0; k < size; k++) start[k + 1] += start[k];
		const fill = start.slice(0, size);
		const order = new Uint32Array(count);
		for (let v = 0; v < count; v++) order[fill[keys[v]]++] = v;
		this.start = start;
		this.order = order;
	}

	key(i, j, k) {
		return (Math.imul(i, 73856093) ^ Math.imul(j, 19349663) ^ Math.imul(k, 83492791)) & this.mask;
	}

	/** Index of the vertex nearest to p, searching outward shell by shell until no nearer one can exist. */
	nearest(p) {
		const P = this.positions;
		const ci = Math.floor(p[0] / CELL), cj = Math.floor(p[1] / CELL), ck = Math.floor(p[2] / CELL);
		let best = -1;
		let bestD = Infinity;
		for (let r = 0; r < 64; r++) {
			for (let i = ci - r; i <= ci + r; i++) {
				for (let j = cj - r; j <= cj + r; j++) {
					for (let k = ck - r; k <= ck + r; k++) {
						if (Math.max(Math.abs(i - ci), Math.abs(j - cj), Math.abs(k - ck)) !== r) continue;
						const key = this.key(i, j, k);
						for (let n = this.start[key]; n < this.start[key + 1]; n++) {
							const v = this.order[n];
							const d = (P[3 * v] - p[0]) ** 2 + (P[3 * v + 1] - p[1]) ** 2 + (P[3 * v + 2] - p[2]) ** 2;
							if (d < bestD) {
								bestD = d;
								best = v;
							}
						}
					}
				}
			}
			// Every point outside the shells searched so far is at least r cells away.
			if (best >= 0 && Math.sqrt(bestD) <= r * CELL) return best;
		}
		return best;
	}
}

/**
 * The faces around each vertex, to find the face a surface point lies on. The point's nearest vertex is a corner of
 * that face or of a face next to it, so the faces around it and around their corners are searched.
 */
export class FaceIndex {
	constructor({ positions, indices }) {
		const vertices = positions.length / 3;
		const start = new Uint32Array(vertices + 1);
		for (let i = 0; i < indices.length; i++) start[indices[i] + 1]++;
		for (let v = 0; v < vertices; v++) start[v + 1] += start[v];
		const fill = start.slice(0, vertices);
		const faces = new Uint32Array(indices.length);
		for (let i = 0; i < indices.length; i++) faces[fill[indices[i]]++] = (i / 3) | 0;
		this.positions = positions;
		this.indices = indices;
		this.start = start;
		this.faces = faces;
	}

	/** The face nearest to point p, whose nearest vertex is `vertex`. */
	faceAt(p, vertex) {
		const I = this.indices;
		let best = -1;
		let bestD = Infinity;
		for (let n = this.start[vertex]; n < this.start[vertex + 1]; n++) {
			const around = this.faces[n];
			for (let k = 0; k < 3; k++) {
				const corner = I[3 * around + k];
				for (let m = this.start[corner]; m < this.start[corner + 1]; m++) {
					const f = this.faces[m];
					const d = this.distance(p, f);
					if (d < bestD) {
						bestD = d;
						best = f;
					}
				}
			}
		}
		return best;
	}

	/** Squared distance from p to face f: to its plane when p projects inside it, else to its nearest edge. */
	distance(p, f) {
		const P = this.positions, I = this.indices;
		const a = 3 * I[3 * f], b = 3 * I[3 * f + 1], c = 3 * I[3 * f + 2];
		const e0 = [P[b] - P[a], P[b + 1] - P[a + 1], P[b + 2] - P[a + 2]];
		const e1 = [P[c] - P[a], P[c + 1] - P[a + 1], P[c + 2] - P[a + 2]];
		const q = [p[0] - P[a], p[1] - P[a + 1], p[2] - P[a + 2]];
		const d00 = dot(e0, e0), d01 = dot(e0, e1), d11 = dot(e1, e1), d20 = dot(q, e0), d21 = dot(q, e1);
		const denom = d00 * d11 - d01 * d01;
		const v = (d11 * d20 - d01 * d21) / denom;
		const w = (d00 * d21 - d01 * d20) / denom;
		if (v >= 0 && w >= 0 && v + w <= 1) {
			const r = [q[0] - v * e0[0] - w * e1[0], q[1] - v * e0[1] - w * e1[1], q[2] - v * e0[2] - w * e1[2]];
			return dot(r, r);
		}
		const corner = (i) => [P[i], P[i + 1], P[i + 2]];
		return Math.min(segment(p, corner(a), corner(b)), segment(p, corner(b), corner(c)), segment(p, corner(c), corner(a)));
	}
}

const dot = (u, v) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];

/** Squared distance from p to the segment from a to b. */
function segment(p, a, b) {
	const ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
	const ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
	const t = Math.min(1, Math.max(0, dot(ap, ab) / (dot(ab, ab) || 1)));
	const r = [ap[0] - t * ab[0], ap[1] - t * ab[1], ap[2] - t * ab[2]];
	return dot(r, r);
}
