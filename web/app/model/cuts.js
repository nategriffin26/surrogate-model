// Where an axis-aligned plane cuts the car's surface: the outlines the plan-view outline reads and the surface
// drawings show, drawn on the car and flat. The triangles are bucketed once along y and z, so a cut visits only the
// triangles that reach its plane and keeps up with a point dragged across the car.

/** Cuts kept per car; a drag across the whole car visits fewer distinct planes than this. */
const KEPT = 96;

export class SurfaceCutter {
	/**
	 * @param {import('../../engine/result.js').DisplayMesh} mesh
	 * @param {number} step  bucket size in metres, the raster step of the plan-view outline
	 */
	constructor({ positions, normals, indices }, step) {
		this.positions = positions;
		this.normals = normals;
		this.indices = indices;
		this.step = step;
		this.buckets = { 1: this.bucket(1), 2: this.bucket(2) };
		this.kept = new Map();
	}

	/** Triangle ids by the bins of `step` their extent along `axis` touches, as offsets into one id array. */
	bucket(axis) {
		const { positions: P, indices: I, step } = this;
		const triangles = I.length / 3;
		let lo = Infinity;
		let hi = -Infinity;
		for (let v = axis; v < P.length; v += 3) {
			if (P[v] < lo) lo = P[v];
			if (P[v] > hi) hi = P[v];
		}
		const bins = Math.floor((hi - lo) / step) + 1;
		const first = new Int32Array(triangles);
		const last = new Int32Array(triangles);
		const start = new Uint32Array(bins + 1);
		for (let t = 0; t < triangles; t++) {
			const a = P[3 * I[3 * t] + axis], b = P[3 * I[3 * t + 1] + axis], c = P[3 * I[3 * t + 2] + axis];
			first[t] = Math.floor((Math.min(a, b, c) - lo) / step);
			last[t] = Math.floor((Math.max(a, b, c) - lo) / step);
			for (let k = first[t]; k <= last[t]; k++) start[k + 1]++;
		}
		for (let k = 0; k < bins; k++) start[k + 1] += start[k];
		const fill = start.slice(0, bins);
		const ids = new Uint32Array(start[bins]);
		for (let t = 0; t < triangles; t++) for (let k = first[t]; k <= last[t]; k++) ids[fill[k]++] = t;
		return { lo, bins, start, ids };
	}

	/**
	 * The surface where coordinate `axis` (1 for y, 2 for z) equals `value`, as segments: `positions` and the
	 * surface `normals` at their ends, six floats per segment each.
	 * @returns {{ positions: Float32Array, normals: Float32Array }}
	 */
	cut(axis, value) {
		const key = `${axis}:${value}`;
		const kept = this.kept.get(key);
		if (kept) return kept;
		const { positions: P, normals: N, indices: I } = this;
		const { lo, bins, start, ids } = this.buckets[axis];
		const bin = Math.floor((value - lo) / this.step);
		const points = [];
		const normals = [];
		if (bin >= 0 && bin < bins) {
			const d = [0, 0, 0];
			const v = [0, 0, 0];
			for (let n = start[bin]; n < start[bin + 1]; n++) {
				const t = ids[n];
				for (let k = 0; k < 3; k++) {
					v[k] = I[3 * t + k];
					d[k] = P[3 * v[k] + axis] - value;
				}
				// A vertex on the plane counts as above it, so each crossing edge is found once.
				const above = (d[0] >= 0) + (d[1] >= 0) + (d[2] >= 0);
				if (above === 0 || above === 3) continue;
				for (let k = 0; k < 3; k++) {
					const a = k, b = (k + 1) % 3;
					if ((d[a] >= 0) === (d[b] >= 0)) continue;
					const s = d[a] / (d[a] - d[b]);
					for (let c = 0; c < 3; c++) {
						points.push(P[3 * v[a] + c] + s * (P[3 * v[b] + c] - P[3 * v[a] + c]));
						normals.push(N[3 * v[a] + c] + s * (N[3 * v[b] + c] - N[3 * v[a] + c]));
					}
				}
			}
		}
		const cut = { positions: Float32Array.from(points), normals: Float32Array.from(normals) };
		this.kept.set(key, cut);
		if (this.kept.size > KEPT) this.kept.delete(this.kept.keys().next().value);
		return cut;
	}
}
