// Nearest mesh vertex to a surface point. The GPU says where the cursor meets the surface; this grid says which
// vertex of the working mesh sits there, so every readout is a value the mesh actually holds.

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
