// One car's meshes on the GPU: the working mesh the tools run on and, where the engine delivers it, the full mesh
// of the file. Both are indexed; fields are per-vertex attributes. Everything is released by `dispose`.

import * as THREE from 'three';

const FIELD_ARRAYS = ['cp', 'sigma', 'cfd'];

/** Arrays each field needs on a mesh before it can be drawn. */
const NEEDS = { cp: ['cp'], cfd: ['cfd'], error: ['cp', 'cfd'], sigma: ['sigma'], drag: ['cp'], none: [] };

class Layer {
	constructor(mesh) {
		this.mesh = mesh;
		this.triangles = mesh.indices.length / 3;
		this.bounds = boundsOf(mesh.positions);
		const geometry = new THREE.BufferGeometry();
		geometry.setAttribute('position', new THREE.BufferAttribute(mesh.positions, 3));
		geometry.setAttribute('normal', new THREE.BufferAttribute(mesh.normals, 3));
		geometry.setIndex(new THREE.BufferAttribute(mesh.indices, 1));
		// three sorts what it draws by bounding sphere and would otherwise walk every vertex twice to find one.
		const [x0, x1, y0, y1, z0, z1] = this.bounds;
		geometry.boundingBox = new THREE.Box3(new THREE.Vector3(x0, y0, z0), new THREE.Vector3(x1, y1, z1));
		geometry.boundingSphere = geometry.boundingBox.getBoundingSphere(new THREE.Sphere());
		this.geometry = geometry;
		this.edges = null;
		this.edgesPending = null;
		this.setFields(mesh);
	}

	/** Attach whichever of cp, sigma and cfd the source carries; an existing attribute is refilled in place. */
	setFields(source) {
		for (const name of FIELD_ARRAYS) {
			const data = source[name];
			if (!data) continue;
			this.mesh[name] = data;
			const existing = this.geometry.getAttribute(name);
			if (existing) {
				existing.array = data;
				existing.needsUpdate = true;
			} else {
				this.geometry.setAttribute(name, new THREE.BufferAttribute(data, 1));
			}
		}
	}

	has(field) {
		return NEEDS[field].every((name) => this.mesh[name]);
	}

	/** Edge lines, built once on first use without blocking the page for long; `onReady` runs when they exist. */
	edgeGeometry(onReady) {
		if (this.edges) return this.edges;
		this.edgesPending ??= buildEdges(this.mesh).then(({ index, edge }) => {
			if (this.disposed) return;
			const geometry = new THREE.BufferGeometry();
			geometry.setAttribute('position', this.geometry.getAttribute('position'));
			geometry.setAttribute('edge', new THREE.BufferAttribute(edge, 1));
			geometry.setIndex(new THREE.BufferAttribute(index, 1));
			this.edges = geometry;
			onReady();
		});
		return null;
	}

	dispose() {
		this.disposed = true;
		// The edge geometry shares the position buffer, so it goes first and the surface's own dispose frees the rest.
		this.edges?.dispose();
		this.geometry.dispose();
	}
}

export class Surface {
	constructor(working) {
		this.working = new Layer(working);
		this.full = null;
		this.fullFraction = 0;
	}

	/** A better working mesh, or the same mesh now carrying its fields. */
	setWorking(mesh) {
		if (mesh.positions === this.working.mesh.positions) {
			this.working.setFields(mesh);
			return;
		}
		if (this.working !== this.full) this.working.dispose();
		this.working = new Layer(mesh);
	}

	/** The file's every triangle. When it was also the first mesh drawn, both roles share one copy on the GPU. */
	setFull(mesh) {
		if (this.full && this.full !== this.working) this.full.dispose();
		this.full = mesh.positions === this.working.mesh.positions ? this.working : new Layer(mesh);
		this.fullFraction = 0;
	}

	setFullFields(fields, fraction) {
		this.full.setFields(fields);
		this.fullFraction = fraction;
	}

	dispose() {
		this.working.dispose();
		if (this.full !== this.working) this.full?.dispose();
	}
}

/** [xmin, xmax, ymin, ymax, zmin, zmax] of packed xyz positions. */
function boundsOf(positions) {
	const b = [Infinity, -Infinity, Infinity, -Infinity, Infinity, -Infinity];
	for (let i = 0; i < positions.length; i += 3) {
		for (let k = 0; k < 3; k++) {
			const v = positions[i + k];
			if (v < b[2 * k]) b[2 * k] = v;
			if (v > b[2 * k + 1]) b[2 * k + 1] = v;
		}
	}
	return b;
}

const yieldToPage = () => new Promise((resolve) => setTimeout(resolve));
const CHUNK = 1 << 21;

/**
 * Each undirected edge once: a closed surface wound consistently holds every edge as a→b in one triangle and b→a in
 * its neighbour, so keeping the direction with a < b needs no lookup table. Also the mean length of the edges at each
 * vertex, which the edge shader uses to fade lines that would crowd together.
 */
async function buildEdges({ positions, indices }) {
	const triangles = indices.length / 3;
	let count = 0;
	for (let t = 0; t < triangles; t++) {
		const a = indices[3 * t], b = indices[3 * t + 1], c = indices[3 * t + 2];
		count += (a < b) + (b < c) + (c < a);
		if (t % CHUNK === CHUNK - 1) await yieldToPage();
	}
	const index = new Uint32Array(2 * count);
	const vertices = positions.length / 3;
	const edge = new Float32Array(vertices);
	const meeting = new Uint16Array(vertices);
	let k = 0;
	const add = (a, b) => {
		index[k++] = a;
		index[k++] = b;
		const length = Math.hypot(positions[3 * a] - positions[3 * b], positions[3 * a + 1] - positions[3 * b + 1], positions[3 * a + 2] - positions[3 * b + 2]);
		edge[a] += length;
		edge[b] += length;
		meeting[a]++;
		meeting[b]++;
	};
	for (let t = 0; t < triangles; t++) {
		const a = indices[3 * t], b = indices[3 * t + 1], c = indices[3 * t + 2];
		if (a < b) add(a, b);
		if (b < c) add(b, c);
		if (c < a) add(c, a);
		if (t % CHUNK === CHUNK - 1) await yieldToPage();
	}
	for (let v = 0; v < vertices; v++) if (meeting[v]) edge[v] /= meeting[v];
	return { index, edge };
}
