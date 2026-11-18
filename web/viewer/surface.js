// One car's meshes on the GPU: the working mesh the tools run on and, where the engine delivers it, the full mesh
// of the file. Both are indexed; fields are per-vertex attributes. Everything is released by `dispose`.
//
// Arrays reach the GPU in pieces, a few megabytes a frame (`upload`), into buffers three draws from as they are.
// Handing three a 17.7M-triangle mesh whole would copy 430 MB in one task: 90 ms of a frozen page, then frames
// held back while the GPU process takes the data in.

import * as THREE from 'three';

const FIELD_ARRAYS = ['cp', 'sigma', 'cfd'];

/** Arrays each field needs on a mesh before it can be drawn. */
const NEEDS = { cp: ['cp'], cfd: ['cfd'], error: ['cp', 'cfd'], sigma: ['sigma'], drag: ['cp'], none: [] };

/**
 * Bytes per bufferSubData call. A larger call can outgrow the browser's transfer buffer, which then waits on the GPU
 * process to make room; that process also runs the networks and may be busy for tens of milliseconds.
 */
const CALL_BYTES = 1 << 20;

class Layer {
	constructor(mesh) {
		this.mesh = mesh;
		this.triangles = mesh.indices.length / 3;
		this.bounds = mesh.bounds;
		this.disposed = false;
		this.gl = null;
		this.stageAll();
	}

	/** Queue every array for the GPU on a fresh geometry: when the layer is made, and again after the context is lost. */
	stageAll() {
		const mesh = this.mesh;
		const geometry = new THREE.BufferGeometry();
		// three culls and sorts by these, and cannot measure buffers it never saw as arrays.
		const [x0, x1, y0, y1, z0, z1] = this.bounds;
		geometry.boundingBox = new THREE.Box3(new THREE.Vector3(x0, y0, z0), new THREE.Vector3(x1, y1, z1));
		geometry.boundingSphere = geometry.boundingBox.getBoundingSphere(new THREE.Sphere());
		this.geometry = geometry;
		this.edges = null;
		this.edgesPending = null;
		/** Arrays on their way to the GPU, in order: {geometry, name, array, itemSize, sent, buffer}. */
		this.uploads = [];
		this.stage(geometry, 'index', mesh.indices, 1);
		this.stage(geometry, 'position', mesh.positions, 3);
		this.stage(geometry, 'normal', mesh.normals, 3);
		this.setFields(mesh);
	}

	/** Whether the mesh can be drawn: its triangles, positions and normals are on the GPU. Fields may still be on their way. */
	get ready() {
		const g = this.geometry;
		return Boolean(g.index && g.getAttribute('position') && g.getAttribute('normal'));
	}

	get uploading() {
		return this.uploads.length > 0;
	}

	/** Queue whichever of cp, sigma and cfd the source carries; a field already on the GPU is refilled in place. */
	setFields(source) {
		for (const name of FIELD_ARRAYS) {
			if (!source[name]) continue;
			this.mesh[name] = source[name];
			this.stage(this.geometry, name, source[name], 1);
		}
	}

	/** Whether every array the field needs is on the GPU; a field the viewer does not know is drawn from `custom`. */
	has(field) {
		return (NEEDS[field] ?? ['custom']).every((name) => this.geometry.getAttribute(name));
	}

	/** Queue an array for an attribute (or the index); a newer array replaces one still on its way and keeps its buffer. */
	stage(geometry, name, array, itemSize) {
		const attached = name === 'index' ? geometry.index : geometry.getAttribute(name);
		const queued = this.uploads.find((u) => u.geometry === geometry && u.name === name);
		this.uploads = this.uploads.filter((u) => u !== queued);
		this.uploads.push({ geometry, name, array, itemSize, sent: 0, buffer: attached?.buffer ?? queued?.buffer ?? null });
	}

	/**
	 * Hand the GPU the next pieces of the queued arrays, up to `budget` bytes. An array is drawn from once all of it is
	 * there; a field refilled in place shows its previous values until then. Returns the budget left and whether an
	 * array finished arriving, which can change the picture.
	 */
	upload(renderer, budget) {
		if (this.uploads.length === 0) return { left: budget, finished: false };
		const gl = renderer.getContext();
		this.gl = gl;
		// An index buffer binds to the bound vertex array object; none of three's may take this one on.
		gl.bindVertexArray(null);
		let finished = false;
		while (this.uploads.length > 0 && budget > 0) {
			const u = this.uploads[0];
			const target = u.name === 'index' ? gl.ELEMENT_ARRAY_BUFFER : gl.ARRAY_BUFFER;
			if (u.buffer) {
				gl.bindBuffer(target, u.buffer);
			} else {
				u.buffer = gl.createBuffer();
				gl.bindBuffer(target, u.buffer);
				gl.bufferData(target, u.array.byteLength, gl.STATIC_DRAW);
			}
			const bytes = u.array.BYTES_PER_ELEMENT;
			const count = Math.min(u.array.length - u.sent, Math.max(1, Math.floor(Math.min(budget, CALL_BYTES) / bytes)));
			gl.bufferSubData(target, u.sent * bytes, u.array, u.sent, count);
			u.sent += count;
			budget -= count * bytes;
			if (u.sent < u.array.length) continue;
			this.uploads.shift();
			this.attach(u, gl);
			finished = true;
		}
		// three caches the bindings it made; these calls changed them behind its back.
		renderer.resetState();
		return { left: budget, finished };
	}

	/** An array is all on the GPU: draw from it. A refilled one is already attached. */
	attach({ geometry, name, array, itemSize, buffer }, gl) {
		if (name === 'index') {
			if (!geometry.index) geometry.setIndex(new THREE.GLBufferAttribute(buffer, gl.UNSIGNED_INT, 1, 4, array.length));
		} else if (!geometry.getAttribute(name)) {
			geometry.setAttribute(name, new THREE.GLBufferAttribute(buffer, gl.FLOAT, itemSize, 4, array.length / itemSize));
		}
	}

	/** Edge lines, built once on first use without blocking the page for long; `onReady` runs once they are on the GPU. */
	edgeGeometry(onReady) {
		if (this.edges) return this.edges.index ? this.edges : null;
		const surface = this.geometry;
		this.edgesPending ??= buildEdges(this.mesh).then(({ index, edge }) => {
			if (this.disposed || this.geometry !== surface) return;
			const geometry = new THREE.BufferGeometry();
			geometry.setAttribute('position', surface.getAttribute('position'));
			geometry.boundingBox = surface.boundingBox;
			geometry.boundingSphere = surface.boundingSphere;
			this.stage(geometry, 'edge', edge, 1);
			this.stage(geometry, 'index', index, 1);
			this.edges = geometry;
			onReady();
		});
		return null;
	}

	dispose() {
		this.disposed = true;
		// Buffers still on their way are unknown to three; deleting one twice does nothing.
		for (const { buffer } of this.uploads) if (buffer) this.gl.deleteBuffer(buffer);
		this.uploads = [];
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

	get uploading() {
		return this.working.uploading || Boolean(this.full?.uploading);
	}

	/** Upload up to `budget` bytes, the working mesh first; returns whether an array finished arriving. */
	upload(renderer, budget) {
		const working = this.working.upload(renderer, budget);
		const full = this.full && this.full !== this.working ? this.full.upload(renderer, working.left) : { finished: false };
		return working.finished || full.finished;
	}

	/** The GPU context was lost and is back: every array goes to the GPU again. */
	restage() {
		this.working.stageAll();
		if (this.full && this.full !== this.working) this.full.stageAll();
	}

	dispose() {
		this.working.dispose();
		if (this.full !== this.working) this.full?.dispose();
	}
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
