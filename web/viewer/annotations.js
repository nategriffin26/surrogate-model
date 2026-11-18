// Annotations on the car: marks anchored in 3D (points, lines, arrows, outlines, boxes, planes) drawn in the viewer's
// scene with the car, and text labels on the overlay over the view, the way probe markers are.
//
// Marks are sized in screen pixels, so a line keeps its width and a point its size at any distance, and each has a
// white halo so it reads over any colour of the car. Where the car hides part of a mark, that part is drawn faint
// rather than dropped. Every mark of every layer goes into one instanced draw per kind and pass, so a few thousand
// arrows cost about as much as one. Labels keep clear of each other where a free place is near, with a leader back to
// their point once moved, and fade where the car hides their point.

import * as THREE from 'three';
import { FaceIndex, VertexGrid } from './picking.js';

/**
 * The layers to draw, in order, later on top.
 * @typedef {Object} DrawnLayer
 * @property {string} colour  sRGB hex, for marks without colours of their own
 * @property {number} width   line width, CSS px
 * @property {Mark[]} marks
 *
 * A mark, points as flat (N, 3) arrays in metres; `colours`, where allowed, RGBA bytes per item.
 * @typedef {{ kind: 'lines', points: Float32Array, starts: Uint32Array, closed: boolean }
 *   | { kind: 'segments', a: Float32Array, b: Float32Array }
 *   | { kind: 'arrows', points: Float32Array, vectors: Float32Array, colours: Uint8Array | null }
 *   | { kind: 'points', points: Float32Array, size: number, colours: Uint8Array | null }
 *   | { kind: 'labels', points: Float32Array, texts: string[], offset: number[] | null }
 *   | { kind: 'box', min: number[], max: number[] }
 *   | { kind: 'plane', point: number[], normal: number[], size: number[] }} Mark
 *   `lines` holds polylines one after another, `starts` the index of each one's first point and then the count of
 *   points; `closed` joins each one's last point to its first. `segments` are separate lines a[i] to b[i].
 */

/** White beyond a mark's edge, CSS px, and its opacity. */
const HALO = 1.25;
/** How strongly the part of a mark behind the car shows. */
const BEHIND = 0.3;
/** Box faces and planes are washed in their colour this strongly. */
const FILL = 0.07;

/** The quad every instanced mark is drawn on: x 0 to 1 along it, y −1 to 1 across. */
function quad() {
	const geometry = new THREE.InstancedBufferGeometry();
	geometry.setAttribute('corner', new THREE.Float32BufferAttribute([0, -1, 1, -1, 0, 1, 1, 1], 2));
	geometry.setIndex([0, 1, 2, 2, 1, 3]);
	return geometry;
}

const COMMON = /* glsl */ `
uniform vec2 uViewport;      // the pane drawn, device px
uniform float uPixelRatio;
uniform float uOrtho;
uniform float uHalo;         // CSS px of white beyond the edge; 0 draws the mark itself
uniform float uOpacity;
const float FRONT = -0.001;  // eye-space z a point must lie below to be in front of the camera
vec3 linearRgb(vec3 c) { return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c)); }
/**
 * Eye position of a point moved toward the camera by a little more than the working mesh strays from the file's
 * surface, so a mark on the surface is in front of it.
 */
vec4 eyeOf(vec3 p) {
	vec4 eye = viewMatrix * vec4(p, 1.0);
	if (uOrtho > 0.5) eye.z += 0.006;
	else eye.xyz *= 1.0 - (0.004 + 0.001 * length(eye.xyz)) / length(eye.xyz);
	return eye;
}
vec2 screenOf(vec4 clip) { return clip.xy / clip.w * 0.5 * uViewport; }
vec4 placed(vec2 screen, float z) { return vec4(screen / (0.5 * uViewport), z, 1.0); }
/** An arrowhead's length, device px, for a line of this width and a shaft this long on screen. */
float headLength(float width, float shaft) { return min((3.5 * width + 4.0) * uPixelRatio, 0.5 * shaft); }
/** Coverage of a shape at signed distance d (device px) from its edge, widened by the halo in the halo pass. */
float cover(float d) { float edge = uHalo * uPixelRatio; return 1.0 - smoothstep(edge - 0.6, edge + 0.6, d); }
`;

const OUTPUT = /* glsl */ `
	if (a <= 0.003) discard;
	gl_FragColor = uHalo > 0.0 ? vec4(1.0, 1.0, 1.0, 0.85 * a * uOpacity) : vec4(vColour.rgb, vColour.a * a * uOpacity);
	#include <colorspace_fragment>
`;

const SEGMENT_VERTEX = /* glsl */ `${COMMON}
attribute vec2 corner;
attribute vec3 a;
attribute vec3 b;
attribute vec4 colour;
attribute float width;       // CSS px
attribute float head;        // 1 for an arrow's shaft, which stops under its head
varying vec2 vLocal;         // device px along from a and across
varying float vLength;
varying float vHalf;
varying vec4 vColour;
void main() {
	vec4 ea = eyeOf(a), eb = eyeOf(b);
	if (uOrtho < 0.5) {
		// The part behind the camera is cut off.
		if (ea.z > FRONT && eb.z > FRONT) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
		if (ea.z > FRONT) ea = mix(ea, eb, (ea.z - FRONT) / (ea.z - eb.z));
		if (eb.z > FRONT) eb = mix(eb, ea, (eb.z - FRONT) / (eb.z - ea.z));
	}
	vec4 ca = projectionMatrix * ea, cb = projectionMatrix * eb;
	vec2 sa = screenOf(ca), sb = screenOf(cb);
	float span = length(sb - sa);
	vec2 along = span > 1e-3 ? (sb - sa) / span : vec2(1.0, 0.0);
	vec2 across = vec2(-along.y, along.x);
	float len = max(span - (head > 0.5 ? 0.8 * headLength(width, span) : 0.0), 0.0);
	vHalf = 0.5 * width * uPixelRatio;
	float reach = vHalf + uHalo * uPixelRatio + 1.0;
	float s = mix(-reach, len + reach, corner.x);
	// Depth after projection is linear along a line on screen.
	float t = span > 1e-3 ? clamp(s / span, 0.0, 1.0) : 0.0;
	gl_Position = placed(sa + along * s + across * corner.y * reach, mix(ca.z / ca.w, cb.z / cb.w, t));
	vLocal = vec2(s, corner.y * reach);
	vLength = len;
	vColour = vec4(linearRgb(colour.rgb), colour.a);
}`;

const SEGMENT_FRAGMENT = /* glsl */ `${COMMON}
varying vec2 vLocal;
varying float vLength;
varying float vHalf;
varying vec4 vColour;
void main() {
	float beyond = max(max(-vLocal.x, vLocal.x - vLength), 0.0);
	float a = cover(length(vec2(beyond, vLocal.y)) - vHalf);
	${OUTPUT}
}`;

const HEAD_VERTEX = /* glsl */ `${COMMON}
attribute vec2 corner;
attribute vec3 tip;
attribute vec3 tail;
attribute vec4 colour;
attribute float width;
varying vec2 vLocal;         // device px back from the tip and across
varying vec2 vSize;          // the head's length and half its base
varying vec4 vColour;
void main() {
	vec4 et = eyeOf(tip), eq = eyeOf(tail);
	if (uOrtho < 0.5) {
		if (et.z > FRONT) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
		if (eq.z > FRONT) eq = mix(eq, et, (eq.z - FRONT) / (eq.z - et.z));
	}
	vec4 ct = projectionMatrix * et;
	vec2 st = screenOf(ct), sq = screenOf(projectionMatrix * eq);
	float span = length(st - sq);
	vec2 along = span > 1e-3 ? (st - sq) / span : vec2(1.0, 0.0);
	float length = headLength(width, span);
	vSize = vec2(length, 0.42 * length);
	float margin = uHalo * uPixelRatio + 1.0;
	vLocal = vec2(mix(-margin, length + margin, corner.x), corner.y * (vSize.y + margin));
	gl_Position = placed(st - along * vLocal.x + vec2(-along.y, along.x) * vLocal.y, ct.z / ct.w);
	vColour = vec4(linearRgb(colour.rgb), colour.a);
}`;

const HEAD_FRAGMENT = /* glsl */ `${COMMON}
varying vec2 vLocal;
varying vec2 vSize;
varying vec4 vColour;
void main() {
	if (vSize.x < 0.5) discard;
	float k = vSize.y / vSize.x;
	float side = (abs(vLocal.y) - vLocal.x * k) / sqrt(1.0 + k * k);
	float a = cover(max(vLocal.x - vSize.x, side));
	${OUTPUT}
}`;

const DOT_VERTEX = /* glsl */ `${COMMON}
attribute vec2 corner;
attribute vec3 centre;
attribute vec4 colour;
attribute float radius;      // CSS px
varying vec2 vLocal;
varying float vRadius;
varying vec4 vColour;
void main() {
	vec4 e = eyeOf(centre);
	if (uOrtho < 0.5 && e.z > FRONT) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
	vec4 c = projectionMatrix * e;
	vRadius = radius * uPixelRatio;
	vLocal = vec2(2.0 * corner.x - 1.0, corner.y) * (vRadius + uHalo * uPixelRatio + 1.0);
	gl_Position = placed(screenOf(c) + vLocal, c.z / c.w);
	vColour = vec4(linearRgb(colour.rgb), colour.a);
}`;

const DOT_FRAGMENT = /* glsl */ `${COMMON}
varying vec2 vLocal;
varying float vRadius;
varying vec4 vColour;
void main() {
	float a = cover(length(vLocal) - vRadius);
	${OUTPUT}
}`;

const FILL_VERTEX = /* glsl */ `${COMMON}
attribute vec4 colour;
varying vec4 vColour;
void main() {
	gl_Position = projectionMatrix * eyeOf(position);
	vColour = vec4(linearRgb(colour.rgb), colour.a);
}`;

const FILL_FRAGMENT = /* glsl */ `${COMMON}
varying vec4 vColour;
void main() {
	float a = 1.0;
	${OUTPUT}
}`;

/** Growable typed arrays for one kind of instance, filled as the layers are read. */
class Batch {
	constructor(layout) {
		this.layout = layout;
		this.count = 0;
		this.arrays = Object.fromEntries(Object.entries(layout).map(([name, { size, type }]) => [name, new type(size * 256)]));
	}

	/** Room for `n` more items; returns the index of the first. */
	reserve(n) {
		const first = this.count;
		this.count += n;
		for (const [name, { size, type }] of Object.entries(this.layout)) {
			const array = this.arrays[name];
			if (array.length >= this.count * size) continue;
			const grown = new type(Math.max(this.count * size, 2 * array.length));
			grown.set(array);
			this.arrays[name] = grown;
		}
		return first;
	}

	/** The filled part of each array. */
	filled(name) {
		return this.arrays[name].subarray(0, this.count * this.layout[name].size);
	}
}

const POINT = { size: 3, type: Float32Array };
const COLOUR = { size: 4, type: Uint8Array };
const NUMBER = { size: 1, type: Float32Array };

/** sRGB bytes of a '#rrggbb' colour. */
export function rgbOf(hex) {
	const n = parseInt(hex.slice(1), 16);
	return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export class Annotations {
	/** viewer: the 3D view; host: the overlay over it, where labels go. */
	constructor(viewer, host) {
		this.viewer = viewer;
		this.group = new THREE.Group();
		this.group.visible = false;
		this.uniforms = {
			uViewport: { value: new THREE.Vector2(1, 1) },
			uPixelRatio: { value: 1 },
			uOrtho: viewer.uniforms.uOrtho,
		};
		this.parts = {
			segments: this.part(true, SEGMENT_VERTEX, SEGMENT_FRAGMENT, { a: POINT, b: POINT, colour: COLOUR, width: NUMBER, head: NUMBER }),
			heads: this.part(true, HEAD_VERTEX, HEAD_FRAGMENT, { tip: POINT, tail: POINT, colour: COLOUR, width: NUMBER }),
			dots: this.part(true, DOT_VERTEX, DOT_FRAGMENT, { centre: POINT, colour: COLOUR, radius: NUMBER }),
			fills: this.part(false, FILL_VERTEX, FILL_FRAGMENT, { position: POINT, colour: COLOUR }),
		};
		// Faint where the car hides them, under what it does not; halos under every mark's own colour.
		const order = [['fills', 'behind', false], ['segments', 'behind', false], ['heads', 'behind', false], ['dots', 'behind', false],
			['fills', 'front', false], ['segments', 'front', true], ['heads', 'front', true], ['dots', 'front', true],
			['segments', 'front', false], ['heads', 'front', false], ['dots', 'front', false]];
		order.forEach(([name, side, halo], i) => this.group.add(this.mesh(this.parts[name], side, halo, 10 + i)));
		this.labels = new Labels(viewer, host);
	}

	/** One kind of mark: instanced on a quad, or plain triangles. */
	part(instanced, vertexShader, fragmentShader, layout) {
		return { instanced, geometry: instanced ? quad() : new THREE.BufferGeometry(), vertexShader, fragmentShader, layout, meshes: [] };
	}

	/** One pass of one kind: what the car hides drawn faint, or what it does not with or without its halo. */
	mesh(part, side, halo, renderOrder) {
		const material = new THREE.ShaderMaterial({
			uniforms: { ...this.uniforms, uHalo: { value: halo ? HALO : 0 }, uOpacity: { value: side === 'behind' ? BEHIND : 1 } },
			vertexShader: part.vertexShader,
			fragmentShader: part.fragmentShader,
			transparent: true,
			depthWrite: false,
			depthFunc: side === 'behind' ? THREE.GreaterDepth : THREE.LessEqualDepth,
			side: THREE.DoubleSide,
		});
		const mesh = new THREE.Mesh(part.geometry, material);
		part.meshes.push(mesh);
		mesh.frustumCulled = false;
		mesh.renderOrder = renderOrder;
		// The pane being drawn sets how pixels map to the screen.
		mesh.onBeforeRender = (renderer) => {
			renderer.getCurrentViewport(viewport);
			this.uniforms.uViewport.value.set(viewport.z, viewport.w);
			this.uniforms.uPixelRatio.value = renderer.getPixelRatio();
			material.uniformsNeedUpdate = true;
		};
		return mesh;
	}

	/**
	 * Draw these layers (DrawnLayer[]), on the case whose working mesh is `mesh` (its labels fade where it hides them).
	 * Nothing is drawn, and nothing costs a frame, when there are none.
	 */
	show(layers, mesh) {
		const batches = Object.fromEntries(Object.entries(this.parts).map(([name, part]) => [name, new Batch(part.layout)]));
		const labels = [];
		for (const layer of layers) {
			const colour = [...rgbOf(layer.colour), 255];
			for (const mark of layer.marks) {
				if (mark.kind === 'labels') labels.push({ mark, colour: layer.colour });
				else addMark(batches, mark, colour, layer.width);
			}
		}
		for (const [name, part] of Object.entries(this.parts)) {
			const geometry = geometryOf(part, batches[name]);
			for (const mesh of part.meshes) Object.assign(mesh, { geometry, visible: batches[name].count > 0 });
			part.geometry.dispose();
			part.geometry = geometry;
		}
		const any = Object.values(batches).some((batch) => batch.count > 0);
		this.group.visible = any;
		this.viewer.setAnnotations(any ? this.group : null);
		this.labels.show(labels, mesh);
	}

	/** Paint the labels as the view shows them into a 2D canvas, `scale` canvas px a CSS px (the Picture export). */
	paintLabels(ctx, scale) {
		this.labels.paint(ctx, scale);
	}
}

const viewport = new THREE.Vector4();

/** A kind's marks as a new geometry: its instances' attributes, or its triangles. */
function geometryOf(part, batch) {
	const geometry = part.instanced ? quad() : new THREE.BufferGeometry();
	for (const [name, { size }] of Object.entries(part.layout)) {
		const array = batch.filled(name).slice();
		const normalized = array instanceof Uint8Array;
		geometry.setAttribute(name, part.instanced ? new THREE.InstancedBufferAttribute(array, size, normalized) : new THREE.BufferAttribute(array, size, normalized));
	}
	if (part.instanced) geometry.instanceCount = batch.count;
	return geometry;
}

/** One mark's instances, in the layer's colour and line width unless it carries colours of its own. */
function addMark(batches, mark, colour, width) {
	const segment = (a, b, rgba, head = 0) => {
		const s = batches.segments;
		const i = s.reserve(1);
		s.arrays.a.set(a, 3 * i);
		s.arrays.b.set(b, 3 * i);
		s.arrays.colour.set(rgba, 4 * i);
		s.arrays.width[i] = width;
		s.arrays.head[i] = head;
	};
	const triangle = (p, q, r, rgba) => {
		const f = batches.fills;
		const i = f.reserve(3);
		[p, q, r].forEach((point, k) => {
			f.arrays.position.set(point, 3 * (i + k));
			f.arrays.colour.set(rgba, 4 * (i + k));
		});
	};
	const point = (array, i) => array.subarray(3 * i, 3 * i + 3);
	const own = (i) => (mark.colours ? mark.colours.subarray(4 * i, 4 * i + 4) : colour);
	const wash = [colour[0], colour[1], colour[2], Math.round(255 * FILL)];
	switch (mark.kind) {
		case 'lines': {
			const { points, starts } = mark;
			for (let line = 0; line + 1 < starts.length; line++) {
				const first = starts[line], end = starts[line + 1];
				for (let i = first; i + 1 < end; i++) segment(point(points, i), point(points, i + 1), colour);
				if (mark.closed && end - first > 2) segment(point(points, end - 1), point(points, first), colour);
			}
			return;
		}
		case 'segments':
			for (let i = 0; i < mark.a.length / 3; i++) segment(point(mark.a, i), point(mark.b, i), colour);
			return;
		case 'arrows': {
			const h = batches.heads;
			const count = mark.points.length / 3;
			const first = h.reserve(count);
			for (let i = 0; i < count; i++) {
				const from = point(mark.points, i);
				const to = [0, 1, 2].map((k) => from[k] + mark.vectors[3 * i + k]);
				segment(from, to, own(i), 1);
				h.arrays.tip.set(to, 3 * (first + i));
				h.arrays.tail.set(from, 3 * (first + i));
				h.arrays.colour.set(own(i), 4 * (first + i));
				h.arrays.width[first + i] = width;
			}
			return;
		}
		case 'points': {
			const d = batches.dots;
			const count = mark.points.length / 3;
			const first = d.reserve(count);
			d.arrays.centre.set(mark.points, 3 * first);
			for (let i = 0; i < count; i++) {
				d.arrays.colour.set(own(i), 4 * (first + i));
				d.arrays.radius[first + i] = mark.size / 2;
			}
			return;
		}
		case 'box': {
			const [x0, y0, z0] = mark.min, [x1, y1, z1] = mark.max;
			const c = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]];
			for (const [i, j] of [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]) segment(c[i], c[j], colour);
			for (const [p, q, r, s] of [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [3, 2, 6, 7], [0, 3, 7, 4], [1, 2, 6, 5]]) {
				triangle(c[p], c[q], c[r], wash);
				triangle(c[p], c[r], c[s], wash);
			}
			return;
		}
		case 'plane': {
			const corners = planeCorners(mark);
			for (let k = 0; k < 4; k++) segment(corners[k], corners[(k + 1) % 4], colour);
			triangle(corners[0], corners[1], corners[2], wash);
			triangle(corners[0], corners[2], corners[3], wash);
			return;
		}
		default:
			throw new Error(`Unknown mark ${mark.kind}`);
	}
}

/** A plane's rectangle: `size` [u, v] metres about `point`, its sides along the frame's axes where they lie in it. */
function planeCorners({ point, normal, size }) {
	const n = new THREE.Vector3(...normal).normalize();
	// The first side runs along the axis the normal points least along, made square to the normal.
	const axis = [Math.abs(n.x), Math.abs(n.y), Math.abs(n.z)];
	const least = axis.indexOf(Math.min(...axis));
	const u = new THREE.Vector3().setComponent(least, 1).addScaledVector(n, -n.getComponent(least)).normalize();
	const v = new THREE.Vector3().crossVectors(n, u);
	const p = new THREE.Vector3(...point);
	return [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([i, j]) => p.clone().addScaledVector(u, (i * size[0]) / 2).addScaledVector(v, (j * size[1]) / 2).toArray());
}

// ---- labels -------------------------------------------------------------------------------------------------------------

/** Space between a label's point and its box when it sits beside it, and the leader lengths tried farther out, CSS px. */
const GAP = 9;
const RINGS = [34, 64, 104];
/** Labels keep this far from each other and from the view's edges. */
const CLEAR = 3;
const BUCKET = 96;

/** Edge of the cubes the car is held in for finding whether it hides a label's point, metres. */
const VOXEL = 0.02;

/**
 * Which cubes of space the car's surface passes through, to find whether the car hides a point: the line from the
 * point to the camera is walked from clear of the surface the point sits on until it leaves the car's box.
 */
class Occupancy {
	constructor(mesh) {
		const { positions, indices, bounds } = mesh;
		this.mesh = mesh;
		this.grid = new VertexGrid(positions);
		this.origin = [bounds[0] - VOXEL, bounds[2] - VOXEL, bounds[4] - VOXEL];
		this.size = [0, 1, 2].map((k) => Math.ceil((bounds[2 * k + 1] - this.origin[k]) / VOXEL) + 2);
		const [nx, ny] = this.size;
		const cells = new Uint8Array(nx * ny * this.size[2]);
		const P = positions;
		const mark = (x, y, z) => {
			const i = Math.floor((x - this.origin[0]) / VOXEL), j = Math.floor((y - this.origin[1]) / VOXEL), k = Math.floor((z - this.origin[2]) / VOXEL);
			cells[i + nx * (j + ny * k)] = 1;
		};
		for (let t = 0; t < indices.length; t += 3) {
			const a = 3 * indices[t], b = 3 * indices[t + 1], c = 3 * indices[t + 2];
			const edge = Math.max(
				Math.hypot(P[b] - P[a], P[b + 1] - P[a + 1], P[b + 2] - P[a + 2]),
				Math.hypot(P[c] - P[a], P[c + 1] - P[a + 1], P[c + 2] - P[a + 2]),
				Math.hypot(P[c] - P[b], P[c + 1] - P[b + 1], P[c + 2] - P[b + 2]),
			);
			// Points on the triangle closer together than half a cube touch every cube it passes through.
			const n = Math.max(1, Math.ceil(edge / (0.5 * VOXEL)));
			for (let i = 0; i <= n; i++) {
				for (let j = 0; i + j <= n; j++) {
					const u = i / n, v = j / n, w = 1 - u - v;
					mark(w * P[a] + u * P[b] + v * P[c], w * P[a + 1] + u * P[b + 1] + v * P[c + 1], w * P[a + 2] + u * P[b + 2] + v * P[c + 2]);
				}
			}
		}
		this.cells = cells;
	}

	/** The outward normal of the surface at a point on it (within a cube of its face), or null for a point off it. */
	normalAt(point) {
		this.faces ??= new FaceIndex(this.mesh);
		const f = this.faces.faceAt(point, this.grid.nearest(point));
		if (f < 0 || Math.sqrt(this.faces.distance(point, f)) > VOXEL) return null;
		const { positions: P, indices: I, normals: N } = this.mesh;
		const [a, b, c] = [3 * I[3 * f], 3 * I[3 * f + 1], 3 * I[3 * f + 2]];
		const u = [P[b] - P[a], P[b + 1] - P[a + 1], P[b + 2] - P[a + 2]];
		const v = [P[c] - P[a], P[c + 1] - P[a + 1], P[c + 2] - P[a + 2]];
		const n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
		// Turned the way the vertex normals point: out of the car.
		const out = (N[a] + N[b] + N[c]) * n[0] + (N[a + 1] + N[b + 1] + N[c + 1]) * n[1] + (N[a + 2] + N[b + 2] + N[c + 2]) * n[2];
		const length = Math.hypot(...n) * (out < 0 ? -1 : 1);
		return length ? n.map((x) => x / length) : null;
	}

	/**
	 * Whether the surface lies between `point` and the eye, along `towards` (unit), outside what `clipped` cuts away.
	 * A point on a face turned away from the eye is behind the car; on one turned toward it, the walk starts where the
	 * line is farther from the face than a cube's diagonal, so the face's own cubes are not taken for another surface.
	 */
	hides(point, towards, distance, clipped, normal) {
		if (clipped(point)) return false;
		let start = 2.5 * VOXEL;
		if (normal) {
			const facing = normal[0] * towards[0] + normal[1] * towards[1] + normal[2] * towards[2];
			if (facing <= 0) return true;
			start = Math.min(Math.max(start, (Math.sqrt(3) * VOXEL) / facing), 0.3);
		}
		const [nx, ny, nz] = this.size;
		const step = 0.5 * VOXEL;
		const p = [0, 0, 0];
		for (let s = start; s < distance; s += step) {
			for (let k = 0; k < 3; k++) p[k] = point[k] + towards[k] * s;
			const i = Math.floor((p[0] - this.origin[0]) / VOXEL), j = Math.floor((p[1] - this.origin[1]) / VOXEL), k = Math.floor((p[2] - this.origin[2]) / VOXEL);
			if (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz) {
				// Out of the box for good once it is walking away from it.
				if ((i < 0 && towards[0] <= 0) || (i >= nx && towards[0] >= 0) || (j < 0 && towards[1] <= 0) || (j >= ny && towards[1] >= 0) || (k < 0 && towards[2] <= 0) || (k >= nz && towards[2] >= 0)) return false;
				continue;
			}
			if (this.cells[i + nx * (j + ny * k)] && !clipped(p)) return true;
		}
		return false;
	}
}

const occupancies = new WeakMap();

/**
 * Labels on the overlay: a dot at each point, the text in a small box beside it, and a leader when the box had to
 * move away to keep clear of another. Placed again whenever the view is drawn.
 */
class Labels {
	constructor(viewer, host) {
		this.viewer = viewer;
		this.leaders = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
		this.leaders.setAttribute('class', 'annotation-leaders');
		this.el = document.createElement('div');
		this.el.className = 'annotation-labels';
		this.el.append(this.leaders);
		host.append(this.el);
		/** {text, point, colour, offset, dot, box, w, h, choice, at: {x, y, behind, shown}} per label. */
		this.items = [];
		this.mesh = null;
		viewer.addEventListener('camera', () => this.place());
	}

	/** labels: [{mark, colour}] in drawing order; mesh: the working mesh whose surface hides them. */
	show(labels, mesh) {
		const items = labels.flatMap(({ mark, colour }) => mark.texts.map((text, i) => ({ text, colour, offset: mark.offset, point: Array.from(mark.points.subarray(3 * i, 3 * i + 3)) })));
		const same = items.length === this.items.length && items.every((item, i) => {
			const was = this.items[i];
			return item.text === was.text && item.colour === was.colour && item.offset === was.offset && item.point.every((v, k) => v === was.point[k]);
		});
		this.mesh = mesh;
		if (same) return this.place();
		for (const item of this.items) item.dot.remove(), item.box.remove();
		for (const item of items) {
			item.dot = document.createElement('div');
			item.dot.className = 'annotation-dot';
			item.dot.style.background = item.colour;
			item.box = document.createElement('div');
			item.box.className = 'annotation-label';
			item.box.style.color = item.colour;
			item.box.textContent = item.text;
			item.choice = 0;
			this.el.append(item.dot, item.box);
		}
		// Measured together, after every box is in, so the page lays out once.
		for (const item of items) [item.w, item.h] = [item.box.offsetWidth, item.box.offsetHeight];
		this.items = items;
		this.place();
	}

	/** Where each label goes for the camera as it is now. */
	place() {
		const viewer = this.viewer;
		const W = viewer.host.clientWidth, H = viewer.host.clientHeight;
		const occupancy = this.items.length && this.mesh ? occupancyOf(this.mesh) : null;
		const camera = viewer.camera.position.toArray();
		const look = new THREE.Vector3();
		viewer.camera.getWorldDirection(look);
		const clipped = (p) => viewer.clipped(p);
		for (const item of this.items) {
			const p = viewer.project(item.point);
			const shown = p.visible && p.x > -40 && p.y > -40 && p.x < W + 40 && p.y < H + 40;
			let behind = false;
			if (shown && occupancy) {
				const towards = viewer.camera.isOrthographicCamera ? look.toArray().map((v) => -v) : camera.map((v, k) => v - item.point[k]);
				const distance = Math.hypot(...towards);
				if (item.normal === undefined) item.normal = occupancy.normalAt(item.point);
				behind = occupancy.hides(item.point, towards.map((v) => v / distance), viewer.camera.isOrthographicCamera ? Infinity : distance, clipped, item.normal);
			}
			item.at = { x: Math.round(p.x), y: Math.round(p.y), behind, shown };
		}
		this.arrange(W, H);
		const paths = new Map();
		for (const item of this.items) {
			const { at, rect } = item;
			item.dot.hidden = !at.shown;
			item.box.hidden = !at.shown || !rect;
			if (!at.shown) continue;
			item.dot.classList.toggle('behind', at.behind);
			item.dot.style.transform = `translate(${at.x}px, ${at.y}px)`;
			if (!rect) continue;
			item.box.classList.toggle('behind', at.behind);
			item.box.style.transform = `translate(${rect.x}px, ${rect.y}px)`;
			if (!rect.leader) continue;
			const [x, y] = nearestOnRect(rect, at);
			const key = `${item.colour}${at.behind ? ' behind' : ''}`;
			paths.set(key, `${paths.get(key) ?? ''}M${at.x} ${at.y}L${x} ${y}`);
		}
		this.leaders.replaceChildren(...[...paths].flatMap(([key, d]) => {
			const [colour, behind] = key.split(' ');
			const halo = document.createElementNS('http://www.w3.org/2000/svg', 'path');
			halo.setAttribute('class', `halo${behind ? ' behind' : ''}`);
			halo.setAttribute('d', d);
			const line = halo.cloneNode();
			line.setAttribute('class', `line${behind ? ' behind' : ''}`);
			line.style.stroke = colour;
			return [halo, line];
		}));
	}

	/**
	 * Boxes beside their points, each in the first place free of those placed before it: the place it had last (so
	 * labels hold still while the camera moves), its own offset, then right, left, above and below its point, then
	 * farther out on a leader. Labels in clear view are placed first, in the order given, then those the car hides. A
	 * label with no free place near shows its dot alone until the view leaves it room.
	 */
	arrange(W, H) {
		const taken = new Map();
		const cellsOf = (r) => {
			const keys = [];
			for (let x = Math.floor((r.x - CLEAR) / BUCKET); x <= Math.floor((r.x + r.w + CLEAR) / BUCKET); x++) {
				for (let y = Math.floor((r.y - CLEAR) / BUCKET); y <= Math.floor((r.y + r.h + CLEAR) / BUCKET); y++) keys.push(`${x},${y}`);
			}
			return keys;
		};
		const free = (r) => r.x >= CLEAR && r.y >= CLEAR && r.x + r.w <= W - CLEAR && r.y + r.h <= H - CLEAR
			&& cellsOf(r).every((key) => (taken.get(key) ?? []).every((o) => r.x + r.w + CLEAR <= o.x || o.x + o.w + CLEAR <= r.x || r.y + r.h + CLEAR <= o.y || o.y + o.h + CLEAR <= r.y));
		const take = (r) => cellsOf(r).forEach((key) => (taken.get(key) ?? taken.set(key, []).get(key)).push(r));
		const shown = this.items.filter((item) => item.at.shown);
		// Every point is kept clear first, so no box covers another label's point.
		for (const { at } of shown) take({ x: at.x - 4, y: at.y - 4, w: 8, h: 8 });
		for (const item of [...shown.filter((i) => !i.at.behind), ...shown.filter((i) => i.at.behind)]) {
			const options = candidates(item);
			const pick = [item.choice, ...options.keys()].find((i) => i < options.length && free(options[i]));
			if (pick !== undefined) item.choice = pick;
			item.rect = pick === undefined ? null : options[pick];
			if (item.rect) take(item.rect);
		}
	}

	paint(ctx, scale) {
		const font = `500 ${11.5 * scale}px "Hanken Grotesk", system-ui, sans-serif`;
		for (const item of this.items) {
			const { at, rect } = item;
			if (!at?.shown) continue;
			ctx.save();
			ctx.globalAlpha = at.behind ? 0.35 : 1;
			if (rect?.leader) {
				const [x, y] = nearestOnRect(rect, at);
				ctx.lineCap = 'round';
				ctx.beginPath();
				ctx.moveTo(at.x * scale, at.y * scale);
				ctx.lineTo(x * scale, y * scale);
				ctx.strokeStyle = 'rgba(255, 255, 255, 0.85)';
				ctx.lineWidth = 3 * scale;
				ctx.stroke();
				ctx.strokeStyle = item.colour;
				ctx.lineWidth = 1.25 * scale;
				ctx.stroke();
			}
			ctx.beginPath();
			ctx.arc(at.x * scale, at.y * scale, 5.5 * scale, 0, 2 * Math.PI);
			ctx.fillStyle = '#ffffff';
			ctx.fill();
			ctx.beginPath();
			ctx.arc(at.x * scale, at.y * scale, 3.5 * scale, 0, 2 * Math.PI);
			ctx.fillStyle = item.colour;
			ctx.fill();
			if (!rect) {
				ctx.restore();
				continue;
			}
			ctx.beginPath();
			ctx.roundRect((rect.x + 0.5) * scale, (rect.y + 0.5) * scale, (rect.w - 1) * scale, (rect.h - 1) * scale, 5 * scale);
			ctx.fillStyle = 'rgba(255, 255, 255, 0.94)';
			ctx.fill();
			ctx.strokeStyle = '#cddce4';
			ctx.lineWidth = scale;
			ctx.stroke();
			ctx.font = font;
			ctx.textBaseline = 'middle';
			ctx.fillStyle = item.colour;
			ctx.fillText(item.text, (rect.x + 7) * scale, (rect.y + rect.h / 2 + 0.5) * scale, (rect.w - 14) * scale);
			ctx.restore();
		}
	}
}

/** The places a label's box may take, first choice first: {x, y, w, h, leader}. */
function candidates({ at, w, h, offset }) {
	const box = (cx, cy, leader) => ({ x: Math.round(cx - w / 2), y: Math.round(cy - h / 2), w, h, leader });
	const out = [];
	if (offset) out.push(box(at.x + offset[0], at.y + offset[1], true));
	out.push(
		box(at.x + GAP + w / 2, at.y, false),
		box(at.x - GAP - w / 2, at.y, false),
		box(at.x, at.y - GAP - h / 2, false),
		box(at.x, at.y + GAP + h / 2, false),
	);
	for (const ring of RINGS) {
		for (let k = 0; k < 8; k++) {
			const angle = (k * Math.PI) / 4 - Math.PI / 4;
			out.push(box(at.x + Math.cos(angle) * (ring + w / 2), at.y + Math.sin(angle) * (ring + h / 2), true));
		}
	}
	return out;
}

/** The point of a rectangle nearest a point. */
function nearestOnRect(r, p) {
	return [Math.min(Math.max(p.x, r.x), r.x + r.w), Math.min(Math.max(p.y, r.y), r.y + r.h)];
}

function occupancyOf(mesh) {
	if (!occupancies.has(mesh)) occupancies.set(mesh, new Occupancy(mesh));
	return occupancies.get(mesh);
}
