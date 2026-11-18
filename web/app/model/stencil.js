// The marks of `measures.js` for one group of inputs, drawn on the page's car in the group's colour. Everything is
// depth-tested against the car: what lies on the surface or in front of it is drawn in full, what the car hides is
// drawn faintly, so the eye can tell which is which. Cuts are the car's own outlines, lifted a hair off the surface;
// boxes are shaded volumes, and the surface inside them is tinted by the weight each voxel has in the input. The
// point is a handle the reader drags.

import * as THREE from 'three';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { SurfaceCutter } from './cuts.js';

/** Length of the drawn normal, in metres. */
const NORMAL_LENGTH = 0.2;
/** Cuts sit this far off the surface, in metres, so the car does not hide them. */
const CUT_LIFT = 0.003;
/** Cuts are drawn this far beyond the stations at their ends, in metres. */
const SPAN_MARGIN = 0.06;
/** Dots are drawn this far towards the camera, in metres, so those on the surface are not lost in it. */
const DOT_LIFT = 0.012;
/** Dot diameters in CSS pixels. */
const DOT = { handle: 21, reading: 6, flag: 8 };
/** How near the pointer, in CSS pixels, the handle must be to be under it. */
const REACH = 14;
/** At most this many boxes are tinted on the surface at once (four sizes, three places each). */
const BLOCKS = 12;
const INK = '#0d2233';

const DOT_VERTEX = /* glsl */ `
attribute vec3 color;
attribute float size;
attribute float style;
uniform float uPixelRatio;
uniform float uLift;
varying vec3 vColor;
varying float vStyle;
varying float vSize;
void main() {
	vec4 view = modelViewMatrix * vec4(position, 1.0);
	view.xyz += (isOrthographic ? vec3(0.0, 0.0, 1.0) : normalize(-view.xyz)) * uLift;
	gl_Position = projectionMatrix * view;
	vSize = size * uPixelRatio;
	gl_PointSize = vSize;
	vColor = color;
	vStyle = style;
}`;

// A disc in the mark's colour inside a white rim (style 0), a ring around a white middle (1), or the handle: a white
// disc with an ink ring and core on a soft shadow (2).
const DOT_FRAGMENT = /* glsl */ `
uniform float uOpacity;
uniform float uPixelRatio;
varying vec3 vColor;
varying float vStyle;
varying float vSize;
void main() {
	float r = length(gl_PointCoord - 0.5) * vSize;
	float edge = 0.5 * vSize;
	float px = uPixelRatio;
	vec3 color;
	float cover;
	if (vStyle > 1.5) {
		float disc = 1.0 - smoothstep(edge - 3.0 * px, edge - 2.0 * px, r);
		float shadow = (1.0 - smoothstep(edge - 3.0 * px, edge, r)) * 0.35;
		float ring = smoothstep(edge - 6.0 * px, edge - 5.0 * px, r);
		float core = 1.0 - smoothstep(2.0 * px, 3.0 * px, r);
		color = mix(vec3(1.0), vColor, max(ring, core));
		cover = max(disc, shadow);
		color = mix(vec3(0.05, 0.13, 0.2), color, disc / max(cover, 1e-3));
	} else {
		cover = 1.0 - smoothstep(edge - px, edge, r);
		float ink = 1.0 - smoothstep(edge - 2.5 * px, edge - 1.5 * px, r);
		float hole = vStyle * (1.0 - smoothstep(edge - 4.5 * px, edge - 3.5 * px, r));
		color = mix(vec3(1.0), vColor, ink * (1.0 - hole));
	}
	if (cover <= 0.0) discard;
	gl_FragColor = vec4(color, cover * uOpacity);
	#include <colorspace_fragment>
}`;

const BOX_VERTEX = /* glsl */ `
varying vec3 vNormal;
void main() {
	vNormal = normalize(mat3(modelMatrix) * normal);
	gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;

// Each face lit by the viewer's key light, so a box reads as a solid.
const BOX_FRAGMENT = /* glsl */ `
uniform vec3 uColor;
uniform float uOpacity;
varying vec3 vNormal;
void main() {
	float light = 0.55 + 0.45 * max(dot(normalize(vNormal), normalize(vec3(-0.45, -0.55, 0.7))), 0.0);
	gl_FragColor = vec4(uColor * light, uOpacity * (gl_FrontFacing ? 1.0 : 0.55));
	#include <colorspace_fragment>
}`;

const TINT_VERTEX = /* glsl */ `
uniform float uLift;
varying vec3 vWorld;
void main() {
	vec4 world = modelMatrix * vec4(position + normal * uLift, 1.0);
	vWorld = world.xyz;
	gl_Position = projectionMatrix * viewMatrix * world;
}`;

// The weight a surface element has in a regional input: the trilinear weight of the voxel it lies in.
const TINT_FRAGMENT = /* glsl */ `
uniform vec3 uMin[${BLOCKS}];
uniform vec3 uWeights[${BLOCKS}];
uniform float uCell[${BLOCKS}];
uniform float uStrength[${BLOCKS}];
uniform float uSizes;
uniform vec3 uColor;
varying vec3 vWorld;
float blockWeight(vec3 q, int b) {
	vec3 cell = floor((q - uMin[b]) / uCell[b]);
	if (any(lessThan(cell, vec3(0.0))) || any(greaterThan(cell, vec3(1.0)))) return 0.0;
	vec3 w = mix(vec3(1.0) - uWeights[b], uWeights[b], cell);
	return w.x * w.y * w.z;
}
void main() {
	float total = 0.0;
	for (int b = 0; b < ${BLOCKS}; b++) {
		if (uStrength[b] > 0.0) total += uStrength[b] * blockWeight(vWorld, b);
	}
	float alpha = 0.75 * clamp(total / max(uSizes, 1.0), 0.0, 1.0);
	if (alpha < 0.01) discard;
	gl_FragColor = vec4(uColor, alpha);
	#include <colorspace_fragment>
}`;

/** Draw order: the surface tint, box backs, lines, box fronts, dots, the handle. */
const ORDER = { tint: 3, back: 4, line: 5, front: 6, dot: 7, handle: 8 };

const BOX_EDGES = [[0, 1], [1, 3], [3, 2], [2, 0], [4, 5], [5, 7], [7, 6], [6, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
const corner = ({ min, max }, k) => [k & 4 ? max[0] : min[0], k & 2 ? max[1] : min[1], k & 1 ? max[2] : min[2]];

export class Stencil {
	constructor(viewer) {
		this.viewer = viewer;
		this.object = null;
		this.point = null;
		this.car = null;
		const pixelRatio = viewer.renderer.getPixelRatio();
		const line = (linewidth, opacity, hidden) => new LineMaterial({
			vertexColors: true, transparent: true, depthWrite: false, linewidth, opacity,
			...(hidden ? { depthFunc: THREE.GreaterDepth, opacity: opacity * 0.2 } : {}),
		});
		// Per width: the part of a line in view, then the part the car hides.
		this.lines = {};
		for (const [name, width] of [['major', 2.5], ['minor', 1.5]]) this.lines[name] = [line(width, 1, false), line(width * 0.7, 1, true)];
		const dots = (opacity, depthFunc) => new THREE.ShaderMaterial({
			uniforms: { uOpacity: { value: opacity }, uPixelRatio: { value: pixelRatio }, uLift: { value: DOT_LIFT } },
			vertexShader: DOT_VERTEX, fragmentShader: DOT_FRAGMENT, transparent: true, depthWrite: false, depthFunc,
		});
		this.dots = [dots(1, THREE.LessEqualDepth), dots(0.3, THREE.GreaterDepth)];
		const box = (side, opacity) => new THREE.ShaderMaterial({
			uniforms: { uColor: { value: new THREE.Color() }, uOpacity: { value: opacity } },
			vertexShader: BOX_VERTEX, fragmentShader: BOX_FRAGMENT, transparent: true, depthWrite: false, side,
		});
		this.boxes = { back: box(THREE.BackSide, 0.045), front: box(THREE.FrontSide, 0.07) };
		this.boxGeometry = new THREE.BoxGeometry(1, 1, 1);
		this.cone = new THREE.Mesh(new THREE.ConeGeometry(0.013, 0.04, 20), new THREE.MeshBasicMaterial({ color: INK }));
		this.cone.renderOrder = ORDER.handle;
		this.tintMaterial = new THREE.ShaderMaterial({
			uniforms: {
				uMin: { value: Array.from({ length: BLOCKS }, () => new THREE.Vector3()) },
				uWeights: { value: Array.from({ length: BLOCKS }, () => new THREE.Vector3()) },
				uCell: { value: new Array(BLOCKS).fill(1) },
				uStrength: { value: new Array(BLOCKS).fill(0) },
				uSizes: { value: 1 },
				uColor: { value: new THREE.Color() },
				uLift: { value: 0.002 },
			},
			vertexShader: TINT_VERTEX, fragmentShader: TINT_FRAGMENT, transparent: true, depthWrite: false,
			polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -4,
		});
	}

	/** The car the marks are drawn on; its working mesh is cut for the slices and tinted inside the boxes. */
	setCar(result) {
		if (this.car?.result === result) return;
		this.car?.tint.geometry.dispose();
		const { mesh, lookups } = result;
		const geometry = new THREE.BufferGeometry();
		geometry.setAttribute('position', new THREE.BufferAttribute(mesh.positions, 3));
		geometry.setAttribute('normal', new THREE.BufferAttribute(mesh.normals, 3));
		geometry.setIndex(new THREE.BufferAttribute(mesh.indices, 1));
		const tint = new THREE.Mesh(geometry, this.tintMaterial);
		tint.renderOrder = ORDER.tint;
		tint.frustumCulled = false;
		this.car = { result, cutter: new SurfaceCutter(mesh, lookups.definition.rasterStep), tint, segments: new WeakMap() };
	}

	/**
	 * The part of a cut its inputs are read along: within its span of x (and a little beyond, so its ends are
	 * seen to be ends), and for a plan-view outline on the point's side of the centre plane.
	 */
	segments(mark) {
		const kept = this.car.segments.get(mark);
		if (kept) return kept;
		const { positions, normals } = this.car.cutter.cut(mark.axis, mark.value);
		const yCenter = this.car.result.lookups.yCenter;
		const [lo, hi] = [mark.span[0] - SPAN_MARGIN, mark.span[1] + SPAN_MARGIN];
		const p = [];
		const n = [];
		for (let i = 0; i < positions.length; i += 6) {
			const x = (positions[i] + positions[i + 3]) / 2;
			const y = (positions[i + 1] + positions[i + 4]) / 2;
			if (x < lo || x > hi || (mark.axis === 2 && (y - yCenter) * mark.side < 0)) continue;
			for (let k = 0; k < 6; k++) {
				p.push(positions[i + k]);
				n.push(normals[i + k]);
			}
		}
		const segments = { positions: Float32Array.from(p), normals: Float32Array.from(n) };
		this.car.segments.set(mark, segments);
		return segments;
	}

	/**
	 * Draw the point, and one group's marks in its colour.
	 * @param {{ position: number[], normal: number[] }} point
	 * @param {import('./measures.js').Mark[]} marks
	 * @param {string} hex  the group's colour
	 */
	draw(point, marks, hex) {
		const group = new THREE.Group();
		const lines = {};
		const addLine = (style, from, to, color) => {
			lines[style] ??= { positions: [], colors: [] };
			lines[style].positions.push(...from, ...to);
			lines[style].colors.push(color.r, color.g, color.b, color.r, color.g, color.b);
		};
		const dots = [];
		const color = new THREE.Color(hex);
		const ink = new THREE.Color(INK);
		const tint = this.tintMaterial.uniforms;
		tint.uStrength.value.fill(0);
		tint.uColor.value.copy(color);
		for (const box of Object.values(this.boxes)) box.uniforms.uColor.value.copy(color);
		let blocks = 0;
		const tinted = new Set();

		for (const mark of marks) {
			if (mark.type === 'reading') {
				dots.push({ at: mark.at, size: mark.kind === 'flag' ? DOT.flag : DOT.reading, style: mark.hollow ? 1 : 0, color });
			} else if (mark.type === 'guide') {
				addLine('minor', mark.from, mark.to, color);
			} else if (mark.type === 'cut') {
				const { positions, normals } = this.segments(mark);
				const style = mark.own ? 'major' : 'minor';
				for (let i = 0; i < positions.length; i += 6) {
					const lift = (k) => positions[i + k] + normals[i + k] * CUT_LIFT;
					addLine(style, [lift(0), lift(1), lift(2)], [lift(3), lift(4), lift(5)], color);
				}
			} else if (mark.type === 'block') {
				for (const [a, b] of BOX_EDGES) addLine(mark.sample === 'here' ? 'major' : 'minor', corner(mark, a), corner(mark, b), color);
				if (mark.sample === 'here') {
					for (const side of ['back', 'front']) {
						const box = new THREE.Mesh(this.boxGeometry, this.boxes[side]);
						box.position.set(...mark.min.map((v, k) => (v + mark.max[k]) / 2));
						box.scale.set(...mark.max.map((v, k) => v - mark.min[k]));
						box.renderOrder = ORDER[side];
						group.add(box);
					}
				}
				if (blocks < BLOCKS) {
					tint.uMin.value[blocks].fromArray(mark.min);
					tint.uWeights.value[blocks].fromArray(mark.weights);
					tint.uCell.value[blocks] = mark.cell;
					tint.uStrength.value[blocks] = 1;
					tinted.add(mark.scale);
					blocks++;
				}
			}
		}
		// Each size's weights are averaged, so surface near the point, counted at every size, is tinted the most.
		tint.uSizes.value = tinted.size;
		if (blocks) group.add(this.car.tint);

		// The point: its facing as an arrow, and the handle.
		const tip = point.position.map((p, k) => p + point.normal[k] * NORMAL_LENGTH);
		addLine('major', point.position, tip, ink);
		this.cone.position.fromArray(tip);
		this.cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(...point.normal));
		group.add(this.cone);
		dots.push({ at: point.position, size: DOT.handle, style: 2, color: ink });

		for (const [key, { positions, colors }] of Object.entries(lines)) {
			const geometry = new LineSegmentsGeometry().setPositions(positions).setColors(colors);
			for (const material of this.lines[key]) {
				const segments = new LineSegments2(geometry, material);
				segments.renderOrder = ORDER.line;
				group.add(segments);
			}
		}
		this.addDots(group, dots);
		this.release();
		this.object = group;
		this.point = point.position;
		this.viewer.setMarkers(group);
	}

	addDots(group, dots) {
		const geometry = new THREE.BufferGeometry();
		geometry.setAttribute('position', new THREE.Float32BufferAttribute(dots.flatMap((d) => d.at), 3));
		geometry.setAttribute('color', new THREE.Float32BufferAttribute(dots.flatMap((d) => d.color.toArray()), 3));
		geometry.setAttribute('size', new THREE.Float32BufferAttribute(dots.map((d) => d.size), 1));
		geometry.setAttribute('style', new THREE.Float32BufferAttribute(dots.map((d) => d.style), 1));
		for (const material of this.dots) {
			const points = new THREE.Points(geometry, material);
			points.frustumCulled = false;
			points.renderOrder = ORDER.dot;
			group.add(points);
		}
	}

	/** Take the marks off the car. */
	clear() {
		this.release();
		this.point = null;
		this.viewer.setMarkers(null);
	}

	/** Free what the last drawing made; materials, the box, the cone and the car's tint mesh are kept. */
	release() {
		if (!this.object) return;
		const freed = new Set();
		this.object.traverse((o) => {
			if (!o.geometry || o === this.cone || o === this.car?.tint || o.geometry === this.boxGeometry || freed.has(o.geometry)) return;
			freed.add(o.geometry);
			o.geometry.dispose();
		});
		this.object = null;
	}

	/** Screen position, in client pixels, of world points as the camera sees them in the pane under (clientX, clientY). */
	projector(clientX, clientY) {
		const at = this.viewer.paneAt(clientX, clientY);
		if (!at) return null;
		const rect = this.viewer.canvas.getBoundingClientRect();
		const camera = this.viewer.camera;
		const m = new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse).elements;
		const left = rect.left + at.index * at.w;
		return (p) => {
			const w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
			const x = (m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12]) / w;
			const y = (m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13]) / w;
			return [left + (x * 0.5 + 0.5) * at.w, rect.top + (0.5 - y * 0.5) * at.h];
		};
	}

	/** Whether (clientX, clientY) is on the point's handle. */
	onHandle(clientX, clientY) {
		const project = this.point && this.projector(clientX, clientY);
		if (!project) return false;
		const [x, y] = project(this.point);
		return Math.hypot(x - clientX, y - clientY) <= REACH;
	}
}
