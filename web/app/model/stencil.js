// The marks of `measures.js` for one group of inputs, drawn on the page's car in the group's colour. Everything is
// depth-tested against the car: what lies on the surface or in front of it is drawn in full, what the car hides is
// drawn faintly, so the eye can tell which is which. Cuts are the car's own outlines, lifted a hair off the surface;
// nearby surface is tinted by its Gaussians about the point, with each size's mean normal; the potential flow is an
// arrow of its velocity along the surface. The point is a handle the reader drags.

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
const DOT = { handle: 21, reading: 6 };
/** How near the pointer, in CSS pixels, the handle must be to be under it. */
const REACH = 14;
/** At most this many sizes of nearby surface are tinted at once. */
const GAUSSIANS = 4;
/** Length of the drawn flow velocity, in metres per unit of the stream's speed. */
const FLOW_LENGTH = 0.3;
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

const TINT_VERTEX = /* glsl */ `
uniform float uLift;
varying vec3 vWorld;
void main() {
	vec4 world = modelMatrix * vec4(position + normal * uLift, 1.0);
	vWorld = world.xyz;
	gl_Position = projectionMatrix * viewMatrix * world;
}`;

// The weight a surface element has in a size of nearby surface: its Gaussian about the point, averaged over the sizes.
const SOFT_FRAGMENT = /* glsl */ `
uniform vec3 uCentre;
uniform float uSigma[${GAUSSIANS}];
uniform float uSizes;
uniform vec3 uColor;
varying vec3 vWorld;
void main() {
	vec3 d = vWorld - uCentre;
	float r2 = dot(d, d);
	float total = 0.0;
	for (int s = 0; s < ${GAUSSIANS}; s++) {
		if (uSigma[s] > 0.0) total += exp(-0.5 * r2 / (uSigma[s] * uSigma[s]));
	}
	float alpha = 0.75 * total / max(uSizes, 1.0);
	if (alpha < 0.01) discard;
	gl_FragColor = vec4(uColor, alpha);
	#include <colorspace_fragment>
}`;

/** Draw order: the surface tint, lines, dots, the handle. */
const ORDER = { tint: 3, line: 5, dot: 7, handle: 8 };

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
		this.cone = new THREE.Mesh(new THREE.ConeGeometry(0.013, 0.04, 20), new THREE.MeshBasicMaterial({ color: INK }));
		this.cone.renderOrder = ORDER.handle;
		this.flowCone = new THREE.Mesh(this.cone.geometry, new THREE.MeshBasicMaterial());
		this.flowCone.renderOrder = ORDER.handle;
		this.softMaterial = new THREE.ShaderMaterial({
			uniforms: {
				uCentre: { value: new THREE.Vector3() },
				uSigma: { value: new Array(GAUSSIANS).fill(0) },
				uSizes: { value: 1 },
				uColor: { value: new THREE.Color() },
				uLift: { value: 0.002 },
			},
			vertexShader: TINT_VERTEX, fragmentShader: SOFT_FRAGMENT, transparent: true, depthWrite: false,
			polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -4,
		});
	}

	/** The car the marks are drawn on; its working mesh is cut for the outline and tinted under the Gaussians. */
	setCar(result) {
		if (this.car?.result === result) return;
		this.car?.soft.geometry.dispose();
		const { mesh, lookups } = result;
		const geometry = new THREE.BufferGeometry();
		geometry.setAttribute('position', new THREE.BufferAttribute(mesh.positions, 3));
		geometry.setAttribute('normal', new THREE.BufferAttribute(mesh.normals, 3));
		geometry.setIndex(new THREE.BufferAttribute(mesh.indices, 1));
		const soft = new THREE.Mesh(geometry, this.softMaterial);
		soft.renderOrder = ORDER.tint;
		soft.frustumCulled = false;
		this.car = { result, cutter: new SurfaceCutter(mesh, lookups.definition.rasterStep), soft, segments: new WeakMap() };
	}

	/**
	 * The part of a cut its inputs are read along: within its span of x (and a little beyond, so its ends are
	 * seen to be ends), on the point's side of the centre plane.
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
			if (x < lo || x > hi || (y - yCenter) * mark.side < 0) continue;
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
		const soft = this.softMaterial.uniforms;
		soft.uSigma.value.fill(0);
		soft.uColor.value.copy(color);
		soft.uCentre.value.fromArray(point.position);
		let sizes = 0;

		for (const mark of marks) {
			if (mark.type === 'reading') {
				dots.push({ at: mark.at, size: DOT.reading, style: mark.hollow ? 1 : 0, color });
			} else if (mark.type === 'guide') {
				addLine('minor', mark.from, mark.to, color);
			} else if (mark.type === 'cut') {
				const { positions, normals } = this.segments(mark);
				const style = mark.own ? 'major' : 'minor';
				for (let i = 0; i < positions.length; i += 6) {
					const lift = (k) => positions[i + k] + normals[i + k] * CUT_LIFT;
					addLine(style, [lift(0), lift(1), lift(2)], [lift(3), lift(4), lift(5)], color);
				}
			} else if (mark.type === 'neighbourhood') {
				// The Gaussian on the surface, and the mean normal under it as a line from the point.
				if (sizes < GAUSSIANS) soft.uSigma.value[sizes++] = mark.sigma;
				const length = Math.hypot(...mark.normal);
				if (length > 0) addLine('minor', mark.at, mark.at.map((p, k) => p + (mark.normal[k] / length) * NORMAL_LENGTH), color);
			} else if (mark.type === 'velocity') {
				// At a stagnation point the flow has no velocity along the surface, and nothing is drawn.
				const speed = Math.hypot(...mark.velocity);
				if (speed === 0) continue;
				const tip = mark.at.map((p, k) => p + mark.velocity[k] * FLOW_LENGTH);
				addLine('major', mark.at, tip, color);
				this.flowCone.material.color.copy(color);
				this.flowCone.position.fromArray(tip);
				this.flowCone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(...mark.velocity.map((v) => v / speed)));
				group.add(this.flowCone);
			}
		}
		// Each size's weights are averaged, so surface near the point, counted at every size, is tinted the most.
		soft.uSizes.value = sizes;
		if (sizes) group.add(this.car.soft);

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

	/** Free what the last drawing made; materials, the cones and the car's tinted surface are kept. */
	release() {
		if (!this.object) return;
		const freed = new Set();
		const kept = new Set([this.cone.geometry, this.car?.soft.geometry]);
		this.object.traverse((o) => {
			if (!o.geometry || kept.has(o.geometry) || freed.has(o.geometry)) return;
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
