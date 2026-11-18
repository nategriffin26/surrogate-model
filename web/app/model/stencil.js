// The marks of `measures.js` for one group of inputs, drawn on the page's car in the group's colour. Everything is
// depth-tested against the car: what lies on the surface or in front of it is drawn in full, what the car hides is
// drawn faintly, so the eye can tell which is which. Nearby surface is a crisp ring on the surface at each size, with
// each size's mean normal as an arrow; a slice's read is a dot where it ends, with a faint hairline from its station
// on the point's fore-aft line. The point is a handle the reader drags.

import * as THREE from 'three';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { sizeShades } from './charts.js';
import { SurfaceCutter } from './cuts.js';

/** Length of the drawn normal, in metres. */
const NORMAL_LENGTH = 0.2;
/** Dots are drawn this far towards the camera, in metres, so those on the surface are not lost in it. */
const DOT_LIFT = 0.012;
/** Dot diameters in CSS pixels: the handle, a reading, a slice's read, and a read of the station shown. */
const DOT = { handle: 21, reading: 6, slice: 8, shown: 11 };
/**
 * While the rings of nearby surface are drawn, the handle is a small ink dot, so the smallest ring (3 cm, about 7 px
 * across at the page's close view on a phone) is seen around it.
 */
const SMALL_HANDLE = 9;
/** How near the pointer, in CSS pixels, the handle must be to be under it. */
const REACH = 14;
/** At most this many sizes of nearby surface are ringed at once. */
const SIZES = 4;
/** Length of the mean normal drawn for the smallest size of nearby surface, in metres, and how much longer each next one is. */
const MEAN_NORMAL = [0.1, 0.06];
/** How strongly the sizes of nearby surface not shown are ringed, against the shown size's 1. */
const FAINT_SIZE = 0.3;
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
const vec3 SHADOW = vec3(0.05, 0.13, 0.2);  // the handle's soft shadow, in linear light
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
		color = mix(SHADOW, color, disc / max(cover, 1e-3));
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

// The car's surface, lifted a little along its normals so the rings drawn on it are not lost in it.
const RING_VERTEX = /* glsl */ `
uniform float uLift;
varying vec3 vWorld;
void main() {
	vec4 world = modelMatrix * vec4(position + normal * uLift, 1.0);
	vWorld = world.xyz;
	gl_Position = projectionMatrix * viewMatrix * world;
}`;

// The sizes of nearby surface about the point: a crisp ring at each size's radius, about 1.5 pixels wide whatever the
// zoom, and inside it a light fill in the shade of the smallest ring that holds the place; each size at its strength,
// the shown size's full and the others' faint.
const RING_FRAGMENT = /* glsl */ `
uniform vec3 uCentre;
uniform float uRadius[${SIZES}];
uniform vec3 uShade[${SIZES}];
uniform float uStrength[${SIZES}];
varying vec3 vWorld;
void main() {
	float d = length(vWorld - uCentre);
	float w = max(fwidth(d), 1e-6);
	float fill = 0.0;
	float line = 0.0;
	vec3 lineColor = uShade[0];
	vec3 fillColor = uShade[0];
	bool inside = false;
	for (int s = 0; s < ${SIZES}; s++) {
		if (uRadius[s] <= 0.0) continue;
		float l = (1.0 - smoothstep(0.7 * w, 1.7 * w, abs(d - uRadius[s]))) * uStrength[s];
		if (l > line) {
			line = l;
			lineColor = uShade[s];
		}
		if (d < uRadius[s]) {
			fill += 0.08 * uStrength[s];
			if (!inside) {
				fillColor = uShade[s];
				inside = true;
			}
		}
	}
	float alpha = max(line * 0.95, fill);
	if (alpha < 0.01) discard;
	gl_FragColor = vec4(mix(fillColor, lineColor, line), alpha);
	#include <colorspace_fragment>
}`;

/** Draw order: the rings on the surface, lines, dots, the handle. */
const ORDER = { rings: 3, line: 5, dot: 7, handle: 8 };

export class Stencil {
	constructor(viewer) {
		this.viewer = viewer;
		this.object = null;
		this.point = null;
		this.car = null;
		const pixelRatio = viewer.renderer.getPixelRatio();
		const line = (linewidth, opacity, hidden) =>
			new LineMaterial({
				vertexColors: true,
				transparent: true,
				depthWrite: false,
				linewidth,
				opacity,
				...(hidden ? { depthFunc: THREE.GreaterDepth, opacity: opacity * 0.2 } : {}),
			});
		// Per width: the part of a line in view, then the part the car hides.
		this.lines = {};
		for (const [name, width] of [
			['major', 2.5],
			['minor', 1.5],
		])
			this.lines[name] = [line(width, 1, false), line(width * 0.7, 1, true)];
		// A hairline is faint and drawn over the car, never hidden by it; the shown station's are bold.
		const over = (linewidth, opacity) =>
			new LineMaterial({
				vertexColors: true,
				transparent: true,
				depthWrite: false,
				depthTest: false,
				linewidth,
				opacity,
			});
		this.lines.hair = [over(1, 0.55)];
		this.lines.shown = [over(2.4, 0.95)];
		// The shown station's slice, fainter where the car hides it.
		this.lines.slice = [
			new LineMaterial({ vertexColors: true, transparent: true, depthWrite: false, linewidth: 1.6, opacity: 0.85 }),
			new LineMaterial({
				vertexColors: true,
				transparent: true,
				depthWrite: false,
				linewidth: 1.2,
				opacity: 0.22,
				depthFunc: THREE.GreaterDepth,
			}),
		];
		const dots = (opacity, depthFunc) =>
			new THREE.ShaderMaterial({
				uniforms: { uOpacity: { value: opacity }, uPixelRatio: { value: pixelRatio }, uLift: { value: DOT_LIFT } },
				vertexShader: DOT_VERTEX,
				fragmentShader: DOT_FRAGMENT,
				transparent: true,
				depthWrite: false,
				depthFunc,
			});
		this.dots = [dots(1, THREE.LessEqualDepth), dots(0.3, THREE.GreaterDepth)];
		this.cone = new THREE.Mesh(new THREE.ConeGeometry(0.013, 0.04, 20), new THREE.MeshBasicMaterial({ color: INK }));
		this.cone.renderOrder = ORDER.handle;
		this.ringMaterial = new THREE.ShaderMaterial({
			uniforms: {
				uCentre: { value: new THREE.Vector3() },
				uRadius: { value: new Array(SIZES).fill(0) },
				uShade: { value: Array.from({ length: SIZES }, () => new THREE.Color()) },
				uStrength: { value: new Array(SIZES).fill(1) },
				uLift: { value: 0.002 },
			},
			vertexShader: RING_VERTEX,
			fragmentShader: RING_FRAGMENT,
			transparent: true,
			depthWrite: false,
			polygonOffset: true,
			polygonOffsetFactor: -1,
			polygonOffsetUnits: -4,
		});
	}

	/** The car the marks are drawn on; its working mesh is cut for the drawings and ringed for nearby surface. */
	setCar(result) {
		if (this.car?.result === result) return;
		this.car?.rings.geometry.dispose();
		const { mesh, lookups } = result;
		const geometry = new THREE.BufferGeometry();
		geometry.setAttribute('position', new THREE.BufferAttribute(mesh.positions, 3));
		geometry.setAttribute('normal', new THREE.BufferAttribute(mesh.normals, 3));
		geometry.setIndex(new THREE.BufferAttribute(mesh.indices, 1));
		const rings = new THREE.Mesh(geometry, this.ringMaterial);
		rings.renderOrder = ORDER.rings;
		rings.frustumCulled = false;
		this.car = { result, cutter: new SurfaceCutter(mesh, lookups.definition.rasterStep), rings };
	}

	/**
	 * Draw the point, and one group's marks in its colour.
	 * @param {{ position: number[], normal: number[] }} point
	 * @param {import('./measures.js').Mark[]} marks
	 * @param {string} hex  the group's colour
	 * @param {number | null} shown  the station of the slices, or the size of nearby surface, drawn as the one shown
	 */
	draw(point, marks, hex, shown) {
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
		const sizes = marks
			.filter((m) => m.type === 'neighbourhood')
			.sort((p, q) => p.scale - q.scale)
			.slice(0, SIZES);
		const shades = sizeShades(hex, sizes.length).map((shade) => new THREE.Color(shade));

		for (const mark of marks) {
			const isShown = mark.station === shown;
			if (mark.type === 'reading') {
				const size = mark.layer !== 'slice' ? DOT.reading : isShown ? DOT.shown : DOT.slice;
				dots.push({ at: mark.at, size, style: mark.hollow ? 1 : 0, color });
			} else if (mark.type === 'guide') {
				addLine('minor', mark.from, mark.to, color);
			} else if (mark.type === 'hairline') {
				// A read that runs to the ground for want of body is drawn only for the station shown.
				if (isShown || !mark.ground) addLine(isShown ? 'shown' : 'hair', mark.from, mark.to, color);
			} else if (mark.type === 'stations') {
				addLine('hair', mark.from, mark.to, ink);
			}
		}
		// The shown station's slice, the one its area is read from.
		if (marks.some((m) => m.type === 'stations') && shown !== null) {
			const { positions } = this.car.cutter.cut(0, point.position[0] + shown);
			for (let i = 0; i < positions.length; i += 6)
				addLine('slice', positions.subarray(i, i + 3), positions.subarray(i + 3, i + 6), color);
		}
		// Nearby surface: a ring at each size's radius, the shown size's strong and the others' faint; under the shown
		// size, its mean normal as an arrow from the point, and across it the line between the part ahead and the part
		// behind.
		if (sizes.length) {
			const rings = this.ringMaterial.uniforms;
			rings.uCentre.value.fromArray(point.position);
			rings.uRadius.value.fill(0);
			sizes.forEach((size, k) => {
				const isShown = shown === null || size.scale === shown;
				rings.uRadius.value[k] = size.scale;
				rings.uShade.value[k].copy(shades[k]);
				rings.uStrength.value[k] = isShown ? 1 : FAINT_SIZE;
				if (!isShown) return;
				const length = Math.hypot(...size.normal);
				const reach = MEAN_NORMAL[0] + MEAN_NORMAL[1] * k;
				if (length > 0)
					addLine(
						'major',
						size.at,
						size.at.map((p, c) => p + (size.normal[c] / length) * reach),
						shades[k],
					);
				const [, ny, nz] = point.normal;
				const across = Math.hypot(ny, nz);
				if (across > 1e-3) {
					const t = [0, nz / across, -ny / across];
					addLine(
						'hair',
						size.at.map((p, c) => p - t[c] * size.scale),
						size.at.map((p, c) => p + t[c] * size.scale),
						ink,
					);
				}
			});
			group.add(this.car.rings);
		}

		// The point: its facing as an arrow, and the handle.
		const tip = point.position.map((p, k) => p + point.normal[k] * NORMAL_LENGTH);
		addLine('major', point.position, tip, ink);
		this.cone.position.fromArray(tip);
		this.cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(...point.normal));
		group.add(this.cone);
		dots.push(
			sizes.length
				? { at: point.position, size: SMALL_HANDLE, style: 0, color: ink }
				: { at: point.position, size: DOT.handle, style: 2, color: ink },
		);

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
		this.viewer.setOverlay(group);
	}

	addDots(group, dots) {
		const geometry = new THREE.BufferGeometry();
		geometry.setAttribute(
			'position',
			new THREE.Float32BufferAttribute(
				dots.flatMap((d) => d.at),
				3,
			),
		);
		geometry.setAttribute(
			'color',
			new THREE.Float32BufferAttribute(
				dots.flatMap((d) => d.color.toArray()),
				3,
			),
		);
		geometry.setAttribute(
			'size',
			new THREE.Float32BufferAttribute(
				dots.map((d) => d.size),
				1,
			),
		);
		geometry.setAttribute(
			'style',
			new THREE.Float32BufferAttribute(
				dots.map((d) => d.style),
				1,
			),
		);
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
		this.viewer.setOverlay(null);
	}

	/** Free what the last drawing made; materials, the cone and the car's ringed surface are kept. */
	release() {
		if (!this.object) return;
		const freed = new Set();
		const kept = new Set([this.cone.geometry, this.car?.rings.geometry]);
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
