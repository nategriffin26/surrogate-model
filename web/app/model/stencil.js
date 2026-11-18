// What the networks were told at a point, drawn on the car where each input was read, at true scale: the point and
// its normal; a sphere for each regional scale the bend is averaged over; the stations of its own slice, the
// neighbouring slices and the plan-view outline; and, from each station, the distance the input measured, out to the
// body surface or plane it found. Lines and dots hidden by the car are drawn dashed and faint.

import * as THREE from 'three';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { KIND_COLORS } from './inputs.js';

/** Length of the drawn normal, in metres. */
const NORMAL_LENGTH = 0.22;
/** Dot diameters in CSS pixels. */
const DOT = { point: 11, flag: 7, station: 5, end: 4 };
/** Dots are drawn this far towards the camera, in metres, so those lying on the surface are not lost in it. */
const LIFT = 0.012;

/**
 * @typedef {{ type: 'segment', from: number[], to: number[] } | { type: 'dot', at: number[], size: number, hollow?: boolean }
 *   | { type: 'sphere', at: number[], radius: number } | { type: 'arrow', from: number[], to: number[] }} Shape
 * @typedef {Shape & { kind: string, group: string | null }} Mark
 *   `group` null for the point and its normal, which every input is about.
 */

const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const times = (a, k) => [a[0] * k, a[1] * k, a[2] * k];

/**
 * Every mark of the stencil for an explanation. Offsets and directions are given outboard-positive, so on the car's
 * −y side their y flips, as the engine reads them.
 * @param {import('../../engine/result.js').ModelColumn[]} columns
 * @param {import('./inputs.js').InputGroup[]} groups
 * @param {import('../../engine/result.js').Explanation} explanation
 * @param {number} yCenter  the car's centre plane
 * @returns {Mark[]}
 */
export function stencilMarks(columns, groups, { position, normal, features }, yCenter) {
	const side = position[1] - yCenter >= 0 ? 1 : -1;
	const world = ([x, y, z]) => [x, side * y, z];
	const marks = [];
	for (const group of groups) {
		const base = { kind: group.kind, group: group.label };
		if (group.kind === 'bend') marks.push({ ...base, type: 'sphere', at: position, radius: group.scale });
		const stations = new Set();
		for (const i of group.columns) {
			const { offset, along } = columns[i];
			const place = add(position, world(offset));
			if (along) {
				const to = add(place, times(world(along), features[i]));
				marks.push({ ...base, type: 'segment', from: place, to }, { ...base, type: 'dot', at: to, size: DOT.end });
			}
			if (group.kind === 'flag') {
				marks.push({ ...base, type: 'dot', at: place, size: DOT.flag, hollow: features[i] < 0.5 });
			} else if (offset.some(Boolean) && !stations.has(place.join())) {
				stations.add(place.join());
				marks.push({ ...base, type: 'dot', at: place, size: DOT.station });
			}
		}
	}
	marks.push(
		{ kind: 'position', group: null, type: 'arrow', from: position, to: add(position, times(normal, NORMAL_LENGTH)) },
		{ kind: 'position', group: null, type: 'dot', at: position, size: DOT.point },
	);
	return marks;
}

/** Whether a mark stands out: no input group is focused, or the focused one is its own. */
export function emphasized(mark, focusedGroup) {
	return !focusedGroup || mark.group === null || mark.group === focusedGroup;
}

const DOT_VERTEX = /* glsl */ `
attribute vec3 color;
attribute float size;
attribute float hollow;
uniform float uPixelRatio;
uniform float uLift;
varying vec3 vColor;
varying float vHollow;
varying float vSize;
void main() {
	vec4 view = modelViewMatrix * vec4(position, 1.0);
	view.xyz += (isOrthographic ? vec3(0.0, 0.0, 1.0) : normalize(-view.xyz)) * uLift;
	gl_Position = projectionMatrix * view;
	vSize = size * uPixelRatio;
	gl_PointSize = vSize;
	vColor = color;
	vHollow = hollow;
}`;

// A disc in the mark's colour inside a white rim, or a ring with a white middle for a hollow dot.
const DOT_FRAGMENT = /* glsl */ `
uniform float uOpacity;
uniform float uPixelRatio;
varying vec3 vColor;
varying float vHollow;
varying float vSize;
void main() {
	float r = length(gl_PointCoord - 0.5) * vSize;
	float edge = 0.5 * vSize;
	float px = uPixelRatio;
	float inside = 1.0 - smoothstep(edge - px, edge, r);
	if (inside <= 0.0) discard;
	float ink = 1.0 - smoothstep(edge - 2.5 * px, edge - 1.5 * px, r);
	float hole = vHollow * (1.0 - smoothstep(edge - 4.5 * px, edge - 3.5 * px, r));
	gl_FragColor = vec4(mix(vec3(1.0), vColor, ink * (1.0 - hole)), inside * uOpacity);
	#include <colorspace_fragment>
}`;

// A sphere seen as its outline: opaque at the silhouette, nearly clear facing the camera.
const SPHERE_VERTEX = /* glsl */ `
varying vec3 vNormal;
varying vec3 vView;
void main() {
	vec4 view = modelViewMatrix * vec4(position, 1.0);
	vNormal = normalize(normalMatrix * normal);
	vView = isOrthographic ? vec3(0.0, 0.0, 1.0) : normalize(-view.xyz);
	gl_Position = projectionMatrix * view;
}`;

const SPHERE_FRAGMENT = /* glsl */ `
uniform vec3 uColor;
uniform float uOpacity;
varying vec3 vNormal;
varying vec3 vView;
void main() {
	float rim = 1.0 - abs(dot(normalize(vNormal), normalize(vView)));
	gl_FragColor = vec4(uColor, uOpacity * (0.03 + pow(rim, 5.0)));
	#include <colorspace_fragment>
}`;

/** Draw order: spheres, then the lines' halos, the lines, the dots and the normal's arrowhead. */
const ORDER = { sphere: 4, halo: 5, line: 6, dot: 7, arrow: 8 };

/** The stencil on the viewer's car. One instance per viewer; `draw` replaces what it drew before. */
export class Stencil {
	constructor(viewer) {
		this.viewer = viewer;
		this.object = null;
		const pixelRatio = viewer.renderer.getPixelRatio();
		const line = (options) => new LineMaterial({ vertexColors: true, transparent: true, depthWrite: false, ...options });
		const hidden = { dashed: true, dashSize: 0.025, gapSize: 0.02, depthFunc: THREE.GreaterDepth };
		// Strong lines sit on a white halo so they read over any colour the car shows.
		const halo = new LineMaterial({ color: 0xffffff, linewidth: 5, transparent: true, opacity: 0.85, depthWrite: false });
		this.lines = {
			strong: [[halo, ORDER.halo], [line({ linewidth: 2 }), ORDER.line], [line({ linewidth: 1.5, opacity: 0.45, ...hidden }), ORDER.line]],
			faint: [[line({ linewidth: 1.5, opacity: 0.16 }), ORDER.line], [line({ linewidth: 1, opacity: 0.08, ...hidden }), ORDER.line]],
		};
		const dots = (opacity, depthFunc) => new THREE.ShaderMaterial({
			uniforms: { uOpacity: { value: opacity }, uPixelRatio: { value: pixelRatio }, uLift: { value: LIFT } },
			vertexShader: DOT_VERTEX, fragmentShader: DOT_FRAGMENT, transparent: true, depthWrite: false, depthFunc,
		});
		this.dots = {
			strong: [dots(1, THREE.LessEqualDepth), dots(0.35, THREE.GreaterDepth)],
			faint: [dots(0.2, THREE.LessEqualDepth), dots(0.07, THREE.GreaterDepth)],
		};
		const sphere = (opacity) => new THREE.ShaderMaterial({
			uniforms: { uColor: { value: new THREE.Color(KIND_COLORS.bend) }, uOpacity: { value: opacity } },
			vertexShader: SPHERE_VERTEX, fragmentShader: SPHERE_FRAGMENT, transparent: true, depthWrite: false, depthTest: false,
		});
		this.spheres = { strong: sphere(0.6), faint: sphere(0.12) };
		this.sphereGeometry = new THREE.SphereGeometry(1, 64, 32);
		this.cone = new THREE.Mesh(new THREE.ConeGeometry(0.014, 0.045, 20), new THREE.MeshBasicMaterial({ color: KIND_COLORS.position }));
		this.cone.renderOrder = ORDER.arrow;
	}

	/** Draw marks, each strong or faint by `strong(mark)`. */
	draw(marks, strong) {
		const group = new THREE.Group();
		for (const level of ['strong', 'faint']) {
			const chosen = marks.filter((m) => strong(m) === (level === 'strong'));
			this.addLines(group, chosen, this.lines[level]);
			this.addDots(group, chosen, this.dots[level]);
			for (const m of chosen) {
				if (m.type !== 'sphere') continue;
				const mesh = new THREE.Mesh(this.sphereGeometry, this.spheres[level]);
				mesh.position.fromArray(m.at);
				mesh.scale.setScalar(m.radius);
				mesh.renderOrder = ORDER.sphere;
				group.add(mesh);
			}
		}
		const arrow = marks.find((m) => m.type === 'arrow');
		if (arrow) {
			const direction = new THREE.Vector3(...arrow.to).sub(new THREE.Vector3(...arrow.from)).normalize();
			this.cone.position.fromArray(arrow.to);
			this.cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
			group.add(this.cone);
		}
		this.release();
		this.object = group;
		this.viewer.setMarkers(group);
	}

	addLines(group, marks, materials) {
		const positions = [];
		const colors = [];
		const color = new THREE.Color();
		for (const m of marks) {
			if (m.type !== 'segment' && m.type !== 'arrow') continue;
			positions.push(...m.from, ...m.to);
			color.set(KIND_COLORS[m.kind]);
			colors.push(color.r, color.g, color.b, color.r, color.g, color.b);
		}
		if (!positions.length) return;
		const geometry = new LineSegmentsGeometry().setPositions(positions).setColors(colors);
		for (const [material, order] of materials) {
			const lines = new LineSegments2(geometry, material).computeLineDistances();
			lines.renderOrder = order;
			group.add(lines);
		}
	}

	addDots(group, marks, materials) {
		const dots = marks.filter((m) => m.type === 'dot').sort((a, b) => a.size - b.size);
		if (!dots.length) return;
		const geometry = new THREE.BufferGeometry();
		const color = new THREE.Color();
		geometry.setAttribute('position', new THREE.Float32BufferAttribute(dots.flatMap((m) => m.at), 3));
		geometry.setAttribute('color', new THREE.Float32BufferAttribute(dots.flatMap((m) => color.set(KIND_COLORS[m.kind]).toArray()), 3));
		geometry.setAttribute('size', new THREE.Float32BufferAttribute(dots.map((m) => m.size), 1));
		geometry.setAttribute('hollow', new THREE.Float32BufferAttribute(dots.map((m) => (m.hollow ? 1 : 0)), 1));
		for (const material of materials) {
			const points = new THREE.Points(geometry, material);
			points.frustumCulled = false;
			points.renderOrder = ORDER.dot;
			group.add(points);
		}
	}

	/** Take the stencil off the car. */
	clear() {
		this.release();
		this.viewer.setMarkers(null);
	}

	/** Free the geometry of what was drawn; materials, the sphere and the cone are kept for the next drawing. */
	release() {
		if (!this.object) return;
		const freed = new Set();
		this.object.traverse((o) => {
			if (o === this.cone || o.geometry === this.sphereGeometry || !o.geometry || freed.has(o.geometry)) return;
			freed.add(o.geometry);
			o.geometry.dispose();
		});
		this.object = null;
	}
}
