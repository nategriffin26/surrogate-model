// The orientation axes in the corner of the view, drawn as a post-processor draws them: three slim arrows from one
// origin, x downstream, y, z up, turning with the camera and foreshortened as the camera's projection would.
// Clicking an axis views the car from that axis's positive side; clicking it again views it from the negative side.

import * as THREE from 'three';

const NS = 'http://www.w3.org/2000/svg';
const SIZE = 92;
/** Arrow length in pixels. */
const R = 30;
/** Distance of the eye from the axes' origin in arrow lengths, for the perspective foreshortening. */
const EYE = 4;
const HEAD = { length: 7, halfWidth: 3 };
const AXES = [
	{ name: 'x', dir: [1, 0, 0], colour: '#c0392b' },
	{ name: 'y', dir: [0, 1, 0], colour: '#237f50' },
	{ name: 'z', dir: [0, 0, 1], colour: '#0b6fbf' },
];

function el(name, attrs) {
	const e = document.createElementNS(NS, name);
	for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
	return e;
}

export function createTriad(viewer) {
	const svg = el('svg', {
		class: 'triad',
		viewBox: `${-SIZE / 2} ${-SIZE / 2} ${SIZE} ${SIZE}`,
		width: SIZE,
		height: SIZE,
		role: 'group',
		'aria-label': 'Axes',
	});
	const axes = AXES.map((axis) => {
		const g = el('g', { class: 'triad-axis', role: 'button', tabindex: '0' });
		const title = el('title', {});
		title.textContent = `View from +${axis.name}, again for −${axis.name}`;
		const hit = el('line', { class: 'hit', x1: 0, y1: 0 });
		const shaft = el('line', { class: 'shaft', x1: 0, y1: 0, stroke: axis.colour });
		const head = el('polygon', { fill: axis.colour });
		const label = el('text', {
			class: 'label',
			fill: axis.colour,
			'text-anchor': 'middle',
			'dominant-baseline': 'central',
		});
		label.textContent = axis.name;
		g.append(title, hit, shaft, head, label);
		const dir = new THREE.Vector3(...axis.dir);
		const look = () => {
			// From the positive side, unless the view already is from there.
			const from = viewer.camera.position.clone().sub(viewer.target).normalize();
			const sign = from.dot(dir) > 0.999 ? -1 : 1;
			viewer.lookAlong(axis.dir.map((v, i) => (axis.name === 'z' && i === 1 ? -1e-4 : v * sign)));
		};
		g.addEventListener('click', look);
		g.addEventListener('keydown', (e) => (e.key === 'Enter' || e.key === ' ') && look());
		svg.append(g);
		return { g, hit, shaft, head, label, dir, depth: 0 };
	});
	svg.prepend(el('circle', { class: 'origin', r: 1.8 }));

	const q = new THREE.Quaternion();
	const v = new THREE.Vector3();
	function update() {
		q.copy(viewer.camera.quaternion).invert();
		const perspective = viewer.projection === 'perspective';
		for (const axis of axes) {
			// The axis in view space: x right, y up, z towards the eye.
			v.copy(axis.dir).applyQuaternion(q);
			const scale = perspective ? EYE / (EYE - v.z) : 1;
			const tx = v.x * R * scale,
				ty = -v.y * R * scale;
			const length = Math.hypot(tx, ty);
			const ux = length ? tx / length : 0,
				uy = length ? ty / length : 0;
			const headLength = Math.min(HEAD.length * scale, length * 0.5);
			const halfWidth = HEAD.halfWidth * scale * (headLength / (HEAD.length * scale));
			const bx = tx - ux * headLength,
				by = ty - uy * headLength;
			axis.shaft.setAttribute('x2', bx);
			axis.shaft.setAttribute('y2', by);
			axis.hit.setAttribute('x2', tx);
			axis.hit.setAttribute('y2', ty);
			axis.head.setAttribute(
				'points',
				`${tx},${ty} ${bx - uy * halfWidth},${by + ux * halfWidth} ${bx + uy * halfWidth},${by - ux * halfWidth}`,
			);
			// The label sits beyond the tip, or beside the origin when the axis points at the eye.
			const gap = length > 4 ? 8 : 0;
			axis.label.setAttribute('x', tx + ux * gap + (length > 4 ? 0 : 7));
			axis.label.setAttribute('y', ty + uy * gap - (length > 4 ? 0 : 7));
			axis.depth = v.z;
		}
		// Axes nearer the eye draw on top.
		for (const axis of [...axes].sort((a, b) => a.depth - b.depth)) svg.append(axis.g);
	}
	viewer.addEventListener('camera', update);
	return svg;
}
