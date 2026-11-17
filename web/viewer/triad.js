// The axis triad in the corner of the view: x downstream, y, z up, turning with the camera. Clicking an axis end
// looks at the car from that side.

import * as THREE from 'three';

const NS = 'http://www.w3.org/2000/svg';
const SIZE = 84;
const R = 28;
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
	const svg = el('svg', { class: 'triad', viewBox: `${-SIZE / 2} ${-SIZE / 2} ${SIZE} ${SIZE}`, width: SIZE, height: SIZE, role: 'group', 'aria-label': 'Axes' });
	const ends = [];
	for (const axis of AXES) {
		for (const sign of [1, -1]) {
			const g = el('g', { class: `triad-end ${sign > 0 ? 'positive' : 'negative'}`, role: 'button', tabindex: '0' });
			const label = `${sign > 0 ? '+' : '−'}${axis.name}`;
			const title = el('title', {});
			title.textContent = `View from ${label}`;
			const line = el('line', { x1: 0, y1: 0, stroke: axis.colour });
			const dot = el('circle', { r: sign > 0 ? 9 : 6, fill: sign > 0 ? axis.colour : '#ffffff', stroke: axis.colour });
			const text = el('text', { 'text-anchor': 'middle', dy: '0.35em' });
			text.textContent = sign > 0 ? axis.name : '';
			g.append(title, line, dot, text);
			const dir = axis.dir.map((v) => v * sign);
			const look = () => viewer.lookAlong(dir.map((v, i) => (axis.name === 'z' && i === 1 ? -1e-4 : v)));
			g.addEventListener('click', look);
			g.addEventListener('keydown', (e) => (e.key === 'Enter' || e.key === ' ') && look());
			svg.append(g);
			ends.push({ g, line, dot, text, dir: new THREE.Vector3(...dir) });
		}
	}
	const q = new THREE.Quaternion();
	const v = new THREE.Vector3();
	function update() {
		q.copy(viewer.camera.quaternion).invert();
		for (const end of ends) {
			v.copy(end.dir).applyQuaternion(q);
			const x = v.x * R, y = -v.y * R;
			end.line.setAttribute('x2', x);
			end.line.setAttribute('y2', y);
			end.dot.setAttribute('cx', x);
			end.dot.setAttribute('cy', y);
			end.text.setAttribute('x', x);
			end.text.setAttribute('y', y);
			end.depth = v.z;
		}
		// Ends nearer the viewer draw on top.
		for (const end of [...ends].sort((a, b) => a.depth - b.depth)) svg.append(end.g);
	}
	viewer.addEventListener('camera', update);
	return svg;
}
