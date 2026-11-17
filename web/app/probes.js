// Probing the surface: the value under the cursor, and numbered probes pinned by a click, listed with their position
// and predicted values. Values are read from the working mesh at the vertex the cursor is on.

import { download, h } from './dom.js';
import { baseName } from './export.js';
import { FIELDS, valueAt } from './fields.js';
import { signed } from './format.js';
import { icon } from './icons.js';

/** The columns of a probe row. */
const COLUMNS = [
	{ key: 'x', label: 'x', unit: 'm', read: (mesh, v) => mesh.positions[3 * v] },
	{ key: 'y', label: 'y', unit: 'm', read: (mesh, v) => mesh.positions[3 * v + 1] },
	{ key: 'z', label: 'z', unit: 'm', read: (mesh, v) => mesh.positions[3 * v + 2] },
	{ key: 'cp', label: 'Cp', read: (mesh, v) => valueAt(mesh, 'cp', v) },
	{ key: 'sigma', label: 'σ', read: (mesh, v) => valueAt(mesh, 'sigma', v) },
];

export function probesCsv(c) {
	const mesh = c.result.mesh;
	const head = ['probe', 'vertex', ...COLUMNS.map((col) => (col.unit ? `${col.key}_${col.unit}` : col.key))];
	const rows = c.probes.map((p) => [p.n, p.vertex, ...COLUMNS.map((col) => col.read(mesh, p.vertex).toFixed(6))]);
	return [head, ...rows].map((r) => r.join(',')).join('\n') + '\n';
}

/**
 * host: the overlay over the 3D view; view: the shared view (its field is what the hover readout shows).
 * changed() runs after probes are added or removed.
 */
export function createProbes({ host, viewer, active, view, changed }) {
	const chip = h('div', { class: 'hover-chip', hidden: true, role: 'status' });
	const markers = h('div', { class: 'probe-markers' });
	host.append(markers, chip);

	function hover(hit, clientX, clientY) {
		const c = active();
		if (!hit || !c?.result) {
			chip.hidden = true;
			return;
		}
		const mesh = c.result.mesh;
		const v = hit.vertex;
		const value = valueAt(mesh, view.field, v);
		const rect = host.getBoundingClientRect();
		chip.replaceChildren(
			h('div', { class: 'chip-value' }, h('span', {}, FIELDS[view.field].symbol), h('b', {}, signed(value, 3))),
			h('div', { class: 'chip-at' }, ['x', 'y', 'z'].map((axis, k) => h('span', {}, `${axis} ${signed(mesh.positions[3 * v + k], 3)}`))),
		);
		const x = clientX - rect.left + 14;
		const y = clientY - rect.top + 16;
		chip.style.transform = `translate(${Math.min(x, rect.width - 190)}px, ${Math.min(y, rect.height - 60)}px)`;
		chip.hidden = false;
	}

	function pin(hit) {
		const c = active();
		if (!hit || !c?.result) return;
		const P = c.result.mesh.positions;
		c.probes.push({ n: c.nextProbe++, vertex: hit.vertex, point: [P[3 * hit.vertex], P[3 * hit.vertex + 1], P[3 * hit.vertex + 2]] });
		changed();
	}

	function place() {
		const c = active();
		const probes = c?.result ? c.probes : [];
		while (markers.children.length < probes.length) markers.append(h('div', { class: 'probe-marker' }, h('span', {})));
		while (markers.children.length > probes.length) markers.lastChild.remove();
		probes.forEach((probe, i) => {
			const marker = markers.children[i];
			const p = viewer.project(probe.point);
			marker.firstChild.textContent = probe.n;
			marker.hidden = !p.visible || viewer.clipped(probe.point);
			marker.style.transform = `translate(${p.x}px, ${p.y}px)`;
		});
	}
	viewer.addEventListener('camera', place);

	/** The probe table. */
	function table(c) {
		const mesh = c.result.mesh;
		const remove = (n) => {
			c.probes = c.probes.filter((p) => p.n !== n);
			changed();
		};
		if (c.probes.length === 0) return h('p', { class: 'plot-empty' }, 'No probes');
		return h('div', { class: 'probes' },
			h('table', { class: 'probe-table' },
				h('thead', {}, h('tr', {}, h('th', {}, '#'), COLUMNS.map((col) => h('th', {}, col.label, col.unit ? h('span', { class: 'unit' }, ` ${col.unit}`) : null)), h('th', {}))),
				h('tbody', {}, c.probes.map((p) => h('tr', { 'data-probe': p.n },
					h('td', { class: 'probe-n' }, p.n),
					COLUMNS.map((col) => h('td', { class: 'num' }, signed(col.read(mesh, p.vertex), 3))),
					h('td', {}, h('button', { type: 'button', class: 'icon-button', title: `Remove probe ${p.n}`, onclick: () => remove(p.n) }, icon('close'))))))));
	}

	/** Copy, save and clear, for the dock's header. */
	function actions(c) {
		const none = c.probes.length === 0;
		const copy = () => navigator.clipboard.writeText(probesCsv(c).replaceAll(',', '\t'));
		const save = () => download(new Blob([probesCsv(c)], { type: 'text/csv' }), `${baseName(c)}-probes.csv`);
		const clear = () => {
			c.probes = [];
			changed();
		};
		return h('div', { class: 'probe-actions' },
			h('button', { type: 'button', class: 'tool', disabled: none, onclick: copy }, icon('copy'), 'Copy'),
			h('button', { type: 'button', class: 'tool', disabled: none, onclick: save }, icon('download'), 'CSV'),
			h('button', { type: 'button', class: 'tool', disabled: none, onclick: clear }, icon('close'), 'Clear'));
	}

	return { hover, pin, place, table, actions };
}
