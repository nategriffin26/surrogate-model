// Probing the surface: the value under the cursor, and numbered probes pinned by a click, listed with every field at
// their vertex. Values are read from the working mesh at the vertex the cursor is on.

import { download, h } from './dom.js';
import { baseName } from './export.js';
import { FIELDS, valueAt } from './fields.js';
import { signed } from './format.js';
import { icon } from './icons.js';

/** The columns a probe row holds; CFD and error only when the case has CFD. */
function probeColumns(hasCfd) {
	return [
		{ key: 'x', label: 'x', unit: 'm', read: (mesh, v) => mesh.positions[3 * v] },
		{ key: 'y', label: 'y', unit: 'm', read: (mesh, v) => mesh.positions[3 * v + 1] },
		{ key: 'z', label: 'z', unit: 'm', read: (mesh, v) => mesh.positions[3 * v + 2] },
		{ key: 'cp', label: 'Cp', read: (mesh, v) => valueAt(mesh, 'cp', v) },
		{ key: 'sigma', label: 'σ', read: (mesh, v) => valueAt(mesh, 'sigma', v) },
		...(hasCfd ? [
			{ key: 'cfd', label: 'Cp CFD', read: (mesh, v) => valueAt(mesh, 'cfd', v) },
			{ key: 'error', label: 'ΔCp', read: (mesh, v) => valueAt(mesh, 'error', v) },
		] : []),
	];
}

export function probesCsv(c) {
	const mesh = c.result.mesh;
	const columns = probeColumns(c.result.source.hasCfd);
	const head = ['probe', 'vertex', ...columns.map((col) => (col.unit ? `${col.key}_${col.unit}` : col.key))];
	const rows = c.view.probes.map((p) => [p.n, p.vertex, ...columns.map((col) => col.read(mesh, p.vertex).toFixed(6))]);
	return [head, ...rows].map((r) => r.join(',')).join('\n') + '\n';
}

/** host: the overlay over the 3D view. actions.changed() runs after probes are added or removed. */
export function createProbes({ host, viewer, active, changed }) {
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
		const field = c.view.compare ? ['cp', 'cfd'][hit.pane] : c.view.field;
		const v = hit.vertex;
		const value = valueAt(mesh, field, v);
		const rect = host.getBoundingClientRect();
		chip.replaceChildren(
			value === null ? null : h('div', { class: 'chip-value' }, h('span', {}, FIELDS[field].symbol), h('b', {}, signed(value, 3))),
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
		const view = c.view;
		const P = c.result.mesh.positions;
		view.probes.push({ n: view.nextProbe++, vertex: hit.vertex, point: [P[3 * hit.vertex], P[3 * hit.vertex + 1], P[3 * hit.vertex + 2]] });
		changed();
	}

	function place() {
		const c = active();
		const probes = c?.result ? c.view.probes : [];
		const panes = viewer.paneRects().length;
		const wanted = probes.length * panes;
		while (markers.children.length < wanted) markers.append(h('div', { class: 'probe-marker' }, h('span', {})));
		while (markers.children.length > wanted) markers.lastChild.remove();
		let i = 0;
		for (let pane = 0; pane < panes; pane++) {
			for (const probe of probes) {
				const marker = markers.children[i++];
				const p = viewer.project(probe.point, pane);
				marker.firstChild.textContent = probe.n;
				marker.hidden = !p.visible || viewer.clipped(probe.point);
				marker.style.transform = `translate(${p.x}px, ${p.y}px)`;
			}
		}
	}
	viewer.addEventListener('camera', place);

	/** The probe table for the side panel. */
	function table(c) {
		const columns = probeColumns(c.result.source.hasCfd);
		const mesh = c.result.mesh;
		const probes = c.view.probes;
		const remove = (n) => {
			c.view.probes = probes.filter((p) => p.n !== n);
			changed();
		};
		const copy = () => navigator.clipboard.writeText(probesCsv(c).replaceAll(',', '\t'));
		const save = () => download(new Blob([probesCsv(c)], { type: 'text/csv' }), `${baseName(c)}-probes.csv`);
		const clear = () => {
			c.view.probes = [];
			changed();
		};
		return h('div', { class: 'probes' },
			probes.length === 0
				? h('p', { class: 'empty' }, 'No probes')
				: h('table', { class: 'probe-table' },
					h('thead', {}, h('tr', {}, h('th', {}, '#'), columns.map((col) => h('th', {}, col.label, col.unit ? h('span', { class: 'unit' }, ` ${col.unit}`) : null)), h('th', {}))),
					h('tbody', {}, probes.map((p) => h('tr', { 'data-probe': p.n },
						h('td', { class: 'probe-n' }, p.n),
						columns.map((col) => h('td', { class: 'num' }, signed(col.read(mesh, p.vertex), 3))),
						h('td', {}, h('button', { type: 'button', class: 'icon-button', title: `Remove probe ${p.n}`, onclick: () => remove(p.n) }, icon('close'))))))),
			h('div', { class: 'probe-actions' },
				h('button', { type: 'button', class: 'button', disabled: !probes.length, onclick: copy }, icon('copy'), 'Copy'),
				h('button', { type: 'button', class: 'button', disabled: !probes.length, onclick: save }, icon('download'), 'CSV'),
				h('button', { type: 'button', class: 'button button-quiet', disabled: !probes.length, onclick: clear }, 'Clear')));
	}

	return { hover, pin, place, table };
}
