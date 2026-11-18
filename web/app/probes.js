// Probing the surface. Hovering previews the probe a click would place: the cell under the cursor outlined at its true
// size, a marker at its centre and a readout of its values. On a dropped file the cell is the file's own polygon with
// the values predicted for it, which the engine looks up. A bundled example carries no file polygons, and the engine
// holds only the last file's, so otherwise it is the face of the working mesh, valued at its centre. A click places
// exactly what the preview shows.

import { VertexGrid, FaceIndex } from '../viewer/picking.js';
import { round, triple } from './checks.js';
import { h, s } from './dom.js';
import { baseName, defineExport } from './export.js';
import { FIELDS } from './fields.js';
import { signed } from './format.js';
import { icon } from './icons.js';
import { defineOperation } from './operations.js';

/** A probe site's value of each field the Result view shows. */
const VALUE = { cp: (site) => site.cp, sigma: (site) => site.sigma, drag: (site) => -site.cp * site.normal[0] };

/** The columns of a probe row. */
const COLUMNS = [
	{ key: 'x', label: 'x', unit: 'm', read: (p) => p.point[0] },
	{ key: 'y', label: 'y', unit: 'm', read: (p) => p.point[1] },
	{ key: 'z', label: 'z', unit: 'm', read: (p) => p.point[2] },
	{ key: 'cp', label: 'Cp', read: (p) => p.cp },
	{ key: 'sigma', label: 'σ', read: (p) => p.sigma },
];

/** Half the preview marker's width, and the space kept between the readout, the cell and the view's edge, in px. */
const MARKER = 6;
const GAP = 12;
const MARGIN = 8;
/** How far off a cell's plane a picked point may lie and still be on that cell: the working mesh strays about 1 mm. */
const OFF_PLANE = 0.002;
/** A click this close to where the preview was looked up places the preview: the viewer's own tolerance for a click. */
const CLICK_PX = 4;

export function probesCsv(c) {
	// A dropped file's probes name the file's cell, so each can be found again in the file.
	const cells = c.kind === 'file';
	const head = ['probe', ...(cells ? ['cell'] : []), ...COLUMNS.map((col) => (col.unit ? `${col.key}_${col.unit}` : col.key))];
	const rows = c.probes.map((p) => [p.n, ...(cells ? [p.cell] : []), ...COLUMNS.map((col) => col.read(p).toFixed(6))]);
	return [head, ...rows].map((r) => r.join(',')).join('\n') + '\n';
}

const dot = (u, v) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
const cross = (u, v) => [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
const corner = (corners, k) => [corners[3 * k], corners[3 * k + 1], corners[3 * k + 2]];
const minus = (u, v) => [u[0] - v[0], u[1] - v[1], u[2] - v[2]];

/** Face f of a mesh as a probe site: its corners, its centre, its outward normal and its values at the centre. */
function faceSite(mesh, f) {
	const { indices: I, positions: P, normals: N } = mesh;
	const vertices = [I[3 * f], I[3 * f + 1], I[3 * f + 2]];
	const corners = new Float32Array(9);
	vertices.forEach((v, k) => corners.set(P.subarray(3 * v, 3 * v + 3), 3 * k));
	const [a, b, c] = [0, 1, 2].map((k) => corner(corners, k));
	const point = [0, 1, 2].map((k) => (a[k] + b[k] + c[k]) / 3);
	const normal = cross(minus(b, a), minus(c, a));
	// Turned the way the vertex normals point: out of the car.
	const outward = [0, 1, 2].map((k) => N[3 * vertices[0] + k] + N[3 * vertices[1] + k] + N[3 * vertices[2] + k]);
	const length = Math.hypot(...normal) * (dot(normal, outward) < 0 ? -1 : 1);
	const mean = (field) => (field[vertices[0]] + field[vertices[1]] + field[vertices[2]]) / 3;
	return { cell: null, corners, point, normal: normal.map((x) => x / length), cp: mean(mesh.cp), sigma: mean(mesh.sigma) };
}

/** Whether a surface point lies on a site's cell: near its plane and inside its outline, whichever way that winds. */
function holds(site, p) {
	const { corners, normal } = site;
	if (Math.abs(dot(minus(p, corner(corners, 0)), normal)) > OFF_PLANE) return false;
	const count = corners.length / 3;
	let sign = 0;
	for (let k = 0; k < count; k++) {
		const a = corner(corners, k);
		const side = dot(cross(minus(corner(corners, (k + 1) % count), a), minus(p, a)), normal);
		if (side * sign < 0) return false;
		if (side !== 0) sign = side;
	}
	return true;
}

/** Per working mesh, the lookups that find the face under a point; built on first use. */
const lookups = new WeakMap();

/** The working-mesh face nearest a point, as a probe site. */
function faceNear(mesh, point) {
	let lookup = lookups.get(mesh);
	if (!lookup) lookups.set(mesh, (lookup = { grid: new VertexGrid(mesh.positions), faces: new FaceIndex(mesh) }));
	return faceSite(mesh, lookup.faces.faceAt(point, lookup.grid.nearest(point)));
}

/**
 * The cell a probe at `point` reads, from the working-mesh face nearest the point (`face`, when the caller already
 * knows it): on the file analysed last, while the engine holds it, the file polygon there; otherwise that face.
 */
async function siteAtPoint(engine, c, point, face = faceNear(c.result.mesh, point)) {
	if (c.kind !== 'file') return face;
	// The face's normal keeps a point on a thin panel on its own side.
	const polygon = await engine.polygonAt(c.result, point, face.normal);
	if (!polygon) return face;
	return { cell: polygon.index, corners: polygon.corners, point: polygon.center, normal: polygon.normal, cp: polygon.cp, sigma: polygon.sigma };
}

/** A probe as operations report it. */
const probeResult = (p) => ({ number: p.n, ...(p.cell === null ? {} : { cell: p.cell }), point_m: p.point.map((v) => round(v, 5)), normal: p.normal.map((v) => round(v, 4)), cp: round(p.cp), sigma: round(p.sigma) });

/** app: { session, engine, update }. Probes as operations, and the probe table as an export. */
export function defineProbeOperations({ session, engine, update }) {
	const target = (case_id) => {
		const c = case_id === undefined ? session.active : session.byId(case_id);
		if (!c?.result) throw new Error(case_id === undefined ? 'No case with a result is on screen. Open a file or an example first.' : `Case ${case_id} has no result yet (${c.status}).`);
		return c;
	};
	const caseId = { type: 'integer', description: 'Default the case on screen.' };

	defineOperation({
		name: 'place_probe',
		description: 'Pin a probe on a car at the surface cell nearest a point: on a dropped file, the file\'s own polygon with the values predicted for it; on a bundled example, the working-mesh triangle, valued at its centre. Probes are numbered from 1 on each car and listed in the dock\'s probe table. Returns the probe: its number, the file cell index when there is one, its centre and outward normal, Cp and the spread σ(Cp).',
		parameters: {
			type: 'object',
			properties: { point: { type: 'array', items: { type: 'number' }, minItems: 3, maxItems: 3, description: '[x, y, z] in metres; snapped to the nearest surface cell.' }, case_id: caseId },
			required: ['point'],
			additionalProperties: false,
		},
		async run({ point, case_id }) {
			const c = target(case_id);
			const site = await siteAtPoint(engine, c, triple('point', point));
			const probe = { n: c.nextProbe++, ...site };
			c.probes.push(probe);
			update('dock', 'layout', 'markers');
			return probeResult(probe);
		},
	});

	defineOperation({
		name: 'list_probes',
		description: 'The probes pinned on a car, in order, with each one\'s cell, centre, normal, Cp and σ(Cp).',
		parameters: { type: 'object', properties: { case_id: caseId }, additionalProperties: false },
		run: ({ case_id }) => ({ probes: target(case_id).probes.map(probeResult) }),
	});

	defineOperation({
		name: 'remove_probe',
		description: 'Remove one probe from a car by its number. The others keep their numbers.',
		parameters: { type: 'object', properties: { number: { type: 'integer' }, case_id: caseId }, required: ['number'], additionalProperties: false },
		run({ number, case_id }) {
			const c = target(case_id);
			if (!c.probes.some((p) => p.n === number)) throw new Error(`Case ${c.id} has no probe ${number}. Its probes: ${c.probes.map((p) => p.n).join(', ') || 'none'}.`);
			c.probes = c.probes.filter((p) => p.n !== number);
			update('dock', 'layout', 'markers');
			return { removed: number };
		},
	});

	defineOperation({
		name: 'clear_probes',
		description: 'Remove every probe from a car.',
		parameters: { type: 'object', properties: { case_id: caseId }, additionalProperties: false },
		run({ case_id }) {
			const c = target(case_id);
			const removed = c.probes.length;
			c.probes = [];
			update('dock', 'layout', 'markers');
			return { removed };
		},
	});

	defineExport({
		name: 'probes',
		label: 'Probes',
		format: 'CSV',
		title: 'The probe table: number, file cell, position, Cp and σ',
		unavailable: (c) => (c.probes.length ? null : 'There are no probes on this car. Click the car to pin one.'),
		make: async (c) => new Blob([probesCsv(c)], { type: 'text/csv' }),
		fileName: (c) => `${baseName(c)}-probes.csv`,
	});
}

/**
 * host: the overlay over the 3D view; view: the shared view (its field is what the readout shows).
 * run(name, args) runs an operation: a click pins its probe through `place_probe`.
 */
export function createProbes({ host, engine, viewer, active, view, run }) {
	const edge = s('polygon', { class: 'edge' });
	const outline = s('svg', { class: 'cell-outline' }, s('polygon', { class: 'halo' }), edge);
	const marker = h('div', { class: 'preview-marker' });
	const symbol = h('span', { class: 'tip-symbol' });
	const value = h('b', { class: 'tip-value' });
	const coordinates = ['x', 'y', 'z'].map(() => h('span', { class: 'tip-number' }));
	const tip = h('div', { class: 'probe-tip', role: 'status' },
		h('div', { class: 'tip-head' }, symbol, value),
		h('div', { class: 'tip-at' }, ['x', 'y', 'z'].map((axis, k) => h('span', {}, axis, ' ', coordinates[k])), h('span', { class: 'tip-unit' }, 'm')));
	const preview = h('div', { class: 'probe-preview', hidden: true }, outline, marker, tip);
	const markers = h('div', { class: 'probe-markers' });
	host.append(markers, preview);

	/**
	 * The site a click would place: its case, and the pointer position it was looked up for. Kept while the pointer is
	 * down, so the click places it even though the preview hides for the press.
	 */
	let current = null;
	/** Whether the pointer is over the car, so the preview shows. */
	let over = false;
	/** The latest position still to look up, and whether a lookup is running: lookups that fall behind collapse into it. */
	let pending = null;
	let looking = false;

	/** The site under a pick: the file polygon there, or the working mesh's face where the engine holds no polygons. */
	const siteAt = (c, hit) => siteAtPoint(engine, c, hit.point, faceSite(c.result.mesh, hit.face));

	/** The cursor moved over the car (a hit at a client position) or off it (null). */
	async function hover(hit, clientX, clientY) {
		const c = active();
		over = Boolean(hit && c?.result);
		if (!over) {
			pending = null;
			draw();
			return;
		}
		if (current?.c === c && holds(current.site, hit.point)) {
			Object.assign(current, { clientX, clientY });
			draw();
			return;
		}
		pending = { hit, clientX, clientY };
		if (looking) return;
		looking = true;
		while (pending) {
			const next = pending;
			pending = null;
			const site = await siteAt(c, next.hit);
			if (c !== active()) break;
			current = site ? { c, site, clientX: next.clientX, clientY: next.clientY } : null;
			draw();
		}
		looking = false;
	}

	/**
	 * A click pins the previewed site when it lands where the preview was looked up, so the two never disagree;
	 * otherwise (a touch, or a click before the preview caught up) the site under the click. `place_probe` finds the
	 * same cell again from its centre.
	 */
	async function pin(hit, clientX, clientY) {
		const c = active();
		if (!hit || !c?.result) return;
		const previewed = current?.c === c && Math.hypot(clientX - current.clientX, clientY - current.clientY) <= CLICK_PX;
		const site = previewed ? current.site : await siteAt(c, hit);
		if (!site || c !== active()) return;
		current = { c, site, clientX, clientY };
		await run('place_probe', { case_id: c.id, point: site.point });
	}

	/** The preview: outline, marker and readout, placed for the camera as it is now. */
	function draw() {
		const site = over && current?.c === active() ? current.site : null;
		const centre = site && !viewer.clipped(site.point) ? viewer.project(site.point) : null;
		preview.hidden = !centre?.visible || !VALUE[view.field];
		if (preview.hidden) return;
		const corners = [];
		for (let k = 0; k < site.corners.length / 3; k++) corners.push(viewer.project(corner(site.corners, k)));
		const points = corners.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ');
		for (const polygon of outline.children) polygon.setAttribute('points', points);
		marker.style.transform = `translate(${centre.x}px, ${centre.y}px)`;
		symbol.textContent = FIELDS[view.field].symbol;
		value.textContent = signed(VALUE[view.field](site), 3);
		site.point.forEach((x, k) => (coordinates[k].textContent = signed(x, 3)));
		// Beside the cell and its marker: on the right unless that leaves the view, centred on the marker's height.
		const left = Math.min(centre.x - MARKER, ...corners.map((p) => p.x));
		const right = Math.max(centre.x + MARKER, ...corners.map((p) => p.x));
		const { offsetWidth: w, offsetHeight: tall } = tip;
		const fits = right + GAP + w <= host.clientWidth - MARGIN;
		const x = Math.max(MARGIN, Math.min(host.clientWidth - w - MARGIN, fits ? right + GAP : left - GAP - w));
		const y = Math.max(MARGIN, Math.min(host.clientHeight - tall - MARGIN, centre.y - tall / 2));
		tip.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
	}

	/** Placed probes' numbered markers, and the preview, where the camera now puts them. */
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
		draw();
	}
	viewer.addEventListener('camera', place);

	/** The probe table. */
	function table(c) {
		const remove = (n) => run('remove_probe', { case_id: c.id, number: n });
		if (c.probes.length === 0) return h('p', { class: 'plot-empty' }, 'No probes');
		return h('div', { class: 'probes' },
			h('table', { class: 'probe-table' },
				h('thead', {}, h('tr', {}, h('th', {}, '#'), COLUMNS.map((col) => h('th', {}, col.label, col.unit ? h('span', { class: 'unit' }, ` ${col.unit}`) : null)), h('th', {}))),
				h('tbody', {}, c.probes.map((p) => h('tr', { 'data-probe': p.n },
					h('td', { class: 'probe-n' }, p.n),
					COLUMNS.map((col) => h('td', { class: 'num' }, signed(col.read(p), 3))),
					h('td', {}, h('button', { type: 'button', class: 'icon-button', title: `Remove probe ${p.n}`, onclick: () => remove(p.n) }, icon('close'))))))));
	}

	/** Copy, save and clear, for the dock's header. */
	function actions(c) {
		const none = c.probes.length === 0;
		const copy = () => navigator.clipboard.writeText(probesCsv(c).replaceAll(',', '\t'));
		const save = () => run('download_export', { export: 'probes' });
		const clear = () => run('clear_probes', { case_id: c.id });
		return h('div', { class: 'probe-actions' },
			h('button', { type: 'button', class: 'tool', disabled: none, onclick: copy }, icon('copy'), 'Copy'),
			h('button', { type: 'button', class: 'tool', disabled: none, onclick: save }, icon('download'), 'CSV'),
			h('button', { type: 'button', class: 'tool', disabled: none, onclick: clear }, icon('close'), 'Clear'));
	}

	return { hover, pin, place, table, actions };
}
