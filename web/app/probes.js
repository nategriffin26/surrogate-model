// Probing the surface. Hovering previews the probe a click would place: the cell under the cursor outlined at its true
// size, a marker at its centre and a readout of its values. On a dropped file the cell is the file's own polygon with
// the values predicted for it, which the engine looks up. A bundled example carries no file polygons, and the engine
// holds only the last file's, so otherwise it is the face of the working mesh, valued at its centre. A click places
// exactly what the preview shows.
//
// Probes come in named sets that belong to the session, each probe a point in metres, so a set made on one car reads
// the same places on another: every car reads each probe at its own cell nearest the point. A set grows by clicks or by
// stations along the section; the dock's table shows the set in use, read on the car shown.

import { VertexGrid, FaceIndex } from '../viewer/picking.js';
import { oneOf, round, triple } from './checks.js';
import { CUT_AXES } from './mesh-cut.js';
import { sectionStations } from './section.js';
import { failureNote, h, put, s } from './dom.js';
import { baseName, defineExport } from './export.js';
import { FIELDS } from './fields.js';
import { signed } from './format.js';
import { icon } from './icons.js';
import { menu } from './menu.js';
import { defineOperation } from './operations.js';

/** A probe site's value of each field the Result view shows. */
const VALUE = { cp: (site) => site.cp, sigma: (site) => site.sigma, drag: (site) => -site.cp * site.normal[0] };

/** The columns of a probe row, read from its site on a car. */
const COLUMNS = [
	{ key: 'x', label: 'x', unit: 'm', read: (site) => site.point[0] },
	{ key: 'y', label: 'y', unit: 'm', read: (site) => site.point[1] },
	{ key: 'z', label: 'z', unit: 'm', read: (site) => site.point[2] },
	{ key: 'cp', label: 'Cp', read: (site) => site.cp },
	{ key: 'sigma', label: 'σ', read: (site) => site.sigma },
];

/** Half the preview marker's width, and the space kept between the readout, the cell and the view's edge, in px. */
const MARKER = 6;
const GAP = 12;
const MARGIN = 8;
/** How far off a cell's plane a picked point may lie and still be on that cell: the working mesh strays about 1 mm. */
const OFF_PLANE = 0.002;
/** A click this close to where the preview was looked up places the preview: the viewer's own tolerance for a click. */
const CLICK_PX = 4;
/** A screen used by touch, where a probe is placed by a tap. */
const TOUCH = matchMedia('(pointer: coarse)');
/** Stations along a section: at most this many, and this many when none is said. */
const MOST_STATIONS = 200;
const STATIONS = 20;

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

/** A field added at run time (fields.addField) on working-mesh face f of case c: the mean of its corners' values; null where the case has none. */
function addedValue(c, field, f) {
	const values = FIELDS[field]?.values?.(c);
	if (!values) return null;
	const I = c.result.mesh.indices;
	return (values[I[3 * f]] + values[I[3 * f + 1]] + values[I[3 * f + 2]]) / 3;
}

/** Each probe's site on a case, by probe; looked up once per case. */
const sitesOn = (c) => (c.sites ??= new WeakMap());

/** A set's probes read on a case: each probe's site there, looking up the ones not yet known. */
export async function readSet(engine, c, set) {
	const sites = sitesOn(c);
	await Promise.all(set.probes.filter((p) => !sites.has(p)).map(async (p) => sites.set(p, await siteAtPoint(engine, c, p.point))));
	return set.probes.map((p) => sites.get(p));
}

/** The probe table of a set on a case as CSV; a dropped file's probes name the file's cell, so each can be found again in the file. */
function probesCsv(c, set, sites) {
	const cells = c.kind === 'file';
	const head = ['probe', ...(cells ? ['cell'] : []), ...COLUMNS.map((col) => (col.unit ? `${col.key}_${col.unit}` : col.key))];
	const rows = set.probes.map((p, i) => [p.n, ...(cells ? [sites[i].cell ?? ''] : []), ...COLUMNS.map((col) => col.read(sites[i]).toFixed(6))]);
	return [head, ...rows].map((r) => r.join(',')).join('\n') + '\n';
}

/** The first "Set n" no set is named. */
function freeSetName(sets) {
	for (let n = sets.length + 1; ; n++) if (!sets.some((x) => x.name === `Set ${n}`)) return `Set ${n}`;
}

/** A probe as operations report it, read on a case. */
const probeResult = (p, site) => ({
	number: p.n,
	...(site.cell === null ? {} : { cell: site.cell }),
	point_m: site.point.map((v) => round(v, 5)),
	normal: site.normal.map((v) => round(v, 4)),
	cp: round(site.cp),
	sigma: round(site.sigma),
});

/** context: { session, engine, update }. Probe sets as operations, and the probe table as an export. */
export function defineProbeOperations({ session, engine, update }) {
	const view = session.view;
	let nextSet = 1;
	const target = (case_id) => {
		const c = case_id === undefined ? session.active : session.byId(case_id);
		if (!c?.result) throw new Error(case_id === undefined ? 'No case with a result is on screen. Open a file or an example first.' : `Case ${case_id} has no result yet (${c.status}).`);
		return c;
	};
	const setById = (set_id) => {
		const set = session.probeSets.find((x) => x.id === set_id);
		if (!set) throw new Error(`There is no probe set ${set_id}. Sets: ${session.probeSets.map((x) => `${x.id} (${x.name})`).join(', ') || 'none'}.`);
		return set;
	};
	/** The set an operation acts on: the one named, or the one in use. */
	const chosen = (set_id) => {
		if (set_id !== undefined) return setById(set_id);
		const set = session.probeSets.find((x) => x.id === view.probeSet);
		if (!set) throw new Error('No probe set is in use. Place a probe or make a set with use_probe_set first.');
		return set;
	};
	const checkedSetName = (name) => {
		const label = String(name).trim();
		if (!label) throw new Error('name must not be empty.');
		if (session.probeSets.some((x) => x.name === label)) throw new Error(`A probe set is already named ${JSON.stringify(label)}. Choose another name.`);
		return label;
	};
	/** A new set in use, holding probes at `points` ([x, y, z] each) snapped to the cells of case c. */
	async function addSet(name, points, c) {
		const set = { id: nextSet++, name: checkedSetName(name), probes: [], next: 1 };
		await addProbes(set, points, c);
		session.probeSets.push(set);
		view.probeSet = set.id;
		update('dock', 'layout', 'markers');
		return set;
	}
	/** Probes at points, each at the centre of its cell on case c, which is where any other car looks for it. */
	async function addProbes(set, points, c) {
		const sites = await Promise.all(points.map((point) => siteAtPoint(engine, c, point)));
		return sites.map((site) => {
			const probe = { n: set.next++, point: site.point };
			sitesOn(c).set(probe, site);
			set.probes.push(probe);
			return probe;
		});
	}
	const setSummary = (set) => ({ set_id: set.id, name: set.name, probes: set.probes.length });
	const caseId = { type: 'integer', description: 'Default the case on screen.' };
	const setId = { type: 'integer', description: 'Default the set in use.' };

	defineOperation({
		name: 'place_probe',
		description: 'Place a probe at the surface cell nearest a point, in a probe set (default the set in use; a first set is made if there is none). On a dropped file the cell is the file\'s own polygon with the values predicted for it; on a bundled example, the working-mesh triangle, valued at its centre. The probe keeps that cell\'s centre, so every car reads it at its own cell nearest there. Returns the probe read on the case: its number in the set, the file cell index when there is one, its centre and outward normal, Cp and the spread σ(Cp).',
		parameters: {
			type: 'object',
			properties: { point: { type: 'array', items: { type: 'number' }, minItems: 3, maxItems: 3, description: '[x, y, z] in metres; snapped to the nearest surface cell.' }, case_id: caseId, set_id: setId },
			required: ['point'],
			additionalProperties: false,
		},
		async run({ point, case_id, set_id }) {
			const c = target(case_id);
			const at = triple('point', point);
			if (set_id === undefined && !session.probeSets.some((x) => x.id === view.probeSet)) await addSet(freeSetName(session.probeSets), [], c);
			const set = chosen(set_id);
			const [probe] = await addProbes(set, [at], c);
			update('dock', 'layout', 'markers');
			return { set_id: set.id, ...probeResult(probe, sitesOn(c).get(probe)) };
		},
	});

	defineOperation({
		name: 'place_probes_along_section',
		description: `Make a probe set of stations along the section: count probes spaced evenly along the plot axis over one branch of the outer surface it cuts (upper or lower; for a z section, y > 0 or y < 0), each in the middle of its share of the branch. The set comes into use. At most ${MOST_STATIONS}.`,
		parameters: {
			type: 'object',
			properties: {
				count: { type: 'integer', description: `How many stations; default ${STATIONS}.` },
				branch: { type: 'string', enum: ['upper', 'lower'], description: 'Default upper.' },
				name: { type: 'string', description: 'Name of the set; default the section and branch.' },
				case_id: caseId,
			},
			additionalProperties: false,
		},
		async run({ count = STATIONS, branch = 'upper', name, case_id }) {
			const c = target(case_id);
			if (!Number.isInteger(count) || count < 1 || count > MOST_STATIONS) throw new Error(`count must be a whole number from 1 to ${MOST_STATIONS}; got ${count}.`);
			const b = ['upper', 'lower'].indexOf(oneOf('branch', branch, ['upper', 'lower']));
			const plane = view.section;
			if (!plane.on) throw new Error('The section is off. Turn it on with set_section first.');
			const points = sectionStations(c, plane, count, b);
			if (points.length === 0) throw new Error('The section cuts no outer surface there. Move it onto the car.');
			const axis = ['x', 'y', 'z'][plane.axis];
			const set = await addSet(name ?? `${axis} = ${signed(plane.offset, 3)} m, ${CUT_AXES[plane.axis].branches[b].toLowerCase()}`, points, c);
			return { ...setSummary(set), probes: set.probes.map((p) => probeResult(p, sitesOn(c).get(p))) };
		},
	});

	defineOperation({
		name: 'list_probes',
		description: 'The probe sets, and the probes of one (default the set in use) read on a case: each one\'s number, cell, centre, normal, Cp and σ.',
		parameters: { type: 'object', properties: { case_id: caseId, set_id: setId }, additionalProperties: false },
		async run({ case_id, set_id }) {
			const sets = session.probeSets.map(setSummary);
			const set = set_id === undefined ? session.probeSets.find((x) => x.id === view.probeSet) : setById(set_id);
			if (!set) return { sets, in_use: null };
			const c = target(case_id);
			const sites = await readSet(engine, c, set);
			return { sets, in_use: view.probeSet, set_id: set.id, name: set.name, probes: set.probes.map((p, i) => probeResult(p, sites[i])) };
		},
	});

	defineOperation({
		name: 'use_probe_set',
		description: 'Put a probe set in use, the one clicks add to and the table shows: an existing one by set_id (renamed when a name is given), or a new empty one with the given name.',
		parameters: { type: 'object', properties: { set_id: { type: 'integer' }, name: { type: 'string' } }, additionalProperties: false },
		async run({ set_id, name }) {
			if (set_id === undefined) {
				if (name === undefined) throw new Error('Give set_id to use a set, or name to make one.');
				return setSummary(await addSet(name, [], null));
			}
			const set = setById(set_id);
			if (name !== undefined && name.trim() !== set.name) set.name = checkedSetName(name);
			view.probeSet = set.id;
			update('dock', 'layout', 'markers');
			return setSummary(set);
		},
	});

	defineOperation({
		name: 'delete_probe_set',
		description: 'Delete a probe set and its probes.',
		parameters: { type: 'object', properties: { set_id: { type: 'integer' } }, required: ['set_id'], additionalProperties: false },
		run({ set_id }) {
			const set = setById(set_id);
			session.probeSets.splice(session.probeSets.indexOf(set), 1);
			if (view.probeSet === set.id) view.probeSet = session.probeSets.at(-1)?.id ?? null;
			update('dock', 'layout', 'markers');
			return { deleted: set_id, in_use: view.probeSet };
		},
	});

	defineOperation({
		name: 'remove_probe',
		description: 'Remove one probe from a set by its number. The others keep their numbers.',
		parameters: { type: 'object', properties: { number: { type: 'integer' }, set_id: setId }, required: ['number'], additionalProperties: false },
		run({ number, set_id }) {
			const set = chosen(set_id);
			if (!set.probes.some((p) => p.n === number)) throw new Error(`Set ${set.id} (${set.name}) has no probe ${number}. Its probes: ${set.probes.map((p) => p.n).join(', ') || 'none'}.`);
			set.probes = set.probes.filter((p) => p.n !== number);
			update('dock', 'layout', 'markers');
			return { removed: number };
		},
	});

	defineOperation({
		name: 'clear_probes',
		description: 'Remove every probe from a set; the set stays.',
		parameters: { type: 'object', properties: { set_id: setId }, additionalProperties: false },
		run({ set_id }) {
			const set = chosen(set_id);
			const removed = set.probes.length;
			set.probes = [];
			update('dock', 'layout', 'markers');
			return { removed };
		},
	});

	defineExport({
		name: 'probes',
		label: 'Probes',
		format: 'CSV',
		title: 'The probe table of the set in use: number, file cell, position, Cp and σ',
		unavailable: () => (session.probeSets.find((x) => x.id === view.probeSet)?.probes.length ? null : 'There are no probes in use. Click the car to place one.'),
		make: async (c) => {
			const set = chosen();
			return new Blob([probesCsv(c, set, await readSet(engine, c, set))], { type: 'text/csv' });
		},
		fileName: (c) => `${baseName(c)}-${chosen().name.replace(/[^\w.-]+/g, '-')}.csv`,
	});
}

/**
 * host: the overlay over the 3D view; view: the shared view (its probe set is the one in use). run(name, args) runs
 * an operation: a click pins its probe through `place_probe`. update(...parts) redraws once probes read on another car
 * are looked up. fieldIn(pane) names the field drawn in a pane of the 3D view: the readout shows the one under the
 * cursor.
 */
export function createProbes({ host, engine, viewer, active, session, update, run, fieldIn }) {
	const view = session.view;
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

	const inUse = () => session.probeSets.find((x) => x.id === view.probeSet) ?? null;

	/**
	 * The set in use read on a case, as far as its sites are known; the rest are looked up and the dock and markers
	 * redrawn when they are.
	 */
	const reading = new WeakSet();
	function known(c, set) {
		const sites = set.probes.map((p) => sitesOn(c).get(p));
		if (sites.includes(undefined) && !reading.has(c)) {
			reading.add(c);
			readSet(engine, c, set).then(() => {
				reading.delete(c);
				update('dock', 'markers');
			});
		}
		return sites;
	}

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
			Object.assign(current, { clientX, clientY, pane: hit.pane, face: hit.face });
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
			current = site ? { c, site, pane: next.hit.pane, face: next.hit.face, clientX: next.clientX, clientY: next.clientY } : null;
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
		current = { c, site, pane: hit.pane, face: hit.face, clientX, clientY };
		await run('place_probe', { case_id: c.id, point: site.point });
	}

	/** The preview: outline, marker and readout, placed for the camera as it is now, in the pane under the cursor. */
	function draw() {
		const site = over && current?.c === active() ? current.site : null;
		const pane = current?.pane ?? 0;
		const centre = site ? viewer.project(site.point, pane) : null;
		const field = fieldIn(pane);
		const reading = site ? (VALUE[field] ? VALUE[field](site) : addedValue(current.c, field, current.face)) : null;
		preview.hidden = !centre?.visible || reading === null || reading === undefined;
		if (preview.hidden) return;
		const corners = [];
		for (let k = 0; k < site.corners.length / 3; k++) corners.push(viewer.project(corner(site.corners, k), pane));
		const points = corners.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ');
		for (const polygon of outline.children) polygon.setAttribute('points', points);
		marker.style.transform = `translate(${centre.x}px, ${centre.y}px)`;
		symbol.textContent = FIELDS[field].symbol;
		value.textContent = signed(reading, 3);
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

	/** The set in use's numbered markers on the car shown, and the preview, where the camera now puts them. */
	function place() {
		const c = active();
		const set = inUse();
		const sites = c?.result && set ? known(c, set) : [];
		while (markers.children.length < sites.length) markers.append(h('div', { class: 'probe-marker' }, h('span', {})));
		while (markers.children.length > sites.length) markers.lastChild.remove();
		sites.forEach((site, i) => {
			const marker = markers.children[i];
			marker.hidden = !site;
			if (!site) return;
			const p = viewer.project(site.point);
			marker.firstChild.textContent = set.probes[i].n;
			marker.hidden = !p.visible;
			marker.style.transform = `translate(${p.x}px, ${p.y}px)`;
		});
		draw();
	}
	viewer.addEventListener('camera', place);

	/** How many probes the set in use holds, for the dock's tab. */
	const count = () => inUse()?.probes.length ?? 0;

	/** The probe table: the set in use, read on the car shown, under the note of a failed action from the tab. */
	function table(c) {
		const set = inUse();
		// Placing a probe is a click or a tap on the car, which no control in the tab shows.
		if (!set || set.probes.length === 0) return h('div', { class: 'probes' }, failure.el, h('p', { class: 'plot-empty' }, `${TOUCH.matches ? 'Tap' : 'Click'} the car to place a probe`));
		const sites = known(c, set);
		return h('div', { class: 'probes' },
			failure.el,
			h('table', { class: 'probe-table' },
				h('thead', {}, h('tr', {}, h('th', {}, '#'), COLUMNS.map((col) => h('th', {}, col.label, col.unit ? h('span', { class: 'unit' }, ` ${col.unit}`) : null)), h('th', {}))),
				h('tbody', {}, set.probes.map((p, i) => h('tr', { 'data-probe': p.n },
					h('td', { class: 'probe-n' }, p.n),
					COLUMNS.map((col) => h('td', { class: 'num' }, sites[i] ? signed(col.read(sites[i]), 3) : '')),
					h('td', {}, h('button', { type: 'button', class: 'icon-button', title: `Remove probe ${p.n}`, onclick: () => act('remove_probe', { number: p.n }) }, icon('close'))))))));
	}

	// ---- the dock's header: the set in use, stations along the section, and clear; the table is exported from Export ----

	const failure = failureNote();
	const act = (name, args) => failure.run(run(name, args));
	const setList = h('div', { class: 'menu-list', role: 'menu' });
	const setButton = h('button', { type: 'button', class: 'tool probe-set', title: 'The probe set in use' });
	const sets = menu(setButton, setList);
	const naming = h('input', { class: 'number probe-set-name', type: 'text', 'aria-label': 'Set name', spellcheck: 'false' });
	naming.addEventListener('keydown', (e) => {
		if (e.key === 'Escape') return renderSets();
		if (e.key !== 'Enter') return;
		const set = inUse();
		if (set && naming.value.trim()) act('use_probe_set', { set_id: set.id, name: naming.value.trim() });
		sets.close();
	});
	function renderSets(renaming = false) {
		const set = inUse();
		const item = (label, onclick, extra = {}) => h('button', { type: 'button', role: 'menuitem', class: 'menu-item', onclick, ...extra }, label);
		put(setList,
			...session.probeSets.map((x) => item([x.name, h('span', { class: 'menu-count' }, String(x.probes.length))], () => { sets.close(); act('use_probe_set', { set_id: x.id }); }, { role: 'menuitemradio', 'aria-checked': String(x === set) })),
			session.probeSets.length ? h('hr', { class: 'menu-rule' }) : null,
			item('New set', () => { sets.close(); act('use_probe_set', { name: freeSetName(session.probeSets) }); }),
			set && !renaming ? item('Rename', () => { renderSets(true); naming.value = set.name; naming.focus(); naming.select(); }) : null,
			set && renaming ? h('div', { class: 'menu-field' }, naming) : null,
			set ? item(`Delete ${set.name}`, () => { sets.close(); act('delete_probe_set', { set_id: set.id }); }, { class: 'menu-item danger' }) : null);
	}
	setButton.addEventListener('click', () => renderSets());

	const stationCount = h('input', { class: 'number', type: 'text', inputmode: 'numeric', value: String(STATIONS), 'aria-label': 'Stations' });
	const branchButtons = [0, 1].map((b) => h('button', { type: 'button', role: 'radio', class: 'segment', 'aria-checked': String(b === 0), onclick: () => branchButtons.forEach((x, k) => x.setAttribute('aria-checked', String(k === b))) }));
	const stationsForm = h('form', { class: 'stations', onsubmit: (e) => {
		e.preventDefault();
		const count = Number(stationCount.value);
		const branch = branchButtons[1].getAttribute('aria-checked') === 'true' ? 'lower' : 'upper';
		stations.close();
		act('place_probes_along_section', { count, branch });
	} },
		h('label', { class: 'stations-row' }, h('span', { class: 'control-label' }, 'Stations'), stationCount),
		h('div', { class: 'segmented', role: 'radiogroup', 'aria-label': 'Branch' }, branchButtons),
		h('button', { type: 'submit', class: 'button' }, 'Place'));
	const stationsButton = h('button', { type: 'button', class: 'tool', title: 'Probes at stations evenly spaced along the section' }, icon('section'), 'Along section');
	const stations = menu(stationsButton, stationsForm);
	stationsButton.addEventListener('click', () => {
		const names = CUT_AXES[view.section.axis].branches;
		branchButtons.forEach((b, k) => (b.textContent = names[k]));
	});

	const clear = h('button', { type: 'button', class: 'tool', onclick: () => act('clear_probes', {}) }, icon('close'), 'Clear');
	const actionsEl = h('div', { class: 'probe-actions' }, sets.el, stations.el, clear);

	/** The dock's header for the probes, brought up to date: the set in use and what can be done with it. */
	function actions() {
		const set = inUse();
		setButton.replaceChildren(set ? set.name : 'No set', icon('chevron'));
		stationsButton.disabled = !view.section.on;
		stationsButton.title = view.section.on ? 'Probes at stations evenly spaced along the section' : 'Turn the section on to place probes along it';
		clear.disabled = !set || set.probes.length === 0;
		return actionsEl;
	}

	return { hover, pin, place, table, actions, count };
}
