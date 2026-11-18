// Variants of a car: its geometry changed on the page and predicted again. A variant stretches the car along an axis,
// raises or lowers the body on its wheels, or moves a region of it with the surface around the region following. It
// is made from the simplified working mesh the car is drawn with (about 600,000 triangles) and predicted there by the
// engine's quick prediction, in a second or two. The car itself is predicted unchanged on the same mesh, once, and a
// variant is compared with that (`comparedAs`), so the difference between them is the change alone. Each variant
// opens as a case beside its car, its surface kept in the workspace under variants/ with the prediction as cell
// arrays.
//
// A car and its open variants are one study, shown as one table whoever made the rows: the Variants tab and the
// assistant both call make_variant, and read_study reads the table.

import { boundsOf } from '../engine/geometry.js';
import { writeVtp } from '../engine/vtp.js';
import { finite, oneOf, round } from './checks.js';
import { h } from './dom.js';
import { baseName, defineExport } from './export.js';
import { signed } from './format.js';
import { icon } from './icons.js';
import { menu } from './menu.js';
import { defineOperation } from './operations.js';

const KINDS = ['stretch', 'ride_height', 'move'];
const KIND_LABELS = { stretch: 'Stretch', ride_height: 'Ride height', move: 'Move region' };
const AXES = ['x', 'y', 'z'];
/** What a stretch along each axis changes. */
const DIMENSIONS = { x: 'length', y: 'width', z: 'height' };
/** Which way a region moves along each axis. */
const DIRECTIONS = { x: 'Downstream', y: 'Away from the centre plane, on each side', z: 'Up' };
/** DrivAerML's wheels, the same on every car of the set: 0.318 m from where the tyre meets the ground to the axle. */
const WHEEL_RADIUS = 0.318;
/** Vertices within this height of the lowest are where the tyres touch the ground. */
const CONTACT = 0.005;
/**
 * In a ride-height change the wheels stay where they are. A vertex within the tyre's radius plus WHEEL_GAP[0] of an
 * axle, across the tyres' width, stays put; the body moves fully from WHEEL_GAP[1] beyond the tyre's radius, short of
 * the wheel arch, and from TYRE_SIDE[1] to either side of the tyre. Between, it follows in part.
 */
const WHEEL_GAP = [0.01, 0.12];
const TYRE_SIDE = [0.03, 0.13];
/** Bounds on a change: beyond them the surface would describe another car rather than a variant of this one. */
const LIMITS = { percent: 20, offset_mm: 200, falloff_mm: 1000 };
const FALLOFF_MM = 100;
const PREDICTED_ON = 'Each variant is predicted on the simplified working mesh, about 600,000 triangles, and so is its car, unchanged: differences between them are the change alone.';

const CHANGE_SCHEMA = {
	type: 'object',
	description: 'One change. stretch: the car scaled along an axis by percent: x (length) about its middle, y (width) about its centre plane, z (height) from the ground up. ride_height: the body moved up (positive) or down by offset_mm, the wheels staying on the ground. move: the faces of a region (region_id, or the selection when it is left out) moved by offset_mm along an axis (x downstream, z up, y away from the centre plane on each side), the surface around them following over falloff_mm measured along the surface.',
	properties: {
		kind: { type: 'string', enum: KINDS },
		axis: { type: 'string', enum: AXES, description: 'stretch and move.' },
		percent: { type: 'number', description: `stretch: the change in size, %, from −${LIMITS.percent} to ${LIMITS.percent}.` },
		offset_mm: { type: 'number', description: `ride_height and move: how far, mm, from −${LIMITS.offset_mm} to ${LIMITS.offset_mm}.` },
		region_id: { type: 'integer', description: 'move: a saved region; default the selection.' },
		falloff_mm: { type: 'number', description: `move: how far along the surface the faces around the region follow it, mm; default ${FALLOFF_MM}.` },
	},
	required: ['kind'],
	additionalProperties: false,
};

/**
 * Case x as it is compared with case y. A car beside one of its variants is its surface predicted unchanged the way
 * the variant was, so the two differ by the change alone; any other case is itself.
 */
export function comparedAs(x, y) {
	return x.baseline && y?.variant?.root === x ? { ...x, result: x.baseline } : x;
}

/**
 * Whether two cases have the same vertices: a car and its variants, all made from its simplified mesh. They are
 * compared vertex for vertex, each point of the body against itself before the change, where the nearest point would
 * be another part of the body once it has moved.
 */
export function sameVertices(x, y) {
	const root = (c) => (c.variant?.root ?? c).id;
	return Boolean(x.variant || y.variant) && root(x) === root(y);
}

/** The name a case goes by beside its car: a variant's own name without the car's at its start ("length +5 %"). */
export function shortName(c) {
	const stem = c.variant ? `${stemOf(c.variant.root)}, ` : null;
	return stem && c.name.startsWith(stem) ? c.name.slice(stem.length) : c.name;
}

const stemOf = (c) => c.name.replace(/\.vtp$/i, '');

// ---- changing a surface ---------------------------------------------------------------------------------------------

/** 1 up to `from`, 0 from `to`, and a half cosine between: how much of a change a point takes as it lies farther out. */
function fade(x, from, to) {
	if (x <= from) return 1;
	if (x >= to) return 0;
	return 0.5 * (1 + Math.cos((Math.PI * (x - from)) / (to - from)));
}

/** Scale the points along an axis by `factor`: the length about the car's middle, the width about its centre plane, the height from the ground up. */
function stretch(points, axis, factor) {
	const k = AXES.indexOf(axis);
	const bounds = boundsOf(points);
	const origin = [(bounds[0] + bounds[1]) / 2, 0, bounds[4]][k];
	for (let i = k; i < points.length; i += 3) points[i] = origin + (points[i] - origin) * factor;
}

/**
 * Move the body up by `offset` (m, down when negative) and leave the wheels. The tyres touch the ground in four
 * patches; the axles lie a wheel radius above the middles of the front and the rear pair.
 */
function rideHeight(points, offset) {
	const count = points.length / 3;
	let ground = Infinity;
	for (let v = 0; v < count; v++) ground = Math.min(ground, points[3 * v + 2]);
	const touching = [];
	for (let v = 0; v < count; v++) if (points[3 * v + 2] < ground + CONTACT) touching.push(v);
	let first = Infinity, last = -Infinity, inner = Infinity, outer = -Infinity;
	for (const v of touching) {
		first = Math.min(first, points[3 * v]);
		last = Math.max(last, points[3 * v]);
		inner = Math.min(inner, Math.abs(points[3 * v + 1]));
		outer = Math.max(outer, Math.abs(points[3 * v + 1]));
	}
	const middle = (first + last) / 2;
	const mean = (vs) => vs.reduce((sum, v) => sum + points[3 * v], 0) / vs.length;
	const front = touching.filter((v) => points[3 * v] < middle), rear = touching.filter((v) => points[3 * v] >= middle);
	if (!front.length || !rear.length) throw new Error('The tyres\' contact with the ground was not found, so the body cannot be told from the wheels.');
	const axles = [mean(front), mean(rear)];
	const axleHeight = ground + WHEEL_RADIUS;
	for (let v = 0; v < count; v++) {
		const x = points[3 * v], y = Math.abs(points[3 * v + 1]), z = points[3 * v + 2];
		const r = Math.min(Math.hypot(x - axles[0], z - axleHeight), Math.hypot(x - axles[1], z - axleHeight));
		const across = y < inner ? inner - y : y > outer ? y - outer : 0;
		const wheel = fade(r, WHEEL_RADIUS + WHEEL_GAP[0], WHEEL_RADIUS + WHEEL_GAP[1]) * fade(across, TYRE_SIDE[0], TYRE_SIDE[1]);
		points[3 * v + 2] += offset * (1 - wheel);
	}
}

/**
 * Move the vertices of the faces in `mask` by `offset` (m) along an axis, along y away from the centre plane on each
 * side, and the vertices around them by less the farther along the surface they lie, to nothing at `falloff` (m).
 */
function moveRegion(points, indices, mask, axis, offset, falloff) {
	const inside = new Uint8Array(points.length / 3);
	for (let f = 0; f < mask.length; f++) {
		if (mask[f]) inside[indices[3 * f]] = inside[indices[3 * f + 1]] = inside[indices[3 * f + 2]] = 1;
	}
	const distance = surfaceDistances(points, indices, inside, falloff);
	const k = AXES.indexOf(axis);
	for (let v = 0; v < inside.length; v++) {
		const weight = fade(distance[v], 0, falloff);
		if (weight === 0) continue;
		points[3 * v + k] += (k === 1 ? Math.sign(points[3 * v + 1]) : 1) * offset * weight;
	}
}

/**
 * Each vertex's distance along the surface from the nearest vertex of `sources`, up to `limit`, and Infinity beyond:
 * Dijkstra over the mesh's edges. Along the surface rather than straight, so a moved bumper does not drag a wheel or
 * a panel on the far side of a gap.
 */
function surfaceDistances(points, indices, sources, limit) {
	const count = sources.length;
	// Each vertex's neighbours, compressed: the edges of every triangle, both ways.
	const start = new Uint32Array(count + 1);
	for (const v of indices) start[v + 1] += 2;
	for (let v = 0; v < count; v++) start[v + 1] += start[v];
	const neighbours = new Uint32Array(start[count]);
	const filled = start.slice(0, count);
	for (let t = 0; t < indices.length; t += 3) {
		const a = indices[t], b = indices[t + 1], c = indices[t + 2];
		neighbours[filled[a]++] = b;
		neighbours[filled[a]++] = c;
		neighbours[filled[b]++] = c;
		neighbours[filled[b]++] = a;
		neighbours[filled[c]++] = a;
		neighbours[filled[c]++] = b;
	}
	const distance = new Float32Array(count).fill(Infinity);
	const frontier = new Frontier();
	for (let v = 0; v < count; v++) {
		if (!sources[v]) continue;
		distance[v] = 0;
		frontier.push(0, v);
	}
	while (frontier.size) {
		const [d, v] = frontier.pop();
		if (d > distance[v]) continue;
		for (let k = start[v]; k < start[v + 1]; k++) {
			const u = neighbours[k];
			const next = d + Math.hypot(points[3 * u] - points[3 * v], points[3 * u + 1] - points[3 * v + 1], points[3 * u + 2] - points[3 * v + 2]);
			if (next < distance[u] && next < limit) {
				distance[u] = next;
				frontier.push(next, u);
			}
		}
	}
	return distance;
}

/** A binary min-heap of vertices by distance. A vertex may be in it twice; the nearer entry comes out first. */
class Frontier {
	keys = [];
	ids = [];

	get size() {
		return this.keys.length;
	}

	push(key, id) {
		const { keys, ids } = this;
		let i = keys.length;
		keys.push(key);
		ids.push(id);
		while (i > 0) {
			const parent = (i - 1) >> 1;
			if (keys[parent] <= key) break;
			keys[i] = keys[parent];
			ids[i] = ids[parent];
			i = parent;
		}
		keys[i] = key;
		ids[i] = id;
	}

	pop() {
		const { keys, ids } = this;
		const top = [keys[0], ids[0]];
		const key = keys.pop(), id = ids.pop();
		if (keys.length) {
			let i = 0;
			for (;;) {
				let child = 2 * i + 1;
				if (child >= keys.length) break;
				if (child + 1 < keys.length && keys[child + 1] < keys[child]) child++;
				if (keys[child] >= key) break;
				keys[i] = keys[child];
				ids[i] = ids[child];
				i = child;
			}
			keys[i] = key;
			ids[i] = id;
		}
		return top;
	}
}

/** A surface for the engine from points and the triangles of a mesh, each array its own buffer, as the engine takes them. */
function surfaceOf(name, points, indices) {
	return { name, points, connectivity: Uint32Array.from(indices), offsets: Uint32Array.from({ length: indices.length / 3 + 1 }, (_, i) => 3 * i) };
}

// ---- describing changes and studies ---------------------------------------------------------------------------------

const plus = (x) => `${x > 0 ? '+' : ''}${signed(x, Number.isInteger(x) ? 0 : 1)}`;

/** A change in a few words, as variant names and the study table give it. */
function describe(change) {
	switch (change.kind) {
		case 'stretch':
			return `${DIMENSIONS[change.axis]} ${plus(change.percent)} %`;
		case 'ride_height':
			return `ride height ${plus(change.offset_mm)} mm`;
		case 'move':
			return `${change.region} ${plus(change.offset_mm)} mm in ${change.axis}`;
		default:
			throw new Error(`Unknown change ${change.kind}`);
	}
}

/** A Result's whole-car numbers, as the study compares them. */
const numbers = (r) => ({ cx: r.totals.coefficients[0], cy: r.totals.coefficients[1], cz: r.totals.coefficients[2], mean_cp: r.totals.meanCp, frontal_area_m2: r.geometry.frontalArea });
const DIFFERENCES = ['cx', 'cy', 'cz', 'mean_cp'];
const rounded = (values) => Object.fromEntries(Object.entries(values).map(([key, value]) => [key, round(value, 5)]));

// ---- the operations and the tab ---------------------------------------------------------------------------------------

/** app: { session, engine, workspace, regions, update, run }. The variant operations, the study export and the Variants tab. */
export function createVariants({ session, engine, workspace, regions, update, run }) {
	const view = session.view;
	/** Per car, its unchanged prediction while it is being made, so it is asked for once. */
	const baselines = new WeakMap();

	function baselineOf(root) {
		if (root.baseline) return root.baseline;
		if (!baselines.has(root)) {
			const { mesh } = root.result;
			baselines.set(root, engine.predictQuick(surfaceOf(root.name, Float32Array.from(mesh.positions), mesh.indices)).then(
				(result) => (root.baseline = result),
				(error) => {
					baselines.delete(root);
					throw error;
				},
			));
		}
		return baselines.get(root);
	}

	/** A change from the operation's arguments, checked, with the faces of a move's region picked on `base`. */
	function checkChange(change, i, base) {
		const name = `changes[${i}]`;
		const kind = oneOf(`${name}.kind`, change?.kind, KINDS);
		const within = (key, low, high) => {
			const value = finite(`${name}.${key}`, change[key]);
			if (value < low || value > high) throw new Error(`${name}.${key} must lie from ${low} to ${high}; got ${value}.`);
			return value;
		};
		switch (kind) {
			case 'stretch':
				return { kind, axis: oneOf(`${name}.axis`, change.axis, AXES), percent: within('percent', -LIMITS.percent, LIMITS.percent) };
			case 'ride_height':
				return { kind, offset_mm: within('offset_mm', -LIMITS.offset_mm, LIMITS.offset_mm) };
			case 'move': {
				const saved = session.regions.map((r) => `${r.id} (${r.name})`).join(', ') || 'none';
				const region = change.region_id === undefined ? null : session.regions.find((r) => r.id === change.region_id);
				if (change.region_id !== undefined && !region) throw new Error(`There is no saved region ${change.region_id}. Saved regions: ${saved}.`);
				const terms = region ? [{ op: 'replace', rule: { kind: 'region', region_id: region.id } }] : view.selection?.terms;
				if (!terms) throw new Error(`${name} moves a region: select faces first, or give region_id. Saved regions: ${saved}.`);
				const mask = regions.maskOf(base, terms);
				const faces = mask.reduce((sum, m) => sum + m, 0);
				if (!faces) throw new Error(`The ${region ? `region ${region.name}` : 'selection'} picks no faces of ${base.name}.`);
				const falloff = change.falloff_mm === undefined ? FALLOFF_MM : within('falloff_mm', 0, LIMITS.falloff_mm);
				return { kind, region: region?.name ?? 'selection', ...(region ? { region_id: region.id } : {}), faces, axis: oneOf(`${name}.axis`, change.axis, AXES), offset_mm: within('offset_mm', -LIMITS.offset_mm, LIMITS.offset_mm), falloff_mm: falloff, mask };
			}
			default:
				throw new Error(`Unknown change ${kind}`);
		}
	}

	/** Change `points` as `change` says; what the study keeps of the change. */
	function apply({ mask, ...change }, points, base) {
		if (change.kind === 'stretch') stretch(points, change.axis, 1 + change.percent / 100);
		else if (change.kind === 'ride_height') rideHeight(points, change.offset_mm / 1000);
		else moveRegion(points, base.result.mesh.indices, mask, change.axis, change.offset_mm / 1000, change.falloff_mm / 1000);
		return change;
	}

	/** A name no open case and no file under variants/ has yet: `wanted`, or it with a number after it. */
	function unique(wanted) {
		const taken = (name) => session.cases.some((c) => c.name === name) || workspace.file(`variants/${name}.vtp`);
		let name = wanted.replaceAll('/', '-');
		for (let n = 2; taken(name); n++) name = `${wanted.replaceAll('/', '-')} (${n})`;
		return name;
	}

	async function makeVariant({ case_id, name, changes }, by) {
		const base = case_id === undefined ? session.active : session.byId(case_id);
		if (!base) throw new Error('No case is on screen. Open a file or an example first.');
		if (base.status !== 'ready') throw new Error(`Case ${base.id} (${base.name}) is ${base.status}; a variant is made from a result.`);
		if (!Array.isArray(changes) || changes.length === 0) throw new Error('changes must list at least one change, e.g. [{"kind": "stretch", "axis": "x", "percent": 5}].');
		const checked = changes.map((change, i) => checkChange(change, i, base));
		const root = base.variant?.root ?? base;
		const { mesh } = base.result;
		const points = Float32Array.from(mesh.positions);
		const made = [...(base.variant?.changes ?? []), ...checked.map((change) => apply(change, points, base))];
		const label = unique(name?.trim() || `${stemOf(root)}, ${made.map(describe).join(', ')}`);
		const [result] = await Promise.all([engine.predictQuick(surfaceOf(label, points, mesh.indices)), baselineOf(root)]);
		const path = `variants/${label}.vtp`;
		const { cells } = result;
		workspace.write(path, writeVtp({ name: `${label}.vtp`, points: cells.points, connectivity: cells.connectivity, offsets: cells.offsets }, { CpPredicted: cells.cp, CpSpread: cells.sigma }), by);
		const c = session.openVariant(result, { name: label, root, changes: made, path });
		workspace.link(path, c.id);
		// Compared with its car from the start: the difference map and the table are one choice away.
		if (session.cases.includes(root)) view.compare = { with: root.id, sideBySide: view.compare?.sideBySide ?? false };
		update('toolbar', 'legend', 'viewer', 'dock');
		return { ...rowOut(c, root), compared_with: root.id };
	}

	/** A car and its open variants, for any case of the study. */
	function studyOf(c) {
		const root = c.variant?.root ?? c;
		return { root, variants: session.cases.filter((x) => x.variant?.root === root) };
	}

	function rowOut(v, root) {
		const mine = numbers(v.result), base = numbers(root.baseline);
		return {
			case_id: v.id,
			name: v.name,
			path: v.path,
			changes: v.variant.changes,
			...rounded(mine),
			difference: rounded(Object.fromEntries(DIFFERENCES.map((key) => [key, mine[key] - base[key]]))),
		};
	}

	function studyOut(c) {
		const { root, variants } = studyOf(c);
		return {
			car: { case_id: root.id, name: root.name, ...(root.baseline ? { predicted_unchanged: rounded(numbers(root.baseline)) } : {}) },
			variants: variants.map((v) => rowOut(v, root)),
			predicted_on: PREDICTED_ON,
		};
	}

	function studyCsv(c) {
		const { root, variants } = studyOf(c);
		const keys = ['cx', 'cy', 'cz', 'mean_cp', 'frontal_area_m2'];
		const line = (name, changes, values, difference) => [`"${name.replaceAll('"', '""')}"`, `"${changes}"`, ...keys.map((k) => values[k].toFixed(5)), ...DIFFERENCES.map((k) => (difference ? difference[k].toFixed(5) : ''))].join(',');
		const base = numbers(root.baseline);
		return [
			`# ${root.name} and its variants. ${PREDICTED_ON}`,
			['case', 'changes', ...keys, ...DIFFERENCES.map((k) => `${k}_difference`)].join(','),
			line(root.name, 'unchanged', base, null),
			...variants.map((v) => {
				const mine = numbers(v.result);
				return line(v.name, v.variant.changes.map(describe).join('; '), mine, Object.fromEntries(DIFFERENCES.map((k) => [k, mine[k] - base[k]])));
			}),
		].join('\n') + '\n';
	}

	defineOperation({
		name: 'make_variant',
		description: 'Make a variant of a car and predict it: its surface changed (stretched along an axis, the ride height raised or lowered, or a region moved with the surface around it following) and predicted on the simplified working mesh in a second or two. It opens as a case beside its car and is compared with it at once (set_field "difference" shows where the pressure changed); its surface is saved under variants/ in the workspace. A variant of a variant adds its changes to the first one\'s. Returns its Cx, Cy, Cz, mean Cp and frontal area, and their differences from the car predicted unchanged on the same mesh, so the differences are the change alone. For a study, call it once per variant, then read_study.',
		parameters: {
			type: 'object',
			properties: {
				case_id: { type: 'integer', description: 'The car to change; default the case on screen.' },
				name: { type: 'string', description: 'Default: the car\'s name and the changes.' },
				changes: { type: 'array', items: CHANGE_SCHEMA, minItems: 1, description: 'Applied in order.' },
			},
			required: ['changes'],
			additionalProperties: false,
		},
		run: (args, context) => makeVariant(args, context.by),
	});

	defineOperation({
		name: 'read_study',
		description: 'The study of a car: the car predicted unchanged on its simplified mesh, as its variants are, and each open variant with its changes, Cx, Cy, Cz, mean Cp and frontal area, and their differences from the unchanged car. The same table the Variants tab shows, whoever made the variants. Give any case of the study; default the case on screen.',
		parameters: { type: 'object', properties: { case_id: { type: 'integer', description: 'Default the case on screen.' } }, additionalProperties: false },
		run({ case_id }) {
			const c = case_id === undefined ? session.active : session.byId(case_id);
			if (!c?.result) throw new Error('No case with a result is on screen. Open a file or an example first.');
			return studyOut(c);
		},
	});

	defineExport({
		name: 'study',
		label: 'Study',
		format: 'CSV',
		title: 'The study table: the car predicted unchanged and each variant, with their differences',
		unavailable: (c) => (studyOf(c).variants.length ? null : 'This car has no variants open. Make one in the Variants tab.'),
		make: async (c) => new Blob([studyCsv(c)], { type: 'text/csv' }),
		fileName: (c) => `${baseName(studyOf(c).root)}-study.csv`,
	});

	// ---- the Variants tab ----

	let kind = 'stretch';
	let stretchAxis = 'x';
	let moveAxis = 'z';
	/** The saved region a move takes, or null for the selection. */
	let regionId = null;
	let making = false;

	const segmented = (label, options, chosen, choose) => {
		const buttons = options.map(([value, text, title]) => h('button', { type: 'button', role: 'radio', class: 'segment', title, 'data-value': value, onclick: () => { choose(value); render(); } }, text));
		const el = h('div', { class: 'segmented', role: 'radiogroup', 'aria-label': label }, buttons);
		return { el, update: () => buttons.forEach((b) => b.setAttribute('aria-checked', String(b.dataset.value === chosen()))) };
	};
	const kinds = segmented('Change', KINDS.map((k) => [k, KIND_LABELS[k]]), () => kind, (k) => (kind = k));
	const stretchAxes = segmented('Along', AXES.map((a) => [a, DIMENSIONS[a][0].toUpperCase() + DIMENSIONS[a].slice(1)]), () => stretchAxis, (a) => (stretchAxis = a));
	const moveAxes = segmented('Along', AXES.map((a) => [a, a, DIRECTIONS[a]]), () => moveAxis, (a) => (moveAxis = a));
	const number = (value, label) => h('input', { class: 'number', type: 'text', inputmode: 'decimal', value, 'aria-label': label, spellcheck: 'false' });
	const unit = (text) => h('span', { class: 'unit' }, text);
	const percent = number('5', 'Change in size, %');
	const height = number('-20', 'Ride height change, mm');
	const distance = number('20', 'Distance, mm');
	const falloff = number(String(FALLOFF_MM), 'Falloff along the surface, mm');

	const regionList = h('div', { class: 'menu-list', role: 'menu' });
	const regionButton = h('button', { type: 'button', class: 'tool' });
	const regionMenu = menu(regionButton, regionList);

	const parts = {
		stretch: h('span', { class: 'variants-part' }, stretchAxes.el, percent, unit('%')),
		ride_height: h('span', { class: 'variants-part' }, height, unit('mm')),
		move: h('span', { class: 'variants-part' }, regionMenu.el, moveAxes.el, distance, unit('mm'), h('span', { class: 'variants-label' }, 'falloff'), falloff, unit('mm')),
	};
	const make = h('button', { type: 'submit', class: 'button' }, 'Make variant');
	const form = h('form', { class: 'variants-form', onsubmit: (e) => submit(e) }, kinds.el, ...Object.values(parts), make);
	const problem = h('p', { class: 'variants-problem', role: 'alert', hidden: true });
	const table = h('table', { class: 'variants-table' });
	const basis = h('p', { class: 'variants-basis' }, 'Variants are predicted on the simplified mesh of 600,000 triangles, and compared with their car predicted unchanged on it.');
	const el = h('div', { class: 'variants' }, form, problem, table, basis);
	const csv = h('button', { type: 'button', class: 'tool', onclick: () => run('download_export', { export: 'study' }).catch(() => {}) }, icon('download'), 'CSV');
	const controls = h('div', { class: 'variants-controls' }, csv);

	async function submit(e) {
		e.preventDefault();
		problem.hidden = true;
		const value = (input) => Number(input.value.trim().replace('\u2212', '-'));
		const change = kind === 'stretch' ? { kind, axis: stretchAxis, percent: value(percent) }
			: kind === 'ride_height' ? { kind, offset_mm: value(height) }
				: { kind, axis: moveAxis, offset_mm: value(distance), falloff_mm: value(falloff), ...(regionId !== null ? { region_id: regionId } : {}) };
		making = true;
		render();
		try {
			await run('make_variant', { changes: [change] });
		} catch (error) {
			problem.textContent = error.message;
			problem.hidden = false;
		} finally {
			making = false;
			render();
		}
	}

	function render() {
		const c = session.active;
		for (const control of [kinds, stretchAxes, moveAxes]) control.update();
		for (const [k, part] of Object.entries(parts)) part.hidden = k !== kind;
		if (regionId !== null && !session.regions.some((r) => r.id === regionId)) regionId = null;
		regionButton.replaceChildren(regionId === null ? 'Selection' : session.regions.find((r) => r.id === regionId).name, icon('chevron'));
		const pick = (id) => () => {
			regionId = id;
			regionMenu.close();
			render();
		};
		regionList.replaceChildren(
			h('button', { type: 'button', role: 'menuitem', class: 'menu-item', 'aria-checked': String(regionId === null), onclick: pick(null) }, 'Selection'),
			...session.regions.map((r) => h('button', { type: 'button', role: 'menuitem', class: 'menu-item', 'aria-checked': String(regionId === r.id), onclick: pick(r.id) }, r.name)));
		make.disabled = making || !c?.result;
		make.textContent = making ? 'Predicting…' : 'Make variant';
		const { root, variants } = c?.result ? studyOf(c) : { root: null, variants: [] };
		table.hidden = !variants.length || !root.baseline;
		csv.disabled = table.hidden;
		if (table.hidden) return table.replaceChildren();
		const base = numbers(root.baseline);
		const row = (x, label, values, title) => h('tr', { class: x === c ? 'current' : '', title, onclick: () => run('show_case', { case_id: x.id }).catch(() => {}) },
			h('td', { class: 'variants-name' }, label),
			...['cx', 'cz', 'mean_cp'].flatMap((key) => [h('td', {}, signed(values[key], 3)), h('td', { class: 'variants-delta' }, x === root ? '' : signed(values[key] - base[key], 3))]),
			h('td', {}, values.frontal_area_m2.toFixed(3)));
		table.replaceChildren(
			h('thead', {}, h('tr', {}, ...['Variant', 'Cx', 'ΔCx', 'Cz', 'ΔCz', 'Mean Cp', 'ΔMean Cp', 'Frontal area, m²'].map((text) => h('th', {}, text)))),
			h('tbody', {},
				row(root, 'Unchanged', base, `${root.name}, predicted unchanged on the simplified mesh`),
				...variants.map((v) => row(v, shortName(v), numbers(v.result), v.name))));
	}

	const count = () => (session.active?.result ? studyOf(session.active).variants.length : 0);

	return { panel: { el, controls, render, count, stop: () => {} } };
}
