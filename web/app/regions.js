// Regions of the car: the faces an engineer picks out to read the pressure on them. A selection is a list of rules,
// each replacing, adding to, taking from or intersecting what came before: faces facing a direction, faces whose value
// lies in a range, faces inside a slab along an axis, faces near a patch painted or lassoed on the car, or a saved
// region. Rules rather than lists of faces, so a region picked on one car picks the same part of another.
//
// The selection is part of the shared view: tinted on the car and read out on the result card. Saved regions belong
// to the session. Readouts sum a dropped file's own polygons (`engine.fileCells`), so they agree with the whole car's
// totals; a bundled example carries only its simplified working mesh (`engine.workingCells`), and is summed on that.
// The tint and the histogram run on the working mesh, as drawn.

import { caseSummary } from './cases.js';
import { finite, oneOf, round, triple } from './checks.js';
import { h } from './dom.js';
import { signed } from './format.js';
import { icon } from './icons.js';
import { menu } from './menu.js';
import { defineOperation } from './operations.js';

/** Directions a face can face, by the sign of one component of its outward normal. */
export const FACINGS = {
	forward: { label: 'Forward-facing', axis: 0, sign: -1, title: 'Faces whose outward normal points upstream (nₓ < 0)' },
	rearward: { label: 'Rearward-facing', axis: 0, sign: 1, title: 'Faces whose outward normal points downstream (nₓ > 0)' },
	upward: { label: 'Upward-facing', axis: 2, sign: 1, title: 'Faces whose outward normal points up (n_z > 0)' },
	downward: { label: 'Downward-facing', axis: 2, sign: -1, title: 'Faces whose outward normal points down (n_z < 0)' },
	'+y': { label: 'Facing +y', axis: 1, sign: 1, title: 'Faces whose outward normal points toward +y' },
	'-y': { label: 'Facing −y', axis: 1, sign: -1, title: 'Faces whose outward normal points toward −y' },
};
const OPS = ['replace', 'add', 'subtract', 'intersect'];
const OP_LABELS = { replace: 'Replace', add: 'Add', subtract: 'Subtract', intersect: 'Intersect' };
/** How a step combines, beside it in the list of the selection's steps. */
const OP_SIGNS = { replace: '', add: '+', subtract: '−', intersect: '∩' };
const AXES = ['x', 'y', 'z'];
const VALUE_FIELDS = ['cp', 'drag', 'sigma'];
/** Regions may name regions; deeper than this is a loop. */
const MAX_DEPTH = 8;
/** Brush radius for painting, in CSS pixels at the first dab. */
const BRUSH_PX = 14;
/**
 * A lasso's points lie about LASSO_STEP drawing-buffer pixels apart at the camera's pivot, farther apart for a lasso
 * wide enough that this would give more than about LASSO_POINTS.
 */
const LASSO_STEP = 3;
const LASSO_POINTS = 20000;

// ---- rules ----------------------------------------------------------------------------------------------------------

/**
 * Points of a patch hashed by cubes the size of its radius, to find one within the radius of a place. Hash
 * collisions only add candidates; each is tested by distance.
 */
class PatchGrid {
	constructor(radius) {
		this.r2 = radius * radius;
		this.size = radius;
		this.cells = new Map();
		this.box = [Infinity, -Infinity, Infinity, -Infinity, Infinity, -Infinity];
	}

	key(i, j, k) {
		return (i * 73856093) ^ (j * 19349663) ^ (k * 83492791);
	}

	add(p) {
		const s = this.size;
		const key = this.key(Math.floor(p[0] / s), Math.floor(p[1] / s), Math.floor(p[2] / s));
		(this.cells.get(key) ?? this.cells.set(key, []).get(key)).push(p);
		for (let k = 0; k < 3; k++) {
			this.box[2 * k] = Math.min(this.box[2 * k], p[k] - s);
			this.box[2 * k + 1] = Math.max(this.box[2 * k + 1], p[k] + s);
		}
	}

	/** A point of the patch within the radius of (x, y, z), or null. */
	near(x, y, z) {
		const b = this.box;
		if (x < b[0] || x > b[1] || y < b[2] || y > b[3] || z < b[4] || z > b[5]) return null;
		const s = this.size;
		const i = Math.floor(x / s), j = Math.floor(y / s), k = Math.floor(z / s);
		for (let di = -1; di <= 1; di++) {
			for (let dj = -1; dj <= 1; dj++) {
				for (let dk = -1; dk <= 1; dk++) {
					for (const p of this.cells.get(this.key(i + di, j + dj, k + dk)) ?? []) {
						if ((p[0] - x) ** 2 + (p[1] - y) ** 2 + (p[2] - z) ** 2 <= this.r2) return p;
					}
				}
			}
		}
		return null;
	}
}

/** Each face's value of a field: Cp, the drag density −Cp·nₓ, or σ. */
export function faceValues(cells, field) {
	if (field !== 'drag') return cells[field];
	const values = new Float32Array(cells.areas.length);
	for (let c = 0; c < values.length; c++) values[c] = -cells.cp[c] * cells.normals[3 * c];
	return values;
}

/** The faces of `cells` one rule picks, as 0 or 1 per face. */
function ruleMask(cells, rule, regions, depth) {
	const count = cells.areas.length;
	const mask = new Uint8Array(count);
	const { normals, centers } = cells;
	switch (rule.kind) {
		case 'facing': {
			const { axis, sign } = FACINGS[rule.facing];
			const least = Math.cos(((rule.within_deg ?? 90) * Math.PI) / 180);
			for (let c = 0; c < count; c++) mask[c] = sign * normals[3 * c + axis] > least ? 1 : 0;
			return mask;
		}
		case 'field': {
			const values = faceValues(cells, rule.field);
			const lo = rule.min ?? -Infinity, hi = rule.max ?? Infinity;
			for (let c = 0; c < count; c++) mask[c] = values[c] >= lo && values[c] <= hi ? 1 : 0;
			return mask;
		}
		case 'slab': {
			const axis = AXES.indexOf(rule.axis);
			const lo = rule.min_m ?? -Infinity, hi = rule.max_m ?? Infinity;
			for (let c = 0; c < count; c++) mask[c] = centers[3 * c + axis] >= lo && centers[3 * c + axis] <= hi ? 1 : 0;
			return mask;
		}
		case 'patch': {
			const grid = new PatchGrid(rule.radius_m);
			for (const p of rule.points_m) grid.add(p);
			for (let c = 0; c < count; c++) mask[c] = grid.near(centers[3 * c], centers[3 * c + 1], centers[3 * c + 2]) ? 1 : 0;
			return mask;
		}
		case 'region': {
			const region = regions.find((r) => r.id === rule.region_id);
			if (!region) throw new Error(`There is no saved region ${rule.region_id}. Saved regions: ${regions.map((r) => `${r.id} (${r.name})`).join(', ') || 'none'}.`);
			if (depth >= MAX_DEPTH) throw new Error(`Region ${region.id} names regions ${MAX_DEPTH} deep; one of them names itself.`);
			return maskOf(cells, region.terms, regions, depth + 1);
		}
		default:
			throw new Error(`Unknown rule ${rule.kind}`);
	}
}

/** The faces a list of terms picks, applied in order. An empty list picks nothing. */
export function maskOf(cells, terms, regions, depth = 0) {
	let mask = new Uint8Array(cells.areas.length);
	for (const { op, rule } of terms) {
		const picked = ruleMask(cells, rule, regions, depth);
		if (op === 'replace') mask = picked;
		else if (op === 'add') for (let c = 0; c < mask.length; c++) mask[c] |= picked[c];
		else if (op === 'subtract') for (let c = 0; c < mask.length; c++) mask[c] &= 1 - picked[c];
		else for (let c = 0; c < mask.length; c++) mask[c] &= picked[c];
	}
	return mask;
}

/**
 * Sums over the faces a mask picks (every face without one): area, area-weighted mean Cp, the pressure force
 * coefficients Cx, Cy, Cz = −Σ Cp n A / reference area, and where Cp is lowest and highest.
 */
export function measure(cells, mask, referenceArea) {
	const { areas, cp, normals, centers } = cells;
	let count = 0, area = 0, cpArea = 0, fx = 0, fy = 0, fz = 0, lo = -1, hi = -1;
	for (let c = 0; c < areas.length; c++) {
		if (mask && !mask[c]) continue;
		const a = areas[c];
		count++;
		area += a;
		cpArea += cp[c] * a;
		fx -= cp[c] * normals[3 * c] * a;
		fy -= cp[c] * normals[3 * c + 1] * a;
		fz -= cp[c] * normals[3 * c + 2] * a;
		if (lo < 0 || cp[c] < cp[lo]) lo = c;
		if (hi < 0 || cp[c] > cp[hi]) hi = c;
	}
	const at = (c) => (c < 0 ? null : { value: cp[c], center: [centers[3 * c], centers[3 * c + 1], centers[3 * c + 2]], normal: [normals[3 * c], normals[3 * c + 1], normals[3 * c + 2]] });
	return { faces: count, area, meanCp: area > 0 ? cpArea / area : null, coefficients: [fx, fy, fz].map((f) => f / referenceArea), cpMin: at(lo), cpMax: at(hi) };
}

/** Per working-mesh vertex, the share of the area of the faces around it that a mask picks: the tint's input. */
function vertexShares(mesh, cells, mask) {
	const { indices } = mesh;
	const count = mesh.positions.length / 3;
	const picked = new Float32Array(count);
	const around = new Float32Array(count);
	for (let t = 0; t < cells.areas.length; t++) {
		const a = cells.areas[t];
		for (let k = 0; k < 3; k++) {
			const v = indices[3 * t + k];
			around[v] += a;
			if (mask[t]) picked[v] += a;
		}
	}
	for (let v = 0; v < count; v++) picked[v] = around[v] > 0 ? picked[v] / around[v] : 0;
	return picked;
}

/** One line naming a rule, for the list of the selection's steps and for the assistant. */
export function describeRule(rule, regions) {
	switch (rule.kind) {
		case 'facing':
			return `${FACINGS[rule.facing].label}${rule.within_deg !== undefined && rule.within_deg !== 90 ? ` within ${rule.within_deg}°` : ''}`;
		case 'field': {
			const name = { cp: 'Cp', drag: '−Cp nₓ', sigma: 'σ' }[rule.field];
			if (rule.min !== undefined && rule.max !== undefined) return `${name} from ${signed(rule.min, 2)} to ${signed(rule.max, 2)}`;
			return rule.min !== undefined ? `${name} at least ${signed(rule.min, 2)}` : `${name} at most ${signed(rule.max, 2)}`;
		}
		case 'slab':
			return `${rule.axis} from ${rule.min_m ?? '−∞'} to ${rule.max_m ?? '∞'} m`;
		case 'patch':
			return `${rule.drawn === 'lasso' ? 'Lassoed' : 'Painted'} patch`;
		case 'region':
			return regions.find((r) => r.id === rule.region_id)?.name ?? `Region ${rule.region_id}`;
		default:
			throw new Error(`Unknown rule ${rule.kind}`);
	}
}

// ---- checking terms from an operation -------------------------------------------------------------------------------

const RULE_SCHEMA = {
	type: 'object',
	description: 'One rule. facing: faces whose outward normal points a way (forward = upstream, nₓ < 0), optionally within_deg of it (default 90, a hemisphere). field: faces whose cp, drag (−Cp·nₓ) or sigma lies between min and max (either may be left out). slab: faces whose centre lies between min_m and max_m along an axis. patch: faces within radius_m of any of points_m (painted or lassoed spots on the surface). region: a saved region by its id.',
	properties: {
		kind: { type: 'string', enum: ['facing', 'field', 'slab', 'patch', 'region'] },
		facing: { type: 'string', enum: Object.keys(FACINGS) },
		within_deg: { type: 'number' },
		field: { type: 'string', enum: VALUE_FIELDS },
		min: { type: 'number' },
		max: { type: 'number' },
		axis: { type: 'string', enum: AXES },
		min_m: { type: 'number' },
		max_m: { type: 'number' },
		points_m: { type: 'array', items: { type: 'array', items: { type: 'number' } } },
		radius_m: { type: 'number' },
		drawn: { type: 'string', enum: ['paint', 'lasso'], description: 'How a patch was made on screen, for its label.' },
		region_id: { type: 'integer' },
	},
	required: ['kind'],
	additionalProperties: false,
};

const TERMS_SCHEMA = {
	type: 'array',
	description: 'Steps applied in order: each rule replaces what came before (replace), adds to it, subtracts from it or intersects it. The first step\'s op is taken as replace.',
	items: {
		type: 'object',
		properties: { op: { type: 'string', enum: OPS }, rule: RULE_SCHEMA },
		required: ['op', 'rule'],
		additionalProperties: false,
	},
};

/** A rule from an operation's arguments, checked, with only the keys its kind uses. */
function checkRule(rule, name, regions) {
	oneOf(`${name}.kind`, rule?.kind, ['facing', 'field', 'slab', 'patch', 'region']);
	switch (rule.kind) {
		case 'facing': {
			const within = rule.within_deg === undefined ? undefined : finite(`${name}.within_deg`, rule.within_deg);
			if (within !== undefined && (within <= 0 || within > 180)) throw new Error(`${name}.within_deg must be above 0 and at most 180; got ${within}.`);
			return { kind: 'facing', facing: oneOf(`${name}.facing`, rule.facing, Object.keys(FACINGS)), ...(within !== undefined ? { within_deg: within } : {}) };
		}
		case 'field': {
			if (rule.min === undefined && rule.max === undefined) throw new Error(`${name} needs min, max or both.`);
			return { kind: 'field', field: oneOf(`${name}.field`, rule.field ?? 'cp', VALUE_FIELDS), ...(rule.min !== undefined ? { min: finite(`${name}.min`, rule.min) } : {}), ...(rule.max !== undefined ? { max: finite(`${name}.max`, rule.max) } : {}) };
		}
		case 'slab': {
			if (rule.min_m === undefined && rule.max_m === undefined) throw new Error(`${name} needs min_m, max_m or both.`);
			return { kind: 'slab', axis: oneOf(`${name}.axis`, rule.axis, AXES), ...(rule.min_m !== undefined ? { min_m: finite(`${name}.min_m`, rule.min_m) } : {}), ...(rule.max_m !== undefined ? { max_m: finite(`${name}.max_m`, rule.max_m) } : {}) };
		}
		case 'patch': {
			if (!Array.isArray(rule.points_m) || rule.points_m.length === 0) throw new Error(`${name}.points_m must be a list of points [x, y, z]; got ${JSON.stringify(rule.points_m)}.`);
			const radius = finite(`${name}.radius_m`, rule.radius_m);
			if (radius <= 0) throw new Error(`${name}.radius_m must be above 0; got ${radius}.`);
			return { kind: 'patch', points_m: rule.points_m.map((p, i) => triple(`${name}.points_m[${i}]`, p)), radius_m: radius, ...(rule.drawn !== undefined ? { drawn: oneOf(`${name}.drawn`, rule.drawn, ['paint', 'lasso']) } : {}) };
		}
		case 'region': {
			if (!regions.some((r) => r.id === rule.region_id)) throw new Error(`There is no saved region ${rule.region_id}. Saved regions: ${regions.map((r) => `${r.id} (${r.name})`).join(', ') || 'none'}.`);
			return { kind: 'region', region_id: rule.region_id };
		}
		default:
			throw new Error(`Unknown rule ${rule.kind}`);
	}
}

/** Steps from an operation's arguments, checked. An op left out replaces; a selection starts as nothing. */
function checkTerms(terms, regions) {
	if (!Array.isArray(terms)) throw new Error(`terms must be a list of steps {op, rule}; got ${JSON.stringify(terms)}.`);
	return terms.map((t, i) => ({ op: oneOf(`terms[${i}].op`, t?.op ?? 'replace', OPS), rule: checkRule(t?.rule, `terms[${i}].rule`, regions) }));
}

// ---- the regions as the app holds them -------------------------------------------------------------------------------

/**
 * app: { session, engine, update }. Keeps the selection's faces and tint for the case on screen, and defines the
 * region operations. Returns what the card and the dock read.
 */
export function createRegions({ session, engine, update }) {
	const view = session.view;
	view.selection ??= null;
	session.regions ??= [];
	let nextId = 1;
	/**
	 * Per result: its working mesh's faces (`cells`), which masks for the tint and the histogram run over; the file's
	 * own polygons for sums (`file`), once fetched; and the selection's mask and readout while the selection holds.
	 */
	const perResult = new WeakMap();
	/** Steps tinted in place of the selection while a patch is being painted (preview), or null. */
	let previewTerms = null;

	const known = (result) => perResult.get(result) ?? perResult.set(result, { cells: engine.workingCells(result), file: null }).get(result);
	const cellsOf = (result) => known(result).cells;
	const withResult = (case_id) => {
		const c = case_id === undefined ? session.active : session.byId(case_id);
		if (!c?.result) throw new Error(c ? `Case ${c.id} has no result yet (${c.status}).` : 'No case is on screen. Open a file or an example first.');
		return c;
	};

	/**
	 * The cells a case's sums run over, and what operations call them: a dropped file's own polygons, fetched from the
	 * engine the first time they are wanted once every polygon is predicted (and again when a reference is attached),
	 * and kept with the result; until then, and for a bundled example, which carries only the working mesh, that
	 * mesh's faces. A variant was predicted on its faces themselves, so they are its own polygons.
	 */
	function sumBasis(c) {
		const held = known(c.result);
		if (c.result.form === 'quick') return { cells: held.cells, over: `every face it was predicted on (${held.cells.areas.length})`, exact: true };
		const { reference } = c.result;
		if (held.file?.cells && held.file.reference === reference) return held.file;
		if (c.kind === 'file' && held.file?.reference !== reference && engine.canExportVtp(c.result)) {
			held.file = { cells: null, reference };
			engine.fileCells(c.result).then((set) => {
				if (!set || held.file.reference !== reference) return;
				// Only what sums read is kept: the points and polygons themselves would double the memory.
				const { centers, normals, areas, cp, sigma, cfd } = set;
				held.file = { cells: { centers, normals, areas, cp, sigma, cfd }, over: `every polygon of the file (${areas.length})`, exact: true, reference };
				update('card', 'dock');
			});
		}
		return { cells: held.cells, over: `${held.cells.areas.length} faces of the simplified working mesh (Cp at each face the mean of its corners, as drawn)`, exact: false };
	}

	/** Masks and sums per set of cells, by the terms they were made for (a terms array is never changed in place). */
	const masks = new WeakMap();
	const sums = new WeakMap();
	const WHOLE = [];
	const kept = (cache, cells, key, make) => {
		const byKey = cache.get(cells) ?? cache.set(cells, new WeakMap()).get(cells);
		if (!byKey.has(key)) byKey.set(key, make());
		return byKey.get(key);
	};
	/** The reference field's cells in place of the prediction's, for summing the CFD over the same polygons. */
	const referenceViews = new WeakMap();
	const reference = (cells) => referenceViews.get(cells) ?? referenceViews.set(cells, { ...cells, cp: cells.cfd }).get(cells);

	/** The faces some terms pick on the cells a case's sums run over, as 0 or 1 each. */
	function maskOn(c, terms) {
		const { cells } = sumBasis(c);
		return kept(masks, cells, terms, () => maskOf(cells, terms, session.regions));
	}

	/**
	 * Sums over the faces some terms pick on a case (every face for null), on its sum basis, of its prediction or,
	 * with `cfd`, of its reference field: {part, over, exact}.
	 */
	function measureOn(c, terms, { cfd = false } = {}) {
		const basis = sumBasis(c);
		const mask = terms ? maskOn(c, terms) : null;
		const cells = cfd ? reference(basis.cells) : basis.cells;
		return { part: kept(sums, cells, terms ?? WHOLE, () => measure(cells, mask, c.result.geometry.frontalArea)), over: basis.over, exact: basis.exact };
	}

	/** The readout of some terms on a case, with the whole car summed over the same cells for shares. */
	function readout(c, terms) {
		const { part, over } = measureOn(c, terms);
		return { part, whole: measureOn(c, null).part, over };
	}

	/**
	 * The current selection on a case: its mask on the working mesh, for the tint, and its readout, kept until the
	 * selection, the case's result or the cells summed over change.
	 */
	function selectionOn(c) {
		if (!view.selection || !c?.result) return null;
		const held = known(c.result);
		const basis = sumBasis(c);
		if (held.terms !== view.selection.terms || held.over !== basis.over) {
			Object.assign(held, readout(c, view.selection.terms), { terms: view.selection.terms, mask: maskOf(held.cells, view.selection.terms, session.regions), shares: null });
		}
		return held;
	}

	/** The tint for the case on screen: per working-mesh vertex, the share of its area selected; null for none. */
	function tintOf(c) {
		if (previewTerms && c?.result) {
			const cells = cellsOf(c.result);
			return vertexShares(c.result.mesh, cells, maskOf(cells, previewTerms, session.regions));
		}
		const selection = selectionOn(c);
		if (!selection) return null;
		selection.shares ??= vertexShares(c.result.mesh, selection.cells, selection.mask);
		return selection.shares;
	}

	const report = (m, whole) => ({
		faces: m.faces,
		area_m2: round(m.area, 5),
		mean_cp: m.meanCp === null ? null : round(m.meanCp, 5),
		cx: round(m.coefficients[0], 5),
		cy: round(m.coefficients[1], 5),
		cz: round(m.coefficients[2], 5),
		...(whole ? { share_of_car: { area: round(m.area / whole.area, 4), cx: round(m.coefficients[0] / whole.coefficients[0], 4), cz: round(m.coefficients[2] / whole.coefficients[2], 4) } } : {}),
		cp_min: m.cpMin && { value: round(m.cpMin.value, 4), center_m: m.cpMin.center.map((v) => round(v, 4)) },
		cp_max: m.cpMax && { value: round(m.cpMax.value, 4), center_m: m.cpMax.center.map((v) => round(v, 4)) },
	});
	/** The whole car as the result card gives it: the Result's totals, and what they were summed over. */
	const wholeCar = (c) => {
		const s = caseSummary(c, null);
		return { area_m2: s.area_m2, mean_cp: s.mean_cp, cx: s.cx, cy: s.cy, cz: s.cz, summed_over: s.totals_over };
	};
	/** Steps as an operation returns them: each labelled, a patch by its point count rather than its points. */
	const stepsOut = (terms) => terms.map(({ op, rule }) => ({
		op,
		label: describeRule(rule, session.regions),
		rule: rule.kind === 'patch' ? { kind: 'patch', points: rule.points_m.length, radius_m: rule.radius_m, drawn: rule.drawn } : rule,
	}));

	/** Set or extend the selection; every control and the assistant come through here. */
	function select(terms, mode) {
		const steps = mode === 'extend' && view.selection ? [...view.selection.terms, ...terms] : terms;
		view.selection = steps.length ? { terms: steps } : null;
		update('viewer', 'card', 'dock');
	}

	/**
	 * Tint a selection on the car without making it the selection: the patch being painted, shown as it grows. At
	 * most once a frame, however fast the dabs come; null goes back to the selection.
	 */
	let pending = null;
	function preview(terms) {
		previewTerms = terms;
		pending ??= requestAnimationFrame(() => {
			pending = null;
			update('viewer');
		});
	}

	defineOperation({
		name: 'select_region',
		description: 'Select faces of the car by rules, tinting them on the car and reading them out on the result card: area, mean Cp, the pressure force coefficients Cx, Cy, Cz and their share of the whole car, and where Cp is lowest and highest. The selection follows the view, so it picks the same part of every car shown. mode set (default) replaces the selection with these steps; extend appends them to it. Returns the readout on the case on screen.',
		parameters: {
			type: 'object',
			properties: { terms: TERMS_SCHEMA, mode: { type: 'string', enum: ['set', 'extend'] } },
			required: ['terms'],
			additionalProperties: false,
		},
		run({ terms, mode = 'set' }) {
			oneOf('mode', mode, ['set', 'extend']);
			select(checkTerms(terms, session.regions), mode);
			const c = session.active;
			const s = selectionOn(c);
			return { selection: stepsOut(view.selection?.terms ?? []), ...(s ? { readout: report(s.part, s.whole), summed_over: s.over } : {}) };
		},
	});

	defineOperation({
		name: 'clear_selection',
		description: 'Clear the selection: the car is shown whole and the result card reads out the whole car.',
		parameters: { type: 'object', properties: {}, additionalProperties: false },
		run() {
			select([], 'set');
			return { selection: [] };
		},
	});

	defineOperation({
		name: 'measure_region',
		description: 'Read out faces of a case without changing the selection: a saved region (region_id), steps of rules (terms, as select_region takes), or the current selection (neither). Gives area, mean Cp, Cx, Cy, Cz, their share of the whole car summed over the same polygons (summed_over names them: a dropped file\'s own polygons, or a bundled example\'s simplified mesh), the lowest and highest Cp with where they are, and the whole car as the result card gives it.',
		parameters: {
			type: 'object',
			properties: { case_id: { type: 'integer', description: 'Default the case on screen.' }, region_id: { type: 'integer' }, terms: TERMS_SCHEMA },
			additionalProperties: false,
		},
		run({ case_id, region_id, terms }) {
			const c = withResult(case_id);
			const steps = namedSteps(region_id, terms) ?? view.selection?.terms;
			if (!steps) throw new Error('Nothing is selected. Give region_id or terms, or select first with select_region.');
			const { part, whole, over } = readout(c, steps);
			return { case_id: c.id, readout: report(part, whole), summed_over: over, whole_car: wholeCar(c) };
		},
	});

	/** The steps an operation's arguments name: a saved region, or terms, checked; undefined when they name neither. */
	function namedSteps(region_id, terms) {
		if (region_id !== undefined && terms !== undefined) throw new Error('Give region_id or terms, not both.');
		if (region_id !== undefined) return [{ op: 'replace', rule: checkRule({ kind: 'region', region_id }, 'region', session.regions) }];
		if (terms !== undefined) return checkTerms(terms, session.regions);
		return undefined;
	}

	defineOperation({
		name: 'save_region',
		description: 'Save the current selection as a named region, so it can be selected again, measured on any car and used as a rule in other selections. Returns its region_id.',
		parameters: { type: 'object', properties: { name: { type: 'string' } }, required: ['name'], additionalProperties: false },
		run({ name }) {
			if (!view.selection) throw new Error('Nothing is selected to save. Select faces first with select_region.');
			const label = String(name).trim();
			if (!label) throw new Error('name must not be empty.');
			if (session.regions.some((r) => r.name === label)) throw new Error(`A region is already named ${JSON.stringify(label)}. Choose another name.`);
			const region = { id: nextId++, name: label, terms: view.selection.terms };
			session.regions.push(region);
			update('dock');
			return { region_id: region.id, name: region.name };
		},
	});

	defineOperation({
		name: 'list_regions',
		description: 'The saved regions: each with its region_id, name and steps, and its area and Cx share of the car on screen when one is.',
		parameters: { type: 'object', properties: {}, additionalProperties: false },
		run() {
			const c = session.active?.result ? session.active : null;
			return {
				regions: session.regions.map((r) => {
					const out = { region_id: r.id, name: r.name, terms: stepsOut(r.terms) };
					if (!c) return out;
					const { part, whole } = readout(c, r.terms);
					return { ...out, area_m2: round(part.area, 5), share_of_car_cx: round(part.coefficients[0] / whole.coefficients[0], 4) };
				}),
			};
		},
	});

	defineOperation({
		name: 'rename_region',
		description: 'Rename a saved region.',
		parameters: { type: 'object', properties: { region_id: { type: 'integer' }, name: { type: 'string' } }, required: ['region_id', 'name'], additionalProperties: false },
		run({ region_id, name }) {
			const region = session.regions.find((r) => r.id === region_id);
			if (!region) throw new Error(`There is no saved region ${region_id}.`);
			const label = String(name).trim();
			if (!label) throw new Error('name must not be empty.');
			if (session.regions.some((r) => r !== region && r.name === label)) throw new Error(`A region is already named ${JSON.stringify(label)}.`);
			region.name = label;
			update('dock', 'card');
			return { region_id, name: label };
		},
	});

	defineOperation({
		name: 'delete_region',
		description: 'Delete a saved region. A region another one names as a rule cannot be deleted until that one is.',
		parameters: { type: 'object', properties: { region_id: { type: 'integer' } }, required: ['region_id'], additionalProperties: false },
		run({ region_id }) {
			const i = session.regions.findIndex((r) => r.id === region_id);
			if (i < 0) throw new Error(`There is no saved region ${region_id}.`);
			const user = session.regions.find((r) => r.terms.some((t) => t.rule.kind === 'region' && t.rule.region_id === region_id));
			if (user) throw new Error(`Region ${user.id} (${user.name}) names this one as a rule; delete it first.`);
			const named = view.selection?.terms.some((t) => t.rule.kind === 'region' && t.rule.region_id === region_id);
			session.regions.splice(i, 1);
			if (named) select([], 'set');
			update('dock');
			return { deleted: region_id };
		},
	});

	return {
		selectionOn, tintOf, select, preview, cellsOf, namedSteps, measureOn, maskOn, sumBasis,
		/** The faces of a case some steps pick, as 0 or 1 per working-mesh face. */
		maskOf: (c, steps) => maskOf(cellsOf(c.result), steps, session.regions),
		describe: (rule) => describeRule(rule, session.regions),
		/** Some terms read on a case as measure_region gives them: the readout, with shares, and what it summed over. */
		readoutOf: (c, terms) => {
			const { part, whole, over } = readout(c, terms);
			return { readout: report(part, whole), summed_over: over };
		},
	};
}

// ---- the dock's Regions tab, and painting and lassoing on the car ---------------------------------------------------

/**
 * The Regions tab: how the next rule combines, the rules to add (facing, painting, a lasso), the steps of the
 * selection, and the saved regions. host: the overlay over the view, for the lasso's outline.
 */
export function createRegionsPanel({ regions, session, viewer, host, run }) {
	const view = session.view;
	let op = 'replace';
	let drawing = null;
	const outline = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
	outline.setAttribute('class', 'lasso');
	const path = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
	outline.append(path);
	outline.style.display = 'none';
	// The brush's reach on screen, round the pointer while painting.
	const brush = h('div', { class: 'brush', hidden: true });
	brush.style.setProperty('--brush', `${2 * BRUSH_PX}px`);
	host.append(outline, brush);
	viewer.canvas.addEventListener('pointermove', (e) => {
		brush.hidden = drawing !== paint;
		const r = host.getBoundingClientRect();
		brush.style.transform = `translate(${e.clientX - r.left}px, ${e.clientY - r.top}px)`;
	});
	viewer.canvas.addEventListener('pointerleave', () => (brush.hidden = true));

	const term = (rule) => [{ op, rule }];
	const apply = (rule) => run('select_region', { terms: term(rule), mode: op === 'replace' ? 'set' : 'extend' }).catch(() => {});
	/** The steps the selection would have with this rule combined in as the mode says. */
	const combined = (rule) => (op === 'replace' ? term(rule) : [...(view.selection?.terms ?? []), ...term(rule)]);

	/**
	 * Painting: dabs where the pointer passes over the car, tinted as they come and joined into one patch rule when
	 * the button lifts. Each dab is a pick, read back a frame or two later, so they queue in order.
	 */
	const paint = {
		points: [],
		radius: 0,
		queue: Promise.resolve(),
		down(at) {
			this.points = [];
			this.radius = 0;
			this.dab(at);
		},
		move(at) {
			this.dab(at);
		},
		dab({ clientX, clientY }) {
			const picked = viewer.pick(clientX, clientY);
			this.queue = this.queue.then(async () => {
				const hit = await picked;
				if (!hit) return;
				this.radius ||= BRUSH_PX * viewer.worldPerPixel(hit.point);
				const last = this.points.at(-1);
				if (last && Math.hypot(...last.map((v, k) => v - hit.point[k])) < this.radius / 2) return;
				this.points.push(hit.point.map((v) => round(v, 4)));
				regions.preview(combined(this.patch()));
			});
		},
		patch() {
			return { kind: 'patch', points_m: this.points, radius_m: round(this.radius, 4), drawn: 'paint' };
		},
		async up() {
			await this.queue;
			regions.preview(null);
			if (this.points.length) apply(this.patch());
			this.points = [];
		},
	};

	/**
	 * A lasso: an outline drawn over the view; the surface seen inside it is the patch. Every pixel of that surface is
	 * visited, and a point is kept wherever none kept so far lies within the radius, so the points reach every spot seen
	 * however steeply the surface leans away. The radius is a few pixels at the camera's pivot, more for a wide lasso.
	 */
	const lasso = {
		corners: [],
		down({ clientX, clientY }) {
			this.corners = [[clientX, clientY]];
			outline.style.display = '';
			this.draw();
		},
		move({ clientX, clientY }) {
			this.corners.push([clientX, clientY]);
			this.draw();
		},
		draw() {
			const r = host.getBoundingClientRect();
			path.setAttribute('points', this.corners.map(([x, y]) => `${x - r.left},${y - r.top}`).join(' '));
		},
		async up() {
			outline.style.display = 'none';
			const corners = this.corners;
			this.corners = [];
			if (corners.length < 3) return;
			const seen = await viewer.readSurface();
			const polygon = corners.map(([x, y]) => seen.toPixel(x, y));
			const step = Math.max(LASSO_STEP, Math.sqrt(areaOf(polygon) / LASSO_POINTS));
			const radius = round((step / (seen.width / viewer.canvas.getBoundingClientRect().width)) * viewer.worldPerPixel(viewer.controls.target.toArray()), 4);
			const grid = new PatchGrid(radius);
			const points = [];
			let cover = null;
			const ys = polygon.map((p) => p[1]);
			for (let y = Math.max(0, Math.min(...ys)); y <= Math.min(seen.height - 1, Math.max(...ys)); y++) {
				const across = crossings(polygon, y + 0.5);
				for (let s = 0; s + 1 < across.length; s += 2) {
					for (let x = Math.max(0, Math.ceil(across[s] - 0.5)); x <= Math.min(seen.width - 1, Math.floor(across[s + 1] - 0.5)); x++) {
						const i = 4 * (y * seen.width + x);
						if (seen.data[i + 3] === 0) continue;
						const px = seen.data[i], py = seen.data[i + 1], pz = seen.data[i + 2];
						// The point that covered the last pixel usually covers this one too, sparing the grid.
						if (cover && (cover[0] - px) ** 2 + (cover[1] - py) ** 2 + (cover[2] - pz) ** 2 <= grid.r2) continue;
						cover = grid.near(px, py, pz);
						if (cover) continue;
						cover = [round(px, 4), round(py, 4), round(pz, 4)];
						grid.add(cover);
						points.push(cover);
					}
				}
			}
			if (points.length) apply({ kind: 'patch', points_m: points, radius_m: radius, drawn: 'lasso' });
		},
	};

	function use(tool) {
		drawing = drawing === tool ? null : tool;
		viewer.setTool(drawing);
		viewer.hoverPicking = !drawing;
		render();
	}

	const segments = OPS.map((name) => h('button', { type: 'button', role: 'radio', class: 'segment', onclick: () => { op = name; render(); } }, OP_LABELS[name]));
	const facingList = h('div', { class: 'menu-list', role: 'menu' }, Object.entries(FACINGS).map(([key, f]) =>
		h('button', { type: 'button', role: 'menuitem', class: 'menu-item', title: f.title, onclick: () => { facing.close(); apply({ kind: 'facing', facing: key }); } }, f.label)));
	const facing = menu(h('button', { type: 'button', class: 'tool', title: 'Faces facing a direction' }, 'Facing', icon('chevron')), facingList);
	const tool = (which, label, glyph, title) => h('button', { type: 'button', class: 'tool', title, onclick: () => use(which) }, icon(glyph), label);
	const painting = tool(paint, 'Paint', 'brush', 'Paint faces on the car with the left button; the right button still pans');
	const lassoing = tool(lasso, 'Lasso', 'lasso', 'Draw round faces on the car with the left button; only faces in sight are picked');
	const clear = h('button', { type: 'button', class: 'tool', onclick: () => run('clear_selection', {}) }, 'Clear');
	const controls = h('div', { class: 'regions-controls' },
		h('div', { class: 'segmented', role: 'radiogroup', 'aria-label': 'How the next rule combines with the selection' }, segments),
		facing.el, painting, lassoing, clear);

	const steps = h('ol', { class: 'regions-steps' });
	const name = h('input', { class: 'number regions-name', type: 'text', placeholder: 'Name', 'aria-label': 'Region name', spellcheck: 'false' });
	const save = h('form', { class: 'regions-save', onsubmit: (e) => {
		e.preventDefault();
		if (!name.value.trim()) return;
		run('save_region', { name: name.value.trim() }).then(() => (name.value = ''), () => {});
	} }, name, h('button', { type: 'submit', class: 'button' }, 'Save region'));
	const saved = h('div', { class: 'regions-saved' });
	const el = h('div', { class: 'regions' }, h('div', { class: 'regions-selection' }, steps, save), saved);

	function render() {
		segments.forEach((button, i) => button.setAttribute('aria-checked', String(OPS[i] === op)));
		painting.setAttribute('aria-pressed', String(drawing === paint));
		lassoing.setAttribute('aria-pressed', String(drawing === lasso));
		clear.disabled = !view.selection;
		const terms = view.selection?.terms ?? [];
		steps.replaceChildren(...terms.map((t) => h('li', {}, h('span', { class: 'regions-op', title: OP_LABELS[t.op] }, OP_SIGNS[t.op]), regions.describe(t.rule))));
		save.hidden = terms.length === 0;
		saved.replaceChildren(...session.regions.map((r) => h('div', { class: 'regions-row' },
			h('span', { class: 'regions-row-name' }, r.name),
			h('button', { type: 'button', class: 'link-button', title: `Combine ${r.name} with the selection as the mode says`, onclick: () => apply({ kind: 'region', region_id: r.id }) }, 'Select'),
			h('button', { type: 'button', class: 'icon-button', title: `Delete ${r.name}`, onclick: () => run('delete_region', { region_id: r.id }).catch(() => {}) }, icon('close')))));
	}

	/** Leave painting or lassoing, as when the tab is left. */
	function stop() {
		if (drawing) use(drawing);
	}

	return { el, controls, render, stop };
}

/** Where the line at height y crosses a polygon's edges, in order: by the even-odd rule, inside runs between pairs. */
function crossings(polygon, y) {
	const xs = [];
	for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
		const [xi, yi] = polygon[i], [xj, yj] = polygon[j];
		if (yi > y !== yj > y) xs.push(xi + ((y - yi) * (xj - xi)) / (yj - yi));
	}
	return xs.sort((a, b) => a - b);
}

/** A polygon's area, by the shoelace formula. */
function areaOf(polygon) {
	let twice = 0;
	for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) twice += polygon[j][0] * polygon[i][1] - polygon[i][0] * polygon[j][1];
	return Math.abs(twice) / 2;
}
