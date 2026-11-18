// Comparing the car on screen with the CFD its file carries (the back-check of the prediction) or with another open
// case. The other's pressure is brought onto this car's working mesh: the CFD as the file gives it, another car's
// from its nearest vertex facing the same way (sampleFields), a car's and its variants' vertex for vertex, so the two
// can be drawn and subtracted point by point.
// A comparison offers two fields, the other's Cp on this car and the difference ΔCp, this car's minus the other's;
// draws this car beside the other on request; adds the other's line to the section plot; and reads the whole car on
// both into one row, each side summed on its own polygons: the CFD on this car's, another car on its own. A car and
// one of its variants are compared like for like: the car's side is its surface predicted unchanged the way the
// variant was.

import { sampleFields } from '../engine/nearest.js';
import { sampled, totalsOver } from './cases.js';
import { round } from './checks.js';
import { coefficient, failureNote, h } from './dom.js';
import { baseName, defineExport } from './export.js';
import { MAPS, addField, removeField, scaleEnd } from './fields.js';
import { signed } from './format.js';
import { icon } from './icons.js';
import { menu } from './menu.js';
import { defineOperation } from './operations.js';
import { comparedAs, sameVertices } from './variants.js';

/** The fields a comparison offers: the other's Cp on this car, and this car's Cp minus it. */
export const COMPARED = 'compared';
const DIFFERENCE = 'difference';
const DIFFERENCE_RAMPS = MAPS.error.ramps;

/**
 * What case c is compared with under the view's comparison: {other (a case, or null for the CFD), self (c as it is
 * compared), label (the other's name), mine (this side's name: Predicted beside the CFD, else the case's), unchanged
 * (whether a car is read as predicted unchanged beside its variant)}; or null when there is nothing to compare c with:
 * no comparison, no CFD on c, or the other case is c itself or has no result.
 */
export function counterpart(view, cases, c) {
	const compare = view.compare;
	if (!compare || !c?.result) return null;
	if (compare.with === 'cfd') return c.result.mesh.cfd ? { other: null, self: c, label: 'CFD', mine: 'Predicted', unchanged: false } : null;
	const other = cases.find((x) => x.id === compare.with);
	if (!other?.result || other === c) return null;
	const [mine, label] = shortNames(c.name, other.name);
	const self = comparedAs(c, other), them = comparedAs(other, c);
	return { other: them, self, label, mine, unchanged: self !== c || them !== other };
}

/**
 * Two case names without the words they start with in common, keeping the last of those before a number: "run 1" and
 * "run 500", not "DrivAerML run 1" and "DrivAerML run 500".
 */
function shortNames(a, b) {
	const x = a.split(' '), y = b.split(' ');
	let k = 0;
	while (k < x.length - 1 && k < y.length - 1 && x[k] === y[k]) k++;
	if (k > 0 && /^\d/.test(x[k]) && /^\d/.test(y[k])) k--;
	return [x.slice(k).join(' '), y.slice(k).join(' ')];
}

/**
 * The other's Cp at each working-mesh vertex of case c, this car's minus it, and the end of a scale spanning 98 % of
 * the difference; worked out once per pair of results and kept with c's result. This car's Cp is its own, or beside
 * one of its variants its surface's predicted unchanged, which has the same vertices.
 */
const pairs = new WeakMap();
function comparedOn(view, cases, c) {
	const pair = counterpart(view, cases, c);
	if (!pair) return null;
	// Kept per other car, or per reference field, which a newly attached reference replaces.
	const key = pair.other?.result ?? c.result.mesh.cfd;
	let known = pairs.get(c.result);
	if (known?.key !== key) {
		const { mesh } = pair.self.result;
		const from = pair.other?.result.mesh;
		const compared = !from ? mesh.cfd : sameVertices(c, pair.other) ? from.cp : sampleFields({ positions: from.positions, normals: from.normals, cp: from.cp }, mesh).cp;
		const difference = Float32Array.from(mesh.cp, (cp, v) => cp - compared[v]);
		known = { key, compared, difference, end: scaleEnd(difference.length, (v) => difference[v]) };
		pairs.set(c.result, known);
	}
	return { ...pair, ...known };
}

/** Each quantity of the table; its name is built per table, since an element stands in one place only. */
const QUANTITIES = [
	{ key: 'cx', name: () => coefficient('x'), of: (m) => m.coefficients[0] },
	{ key: 'cy', name: () => coefficient('y'), of: (m) => m.coefficients[1] },
	{ key: 'cz', name: () => coefficient('z'), of: (m) => m.coefficients[2] },
	{ key: 'mean_cp', name: () => 'Mean Cp', of: (m) => m.meanCp },
];
/** The quantities the table shows, which with the section plot beside it fit a phone's width; the operation and the CSV give Cy and mean Cp too. */
const SHOWN = ['cx', 'cz'];

/** A quantity of the whole car on this car, on the other, and their difference. */
function valuesOf(q, row) {
	const a = q.of(row.mine), b = q.of(row.other);
	return [a, b, a - b];
}

/** Sums over every face of some cells: area, area-weighted mean of `cp`, and the pressure force coefficients Cx, Cy, Cz = −Σ Cp n A / reference area. */
function measure(cells, cp, referenceArea) {
	const { areas, normals } = cells;
	let area = 0, cpArea = 0, fx = 0, fy = 0, fz = 0;
	for (let c = 0; c < areas.length; c++) {
		const a = areas[c];
		area += a;
		cpArea += cp[c] * a;
		fx -= cp[c] * normals[3 * c] * a;
		fy -= cp[c] * normals[3 * c + 1] * a;
		fz -= cp[c] * normals[3 * c + 2] * a;
	}
	return { faces: areas.length, area, meanCp: area > 0 ? cpArea / area : null, coefficients: [fx, fy, fz].map((f) => f / referenceArea) };
}

/**
 * app: { session, engine, update, run }. The comparison: its fields, its side-by-side panes, its operations and the
 * Compare tab.
 */
export function createComparison({ session, engine, update, run }) {
	const view = session.view;
	const compared = (c) => comparedOn(view, session.cases, c);

	/** Which pair the comparison fields are offered for; a new pair restarts the difference's colour scale. */
	let offeredFor = null;
	function offerFields(pair) {
		const key = `${pair.mine}|${pair.label}|${pair.other ? pair.other.id : 'cfd'}`;
		if (key === offeredFor) return;
		const start = { ramp: 'coolwarm', lo: -pair.end, hi: pair.end, bands: 0 };
		const title = pair.other ? `ΔCp ${pair.mine} − ${pair.label}` : 'ΔCp predicted − CFD';
		addField(COMPARED, { label: pair.label, title: `Cp ${pair.label}`, symbol: 'Cp', tip: pair.other ? `Pressure coefficient of ${pair.label}, on this car's surface` : 'Pressure coefficient from the CFD solution in the file', values: (c) => compared(c)?.compared ?? null }, 'cp');
		addField(DIFFERENCE, { label: 'Difference', title, symbol: 'ΔCp', tip: title, values: (c) => compared(c)?.difference ?? null }, { ramps: DIFFERENCE_RAMPS, diverging: true, start });
		view.maps ??= {};
		view.maps[DIFFERENCE] = { ...start };
		offeredFor = key;
	}

	/** The comparison's fields on case c: the other's Cp and the difference, when c is compared with something. */
	function fields(c) {
		const pair = compared(c);
		if (!pair) return [];
		offerFields(pair);
		return [COMPARED, DIFFERENCE];
	}

	/** Side by side: this car's Cp beside the other's, while Cp or the other's Cp is the field shown; or null. */
	function panes(c, field) {
		if (!view.compare?.sideBySide || (field !== 'cp' && field !== COMPARED)) return null;
		const pair = compared(c);
		return pair ? [{ field: 'cp', label: pair.mine }, { field: COMPARED, label: pair.label }] : null;
	}

	/** Per result: its working mesh's faces, and the file's own polygons for sums, once fetched. */
	const perResult = new WeakMap();
	const known = (result) => perResult.get(result) ?? perResult.set(result, { cells: engine.workingCells(result), file: null }).get(result);

	/**
	 * The cells a case's sums run over: a dropped file's own polygons, fetched from the engine the first time they are
	 * wanted once every polygon is predicted (and again when a reference is attached), and kept with the result; until
	 * then, and for a bundled example, which carries only the working mesh, that mesh's faces. A variant was predicted
	 * on its faces themselves, so they are its own polygons.
	 */
	function sumBasis(c) {
		const kept = known(c.result);
		if (c.result.form === 'quick') return { cells: kept.cells, over: `every face it was predicted on (${kept.cells.areas.length})`, exact: true };
		const { reference } = c.result;
		if (kept.file?.cells && kept.file.reference === reference) return kept.file;
		if (c.kind === 'file' && kept.file?.reference !== reference && engine.canExportVtp(c.result)) {
			kept.file = { cells: null, reference };
			engine.fileCells(c.result).then((set) => {
				if (!set || kept.file.reference !== reference) return;
				// Only what sums read is kept: the points and polygons themselves would double the memory.
				const { centers, normals, areas, cp, sigma, cfd } = set;
				kept.file = { cells: { centers, normals, areas, cp, sigma, cfd }, over: `every polygon of the file (${areas.length})`, exact: true, reference };
				update('dock');
			});
		}
		return { cells: kept.cells, over: `${kept.cells.areas.length} faces of the working mesh (Cp at each face the mean of its corners, as drawn)`, exact: false };
	}

	/** The whole car on both sides, with the RMS of ΔCp over this car's faces. */
	function wholeCar(c) {
		const pair = compared(c);
		return { mine: side(pair.self, false), other: side(pair.other ?? pair.self, !pair.other), rms: rms(pair.self, pair) };
	}

	/** Sums of a case's prediction, and of its reference field, kept per set of cells. */
	const sums = new WeakMap();

	/**
	 * One side: case x's prediction, or with `cfd` its reference field, summed on its own polygons. Where those are
	 * not yet at hand the result's exact totals stand in: the reference's are summed over every polygon from the start,
	 * the prediction's once every polygon is predicted.
	 */
	function side(x, cfd) {
		const basis = sumBasis(x);
		if (basis.exact) {
			const measured = sums.get(basis.cells) ?? sums.set(basis.cells, {}).get(basis.cells);
			const field = cfd ? 'cfd' : 'cp';
			measured[field] ??= measure(basis.cells, basis.cells[field], x.result.geometry.frontalArea);
			return { ...measured[field], over: basis.over };
		}
		const r = x.result;
		if (cfd) return { faces: r.source.cells, area: r.totals.area, ...r.referenceTotals, over: 'every polygon' };
		return { faces: r.source.cells, area: r.totals.area, meanCp: r.totals.meanCp, coefficients: r.totals.coefficients, over: totalsOver(sampled(x)) };
	}

	/**
	 * Area-weighted RMS of ΔCp over this car's faces: against the CFD on the polygons it is summed on, against another
	 * car on the working mesh, where the other's Cp is brought. Kept per field.
	 */
	const rmsKept = new WeakMap();
	function rms(c, pair) {
		const cfd = !pair.other;
		const cells = cfd ? sumBasis(c).cells : known(c.result).cells;
		const key = cfd ? cells : pair.difference;
		if (!rmsKept.has(key)) {
			const I = c.result.mesh.indices, d = pair.difference;
			rmsKept.set(key, rmsOver(cells.areas, cfd ? (f) => cells.cp[f] - cells.cfd[f] : (f) => (d[I[3 * f]] + d[I[3 * f + 1]] + d[I[3 * f + 2]]) / 3));
		}
		return rmsKept.get(key);
	}

	/** Whether RMS ΔCp is taken on the simplified mesh rather than on the polygons of a file, said once under the table; or null. */
	function simplified(c) {
		const { self, other } = compared(c);
		// A car and its variant are both predicted on the simplified mesh, so it is the surface the difference is taken on.
		const quick = (x) => x.result.form === 'quick';
		const onSimplified = (Boolean(other) && !(quick(self) && quick(other))) || !sumBasis(self).exact;
		return onSimplified ? 'RMS ΔCp is taken on the simplified mesh.' : null;
	}

	const sideOut = (m) => ({ faces: m.faces, area_m2: round(m.area, 5), ...Object.fromEntries(QUANTITIES.map((q) => [q.key, round(q.of(m), 5)])), summed_over: m.over });

	function comparisonOut(c) {
		const pair = compared(c);
		const whole = wholeCar(c);
		return {
			case_id: c.id,
			compared_with: pair.other ? { case_id: pair.other.id, name: pair.other.name } : 'cfd',
			...(pair.unchanged ? { like_for_like: 'The car is read as its surface predicted unchanged on the working mesh, as its variant is, so the difference is the change alone.' } : {}),
			side_by_side: view.compare.sideBySide,
			whole_car: {
				this_car: sideOut(whole.mine),
				other: sideOut(whole.other),
				difference: Object.fromEntries(QUANTITIES.map((q) => [q.key, round(valuesOf(q, whole)[2], 5)])),
				rms_dcp: round(whole.rms, 5),
			},
			rms_dcp_over: pair.other || !sumBasis(pair.self).exact ? 'this car\'s working mesh, of the difference field as drawn' : 'this car\'s polygons, Cp minus CFD on each',
		};
	}

	defineOperation({
		name: 'set_comparison',
		description: 'Compare the car on screen with the CFD its file carries (with "cfd": the back-check of the prediction) or with another open case (with its case_id), or stop comparing (with null). The other\'s pressure is brought onto this car\'s surface, another car\'s from its nearest point facing the same way. A comparison offers two fields for set_field: compared (the other\'s Cp on this car) and difference (ΔCp, this car\'s minus the other\'s). side_by_side draws this car\'s Cp beside the other\'s under one camera. The section plot adds the other\'s line. Returns the whole car on both, each side summed on its own polygons, with the difference and the RMS of ΔCp.',
		parameters: {
			type: 'object',
			properties: {
				with: { type: ['string', 'integer', 'null'], description: '"cfd", a case_id, or null to stop comparing. Default: keep what is compared.' },
				side_by_side: { type: 'boolean', description: 'Draw this car\'s Cp beside the other\'s.' },
			},
			additionalProperties: false,
		},
		run(args) {
			const c = session.active;
			if (!c?.result) throw new Error('No case with a result is on screen. Open a file or an example first.');
			const target = args.with;
			if (target === null) {
				view.compare = null;
				removeField(COMPARED);
				removeField(DIFFERENCE);
				delete view.maps?.[DIFFERENCE];
				labelled = null;
				if (view.field === COMPARED || view.field === DIFFERENCE) view.field = 'cp';
				update('toolbar', 'legend', 'viewer', 'dock');
				return { comparing: false };
			}
			if (target !== undefined) {
				if (target === 'cfd') {
					if (!c.result.mesh.cfd) throw new Error(`${c.name} carries no CFD pressure to compare with. Compare it with another open case by its case_id instead.`);
				} else if (Number.isInteger(target)) {
					const other = session.byId(target);
					if (other === c) throw new Error(`Case ${target} is the car on screen. Give another case_id, or "cfd".`);
					if (!other.result) throw new Error(`Case ${target} has no result yet (${other.status}).`);
				} else {
					throw new Error(`with must be "cfd", a case_id or null; got ${JSON.stringify(target)}.`);
				}
				view.compare = { with: target, sideBySide: view.compare?.sideBySide ?? false };
			}
			if (!view.compare) throw new Error('Nothing is compared. Give with: "cfd" or a case_id.');
			if (args.side_by_side !== undefined) view.compare.sideBySide = Boolean(args.side_by_side);
			if (!compared(c)) throw new Error(`${c.name} cannot be compared with that: it is the other case itself, or carries no CFD.`);
			view.dock = 'compare';
			view.dockCollapsed = false;
			update('toolbar', 'legend', 'viewer', 'dock', 'layout');
			return comparisonOut(c);
		},
	});

	defineOperation({
		name: 'read_comparison',
		description: 'The comparison for a case, without changing the view: the whole car\'s area, Cx, Cy, Cz and mean Cp on this car and on what it is compared with (each side summed on its own polygons, as summed_over says), their difference, and the area-weighted RMS of ΔCp over this car\'s faces.',
		parameters: { type: 'object', properties: { case_id: { type: 'integer', description: 'Default the case on screen.' } }, additionalProperties: false },
		run({ case_id }) {
			const c = case_id === undefined ? session.active : session.byId(case_id);
			if (!c?.result) throw new Error('No case with a result is on screen. Open a file or an example first.');
			if (!compared(c)) throw new Error('Nothing is compared with this case. Start with set_comparison.');
			return comparisonOut(c);
		},
	});

	defineExport({
		name: 'comparison',
		label: 'Comparison',
		format: 'CSV',
		title: 'The whole car on this car and on what it is compared with, and their difference',
		unavailable: (c) => (compared(c) ? null : 'Nothing is compared. Choose what to compare with in the Compare tab.'),
		make: async (c) => new Blob([comparisonCsv(c)], { type: 'text/csv' }),
		fileName: (c) => `${baseName(c)}-comparison.csv`,
	});

	function comparisonCsv(c) {
		const pair = compared(c);
		const whole = wholeCar(c);
		const head = ['area_m2', ...QUANTITIES.flatMap((q) => [`${q.key}_this`, `${q.key}_other`, `${q.key}_difference`]), 'rms_dcp'];
		const values = [whole.mine.area.toFixed(5), ...QUANTITIES.flatMap((q) => valuesOf(q, whole).map((v) => v.toFixed(5))), whole.rms.toFixed(5)];
		return `# ${c.name} compared with ${pair.other ? pair.other.name : 'CFD'}\n${head.join(',')}\n${values.join(',')}\n`;
	}

	// ---- the Compare tab ----

	const failure = failureNote();
	const compare = (args) => failure.run(run('set_comparison', args));
	const choices = h('div', { class: 'menu-list', role: 'menu' });
	const withButton = h('button', { type: 'button', class: 'tool compare-with' });
	const chooser = menu(withButton, choices);
	const sideBySide = h('button', { type: 'button', class: 'tool', 'aria-pressed': 'false', onclick: () => compare({ side_by_side: !view.compare?.sideBySide }) }, icon('compare'), 'Side by side');
	const controls = h('div', { class: 'compare-controls' }, chooser.el, sideBySide);
	const table = h('table', { class: 'compare-table' });
	const note = h('p', { class: 'compare-note' });
	const empty = h('div', { class: 'compare-empty' });
	// The section plot sits beside the table, so a difference in the numbers can be found along the car.
	const sectionSlot = h('div', { class: 'section-slot' });
	const el = h('div', { class: 'compare dock-row' }, h('div', { class: 'dock-numbers' }, failure.el, empty, table, note), sectionSlot);

	/** What case c can be compared with: its CFD when it carries one, and every other case with a result. */
	const targets = (c) => [
		...(c.result.mesh.cfd ? [{ with: 'cfd', label: 'CFD' }] : []),
		...session.cases.filter((x) => x !== c && x.result).map((x) => ({ with: x.id, label: x.name })),
	];

	function render() {
		const c = session.active;
		if (!c?.result) return;
		const pair = compared(c);
		const options = targets(c);
		const current = view.compare?.with;
		withButton.replaceChildren(pair ? `With ${pair.label}` : 'Compare with', icon('chevron'));
		choices.replaceChildren(
			...options.map((o) => h('button', {
				type: 'button', role: 'menuitemradio', class: 'menu-item', 'aria-checked': String(o.with === current),
				onclick: () => {
					chooser.close();
					compare({ with: o.with });
				},
			}, o.label)),
			...(view.compare ? [h('hr', { class: 'menu-rule' }), h('button', { type: 'button', role: 'menuitem', class: 'menu-item', onclick: () => {
				chooser.close();
				compare({ with: null });
			} }, 'Stop comparing')] : []),
		);
		sideBySide.disabled = !pair;
		sideBySide.setAttribute('aria-pressed', String(Boolean(pair && view.compare.sideBySide)));
		empty.hidden = Boolean(pair);
		table.hidden = !pair;
		note.textContent = pair ? simplified(c) ?? '' : '';
		if (!pair) {
			empty.replaceChildren(...(options.length
				? [h('span', {}, `Compare ${c.name} with`), ...options.map((o) => h('button', { type: 'button', class: 'tool', onclick: () => compare({ with: o.with }) }, o.label))]
				: [h('span', {}, 'Open another car to compare with.')]));
			return;
		}
		const shown = QUANTITIES.filter((q) => SHOWN.includes(q.key));
		const whole = wholeCar(c);
		const cells = shown.flatMap((q) => valuesOf(q, whole)).map((v, k) => h('td', { class: k % 3 === 2 ? 'compare-delta' : '' }, signed(v, 3)));
		table.replaceChildren(
			h('thead', {},
				h('tr', { class: 'compare-groups' }, h('th', {}), ...shown.map((q) => h('th', { colspan: 3 }, q.name())), h('th', { title: 'Area-weighted root mean square of ΔCp over this car\'s faces' }, 'RMS ΔCp')),
				h('tr', {}, h('th', {}), ...shown.flatMap(() => [h('th', {}, pair.mine), h('th', {}, pair.label), h('th', {}, 'Δ')]), h('th', {}))),
			h('tbody', {}, h('tr', {}, h('td', { class: 'compare-name' }, 'Whole car'), ...cells, h('td', {}, whole.rms.toFixed(3)))),
		);
	}

	return {
		fields,
		panes,
		/** The comparison table of case c, as read_comparison gives it, or null when c is compared with nothing. */
		read: (c) => (compared(c) ? comparisonOut(c) : null),
		panel: { el, controls, render, sectionSlot },
	};
}

/** Area-weighted root mean square of a per-face difference, delta(f), over every face where it is defined. */
function rmsOver(areas, delta) {
	let sum = 0, area = 0;
	for (let f = 0; f < areas.length; f++) {
		const d = delta(f);
		if (Number.isNaN(d)) continue;
		sum += areas[f] * d * d;
		area += areas[f];
	}
	return area > 0 ? Math.sqrt(sum / area) : 0;
}
