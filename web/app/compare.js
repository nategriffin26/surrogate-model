// Comparing the car on screen with the CFD its file carries (the back-check of the prediction) or with another open
// case. The other's pressure is brought onto this car's working mesh: the CFD as the file gives it, another car's
// from its nearest vertex facing the same way (sampleFields), so the two can be drawn and subtracted point by point.
// A comparison offers two fields, the other's Cp on this car and the difference ΔCp, this car's minus the other's;
// draws this car beside the other on request; adds the other's line to the section plot; and reads the whole car,
// the selection and every saved region on both into one table. In the table each side is measured on its own faces:
// the CFD on this car's, another car on its own, where the same rules pick its regions.

import { sampleFields } from '../engine/nearest.js';
import { round } from './checks.js';
import { h } from './dom.js';
import { baseName, defineExport } from './export.js';
import { addField, removeField, scaleEnd } from './fields.js';
import { signed } from './format.js';
import { icon } from './icons.js';
import { menu } from './menu.js';
import { defineOperation } from './operations.js';
import { measure } from './regions.js';

/** The fields a comparison offers: the other's Cp on this car, and this car's Cp minus it. */
export const COMPARED = 'compared';
export const DIFFERENCE = 'difference';
const DIFFERENCE_RAMPS = ['coolwarm', 'purplegreen', 'brownteal'];

/**
 * What case c is compared with under the view's comparison: {other (a case, or null for the CFD), label (the other's
 * name), mine (this side's name: Predicted beside the CFD, else the case's)}; or null when there is nothing to compare
 * c with: no comparison, no CFD on c, or the other case is c itself or has no result.
 */
export function counterpart(view, cases, c) {
	const compare = view.compare;
	if (!compare || !c?.result) return null;
	if (compare.with === 'cfd') return c.result.mesh.cfd ? { other: null, label: 'CFD', mine: 'Predicted' } : null;
	const other = cases.find((x) => x.id === compare.with);
	if (!other?.result || other === c) return null;
	const [mine, label] = shortNames(c.name, other.name);
	return { other, label, mine };
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
 * the difference; worked out once per pair of results and kept with c's result.
 */
const held = new WeakMap();
export function comparedOn(view, cases, c) {
	const pair = counterpart(view, cases, c);
	if (!pair) return null;
	const key = pair.other?.result ?? 'cfd';
	let known = held.get(c.result);
	if (known?.key !== key) {
		const { mesh } = c.result;
		const from = pair.other?.result.mesh;
		const compared = from ? sampleFields({ positions: from.positions, normals: from.normals, cp: from.cp }, mesh).cp : mesh.cfd;
		const difference = Float32Array.from(mesh.cp, (cp, v) => cp - compared[v]);
		known = { key, compared, difference, end: scaleEnd(difference.length, (v) => difference[v]) };
		held.set(c.result, known);
	}
	return { ...pair, ...known };
}

const QUANTITIES = [
	{ key: 'cx', label: 'Cx', of: (m) => m.coefficients[0] },
	{ key: 'cy', label: 'Cy', of: (m) => m.coefficients[1] },
	{ key: 'cz', label: 'Cz', of: (m) => m.coefficients[2] },
	{ key: 'mean_cp', label: 'Mean Cp', of: (m) => m.meanCp },
];
/** The quantities the table shows; the operation and the CSV give Cy too. */
const SHOWN = ['cx', 'cz', 'mean_cp'];

/** A quantity of a table row on this car, on the other, and their difference; null where a side has no faces. */
function valuesOf(q, row) {
	const a = q.of(row.mine), b = q.of(row.other);
	return [a, b, a === null || b === null ? null : a - b];
}

/**
 * app: { session, regions, update, run }. The comparison: its fields, its side-by-side panes, its operations and the
 * Compare tab.
 */
export function createComparison({ session, regions, update, run }) {
	const view = session.view;
	const compared = (c) => comparedOn(view, session.cases, c);

	/** Which pair the comparison fields are labelled for; a new pair restarts the difference's colour scale. */
	let labelled = null;
	function label(pair) {
		const key = `${pair.mine}|${pair.label}|${pair.key === 'cfd' ? 'cfd' : pair.other.id}`;
		if (key === labelled) return;
		const start = { ramp: 'coolwarm', lo: -pair.end, hi: pair.end, bands: 0 };
		const title = pair.other ? `ΔCp ${pair.mine} − ${pair.label}` : 'ΔCp predicted − CFD';
		addField(COMPARED, { label: pair.label, title: `Cp ${pair.label}`, symbol: 'Cp', tip: pair.other ? `Pressure coefficient of ${pair.label}, on this car's surface` : 'Pressure coefficient from the CFD solution in the file', values: (c) => compared(c)?.compared ?? null }, 'cp');
		addField(DIFFERENCE, { label: 'Difference', title, symbol: 'ΔCp', tip: title, values: (c) => compared(c)?.difference ?? null }, { ramps: DIFFERENCE_RAMPS, diverging: true, start });
		view.maps ??= {};
		view.maps[DIFFERENCE] = { ...start };
		labelled = key;
	}

	/** The comparison's fields on case c: the other's Cp and the difference, when c is compared with something. */
	function fields(c) {
		const pair = compared(c);
		if (!pair) return [];
		label(pair);
		return [COMPARED, DIFFERENCE];
	}

	/** Side by side: this car's Cp beside the other's, while Cp or the other's Cp is the field shown; or null. */
	function panes(c, field) {
		if (!view.compare?.sideBySide || (field !== 'cp' && field !== COMPARED)) return null;
		const pair = compared(c);
		return pair ? [{ field: 'cp', label: pair.mine }, { field: COMPARED, label: pair.label }] : null;
	}

	/** The rows of the table: whole car, selection, saved regions; each measured on both sides, with the RMS of ΔCp. */
	const measured = new WeakMap();
	const WHOLE = [];
	function rows(c) {
		const pair = compared(c);
		if (!pair) return [];
		const cells = regions.cellsOf(c.result);
		const theirs = pair.other ? { case: pair.other, cells: regions.cellsOf(pair.other.result) } : { case: c, cells: { ...cells, cp: cells.cfd } };
		const items = [
			{ name: 'Whole car', terms: WHOLE },
			...(view.selection ? [{ name: 'Selection', terms: view.selection.terms }] : []),
			...session.regions.map((r) => ({ name: r.name, region_id: r.id, terms: [{ op: 'replace', rule: { kind: 'region', region_id: r.id } }], held: r.terms })),
		];
		// Kept per pair of results and per region, which a selection's or region's terms stand for: they are never changed in place.
		const perPair = measured.get(c.result)?.get(theirs.case.result) ?? new WeakMap();
		measured.set(c.result, (measured.get(c.result) ?? new Map()).set(theirs.case.result, perPair));
		return items.map((item) => {
			const id = item.held ?? item.terms;
			if (!perPair.has(id)) {
				const whole = item.terms === WHOLE;
				const mask = whole ? null : regions.maskOf(c, item.terms);
				const mine = measure(cells, mask, c.result.geometry.frontalArea);
				const other = measure(theirs.cells, whole ? null : pair.other ? regions.maskOf(pair.other, item.terms) : mask, theirs.case.result.geometry.frontalArea);
				perPair.set(id, { mine, other, rms: rmsDifference(c.result.mesh, cells, pair.compared, mask) });
			}
			return { ...item, ...perPair.get(id) };
		});
	}

	const rowOut = (row) => ({
		name: row.name,
		...(row.region_id !== undefined ? { region_id: row.region_id } : {}),
		this_car: sideOut(row.mine),
		other: sideOut(row.other),
		difference: Object.fromEntries(QUANTITIES.map((q) => [q.key, r5(valuesOf(q, row)[2])])),
		rms_dcp: round(row.rms, 5),
	});
	const sideOut = (m) => ({ faces: m.faces, area_m2: round(m.area, 5), ...Object.fromEntries(QUANTITIES.map((q) => [q.key, r5(q.of(m))])) });
	const r5 = (v) => (v === null ? null : round(v, 5));

	/**
	 * Compared with the CFD, the whole car's Cx over every polygon of the file, as the result card gives it: the
	 * figure to check the prediction by, which the working mesh's faces only approach. Null otherwise.
	 */
	function everyPolygon(c) {
		const { forces } = c.result;
		if (compared(c)?.other !== null || forces.cdPCfd == null) return null;
		return { name: 'Whole car, every polygon', cx: [forces.cdP, forces.cdPCfd, forces.cdP - forces.cdPCfd] };
	}

	function comparisonOut(c) {
		const pair = compared(c);
		const forces = c.result.forces;
		return {
			case_id: c.id,
			compared_with: pair.other ? { case_id: pair.other.id, name: pair.other.name } : 'cfd',
			side_by_side: view.compare.sideBySide,
			rows: rows(c).map(rowOut),
			...(everyPolygon(c) ? { every_polygon: { cx_predicted: round(forces.cdP, 5), cx_cfd: round(forces.cdPCfd, 5) } } : {}),
			summed_over: 'each side on its own working-mesh faces (Cp at each face the mean of its corners); rms_dcp over this car\'s faces, of the difference field',
		};
	}

	defineOperation({
		name: 'set_comparison',
		description: 'Compare the car on screen with the CFD its file carries (with "cfd": the back-check of the prediction) or with another open case (with its case_id), or stop comparing (with null). The other\'s pressure is brought onto this car\'s surface, another car\'s from its nearest point facing the same way. A comparison offers two fields for set_field: compared (the other\'s Cp on this car) and difference (ΔCp, this car\'s minus the other\'s). side_by_side draws this car\'s Cp beside the other\'s under one camera. The section plot adds the other\'s line. Returns the comparison table: the whole car, the selection and each saved region on both, each side measured on its own faces, with the difference and the RMS of ΔCp.',
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
		description: 'The comparison table for a case, without changing the view: the whole car, the selection and each saved region, each with area, Cx, Cy, Cz and mean Cp on this car and on what it is compared with (each side on its own faces), their difference, and the area-weighted RMS of ΔCp over this car\'s faces. Compared with CFD, also the whole car\'s Cx over every polygon of the file.',
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
		title: 'The comparison table: each region on this car and on what it is compared with, and their difference',
		unavailable: (c) => (compared(c) ? null : 'Nothing is compared. Choose what to compare with in the Compare tab.'),
		make: async (c) => new Blob([comparisonCsv(c)], { type: 'text/csv' }),
		fileName: (c) => `${baseName(c)}-comparison.csv`,
	});

	function comparisonCsv(c, separator = ',') {
		const pair = compared(c);
		const head = ['region', 'area_m2', ...QUANTITIES.flatMap((q) => [`${q.key}_this`, `${q.key}_other`, `${q.key}_difference`]), 'rms_dcp'];
		const quote = (text) => (separator === ',' ? `"${text.replaceAll('"', '""')}"` : text);
		const lines = [`# ${c.name} compared with ${pair.other ? pair.other.name : 'CFD'}`, head.join(separator)];
		const rowsOut = rows(c).map((row) => [quote(row.name), row.mine.area.toFixed(5), ...QUANTITIES.flatMap((q) => valuesOf(q, row).map((v) => (v === null ? '' : v.toFixed(5)))), row.rms.toFixed(5)].join(separator));
		const exact = everyPolygon(c);
		if (exact) rowsOut.splice(1, 0, [quote(exact.name), '', ...QUANTITIES.flatMap((q) => (q.key === 'cx' ? exact.cx.map((v) => v.toFixed(5)) : ['', '', ''])), ''].join(separator));
		lines.push(...rowsOut);
		return lines.join('\n') + '\n';
	}

	// ---- the Compare tab ----

	const quietly = (promise) => promise.catch(() => {});
	const choices = h('div', { class: 'menu-list', role: 'menu' });
	const withButton = h('button', { type: 'button', class: 'tool compare-with' });
	const chooser = menu(withButton, choices);
	const sideBySide = h('button', { type: 'button', class: 'tool', 'aria-pressed': 'false', onclick: () => quietly(run('set_comparison', { side_by_side: !view.compare?.sideBySide })) }, icon('compare'), 'Side by side');
	const copy = h('button', { type: 'button', class: 'tool', onclick: () => navigator.clipboard.writeText(comparisonCsv(session.active, '\t')) }, icon('copy'), 'Copy');
	const save = h('button', { type: 'button', class: 'tool', onclick: () => quietly(run('download_export', { export: 'comparison' })) }, icon('download'), 'CSV');
	const controls = h('div', { class: 'compare-controls' }, chooser.el, sideBySide, copy, save);
	const table = h('table', { class: 'compare-table' });
	const empty = h('div', { class: 'compare-empty' });
	const el = h('div', { class: 'compare' }, empty, table);

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
					quietly(run('set_comparison', { with: o.with }));
				},
			}, o.label)),
			...(view.compare ? [h('div', { class: 'menu-rule' }), h('button', { type: 'button', role: 'menuitem', class: 'menu-item', onclick: () => {
				chooser.close();
				quietly(run('set_comparison', { with: null }));
			} }, 'Stop comparing')] : []),
		);
		sideBySide.disabled = copy.disabled = save.disabled = !pair;
		sideBySide.setAttribute('aria-pressed', String(Boolean(pair && view.compare.sideBySide)));
		empty.hidden = Boolean(pair);
		table.hidden = !pair;
		if (!pair) {
			empty.replaceChildren(...(options.length
				? [h('span', {}, `Compare ${c.name} with`), ...options.map((o) => h('button', { type: 'button', class: 'tool', onclick: () => quietly(run('set_comparison', { with: o.with })) }, o.label))]
				: [h('span', {}, 'Open another car to compare with.')]));
			return;
		}
		const shown = QUANTITIES.filter((q) => SHOWN.includes(q.key));
		const cells = (values) => values.map((v, k) => h('td', { class: k % 3 === 2 ? 'compare-delta' : '' }, v === null ? '' : signed(v, 3)));
		const body = rows(c).map((row) => h('tr', {}, h('td', { class: 'compare-name' }, row.name), ...cells(shown.flatMap((q) => valuesOf(q, row))), h('td', {}, row.rms.toFixed(3))));
		const exact = everyPolygon(c);
		if (exact) body.splice(1, 0, h('tr', {}, h('td', { class: 'compare-name' }, exact.name), ...cells(shown.flatMap((q) => (q.key === 'cx' ? exact.cx : [null, null, null]))), h('td', {})));
		table.replaceChildren(
			h('thead', {},
				h('tr', { class: 'compare-groups' }, h('th', {}), ...shown.map((q) => h('th', { colspan: 3 }, q.label)), h('th', { title: 'Area-weighted root mean square of ΔCp over this car\'s faces' }, 'RMS ΔCp')),
				h('tr', {}, h('th', {}, 'Region'), ...shown.flatMap(() => [h('th', {}, pair.mine), h('th', {}, pair.label), h('th', {}, 'Δ')]), h('th', {}))),
			h('tbody', {}, ...body),
		);
	}

	return { fields, panes, panel: { el, controls, render, stop: () => {} } };
}

/** Area-weighted root mean square of ΔCp over the faces a mask picks: each face's Cp minus the mean of the other's at its corners. */
function rmsDifference(mesh, cells, compared, mask) {
	const I = mesh.indices;
	let sum = 0, area = 0;
	for (let f = 0; f < cells.areas.length; f++) {
		if (mask && !mask[f]) continue;
		const d = cells.cp[f] - (compared[I[3 * f]] + compared[I[3 * f + 1]] + compared[I[3 * f + 2]]) / 3;
		if (Number.isNaN(d)) continue;
		sum += cells.areas[f] * d * d;
		area += cells.areas[f];
	}
	return area > 0 ? Math.sqrt(sum / area) : 0;
}
