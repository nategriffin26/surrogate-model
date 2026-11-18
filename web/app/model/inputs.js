// The model's inputs in their groups, and what each group bought: switches for the groups the models compared in
// training were given or not, and how each of those models scored on pressure and on drag.

import { h } from '../dom.js';
import { dragCounts } from './evidence.js';

/** Each kind of input's colour, here and on the car. */
export const KIND_COLORS = {
	position: '#0d2233',
	bend: '#9a55e8',
	slice: '#0b6fbf',
	neighbour: '#197b80',
	outline: '#a35b1a',
	flag: '#237f50',
};

/** Names of the compared models' learners; this model's networks given fewer inputs are told apart by those inputs. */
const LEARNERS = { constant: 'Mean only', pointnet: 'PointNet', trees: 'Boosted trees', ensemble: 'Shipped' };
/** The learners that are this model's own networks, given all or fewer of its groups of inputs. */
const OWN = new Set(['mlp', 'ensemble']);

/**
 * @typedef {{ kind: string, label: string, columns: number[], scale: number | null }} InputGroup
 *   `columns` are indices into `ModelCard.columns`; `scale` is the size of the region a bend group averages over,
 *   its farthest offset, as the engine reads the regional grid here and one cell up and downstream.
 * @typedef {{ kinds: string[], groups: InputGroup[], fixed: boolean }} InputUnit
 *   Kinds the compared networks were always given together, switched as one; `fixed` when every one had them.
 */

/**
 * The groups in the order the networks first meet them.
 * @param {import('../../engine/result.js').ModelColumn[]} columns
 * @returns {InputGroup[]}
 */
export function inputGroups(columns) {
	const groups = new Map();
	columns.forEach((column, index) => {
		if (!(column.kind in KIND_COLORS)) throw new Error(`Unknown kind of model input: ${column.kind}`);
		if (!groups.has(column.group)) groups.set(column.group, { kind: column.kind, label: column.group, columns: [], scale: null });
		groups.get(column.group).columns.push(index);
	});
	for (const group of groups.values()) {
		if (group.kind === 'bend') group.scale = Math.max(...group.columns.map((i) => Math.abs(columns[i].offset[0])));
	}
	return [...groups.values()];
}

/** @returns {InputUnit[]} */
export function inputUnits(comparison, groups) {
	const own = comparison.filter((row) => OWN.has(row.learner));
	const units = new Map();
	for (const group of groups) {
		const given = own.map((row) => row.inputs.includes(group.kind));
		const key = given.join();
		if (!units.has(key)) units.set(key, { kinds: [], groups: [], fixed: given.every(Boolean) });
		const unit = units.get(key);
		if (!unit.kinds.includes(group.kind)) unit.kinds.push(group.kind);
		unit.groups.push(group);
	}
	return [...units.values()];
}

/** The index of this model's own networks trained on exactly these kinds, or -1. */
function ownRowWith(comparison, kinds) {
	return comparison.findIndex((row) => OWN.has(row.learner) && row.inputs.length === kinds.size && row.inputs.every((k) => kinds.has(k)));
}

/** The diagram's bar of input groups: one segment per group, as wide as its share of the inputs. */
export function groupBar(ctx) {
	const { groups, given, act } = ctx;
	return h('div', { class: 'group-bar' },
		groups.map((g) => h('span', {
			class: 'group-segment',
			style: `flex: ${g.columns.length}; --kind: ${KIND_COLORS[g.kind]}`,
			'data-off': String(!given.has(g.kind)),
			onpointerenter: (e) => {
				act.focus(g.label);
				act.tip.show(e.currentTarget, h('span', {}, h('b', {}, g.label), ` ${g.columns.length}`));
			},
			onpointerleave: () => {
				act.focus(null);
				act.tip.hide();
			},
		})));
}

/** Each group with its count, under a switch for each set of groups the compared networks were given together. */
function unitList(ctx) {
	const { card, units, given, act } = ctx;
	return h('div', { class: 'input-units' }, units.map((unit) => {
		const on = unit.kinds.every((k) => given.has(k));
		const next = new Set(given);
		for (const k of unit.kinds) on ? next.delete(k) : next.add(k);
		const target = unit.fixed ? -1 : ownRowWith(card.comparison, next);
		return h('div', { class: 'input-unit', 'data-on': String(on) },
			unit.fixed
				? h('span', { class: 'input-switch', 'data-fixed': 'true', role: 'switch', 'aria-checked': 'true', 'aria-disabled': 'true', title: 'Given to every network compared' })
				: h('button', {
					type: 'button',
					class: 'input-switch',
					role: 'switch',
					'aria-checked': String(on),
					'aria-label': unit.groups.map((g) => g.label).join(', '),
					disabled: target < 0,
					title: target < 0 ? 'No network was trained on this combination' : null,
					onclick: () => act.select(target),
				}),
			h('div', { class: 'input-groups' }, unit.groups.map((g) => h('div', {
				class: 'input-group',
				style: `--kind: ${KIND_COLORS[g.kind]}`,
				onpointerenter: () => act.focus(g.label),
				onpointerleave: () => act.focus(null),
			}, h('i', { class: 'swatch' }), h('span', { class: 'input-name' }, g.label), h('span', { class: 'input-count num' }, g.columns.length)))));
	}));
}

/** A bar for a value on [0, top], with the value after it. */
function bar(value, top, text) {
	return h('span', { class: 'trade-bar' },
		h('span', { class: 'trade-track' }, h('span', { class: 'trade-fill', style: `width: ${(100 * Math.max(0, value)) / top}%` })),
		h('span', { class: 'trade-value num' }, text));
}

/**
 * Every compared model better than the mean, this model's own networks first from fewest groups to all, then the
 * others: each with the groups of inputs it was given, as on the diagram's bar, its cross-validated R² of Cp and its
 * pressure-drag error.
 */
function tradeRows(ctx) {
	const { card, groups, selected, act } = ctx;
	const rows = card.comparison.map((row, index) => ({ row, index })).filter(({ row }) => row.crossValidationR2 > 0);
	rows.sort((a, b) => (OWN.has(b.row.learner) - OWN.has(a.row.learner)) || (OWN.has(a.row.learner) ? a.row.inputs.length - b.row.inputs.length : 0));
	const worst = Math.max(...rows.map(({ row }) => dragCounts(row.dragMae)));
	return h('div', { class: 'trade', role: 'listbox', 'aria-label': 'Models compared in training' },
		h('div', { class: 'trade-head' }, h('span'), h('span', {}, 'R²'), h('span', {}, h('span', { class: 'symbol' }, 'C', h('sub', {}, 'D,p')), ' error, counts')),
		rows.map(({ row, index }) => h('button', {
			type: 'button',
			class: 'trade-row',
			role: 'option',
			'aria-selected': String(index === selected),
			title: row.label,
			onclick: () => act.select(index),
		},
		h('span', { class: 'trade-name' },
			h('span', { class: 'trade-inputs' }, groups.map((g) => h('i', {
				style: `flex: ${g.columns.length}; --kind: ${KIND_COLORS[g.kind]}`,
				'data-off': String(!row.inputs.includes(g.kind)),
			}))),
			LEARNERS[row.learner] ?? null),
		bar(row.crossValidationR2, 1, row.crossValidationR2.toFixed(3)),
		bar(dragCounts(row.dragMae), worst, String(dragCounts(row.dragMae))))));
}

/** The groups and their switches, and how every compared model scored. */
export function inputsDetail(ctx) {
	return h('div', { class: 'inputs' },
		unitList(ctx),
		tradeRows(ctx),
		h('p', { class: 'caption' }, `Scored on the ${ctx.card.trainRuns.length} training cars, each predicted by networks that never saw it.`));
}
