// The model's inputs: their groups, the three layers the story tells them in (the sets of groups the compared
// networks were always given together), the strip of a group's values at the point, and the chart of what each
// layer bought in training.

import { h } from '../dom.js';
import { DRAG_COUNT, signed } from '../format.js';

/** Each kind of input's colour, here and on the car. */
export const KIND_COLORS = {
	position: '#0d2233',
	bend: '#9a55e8',
	slice: '#0b6fbf',
	neighbour: '#197b80',
	outline: '#a35b1a',
	flag: '#237f50',
};

/** The learners that are this model's own kind of network, given all or fewer of its layers of inputs. */
const OWN = new Set(['mlp', 'ensemble']);

/** A pressure-drag error in counts, thousandths of the drag coefficient. */
export const dragCounts = (error) => Math.round(error / DRAG_COUNT);

/**
 * @typedef {{ kind: string, label: string, columns: number[], scale: number | null }} InputGroup
 *   `columns` are indices into `ModelCard.columns`; `scale` is a bend group's voxel size, its farthest offset.
 * @typedef {{ kinds: string[], groups: InputGroup[], columns: number[] }} InputLayer
 *   kinds the compared networks were always given together, told as one layer of the story
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

/**
 * The layers of inputs: groups whose kinds every compared network of this model's kind was given or not together.
 * @returns {InputLayer[]}
 */
export function inputLayers(comparison, groups) {
	const own = comparison.filter((row) => OWN.has(row.learner));
	const layers = new Map();
	for (const group of groups) {
		const key = own.map((row) => row.inputs.includes(group.kind)).join();
		if (!layers.has(key)) layers.set(key, { kinds: [], groups: [], columns: [] });
		const layer = layers.get(key);
		if (!layer.kinds.includes(group.kind)) layer.kinds.push(group.kind);
		layer.groups.push(group);
		layer.columns.push(...group.columns);
	}
	return [...layers.values()];
}

/** One group's values at the point, a cell per input coloured as the network reads it. */
export function groupStrip(group, act) {
	const cells = group.columns.map((i) => h('i', {
		class: 'cell',
		'data-column': i,
		onpointerenter: (e) => act.hoverColumns([i], e.currentTarget),
		onpointerleave: () => act.hoverColumns(null),
	}));
	return h('div', { class: 'input-row', style: `--kind: ${KIND_COLORS[group.kind]}` },
		h('span', {
			class: 'input-label',
			onpointerenter: (e) => act.hoverColumns(group.columns, e.currentTarget),
			onpointerleave: () => act.hoverColumns(null),
		}, h('i', { class: 'swatch' }), group.label),
		h('span', { class: 'input-count num' }, group.columns.length),
		h('span', { class: 'input-cells' }, cells));
}

/**
 * The compared models, as rows of the chart: this model's kind of network given one layer, then more, then all
 * (this model), then the other learners; the constant baseline is the chart's reference, not a row.
 */
export function comparisonRows(comparison, layers) {
	const has = (row, layer) => layer.kinds.every((k) => row.inputs.includes(k));
	const own = comparison.filter((row) => OWN.has(row.learner))
		.sort((a, b) => a.inputs.length - b.inputs.length || (a.learner === 'ensemble') - (b.learner === 'ensemble'));
	const rows = own.map((row) => ({ row, layers: layers.map((layer) => has(row, layer)), own: true }));
	const others = comparison.filter((row) => !OWN.has(row.learner) && row.learner !== 'constant');
	rows.push(...others.map((row) => ({ row, layers: layers.map((layer) => has(row, layer)), own: false })));
	return { rows, baseline: comparison.find((row) => row.learner === 'constant') ?? null };
}

/** A value on [0, top] as a bar with the value after it; `reference` marks another value on the same scale. */
export function bar(value, top, text, reference = null) {
	return h('span', { class: 'chart-bar' },
		h('span', { class: 'chart-track' },
			h('span', { class: 'chart-fill', style: `width: ${(100 * Math.max(0, value)) / top}%` }),
			reference === null ? null : h('span', { class: 'chart-reference', style: `left: ${(100 * reference) / top}%` })),
		h('span', { class: 'chart-value num' }, text));
}

/** A value as the column names it: distances in metres, flags as yes or no, the rest as numbers. */
export function columnValue(column, value) {
	if (column.kind === 'flag') return value >= 0.5 ? 'yes' : 'no';
	return column.unit ? `${signed(value, 3)} ${column.unit}` : signed(value, 3);
}
