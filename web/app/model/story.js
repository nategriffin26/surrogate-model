// The model view's column: why the model is built as it is, read top to bottom, with the picked point's values
// in every figure. Built once per model card; `setCase` fills what depends on the car, `update` what depends on
// the point, so dragging the point changes values in place.

import { h, put, s } from '../dom.js';
import { count, signed } from '../format.js';
import { appliesBlock, unseenBlock } from './evidence.js';
import { KIND_COLORS, bar, columnValue, comparisonRows, dragCounts, groupStrip } from './inputs.js';
import { NetworkFigure, shade } from './network.js';

const WORDS = ['no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve'];
const word = (n) => WORDS[n] ?? String(n);
const capital = (text) => text[0].toUpperCase() + text.slice(1);
/** Values as a list in a sentence, with their unit: "a, b, c and d cm". */
const listOf = (values, unit) => `${values.slice(0, -1).join(', ')}${values.length > 1 ? ' and ' : ''}${values.at(-1)} ${unit}`;
const centimetres = (metres) => Math.round(metres * 100);
const distinct = (values) => [...new Set(values)].sort((a, b) => a - b);

const SYMBOL_CDP = () => h('span', { class: 'symbol' }, 'C', h('sub', {}, 'D,p'));
/** The other learners compared in training, by `learner`, in the terms of the story. */
const OTHER_LEARNERS = {
	pointnet: 'A network that learns its own summary of the whole car',
	trees: 'Boosted trees on the same numbers',
};

/**
 * What each of the three layers of inputs is told as, by the first kind in it: the drawing layers that show it
 * on the car, its title and its sentence, written from the card's own distances and sizes.
 */
function layerStories(card, groups) {
	const offsets = (kind, axis) => distinct(card.columns.filter((c) => c.kind === kind && c.offset[axis]).map((c) => centimetres(Math.abs(c.offset[axis]))));
	const scales = groups.filter((g) => g.kind === 'bend').map((g) => centimetres(g.scale));
	const reach = Math.max(...card.columns.filter((c) => c.kind === 'slice').map((c) => Math.abs(c.offset[0])));
	return {
		position: {
			drawn: ['position'],
			title: 'Where it is',
			text: 'How far it is from the nose and from the tail, how far it lies from the centre plane and on which side, its height above the lowest point of the car, and which way the surface faces there.',
		},
		bend: {
			drawn: ['bend'],
			title: 'How the surface bends around it',
			text: `In boxes of ${listOf(scales, 'cm')} around the point: which way the surface inside faces on average, how much surface there is, and how the point’s own facing differs; the same again one box upstream and one downstream. The small boxes see the edge the point sits on, the large ones the hood or the roof. The tint on the car is how much each part of the surface counts.`,
		},
		slice: {
			drawn: ['slice', 'neighbour', 'outline'],
			title: 'The car’s outline along the flow',
			text: `Where the car’s top and bottom are in the streamwise cut through the point, out to ${reach} m ahead and behind; the same in cuts ${listOf(offsets('neighbour', 1), 'cm')} to either side; how far out the car reaches on the point’s side, at the point’s height and ${listOf(offsets('outline', 2), 'cm')} above and below it; and whether there is car at each of those places. It tells the point what the air has passed on its way there, and what lies behind.`,
		},
	};
}

/** Values on one short scale: each network a tick, the mean, σ as a band around it, CFD as a diamond. */
function valueStrip({ width, values, mean, spread, cfd, label }) {
	const all = [...values, mean, ...(cfd === null ? [] : [cfd]), mean - spread, mean + spread];
	const lo = Math.min(...all), hi = Math.max(...all);
	const pad = Math.max((hi - lo) * 0.12, 0.004);
	const x = (v) => 6 + ((v - (lo - pad)) / (hi - lo + 2 * pad)) * (width - 12);
	return s('svg', { class: 'value-strip', width, height: 34, viewBox: `0 0 ${width} 34`, role: 'img', 'aria-label': label },
		s('line', { class: 'track', x1: 6, x2: width - 6, y1: 17, y2: 17 }),
		s('rect', { class: 'spread', x: x(mean - spread), y: 9, width: Math.max(1, x(mean + spread) - x(mean - spread)), height: 16, rx: 2 }, s('title', {}, `σ ${spread.toFixed(3)}`)),
		values.map((v, k) => s('line', { class: 'tick', x1: x(v), x2: x(v), y1: 10, y2: 24 }, s('title', {}, `Network ${k + 1}: ${signed(v, 3)}`))),
		s('line', { class: 'mean', x1: x(mean), x2: x(mean), y1: 5, y2: 29 }, s('title', {}, `Mean ${signed(mean, 3)}`)),
		cfd === null ? null : s('path', { class: 'cfd', d: `M${x(cfd)} 10l5 7-5 7-5-7z` }, s('title', {}, `CFD ${signed(cfd, 3)}`)));
}

/** The car's side outline from its centre-plane cut, with the point on it. */
function sideOutline(cut, bounds, width) {
	const [xmin, xmax, , , zmin, zmax] = bounds;
	const scale = (width - 8) / (xmax - xmin);
	const height = Math.ceil((zmax - zmin) * scale) + 8;
	const X = (x) => 4 + (x - xmin) * scale;
	const Z = (z) => height - 4 - (z - zmin) * scale;
	let d = '';
	for (let i = 0; i < cut.length; i += 6) d += `M${X(cut[i]).toFixed(1)} ${Z(cut[i + 2]).toFixed(1)}L${X(cut[i + 3]).toFixed(1)} ${Z(cut[i + 5]).toFixed(1)}`;
	const point = s('circle', { class: 'side-outline-point', r: 5, visibility: 'hidden' });
	const svg = s('svg', { class: 'side-outline', width, height, viewBox: `0 0 ${width} ${height}`, 'aria-hidden': 'true' }, s('path', { d }), point);
	const place = ([x, , z]) => {
		point.setAttribute('cx', X(x));
		point.setAttribute('cy', Z(z));
		point.setAttribute('visibility', 'visible');
	};
	return { svg, place };
}

/**
 * @param {{ card: import('../../engine/result.js').ModelCard, groups: import('./inputs.js').InputGroup[],
 *   layers: import('./inputs.js').InputLayer[], act: object }} options
 */
export function createStory({ card, groups, layers, act }) {
	const stories = layerStories(card, groups);
	const storyOf = (layer) => {
		if (!(layer.kinds[0] in stories)) throw new Error(`The model card groups its inputs in a layer the view does not tell: ${layer.kinds.join(', ')}`);
		return stories[layer.kinds[0]];
	};
	const total = card.trainRuns.length * card.samplesPerCar;
	const heldOut = card.networks[0].validationRuns.length;
	const width = () => limit.clientWidth;

	// The point, its answer, and how to move it, kept in view while the column scrolls.
	const coords = h('span', { class: 'readout-coords num' });
	const at = h('span', { class: 'at-tag' });
	const answer = h('span', { class: 'readout-answer num' });
	const failure = h('p', { class: 'point-failure', role: 'alert', hidden: true });
	const readout = h('header', { class: 'readout' },
		h('div', { class: 'readout-line' }, h('span', { class: 'readout-title' }, 'Point'), coords, at),
		h('div', { class: 'readout-line' }, answer),
		failure,
		h('p', { class: 'readout-hint' }, h('i', { class: 'handle-glyph', 'aria-hidden': 'true' }), 'Drag the point across the car, or click anywhere on it.'));

	const section = (id, title, ...content) => h('section', { class: 'story-section', 'data-section': id }, h('h2', {}, title), ...content);
	const prose = (text) => h('p', { class: 'prose' }, text);

	// Ten cars, one point at a time.
	const outlineHost = h('div', { class: 'side-outline-host' });
	const limit = section('limit', `${capital(word(card.trainRuns.length))} cars, one point at a time`,
		prose(`The model learned from ${word(card.trainRuns.length)} ${card.dataset.name} cars. That is too few to learn what a car is, so it never looks at a whole car. It predicts the pressure at one point of the surface at a time, from a description of the shape around that point, and each training car gave it ${count(card.samplesPerCar)} points to learn from.`),
		outlineHost,
		h('div', { class: 'stat-line num' },
			h('span', {}, h('b', {}, card.trainRuns.length), ' cars'), h('span', { class: 'times' }, '×'),
			h('span', {}, h('b', {}, count(card.samplesPerCar)), ' points'), h('span', { class: 'times' }, '='),
			h('span', {}, h('b', {}, count(total)), ' examples')));

	// What the point is told: the three layers, each drawn on the car.
	const positionValues = h('dl', { class: 'position-values num' });
	const told = section('told', 'What the point is told',
		prose(`${count(card.columns.length)} numbers, in ${word(layers.length)} layers. Each is drawn on the car; point at a drawing or a number to see which is which.`),
		layers.map((layer) => {
			const story = storyOf(layer);
			return h('div', { class: 'told-layer', 'data-drawn': story.drawn.join(' ') },
				h('h3', {}, story.title, h('span', { class: 'count num' }, layer.columns.length)),
				prose(story.text),
				layer.kinds.includes('position') ? positionValues : null,
				h('div', { class: 'input-rows' }, layer.groups.map((group) => groupStrip(group, act))));
		}));

	// What each layer bought.
	const { rows, baseline } = comparisonRows(card.comparison, layers);
	const worst = Math.max(...rows.map(({ row }) => dragCounts(row.dragMae)), baseline ? dragCounts(baseline.dragMae) : 0);
	const rowLabel = ({ row, layers: given, own }) => {
		if (!own) {
			if (!(row.learner in OTHER_LEARNERS)) throw new Error(`The model card compares an unknown learner: ${row.learner}`);
			return OTHER_LEARNERS[row.learner];
		}
		const names = layers.filter((_, k) => given[k]).map((layer) => storyOf(layer).title.toLowerCase());
		if (row.learner === 'ensemble') return `All ${word(layers.length)}: this model`;
		return names.length === 1 ? `${capital(names[0])} only` : `+ ${names.slice(1).join(' + ')}`;
	};
	const bought = section('bought', 'What each layer bought',
		prose(`Each model below was trained the same way and scored on training cars it never saw: pressure accuracy as R² of Cp over the surface (1 is exact), and the error in pressure drag in counts (0.001 of the drag coefficient).`),
		h('div', { class: 'chart', role: 'table', 'aria-label': 'Models compared in training' },
			h('div', { class: 'chart-head', role: 'row' }, h('span', { role: 'columnheader' }), h('span', { role: 'columnheader' }, 'R² of Cp'),
				h('span', { role: 'columnheader' }, SYMBOL_CDP(), ' error, counts')),
			rows.map((entry) => h('div', { class: 'chart-row', role: 'row', 'data-own': String(entry.own), 'data-shipped': String(entry.row.learner === 'ensemble') },
				h('span', { class: 'chart-name', role: 'rowheader' },
					h('span', { class: 'layer-chips' }, layers.map((layer, k) => h('i', { style: `--kind: ${KIND_COLORS[layer.kinds[0]]}`, 'data-off': String(!entry.layers[k]) }))),
					rowLabel(entry)),
				bar(entry.row.crossValidationR2, 1, entry.row.crossValidationR2.toFixed(3)),
				bar(dragCounts(entry.row.dragMae), worst, String(dragCounts(entry.row.dragMae)), baseline ? dragCounts(baseline.dragMae) : null))),
			baseline ? h('p', { class: 'chart-key' }, h('i', { class: 'reference-key' }), `Predicting every car’s mean drag: ${dragCounts(baseline.dragMae)} counts`) : null),
		prose(`The bend brings the detail in pressure; the outline along the flow brings the drag error down. All ${word(layers.length)} together give the best pressure, but a larger drag error than the outline alone, and larger than predicting every car’s mean drag. A network that learns its own summary of the whole car memorises ${word(card.trainRuns.length)} cars and does worst on pressure. Boosted trees given the same numbers come close on pressure, not on drag.`));

	// One network.
	const network = new NetworkFigure({ card, act });
	const picker = h('div', { class: 'network-picker', role: 'radiogroup', 'aria-label': 'Network' },
		card.networks.map((_, k) => h('button', { type: 'button', role: 'radio', 'aria-checked': 'false', onclick: () => act.network(k) }, `Network ${k + 1}`)));
	const inside = section('network', 'Inside one network',
		prose(`The ${count(card.columns.length)} numbers go into a network of ${word(card.networks[0].layers.length - 2)} layers of ${card.networks[0].layers[1]} units. Each unit adds up every number before it, each with a weight learned in training, and passes the sum on through a smooth threshold (${card.activation}); one number comes out, the pressure coefficient Cp at the point. Each row below is one unit, coloured by its value at your point: blue above zero, orange below.`),
		picker, network.el);

	// Five networks.
	const foldGrid = h('table', { class: 'fold-grid' },
		h('thead', {}, h('tr', {}, h('th', {}), card.trainRuns.map((run) => h('th', { scope: 'col' }, run)))),
		h('tbody', {}, card.networks.map((net, k) => h('tr', { 'data-network': k, onclick: () => act.network(k) },
			h('th', { scope: 'row' }, `Network ${k + 1}`),
			card.trainRuns.map((run) => {
				const fitted = net.trainRuns.includes(run);
				return h('td', { 'data-role': fitted ? 'fit' : 'held', title: `Run ${run}: ${fitted ? 'trained on' : 'never seen'}` }, h('i'));
			})))));
	const fiveStrip = h('div', { class: 'strip-host' });
	const fiveValues = h('div', { class: 'five-values num' });
	const five = section('five', `${capital(word(card.networks.length))} networks, each blind to ${word(heldOut)} cars`,
		prose(`${capital(word(card.networks.length))} networks were trained, each without ${word(heldOut)} of the ${word(card.trainRuns.length)} cars, so every training car is scored by a network that never saw it. The prediction is their mean. Where they disagree, the spread σ is large: σ is their standard deviation × ${card.spreadScale.toFixed(2)}, and on cars outside training CFD falls within σ of the mean on ${Math.round(card.spreadCoverage * 100)} % of the surface.`),
		h('div', { class: 'grid-key' }, h('span', { 'data-role': 'fit' }, h('i'), 'Trained on'), h('span', { 'data-role': 'held' }, h('i'), 'Never seen')),
		foldGrid, h('h3', { class: 'figure-title' }, 'Their answers at your point'), fiveStrip, fiveValues);

	// From pressure to drag.
	const dragValues = h('div', { class: 'five-values num' });
	const dragStrip = h('div', { class: 'strip-host' });
	const polygons = h('span');
	const drag = section('drag', 'From pressure to drag',
		h('p', { class: 'prose' }, 'Pressure drag adds up Cp over every polygon of the file, each counted by its area and by how squarely it faces the flow: ', polygons, '.'),
		h('div', { class: 'formula' }, SYMBOL_CDP(), ' = −Σ Cp ', h('span', { class: 'symbol' }, 'n', h('sub', {}, 'x')), ' A / ', h('span', { class: 'symbol' }, 'A', h('sub', {}, 'ref'))),
		dragValues, dragStrip);

	const unseenHost = h('div');
	const unseen = section('unseen', 'On cars it never saw', unseenHost);
	const appliesHost = h('div');
	const applies = section('applies', 'Where it applies', appliesHost);

	const el = h('div', { class: 'model' }, readout, limit, told, bought, inside, five, drag, unseen, applies);
	let place = null;
	let caseShown = null;
	let shown = null;

	/** Show what depends on the car: its outline, its drag, its scores and its size. */
	function setCase(result, cutter, compare) {
		caseShown = [result, cutter, compare];
		const { bounds } = result.geometry;
		const figure = sideOutline(cutter.cut(1, result.lookups.yCenter).positions, bounds, width());
		place = figure.place;
		put(outlineHost, figure.svg);
		put(polygons, `all ${count(result.source.cells)} of them`);
		put(unseenHost, unseenBlock({ card, result, compare, act }));
		put(appliesHost, appliesBlock({ card, result }));
		const f = result.forces;
		put(dragValues, h('span', {}, SYMBOL_CDP(), ` ${f.cdP.toFixed(3)}`), f.cdPCfd === null ? null : h('span', { class: 'side-value' }, h('i', {}, 'CFD'), f.cdPCfd.toFixed(3)));
		put(dragStrip, valueStrip({ width: width(), values: f.cdPModels, mean: f.cdP, spread: f.cdPStd, cfd: f.cdPCfd, label: 'Each network’s pressure drag' }));
		if (shown) update({ ...shown, sweep: false });
	}

	/**
	 * Show a point: `explanation` (or `failure`) as the engine gave it, CFD at the point if the file has it, the
	 * network the reader picked; `sweep` runs the new values through the network.
	 */
	function update(view) {
		shown = view;
		const { explanation, cfdAt, network: k, sweep } = view;
		failure.hidden = !view.failure;
		failure.textContent = view.failure ?? '';
		for (const [n, b] of [...picker.children].entries()) b.setAttribute('aria-checked', String(n === k));
		for (const row of foldGrid.tBodies[0].rows) row.dataset.selected = String(Number(row.dataset.network) === k);
		if (!explanation) return;
		const { position, at: where, index, cp, sigma, cpModels, activations, features } = explanation;
		put(coords, ['x', 'y', 'z'].map((axis, n) => h('span', {}, h('i', {}, axis), signed(position[n], 3))), h('span', { class: 'unit' }, 'm'));
		at.dataset.at = where;
		at.textContent = where === 'vertex' ? 'Mesh vertex' : `Polygon ${count(index)}`;
		put(answer, h('span', { class: 'readout-cp' }, h('i', {}, 'Cp'), signed(cp, 3)),
			h('span', { class: 'side-value' }, h('i', {}, 'σ'), sigma.toFixed(3)),
			cfdAt === null ? null : h('span', { class: 'side-value' }, h('i', {}, 'CFD'), signed(cfdAt, 3)));
		place?.(position);

		const input = activations[k][0];
		for (const cell of el.querySelectorAll('.cell')) {
			const i = Number(cell.dataset.column);
			cell.style.background = shade(input[i], 1.5);
		}
		put(positionValues, card.columns.flatMap((column, i) => column.kind === 'position' && column.along
			? [h('dt', {}, column.name), h('dd', {}, columnValue(column, features[i]))] : []),
		h('dt', {}, 'Facing'), h('dd', {}, explanation.normal.map((v) => signed(v, 2)).join(', ')));

		network.update(activations, cpModels, k, sweep);
		put(fiveStrip, valueStrip({ width: width(), values: cpModels, mean: cp, spread: sigma, cfd: cfdAt, label: 'Each network’s Cp at the point' }));
		put(fiveValues, cpModels.map((v, n) => h('span', { 'data-selected': String(n === k) }, h('i', {}, n + 1), signed(v, 3))));
	}

	/** Light these inputs' cells and rows, or none. */
	function highlight(columns) {
		const lit = new Set(columns ?? []);
		for (const cell of el.querySelectorAll('.cell')) cell.classList.toggle('lit', lit.has(Number(cell.dataset.column)));
		el.classList.toggle('lighting', lit.size > 0);
		network.highlight(columns);
	}

	/** Figures drawn to the column's width are drawn again when it changes. */
	function resize() {
		if (caseShown) setCase(...caseShown);
	}

	return { el, sections: [...el.querySelectorAll('.told-layer')], setCase, update, highlight, resize };
}
