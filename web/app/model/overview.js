// The whole model in one line, at the top of the page: a new car, one of its cells, what the cell is told, the five
// networks fitted to the ten training cars, their answer and its spread, the pressure on every cell, and the car's
// pressure drag summed from it. Each step but the last links to the section below that opens it up, and each section
// carries the line in small with its own steps marked, so the page reads as this one diagram taken apart.

import { h, s } from '../dom.js';
import { count, signed } from '../format.js';
import { countWord, pictureFrame, runOf } from './charts.js';
import { copied } from './copy.js';
import { closeUp } from './drawings.js';
import { trainingCar } from './training.js';

/**
 * The steps of the line, in order, each named at `overview.steps.<id>`, and the section that opens it up: the car's
 * pressure drag, summed from the pressure, has none.
 */
export const STEPS = [
	{ id: 'car', section: 'accuracy' },
	{ id: 'cell', section: 'point' },
	{ id: 'inputs', section: 'inputs' },
	{ id: 'networks', section: 'networks' },
	{ id: 'answer', section: 'spread' },
	{ id: 'field', section: 'accuracy' },
	{ id: 'drag', section: null },
];

/** The line in small, a dot per step a section opens up, with the steps of section `sectionId` filled in. */
export function locator(sectionId) {
	const opened = STEPS.filter((step) => step.section);
	const gap = 22;
	const width = gap * (opened.length - 1) + 12;
	return s('svg', { class: 'locator', viewBox: `0 0 ${width} 12`, width, height: 12, 'aria-hidden': 'true' },
		s('line', { x1: 6, y1: 6, x2: width - 6, y2: 6 }),
		opened.map((step, i) => s('circle', { cx: 6 + i * gap, cy: 6, r: step.section === sectionId ? 5 : 3.5, class: step.section === sectionId ? 'here' : '' })));
}

/** The five networks' Cp at the point as ticks, their mean as a dot, on a short scale around them. */
function answerGlyph({ cpModels, cp }) {
	const lo = Math.min(...cpModels);
	const hi = Math.max(...cpModels);
	const span = Math.max(hi - lo, 0.04);
	const x = (v) => 58 + ((v - (lo + hi) / 2) / span) * 70;
	return s('svg', { class: 'answer-glyph', viewBox: '0 0 116 40', 'aria-hidden': 'true' },
		s('line', { x1: 4, y1: 22, x2: 112, y2: 22, class: 'axis' }),
		cpModels.map((v) => s('line', { x1: x(v), y1: 12, x2: x(v), y2: 32, class: 'tick' })),
		s('circle', { cx: x(cp), cy: 22, r: 5, class: 'mean' }));
}

/** Five networks as a stack of cards, the front one showing a network's layers of units, every unit joined to the next layer's. */
function networksGlyph(networks) {
	const count = networks.length;
	const [w, hgt, offset] = [70, 52, 5];
	// Units per layer, drawn few: the inputs, the hidden layers and the one output.
	const layers = networks[0].layers.map((_, l, all) => (l === all.length - 1 ? 1 : l === 0 ? 4 : 5));
	const x = (l) => 10 + (l * (w - 20)) / (layers.length - 1);
	const y = (i, n) => hgt / 2 + (i - (n - 1) / 2) * 8.5;
	const front = s('g', { class: 'glyph-network' });
	layers.slice(1).forEach((n, l) => {
		for (let a = 0; a < layers[l]; a++) for (let b = 0; b < n; b++) front.append(s('line', { x1: x(l), y1: y(a, layers[l]), x2: x(l + 1), y2: y(b, n) }));
	});
	layers.forEach((n, l) => {
		for (let i = 0; i < n; i++) front.append(s('circle', { cx: x(l), cy: y(i, n), r: 2.6 }));
	});
	const cards = Array.from({ length: count }, (_, k) => {
		const back = count - 1 - k;
		return s('g', { transform: `translate(${back * offset}, ${(count - 1 - back) * offset})` },
			s('rect', { class: 'glyph-card', x: 0.5, y: 0.5, width: w, height: hgt, rx: 5 }),
			back === 0 ? front : null);
	});
	return s('svg', { class: 'network-glyph', viewBox: `0 0 ${w + 1 + offset * (count - 1)} ${hgt + 1 + offset * (count - 1)}`, 'aria-hidden': 'true' }, cards);
}

/**
 * @param {{ card: import('../../engine/result.js').ModelCard, groups: import('./groups.js').InputGroup[],
 *   training: { step: number, cars: object[] } }} options
 */
export function createOverview({ card, groups, training }) {
	const parts = Object.fromEntries(STEPS.map((step) => [step.id, h('div', { class: 'step-figure' })]));
	const numbers = Object.fromEntries(STEPS.map((step) => [step.id, h('span', { class: 'step-number num' })]));
	const networks = { networks: card.networks.length, networksWord: countWord(card.networks.length) };
	const steps = STEPS.map((step) => h(step.section ? 'a' : 'div', { class: `step step-${step.id}`, href: step.section && `#model-${step.section}` },
		parts[step.id], copied(h('span', { class: 'step-name' }), `overview.steps.${step.id}`, networks), numbers[step.id]));
	const line = h('div', { class: 'overview-line' }, steps.flatMap((step, i) => (i ? [h('span', { class: 'step-arrow', 'aria-hidden': 'true' }), step] : [step])));

	// The training cars the networks are fitted to.
	const grid = h('div', { class: 'training-grid' });
	const fitted = h('a', { class: 'overview-training', href: '#model-networks' },
		grid,
		copied(h('span', { class: 'step-name' }), 'overview.training.name', { cars: training.cars.length }),
		copied(h('span', { class: 'step-number num' }), 'overview.training.sampled', { cells: count(card.samplesPerCar) }));
	const loopLabel = h('span', { class: 'loop-label num' });
	const loop = h('div', { class: 'overview-loop' }, loopLabel);

	// What the cell is told: a bar per group, as long as its count of inputs.
	parts.inputs.append(
		h('div', { class: 'input-bar' }, groups.map((group) => h('span', { class: 'input-part', style: `--group: ${group.color}; flex: ${group.columns.length}` }))),
		h('div', { class: 'input-legend copy-block' }, groups.map((group) => h('span', { class: 'copy-block', style: `--group: ${group.color}` },
			h('i'), copied(h('span'), `overview.legend.${group.words}`, { count: h('span', { class: 'num' }, group.columns.length) })))));
	copied(numbers.inputs, 'overview.steps.inputs.number', { count: card.columns.length });
	parts.networks.append(networksGlyph(card.networks));
	numbers.networks.textContent = card.networks[0].layers.join('-');
	const carFrame = pictureFrame('overview-car');
	const fieldFrame = pictureFrame('overview-field');
	parts.car.append(carFrame.el);
	parts.field.append(fieldFrame.el);
	copied(numbers.drag, 'overview.steps.drag.number');

	const view = { car: null, point: null };
	const overview = {
		el: h('div', { class: 'overview' }, fitted, line, loop),
		/** The car: its pictures (clay and predicted Cp from the side), its numbers and its colour maps. */
		setCar(car) {
			view.car = car;
			const { result, pictures, maps } = car;
			grid.replaceChildren(...training.cars.map((trained) => trainingCar(trained, maps.cp, 'solid')));
			copied(numbers.car, 'car.outside', { run: runOf(result) });
			fieldFrame.show(pictures.fieldSide);
			copied(numbers.field, 'overview.steps.field.number', { cells: count(result.source.cells) });
			copied(loopLabel, 'overview.loop', { cells: count(result.source.cells) });
			parts.drag.replaceChildren(h('span', { class: 'drag-value num' }, h('span', { class: 'symbol' }, 'C', h('sub', {}, 'D,p')), ` ${result.forces.cdP.toFixed(3)}`));
			overview.setPoint(view.point);
		},
		/** The point the page explains: what the engine said there, or why it could not. */
		setPoint(point) {
			view.point = point;
			if (!view.car || !point?.explanation) return;
			const { explanation } = point;
			const { pictures, cutter } = view.car;
			const [u, v] = pictures.claySide.place(explanation.position);
			carFrame.show(pictures.claySide, h('i', { class: 'picture-point', style: `left: ${100 * u}%; top: ${100 * v}%` }));
			parts.cell.replaceChildren(closeUp({ width: 96, explanation, cutter }));
			parts.answer.replaceChildren(answerGlyph(explanation));
			copied(numbers.answer, 'overview.steps.answer.number', { cp: signed(explanation.cp, 2), spread: explanation.sigma.toFixed(2) });
		},
	};
	return overview;
}
