// Why the model reads one point at a time: ten cars are ten examples for a model that reads a whole car, and ten
// times the cells sampled from each for a model that reads a point and the shape around it. The measured evidence is
// the compared models' accuracy on cars they did not see.

import { h } from '../dom.js';
import { count } from '../format.js';
import { bars, chips, countWord, section } from './charts.js';
import { copied } from './copy.js';
import { groupsGiven } from './groups.js';
import { locator } from './overview.js';
import { trainingCar } from './training.js';

/**
 * @param {{ card: import('../../engine/result.js').ModelCard, groups: import('./groups.js').InputGroup[],
 *   training: { cars: object[] } }} options
 */
export function createPointwise({ card, groups, training }) {
	const cars = training.cars.length;
	const examples = cars * card.samplesPerCar;
	const wholeCars = h('div', { class: 'example-grid' });
	const cells = h('div', { class: 'example-cells' });

	// A learned summary of the whole car added to a point's position, the point's position alone, and this model.
	const summary = card.comparison.find((row) => row.learner === 'pointnet');
	const same = summary && card.comparison.find((row) => row.learner === 'mlp' && row.inputs.join() === summary.inputs.join());
	const shipped = card.comparison.find((row) => row.learner === 'ensemble');
	const rows = [
		summary && { label: 'point.chart.summary', row: summary },
		same && { label: 'point.chart.alone', row: same },
		{ label: 'point.chart.shape', row: shipped, strong: true },
	].filter(Boolean);
	const chart = h('figure', { class: 'figure' },
		copied(h('figcaption'), 'point.chart.caption'),
		bars(rows.map(({ label, row, strong }) => ({
			label: [label],
			chips: chips(groups, groupsGiven(groups, row)),
			value: row.crossValidationR2,
			text: row.crossValidationR2.toFixed(2),
			strong,
		})), 1, 'point.chart.inputs'));

	const el = section('point', { cars: count(cars), carsWord: countWord(cars), examples: count(examples) },
		h('div', { class: 'figure-pair' },
			h('figure', { class: 'figure' },
				copied(h('figcaption'), 'point.cars.caption'),
				wholeCars,
				h('p', { class: 'big-number num' }, count(cars))),
			h('figure', { class: 'figure' },
				copied(h('figcaption'), 'point.cells.caption'),
				cells,
				h('p', { class: 'big-number num' }, count(examples)),
				copied(h('p', { class: 'small-number num' }), 'point.cells.note', { cells: count(card.samplesPerCar), cars: count(cars) }))),
		chart);
	el.prepend(locator('point'));

	return {
		el,
		setCase({ maps }) {
			wholeCars.replaceChildren(...training.cars.map((car) => h('span', { class: 'example' }, trainingCar(car, maps.cp, 'solid'))));
			cells.replaceChildren(trainingCar(training.cars[0], maps.cp, 'dots'));
		},
		setPoint() {},
	};
}
