// Why the model reads one point at a time: ten cars are ten examples for a model that reads a whole car, and ten
// times the cells sampled from each for a model that reads a point and the shape around it. The measured evidence is
// the compared models' accuracy on cars they did not see.

import { h } from '../dom.js';
import { count } from '../format.js';
import { bars, chips, countWord, section } from './charts.js';
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
		summary && { label: 'A point and a learned summary of the whole car', row: summary },
		same && { label: 'A point alone', row: same },
		{ label: 'A point and the shape around it', row: shipped, strong: true },
	].filter(Boolean);
	const chart = h('figure', { class: 'figure' },
		h('figcaption', {}, 'Accuracy on cars held out of training, R²'),
		bars(rows.map(({ label, row, strong }) => ({
			label,
			chips: chips(groups, groupsGiven(groups, row)),
			value: row.crossValidationR2,
			text: row.crossValidationR2.toFixed(2),
			strong,
		})), 1));

	const el = section('point', 'Learn one point at a time, not one car at a time',
		`${countWord(cars)} cars are ${countWord(cars).toLowerCase()} examples to a model that reads a whole car, and ${count(examples)} to one that reads a point and the shape around it.`,
		h('div', { class: 'figure-pair' },
			h('figure', { class: 'figure' },
				h('figcaption', {}, 'A car per example'),
				wholeCars,
				h('p', { class: 'big-number num' }, count(cars))),
			h('figure', { class: 'figure' },
				h('figcaption', {}, 'A cell per example'),
				cells,
				h('p', { class: 'big-number num' }, count(examples)),
				h('p', { class: 'small-number num' }, `${count(card.samplesPerCar)} cells from each of ${count(cars)} cars`))),
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
