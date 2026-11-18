// Why the model reads one point at a time: ten cars are ten examples for a model that reads a whole car, and ten
// times the cells sampled from each for a model that reads a point and the shape around it.

import { h } from '../dom.js';
import { count } from '../format.js';
import { countWord, section } from './charts.js';
import { copied } from './copy.js';
import { locator } from './overview.js';
import { trainingCar } from './training.js';

/**
 * @param {{ card: import('../../engine/result.js').ModelCard, training: { cars: object[] } }} options
 */
export function createPointwise({ card, training }) {
	const cars = training.cars.length;
	const examples = cars * card.samplesPerCar;
	const wholeCars = h('div', { class: 'example-grid' });
	const cells = h('div', { class: 'example-cells' });

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
				copied(h('p', { class: 'small-number num' }), 'point.cells.note', { cells: count(card.samplesPerCar), cars: count(cars) }))));
	el.prepend(locator('point'));

	return {
		el,
		setCar({ maps }) {
			wholeCars.replaceChildren(...training.cars.map((car) => h('span', { class: 'example' }, trainingCar(car, maps.cp, 'solid'))));
			cells.replaceChildren(trainingCar(training.cars[0], maps.cp, 'dots'));
		},
		setPoint() {},
	};
}
