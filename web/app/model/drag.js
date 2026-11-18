// Drag, the pressure summed over the car: the training cars' CFD drag beside this car's, this car's drag split by the
// way its surface faces, and every compared model's cross-validated drag error against guessing the training average.

import { h, s } from '../dom.js';
import { DRAG_COUNT, signed } from '../format.js';
import { bars, chips, section } from './charts.js';
import { copied, text } from './copy.js';
import { groupsGiven, learnerName } from './groups.js';
import { locator } from './overview.js';

/** The pressure drag coefficient's symbol, C with D,p below, for a placeholder to show. */
const DRAG_SYMBOL = h('span', { class: 'symbol' }, 'C', h('sub', {}, 'D,p'));

const WIDTH = 560;
const LEFT = 120;
const ROW = 40;

/** The training cars' CFD drag, their average, and the open car's prediction by each network, their mean and its CFD drag (or null). */
function dragLine(training, forces, cfdDrag) {
	const trained = training.cars.map((car) => car.cdP);
	const average = trained.reduce((a, b) => a + b, 0) / trained.length;
	const values = [...trained, ...forces.cdPModels, forces.cdP, ...(cfdDrag === null ? [] : [cfdDrag])];
	const lo = Math.floor(Math.min(...values) / 0.01) * 0.01;
	const hi = Math.ceil(Math.max(...values) / 0.01) * 0.01;
	const x = (v) => LEFT + ((v - lo) / (hi - lo)) * (WIDTH - LEFT - 12);
	const ticks = [];
	for (let v = lo; v <= hi + 1e-9; v += 0.01) ticks.push(v);
	const height = 2 * ROW + 36;
	return s('svg', { class: 'drag-line', viewBox: `0 0 ${WIDTH} ${height}`, width: WIDTH, height, role: 'img', 'aria-label': text('drag.line.caption', { symbol: 'CD,p' }) },
		ticks.map((v) => [
			s('line', { x1: x(v), y1: 6, x2: x(v), y2: 2 * ROW + 8, class: 'grid' }),
			s('text', { x: x(v), y: 2 * ROW + 24, 'text-anchor': 'middle' }, v.toFixed(2)),
		]),
		copied(s('text', { x: 0, y: ROW / 2 + 10, class: 'row-name' }), 'drag.line.training'),
		trained.map((v) => s('circle', { cx: x(v), cy: ROW / 2 + 6, r: 5, class: 'training-drag' })),
		s('line', { x1: x(average), y1: ROW / 2 - 8, x2: x(average), y2: ROW / 2 + 20, class: 'average' }),
		copied(s('text', { x: x(average), y: ROW / 2 - 11, 'text-anchor': 'middle', class: 'average-label' }), 'drag.line.average'),
		copied(s('text', { x: 0, y: ROW * 1.5 + 10, class: 'row-name' }), 'drag.line.car'),
		forces.cdPModels.map((v) => s('line', { x1: x(v), y1: ROW * 1.5 - 2, x2: x(v), y2: ROW * 1.5 + 14, class: 'network-drag' })),
		s('circle', { cx: x(forces.cdP), cy: ROW * 1.5 + 6, r: 5.5, class: 'predicted-drag' }),
		cfdDrag === null ? null : s('path', { d: `M${x(cfdDrag)} ${ROW * 1.5 - 1} l7 7 l-7 7 l-7 -7 z`, class: 'cfd-drag' }));
}

/** This car's predicted drag as the sum of what its rearward-facing and its forward-facing surfaces contribute. */
function facingSplit({ cdP, byFacing }) {
	const rows = [
		{ key: 'drag.facing.rearward', value: byFacing.rearward },
		{ key: 'drag.facing.forward', value: byFacing.forward },
		{ key: 'drag.facing.sum', value: cdP, sum: true },
	];
	const lo = Math.min(0, ...rows.map((r) => r.value));
	const hi = Math.max(0, ...rows.map((r) => r.value));
	const x = (v) => (100 * (v - lo)) / (hi - lo);
	return h('div', { class: 'facing-split' }, rows.map((row) => h('div', { class: `facing-row${row.sum ? ' sum' : ''}` },
		copied(h('span', { class: 'bar-label' }), row.key),
		h('span', { class: 'facing-track' },
			h('i', { class: 'facing-zero', style: `left: ${x(0)}%` }),
			h('i', { class: 'facing-fill', style: `left: ${x(Math.min(0, row.value))}%; width: ${x(Math.max(0, row.value)) - x(Math.min(0, row.value))}%` })),
		h('span', { class: 'bar-value num' }, signed(row.value, 3)))));
}

/**
 * @param {{ card: import('../../engine/result.js').ModelCard, groups: import('./groups.js').InputGroup[],
 *   training: { cars: { cdP: number }[] } }} options
 */
export function createDrag({ card, groups, training }) {
	const line = h('div');
	const split = h('div');
	const thisCar = h('p', { class: 'small-number num' });
	const rows = card.comparison.map((row) => ({
		label: learnerName(row, groups),
		chips: chips(groups, groupsGiven(groups, row)),
		value: row.dragMae / DRAG_COUNT,
		text: `${Math.round(row.dragMae / DRAG_COUNT)}`,
		strong: row.learner === 'ensemble' || row.learner === 'constant',
	}));
	const max = Math.max(...rows.map((r) => r.value));
	const counts = (learner) => Math.round(card.comparison.find((row) => row.learner === learner).dragMae / DRAG_COUNT);

	const el = section('drag', { model: counts('ensemble'), average: counts('constant') },
		h('div', { class: 'drag-layout' },
			h('figure', { class: 'figure' },
				copied(h('figcaption'), 'drag.line.caption', { symbol: DRAG_SYMBOL }),
				line,
				h('p', { class: 'drag-key copy-block' },
					h('span', { class: 'copy-block' }, h('i', { class: 'key-network' }), copied(h('span'), 'drag.key.network')),
					h('span', { class: 'copy-block' }, h('i', { class: 'key-mean' }), copied(h('span'), 'drag.key.mean')),
					h('span', { class: 'copy-block' }, h('i', { class: 'key-cfd' }), copied(h('span'), 'drag.key.cfd'))),
				thisCar,
				copied(h('figcaption'), 'drag.facing.caption'),
				split),
			h('figure', { class: 'figure' },
				copied(h('figcaption'), 'drag.learners.caption', { count: DRAG_COUNT }),
				bars(rows, max, 'drag.learners.inputs'))));
	el.prepend(locator('drag'));

	return {
		el,
		setCase({ result }) {
			const { forces } = result;
			const cfdDrag = result.referenceTotals?.coefficients[0] ?? null;
			line.replaceChildren(dragLine(training, forces, cfdDrag));
			split.replaceChildren(facingSplit(forces));
			const compared = cfdDrag === null ? {} : { cfd: cfdDrag.toFixed(3), counts: signed(Math.round((forces.cdP - cfdDrag) / DRAG_COUNT), 0) };
			copied(thisCar, cfdDrag === null ? 'drag.car.predicted' : 'drag.car.compared', { predicted: forces.cdP.toFixed(3), ...compared });
		},
		setPoint() {},
	};
}
