// Whether the model can stand in for CFD: this car against its CFD when the file carries it, otherwise the runs it was
// never trained on, and the training cars each scored by the networks that never saw it; and where it applies: the
// data it was trained on and the size of the cars in it.

import { h, s } from '../dom.js';
import { DRAG_COUNT, counts, signed } from '../format.js';
import { icon } from '../icons.js';
import { EXAMPLES } from '../examples.js';

const DIMENSIONS = { length: 'Length', width: 'Width', height: 'Height' };

/** A pressure-drag error in counts, thousandths of the drag coefficient. */
export const dragCounts = (error) => Math.round(error / DRAG_COUNT);

export function evidenceBlock({ card, result, compare, act }) {
	const { accuracy, forces } = result;
	const rows = [];
	if (accuracy) {
		rows.push({ label: 'This car', r2: accuracy.r2, error: Math.abs(counts(forces.cdP, forces.cdPCfd)), kind: 'this' });
	} else {
		for (const run of card.outsideRuns) {
			const example = EXAMPLES.find((e) => e.id === `run_${run.run}`);
			rows.push({ label: `Run ${run.run}`, r2: run.r2, error: dragCounts(run.dragMae), kind: 'outside', example });
		}
	}
	rows.push({ label: `${card.trainRuns.length} training cars, each held out`, r2: card.crossValidation.meanR2, error: dragCounts(card.crossValidation.dragMae), kind: 'cv' });
	return h('section', { class: 'evidence' },
		h('div', { class: 'evidence-head' },
			h('h3', {}, 'Against CFD'),
			accuracy ? h('button', {
				type: 'button',
				class: 'tool compare-toggle',
				'aria-pressed': String(compare),
				onclick: () => act.compare(!compare),
			}, icon('compare'), 'Side by side') : null),
		h('table', { class: 'evidence-table' },
			h('thead', {}, h('tr', {},
				h('th', {}),
				h('th', { class: 'num' }, 'R²'),
				h('th', { class: 'num' }, h('span', { class: 'symbol' }, 'C', h('sub', {}, 'D,p')), ' error, counts'))),
			h('tbody', {}, rows.map(({ label, r2, error, kind, example }) => h('tr', { 'data-kind': kind },
				h('th', { scope: 'row' }, label, example ? h('button', {
					type: 'button',
					class: 'open-example',
					title: `Open ${example.label}, with its CFD`,
					onclick: () => act.openExample(example),
				}, 'Open') : null),
				h('td', { class: 'num' }, signed(r2, 3)),
				h('td', { class: 'num' }, String(error)))))));
}

/** A dimension on the training cars' range: the range as a band, this car as a tick, outside values marked. */
function rangeRow(name, value, { min, max }) {
	const outside = value < min || value > max;
	const lo = Math.min(min, value), hi = Math.max(max, value);
	const pad = (hi - lo) * 0.15;
	const W = 120;
	const x = (v) => 2 + ((v - (lo - pad)) / (hi - lo + 2 * pad)) * (W - 4);
	return h('div', { class: `row range${outside ? ' outside' : ''}`, title: `This car ${value.toFixed(3)} m` },
		h('span', { class: 'row-label' }, DIMENSIONS[name]),
		s('svg', { class: 'range-bar', viewBox: `0 0 ${W} 14`, 'aria-hidden': 'true' },
			s('rect', { x: x(min), y: 4, width: Math.max(1, x(max) - x(min)), height: 6, rx: 1.5, class: 'train' }),
			s('line', { x1: x(value), x2: x(value), y1: 0, y2: 14, class: 'tick' })),
		h('span', { class: 'row-value num' }, `${min.toFixed(2)}–${max.toFixed(2)} m`));
}

const sourceName = (url) => {
	const { hostname, pathname } = new URL(url);
	return hostname.endsWith('arxiv.org') ? `arXiv ${pathname.split('/').pop()}` : hostname;
};

/** The training data's conditions, the training cars' size against this car's, and the runs trained on. */
export function appliesBlock({ card, result }) {
	const d = card.dataset;
	const g = result.geometry;
	return h('section', { class: 'applies' },
		h('h3', {}, 'Where it applies'),
		h('div', { class: 'conditions' },
			[d.name, d.family, `${d.freestreamSpeed.toFixed(1)} m/s`, `${d.yawDegrees}° yaw`, `${d.wheels} wheels`].map((text) => h('span', { class: 'condition' }, text)),
			h('a', { class: 'source', href: d.source, target: '_blank', rel: 'noopener' }, sourceName(d.source), ' ↗')),
		Object.keys(DIMENSIONS).map((name) => rangeRow(name, g[name], card.dimensions[name])),
		h('div', { class: 'runs' }, h('span', { class: 'row-label' }, 'Trained on'),
			h('span', { class: 'run-list num' }, [...card.trainRuns].sort((a, b) => a - b).map((run) => h('span', {}, run)))));
}
