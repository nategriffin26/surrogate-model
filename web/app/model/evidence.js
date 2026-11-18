// How the model does on cars it never saw: this car against its CFD when the file carries it, otherwise the runs
// outside training, and the training cars each scored by the networks that never saw it; and where it applies:
// the data it was trained on and the size of the cars in it.

import { h, s } from '../dom.js';
import { counts, signed } from '../format.js';
import { icon } from '../icons.js';
import { EXAMPLES } from '../examples.js';
import { dragCounts } from './inputs.js';

const DIMENSIONS = { length: 'Length', width: 'Width', height: 'Height' };

/** The scores against CFD, with the switch that shows this car's prediction beside its CFD. */
export function unseenBlock({ card, result, compare, act }) {
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
	return h('div', { class: 'unseen' },
		h('table', { class: 'evidence-table' },
			h('thead', {}, h('tr', {},
				h('th', {}),
				h('th', { class: 'num' }, 'R² of Cp'),
				h('th', { class: 'num' }, h('span', { class: 'symbol' }, 'C', h('sub', {}, 'D,p')), ' error, counts'))),
			h('tbody', {}, rows.map(({ label, r2, error, kind, example }) => h('tr', { 'data-kind': kind },
				h('th', { scope: 'row' }, label, example ? h('button', {
					type: 'button',
					class: 'link-button',
					title: `Open ${example.label}, with its CFD`,
					onclick: () => act.openExample(example),
				}, 'Open') : null),
				h('td', { class: 'num' }, signed(r2, 3)),
				h('td', { class: 'num' }, String(error)))))),
		accuracy ? h('button', {
			type: 'button',
			class: 'control-button',
			'aria-pressed': String(compare),
			onclick: () => act.compare(!compare),
		}, icon('compare'), compare ? 'Back to the drawing' : 'Show prediction beside CFD') : null);
}

/** A dimension on the training cars' range: the range as a band, this car as a tick, outside values marked. */
function rangeRow(name, value, { min, max }) {
	const outside = value < min || value > max;
	const lo = Math.min(min, value), hi = Math.max(max, value);
	const pad = (hi - lo) * 0.15;
	const W = 120;
	const x = (v) => 2 + ((v - (lo - pad)) / (hi - lo + 2 * pad)) * (W - 4);
	return h('div', { class: `range${outside ? ' outside' : ''}` },
		h('span', { class: 'range-label' }, DIMENSIONS[name]),
		s('svg', { class: 'range-bar', viewBox: `0 0 ${W} 14`, 'aria-hidden': 'true' },
			s('rect', { x: x(min), y: 4, width: Math.max(1, x(max) - x(min)), height: 6, rx: 1.5, class: 'train' }),
			s('line', { x1: x(value), x2: x(value), y1: 0, y2: 14, class: 'tick' })),
		h('span', { class: 'range-value num' }, `${value.toFixed(2)} m`),
		h('span', { class: 'range-train num' }, `trained ${min.toFixed(2)}–${max.toFixed(2)} m`));
}

const sourceName = (url) => {
	const { hostname, pathname } = new URL(url);
	return hostname.endsWith('arxiv.org') ? `arXiv ${pathname.split('/').pop()}` : hostname;
};

/** The training data's conditions, the training cars' size against this car's, and the runs trained on. */
export function appliesBlock({ card, result }) {
	const d = card.dataset;
	const g = result.geometry;
	return h('div', { class: 'applies' },
		h('div', { class: 'conditions' },
			[d.name, d.family, `${d.freestreamSpeed.toFixed(1)} m/s`, `${d.yawDegrees}° yaw`, `${d.wheels} wheels`].map((text) => h('span', { class: 'condition' }, text)),
			h('a', { class: 'source', href: d.source, target: '_blank', rel: 'noopener' }, sourceName(d.source), ' ↗')),
		Object.keys(DIMENSIONS).map((name) => rangeRow(name, g[name], card.dimensions[name])),
		h('div', { class: 'runs' }, h('span', { class: 'range-label' }, 'Trained on'),
			h('span', { class: 'run-list num' }, [...card.trainRuns].sort((a, b) => a - b).map((run) => h('span', {}, `Run ${run}`)))));
}
