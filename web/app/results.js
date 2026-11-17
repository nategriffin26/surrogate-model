// The result card over the view: the pressure drag, the car's reference area and size, the dimensions that lie
// outside what the model was trained on (only when there are any), and one line about the run.

import { h } from './dom.js';
import { millions, seconds } from './format.js';

const DIMENSION_LABELS = { length: 'Length', width: 'Width', height: 'Height' };

export function renderResultCard(c) {
	const r = c.result;
	const f = r.forces;
	const g = r.geometry;
	const lowest = Math.min(...f.cdPModels);
	const highest = Math.max(...f.cdPModels);
	const outside = r.coverage.dimensions.filter((d) => d.value < d.trainMin || d.value > d.trainMax);
	const took = r.timings.reduce((sum, t) => sum + t.seconds, 0);
	return h('div', { class: 'result-card' },
		h('div', { class: 'result-label' }, 'Pressure drag'),
		h('div', { class: 'headline' },
			h('span', { class: 'symbol' }, 'C', h('sub', {}, 'D,p')),
			h('span', { class: 'big num' }, f.cdP.toFixed(3))),
		h('div', { class: 'result-range num', title: 'Lowest and highest pressure drag among the networks; the prediction is their mean' },
			`${lowest.toFixed(3)} to ${highest.toFixed(3)} across ${f.cdPModels.length} networks`),
		h('dl', { class: 'result-facts' },
			h('dt', {}, 'Frontal area'), h('dd', { class: 'num' }, `${g.frontalArea.toFixed(2)} m²`),
			h('dt', {}, 'L × W × H'), h('dd', { class: 'num' }, `${g.length.toFixed(2)} × ${g.width.toFixed(2)} × ${g.height.toFixed(2)} m`)),
		outside.length === 0 ? null : h('div', { class: 'result-outside', role: 'note' },
			h('div', {}, 'Outside the training range'),
			outside.map((d) => h('div', { class: 'num' }, `${DIMENSION_LABELS[d.name]} ${d.value.toFixed(2)} m, trained ${d.trainMin.toFixed(2)} to ${d.trainMax.toFixed(2)} m`))),
		h('div', { class: 'result-run num' }, `${millions(r.source.cells)} cells · ${seconds(took)} · ${r.device}`));
}
