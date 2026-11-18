// How good the model is on a car it never saw: the reference car's CFD, prediction and their difference from the side
// and from below, every cell's prediction against its CFD, the error by region, and the accuracy measured in training.

import { h, s } from '../dom.js';
import { signed } from '../format.js';
import { colorBar, pictureFrame, runOf, section } from './charts.js';
import { copied } from './copy.js';
import { locator } from './overview.js';

const SCATTER = 260;
const AXIS = 44;

/** Every cell's predicted Cp against its CFD Cp as the area in each square of a grid, darker where there is more. */
function scatterFigure({ lo, hi, bins, density }) {
	let max = 0;
	for (const d of density) max = Math.max(max, d);
	const canvas = h('canvas', { width: bins, height: bins, class: 'scatter-density', style: `left: ${AXIS}px; width: ${SCATTER}px; height: ${SCATTER}px` });
	const g = canvas.getContext('2d');
	const image = g.createImageData(bins, bins);
	for (let j = 0; j < bins; j++) {
		for (let i = 0; i < bins; i++) {
			const d = density[j * bins + i];
			if (!d) continue;
			const strength = Math.log1p((200 * d) / max) / Math.log1p(200);
			const at = 4 * ((bins - 1 - j) * bins + i);
			image.data.set([11, 111, 191, Math.round(40 + 215 * strength)], at);
		}
	}
	g.putImageData(image, 0, 0);
	const ticks = [];
	for (let v = Math.ceil(lo * 2) / 2; v <= hi + 1e-9; v += 0.5) ticks.push(v);
	const x = (v) => AXIS + ((v - lo) / (hi - lo)) * SCATTER;
	const y = (v) => SCATTER - ((v - lo) / (hi - lo)) * SCATTER;
	const axes = s('svg', { class: 'scatter-axes', viewBox: `0 0 ${SCATTER + AXIS} ${SCATTER + AXIS}`, width: SCATTER + AXIS, height: SCATTER + AXIS },
		s('rect', { x: AXIS, y: 0, width: SCATTER, height: SCATTER, class: 'frame' }),
		s('line', { x1: x(lo), y1: y(lo), x2: x(hi), y2: y(hi), class: 'diagonal' }),
		ticks.map((v) => [
			s('text', { x: x(v), y: SCATTER + 14, 'text-anchor': 'middle' }, signed(v, 1)),
			s('text', { x: AXIS - 5, y: y(v) + 4, 'text-anchor': 'end' }, signed(v, 1)),
		]),
		copied(s('text', { x: AXIS + SCATTER / 2, y: SCATTER + AXIS - 2, 'text-anchor': 'middle', class: 'axis-name' }), 'accuracy.scatter.x'),
		copied(s('text', { x: 8, y: SCATTER / 2, transform: `rotate(-90 8 ${SCATTER / 2})`, 'text-anchor': 'middle', class: 'axis-name' }), 'accuracy.scatter.y'));
	return h('div', { class: 'scatter', style: `width: ${SCATTER + AXIS}px` }, axes, canvas);
}

/** Mean |predicted − CFD| by region, largest first, with each region's share of the surface. */
function regionFigure(regions) {
	const max = Math.max(...regions.map((r) => r.error));
	return h('div', { class: 'bars' }, [...regions].sort((a, b) => b.error - a.error).map((region) => h('div', { class: 'bar-row' },
		copied(h('span', { class: 'bar-label' }), `accuracy.regions.${region.id}`),
		h('span', { class: 'bar-share num' }, `${Math.round(100 * region.share)} %`),
		h('span', { class: 'bar-track' }, h('span', { class: 'bar-fill', style: `width: ${(100 * region.error) / max}%` })),
		h('span', { class: 'bar-value num' }, region.error.toFixed(3)))));
}

/** @param {{ card: import('../../engine/result.js').ModelCard }} options */
export function createAccuracy({ card }) {
	const frames = Object.fromEntries(['cfdSide', 'cpSide', 'errorSide', 'cfdBelow', 'cpBelow', 'errorBelow'].map((id) => [id, pictureFrame(id)]));
	const scales = h('div', { class: 'picture-scales' });
	const outsideNumber = h('div', { class: 'headline' });
	const carNumber = h('div', { class: 'headline' });
	const charts = h('div', { class: 'accuracy-charts' });

	const el = section('accuracy', {},
		h('div', { class: 'headlines' },
			h('div', { class: 'headline' }, h('span', { class: 'big-number num' }, card.crossValidation.meanR2.toFixed(3)),
				copied(h('span', { class: 'headline-label' }), 'accuracy.headlines.crossValidation')),
			outsideNumber,
			carNumber),
		h('figure', { class: 'figure' },
			h('div', { class: 'picture-grid three' },
				h('div', { class: 'picture-heads copy-block' },
					copied(h('span', { class: 'picture-head' }), 'accuracy.pictures.cfd'),
					copied(h('span', { class: 'picture-head' }), 'accuracy.pictures.predicted'),
					copied(h('span', { class: 'picture-head' }), 'accuracy.pictures.error')),
				frames.cfdSide.el, frames.cpSide.el, frames.errorSide.el,
				frames.cfdBelow.el, frames.cpBelow.el, frames.errorBelow.el),
			scales),
		charts);
	el.prepend(locator('accuracy'));

	return {
		el,
		setCar({ result, pictures, stats, maps }) {
			// The car's own number replaces the card's for the same run: two R² for one car would read as a contradiction.
			const outside = card.outsideRuns.filter((r) => r.run !== runOf(result));
			outsideNumber.hidden = outside.length === 0;
			outsideNumber.replaceChildren(
				h('span', { class: 'big-number num' }, outside.map((r) => r.r2.toFixed(3)).join(' · ')),
				copied(h('span', { class: 'headline-label' }), `accuracy.headlines.outside.${new Intl.PluralRules('en').select(outside.length)}`, {
					runs: new Intl.ListFormat('en').format(outside.map((r) => String(r.run))),
				}));
			carNumber.replaceChildren(
				h('span', { class: 'big-number num' }, result.accuracy.r2.toFixed(3)),
				copied(h('span', { class: 'headline-label' }), 'accuracy.headlines.car'));
			for (const [id, frame] of Object.entries(frames)) frame.show(pictures[id]);
			scales.replaceChildren(colorBar(maps.cp, 'accuracy.scales.cp'), colorBar(maps.error, 'accuracy.scales.error'));
			charts.replaceChildren(
				h('figure', { class: 'figure' }, scatterFigure(stats.scatter)),
				h('figure', { class: 'figure' }, copied(h('figcaption'), 'accuracy.regions.caption'), regionFigure(stats.regions)));
		},
		setPoint() {},
	};
}
