// The model from a point on the surface to the pressure drag, one part per step, each a line of the diagram with a
// detail that opens in place: the point, the inputs it is told there, the networks, their Cp with its spread, and the
// drag summed from Cp over every polygon.

import { h, s } from '../dom.js';
import { count, signed } from '../format.js';
import { groupBar, inputsDetail } from './inputs.js';
import { networksDetail, networksFigure } from './networks.js';

const symbol = (base, sub) => h('span', { class: 'symbol' }, base, sub ? h('sub', {}, sub) : null);

/** Where the point is and whether the networks answered at a file polygon or a mesh vertex; or why they could not. */
function pointLine({ shown, failure }) {
	if (failure) return h('p', { class: 'point-failure', role: 'alert' }, failure);
	if (!shown) return h('div', { class: 'point-line' });
	const { position, at, index } = shown.explanation;
	return h('div', { class: 'point-line num' },
		['x', 'y', 'z'].map((axis, k) => h('span', {}, h('i', {}, axis), signed(position[k], 3))),
		h('span', { class: 'unit' }, 'm'),
		h('span', { class: 'at-tag', 'data-at': at }, at === 'vertex' ? 'Mesh vertex' : `Polygon ${count(index)}`));
}

/** Values on one short scale: each network a tick, the mean, the spread as a band, CFD as a diamond. */
function valueStrip({ width, values, mean, spread, cfd, label }) {
	const all = [...values, mean, ...(cfd === null ? [] : [cfd]), mean - spread, mean + spread];
	const lo = Math.min(...all), hi = Math.max(...all);
	const pad = (hi - lo) * 0.14 || 0.01;
	const H = 30;
	const x = (v) => 8 + ((v - (lo - pad)) / (hi - lo + 2 * pad)) * (width - 16);
	return s('svg', { class: 'value-strip', width, height: H, viewBox: `0 0 ${width} ${H}`, role: 'img', 'aria-label': label },
		s('line', { x1: 8, x2: width - 8, y1: 14, y2: 14, class: 'track' }),
		s('rect', { x: x(mean - spread), y: 9, width: Math.max(1, x(mean + spread) - x(mean - spread)), height: 10, rx: 2, class: 'spread' }, s('title', {}, `σ ${spread.toFixed(3)}`)),
		values.map((v, k) => s('line', { x1: x(v), x2: x(v), y1: 8, y2: 20, class: 'tick' }, s('title', {}, `Network ${k + 1}: ${signed(v, 3)}`))),
		s('line', { x1: x(mean), x2: x(mean), y1: 3, y2: 25, class: 'mean' }, s('title', {}, `Mean ${signed(mean, 3)}`)),
		cfd === null ? null : s('path', { d: `M${x(cfd)} 8l5 6-5 6-5-6z`, class: 'cfd' }, s('title', {}, `CFD ${signed(cfd, 3)}`)));
}

function cpFigure({ shown, cfdAt, width }) {
	if (!shown) return null;
	const { cpModels, cp, sigma } = shown.explanation;
	return valueStrip({ width, values: cpModels, mean: cp, spread: sigma, cfd: cfdAt, label: 'Each network’s Cp at the point' });
}

function cpValue({ shown, cfdAt }) {
	if (!shown) return '';
	const { cp, sigma } = shown.explanation;
	return h('span', {}, signed(cp, 3),
		h('span', { class: 'side-value' }, h('i', {}, 'σ'), sigma.toFixed(3)),
		cfdAt === null ? null : h('span', { class: 'side-value' }, h('i', {}, 'CFD'), signed(cfdAt, 3)));
}

function cpDetail({ card }) {
	return h('p', { class: 'caption' },
		`σ is the networks’ standard deviation × ${card.spreadScale.toFixed(2)}: on cars outside training, CFD falls within σ of the mean on ${Math.round(card.spreadCoverage * 100)} % of the surface.`);
}

function dragDetail({ result, width }) {
	const f = result.forces;
	return h('div', {},
		valueStrip({ width, values: f.cdPModels, mean: f.cdP, spread: f.cdPStd, cfd: f.cdPCfd, label: 'Each network’s pressure drag' }),
		h('div', { class: 'formula' }, symbol('C', 'D,p'), ' = −', h('span', { class: 'sum' }, 'Σ'), ' ', symbol('C', 'p'), ' ', symbol('n', 'x'), ' ', symbol('A'), ' / ', symbol('A', 'ref')),
		h('p', { class: 'caption' }, `Summed over all ${count(result.source.cells)} polygons of the file.`));
}

function dragValue({ result }) {
	const f = result.forces;
	return h('span', {}, symbol('C', 'D,p'), ` ${f.cdP.toFixed(3)}`,
		f.cdPCfd === null ? null : h('span', { class: 'side-value' }, h('i', {}, 'CFD'), f.cdPCfd.toFixed(3)));
}

/** The pipeline's parts in order. `value` is the head's right side; `figure` stays visible; `detail` opens. */
export const PIPELINE = [
	{ id: 'point', title: 'Point', value: () => '', figure: pointLine },
	{ id: 'inputs', title: 'Inputs', value: ({ columns }) => count(columns.length), figure: groupBar, detail: inputsDetail },
	{ id: 'networks', title: 'Networks', value: ({ card }) => `${card.networks.length} × ${card.activation} MLP`, figure: networksFigure, detail: networksDetail },
	{ id: 'cp', title: 'Cp', value: cpValue, figure: cpFigure, detail: cpDetail },
	{ id: 'drag', title: 'Pressure drag', value: dragValue, detail: dragDetail },
];
