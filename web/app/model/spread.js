// The spread: the networks disagree where the model is unsure. The reference car's spread σ(Cp) beside its actual
// error on one colour scale, and the error against the spread over cells grouped by spread.

import { h, s } from '../dom.js';
import { colorBar, pictureFrame, section } from './charts.js';
import { copied, text } from './copy.js';
import { locator } from './overview.js';

const CHART = 240;
const AXIS = 46;

/** Root-mean-square error against mean spread for cells grouped by spread; on the diagonal the spread is the error. */
function calibrationFigure(calibration) {
	const top = Math.max(...calibration.flatMap((b) => [b.sigma, b.error])) * 1.05;
	const step = top > 0.2 ? 0.1 : top > 0.1 ? 0.05 : 0.02;
	const ticks = [];
	for (let v = 0; v <= top + 1e-9; v += step) ticks.push(v);
	const x = (v) => AXIS + (v / top) * CHART;
	const y = (v) => CHART - (v / top) * CHART;
	return s(
		'svg',
		{
			class: 'calibration',
			viewBox: `0 0 ${CHART + AXIS + 6} ${CHART + AXIS}`,
			width: CHART + AXIS + 6,
			height: CHART + AXIS,
			role: 'img',
			'aria-label': text('spread.calibration.caption'),
		},
		s('rect', { x: AXIS, y: 0, width: CHART, height: CHART, class: 'frame' }),
		s('line', { x1: x(0), y1: y(0), x2: x(top), y2: y(top), class: 'diagonal' }),
		ticks.map((v) => [
			s('text', { x: x(v), y: CHART + 14, 'text-anchor': 'middle' }, v.toFixed(2)),
			s('text', { x: AXIS - 5, y: y(v) + 4, 'text-anchor': 'end' }, v.toFixed(2)),
		]),
		s('polyline', {
			class: 'calibration-line',
			points: calibration.map((b) => `${x(b.sigma)},${y(b.error)}`).join(' '),
		}),
		calibration.map((b) => s('circle', { cx: x(b.sigma), cy: y(b.error), r: 3.5, class: 'calibration-dot' })),
		copied(
			s('text', { x: AXIS + CHART / 2, y: CHART + AXIS - 2, 'text-anchor': 'middle', class: 'axis-name' }),
			'spread.calibration.x',
		),
		copied(
			s('text', {
				x: 8,
				y: CHART / 2,
				transform: `rotate(-90 8 ${CHART / 2})`,
				'text-anchor': 'middle',
				class: 'axis-name',
			}),
			'spread.calibration.y',
		),
	);
}

export function createSpread() {
	const frames = Object.fromEntries(
		['sigmaSide', 'missSide', 'sigmaBelow', 'missBelow'].map((id) => [id, pictureFrame(id)]),
	);
	const scale = h('div', { class: 'picture-scales' });
	const chart = h('div');
	const grid = h(
		'div',
		{ class: 'picture-grid two' },
		h(
			'div',
			{ class: 'picture-heads copy-block' },
			copied(h('span', { class: 'picture-head' }), 'spread.pictures.spread'),
			copied(h('span', { class: 'picture-head' }), 'spread.pictures.error'),
		),
		frames.sigmaSide.el,
		frames.missSide.el,
		frames.sigmaBelow.el,
		frames.missBelow.el,
	);

	const el = section(
		'spread',
		{},
		h(
			'div',
			{ class: 'spread-layout' },
			h('figure', { class: 'figure' }, grid, scale),
			h('figure', { class: 'figure' }, copied(h('figcaption'), 'spread.calibration.caption'), chart),
		),
	);
	el.prepend(locator('spread'));

	return {
		el,
		setCar({ pictures, stats, maps }) {
			for (const [id, frame] of Object.entries(frames)) frame.show(pictures[id]);
			scale.replaceChildren(colorBar(maps.spread, 'spread.scales.both'));
			chart.replaceChildren(calibrationFigure(stats.calibration));
		},
		setPoint() {},
	};
}
