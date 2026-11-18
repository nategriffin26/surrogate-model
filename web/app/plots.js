// Line plots in SVG: a frame with round ticks and units, lines, and a cursor that follows the pointer. Each plot
// redraws to the size of its box.

import { s } from './dom.js';
import { niceTicks, signed } from './format.js';

export function scale([d0, d1], [r0, r1]) {
	const k = (r1 - r0) / (d1 - d0 || 1);
	const f = (v) => r0 + (v - d0) * k;
	f.invert = (p) => d0 + (p - r0) / k;
	return f;
}

const path = (points) => points.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`).join('');

/**
 * Draw a plot into an empty SVG of the given size. `x` and `y` are {domain: [lo, hi], label, invert?}; series are
 * lines {cls, points: [[x, y]], colour?, dashed?}. Returns the scales and the plotting box so a cursor can be drawn
 * later.
 */
export function drawPlot(svg, { width, height, x, y, series }) {
	const m = { l: 52, r: 16, t: 22, b: 30 };
	svg.replaceChildren();
	svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
	svg.setAttribute('width', width);
	svg.setAttribute('height', height);
	const sx = scale(x.domain, [m.l, width - m.r]);
	const sy = scale(y.domain, y.invert ? [m.t, height - m.b] : [height - m.b, m.t]);
	const axis = s('g', { class: 'axis' });
	const xt = niceTicks(x.domain[0], x.domain[1], Math.max(3, Math.floor((width - m.l - m.r) / 90)));
	const yt = niceTicks(Math.min(...y.domain), Math.max(...y.domain), Math.max(3, Math.floor((height - m.t - m.b) / 26)));
	for (const t of yt.ticks) {
		axis.append(s('line', { x1: m.l, x2: width - m.r, y1: sy(t), y2: sy(t), class: t === 0 ? 'zero' : 'grid' }));
		axis.append(s('text', { x: m.l - 8, y: sy(t) + 3.5, 'text-anchor': 'end' }, signed(t, yt.digits)));
	}
	for (const t of xt.ticks) {
		axis.append(s('line', { x1: sx(t), x2: sx(t), y1: height - m.b, y2: height - m.b + 4, class: 'tick' }));
		axis.append(s('text', { x: sx(t), y: height - m.b + 16, 'text-anchor': 'middle' }, signed(t, xt.digits)));
	}
	axis.append(s('text', { x: width - m.r, y: height - 3, 'text-anchor': 'end', class: 'unit' }, ...labelParts(x.label)));
	axis.append(s('text', { x: m.l - 8, y: 12, 'text-anchor': 'end', class: 'unit' }, ...labelParts(y.label)));
	svg.append(axis);
	const clip = `clip-${Math.random().toString(36).slice(2)}`;
	svg.append(s('defs', {}, s('clipPath', { id: clip }, s('rect', { x: m.l, y: m.t, width: width - m.l - m.r, height: height - m.t - m.b }))));
	const body = s('g', { 'clip-path': `url(#${clip})` });
	for (const item of series) {
		if (item.points.length < 2) continue;
		const line = s('path', { d: path(item.points.map(([a, b]) => [sx(a), sy(b)])), class: `line ${item.cls}`, fill: 'none' });
		if (item.colour) line.style.stroke = item.colour;
		if (item.dashed) line.setAttribute('stroke-dasharray', '5 3');
		body.append(line);
	}
	svg.append(body);
	return { svg, x: sx, y: sy, m, width, height };
}

/** A vertical cursor at data x with a dot on each value ({value, cls, colour?}); null removes it. */
export function drawCursor(frame, at, dots = []) {
	frame.svg.querySelector('.cursor')?.remove();
	if (at === null) return;
	const g = s('g', { class: 'cursor' }, s('line', { x1: frame.x(at), x2: frame.x(at), y1: frame.m.t, y2: frame.height - frame.m.b }));
	for (const { value, cls, colour } of dots) {
		const dot = s('circle', { cx: frame.x(at), cy: frame.y(value), r: 3.5, class: cls });
		if (colour) dot.style.stroke = colour;
		g.append(dot);
	}
	frame.svg.append(g);
}

/** Linear interpolation along a run of [x, ...values] points sorted by x; null outside it. */
export function valueAlong(points, x, column = 1) {
	if (points.length === 0 || x < points[0][0] || x > points.at(-1)[0]) return null;
	for (let i = 1; i < points.length; i++) {
		if (points[i][0] >= x) {
			const a = points[i - 1], b = points[i];
			const k = b[0] === a[0] ? 0 : (x - a[0]) / (b[0] - a[0]);
			return a[column] + (b[column] - a[column]) * k;
		}
	}
	return points.at(-1)[column];
}

/** An axis label; the text after "_" up to the next space is set as a subscript, as in "C_D,p". */
function labelParts(label) {
	const at = label.indexOf('_');
	if (at < 0) return [label];
	const end = label.indexOf(' ', at);
	const sub = label.slice(at + 1, end < 0 ? undefined : end);
	return [label.slice(0, at), s('tspan', { dy: '3', 'font-size': '0.78em' }, sub), ...(end < 0 ? [] : [s('tspan', { dy: '-3' }, label.slice(end))])];
}
