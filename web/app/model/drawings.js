// The flat drawing of the surface around the point for the overview, cut from the car's own geometry, and the
// arrowhead the page's facing arrows use. Metres keep one scale across both axes of a drawing.

import { s } from '../dom.js';

/** Length of the drawn facing arrow, in metres. */
const ARROW = 0.32;
/** Margin around a drawing's content, in pixels. */
const PAD = 10;

/**
 * A drawing area `width` pixels wide over x in [x0, x1] and v in [v0, v1], v up, at one scale on both axes.
 * @returns {{ svg: SVGSVGElement, at: (x: number, v: number) => number[], height: number }}
 */
function area(width, [x0, x1], [v0, v1], label) {
	const scale = (width - 2 * PAD) / (x1 - x0);
	const height = Math.round((v1 - v0) * scale + 2 * PAD);
	const svg = s('svg', { class: 'drawing', viewBox: `0 0 ${width} ${height}`, width, height, role: 'img', 'aria-label': label });
	return { svg, at: (x, v) => [PAD + (x - x0) * scale, height - PAD - (v - v0) * scale], height };
}

/** Clip paths made so far, for unique ids. */
let clips = 0;

/** Segments of a cut (six floats each: two ends) as a path in a view that reads coordinates `a` and `b`. */
function segmentsPath({ positions }, at, a, b) {
	let d = '';
	for (let i = 0; i < positions.length; i += 6) {
		const [x1, y1] = at(positions[i + a], positions[i + b]);
		const [x2, y2] = at(positions[i + 3 + a], positions[i + 3 + b]);
		d += `M${x1.toFixed(1)} ${y1.toFixed(1)}L${x2.toFixed(1)} ${y2.toFixed(1)}`;
	}
	return d;
}

/** The point with its facing as an arrow, in a view reading coordinates `a` and `b`. */
function pointMark(at, position, normal, a, b) {
	const [px, py] = at(position[a], position[b]);
	const [tx, ty] = at(position[a] + normal[a] * ARROW, position[b] + normal[b] * ARROW);
	return s('g', { class: 'point-mark' },
		s('line', { x1: px, y1: py, x2: tx, y2: ty, class: 'facing', 'marker-end': 'url(#model-arrow)' }),
		s('circle', { cx: px, cy: py, r: 4.5, class: 'point-dot' }));
}

/** The surface around the point close up, as the round inset of the overview. */
export function closeUp({ width, explanation, cutter }) {
	const { position, normal } = explanation;
	const reach = 0.3;
	const view = area(width, [position[0] - reach, position[0] + reach], [position[2] - reach, position[2] + reach], 'The surface around the cell');
	const clip = `model-clip-${++clips}`;
	const middle = width / 2;
	view.svg.append(
		s('clipPath', { id: clip }, s('circle', { cx: middle, cy: view.height / 2, r: middle - 1 })),
		s('g', { 'clip-path': `url(#${clip})` }, s('path', { class: 'cut', d: segmentsPath(cutter.cut(1, position[1]), view.at, 0, 2) })),
		pointMark(view.at, position, normal, 0, 2));
	return view.svg;
}

/** The arrowhead the facing arrows use, put once on the page. */
export function arrowDefinition() {
	const head = (id, size) => s('marker', { id, viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: size, markerHeight: size, orient: 'auto-start-reverse' },
		s('path', { d: 'M0 1 L9 5 L0 9 z' }));
	return s('svg', { class: 'definitions', width: 0, height: 0, 'aria-hidden': 'true' },
		s('defs', {}, head('model-arrow', 7)));
}
