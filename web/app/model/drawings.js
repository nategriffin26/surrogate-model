// Flat drawings of what one group of inputs reads at the point, made from the car's own geometry: its outline seen
// from the side and from above, and the cut through the point that the slice inputs read. Distances are drawn where
// they were measured, from the marks of `measures.js`, so each drawing shows the group's inputs and where they come
// from. Metres keep one scale across both axes of a drawing.

import { s } from '../dom.js';
import { signed } from '../format.js';

/** Length of the drawn facing arrow, in metres. */
const ARROW = 0.32;
/** Margin around a drawing's content, in pixels. */
const PAD = 10;
/** Half the size of the close-up around the point, in metres along x. */
const CLOSE = 0.85;
const metres = (v) => `${signed(v, 2)} m`;

/**
 * The car's outline from the side and from above: for each step of x, the highest and lowest z and the outermost
 * y on either side, from the working mesh's vertices.
 * @param {import('../../engine/result.js').DisplayMesh} mesh
 */
export function silhouettes({ positions, bounds }) {
	const step = 0.01;
	const [x0, x1] = bounds;
	const bins = Math.ceil((x1 - x0) / step) + 1;
	const top = new Float32Array(bins).fill(-Infinity);
	const bottom = new Float32Array(bins).fill(Infinity);
	const left = new Float32Array(bins).fill(Infinity);
	const right = new Float32Array(bins).fill(-Infinity);
	for (let v = 0; v < positions.length; v += 3) {
		const b = Math.floor((positions[v] - x0) / step);
		top[b] = Math.max(top[b], positions[v + 2]);
		bottom[b] = Math.min(bottom[b], positions[v + 2]);
		left[b] = Math.min(left[b], positions[v + 1]);
		right[b] = Math.max(right[b], positions[v + 1]);
	}
	const xs = [];
	for (let b = 0; b < bins; b++) if (top[b] > -Infinity) xs.push(b);
	const outline = (upper, lower) => [
		...xs.map((b) => [x0 + (b + 0.5) * step, upper[b]]),
		...xs.toReversed().map((b) => [x0 + (b + 0.5) * step, lower[b]]),
	];
	return { side: outline(top, bottom), plan: outline(right, left) };
}

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

const pathOf = (points, at, close) => points.map(([x, v], i) => `${i ? 'L' : 'M'}${at(x, v).map((p) => p.toFixed(1)).join(' ')}`).join('') + (close ? 'Z' : '');

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

/** A measured distance: a line from where it starts to where it ends, with its value beside the middle. */
function dimension(at, from, to, text, a, b) {
	const [x1, y1] = at(from[a], from[b]);
	const [x2, y2] = at(to[a], to[b]);
	const vertical = Math.abs(y2 - y1) > Math.abs(x2 - x1);
	return s('g', { class: 'dimension' },
		s('line', { x1, y1, x2, y2 }),
		s('circle', { cx: x2, cy: y2, r: 2.5 }),
		s('text', { x: (x1 + x2) / 2 + (vertical ? 5 : 0), y: (y1 + y2) / 2 + (vertical ? 4 : -5), 'text-anchor': vertical ? 'start' : 'middle' }, text));
}

/** Where a column's value says it measured to, from the marks. */
const guideOf = (marks, column) => marks.find((m) => m.type === 'guide' && m.columns.includes(column));

/** Position and facing: where the point is on the car's outline from the side and from above, and which way it faces. */
function positionDrawing({ width, explanation, marks, outlines, columns, bounds, yCenter }) {
	const { position, normal, features } = explanation;
	const [x0, x1, y0, y1, z0, z1] = bounds;
	const xs = [x0 - 0.05, x1 + 0.05];
	const side = area(width, xs, [z0 - 0.05, z1 + 0.05], 'The point on the car from the side');
	const plan = area(width, xs, [y0 - 0.05, y1 + 0.05], 'The point on the car from above');
	side.svg.append(s('path', { class: 'silhouette', d: pathOf(outlines.side, side.at, true) }));
	plan.svg.append(s('path', { class: 'silhouette', d: pathOf(outlines.plan, plan.at, true) }));
	const [cx1, cy] = plan.at(xs[0], yCenter);
	const [cx2] = plan.at(xs[1], yCenter);
	plan.svg.append(s('line', { class: 'centre-line', x1: cx1, y1: cy, x2: cx2, y2: cy }));
	columns.forEach((column, i) => {
		const guide = guideOf(marks, i);
		if (!guide) return;
		const across = column.along[1] !== 0;
		const view = across ? plan : side;
		view.svg.append(dimension(view.at, guide.from, guide.to, metres(Math.abs(features[i])), 0, across ? 1 : 2));
	});
	side.svg.append(pointMark(side.at, position, normal, 0, 2));
	plan.svg.append(pointMark(plan.at, position, normal, 0, 1));
	return [side.svg, plan.svg];
}

/**
 * Nearby surface: the cut through the point close up, and the boxes each size of input averages the surface in, the
 * one at the point drawn solid and those up- and downstream dashed. Sizes are named in a column at the right.
 */
function boxesDrawing({ width, explanation, marks, cutter }) {
	const { position, normal } = explanation;
	const xs = [position[0] - CLOSE, position[0] + CLOSE];
	const view = area(width, xs, [position[2] - CLOSE * 0.55, position[2] + CLOSE * 0.55], 'Boxes of nearby surface around the point');
	const clip = `model-clip-${++clips}`;
	view.svg.append(s('clipPath', { id: clip }, s('rect', { x: 0, y: 0, width, height: view.height })));
	const content = s('g', { 'clip-path': `url(#${clip})` });
	content.append(s('path', { class: 'cut', d: segmentsPath(cutter.cut(1, position[1]), view.at, 0, 2) }));
	const blocks = marks.filter((m) => m.type === 'block').sort((a, b) => b.scale - a.scale);
	const labels = s('g', { class: 'box-labels' });
	const here = blocks.filter((block) => block.sample === 'here');
	const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
	for (const block of blocks) {
		const [left, top] = view.at(block.min[0], block.max[2]);
		const [right, bottom] = view.at(block.max[0], block.min[2]);
		content.append(s('rect', { class: `box ${block.sample}`, x: left, y: top, width: right - left, height: bottom - top }));
		if (block.sample !== 'here') continue;
		const y = PAD + 8 + here.indexOf(block) * 15;
		const corner = [clamp(right, PAD, width - PAD - 44), clamp(top, PAD, view.height - PAD)];
		labels.append(
			s('polyline', { class: 'box-leader', points: `${corner.join(',')} ${width - PAD - 40},${y - 4}` }),
			s('text', { class: 'box-size', x: width - PAD, y, 'text-anchor': 'end' }, `${Math.round(block.scale * 100)} cm`));
	}
	view.svg.append(content, labels, pointMark(view.at, position, normal, 0, 2));
	return [view.svg];
}

/**
 * The car ahead and behind: the cut through the point from the side, with the body's top and bottom read at each
 * station along it, and from above, the slices beside the point and the outline's distance outboard.
 */
function slicesDrawing({ width, explanation, marks, outlines, cutter, bounds }) {
	const { position, normal } = explanation;
	const [x0, x1, y0, y1, z0, z1] = bounds;
	const readings = marks.filter((m) => m.type === 'reading');
	const stationXs = readings.map((m) => m.at[0]);
	const xs = [Math.min(x0, ...stationXs) - 0.05, Math.max(x1, ...stationXs) + 0.05];
	const side = area(width, xs, [z0 - 0.08, z1 + 0.08], 'The body ahead of and behind the point, in its own slice');
	const ownCut = marks.find((m) => m.type === 'cut' && m.layer === 'slice');
	side.svg.append(s('path', { class: 'cut', d: segmentsPath(cutter.cut(1, ownCut.value), side.at, 0, 2) }));
	// Each station's readings of the body above and below, joined; a station with no body there is an open dot.
	const stations = new Map();
	for (const reading of readings.filter((m) => m.layer === 'slice')) {
		const key = reading.at[0].toFixed(4);
		if (!stations.has(key)) stations.set(key, []);
		stations.get(key).push(reading);
	}
	const empty = new Set();
	for (const [key, found] of stations) {
		if (found.some((m) => m.hollow)) {
			empty.add(key);
			const [x, y] = side.at(found[0].at[0], position[2]);
			side.svg.append(s('circle', { class: 'station empty', cx: x, cy: y, r: 3 }));
			continue;
		}
		const zs = found.map((m) => m.at[2]);
		const [x, top] = side.at(found[0].at[0], Math.max(...zs));
		const [, bottom] = side.at(found[0].at[0], Math.min(...zs));
		side.svg.append(s('g', { class: 'station' },
			s('line', { x1: x, y1: top, x2: x, y2: bottom }),
			s('circle', { cx: x, cy: top, r: 2.5 }),
			s('circle', { cx: x, cy: bottom, r: 2.5 })));
	}
	// Distances from the point along its own slice: to the slice's front and rear, and up to its peak.
	for (const guide of marks.filter((m) => m.type === 'guide' && m.layer === 'slice')) {
		const [x1, y1] = side.at(guide.from[0], guide.from[2]);
		const [x2, y2] = side.at(guide.to[0], guide.to[2]);
		side.svg.append(s('line', { class: 'guide', x1, y1, x2, y2 }));
	}
	side.svg.append(pointMark(side.at, position, normal, 0, 2));

	const plan = area(width, xs, [y0 - 0.05, y1 + 0.05], 'Slices beside the point and the outline outboard, from above');
	plan.svg.append(s('path', { class: 'silhouette', d: pathOf(outlines.plan, plan.at, true) }));
	for (const cut of marks.filter((m) => m.type === 'cut' && m.axis === 1)) {
		const [ax, ay] = plan.at(cut.span[0], cut.value);
		const [bx] = plan.at(cut.span[1], cut.value);
		plan.svg.append(s('line', { class: `slice-line${cut.own ? ' own' : ''}`, x1: ax, y1: ay, x2: bx, y2: ay }));
	}
	for (const reading of readings.filter((m) => m.layer === 'outline' && !empty.has(m.at[0].toFixed(4)))) {
		const [ax, ay] = plan.at(reading.at[0], position[1]);
		const [, by] = plan.at(reading.at[0], reading.at[1]);
		plan.svg.append(s('g', { class: 'station' }, s('line', { x1: ax, y1: ay, x2: ax, y2: by }), s('circle', { cx: ax, cy: by, r: 2.5 })));
	}
	plan.svg.append(pointMark(plan.at, position, normal, 0, 1));
	return [side.svg, plan.svg];
}

/** Each group's drawing, by its id. */
const DRAWINGS = { position: positionDrawing, bend: boxesDrawing, slice: slicesDrawing };

/**
 * A group's drawing at the point.
 * @param {import('./groups.js').InputGroup} group
 * @param {{ width: number, explanation: object, marks: object[], outlines: object, cutter: object, bounds: number[],
 *   yCenter: number, columns: object[] }} at  `marks` are the group's own; `columns` the model's, indexed as the
 *   marks' are; `bounds` the car's and `yCenter` its centre plane's y
 * @returns {SVGSVGElement[]}
 */
export function drawGroup(group, at) {
	if (!(group.id in DRAWINGS)) throw new Error(`The page has no drawing for the inputs “${group.name}”.`);
	return DRAWINGS[group.id](at);
}

/** The surface around the point close up, as the round inset of the overview. */
export function closeUp({ width, explanation, cutter }) {
	const { position, normal } = explanation;
	const reach = 0.3;
	const view = area(width, [position[0] - reach, position[0] + reach], [position[2] - reach, position[2] + reach], 'The surface around the point');
	const clip = `model-clip-${++clips}`;
	const middle = width / 2;
	view.svg.append(
		s('clipPath', { id: clip }, s('circle', { cx: middle, cy: view.height / 2, r: middle - 1 })),
		s('g', { 'clip-path': `url(#${clip})` }, s('path', { class: 'cut', d: segmentsPath(cutter.cut(1, position[1]), view.at, 0, 2) })),
		pointMark(view.at, position, normal, 0, 2));
	return view.svg;
}

/** The arrowhead the facing arrows use; put once on the page. */
export function arrowDefinition() {
	return s('svg', { class: 'definitions', width: 0, height: 0, 'aria-hidden': 'true' },
		s('defs', {}, s('marker', { id: 'model-arrow', viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' },
			s('path', { d: 'M0 1 L9 5 L0 9 z' }))));
}
