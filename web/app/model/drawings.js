// Flat drawings of what one group of inputs reads at the point, made from the car's own geometry: its outline seen
// from the side and from above, the cuts through the point that the surface and outline inputs read, and pictures of
// the potential flow over the car. Distances are drawn where they were measured, from the marks of `measures.js`, so
// each drawing shows the group's inputs and where they come from. Metres keep one scale across both axes of a drawing.

import { h, s } from '../dom.js';
import { signed } from '../format.js';
import { colorBar, sizeShades } from './charts.js';

/** Length of the drawn facing arrow, in metres. */
const ARROW = 0.32;
/** Length of a drawn flow velocity in the flow pictures, in metres per unit of the stream's speed. */
const FLOW_ARROW = 0.2;
/** Margin around a drawing's content, in pixels. */
const PAD = 10;
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

/** A picture of the car with an SVG over it in the same pixels, and where a world point falls on both. */
function overlaid(picture, width, label) {
	const height = Math.round((width * picture.canvas.height) / picture.canvas.width);
	Object.assign(picture.canvas.style, { width: `${width}px`, height: `${height}px` });
	const svg = s('svg', { class: 'drawing', viewBox: `0 0 ${width} ${height}`, width, height, role: 'img', 'aria-label': label });
	const at = (point) => {
		const [u, v] = picture.place(point);
		return [u * width, v * height];
	};
	return { el: h('div', { class: 'overlaid' }, picture.canvas, svg), svg, at };
}

/**
 * Potential flow: the ideal flow over the car from the side, coloured by the pressure it would put on the surface and
 * with arrows of its velocity along the surface, the point's own arrow and speed among them; beneath it, CFD's
 * pressure on the same scale when the file has it, so what the ideal flow misses is seen.
 */
function flowDrawing({ width, marks, flow }) {
	if (!flow) return [];
	const ideal = overlaid(flow.picture, width, 'The potential flow over the car from the side: its pressure and its velocity along the surface');
	const arrow = (at, velocity, attributes) => {
		const [x1, y1] = ideal.at(at);
		const [x2, y2] = ideal.at(at.map((p, k) => p + velocity[k] * FLOW_ARROW));
		return s('line', { x1, y1, x2, y2, 'marker-end': 'url(#model-arrow)', ...attributes });
	};
	for (const { at, velocity } of flow.arrows) ideal.svg.append(arrow(at, velocity, { class: 'flow-arrow', 'marker-end': 'url(#model-flow-arrow)' }));
	const point = marks.find((m) => m.type === 'velocity');
	if (point) {
		const [x, y] = ideal.at(point.at);
		ideal.svg.append(s('g', { class: 'flow-point' },
			arrow(point.at, point.velocity, {}),
			s('circle', { cx: x, cy: y, r: 4.5, class: 'point-dot' }),
			s('text', { x: x + 8, y: y - 8 }, `${Math.hypot(...point.velocity).toFixed(2)} × the stream`)));
	}
	const views = [h('div', { class: 'flow-view' }, h('span', { class: 'drawing-label' }, 'Ideal flow'), ideal.el)];
	if (flow.cfdPicture) {
		const cfd = overlaid(flow.cfdPicture, width, 'CFD pressure over the car from the side');
		views.push(h('div', { class: 'flow-view' }, h('span', { class: 'drawing-label' }, 'CFD'), cfd.el));
	}
	return [...views, colorBar(flow.map, 'Cp')];
}

/**
 * Nearby surface: the cut through the point, with each size as a ring of its radius about the point and the mean
 * normal it finds there as an arrow, longer for each larger size. Seen from the side, or from above when the point
 * faces sideways.
 */
function surfaceDrawing({ width, explanation, marks, cutter, color }) {
	const { position, normal } = explanation;
	const sideways = Math.abs(normal[1]) > Math.max(Math.abs(normal[0]), Math.abs(normal[2]));
	const b = sideways ? 1 : 2;
	const cutAxis = sideways ? 2 : 1;
	const sizes = marks.filter((m) => m.type === 'neighbourhood').sort((p, q) => p.scale - q.scale);
	const shades = sizeShades(color, sizes.length);
	const reach = 1.25 * Math.max(...sizes.map((m) => m.scale));
	const view = area(width, [position[0] - reach, position[0] + reach], [position[b] - reach * 0.5, position[b] + reach * 0.62],
		`Nearby surface at ${sizes.length} sizes about the point, ${sideways ? 'from above' : 'from the side'}`);
	const clip = `model-clip-${++clips}`;
	const [px, py] = view.at(position[0], position[b]);
	const perMetre = view.at(position[0] + 1, position[b])[0] - px;
	const shade = (k) => `--shade: ${shades[k]}`;
	const content = s('g', { 'clip-path': `url(#${clip})` },
		sizes.map((size, k) => s('circle', { class: 'size-ring', cx: px, cy: py, r: size.scale * perMetre, style: shade(k) })).toReversed(),
		s('path', { class: 'cut', d: segmentsPath(cutter.cut(cutAxis, position[cutAxis]), view.at, 0, b) }));
	const arrow = 0.8 * reach;
	sizes.forEach((size, k) => {
		const [lx, ly] = view.at(position[0] + size.scale * 0.72, position[b] + size.scale * 0.72);
		content.append(s('text', { class: 'size-label', x: lx + 2, y: ly - 2, style: shade(k) }, `${Math.round(size.scale * 100)}`));
		const length = Math.hypot(...size.normal);
		if (length === 0) return;
		const tip = arrow * (0.55 + 0.15 * k);
		const [tx, ty] = view.at(position[0] + (size.normal[0] / length) * tip, position[b] + (size.normal[b] / length) * tip);
		content.append(s('g', { class: 'size-normal', style: shade(k) },
			s('line', { x1: px, y1: py, x2: tx, y2: ty }),
			s('circle', { cx: tx, cy: ty, r: 2.6 })));
	});
	view.svg.append(
		s('clipPath', { id: clip }, s('rect', { x: 0, y: 0, width, height: view.height })),
		content,
		s('text', { x: width - PAD, y: view.height - PAD, 'text-anchor': 'end' }, 'cm'),
		pointMark(view.at, position, normal, 0, b));
	return [view.svg];
}

/**
 * Plan-view outline: from above, the car cut at the point's height, and at each station ahead and behind how far the
 * body reaches outboard of the point; an open dot where no body stands at that height. The readings a little above
 * and below are the small dots.
 */
function outlineDrawing({ width, explanation, marks, outlines, cutter, bounds }) {
	const { position, normal } = explanation;
	const [x0, x1, y0, y1] = bounds;
	const readings = marks.filter((m) => m.type === 'reading' && m.layer === 'outline');
	const stationXs = readings.map((m) => m.at[0]);
	const plan = area(width, [Math.min(x0, ...stationXs) - 0.05, Math.max(x1, ...stationXs) + 0.05], [y0 - 0.05, y1 + 0.05],
		'The car cut at the point\u2019s height, from above, and how far it reaches outboard ahead of and behind the point');
	plan.svg.append(s('path', { class: 'silhouette', d: pathOf(outlines.plan, plan.at, true) }));
	const own = marks.find((m) => m.type === 'cut' && m.layer === 'outline' && m.own);
	if (own) plan.svg.append(s('path', { class: 'cut', d: segmentsPath(cutter.cut(2, own.value), plan.at, 0, 1) }));
	for (const reading of readings) {
		const [x, from] = plan.at(reading.at[0], position[1]);
		if (Math.abs(reading.at[2] - position[2]) > 1e-6) {
			if (!reading.hollow) plan.svg.append(s('circle', { class: 'station other', cx: x, cy: plan.at(reading.at[0], reading.at[1])[1], r: 1.6 }));
		} else if (reading.hollow) {
			plan.svg.append(s('circle', { class: 'station empty', cx: x, cy: from, r: 3 }));
		} else {
			const [, to] = plan.at(reading.at[0], reading.at[1]);
			plan.svg.append(s('g', { class: 'station' }, s('line', { x1: x, y1: from, x2: x, y2: to }), s('circle', { cx: x, cy: to, r: 2.5 })));
		}
	}
	plan.svg.append(pointMark(plan.at, position, normal, 0, 1));
	return [plan.svg];
}

/** Each group's drawing, by its id. */
const DRAWINGS = { position: positionDrawing, flow: flowDrawing, surface: surfaceDrawing, outline: outlineDrawing };

/**
 * A group's drawing at the point, in the group's colour.
 * @param {import('./groups.js').InputGroup} group
 * @param {{ width: number, explanation: object, marks: object[], outlines: object, cutter: object, bounds: number[],
 *   yCenter: number, columns: object[], flow: object | null }} at  `marks` are the group's own; `columns` the
 *   model's, indexed as the marks' are; `bounds` the car's and `yCenter` its centre plane's y; `flow` the potential
 *   flow's pictures of the car and arrows, when the model reads it
 * @returns {Element[]}
 */
export function drawGroup(group, at) {
	if (!(group.id in DRAWINGS)) throw new Error(`The page has no drawing for the inputs “${group.name}”.`);
	return DRAWINGS[group.id]({ ...at, color: group.color });
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

/** The arrowheads the drawings use, put once on the page: the facing arrows', and the smaller one of the flow field. */
export function arrowDefinition() {
	const head = (id, size) => s('marker', { id, viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: size, markerHeight: size, orient: 'auto-start-reverse' },
		s('path', { d: 'M0 1 L9 5 L0 9 z' }));
	return s('svg', { class: 'definitions', width: 0, height: 0, 'aria-hidden': 'true' },
		s('defs', {}, head('model-arrow', 7), head('model-flow-arrow', 4)));
}
