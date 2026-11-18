// Each group's inputs at the point as the numbers the networks are given, each on its range over the training cars,
// so a reader sees what a number is, what it is now and what it ranges between. A cell per input is coloured by where
// its number sits in that range: white at the training cars' median, brown toward their low end, blue toward their
// high end. A group that reads the same things at several stations or sizes shows them all as a grid and one of them
// as numbers, with a stepper and the grid's columns or rows to choose which.

import { h, s } from '../dom.js';
import { signed } from '../format.js';
import { shade } from './charts.js';

/** Where a value sits in its range: −1 at the low end (2nd percentile), 0 at the median, +1 at the high end (98th). */
function placed(value, [low, middle, high]) {
	return value >= middle ? (value - middle) / Math.max(high - middle, 1e-9) : (value - middle) / Math.max(middle - low, 1e-9);
}

/** An input's colour: where its number sits in its range over the training cars. */
const tone = (value, range) => shade(placed(value, range), 1);

const number = (value, unit) => `${signed(value, 2)}${unit ? ` ${unit}` : ''}`;

/** A section's station as the page names it. */
const stationName = (dx) => (dx === 0 ? 'at the point' : `${Math.round(Math.abs(dx) * 100)} cm ${dx < 0 ? 'ahead' : 'behind'}`);
const centimetres = (metres) => `${Math.round(metres * 100)}`;

/** ◂ the shown station or size ▸ */
function stepper(label, shown, count, show, [back, forward]) {
	return h('div', { class: 'repeat-step' },
		h('button', { type: 'button', 'aria-label': back, disabled: shown === 0, onclick: () => show(shown - 1) }, '\u25c2'),
		h('span', {}, label),
		h('button', { type: 'button', 'aria-label': forward, disabled: shown === count - 1, onclick: () => show(shown + 1) }, '\u25b8'));
}

/** A number's range over the training cars as a track from its low end to its high end, the median a tick, the number a dot. */
function rangeBar(svg, x, y, width, value, range) {
	const [low, middle, high] = range;
	const at = (v) => x + width * Math.max(0, Math.min(1, (v - low) / Math.max(high - low, 1e-9)));
	svg.append(
		s('line', { class: 'range-track', x1: x, y1: y, x2: x + width, y2: y }),
		s('line', { class: 'range-middle', x1: at(middle), y1: y - 4, x2: at(middle), y2: y + 4 }),
		s('circle', { class: 'range-value', cx: at(value), cy: y, r: 4.5, style: `fill: ${tone(value, range)}` }),
		s('text', { class: 'range-end num', x: x - 6, y: y + 3.5, 'text-anchor': 'end' }, signed(low, 2)),
		s('text', { class: 'range-end num', x: x + width + 6, y: y + 3.5 }, signed(high, 2)));
}

/** The cell of an input: its colour where its number sits in its range. */
const cell = (x, y, width, height, value, range) => s('rect', { class: 'read-cell', x: x + 0.5, y: y + 0.5, width: width - 1, height: height - 1, rx: 1.5, style: `fill: ${tone(value, range)}` });

/**
 * Lines of numbers, one an input: its cell, its name (from `nameAt`), its number with its unit (ending at `valueEnd`)
 * and its range bar; `part` names a block of lines on its first. With `stacked`, a line's number ends at the
 * drawing's right edge and its range bar runs under its name.
 */
function numberLines(svg, lines, { top, width, pitch, nameAt, valueEnd, stacked = false }) {
	const barStart = stacked ? nameAt + 30 : valueEnd + 44;
	const barWidth = width - barStart - 40;
	lines.forEach(({ part, name, value, unit, range }, k) => {
		const y = top + k * pitch + (stacked ? 9 : pitch / 2);
		svg.append(
			cell(0, y - 6, 12, 12, value, range),
			s('text', { class: 'read-name', x: nameAt, y: y + 4 }, name),
			s('text', { class: 'read-value num', x: stacked ? width : valueEnd, y: y + 4.5, 'text-anchor': 'end' }, number(value, unit)));
		if (part) svg.append(s('text', { class: 'read-part', x: 16, y: y + 4 }, part));
		rangeBar(svg, barStart, stacked ? y + 13 : y, barWidth, value, range);
	});
	svg.append(s('text', { class: 'range-end', x: width, y: top - 2, 'text-anchor': 'end' }, 'range over the training cars, median ticked'));
}

const drawingOf = (width, height, label) => s('svg', { class: 'drawing', viewBox: `0 0 ${width} ${height}`, width, height, role: 'img', 'aria-label': label });

/** Position and facing: its seven numbers. */
function positionNumbers({ group, width, explanation, columns }) {
	const pitch = 30;
	const top = 14;
	const svg = drawingOf(width, top + group.columns.length * pitch, 'Position and facing as the networks are given them');
	numberLines(svg, group.columns.map((i) => ({ name: columns[i].name, value: explanation.features[i], unit: columns[i].unit, range: columns[i].range })),
		{ top, width, pitch, nameAt: 18, stacked: true });
	return [svg];
}

/**
 * Slices across the car: the 52 inputs as a table of stations by reads, the shown station's column outlined; under it
 * the shown station's four numbers.
 */
function sectionNumbers({ group, width, explanation, columns, shown, show }) {
	const { values: stations, columns: byStation } = group.repeat;
	const reads = byStation[0].map((i) => columns[i].read);
	const left = 56;
	const size = 15;
	const gridWidth = width - left - 2;
	const step = gridWidth / stations.length;
	const linesTop = reads.length * size + 40;
	const pitch = 26;
	const svg = drawingOf(width, linesTop + reads.length * pitch, 'The slices\u2019 inputs, station by station, and the shown station\u2019s numbers');
	reads.forEach((read, r) => svg.append(s('text', { class: 'read-name', x: left - 6, y: r * size + 11, 'text-anchor': 'end' }, read)));
	byStation.forEach((station, k) => {
		station.forEach((i, r) => svg.append(cell(left + k * step, r * size, step, size, explanation.features[i], columns[i].range)));
		svg.append(
			s('text', { class: 'grid-label num', x: left + (k + 0.5) * step, y: reads.length * size + 12, 'text-anchor': 'middle' }, centimetres(Math.abs(stations[k]))),
			s('rect', { class: 'repeat-hit', x: left + k * step, y: 0, width: step, height: reads.length * size, onclick: () => show(k) }));
	});
	svg.append(
		s('rect', { class: 'shown-outline', x: left + shown * step - 1, y: -1, width: step + 2, height: reads.length * size + 2, rx: 2.5 }),
		s('text', { class: 'grid-label', x: left, y: reads.length * size + 24 }, '\u2190 ahead, cm'),
		s('text', { class: 'grid-label', x: left + gridWidth, y: reads.length * size + 24, 'text-anchor': 'end' }, 'cm, behind \u2192'));
	numberLines(svg, byStation[shown].map((i) => ({ name: columns[i].read, value: explanation.features[i], unit: columns[i].unit, range: columns[i].range })),
		{ top: linesTop, width, pitch, nameAt: 18, valueEnd: 108 });
	return [stepper(stationName(stations[shown]), shown, stations.length, show, ['Station ahead', 'Station behind']), svg];
}

/** A nearby-surface input's name within its part: which way the mean facing points, or the density. */
const surfaceName = (name) => name.replace(/^Mean /, '').replace(/, (ahead|behind)$/, '').replace(/^Surface density$/, 'density');

/**
 * Nearby surface: the 48 inputs as a table of sizes by reads, whole, ahead and behind, the shown size's row outlined;
 * under it the shown size's twelve numbers.
 */
function surfaceNumbers({ group, width, explanation, columns, shown, show }) {
	const { values: sizes, columns: bySize } = group.repeat;
	const reads = bySize[0];
	const parts = [...new Set(reads.map((i) => columns[i].part))];
	const left = 30;
	const gap = 6;
	const size = 11;
	const step = (width - left - 2 - gap * (parts.length - 1)) / reads.length;
	const columnAt = (c) => left + c * step + gap * parts.indexOf(columns[reads[c]].part);
	const gridTop = 14;
	const linesTop = gridTop + sizes.length * size + 22;
	const pitch = 15;
	const svg = drawingOf(width, linesTop + reads.length * pitch, 'Nearby surface\u2019s inputs, size by size, and the shown size\u2019s numbers');
	parts.forEach((part) => {
		const first = reads.findIndex((i) => columns[i].part === part);
		const count = reads.filter((i) => columns[i].part === part).length;
		svg.append(s('text', { class: 'read-name', x: columnAt(first) + (count * step) / 2, y: gridTop - 4, 'text-anchor': 'middle' }, part));
	});
	bySize.forEach((row, k) => {
		const y = gridTop + k * size;
		row.forEach((i, c) => svg.append(cell(columnAt(c), y, step, size, explanation.features[i], columns[i].range)));
		svg.append(
			s('text', { class: 'grid-label num', x: left - 6, y: y + size - 2.5, 'text-anchor': 'end' }, centimetres(sizes[k])),
			s('rect', { class: 'repeat-hit', x: left, y, width: width - left, height: size, onclick: () => show(k) }));
	});
	svg.append(
		s('rect', { class: 'shown-outline', x: left - 1, y: gridTop + shown * size - 1, width: width - left, height: size + 2, rx: 2.5 }),
		s('text', { class: 'grid-label', x: left - 6, y: gridTop + sizes.length * size + 10, 'text-anchor': 'end' }, 'cm'));
	numberLines(svg, bySize[shown].map((i, c) => ({
		part: c === 0 || columns[i].part !== columns[reads[c - 1]].part ? columns[i].part : null,
		name: surfaceName(columns[i].name), value: explanation.features[i], unit: columns[i].unit, range: columns[i].range,
	})), { top: linesTop, width, pitch, nameAt: 58, valueEnd: 170 });
	return [stepper(`${centimetres(sizes[shown])} cm`, shown, sizes.length, show, ['Smaller size', 'Larger size']), svg];
}

/** Each group's numbers, by its id. */
const NUMBERS = { position: positionNumbers, surface: surfaceNumbers, section: sectionNumbers };

/**
 * A group's numbers at the point.
 * @param {import('./groups.js').InputGroup} group
 * @param {{ width: number, explanation: object, columns: object[], shown: number | null,
 *   show: (k: number) => void }} at  `columns` the model's; `shown` which of a repeating group's stations or sizes
 *   is shown as numbers, and `show` to show another
 * @returns {Element[]}
 */
export function drawGroup(group, at) {
	if (!(group.id in NUMBERS)) throw new Error(`The page has no drawing for the inputs “${group.name}”.`);
	return NUMBERS[group.id]({ ...at, group });
}
