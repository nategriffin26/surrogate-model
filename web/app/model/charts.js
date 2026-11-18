// Small pieces every figure on the page is built from: a section's head, rows of bars, the chips that say which
// groups of inputs a model was given, colour bars, the shade of a value as a network reads it, and the shades that
// tell the sizes of nearby surface apart.

import { h } from '../dom.js';
import { niceTicks } from '../format.js';
import { rampCss } from '../ramps.js';
import { copied, text } from './copy.js';

const WORDS = ['no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve'];
/** A count as a word, or as digits beyond twelve; a placeholder written capitalised ({CarsWord}) begins a sentence with it. */
export const countWord = (n) => WORDS[n] ?? n.toLocaleString('en-US');

/** The DrivAerML run of the open car, read from the dataset's file naming (`boundary_<run>.vtp`), or null. */
export function runOf(result) {
	const match = /^boundary_(\d+)\.vtp$/.exec(result.source.name);
	return match ? Number(match[1]) : null;
}

/** The words that name the open car: a DrivAerML run by its number and whether training used it; null for any other file. */
function carWords(result, card) {
	const run = runOf(result);
	if (card.trainRuns.includes(run)) return ['car.training', { run }];
	if (card.outsideRuns.some((r) => r.run === run)) return ['car.outside', { run }];
	return null;
}

/** The open car's name as text: its words, or any other file by its name. */
export function carName(result, card) {
	const words = carWords(result, card);
	return words ? text(...words) : result.source.name;
}

/** Show the open car's name in `element`: its words, marked as theirs, or the file's name. */
export function showCarName(element, result, card) {
	const words = carWords(result, card);
	if (words) return copied(element, ...words);
	delete element.dataset.copy;
	element.hidden = false;
	element.textContent = result.source.name;
	return element;
}

/** A section: its takeaway as the title (`<id>.title`), a sentence or two under it (`<id>.lede`), and its figures. */
export function section(id, values, ...figures) {
	return h('section', { class: 'model-section', id: `model-${id}`, 'data-section': id },
		copied(h('h2'), `${id}.title`, values),
		copied(h('p', { class: 'lede' }), `${id}.lede`, values),
		...figures);
}

/** Which groups of inputs a model was given: a chip per group, empty for those it was not, each named by its group. */
export function chips(groups, given) {
	return h('span', { class: 'chips', 'aria-label': given.map((g) => text(`groups.${g.words}`)).join(', ') || null },
		groups.map((group) => h('i', { class: `chip${given.includes(group) ? '' : ' off'}`, style: `--group: ${group.color}`, title: text(`groups.${group.words}`) })));
}

/**
 * Rows of bars on one scale from 0 to `max`: each row its label's words, optionally the chips of its inputs, the bar
 * and its value. `strong` marks the row of this model. With chips, a head row names them with the words at `inputsKey`.
 * @param {{ label: [string, object?], chips?: Element, value: number, text: string, strong?: boolean }[]} rows
 */
export function bars(rows, max, inputsKey = null) {
	const head = inputsKey ? h('div', { class: 'bar-row bar-head copy-block' }, h('span'), copied(h('span'), inputsKey), h('span'), h('span')) : null;
	return h('div', { class: 'bars' }, head, rows.map((row) => h('div', { class: `bar-row${row.strong ? ' strong' : ''}` },
		copied(h('span', { class: 'bar-label' }), ...row.label),
		row.chips ?? h('span'),
		h('span', { class: 'bar-track' }, h('span', { class: 'bar-fill', style: `width: ${(100 * Math.max(0, row.value)) / max}%` })),
		h('span', { class: 'bar-value num' }, row.text))));
}

/** A colour map's ramp with ticks under it, named by the words at `labelKey`. */
export function colorBar({ ramp, lo, hi }, labelKey) {
	const { ticks, digits } = niceTicks(lo, hi, 4);
	return h('div', { class: 'color-bar' },
		copied(h('span', { class: 'color-bar-label' }), labelKey),
		h('span', { class: 'color-bar-ramp', style: `background: ${rampCss(ramp)}` }),
		h('span', { class: 'color-bar-ticks num' }, ticks.map((t) => h('span', { style: `left: ${(100 * (t - lo)) / (hi - lo)}%` }, t.toFixed(digits).replace('-', '−')))));
}

const NEGATIVE = [163, 91, 26];
const POSITIVE = [11, 111, 191];

/** A value's shade: white at zero, towards blue for positive values and orange for negative, saturating smoothly. */
export function shade(value, scale) {
	const strength = Math.tanh(Math.abs(value) / scale);
	const [r, g, b] = value < 0 ? NEGATIVE : POSITIVE;
	const mix = (c) => Math.round(255 + (c - 255) * strength);
	return `rgb(${mix(r)}, ${mix(g)}, ${mix(b)})`;
}

const INK = [13, 34, 51];

/** A colour per size of nearby surface, smallest first: the group's colour, darker for the smallest, lighter for the largest. */
export function sizeShades(hex, count) {
	const colour = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
	return Array.from({ length: count }, (_, k) => {
		const t = -0.35 + (0.95 * k) / Math.max(count - 1, 1);
		const towards = t < 0 ? INK : [255, 255, 255];
		const mixed = colour.map((c, i) => Math.round(c + (towards[i] - c) * Math.abs(t)));
		return `#${mixed.map((c) => c.toString(16).padStart(2, '0')).join('')}`;
	});
}

/** A picture of the car in a frame that keeps its shape, with marks placed on it at shares of its size. */
export function pictureFrame(className) {
	const holder = h('div', { class: `picture ${className}` });
	return {
		el: holder,
		show(picture, ...marks) {
			holder.replaceChildren(picture.canvas, ...marks);
			holder.style.aspectRatio = `${picture.canvas.width} / ${picture.canvas.height}`;
		},
	};
}
