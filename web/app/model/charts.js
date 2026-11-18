// Small pieces the figures on the page are built from: a section's head, the car's run, colour bars, the shade of a
// value as a network reads it, and the shades that tell the sizes of nearby surface apart.

import { h } from '../dom.js';
import { niceTicks } from '../format.js';
import { RAMPS, rampCss } from '../ramps.js';
import { copied, hasCopy } from './copy.js';

const NUMBER_WORDS = [
	'no',
	'one',
	'two',
	'three',
	'four',
	'five',
	'six',
	'seven',
	'eight',
	'nine',
	'ten',
	'eleven',
	'twelve',
];
/** A count as a word, or as digits beyond twelve; a placeholder written capitalised ({CarsWord}) begins a sentence with it. */
export const countWord = (n) => NUMBER_WORDS[n] ?? n.toLocaleString('en-US');

/** The DrivAerML run of a car, read from the dataset's file naming (`boundary_<run>.vtp`), or null. */
export function runOf(result) {
	const match = /^boundary_(\d+)\.vtp$/.exec(result.source.name);
	return match ? Number(match[1]) : null;
}

/**
 * A section: its takeaway as the title (`<id>.title`), a sentence or two under it where copy.json has them
 * (`<id>.lede`), and its figures.
 */
export function section(id, values, ...figures) {
	return h(
		'section',
		{ class: 'model-section', id: `model-${id}`, 'data-section': id },
		copied(h('h2'), `${id}.title`, values),
		hasCopy(`${id}.lede`) ? copied(h('p', { class: 'lede' }), `${id}.lede`, values) : null,
		...figures,
	);
}

/** A colour map's ramp with ticks under it, named by the words at `labelKey`. */
export function colorBar({ ramp, lo, hi }, labelKey) {
	const { ticks, digits } = niceTicks(lo, hi, 4);
	return h(
		'div',
		{ class: 'color-bar' },
		copied(h('span', { class: 'color-bar-label' }), labelKey),
		h('span', { class: 'color-bar-ramp', style: `background: ${rampCss(ramp)}` }),
		h(
			'span',
			{ class: 'color-bar-ticks num' },
			ticks.map((t) =>
				h('span', { style: `left: ${(100 * (t - lo)) / (hi - lo)}%` }, t.toFixed(digits).replace('-', '−')),
			),
		),
	);
}

const rgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
/** The pressure map's ends (coolwarm, `ramps.js`): its red below the mean, its blue above. */
const NEGATIVE = rgb(RAMPS.coolwarm.stops.at(-1)[1]);
const POSITIVE = rgb(RAMPS.coolwarm.stops[0][1]);

/** A value's shade: white at zero, towards blue for positive values and red for negative, saturating smoothly. */
export function shade(value, scale) {
	const strength = Math.tanh(Math.abs(value) / scale);
	const [r, g, b] = value < 0 ? NEGATIVE : POSITIVE;
	const mix = (c) => Math.round(255 + (c - 255) * strength);
	return `rgb(${mix(r)}, ${mix(g)}, ${mix(b)})`;
}

/** The shades as a CSS gradient for their key, from the most negative through white to the most positive. */
export const shadeGradient = () => `linear-gradient(90deg, ${shade(-3, 1)}, #fff 50%, ${shade(3, 1)})`;

const INK = [13, 34, 51];

/** A colour per size of nearby surface, smallest first: the group's colour, darker for the smallest, lighter for the largest. */
export function sizeShades(hex, count) {
	const colour = rgb(hex);
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
