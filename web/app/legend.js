// The colour legend: field, ramp, ticks, and a range that can be typed, dragged or split into discrete bands. It
// draws through the same ramp and band rule as the shader.

import { h } from './dom.js';
import { MAPS } from './fields.js';
import { digitsFor, niceTicks, signed } from './format.js';
import { icon } from './icons.js';
import { menu } from './menu.js';
import { RAMPS, rampCss, rampPosition, sample } from './ramps.js';

const BAR_WIDTH = 264;
const GRIP = 10;

/** Number of bands a banded map splits into: round steps of its range, between 4 and 24. */
export function bandCount(map) {
	if (!map.banded) return 0;
	const { step } = niceTicks(map.lo, map.hi, 10);
	return Math.min(24, Math.max(4, Math.round((map.hi - map.lo) / step)));
}

/**
 * Tick values for a scale `width` pixels wide: band edges when banded, otherwise round values, each far enough from
 * the ends that its centred label clears the end labels (about 6.6 px a character at the legend's 11.5 px).
 */
function ticksFor(map, width) {
	const d = digits(map);
	const chars = Math.max(signed(map.lo, d).length, signed(map.hi, d).length);
	const margin = (1.5 * chars * 6.6 + 8) / width;
	const span = map.hi - map.lo;
	const clear = (t) => (t - map.lo) / span >= margin && (map.hi - t) / span >= margin;
	const bands = bandCount(map);
	if (bands) {
		const every = Math.ceil(bands / 6);
		const out = [];
		for (let k = every; k < bands; k += every) out.push(map.lo + (span * k) / bands);
		return out.filter(clear);
	}
	return niceTicks(map.lo, map.hi, 5).ticks.filter(clear);
}

function digits(map) {
	return digitsFor((map.hi - map.lo) / 20);
}

/** The bar across [lo, hi] as CSS: the ramp, or its bands with hard edges. Painted off the main thread. */
function barGradient(map) {
	const bands = bandCount(map);
	const colour = (t) => `rgb(${sample(map.ramp, t).join(' ')})`;
	const percent = (t) => `${(t * 100).toFixed(2)}%`;
	const stops = [];
	if (bands) for (let k = 0; k < bands; k++) stops.push(`${colour((k + 0.5) / bands)} ${percent(k / bands)} ${percent((k + 1) / bands)}`);
	else for (let i = 0; i <= 64; i++) stops.push(`${colour(i / 64)} ${percent(i / 64)}`);
	return `linear-gradient(90deg, ${stops.join(', ')})`;
}

/** The bar into a canvas, for the picture export. */
function paintBar(canvas, map) {
	const bands = bandCount(map);
	const { width, height } = canvas;
	const image = new ImageData(width, height);
	for (let i = 0; i < width; i++) {
		const value = map.lo + ((map.hi - map.lo) * (i + 0.5)) / width;
		const [r, g, b] = sample(map.ramp, rampPosition(value, map.lo, map.hi, bands));
		for (let j = 0; j < height; j++) image.data.set([r, g, b, 255], 4 * (j * width + i));
	}
	canvas.getContext('2d').putImageData(image, 0, 0);
}

function parse(text) {
	const value = Number(text.trim().replace('\u2212', '-'));
	return Number.isFinite(value) ? value : null;
}

/** actions: change(patch to the map). */
export function createLegend(actions) {
	const title = h('span', { class: 'legend-title' });
	// A gradient, not a canvas: putting pixels into a canvas waits on the GPU process, which the networks keep busy.
	const bar = h('div', { class: 'legend-bar', title: 'Drag an end to set it, the middle to shift the range' });
	const ticks = h('div', { class: 'legend-ticks' });
	const lo = h('input', { class: 'legend-end', 'aria-label': 'Range minimum', inputmode: 'decimal', spellcheck: 'false' });
	const hi = h('input', { class: 'legend-end legend-end-hi', 'aria-label': 'Range maximum', inputmode: 'decimal', spellcheck: 'false' });
	const banded = h('button', { type: 'button', class: 'legend-tool', title: 'Discrete bands', onclick: () => actions.change({ banded: !state.map.banded }) }, icon('bands'));
	const rampList = h('div', { class: 'menu-list', role: 'menu' });
	const swatch = h('span', { class: 'ramp-swatch' });
	const ramps = menu(h('button', { type: 'button', class: 'legend-tool legend-ramp', title: 'Colour ramp' }, swatch, icon('chevron')), rampList, { align: 'right' });
	const el = h('div', { class: 'legend' },
		h('div', { class: 'legend-head' }, title, h('span', { class: 'legend-tools' }, banded, ramps.el)),
		bar,
		h('div', { class: 'legend-scale' }, lo, ticks, hi));
	let state = null;

	for (const [input, end] of [[lo, 'lo'], [hi, 'hi']]) {
		const commit = () => {
			const value = parse(input.value);
			const next = { lo: state.map.lo, hi: state.map.hi, [end]: value };
			if (value === null || next.lo >= next.hi) {
				input.classList.add('invalid');
				setTimeout(() => input.classList.remove('invalid'), 600);
				render(state);
				return;
			}
			actions.change({ [end]: value });
		};
		input.addEventListener('change', commit);
		input.addEventListener('keydown', (e) => {
			if (e.key === 'Enter') input.blur();
			if (e.key === 'Escape') {
				render(state);
				input.blur();
			}
		});
	}

	bar.addEventListener('pointerdown', (e) => {
		const rect = bar.getBoundingClientRect();
		const x = e.clientX - rect.left;
		const grab = x < GRIP ? 'lo' : x > rect.width - GRIP ? 'hi' : 'shift';
		const start = { x: e.clientX, lo: state.map.lo, hi: state.map.hi };
		const perPixel = (start.hi - start.lo) / rect.width;
		const round = (v) => Number(v.toFixed(digits(start) + 1));
		bar.setPointerCapture(e.pointerId);
		const move = (ev) => {
			const d = (ev.clientX - start.x) * perPixel;
			if (grab === 'lo') actions.change({ lo: round(Math.min(start.lo + d, start.hi - perPixel * 8)) });
			else if (grab === 'hi') actions.change({ hi: round(Math.max(start.hi + d, start.lo + perPixel * 8)) });
			else actions.change({ lo: round(start.lo + d), hi: round(start.hi + d) });
		};
		const up = () => {
			bar.removeEventListener('pointermove', move);
			bar.removeEventListener('pointerup', up);
		};
		bar.addEventListener('pointermove', move);
		bar.addEventListener('pointerup', up);
	});
	bar.addEventListener('pointermove', (e) => {
		if (e.buttons) return;
		const x = e.clientX - bar.getBoundingClientRect().left;
		bar.style.cursor = x < GRIP || x > bar.clientWidth - GRIP ? 'ew-resize' : 'grab';
	});

	/** state: {title, map: {ramp, lo, hi, banded} or null, mapName} */
	function render(next) {
		state = next;
		const { map, mapName } = state;
		el.hidden = !map;
		if (!map) return;
		title.textContent = state.title;
		bar.style.background = barGradient(map);
		const d = digits(map);
		if (document.activeElement !== lo) lo.value = signed(map.lo, d);
		if (document.activeElement !== hi) hi.value = signed(map.hi, d);
		ticks.replaceChildren(...ticksFor(map, bar.clientWidth || BAR_WIDTH).map((t) => h('span', { style: `left:${(((t - map.lo) / (map.hi - map.lo)) * 100).toFixed(2)}%` }, signed(t, d))));
		banded.setAttribute('aria-pressed', String(map.banded));
		swatch.style.background = rampCss(map.ramp);
		rampList.replaceChildren(...MAPS[mapName].ramps.map((name) => h('button', {
			type: 'button', role: 'menuitemradio', class: 'menu-item', 'aria-checked': String(name === map.ramp),
			onclick: () => {
				ramps.close();
				actions.change({ ramp: name });
			},
		}, h('span', { class: 'ramp-swatch', style: `background:${rampCss(name)}` }), h('span', {}, RAMPS[name].label))));
	}
	return { el, render };
}

/** Draw the legend into a 2D canvas (for the picture export), at (x, y), `scale` device pixels per CSS pixel. */
export function paintLegend(ctx, x, y, scale, title, map) {
	const width = BAR_WIDTH * scale;
	const font = (size, weight = 400) => `${weight} ${size * scale}px "Hanken Grotesk", system-ui, sans-serif`;
	ctx.save();
	ctx.fillStyle = 'rgba(255,255,255,0.92)';
	ctx.strokeStyle = '#cddce4';
	ctx.lineWidth = scale;
	const pad = 10 * scale;
	ctx.beginPath();
	ctx.roundRect(x, y, width + 2 * pad, 64 * scale, 6 * scale);
	ctx.fill();
	ctx.stroke();
	ctx.fillStyle = '#0d2233';
	ctx.font = font(12.5, 600);
	ctx.textBaseline = 'top';
	ctx.textAlign = 'left';
	ctx.fillText(title, x + pad, y + 9 * scale);
	const bar = document.createElement('canvas');
	bar.width = width;
	bar.height = 1;
	paintBar(bar, map);
	ctx.imageSmoothingEnabled = false;
	ctx.drawImage(bar, x + pad, y + 28 * scale, width, 10 * scale);
	ctx.font = font(11.5);
	ctx.fillStyle = '#1a2b3a';
	const d = digits(map);
	const at = (v) => x + pad + ((v - map.lo) / (map.hi - map.lo)) * width;
	ctx.textAlign = 'left';
	ctx.fillText(signed(map.lo, d), at(map.lo), y + 43 * scale);
	ctx.textAlign = 'right';
	ctx.fillText(signed(map.hi, d), at(map.hi), y + 43 * scale);
	ctx.textAlign = 'center';
	for (const t of ticksFor(map, BAR_WIDTH)) ctx.fillText(signed(t, d), at(t), y + 43 * scale);
	ctx.restore();
}
