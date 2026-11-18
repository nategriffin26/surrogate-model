// One network working on the point shown: its inputs as it reads them, its hidden layers and its Cp, each unit a
// row coloured by the value it holds for this point. A new point's values sweep through it from the inputs to Cp.

import { h } from '../dom.js';
import { signed } from '../format.js';
import { KIND_COLORS } from './inputs.js';

const HEIGHT = 288;
const TOP = 22;
const BOTTOM = 26;
const STRIP = 26;
const KIND_BAND = 5;
const NODE = 21;
/** How long each layer waits after the one before it when a new point's values sweep through. */
const SWEEP_MS = 110;
const REDUCED_MOTION = matchMedia('(prefers-reduced-motion: reduce)');

const NEGATIVE = [163, 91, 26];
const POSITIVE = [11, 111, 191];

/** A unit's colour: white at zero, towards blue for positive values and orange for negative, saturating smoothly. */
export function shade(value, scale) {
	const strength = Math.tanh(Math.abs(value) / scale);
	const [r, g, b] = value < 0 ? NEGATIVE : POSITIVE;
	const mix = (c) => Math.round(255 + (c - 255) * strength);
	return `rgb(${mix(r)}, ${mix(g)}, ${mix(b)})`;
}

export class NetworkFigure {
	/**
	 * @param {{ card: import('../../engine/result.js').ModelCard, act: object }} options
	 *   `act.tip(target, content, clientX, clientY)`, `act.untip()`, `act.focusColumns(columns | null)`
	 */
	constructor({ card, act }) {
		this.card = card;
		this.act = act;
		this.canvas = h('canvas', { class: 'network-canvas', role: 'img' });
		this.el = h('div', { class: 'network-figure' }, this.canvas);
		this.layers = null;
		this.cp = null;
		this.network = 0;
		this.shownUpTo = Infinity;
		this.highlighted = null;
		this.canvas.addEventListener('pointermove', (e) => this.hover(e));
		this.canvas.addEventListener('pointerleave', () => {
			act.untip();
			act.focusColumns(null);
		});
	}

	/** Strips left to right: x of each, and its unit count. */
	geometry() {
		const width = this.el.clientWidth;
		const sizes = this.card.networks[this.network].layers;
		const strips = sizes.length - 1;
		const right = width - 2 * NODE - 8;
		const gap = (right - KIND_BAND - strips * STRIP) / (strips - 1 + 0.6);
		const xs = Array.from({ length: strips }, (_, l) => KIND_BAND + l * (STRIP + gap));
		return { width, sizes, xs, gap, node: [right + NODE + 4, TOP + (HEIGHT - TOP - BOTTOM) / 2] };
	}

	/** Show network `k`'s values for a point; `sweep` runs them through it from the inputs on. */
	update(activations, cpModels, network, sweep) {
		this.network = network;
		this.layers = activations[network];
		this.cp = cpModels[network];
		cancelAnimationFrame(this.frame);
		if (!sweep || REDUCED_MOTION.matches) {
			this.shownUpTo = Infinity;
			this.draw();
			return;
		}
		// The previous point's values stay in the layers the sweep has not reached.
		const start = performance.now();
		const step = (now) => {
			this.shownUpTo = Math.floor((now - start) / SWEEP_MS);
			this.draw();
			if (this.shownUpTo <= this.layers.length) this.frame = requestAnimationFrame(step);
		};
		this.previous = this.drawnLayers ?? this.layers;
		this.previousCp = this.drawnCp ?? this.cp;
		this.frame = requestAnimationFrame(step);
	}

	/** Mark these input rows, or none. */
	highlight(columns) {
		this.highlighted = columns;
		this.draw();
	}

	draw() {
		if (!this.layers) return;
		const ratio = devicePixelRatio;
		const { width, sizes, xs, gap, node } = this.geometry();
		if (this.canvas.width !== Math.round(width * ratio)) {
			this.canvas.width = Math.round(width * ratio);
			this.canvas.height = Math.round(HEIGHT * ratio);
			this.canvas.style.height = `${HEIGHT}px`;
		}
		const g = this.canvas.getContext('2d');
		g.setTransform(ratio, 0, 0, ratio, 0, 0);
		g.clearRect(0, 0, width, HEIGHT);
		const body = getComputedStyle(this.el).fontFamily;
		const span = HEIGHT - TOP - BOTTOM;
		const layers = this.layers.map((values, l) => (l < this.shownUpTo ? values : this.previous[l]));
		const cp = this.shownUpTo > this.layers.length ? this.cp : this.previousCp;
		this.drawnLayers = layers;
		this.drawnCp = cp;

		// Every unit of a layer feeds every unit of the next: a band between the strips.
		for (let l = 0; l < xs.length; l++) {
			const from = xs[l] + STRIP;
			const to = l + 1 < xs.length ? xs[l + 1] : node[0] - NODE;
			const fan = g.createLinearGradient(from, 0, to, 0);
			fan.addColorStop(0, 'rgba(11, 111, 191, 0.09)');
			fan.addColorStop(1, 'rgba(11, 111, 191, 0.04)');
			g.fillStyle = fan;
			g.beginPath();
			g.moveTo(from, TOP);
			g.lineTo(to, l + 1 < xs.length ? TOP : node[1] - NODE * 0.6);
			g.lineTo(to, l + 1 < xs.length ? TOP + span : node[1] + NODE * 0.6);
			g.lineTo(from, TOP + span);
			g.fill();
		}

		layers.forEach((values, l) => {
			const rows = values.length;
			const step = span / rows;
			for (let i = 0; i < rows; i++) {
				g.fillStyle = shade(values[i], l === 0 ? 1.5 : 1);
				g.fillRect(xs[l], TOP + i * step, STRIP, Math.max(step, 0.6) + 0.35);
			}
			g.strokeStyle = 'rgba(13, 34, 51, 0.18)';
			g.lineWidth = 1;
			g.strokeRect(xs[l] - 0.5, TOP - 0.5, STRIP + 1, span + 1);
		});

		// The inputs' kinds beside them, and the rows the reader is pointing at.
		const inputStep = span / sizes[0];
		this.card.columns.forEach((column, i) => {
			g.fillStyle = KIND_COLORS[column.kind];
			g.fillRect(0, TOP + i * inputStep, KIND_BAND - 1.5, inputStep + 0.35);
		});
		if (this.highlighted) {
			g.fillStyle = '#0d2233';
			for (const i of this.highlighted) g.fillRect(xs[0] + STRIP + 2, TOP + i * inputStep, 4, Math.max(inputStep, 1.5));
		}

		// Cp out.
		g.beginPath();
		g.arc(node[0], node[1], NODE, 0, 2 * Math.PI);
		g.fillStyle = '#fff';
		g.fill();
		g.lineWidth = 2;
		g.strokeStyle = '#0d2233';
		g.stroke();
		g.fillStyle = '#0d2233';
		g.textAlign = 'center';
		g.textBaseline = 'middle';
		g.font = `600 11.5px ${body}`;
		g.fillText(signed(cp, 3), node[0], node[1]);

		g.font = `500 11px ${body}`;
		g.fillStyle = '#58727f';
		g.textBaseline = 'alphabetic';
		sizes.slice(0, -1).forEach((size, l) => g.fillText(l === 0 ? `${size} in` : String(size), xs[l] + STRIP / 2, HEIGHT - 8));
		g.fillText('Cp', node[0], HEIGHT - 8);
		g.textBaseline = 'top';
		g.fillText(this.card.activation, (xs[1] + xs[0] + STRIP) / 2, 4);
		for (let l = 1; l < xs.length - 1; l++) g.fillText(this.card.activation, (xs[l + 1] + xs[l] + STRIP) / 2, 4);
		this.gap = gap;
	}

	/** Name the unit under the pointer; an input also lights where it was read on the car. */
	hover(e) {
		if (!this.layers) return;
		const rect = this.canvas.getBoundingClientRect();
		const x = e.clientX - rect.left;
		const y = e.clientY - rect.top;
		const { xs, sizes, node } = this.geometry();
		const span = HEIGHT - TOP - BOTTOM;
		const l = xs.findIndex((left) => x >= left - (left === xs[0] ? KIND_BAND : 0) && x <= left + STRIP);
		if (l >= 0 && y >= TOP && y < TOP + span) {
			const i = Math.min(sizes[l] - 1, Math.floor(((y - TOP) / span) * sizes[l]));
			const value = this.layers[l][i];
			if (l === 0) {
				const column = this.card.columns[i];
				this.act.tip(e.currentTarget, h('span', {}, h('b', {}, column.name), ` ${signed(value, 2)} as read`), e.clientX, e.clientY);
				this.act.focusColumns([i]);
			} else {
				this.act.tip(e.currentTarget, h('span', {}, h('b', {}, `Layer ${l}, unit ${i + 1}`), ` ${signed(value, 3)}`), e.clientX, e.clientY);
				this.act.focusColumns(null);
			}
			return;
		}
		if (Math.hypot(x - node[0], y - node[1]) <= NODE) {
			this.act.tip(e.currentTarget, h('span', {}, h('b', {}, `Network ${this.network + 1}`), ` Cp ${signed(this.cp, 3)}`), e.clientX, e.clientY);
			return;
		}
		this.act.untip();
		this.act.focusColumns(null);
	}
}
