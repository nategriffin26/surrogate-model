// The ensemble's networks: in the diagram, one glyph per network on a bus from the inputs and back to their mean, its
// layers as wide as they really are; in detail, one network's layers with their sizes and activation, and which
// training cars each network was fitted on and which it never saw.

import { h, s } from '../dom.js';
import { count, signed } from '../format.js';
import { KIND_COLORS } from './inputs.js';

/** Horizontal distance from a figure's left edge back to the diagram's rail, in pixels. */
const RAIL_X = -19.5;

const GLYPH = { top: 16, step: 6, bar: 3 };

/** The diagram's networks between the inputs and their mean, each with its Cp at the point. */
export function networksFigure(ctx) {
	const { card, width, shown, act } = ctx;
	const n = card.networks.length;
	const slot = width / n;
	const glyph = Math.min(56, slot * 0.72);
	const widest = Math.max(...card.networks.flatMap((net) => net.layers));
	const layersOut = Math.max(...card.networks.map((net) => net.layers.length));
	const bottom = GLYPH.top + (layersOut - 1) * GLYPH.step + GLYPH.bar;
	const valueY = bottom + 15;
	const busY = valueY + 13;
	const cx = (k) => slot * (k + 0.5);
	return s('svg', { class: 'net-figure', width, height: busY + 4, viewBox: `0 0 ${width} ${busY + 4}`, role: 'group', 'aria-label': `${n} networks` },
		s('path', { class: 'wire', d: `M${RAIL_X} 6H${cx(n - 1)}` }),
		s('path', { class: 'wire', d: `M${cx(0)} ${busY}H${RAIL_X}` }),
		card.networks.map((net, k) => {
			const x = cx(k);
			const cp = shown?.explanation.cpModels[k];
			return s('g', {
				class: 'net-glyph',
				'data-selected': String(k === ctx.network),
				tabindex: 0,
				role: 'button',
				'aria-label': `Network ${k + 1}`,
				onclick: () => act.network(k),
				onkeydown: (e) => (e.key === 'Enter' || e.key === ' ') && act.network(k),
				onpointerenter: (e) => act.tip.show(e.currentTarget, h('span', {}, h('b', {}, `Network ${k + 1}`), ` ${net.layers.join(' · ')}`)),
				onpointerleave: () => act.tip.hide(),
			},
			s('rect', { class: 'net-hit', x: x - slot / 2, y: 0, width: slot, height: busY }),
			s('path', { class: 'wire', d: `M${x} 6V${GLYPH.top - 2}` }),
			net.layers.map((size, l) => {
				const w = Math.max(GLYPH.bar, (glyph * size) / widest);
				return s('rect', { class: 'layer', style: `--l: ${l}`, x: x - w / 2, y: GLYPH.top + l * GLYPH.step, width: w, height: GLYPH.bar, rx: GLYPH.bar / 2 });
			}),
			s('text', { class: 'net-value num', x, y: valueY, 'text-anchor': 'middle' }, cp === undefined ? '' : signed(cp, 3)),
			s('path', { class: 'wire', d: `M${x} ${valueY + 4}V${busY}` }));
		}));
}

/** One network's layers: bars as tall as they are wide, the dense connections between them, and their sizes. */
function layerDiagram(ctx, net) {
	const { card, groups, width } = ctx;
	const { layers } = net;
	const H = 150;
	const mid = 70;
	const tallest = 104;
	const widest = Math.max(...layers);
	const bar = 14;
	const x = (l) => 2 + (l * (width - 4 - bar)) / (layers.length - 1);
	const height = (size) => Math.max(2, (tallest * size) / widest);
	const parts = [];
	for (let l = 0; l + 1 < layers.length; l++) {
		const a = height(layers[l]) / 2, b = height(layers[l + 1]) / 2;
		parts.push(s('path', { class: 'dense', d: `M${x(l) + bar} ${mid - a}L${x(l + 1)} ${mid - b}V${mid + b}L${x(l) + bar} ${mid + a}Z` }));
	}
	layers.forEach((size, l) => {
		const top = mid - height(size) / 2;
		const label = l === 0 ? 'in' : l === layers.length - 1 ? 'Cp' : card.activation;
		if (l === 0) {
			// The input layer, split into its groups.
			let y = top;
			for (const g of groups) {
				const share = (height(size) * g.columns.length) / size;
				parts.push(s('rect', { x: x(l), y, width: bar, height: Math.max(0.5, share - 0.6), fill: KIND_COLORS[g.kind] }, s('title', {}, `${g.label} ${g.columns.length}`)));
				y += share;
			}
		} else {
			parts.push(s('rect', { class: l === layers.length - 1 ? 'layer-out' : 'layer-hidden', x: x(l), y: top, width: bar, height: height(size), rx: 1.5 }));
		}
		parts.push(s('text', { class: 'layer-label', x: x(l) + bar / 2, y: top - 7, 'text-anchor': 'middle' }, label));
		parts.push(s('text', { class: 'layer-size num', x: x(l) + bar / 2, y: H - 4, 'text-anchor': 'middle' }, count(size)));
	});
	return s('svg', { class: 'layer-diagram', width, height: H, viewBox: `0 0 ${width} ${H}`, role: 'img', 'aria-label': `Layers ${layers.join(', ')}` }, parts);
}

/** Which training cars each network was fitted on, and which it never saw. */
function foldGrid(ctx) {
	const { card, act } = ctx;
	const runs = [...card.trainRuns].sort((a, b) => a - b);
	return h('table', { class: 'fold-grid' },
		h('thead', {}, h('tr', {}, h('th', { scope: 'col' }), runs.map((run) => h('th', { scope: 'col', class: 'num' }, run)))),
		h('tbody', {}, card.networks.map((net, k) => h('tr', { 'data-selected': String(k === ctx.network), onclick: () => act.network(k) },
			h('th', { scope: 'row', class: 'num' }, k + 1),
			runs.map((run) => {
				const role = net.validationRuns.includes(run) ? 'held' : net.trainRuns.includes(run) ? 'fit' : 'none';
				return h('td', { 'data-role': role, title: `Run ${run}: ${role === 'held' ? 'held out' : role === 'fit' ? 'fitted' : 'not used'}` }, h('i'));
			})))));
}

/** Why several networks: each car is held out of one, which scores it, and their disagreement is the spread. */
function heldOutCaption(card) {
	const held = new Set(card.networks.map((net) => net.validationRuns.length));
	const n = card.networks.length;
	const kept = held.size === 1 ? `, each kept from ${[...held][0]} of the ${card.trainRuns.length} cars` : '';
	return `${n} networks${kept}: every car is scored by one that never saw it, and where they disagree, σ grows.`;
}

export function networksDetail(ctx) {
	const { card } = ctx;
	return h('div', { class: 'networks-detail' },
		layerDiagram(ctx, card.networks[ctx.network]),
		h('div', { class: 'detail-head' }, h('span', {}, 'Training cars'),
			h('span', { class: 'grid-key' }, h('span', { 'data-role': 'fit' }, h('i'), 'Fitted'), h('span', { 'data-role': 'held' }, h('i'), 'Held out'))),
		foldGrid(ctx),
		h('p', { class: 'caption' }, heldOutCaption(card)));
}
