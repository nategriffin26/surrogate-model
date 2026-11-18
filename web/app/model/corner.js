// The card over the viewer's corner that says what is drawn on the car: a switch per drawing layer, and for the
// boxes one per size. It folds to its title, to clear the car on a small screen.

import { h } from '../dom.js';
import { icon } from '../icons.js';
import { KIND_COLORS } from './inputs.js';
import { LAYERS } from './measures.js';

/**
 * @param {HTMLElement} host  the viewer's corner slot
 * @param {number[]} scales   the boxes' sizes, in metres
 * @param {{ toggleLayer: (id: string) => void, toggleScale: (scale: number) => void,
 *   hoverLayer: (id: string | null) => void }} act
 */
export function createCorner(host, scales, act) {
	const switches = new Map();
	const sizes = new Map();
	const rows = LAYERS.map((layer) => {
		const toggle = h('button', {
			type: 'button',
			class: 'layer-switch',
			role: 'switch',
			'aria-checked': 'true',
			'aria-label': layer.label,
			onclick: () => act.toggleLayer(layer.id),
		});
		switches.set(layer.id, toggle);
		const sizeChips = layer.id === 'bend' ? h('span', { class: 'scale-chips', role: 'group', 'aria-label': 'Box sizes' },
			scales.map((scale) => {
				const chip = h('button', { type: 'button', class: 'scale-chip', 'aria-pressed': 'true', onclick: () => act.toggleScale(scale) }, Math.round(scale * 100));
				sizes.set(scale, chip);
				return chip;
			}), h('span', { class: 'scale-unit' }, 'cm')) : null;
		// The whole row toggles the layer, except a size chip.
		return h('div', {
			class: 'layer-row',
			'data-layer': layer.id,
			onclick: (e) => !e.target.closest('.scale-chip, .layer-switch') && act.toggleLayer(layer.id),
			onpointerenter: () => act.hoverLayer(layer.id),
			onpointerleave: () => act.hoverLayer(null),
		}, h('i', { class: 'swatch', style: `--kind: ${KIND_COLORS[layer.kind]}` }), h('span', { class: 'layer-label' }, layer.label), sizeChips, toggle);
	});
	const head = h('button', {
		type: 'button',
		class: 'model-card-head',
		'aria-expanded': 'true',
		onclick: () => {
			const open = head.getAttribute('aria-expanded') !== 'true';
			head.setAttribute('aria-expanded', String(open));
			list.hidden = !open;
		},
	}, 'Drawn on the car', icon('chevron'));
	const list = h('div', { class: 'layer-list' }, rows);
	const card = h('div', { class: 'model-card' }, head, list);
	host.append(card);

	return {
		/** Show which layers and sizes are on; `hidden` when the car shows prediction beside CFD. */
		render({ shown, scaleShown, hidden }) {
			card.hidden = hidden;
			for (const [id, toggle] of switches) {
				toggle.setAttribute('aria-checked', String(shown.has(id)));
				toggle.parentElement.dataset.on = String(shown.has(id));
			}
			for (const [scale, chip] of sizes) {
				chip.setAttribute('aria-pressed', String(scaleShown.has(scale)));
				chip.disabled = !shown.has('bend');
			}
		},
	};
}
