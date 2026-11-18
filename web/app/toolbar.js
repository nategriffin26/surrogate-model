// The strip above the 3D view: what colours the car (or none), the layers drawn on it, each shown, hidden or removed
// with a tap, the camera, what else is drawn, and the tools.

import { h } from './dom.js';
import { FIELDS } from './fields.js';
import { icon } from './icons.js';
import { menu } from './menu.js';
import { rampCss } from './ramps.js';
import { VIEWS } from '../viewer/viewer.js';

const VIEW_KEYS = { isometric: '0', front: '1', rear: '2', left: '3', right: '4', top: '5', bottom: '6' };
const VIEW_BY_KEY = Object.fromEntries(Object.entries(VIEW_KEYS).map(([name, key]) => [key, name]));
const TOGGLE_BY_KEY = { e: 'edges', g: 'ground', s: 'section' };

function tool(name, label, title, onclick) {
	return h(
		'button',
		{ type: 'button', class: 'tool', title, 'data-tool': name, onclick },
		icon(name),
		h('span', { class: 'tool-label' }, label),
	);
}

/**
 * actions: what the controls and their keys do (field(name), layer(name, visible), removeLayer(name), view, fit,
 * projection, toggle), and hasCar(), whether a car is on screen for the keys to act on.
 */
export function createToolbar(actions) {
	const fields = h('div', { class: 'fields', role: 'radiogroup', 'aria-label': 'Field' });
	const projection = ['perspective', 'orthographic'].map((p) =>
		h(
			'button',
			{
				type: 'button',
				role: 'menuitemradio',
				class: 'menu-item',
				'data-projection': p,
				onclick: () => {
					views.close();
					actions.projection(p);
				},
			},
			h('span', {}, p === 'perspective' ? 'Perspective' : 'Orthographic'),
			h('kbd', {}, p === 'perspective' ? 'P' : 'O'),
		),
	);
	const viewList = h(
		'div',
		{ class: 'menu-list', role: 'menu' },
		Object.entries(VIEWS).map(([name, view]) =>
			h(
				'button',
				{
					type: 'button',
					role: 'menuitem',
					class: 'menu-item',
					'data-view': name,
					onclick: () => {
						views.close();
						actions.view(name);
					},
				},
				h('span', {}, view.label),
				h('kbd', {}, VIEW_KEYS[name]),
			),
		),
		h('hr', { class: 'menu-rule' }),
		projection,
	);
	const views = menu(tool('views', 'Views', 'Standard views and projection', null), viewList);
	const tools = {
		fit: tool('fit', 'Fit', 'Fit to view (F)', () => actions.fit()),
		edges: tool('edges', 'Edges', 'Mesh edges (E)', () => actions.toggle('edges')),
		ground: tool('ground', 'Ground', 'Ground (G)', () => actions.toggle('ground')),
		section: tool('section', 'Section', 'Section cut (S)', () => actions.toggle('section')),
	};
	const controls = menu(
		h('button', { type: 'button', class: 'tool tool-quiet', title: 'Mouse and keys' }, icon('controls')),
		h(
			'dl',
			{ class: 'controls-list' },
			[
				['Rotate', 'Drag'],
				['Pan', 'Right-drag, Shift-drag, two fingers'],
				['Zoom', 'Wheel, pinch'],
				['Orbit centre', 'Double-click'],
				['Probe', 'Click'],
				['Fit', 'F'],
				['Views', '0–6'],
				['Edges · Ground', 'E · G'],
				['Section', 'S'],
			].map(([what, how]) => [h('dt', {}, what), h('dd', {}, how)]),
		),
		{ align: 'right' },
	);
	// Fields and layers share one row that scrolls sideways when it does not fit, so no control is ever cut off; the
	// view tools at the end keep their place.
	const chips = h('div', { class: 'layer-chips', role: 'group', 'aria-label': 'Layers' });
	const scroller = h('div', { class: 'toolbar-row' }, fields, chips);
	const el = h(
		'div',
		{ class: 'toolbar' },
		scroller,
		h(
			'div',
			{ class: 'toolbar-end' },
			h('div', { class: 'tool-group' }, views.el, tools.fit),
			h('div', { class: 'tool-group' }, tools.edges, tools.ground),
			h('div', { class: 'tool-group' }, tools.section),
			controls.el,
		),
	);

	/** Which ends of the row are cut off, for the fade that says there is more. */
	function markClipped() {
		const { scrollLeft, scrollWidth, clientWidth } = scroller;
		const ends = [];
		if (scrollLeft > 1) ends.push('start');
		if (scrollLeft + clientWidth < scrollWidth - 1) ends.push('end');
		scroller.dataset.clipped = ends.join(' ');
	}
	scroller.addEventListener('scroll', markClipped, { passive: true });
	new ResizeObserver(markClipped).observe(scroller);
	// A mouse wheel turns the row too, where it would otherwise scroll nothing.
	scroller.addEventListener(
		'wheel',
		(e) => {
			if (scroller.scrollWidth <= scroller.clientWidth || Math.abs(e.deltaX) >= Math.abs(e.deltaY)) return;
			e.preventDefault();
			scroller.scrollLeft += e.deltaY;
		},
		{ passive: false },
	);

	const removeButton = (name) =>
		h(
			'button',
			{
				type: 'button',
				class: 'chip-remove',
				title: `Remove ${name}`,
				'aria-label': `Remove ${name}`,
				onclick: () => actions.removeLayer(name),
			},
			icon('close'),
		);

	/** Each drawing layer's chip, by name, kept from render to render so showing and hiding it can fade. */
	const chipOf = new Map();
	function chip(layer) {
		let made = chipOf.get(layer.name);
		if (!made) {
			const swatch = h('span', { class: 'chip-swatch' });
			const toggle = h(
				'button',
				{ type: 'button', class: 'chip-toggle' },
				swatch,
				h('span', { class: 'chip-name' }, layer.name),
			);
			made = {
				el: h('span', { class: 'chip', 'data-layer': layer.name }, toggle, removeButton(layer.name)),
				toggle,
				swatch,
			};
			chipOf.set(layer.name, made);
		}
		const { el: chipEl, toggle, swatch } = made;
		chipEl.dataset.hidden = String(!layer.visible);
		toggle.setAttribute('aria-pressed', String(layer.visible));
		toggle.title = `${layer.visible ? 'Hide' : 'Show'} ${layer.name}`;
		toggle.onclick = () => actions.layer(layer.name, !layer.visible);
		swatch.style.setProperty('--swatch', layer.colour);
		return chipEl;
	}

	// The keys the Mouse and keys list names, unless a text field has the focus.
	addEventListener('keydown', (e) => {
		if (e.metaKey || e.ctrlKey || e.altKey || e.target.closest('input, textarea, select, [contenteditable]')) return;
		if (!actions.hasCar()) return;
		const key = e.key.toLowerCase();
		if (VIEW_BY_KEY[key]) actions.view(VIEW_BY_KEY[key]);
		else if (key === 'f') actions.fit();
		else if (key === 'p' || key === 'o') actions.projection(key === 'p' ? 'perspective' : 'orthographic');
		else if (TOGGLE_BY_KEY[key]) actions.toggle(TOGGLE_BY_KEY[key]);
	});

	/**
	 * state: {fields: [names], field ('none' when the car is plain), maps, enabled, projection, display: {edges, ground},
	 * section, layers: the car's layers, [{name, kind, colour, visible, field}]}. Tapping the field shown turns the field
	 * off; a field a layer added has a remove control, as each drawing layer's chip has.
	 */
	function render(state) {
		const fieldLayers = new Map(state.layers.filter((l) => l.kind === 'field').map((l) => [l.field.key, l]));
		fields.replaceChildren(
			...state.fields.map((f) => {
				const map = FIELDS[f].map ? state.maps?.[FIELDS[f].map] : null;
				const button = h(
					'button',
					{
						type: 'button',
						role: 'radio',
						class: 'field',
						'data-field': f,
						title: FIELDS[f].tip,
						'aria-checked': String(state.enabled && f === state.field),
						disabled: !state.enabled,
						onclick: () => actions.field(f === state.field ? 'none' : f),
					},
					h('span', {}, FIELDS[f].label),
					h('span', { class: 'field-ramp', style: map ? `background:${rampCss(map.ramp)}` : null }),
				);
				const layer = fieldLayers.get(f);
				return layer ? h('span', { class: 'field-item' }, button, removeButton(layer.name)) : button;
			}),
		);
		const shown = state.layers.filter((l) => l.kind === 'drawing').map(chip);
		for (const name of chipOf.keys()) if (!state.layers.some((l) => l.name === name)) chipOf.delete(name);
		// Rebuilt only when the chips change: a chip moved in the page would lose its fade.
		if (shown.length !== chips.children.length || shown.some((c, i) => c !== chips.children[i]))
			chips.replaceChildren(...shown);
		chips.hidden = shown.length === 0;
		markClipped();
		for (const b of projection) b.setAttribute('aria-checked', String(b.dataset.projection === state.projection));
		tools.edges.setAttribute('aria-pressed', String(state.display.edges));
		tools.ground.setAttribute('aria-pressed', String(state.display.ground));
		tools.section.setAttribute('aria-pressed', String(state.section));
		tools.section.disabled = !state.enabled;
	}
	return { el, render };
}
