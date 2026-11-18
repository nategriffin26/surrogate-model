// The strip above the 3D view: what colours the car, the camera, what is drawn, and the tools.

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
	return h('button', { type: 'button', class: 'tool', title, 'data-tool': name, onclick }, icon(name), h('span', { class: 'tool-label' }, label));
}

/**
 * actions: what the controls and their keys do, and hasCar(), whether a car is on screen for the keys to act on.
 * layers: the Layers menu, beside the section toggle as another thing drawn on the car.
 */
export function createToolbar(actions, layers) {
	const fields = h('div', { class: 'fields', role: 'radiogroup', 'aria-label': 'Field' });
	const projection = ['perspective', 'orthographic'].map((p) => h('button', {
		type: 'button', role: 'menuitemradio', class: 'menu-item', 'data-projection': p,
		onclick: () => { views.close(); actions.projection(p); },
	}, h('span', {}, p === 'perspective' ? 'Perspective' : 'Orthographic'), h('kbd', {}, p === 'perspective' ? 'P' : 'O')));
	const viewList = h('div', { class: 'menu-list', role: 'menu' },
		Object.entries(VIEWS).map(([name, view]) => h('button', { type: 'button', role: 'menuitem', class: 'menu-item', 'data-view': name, onclick: () => { views.close(); actions.view(name); } },
			h('span', {}, view.label), h('kbd', {}, VIEW_KEYS[name]))),
		h('hr', { class: 'menu-rule' }),
		projection);
	const views = menu(tool('views', 'Views', 'Standard views and projection', null), viewList);
	const tools = {
		fit: tool('fit', 'Fit', 'Fit to view (F)', () => actions.fit()),
		edges: tool('edges', 'Edges', 'Mesh edges (E)', () => actions.toggle('edges')),
		ground: tool('ground', 'Ground', 'Ground (G)', () => actions.toggle('ground')),
		section: tool('section', 'Section', 'Section cut (S)', () => actions.toggle('section')),
	};
	const controls = menu(h('button', { type: 'button', class: 'tool tool-quiet', title: 'Mouse and keys' }, icon('controls')), h('dl', { class: 'controls-list' },
		[['Rotate', 'Drag'], ['Pan', 'Right-drag, Shift-drag, two fingers'], ['Zoom', 'Wheel, pinch'], ['Orbit centre', 'Double-click'],
			['Probe', 'Click'], ['Fit', 'F'], ['Views', '0–6'], ['Edges · Ground', 'E · G'], ['Section', 'S']]
			.map(([what, how]) => [h('dt', {}, what), h('dd', {}, how)])), { align: 'right' });
	const el = h('div', { class: 'toolbar' },
		fields,
		h('div', { class: 'toolbar-end' },
			h('div', { class: 'tool-group' }, views.el, tools.fit),
			h('div', { class: 'tool-group' }, tools.edges, tools.ground),
			h('div', { class: 'tool-group' }, tools.section, layers),
			controls.el));

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

	/** state: {fields: [names], field, maps, enabled, projection, display: {edges, ground}, section} */
	function render(state) {
		fields.replaceChildren(...state.fields.map((f) => {
			const map = FIELDS[f].map ? state.maps?.[FIELDS[f].map] : null;
			return h('button', {
				type: 'button', role: 'radio', class: 'field', 'data-field': f, title: FIELDS[f].tip,
				'aria-checked': String(state.enabled && f === state.field), disabled: !state.enabled,
				onclick: () => actions.field(f),
			}, h('span', {}, FIELDS[f].label), h('span', { class: 'field-ramp', style: map ? `background:${rampCss(map.ramp)}` : null }));
		}));
		for (const b of projection) b.setAttribute('aria-checked', String(b.dataset.projection === state.projection));
		tools.edges.setAttribute('aria-pressed', String(state.display.edges));
		tools.ground.setAttribute('aria-pressed', String(state.display.ground));
		tools.section.setAttribute('aria-pressed', String(state.section));
		tools.section.disabled = !state.enabled;
	}
	return { el, render };
}
