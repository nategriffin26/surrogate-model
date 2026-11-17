// The strip above the 3D view: what colours the car, the camera, what is drawn, and the tools.

import { h } from './dom.js';
import { FIELDS } from './fields.js';
import { icon } from './icons.js';
import { menu } from './menu.js';
import { rampCss } from './ramps.js';
import { VIEWS } from '../viewer/viewer.js';

export const VIEW_KEYS = { isometric: '0', front: '1', rear: '2', left: '3', right: '4', top: '5', bottom: '6' };

function tool(name, label, title, onclick) {
	return h('button', { type: 'button', class: 'tool', title, 'data-tool': name, onclick }, icon(name), h('span', { class: 'tool-label' }, label));
}

export function createToolbar(actions) {
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
		compare: tool('compare', 'Compare', 'Predicted and CFD side by side (C)', () => actions.toggle('compare')),
	};
	const controls = menu(h('button', { type: 'button', class: 'tool tool-quiet', title: 'Mouse and keys' }, icon('controls')), h('dl', { class: 'controls-list' },
		[['Rotate', 'Drag'], ['Pan', 'Right-drag, Shift-drag, two fingers'], ['Zoom', 'Wheel, pinch'], ['Orbit centre', 'Double-click'],
			['Probe', 'Click'], ['Fit', 'F'], ['Views', '0–6'], ['Edges · Ground', 'E · G'], ['Section · Compare', 'S · C']]
			.map(([what, how]) => [h('dt', {}, what), h('dd', {}, how)])), { align: 'right' });
	const el = h('div', { class: 'toolbar' },
		fields,
		h('div', { class: 'toolbar-end' },
			h('div', { class: 'tool-group' }, views.el, tools.fit),
			h('div', { class: 'tool-group' }, tools.edges, tools.ground),
			h('div', { class: 'tool-group' }, tools.section, tools.compare),
			controls.el));

	/** state: {fields: [names], field, maps, enabled, projection, display: {edges, ground}, section, compare, canCompare} */
	function render(state) {
		fields.replaceChildren(...state.fields.map((f) => {
			const map = FIELDS[f].map ? state.maps?.[FIELDS[f].map] : null;
			return h('button', {
				type: 'button', role: 'radio', class: 'field', 'data-field': f, title: FIELDS[f].tip,
				'aria-checked': String(f === state.field), disabled: !state.enabled && f !== 'none',
				onclick: () => actions.field(f),
			}, h('span', {}, FIELDS[f].label), h('span', { class: 'field-ramp', style: map ? `background:${rampCss(map.ramp)}` : null }));
		}));
		for (const b of projection) b.setAttribute('aria-checked', String(b.dataset.projection === state.projection));
		tools.edges.setAttribute('aria-pressed', String(state.display.edges));
		tools.ground.setAttribute('aria-pressed', String(state.display.ground));
		tools.section.setAttribute('aria-pressed', String(state.section));
		tools.section.disabled = !state.enabled;
		tools.compare.setAttribute('aria-pressed', String(state.compare));
		tools.compare.hidden = !state.canCompare;
	}
	return { el, render };
}
