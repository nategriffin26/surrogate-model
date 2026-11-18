// The dock under the 3D view: the section plot, the probe table, and the panels for comparing cars and making
// variants. What each shows is exported from the Export menu.

import { h, s } from './dom.js';
import { icon } from './icons.js';

/** The least height, in px, a section plot is drawn in beside a panel's numbers: below it the curves are too flat to read. */
const SLOT_MIN_PX = 130;
/** The dock's height as its top edge is dragged: at least DOCK_MIN_PX, at most DOCK_MAX_SHARE of the window's height. */
const DOCK_MIN_PX = 150;
const DOCK_MAX_SHARE = 0.6;

/**
 * view: the shared view (its dock tab and section). actions: tab(name), collapse(). resizer: the edge above the dock,
 * dragged to set its height. The section tool supplies the section plot and its controls, the probes their table and
 * its actions. Each of `panels` is a tab of its own, {name, label, el, controls?, render(), count?(), sectionSlot?},
 * its count shown in the tab when above zero. A panel's `sectionSlot` is the place in its layout where the section
 * plot sits, beside its numbers, while the section is on.
 */
export function createDock({ section, probes, panels, active, view, actions, resizer }) {
	const tabs = h('div', { class: 'dock-tabs', role: 'tablist' });
	const controls = h('div', { class: 'dock-controls' });
	const key = h('div', { class: 'plot-key' });
	const collapse = h('button', { type: 'button', class: 'icon-button dock-collapse', title: 'Collapse plots', onclick: () => actions.collapse() }, icon('chevron'));
	const svg = s('svg', { class: 'plot' });
	const readout = h('div', { class: 'plot-readout' });
	const table = h('div', { class: 'dock-table' });
	// The key and the cursor readout sit together at the plot's top left, next to the axis they describe.
	const sectionPlot = h('div', { class: 'section-plot' }, svg, h('div', { class: 'plot-head' }, key, readout));
	const body = h('div', { class: 'dock-body' }, sectionPlot, table);
	const el = h('div', { class: 'dock' }, h('div', { class: 'dock-head' }, tabs, controls, collapse), body);
	new ResizeObserver(() => drawPlotNow()).observe(body);

	resizer.addEventListener('pointerdown', (e) => {
		resizer.setPointerCapture(e.pointerId);
		document.body.classList.add('resizing');
		const move = (ev) => {
			const bottom = el.parentElement.getBoundingClientRect().bottom;
			document.body.style.setProperty('--dock', `${Math.min(innerHeight * DOCK_MAX_SHARE, Math.max(DOCK_MIN_PX, bottom - ev.clientY))}px`);
		};
		const end = () => {
			document.body.classList.remove('resizing');
			resizer.removeEventListener('pointermove', move);
			resizer.removeEventListener('pointerup', end);
			resizer.removeEventListener('pointercancel', end);
		};
		resizer.addEventListener('pointermove', move);
		resizer.addEventListener('pointerup', end);
		resizer.addEventListener('pointercancel', end);
	});

	const panelShown = () => panels.find((p) => p.name === view.dock);
	/** Where the section plot is drawn: the whole dock on the Section tab, beside a panel's numbers, or nowhere. */
	function sectionHost() {
		if (view.dock === 'section') return body;
		return view.section.on ? panelShown()?.sectionSlot ?? null : null;
	}

	function drawPlotNow() {
		const c = active();
		if (!c?.result || body.clientHeight < 40) return;
		readout.replaceChildren();
		table.hidden = view.dock === 'section';
		if (view.dock === 'probes') table.replaceChildren(probes.table(c));
		else if (view.dock !== 'section') {
			const panel = panelShown();
			if (!panel) throw new Error(`Unknown dock tab ${view.dock}`);
			// In the page first, so a panel that draws to its own size can measure it.
			if (table.firstChild !== panel.el) table.replaceChildren(panel.el);
			panel.render();
		}
		const host = sectionHost();
		// Beside a panel whose own rows leave the plot too little height (a phone held sideways), the numbers keep the
		// dock's width rather than sit beside a plot too flat to read.
		if (!host || (host !== body && host.parentElement.clientHeight < SLOT_MIN_PX)) {
			sectionPlot.remove();
			section.forget();
			return;
		}
		if (sectionPlot.parentNode !== host) host.prepend(sectionPlot);
		section.drawSection(svg, readout, host.clientWidth, host.clientHeight);
	}

	/** Redraw for the active case: tabs, controls, key and plot. A case still being analysed has no plots yet. */
	function render() {
		const c = active();
		const result = Boolean(c?.result);
		const counted = (label, count) => (count ? `${label} (${count})` : label);
		const names = [['section', 'Section'], ['probes', counted('Probes', result ? probes.count() : 0)], ...panels.map((p) => [p.name, counted(p.label, result ? p.count?.() : 0)])];
		tabs.replaceChildren(...names.map(([name, label]) =>
			h('button', { type: 'button', role: 'tab', class: 'dock-tab', 'data-tab': name, disabled: !result, 'aria-selected': String(result && name === view.dock), onclick: () => actions.tab(name) }, label)));
		collapse.disabled = !result;
		el.classList.toggle('collapsed', view.dockCollapsed);
		collapse.title = view.dockCollapsed ? 'Expand plots' : 'Collapse plots';
		if (!result) {
			controls.replaceChildren();
			key.replaceChildren();
			readout.replaceChildren();
			table.replaceChildren();
			svg.replaceChildren();
			return;
		}
		const wanted = view.dock === 'section' ? section.controls(c) : view.dock === 'probes' ? probes.actions(c) : panelShown().controls;
		// The section controls are one element updated in place; detaching it would end a drag on its slider.
		if (!wanted) controls.replaceChildren();
		else if (controls.firstChild !== wanted || controls.childNodes.length !== 1) controls.replaceChildren(wanted);
		key.replaceChildren(...(sectionHost() ? section.key() : []));
		drawPlotNow();
	}

	return { el, render };
}
