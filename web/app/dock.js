// The dock under the 3D view: the section plot, the probe table, and panels such as the regions tool. The section
// plot can be saved as CSV; the probe table has its own copy and save.

import { h } from './dom.js';
import { icon } from './icons.js';

/**
 * view: the shared view (its dock tab and section). actions: tab(name), collapse(), save(). The section tool
 * supplies the section plot and its controls, the probes their table and its actions. Each of `panels` is a tab of
 * its own, {name, label, el, controls, render(), count?()}, its count shown in the tab when above zero.
 */
export function createDock({ section, probes, panels, active, view, actions }) {
	const tabs = h('div', { class: 'dock-tabs', role: 'tablist' });
	const controls = h('div', { class: 'dock-controls' });
	const key = h('div', { class: 'plot-key' });
	const save = h('button', { type: 'button', class: 'icon-button dock-save', title: 'Save plot as CSV', onclick: () => actions.save() }, icon('download'));
	const collapse = h('button', { type: 'button', class: 'icon-button dock-collapse', title: 'Collapse plots', onclick: () => actions.collapse() }, icon('chevron'));
	const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
	svg.setAttribute('class', 'plot');
	const readout = h('div', { class: 'plot-readout' });
	const table = h('div', { class: 'dock-table' });
	// The key and the cursor readout sit together at the plot's top left, next to the axis they describe.
	const body = h('div', { class: 'dock-body' }, svg, h('div', { class: 'plot-head' }, key, readout), table);
	const el = h('div', { class: 'dock' }, h('div', { class: 'dock-head' }, tabs, controls, save, collapse), body);
	new ResizeObserver(() => drawPlotNow()).observe(body);

	function drawPlotNow() {
		const c = active();
		if (!c?.result || body.clientHeight < 40) return;
		readout.replaceChildren();
		const plotted = view.dock === 'section';
		svg.style.display = plotted ? '' : 'none';
		table.hidden = plotted;
		switch (view.dock) {
			case 'section':
				section.drawSection(svg, readout, body.clientWidth, body.clientHeight);
				return;
			case 'probes':
				section.forget();
				table.replaceChildren(probes.table(c));
				return;
			default: {
				const panel = panels.find((p) => p.name === view.dock);
				if (!panel) throw new Error(`Unknown dock tab ${view.dock}`);
				section.forget();
				// In the page first, so a panel that draws to its own size can measure it.
				if (table.firstChild !== panel.el) table.replaceChildren(panel.el);
				panel.render();
			}
		}
	}

	/** Redraw for the active case: tabs, controls, key and plot. A case still being analysed has no plots yet. */
	function render() {
		const c = active();
		const result = Boolean(c?.result);
		const counted = (label, count) => (count ? `${label} (${count})` : label);
		const names = [['section', 'Section'], ['probes', counted('Probes', result ? probes.count() : 0)], ...panels.map((p) => [p.name, counted(p.label, result ? p.count?.() : 0)])];
		tabs.replaceChildren(...names.map(([name, label]) =>
			h('button', { type: 'button', role: 'tab', class: 'dock-tab', 'data-tab': name, disabled: !result, 'aria-selected': String(result && name === view.dock), onclick: () => actions.tab(name) }, label)));
		save.hidden = view.dock !== 'section';
		save.disabled = !result;
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
		const wanted = [view.dock === 'section' ? section.controls(c) : view.dock === 'probes' ? probes.actions(c) : panels.find((p) => p.name === view.dock).controls];
		// The section controls are one element updated in place; detaching it would end a drag on its slider.
		if (wanted.length !== controls.childNodes.length || wanted.some((node, i) => node !== controls.childNodes[i])) controls.replaceChildren(...wanted);
		key.replaceChildren(...(view.dock === 'section' ? section.key() : []));
		drawPlotNow();
	}

	return { el, render };
}
