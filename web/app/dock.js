// The dock under the 3D view: the section plot and the probe table. The section plot can be saved as CSV; the probe
// table has its own copy and save.

import { download, h } from './dom.js';
import { icon } from './icons.js';
import { sectionCsv } from './cut.js';
import { baseName } from './export.js';
import { probesCsv } from './probes.js';

/** The data behind the dock's current tab, as CSV. */
export function dockCsv(c, view) {
	switch (view.dock) {
		case 'section':
			return sectionCsv(c, view);
		case 'probes':
			return probesCsv(c);
		default:
			throw new Error(`Unknown dock tab ${view.dock}`);
	}
}

/**
 * view: the shared view (its dock tab and section). actions: tab(name), collapse(). The section tool supplies the
 * section plot and its controls, the probes their table and its actions.
 */
export function createDock({ section, probes, active, view, actions }) {
	const tabs = h('div', { class: 'dock-tabs', role: 'tablist' });
	const controls = h('div', { class: 'dock-controls' });
	const key = h('div', { class: 'plot-key' });
	const save = h('button', { type: 'button', class: 'icon-button dock-save', title: 'Save plot as CSV', onclick: () => {
		const c = active();
		download(new Blob([dockCsv(c, view)], { type: 'text/csv' }), `${baseName(c)}-${view.dock}.csv`);
	} }, icon('download'));
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
			default:
				throw new Error(`Unknown dock tab ${view.dock}`);
		}
	}

	/** Redraw for the active case: tabs, controls, key and plot. A case still being analysed has no plots yet. */
	function render() {
		const c = active();
		const result = Boolean(c?.result);
		const count = result ? c.probes.length : 0;
		tabs.replaceChildren(...[['section', 'Section'], ['probes', count ? `Probes (${count})` : 'Probes']].map(([name, label]) =>
			h('button', { type: 'button', role: 'tab', class: 'dock-tab', 'data-tab': name, disabled: !result, 'aria-selected': String(result && name === view.dock), onclick: () => actions.tab(name) }, label)));
		save.hidden = view.dock === 'probes';
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
		const wanted = view.dock === 'section' ? [section.controls(c)] : [probes.actions(c)];
		// The section controls are one element updated in place; detaching it would end a drag on its slider.
		if (wanted.length !== controls.childNodes.length || wanted.some((node, i) => node !== controls.childNodes[i])) controls.replaceChildren(...wanted);
		key.replaceChildren(...(view.dock === 'section' ? section.key() : []));
		drawPlotNow();
	}

	return { el, render };
}
