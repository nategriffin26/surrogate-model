// The dock under the 3D view: the section plot, pressure drag accumulated from nose to tail, and the probe table.
// The plots can be saved as CSV; the probe table has its own copy and save.

import { download, h } from './dom.js';
import { signed } from './format.js';
import { icon } from './icons.js';
import { drawCursor, drawPlot, extent, valueAlong } from './plots.js';
import { sectionCsv } from './cut.js';
import { baseName } from './export.js';
import { probesCsv } from './probes.js';

/** The data behind the dock's current tab, as CSV. */
export function dockCsv(c, view) {
	switch (view.dock) {
		case 'section':
			return sectionCsv(c, view);
		case 'buildup':
			return ['x_m,cd_p', ...c.result.dragBuildup.map((p) => [p.x, p.cdP].join(','))].join('\n') + '\n';
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
export function createDock({ section, probes, viewer, active, view, actions }) {
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
	const body = h('div', { class: 'dock-body' }, svg, key, readout, table);
	const el = h('div', { class: 'dock' }, h('div', { class: 'dock-head' }, tabs, controls, save, collapse), body);
	new ResizeObserver(() => drawPlotNow()).observe(body);
	let buildup = null;

	/** The cumulative pressure drag along the car, with a cursor that also marks its station on the car. */
	function drawBuildup(r) {
		const x0 = r.geometry.bounds[0];
		const points = [[x0, 0], ...r.dragBuildup.map((p) => [p.x, p.cdP])];
		const series = [{ kind: 'band', cls: 'b0', points: points.map(([x, y]) => [x, 0, y]) }, { kind: 'line', cls: 'b0', points }];
		const [, hi] = extent(series);
		const frame = drawPlot(svg, { width: body.clientWidth, height: body.clientHeight, x: { domain: [x0 - 0.05, r.geometry.bounds[1] + 0.05], label: 'x, m' }, y: { label: 'C_D,p', domain: [0, hi] }, series });
		buildup = { frame, points };
		svg.onpointermove = (e) => {
			const at = frame.x.invert(e.clientX - svg.getBoundingClientRect().left);
			buildupCursor(at);
			viewer.setCursorLine(valueAlong(points, at) === null ? null : { axis: 0, offset: at });
		};
		svg.onpointerleave = () => {
			buildupCursor(null);
			viewer.setCursorLine(null);
		};
	}

	function buildupCursor(at) {
		const value = at === null ? null : valueAlong(buildup.points, at);
		drawCursor(buildup.frame, value === null ? null : at, value === null ? [] : [{ value, cls: 'b0' }]);
		readout.replaceChildren(...(value === null ? [] : [h('span', {}, `x ${signed(at, 3)} m`), h('span', { class: 'readout-b0' }, 'C', h('sub', {}, 'D,p'), ' ', h('b', {}, value.toFixed(3)))]));
	}

	function drawPlotNow() {
		const c = active();
		if (!c?.result || body.clientHeight < 40) return;
		readout.replaceChildren();
		buildup = null;
		const plotted = view.dock !== 'probes';
		svg.style.display = plotted ? '' : 'none';
		table.hidden = plotted;
		switch (view.dock) {
			case 'section':
				section.drawSection(svg, readout, body.clientWidth, body.clientHeight);
				return;
			case 'buildup':
				section.forget();
				drawBuildup(c.result);
				return;
			case 'probes':
				section.forget();
				table.replaceChildren(probes.table(c));
				return;
			default:
				throw new Error(`Unknown dock tab ${view.dock}`);
		}
	}

	/** The cursor follows the car: hovering the surface moves the build-up cursor to the hovered station. */
	function followCar(hit) {
		if (view.dock === 'buildup' && buildup) buildupCursor(hit ? hit.point[0] : null);
		else section.followCar(hit);
	}

	/** Redraw for the active case: tabs, controls, key and plot. A case still being analysed has no plots yet. */
	function render() {
		const c = active();
		const result = Boolean(c?.result);
		const count = result ? c.probes.length : 0;
		tabs.replaceChildren(...[['section', 'Section'], ['buildup', 'Drag build-up'], ['probes', count ? `Probes (${count})` : 'Probes']].map(([name, label]) =>
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
		controls.replaceChildren(...(view.dock === 'section' ? [section.controls(c)] : view.dock === 'probes' ? [probes.actions(c)] : []));
		key.replaceChildren(...(view.dock === 'section' ? section.key() : []));
		drawPlotNow();
	}

	return { el, render, followCar };
}
