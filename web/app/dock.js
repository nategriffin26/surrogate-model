// The plot dock under the 3D view: the section plot, pressure drag accumulated from nose to tail, and the car's
// cross-section area against the training cars' envelope. Each plot can be saved as CSV.

import { download, h } from './dom.js';
import { signed } from './format.js';
import { icon } from './icons.js';
import { drawCursor, drawPlot, extent, valueAlong } from './plots.js';
import { sectionCsv } from './cut.js';
import { baseName } from './export.js';

const DOCK_TABS = { section: 'Section', buildup: 'Drag build-up', area: 'Section area' };

function buildupSeries(r) {
	const x0 = r.geometry.bounds[0];
	const series = [{ kind: 'line', cls: 'b0', points: [[x0, 0], ...r.dragBuildup.map((p) => [p.x, p.cdP])] }];
	if (r.source.hasCfd) series.unshift({ kind: 'dashed', cls: 'b0 cfd', points: [[x0, 0], ...r.dragBuildup.map((p) => [p.x, p.cdPCfd])] });
	return series;
}

function areaSeries(r) {
	const pts = r.coverage.sectionArea;
	const outside = [];
	let run = [];
	for (const p of pts) {
		if (p.area > p.trainMax || p.area < p.trainMin) run.push([p.x, p.area]);
		else if (run.length) {
			outside.push(run);
			run = [];
		}
	}
	if (run.length) outside.push(run);
	return [
		{ kind: 'band', cls: 'train', points: pts.map((p) => [p.x, p.trainMin, p.trainMax]) },
		{ kind: 'line', cls: 'area', points: pts.map((p) => [p.x, p.area]) },
		...outside.map((points) => ({ kind: 'line', cls: 'area outside', points: points.length > 1 ? points : [points[0], [points[0][0] + 0.02, points[0][1]]] })),
	];
}

export function plotCsv(c) {
	const r = c.result;
	switch (c.view.dock) {
		case 'section':
			return sectionCsv(c);
		case 'buildup':
			return ['x_m,cd_p' + (r.source.hasCfd ? ',cd_p_cfd' : ''), ...r.dragBuildup.map((p) => [p.x, p.cdP, ...(r.source.hasCfd ? [p.cdPCfd] : [])].join(','))].join('\n') + '\n';
		case 'area':
			return ['x_m,area_m2,train_min_m2,train_max_m2', ...r.coverage.sectionArea.map((p) => [p.x, p.area, p.trainMin, p.trainMax].join(','))].join('\n') + '\n';
		default:
			throw new Error(`Unknown plot ${c.view.dock}`);
	}
}

/** actions: tab(name), collapse(), changed() — the section tool supplies the section plot and its controls. */
export function createDock({ section, active, actions }) {
	const tabs = h('div', { class: 'dock-tabs', role: 'tablist' }, Object.entries(DOCK_TABS).map(([name, label]) =>
		h('button', { type: 'button', role: 'tab', class: 'dock-tab', 'data-tab': name, onclick: () => actions.tab(name) }, label)));
	const controls = h('div', { class: 'dock-controls' });
	const key = h('div', { class: 'plot-key' });
	const save = h('button', { type: 'button', class: 'icon-button dock-save', title: 'Save plot as CSV', onclick: () => {
		const c = active();
		download(new Blob([plotCsv(c)], { type: 'text/csv' }), `${baseName(c)}-${c.view.dock}.csv`);
	} }, icon('download'));
	const collapse = h('button', { type: 'button', class: 'icon-button dock-collapse', title: 'Collapse plots', onclick: () => actions.collapse() }, icon('chevron'));
	const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
	svg.setAttribute('class', 'plot');
	const readout = h('div', { class: 'plot-readout' });
	const empty = h('div', { class: 'plot-empty' });
	const body = h('div', { class: 'dock-body' }, svg, key, readout, empty);
	const el = h('div', { class: 'dock' }, h('div', { class: 'dock-head' }, tabs, controls, save, collapse), body);
	new ResizeObserver(() => drawPlotNow()).observe(body);

	function simplePlot(series, x, y, readoutFor) {
		const frame = drawPlot(svg, { width: body.clientWidth, height: body.clientHeight, x, y: { ...y, domain: y.domain ?? extent(series) }, series });
		svg.onpointermove = (e) => {
			const at = frame.x.invert(e.clientX - svg.getBoundingClientRect().left);
			const values = readoutFor(at);
			if (!values.length) {
				drawCursor(frame, null);
				readout.replaceChildren();
				return;
			}
			drawCursor(frame, at, values.filter((v) => v.cls).map((v) => ({ value: v.value, cls: v.cls })));
			readout.replaceChildren(h('span', {}, `x ${signed(at, 3)} m`), ...values.map((v) => h('span', { class: v.cls ? `readout-${v.cls.split(' ')[0]}` : '' }, `${v.label} `, h('b', {}, v.text))));
		};
		svg.onpointerleave = () => {
			drawCursor(frame, null);
			readout.replaceChildren();
		};
	}

	function drawPlotNow() {
		const c = active();
		if (!c?.result || el.hidden || body.clientHeight < 40) return;
		readout.replaceChildren();
		empty.hidden = true;
		svg.style.display = '';
		const r = c.result;
		switch (c.view.dock) {
			case 'section':
				if (!c.view.section.on) {
					section.forget();
					svg.style.display = 'none';
					empty.hidden = false;
					empty.textContent = 'Section off';
					return;
				}
				section.drawSection(svg, readout, body.clientWidth, body.clientHeight);
				return;
			case 'buildup': {
				section.forget();
				const series = buildupSeries(r);
				const [, hi] = extent(series);
				simplePlot(series, { domain: [r.geometry.bounds[0] - 0.05, r.geometry.bounds[1] + 0.05], label: 'x, m' }, { label: 'C_D,p', domain: [0, hi] }, (at) => {
					const pts = [[r.geometry.bounds[0], 0, 0], ...r.dragBuildup.map((p) => [p.x, p.cdP, p.cdPCfd])];
					const pred = valueAlong(pts, at, 1);
					if (pred === null) return [];
					const out = [{ label: 'Predicted', value: pred, text: pred.toFixed(3), cls: 'b0' }];
					if (r.source.hasCfd) {
						const cfd = valueAlong(pts, at, 2);
						out.push({ label: 'CFD', value: cfd, text: cfd.toFixed(3), cls: 'b0 cfd' });
					}
					return out;
				});
				return;
			}
			case 'area': {
				section.forget();
				const series = areaSeries(r);
				const [, hi] = extent(series);
				simplePlot(series, { domain: [Math.min(r.geometry.bounds[0], r.coverage.sectionArea[0].x) - 0.05, r.geometry.bounds[1] + 0.05], label: 'x, m' }, { label: 'S, m²', domain: [0, hi] }, (at) => {
					const pts = r.coverage.sectionArea.map((p) => [p.x, p.area, p.trainMin, p.trainMax]);
					const area = valueAlong(pts, at, 1);
					if (area === null) return [];
					return [
						{ label: 'This car', value: area, text: `${area.toFixed(3)} m²`, cls: 'area' },
						{ label: 'Training', text: `${valueAlong(pts, at, 2).toFixed(3)} to ${valueAlong(pts, at, 3).toFixed(3)} m²` },
					];
				});
				return;
			}
			default:
				throw new Error(`Unknown plot ${c.view.dock}`);
		}
	}

	function keyFor(c) {
		switch (c.view.dock) {
			case 'section':
				return c.view.section.on ? section.key(c) : [];
			case 'buildup':
				return [h('span', { class: 'key-item b0' }, h('i'), 'Predicted'), c.result.source.hasCfd ? h('span', { class: 'key-item cfd' }, h('i'), 'CFD') : null];
			case 'area':
				return [h('span', { class: 'key-item area' }, h('i'), 'This car'), h('span', { class: 'key-item train' }, h('i'), 'Training cars'), h('span', { class: 'key-item outside' }, h('i'), 'Outside')];
			default:
				throw new Error(`Unknown plot ${c.view.dock}`);
		}
	}

	/** Redraw for the active case: tabs, controls, key and plot. A case still being analysed has no plots yet. */
	function render() {
		const c = active();
		const result = Boolean(c?.result);
		for (const tab of tabs.children) {
			tab.disabled = !result;
			tab.setAttribute('aria-selected', String(result && tab.dataset.tab === c.view.dock));
		}
		save.disabled = !result || (c.view.dock === 'section' && !c.view.section.on);
		collapse.disabled = !result;
		if (!result) {
			controls.replaceChildren();
			key.replaceChildren();
			readout.replaceChildren();
			empty.textContent = '';
			svg.replaceChildren();
			return;
		}
		controls.replaceChildren(...(c.view.dock === 'section' && c.view.section.on ? [section.controls(c)] : []));
		key.replaceChildren(...keyFor(c).filter(Boolean));
		el.classList.toggle('collapsed', c.view.dockCollapsed);
		collapse.title = c.view.dockCollapsed ? 'Expand plots' : 'Collapse plots';
		drawPlotNow();
	}

	return { el, render, redraw: drawPlotNow };
}
