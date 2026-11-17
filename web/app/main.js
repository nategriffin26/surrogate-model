// The tool's shell: open cases in tabs, the 3D view with its toolbar and legend, the plot dock and the results
// panel, all kept in step with the active case. Each case keeps its own view: camera, field, colour maps, section,
// probes and plot.

import { createEngine } from '../engine/engine.js';
import { createTriad } from '../viewer/triad.js';
import { VIEWS, Viewer } from '../viewer/viewer.js';
import { createSectionTool } from './cut.js';
import { createDock, plotCsv } from './dock.js';
import { download, h } from './dom.js';
import { baseName, savePicture } from './export.js';
import { FIELDS, defaultMaps, fieldsFor } from './fields.js';
import { icon } from './icons.js';
import { bandCount, createLegend } from './legend.js';
import { menu } from './menu.js';
import { createProbes, probesCsv } from './probes.js';
import { createProgress } from './progress.js';
import { drawnText, renderResults } from './results.js';
import { Session } from './session.js';
import { acceptDrops, startScreen } from './start.js';
import { VIEW_KEYS, createToolbar } from './toolbar.js';

const $ = (selector) => document.querySelector(selector);
const body = document.body;
const viewport = $('[data-viewport]');
const overlay = $('[data-overlay]');
const center = $('[data-center]');
const strip = $('[data-strip]');
const paneLabels = $('[data-pane-labels]');
const panel = $('[data-panel]');
const panelBody = $('[data-panel-body]');
const modelHost = $('[data-model]');

const engine = createEngine();
const session = new Session(engine);
const viewer = new Viewer(viewport);
viewport.append(createTriad(viewer));
const display = { edges: false, ground: true };
let mode = 'result';
let drawn = null;

const active = () => session.active;
const ready = () => session.active?.status === 'ready';

// ---- per-case view state -------------------------------------------------------------------------------------------

function newView() {
	return {
		camera: null,
		field: 'none',
		maps: null,
		defaults: null,
		compare: false,
		section: { on: false, axis: 1, offset: 0, clip: 0 },
		probes: [],
		nextProbe: 1,
		dock: 'buildup',
		dockCollapsed: false,
		panel: 'results',
		arrived: false,
	};
}

/** Once fields exist: starting colour maps and the predicted pressure. */
function fillView(c) {
	if (c.view.maps) return;
	c.view.defaults = defaultMaps(c.result.mesh);
	c.view.maps = structuredClone(c.view.defaults);
	c.view.field = 'cp';
}

function mapFor(c, field) {
	const name = FIELDS[field].map;
	if (!name || !c.view.maps) return null;
	const map = c.view.maps[name];
	return { ...map, bands: bandCount(map) };
}

function panesFor(c) {
	if (!c?.view) return [{ field: 'none', map: null }];
	if (c.view.compare) return [{ field: 'cp', label: 'Predicted', map: mapFor(c, 'cp') }, { field: 'cfd', label: 'CFD', map: mapFor(c, 'cfd') }];
	return [{ field: c.view.field, map: mapFor(c, c.view.field) }];
}

// ---- components ------------------------------------------------------------------------------------------------------

const progress = createProgress({ cancel: (c) => session.close(c) });

const toolbar = createToolbar({
	field: (field) => {
		const c = active();
		c.view.field = field;
		c.view.compare = false;
		update('toolbar', 'legend', 'viewer', 'dock');
	},
	view: (name) => viewer.view(name),
	fit: () => viewer.fit(),
	projection: (p) => {
		viewer.setProjection(p);
		update('toolbar');
	},
	toggle: (what) => toggle(what),
});
$('[data-toolbar]').append(toolbar.el);
const panelToggle = h('button', { type: 'button', class: 'tool tool-quiet panel-toggle', title: 'Results panel', onclick: () => {
	body.classList.toggle('panel-closed');
	update('toolbar');
} }, icon('panel'));
toolbar.el.querySelector('.toolbar-end').append(panelToggle);

function toggle(what) {
	const c = active();
	switch (what) {
		case 'edges':
		case 'ground':
			display[what] = !display[what];
			viewer.setDisplay(display);
			break;
		case 'section':
			if (!c?.result) return;
			c.view.section.on = !c.view.section.on;
			if (c.view.section.on) {
				c.view.dock = 'section';
				c.view.dockCollapsed = false;
			}
			break;
		case 'compare':
			if (!c?.result?.source.hasCfd) return;
			c.view.compare = !c.view.compare;
			break;
		default:
			throw new Error(`Unknown toggle ${what}`);
	}
	update('toolbar', 'legend', 'viewer', 'dock', 'layout');
}

const legend = createLegend({
	change: (patch) => {
		const c = active();
		const name = FIELDS[c.view.compare ? 'cp' : c.view.field].map;
		Object.assign(c.view.maps[name], patch);
		update('legend', 'toolbar', 'viewer');
	},
	reset: () => {
		const c = active();
		const name = FIELDS[c.view.compare ? 'cp' : c.view.field].map;
		c.view.maps[name] = structuredClone(c.view.defaults[name]);
		update('legend', 'toolbar', 'viewer');
	},
});
$('[data-legend]').append(legend.el);

const probes = createProbes({ host: overlay, viewer, active, changed: () => update('panel', 'markers') });

const section = createSectionTool({
	host: overlay,
	viewport,
	viewer,
	active,
	changed: (patch) => {
		const c = active();
		Object.assign(c.view.section, patch);
		update('viewer', 'dock');
	},
});

const dock = createDock({
	section,
	active,
	actions: {
		tab: (name) => {
			const c = active();
			c.view.dock = name;
			c.view.dockCollapsed = false;
			update('dock', 'layout');
		},
		collapse: () => {
			const c = active();
			c.view.dockCollapsed = !c.view.dockCollapsed;
			update('dock', 'layout');
		},
	},
});
$('[data-dock]').append(dock.el);

// ---- top bar: case tabs, modes, export -------------------------------------------------------------------------------

const tabs = $('[data-cases]');
function renderTabs() {
	tabs.replaceChildren(
		...session.cases.map((c) => h('div', { class: 'case-tab', role: 'tab', 'aria-selected': String(c === active()), 'data-status': c.status, title: c.name },
			h('button', { type: 'button', class: 'case-open', onclick: () => session.activate(c) },
				h('span', { class: 'case-state', 'aria-hidden': 'true' }),
				h('span', { class: 'case-name' }, c.name)),
			h('button', { type: 'button', class: 'case-close', title: c.status === 'analysing' || c.status === 'queued' ? `Cancel and close ${c.name}` : `Close ${c.name}`, onclick: () => session.close(c) }, icon('close')))),
		h('button', { type: 'button', class: 'case-add', title: 'Open a file or an example', 'aria-pressed': String(active() === null), onclick: () => session.activate(null) }, icon('plus'), h('span', {}, 'Open')));
}

const modes = $('[data-modes]');
modes.append(...['result', 'model'].map((m) => h('button', { type: 'button', role: 'tab', class: 'mode', 'data-mode': m, onclick: () => setMode(m) }, m === 'result' ? 'Result' : 'Model')));

let modelMounted = null;
function setMode(next) {
	mode = next;
	if (mode === 'model' && !modelMounted) {
		modelMounted = import('./model/index.js').then(({ mountModelView }) => mountModelView(modelHost, { engine, session, viewer }));
	}
	update('layout');
}

const exportItems = {
	picture: h('button', { type: 'button', class: 'menu-item', onclick: () => {
		exportMenu.close();
		savePicture(active(), viewer, panesFor(active()).map((pane) => ({ ...pane, map: active().view.maps[FIELDS[pane.field].map] ?? null })));
	} }, h('span', {}, 'Picture'), h('kbd', {}, 'PNG')),
	plot: h('button', { type: 'button', class: 'menu-item', onclick: () => {
		exportMenu.close();
		const c = active();
		download(new Blob([plotCsv(c)], { type: 'text/csv' }), `${baseName(c)}-${c.view.dock}.csv`);
	} }, h('span', {}, 'Plot'), h('kbd', {}, 'CSV')),
	probes: h('button', { type: 'button', class: 'menu-item', onclick: () => {
		exportMenu.close();
		const c = active();
		download(new Blob([probesCsv(c)], { type: 'text/csv' }), `${baseName(c)}-probes.csv`);
	} }, h('span', {}, 'Probes'), h('kbd', {}, 'CSV')),
};
const exportMenu = menu(h('button', { type: 'button', class: 'button export-button' }, icon('download'), 'Export'), h('div', { class: 'menu-list' }, Object.values(exportItems)), { align: 'right' });
$('[data-export]').append(exportMenu.el);

// ---- panel -------------------------------------------------------------------------------------------------------------

const panelTabs = $('[data-panel-tabs]');
function renderPanel() {
	const c = active();
	if (!c || c.status === 'failed' || c.status === 'loading') return;
	const tab = c.result ? c.view.panel : 'results';
	const probeCount = c.view?.probes.length ?? 0;
	panelTabs.replaceChildren(...[['results', 'Results'], ['probes', `Probes${probeCount ? ` (${probeCount})` : ''}`]].map(([name, label]) =>
		h('button', { type: 'button', role: 'tab', class: 'panel-tab', 'aria-selected': String(name === tab), disabled: name === 'probes' && !c.result, onclick: () => {
			c.view.panel = name;
			renderPanel();
		} }, label)));
	const scroll = panelBody.scrollTop;
	panelBody.replaceChildren(tab === 'probes' ? probes.table(c) : renderResults(c, { device: session.device, drawn }));
	panelBody.scrollTop = scroll;
}

function renderDrawn() {
	const cell = panelBody.querySelector('[data-fact="drawn"] .row-value');
	if (cell && drawn) cell.textContent = drawnText(drawn);
}

// ---- the middle of the view: first screen, progress, errors ------------------------------------------------------------

const start = startScreen({ onFile: (file) => session.openFile(file), onExample: (id, label) => session.openExample(id, label) });

function errorCard(c) {
	return h('div', { class: 'notice', role: 'alert' },
		h('div', { class: 'notice-title' }, c.name),
		h('p', { class: 'notice-text' }, c.error),
		h('div', { class: 'notice-actions' },
			h('button', { type: 'button', class: 'button button-primary', onclick: () => session.activate(null) }, 'Open another file'),
			h('button', { type: 'button', class: 'button', onclick: () => session.close(c) }, 'Close')));
}

function loadingCard(c) {
	return h('div', { class: 'notice', role: 'status' }, h('div', { class: 'notice-title' }, c.name), h('div', { class: 'progress-bar indeterminate' }, h('span', { class: 'progress-fill' })));
}

function renderCenter() {
	const c = active();
	if (!c) {
		if (center.firstChild !== start) center.replaceChildren(start);
		strip.replaceChildren();
		return;
	}
	switch (c.status) {
		case 'loading':
			center.replaceChildren(loadingCard(c));
			strip.replaceChildren();
			return;
		case 'queued':
		case 'analysing':
			progress.render(c, session.device);
			if (c.surface) {
				center.replaceChildren();
				if (strip.firstChild !== progress.el) strip.replaceChildren(progress.el);
			} else {
				strip.replaceChildren();
				if (center.firstChild !== progress.el) center.replaceChildren(progress.el);
			}
			return;
		case 'failed':
			center.replaceChildren(errorCard(c));
			strip.replaceChildren();
			return;
		case 'ready':
			center.replaceChildren();
			strip.replaceChildren();
			return;
		default:
			throw new Error(`Unknown status ${c.status}`);
	}
}

// ---- keeping everything in step ------------------------------------------------------------------------------------------

function renderLayout() {
	const c = active();
	const screen = !c ? 'start' : c.status === 'failed' || c.status === 'loading' ? 'notice' : 'case';
	body.dataset.screen = screen;
	if (!ready() && mode === 'model') mode = 'result';
	body.dataset.mode = mode;
	for (const b of modes.children) b.setAttribute('aria-selected', String(b.dataset.mode === mode));
	modes.hidden = !ready();
	$('[data-export]').hidden = !ready();
	modelHost.hidden = mode !== 'model';
	viewer.hoverPicking = ready();
	body.classList.toggle('dock-closed', Boolean(c?.view?.dockCollapsed));
	if (ready()) {
		exportItems.plot.disabled = c.view.dock === 'section' && !c.view.section.on;
		exportItems.probes.disabled = c.view.probes.length === 0;
	}
}

function renderToolbar() {
	const c = active();
	toolbar.render({
		fields: fieldsFor(Boolean(c?.result?.source.hasCfd)),
		field: c?.view && !c.view.compare ? c.view.field : null,
		maps: c?.view?.maps,
		enabled: Boolean(c?.result),
		projection: viewer.projection,
		display,
		section: Boolean(c?.view?.section.on),
		compare: Boolean(c?.view?.compare),
		canCompare: Boolean(c?.result?.source.hasCfd),
	});
	panelToggle.setAttribute('aria-pressed', String(!body.classList.contains('panel-closed')));
}

function renderLegend() {
	const c = active();
	const field = c?.view ? (c.view.compare ? 'cp' : c.view.field) : 'none';
	const map = c?.view?.maps && FIELDS[field].map ? c.view.maps[FIELDS[field].map] : null;
	legend.render({ field, title: c?.view?.compare ? 'Cp' : FIELDS[field].title, map, mapName: FIELDS[field].map });
}

function applyViewer() {
	const c = active();
	const panes = panesFor(c);
	viewer.setPanes(panes);
	paneLabels.replaceChildren(...(panes.length > 1 ? panes.map((pane) => h('span', { class: 'pane-label' }, pane.label)) : []));
	const s = c?.view?.section;
	viewer.setSection(c?.result && s.on ? { axis: s.axis, offset: s.offset, clip: s.clip, plane: Boolean(viewer.section?.plane) } : null);
	viewer.setDisplay(display);
}

const RENDER = {
	layout: renderLayout,
	tabs: renderTabs,
	toolbar: renderToolbar,
	legend: renderLegend,
	viewer: applyViewer,
	dock: () => dock.render(),
	panel: renderPanel,
	center: renderCenter,
	markers: () => probes.place(),
};

function update(...parts) {
	for (const part of parts) RENDER[part]();
}

/**
 * Put a case on screen, or bring the screen up to date with it: its surface, its camera (restored when switching to
 * it, framed when its car first appears) and everything drawn from its view.
 */
function present(c, { switching = false } = {}) {
	if (c?.surface && !c.view) c.view = newView();
	if (c?.result) fillView(c);
	const surface = c?.surface ?? null;
	if (viewer.surface !== surface) {
		viewer.show(surface);
		if (c?.view?.camera) viewer.restoreCamera(c.view.camera);
		else viewer.view('isometric', { animate: false });
	} else if (surface) {
		if (switching && c.view.camera) viewer.restoreCamera(c.view.camera);
		viewer.surfaceChanged();
	}
	viewer.setBackdrop(!c ? 'still' : surface ? 'none' : c.status === 'failed' ? 'still' : 'moving');
	if (surface || !c) viewer.setSweep(null);
	update('layout', 'tabs', 'toolbar', 'legend', 'viewer', 'center', 'panel', 'dock', 'markers');
	if (c?.result && !c.view.arrived) {
		c.view.arrived = true;
		viewer.arrive();
	}
}

let shown = null;
session.addEventListener('change', ({ detail: { case: c, what } }) => {
	switch (what) {
		case 'cases':
			update('tabs');
			return;
		case 'active':
			if (shown?.view) shown.view.camera = viewer.cameraState();
			shown = c;
			present(c, { switching: true });
			return;
		case 'status':
			update('tabs');
			if (c === active()) present(c);
			return;
		case 'stage':
			if (c !== active()) return;
			if (!c.surface) viewer.setSweep(progress.fraction(c));
			update('center', 'panel');
			return;
		case 'geometry':
			if (c !== active()) return;
			present(c);
			return;
		case 'full':
			if (c !== active()) return;
			viewer.surfaceChanged();
			update('panel');
			return;
		case 'numbers':
			if (c === active()) update('panel', 'dock');
			return;
		case 'device':
			if (active()?.status === 'analysing' || active()?.status === 'queued') update('center', 'panel');
			return;
		default:
			throw new Error(`Unknown change ${what}`);
	}
});

// Stage clocks tick while the active case is being analysed.
setInterval(() => {
	const c = active();
	if (c?.status === 'analysing') update('center');
}, 200);

viewer.addEventListener('hover', ({ detail: { hit, clientX, clientY } }) => {
	if (mode !== 'result' || !ready()) return;
	probes.hover(hit, clientX, clientY);
	section.followCar(hit);
});
viewer.addEventListener('tap', ({ detail: { hit } }) => {
	if (mode !== 'result' || !ready()) return;
	probes.pin(hit);
});
viewer.addEventListener('drawn', ({ detail }) => {
	drawn = detail;
	renderDrawn();
});

// ---- panel and dock sizes -------------------------------------------------------------------------------------------------

function resizer(handle, apply) {
	handle.addEventListener('pointerdown', (e) => {
		handle.setPointerCapture(e.pointerId);
		body.classList.add('resizing');
		const move = (ev) => apply(ev);
		const up = () => {
			body.classList.remove('resizing');
			handle.removeEventListener('pointermove', move);
			handle.removeEventListener('pointerup', up);
		};
		handle.addEventListener('pointermove', move);
		handle.addEventListener('pointerup', up);
	});
}
resizer($('[data-panel-resizer]'), (e) => body.style.setProperty('--panel', `${Math.min(560, Math.max(280, innerWidth - e.clientX))}px`));
resizer($('[data-dock-resizer]'), (e) => {
	const top = $('[data-dock]').getBoundingClientRect().bottom;
	body.style.setProperty('--dock', `${Math.min(innerHeight * 0.6, Math.max(150, top - e.clientY))}px`);
});

// ---- keys -------------------------------------------------------------------------------------------------------------------

const VIEW_BY_KEY = Object.fromEntries(Object.entries(VIEW_KEYS).map(([name, key]) => [key, name]));
addEventListener('keydown', (e) => {
	if (e.metaKey || e.ctrlKey || e.altKey || e.target.closest('input, textarea, select')) return;
	const c = active();
	if (!c?.surface) return;
	const key = e.key.toLowerCase();
	if (VIEW_BY_KEY[key] && VIEWS[VIEW_BY_KEY[key]]) viewer.view(VIEW_BY_KEY[key]);
	else if (key === 'f') viewer.fit();
	else if (key === 'p' || key === 'o') {
		viewer.setProjection(key === 'p' ? 'perspective' : 'orthographic');
		update('toolbar');
	} else if (key === 'e') toggle('edges');
	else if (key === 'g') toggle('ground');
	else if (key === 's') toggle('section');
	else if (key === 'c') toggle('compare');
});

acceptDrops((file) => session.openFile(file));
if (innerWidth < 1100) body.classList.add('panel-closed');
present(null);

export const app = { engine, session, viewer, update };
