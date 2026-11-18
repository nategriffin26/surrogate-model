// The tool's shell: open cases in tabs, the 3D view with its toolbar, legend and result card, and the dock of plots
// and probes under it. Every case is seen through one view (camera, field, colour maps, section and dock tab), so
// flipping tabs compares cars from the same angle on the same colour scale; each case keeps only its own probes.

import { createEngine } from '../engine/engine.js';
import { createTriad } from '../viewer/triad.js';
import { VIEWS, Viewer } from '../viewer/viewer.js';
import { createSectionTool } from './cut.js';
import { createDock, dockCsv } from './dock.js';
import { download, h } from './dom.js';
import { baseName, savePicture } from './export.js';
import { FIELDS, RESULT_FIELDS, defaultMaps } from './fields.js';
import { icon } from './icons.js';
import { bandCount, createLegend } from './legend.js';
import { menu } from './menu.js';
import { createProbes } from './probes.js';
import { createProgress } from './progress.js';
import { createResultCard } from './results.js';
import { Session } from './session.js';
import { acceptDrops, startScreen } from './start.js';
import { VIEW_KEYS, createToolbar } from './toolbar.js';

const $ = (selector) => document.querySelector(selector);
const body = document.body;
const viewport = $('[data-viewport]');
const overlay = $('[data-overlay]');
const center = $('[data-center]');
const strip = $('[data-strip]');
const card = $('[data-card]');
const paneLabels = $('[data-pane-labels]');
const modelHost = $('[data-model]');

const engine = createEngine();
const session = new Session(engine);
const viewer = new Viewer(viewport);
viewport.append(createTriad(viewer));
const display = { edges: false, ground: true };
let mode = 'result';

const active = () => session.active;
const ready = () => session.active?.status === 'ready';

// ---- the shared view ----------------------------------------------------------------------------------------------------

/** The colour maps start from the first car with a result and then stay put, so every car reads on one scale. */
const view = {
	field: 'cp',
	maps: null,
	section: { on: true, axis: 1, offset: 0, clip: 0 },
	dock: 'section',
	dockCollapsed: false,
};
/** The camera frames the first car; after that it stays where the engineer put it, whichever car is shown. */
let framed = false;

/**
 * While the Model view is on screen it may choose what the car shows: panes [{ field, label }], or null to show the
 * shared view. The shared view is left untouched, so Result comes back as it was.
 */
let modelPanes = null;
const modelOwnsCar = () => mode === 'model' && modelPanes !== null;

function mapFor(field) {
	const name = FIELDS[field].map;
	if (!name || !view.maps) return null;
	view.maps[name] ??= defaultMaps(active().result.mesh)[name];
	const map = view.maps[name];
	return { ...map, bands: bandCount(map) };
}

/** The panes the car is drawn in: the Model view's while it owns the car, otherwise the shared field. */
function panes() {
	if (modelOwnsCar()) return modelPanes.map((pane) => ({ ...pane, map: mapFor(pane.field) }));
	const field = active()?.result ? view.field : 'none';
	return [{ field, map: mapFor(field) }];
}

/** The field the legend shows and edits: the first pane drawn in colour. */
const legendField = () => panes().find((pane) => FIELDS[pane.field].map)?.field ?? 'none';

// ---- components ------------------------------------------------------------------------------------------------------

const progress = createProgress({ cancel: (c) => session.close(c) });

const toolbar = createToolbar({
	field: (field) => {
		view.field = field;
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

/** Turning the section on brings its plot up; turning it off leaves the probe table in its place. */
function showSection(on) {
	view.section.on = on;
	if (on) {
		view.dock = 'section';
		view.dockCollapsed = false;
	} else if (view.dock === 'section') {
		view.dock = 'probes';
	}
}

function toggle(what) {
	switch (what) {
		case 'edges':
		case 'ground':
			display[what] = !display[what];
			viewer.setDisplay(display);
			break;
		case 'section':
			if (!active()?.result || modelOwnsCar()) return;
			showSection(!view.section.on);
			break;
		default:
			throw new Error(`Unknown toggle ${what}`);
	}
	update('toolbar', 'viewer', 'dock', 'layout');
}

const legend = createLegend({
	change: (patch) => {
		Object.assign(view.maps[FIELDS[legendField()].map], patch);
		update('legend', 'toolbar', 'viewer');
	},
	reset: () => {
		const name = FIELDS[legendField()].map;
		view.maps[name] = defaultMaps(active().result.mesh)[name];
		update('legend', 'toolbar', 'viewer');
	},
});
$('[data-legend]').append(legend.el);

const probes = createProbes({ host: overlay, engine, viewer, active, view, changed: () => update('dock', 'layout', 'markers') });

const section = createSectionTool({
	host: overlay,
	viewport,
	viewer,
	active,
	view,
	changed: (patch) => {
		Object.assign(view.section, patch);
		update('viewer', 'dock');
	},
});

const dock = createDock({
	section,
	probes,
	active,
	view,
	actions: {
		tab: (name) => {
			if (name === 'section') showSection(true);
			view.dock = name;
			view.dockCollapsed = false;
			update('toolbar', 'viewer', 'dock', 'layout');
		},
		collapse: () => {
			view.dockCollapsed = !view.dockCollapsed;
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
		const showPanes = (panes) => {
			modelPanes = panes;
			update('toolbar', 'legend', 'viewer');
		};
		modelMounted = import('./model/index.js').then(({ mountModelView }) => mountModelView(modelHost, { engine, session, viewer, showPanes, corner: $('[data-model-corner]') }));
	}
	update('layout', 'toolbar', 'legend', 'viewer');
}

/** A CSV file of one of the dock's tabs, shown or not. */
function saveCsv(tab) {
	exportMenu.close();
	const c = active();
	download(new Blob([dockCsv(c, { ...view, dock: tab })], { type: 'text/csv' }), `${baseName(c)}-${tab}.csv`);
}

/** The dropped file with the predicted Cp and its spread added as cell arrays, for ParaView and the like. */
async function saveSurface() {
	const item = exportItems.surface;
	const progress = item.querySelector('kbd');
	item.disabled = true;
	exportNote.hidden = true;
	try {
		const file = await engine.exportVtp(active().result, (fraction) => (progress.textContent = `${Math.round(fraction * 100)} %`));
		download(file, file.name);
		exportMenu.close();
	} catch (error) {
		exportNote.textContent = error.message;
		exportNote.hidden = false;
	} finally {
		progress.textContent = 'VTP';
		item.disabled = false;
		refreshExport();
	}
}

const exportItems = {
	picture: h('button', { type: 'button', class: 'menu-item', onclick: () => {
		exportMenu.close();
		savePicture(active(), viewer, { field: view.field, map: view.maps[FIELDS[view.field].map] });
	} }, h('span', {}, 'Picture'), h('kbd', {}, 'PNG')),
	surface: h('button', { type: 'button', class: 'menu-item', title: 'The analysed file with CpPredicted and CpSpread cell arrays added', onclick: saveSurface }, h('span', {}, 'Surface'), h('kbd', {}, 'VTP')),
	section: h('button', { type: 'button', class: 'menu-item', onclick: () => saveCsv('section') }, h('span', {}, 'Section'), h('kbd', {}, 'CSV')),
	probes: h('button', { type: 'button', class: 'menu-item', onclick: () => saveCsv('probes') }, h('span', {}, 'Probes'), h('kbd', {}, 'CSV')),
};
const exportNote = h('p', { class: 'menu-note', role: 'alert', hidden: true });
const exportButton = h('button', { type: 'button', class: 'button export-button' }, icon('download'), 'Export');
const exportMenu = menu(exportButton, h('div', { class: 'menu-list' }, Object.values(exportItems), exportNote), { align: 'right' });
$('[data-export]').append(exportMenu.el);

/**
 * What can be exported right now. The surface file exists only while the engine holds the car's whole predicted
 * surface: for a dropped file, from its exact numbers until the next analysis starts, and never on a device too
 * slow to predict every polygon. Rather than a control that cannot be used, it is offered only then.
 */
function refreshExport() {
	const c = active();
	exportItems.surface.hidden = !engine.canExportVtp(c.result);
	exportItems.section.disabled = !view.section.on;
	exportItems.probes.disabled = c.probes.length === 0;
}
exportButton.addEventListener('click', () => {
	exportNote.hidden = true;
	refreshExport();
});

// ---- result card ----------------------------------------------------------------------------------------------------------

const resultCard = createResultCard({ host: card, overlay, viewer, active });

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
			if (carShown(c)) {
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
	body.dataset.screen = !c ? 'start' : c.status === 'failed' || c.status === 'loading' ? 'notice' : 'case';
	if (!ready() && mode === 'model') mode = 'result';
	body.dataset.mode = mode;
	for (const b of modes.children) b.setAttribute('aria-selected', String(b.dataset.mode === mode));
	modes.hidden = !ready();
	$('[data-export]').hidden = !ready();
	modelHost.hidden = mode !== 'model';
	viewer.hoverPicking = ready();
	body.classList.toggle('dock-closed', view.dockCollapsed);
}

/** While the Model view owns the car, the field picker and the section wait until Result is back. */
function renderToolbar() {
	const c = active();
	const enabled = Boolean(c?.result) && !modelOwnsCar();
	toolbar.render({
		fields: RESULT_FIELDS,
		field: view.field,
		maps: view.maps,
		enabled,
		projection: viewer.projection,
		display,
		section: enabled && view.section.on,
	});
}

function renderLegend() {
	const field = legendField();
	const mapName = FIELDS[field].map;
	const title = panes().length > 1 ? FIELDS[field].symbol : FIELDS[field].title;
	legend.render({ field, title, map: mapName ? view.maps[mapName] : null, mapName });
}

function applyViewer() {
	const c = active();
	const shown = panes();
	viewer.setPanes(shown);
	paneLabels.replaceChildren(...(shown.length > 1 ? shown.map((pane) => h('span', { class: 'pane-label' }, pane.label)) : []));
	const s = view.section;
	viewer.setSection(c?.result && s.on && !modelOwnsCar() ? { axis: s.axis, offset: s.offset, clip: s.clip, plane: Boolean(viewer.section?.plane) } : null);
	viewer.setDisplay(display);
}

const RENDER = {
	layout: renderLayout,
	tabs: renderTabs,
	toolbar: renderToolbar,
	legend: renderLegend,
	viewer: applyViewer,
	dock: () => dock.render(),
	card: () => resultCard.render(),
	center: renderCenter,
	markers: () => probes.place(),
};

function update(...parts) {
	for (const part of parts) RENDER[part]();
}

/** Whether a case's car is on screen: its surface is the one shown and its mesh has reached the GPU. */
function carShown(c) {
	return Boolean(c?.surface) && viewer.surface === c.surface && viewer.placed !== null;
}

/** The analysing backdrop moves until the car is on screen, which takes it over. */
function placeBackdrop(c) {
	viewer.setBackdrop(!c ? 'still' : carShown(c) ? 'none' : c.status === 'failed' ? 'still' : 'moving');
	if (carShown(c) || !c) viewer.setSweep(null);
}

/** Put a case on screen, or bring the screen up to date with it: its surface, its result and everything drawn from it. */
function present(c) {
	const arriving = Boolean(c?.result && !c.arrived);
	if (arriving) {
		c.arrived = true;
		// Every car opens on its centreline section.
		view.section = { on: true, axis: 1, offset: 0, clip: 0 };
		view.dock = 'section';
		view.dockCollapsed = false;
	}
	if (c?.result && !view.maps) view.maps = defaultMaps(c.result.mesh);
	const surface = c?.surface ?? null;
	if (viewer.surface !== surface) {
		viewer.show(surface);
		if (surface && !framed) {
			viewer.view('isometric', { animate: false });
			framed = true;
		}
	} else if (surface) {
		viewer.surfaceChanged();
	}
	placeBackdrop(c);
	update('layout', 'tabs', 'toolbar', 'legend', 'viewer', 'center', 'card', 'dock', 'markers');
	if (arriving) viewer.arrive();
}

session.addEventListener('change', ({ detail: { case: c, what } }) => {
	switch (what) {
		case 'cases':
			update('tabs');
			return;
		case 'active':
			present(c);
			return;
		case 'status':
			update('tabs');
			if (c === active()) present(c);
			return;
		case 'stage':
			if (c !== active()) return;
			if (!carShown(c)) viewer.setSweep(progress.fraction(c));
			update('center');
			return;
		case 'geometry':
			if (c === active()) present(c);
			return;
		case 'full':
			if (c === active()) viewer.surfaceChanged();
			return;
		case 'numbers':
			if (c !== active()) return;
			update('card', 'dock');
			// The surface export becomes available with the exact numbers, perhaps while the menu is open.
			if (ready()) refreshExport();
			return;
		case 'device':
			if (active()?.status === 'analysing' || active()?.status === 'queued') update('center');
			return;
		default:
			throw new Error(`Unknown change ${what}`);
	}
});

// A car's mesh reaches the GPU some frames after it arrives; the analysing screen hands over to it then.
viewer.addEventListener('placed', () => {
	placeBackdrop(active());
	update('center');
});

// Stage clocks tick while the active case is being analysed.
setInterval(() => {
	if (active()?.status === 'analysing') update('center');
}, 200);

viewer.addEventListener('hover', ({ detail: { hit, clientX, clientY } }) => {
	if (mode !== 'result' || !ready()) return;
	probes.hover(hit, clientX, clientY);
	section.followCar(hit);
});
viewer.addEventListener('tap', ({ detail: { hit, clientX, clientY } }) => {
	if (mode !== 'result' || !ready()) return;
	probes.pin(hit, clientX, clientY);
});

// ---- dock height ----------------------------------------------------------------------------------------------------------

const dockResizer = $('[data-dock-resizer]');
dockResizer.addEventListener('pointerdown', (e) => {
	dockResizer.setPointerCapture(e.pointerId);
	body.classList.add('resizing');
	const move = (ev) => {
		const bottom = $('[data-dock]').getBoundingClientRect().bottom;
		body.style.setProperty('--dock', `${Math.min(innerHeight * 0.6, Math.max(150, bottom - ev.clientY))}px`);
	};
	const up = () => {
		body.classList.remove('resizing');
		dockResizer.removeEventListener('pointermove', move);
		dockResizer.removeEventListener('pointerup', up);
	};
	dockResizer.addEventListener('pointermove', move);
	dockResizer.addEventListener('pointerup', up);
});

// ---- keys -------------------------------------------------------------------------------------------------------------------

const VIEW_BY_KEY = Object.fromEntries(Object.entries(VIEW_KEYS).map(([name, key]) => [key, name]));
addEventListener('keydown', (e) => {
	if (e.metaKey || e.ctrlKey || e.altKey || e.target.closest('input, textarea, select')) return;
	if (!active()?.surface) return;
	const key = e.key.toLowerCase();
	if (VIEW_BY_KEY[key] && VIEWS[VIEW_BY_KEY[key]]) viewer.view(VIEW_BY_KEY[key]);
	else if (key === 'f') viewer.fit();
	else if (key === 'p' || key === 'o') {
		viewer.setProjection(key === 'p' ? 'perspective' : 'orthographic');
		update('toolbar');
	} else if (key === 'e') toggle('edges');
	else if (key === 'g') toggle('ground');
	else if (key === 's') toggle('section');
});

acceptDrops((file) => session.openFile(file));
present(null);

export const app = { engine, session, viewer, view, update };
