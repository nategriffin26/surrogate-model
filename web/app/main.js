// The tool's shell: open cases in tabs, the 3D view with its toolbar, legend and result card, the dock of plots and
// probes under it, the workspace's files in a panel on the left and the assistant in one on the right. Every case is
// seen through one view (camera, field, colour maps, section and dock tab), so flipping tabs compares cars from the
// same angle on the same colour scale; each case keeps only its own probes.
//
// Every action is an operation in the registry (operations.js): the controls here run them as the engineer, the
// assistant runs the same ones, so the two never disagree about what an action does.

import { createEngine } from '../engine/engine.js';
import { createTriad } from '../viewer/triad.js';
import { VIEWS, Viewer } from '../viewer/viewer.js';
import { createAnnotations } from './annotations.js';
import { createComparison } from './compare.js';
import { defineCaseOperations } from './cases.js';
import { createSectionTool, defineSectionOperations } from './cut.js';
import { defineDisplayOperations } from './display.js';
import { createDock } from './dock.js';
import { h } from './dom.js';
import { EXPORTS, baseName, defineExport, defineExportOperations, pictureOf } from './export.js';
import { FIELDS, RESULT_FIELDS, defaultMaps } from './fields.js';
import { addFiles, createFilesPanel, defineFileOperations } from './files.js';
import { createHistogramPanel, defineHistogramOperations } from './histogram.js';
import { icon } from './icons.js';
import { bandCount, createLegend } from './legend.js';
import { menu } from './menu.js';
import { onOperation, operations, runOperation } from './operations.js';
import { createProbes, defineProbeOperations } from './probes.js';
import { createProgress } from './progress.js';
import { createRegions, createRegionsPanel } from './regions.js';
import { defineReport } from './report.js';
import { createResultCard } from './results.js';
import { Session } from './session.js';
import { acceptDrops, startScreen } from './start.js';
import { VIEW_KEYS, createToolbar } from './toolbar.js';
import { createVariants, shortName } from './variants.js';
import { createWorkspace } from './workspace.js';

const $ = (selector) => document.querySelector(selector);
const body = document.body;
const viewport = $('[data-viewport]');
const overlay = $('[data-overlay]');
const center = $('[data-center]');
const strip = $('[data-strip]');
const card = $('[data-card]');
const paneLabels = $('[data-pane-labels]');
const modelHost = $('[data-model]');
const assistantHost = $('[data-assistant]');
const REDUCED_MOTION = matchMedia('(prefers-reduced-motion: reduce)');

const engine = createEngine();
const session = new Session(engine);
const workspace = createWorkspace();
const viewer = new Viewer(viewport);
viewport.append(createTriad(viewer));
const view = session.view;
let mode = 'result';

const active = () => session.active;
/** The active case's result is on screen: ready, and its car drawn. Until then the tools wait with it. */
const ready = () => session.active?.status === 'ready' && session.active.drawn;
/** The case the card and the plots describe: the active one once its result is on screen, so they arrive with its car. */
const described = () => (ready() ? active() : null);
/** Run an operation as the engineer: what every control here does. */
const run = (name, args) => runOperation(name, args, { session, viewer, engine, by: 'engineer' });

/** The camera frames the first car; after that it stays where the engineer put it, whichever car is shown. */
let framed = false;

function mapFor(field) {
	const name = FIELDS[field].map;
	if (!name || !view.maps) return null;
	view.maps[name] ??= defaultMaps(active().result.mesh)[name];
	const map = view.maps[name];
	return { ...map, bands: bandCount(map) };
}

/** The fields case c can be coloured by: the model's, then a comparison's and the annotations'. */
const fieldsOf = (c) => [...RESULT_FIELDS, ...comparison.fields(c), ...annotations.fields(c)];

/** The field the car shows: the view's, when the case on screen has it, else pressure. */
const shownField = () => (fieldsOf(active()).includes(view.field) ? view.field : 'cp');

/** The panes the car is drawn in: the shared field, or this car beside what it is compared with. */
function panes() {
	const field = active()?.result ? shownField() : 'none';
	const sideBySide = field === 'none' ? null : comparison.panes(active(), field);
	return (sideBySide ?? [{ field }]).map((pane) => ({ ...pane, map: mapFor(pane.field) }));
}

/** The field the legend shows and edits: the first pane drawn in colour. */
const legendField = () => panes().find((pane) => FIELDS[pane.field].map)?.field ?? 'none';

// ---- operations ---------------------------------------------------------------------------------------------------------

const tool = { session, viewer, engine, workspace, update };
defineDisplayOperations(tool);
defineCaseOperations(tool);
defineSectionOperations(tool);
defineProbeOperations(tool);
const regions = createRegions(tool);
const annotations = createAnnotations({ session, engine, viewer, view, overlay, update, run });
defineHistogramOperations({ session, regions });
const comparison = createComparison({ session, regions, update, run });
const variants = createVariants({ session, engine, workspace, regions, update, run });
defineExportOperations(tool);
const files = defineFileOperations(tool);

defineExport({
	name: 'picture',
	label: 'Picture',
	format: 'PNG',
	title: 'The view as drawn, with the colour legend',
	unavailable: () => null,
	make: (c) => pictureOf(c, viewer, { field: shownField(), map: view.maps[FIELDS[shownField()].map], marks: (ctx, scale) => annotations.paintLabels(ctx, scale) }),
	fileName: (c) => `${baseName(c)}-${shownField().replace(/^custom:/, '')}.png`,
});
defineExport({
	name: 'surface',
	label: 'Surface',
	format: 'VTP',
	title: 'The analysed file with CpPredicted and CpSpread cell arrays added',
	// The engine holds the whole predicted surface of the file analysed last only, from its exact numbers on. A variant's
	// surface was written with its prediction when it was made.
	unavailable: (c) => (engine.canExportVtp(c.result) ? null
		: c.variant ? `The variant's surface, with CpPredicted and CpSpread, is in the workspace at ${c.path}.`
			: 'The surface file exists for the file analysed last, once every polygon is predicted; open or reopen it to export it.'),
	make: (c) => engine.exportVtp(c.result),
	fileName: (c) => `${baseName(c)}_prediction.vtp`,
});

// ---- components ------------------------------------------------------------------------------------------------------

const progress = createProgress({ cancel: (c) => run('close_case', { case_id: c.id }) });

const toolbar = createToolbar({
	field: (field) => run('set_field', { field }),
	view: (name) => run('set_camera_view', { view: name }),
	fit: () => run('fit_view', {}),
	projection: (projection) => run('set_projection', { projection }),
	toggle: (what) => toggle(what),
});
$('[data-toolbar]').append(toolbar.el);

function toggle(what) {
	switch (what) {
		case 'edges':
		case 'ground':
			return run('set_display', { [what]: !view.display[what] });
		case 'section':
			if (!active()?.result) return;
			return run('set_section', { on: !view.section.on });
		default:
			throw new Error(`Unknown toggle ${what}`);
	}
}

const legend = createLegend({
	change: (patch) => run('set_colour_scale', {
		field: legendField(),
		...(patch.lo !== undefined ? { min: patch.lo } : {}),
		...(patch.hi !== undefined ? { max: patch.hi } : {}),
		...(patch.ramp !== undefined ? { ramp: patch.ramp } : {}),
		...(patch.banded !== undefined ? { banded: patch.banded } : {}),
	}),
	reset: () => run('set_colour_scale', { field: legendField(), reset: true }),
});
$('[data-legend]').append(legend.el);

const probes = createProbes({ host: overlay, engine, viewer, active, session, workspace, update, run, fieldIn: (pane) => panes()[pane]?.field ?? shownField() });
const section = createSectionTool({ host: overlay, viewport, viewer, active, cases: () => session.cases, view, run });
defineReport({ session, engine, regions, section, comparison, variants });
const regionsPanel = createRegionsPanel({ regions, session, viewer, host: overlay, run });
/** The dock's panel tabs; each one's `stop` undoes what it shows on the car when another tab is chosen. */
const panels = [
	{ name: 'regions', label: 'Regions', ...regionsPanel },
	{ name: 'histogram', label: 'Histogram', ...createHistogramPanel({ session, regions, active: described, run }) },
	{ name: 'compare', label: 'Compare', ...comparison.panel },
	{ name: 'variants', label: 'Variants', ...variants.panel },
	{ name: 'layers', label: 'Layers', ...annotations.panel },
];

const dock = createDock({
	section,
	probes,
	panels,
	active: described,
	view,
	actions: {
		tab: (name) => {
			// Painting, a lasso or a brushed range belongs to its tab; leaving it gives the car back.
			for (const panel of panels) if (panel.name !== name) panel.stop();
			// Choosing Section turns the section on, which brings its plot up; when it is on already the tab just shows it.
			if (name === 'section' && !view.section.on) return run('set_section', { on: true });
			view.dock = name;
			view.dockCollapsed = false;
			update('toolbar', 'viewer', 'dock', 'layout');
		},
		collapse: () => {
			view.dockCollapsed = !view.dockCollapsed;
			update('dock', 'layout');
		},
		save: () => run('download_export', { export: view.dock }),
	},
});
$('[data-dock]').append(dock.el);

// ---- top bar: files, case tabs, modes, export, assistant -------------------------------------------------------------

const filesPanel = createFilesPanel({ session, workspace, engine, run, progress: (c) => progress.fraction(c) });
$('.work').prepend(filesPanel.el);
$('.brand').after(filesPanel.toggle);

const tabs = $('[data-cases]');
function renderTabs() {
	tabs.replaceChildren(...session.cases.map((c) => h('div', { class: 'case-tab', role: 'tab', 'aria-selected': String(c === active()), 'data-status': c.status, title: c.name },
		h('button', { type: 'button', class: 'case-open', onclick: () => run('show_case', { case_id: c.id }) },
			h('span', { class: 'case-state', 'aria-hidden': 'true' }),
			h('span', { class: 'case-name' }, shortName(c))),
		h('button', { type: 'button', class: 'case-close', title: c.status === 'analysing' || c.status === 'queued' ? `Cancel and close ${c.name}` : `Close ${c.name}`, onclick: () => run('close_case', { case_id: c.id }) }, icon('close')))));
}

/**
 * The model's own page, for reading how it is built, is one click away but not a product control: a quiet toggle in
 * the top bar that hands it the whole area under the bar and gives the tool back as it was.
 */
const modelToggle = h('button', { type: 'button', class: 'model-toggle', 'aria-pressed': 'false', onclick: () => setMode(mode === 'model' ? 'result' : 'model') }, h('span', {}, 'Model'));
$('[data-model-toggle]').append(modelToggle);

let modelMounted = null;
function setMode(next) {
	mode = next;
	if (mode === 'model') modelMounted ??= import('./model/index.js').then(({ mountModelView }) => mountModelView(modelHost, { engine, session, viewer }));
	update('layout');
}

/**
 * The Export menu: every export, downloaded by its row or saved into the workspace by the folder beside it. Those
 * that cannot be made now are disabled with the reason as their tooltip; the surface file is offered only while the
 * engine holds the car's whole predicted surface.
 */
const exportNote = h('p', { class: 'menu-note', role: 'alert', hidden: true });
const exportList = h('div', { class: 'menu-list export-list' });
const exportButton = h('button', { type: 'button', class: 'button export-button' }, icon('download'), 'Export');
const exportMenu = menu(exportButton, h('div', {}, exportList, exportNote), { align: 'right' });
$('[data-export]').append(exportMenu.el);

async function exportTo(row, name, to) {
	exportNote.hidden = true;
	row.classList.add('busy');
	try {
		await run(to === 'download' ? 'download_export' : 'save_export', { export: name });
		exportMenu.close();
	} catch (error) {
		exportNote.textContent = error.message;
		exportNote.hidden = false;
	} finally {
		row.classList.remove('busy');
	}
}

function renderExportMenu() {
	const c = active();
	exportList.replaceChildren(...[...EXPORTS.values()].flatMap((e) => {
		const why = e.unavailable(c);
		if (e.name === 'surface' && why) return [];
		const row = h('div', { class: 'export-row', 'data-export': e.name },
			h('button', { type: 'button', class: 'menu-item', title: why ?? e.title, disabled: Boolean(why), onclick: () => exportTo(row, e.name, 'download') }, h('span', {}, e.label), h('kbd', {}, e.format)),
			h('button', { type: 'button', class: 'icon-button export-keep', title: `Save ${e.label.toLowerCase()} to files`, disabled: Boolean(why), onclick: () => exportTo(row, e.name, 'files') }, icon('keep')));
		return [row];
	}));
}
exportButton.addEventListener('click', () => {
	exportNote.hidden = true;
	renderExportMenu();
});

// ---- the assistant: a panel on the right, open in either view ------------------------------------------------------------

/** The assistant acts only through the operations, as the engineer's controls do; its runs are marked as its own. */
const assistantRun = (name, args) => runOperation(name, args, { session, viewer, engine, by: 'assistant' });
let assistantMounted = null;
const assistantButton = h('button', { type: 'button', class: 'button assistant-button', 'aria-pressed': 'false', 'aria-controls': 'assistant', onclick: () => setAssistant(!body.classList.contains('assistant-open')) }, icon('assistant'), 'Assistant');
$('.topbar-end').append(assistantButton);

/** The assistant's column opens beside the stage, which narrows to make room. */
function setAssistant(open) {
	body.classList.toggle('assistant-open', open);
	assistantButton.setAttribute('aria-pressed', String(open));
	if (open) assistantMounted ??= import('../assistant/index.js').then(({ mountAssistant }) => mountAssistant(assistantHost, { session, engine, viewer, workspace, operations, runOperation: assistantRun, onOperation, close: () => setAssistant(false) }));
}

// ---- result card ----------------------------------------------------------------------------------------------------------

const resultCard = createResultCard({ host: card, overlay, viewer, active: described, regions, run });

// ---- the middle of the view: first screen, progress, errors ------------------------------------------------------------

// An example that fails to load shows why on its own card, so the first screen does not wait on the operation.
const start = startScreen({ onFile: (file) => addFiles([file], { workspace, run }), onExample: (id) => run('open_example', { example: id }).catch(() => {}) });

function errorCard(c) {
	return h('div', { class: 'notice', role: 'alert' },
		h('div', { class: 'notice-title' }, c.name),
		h('p', { class: 'notice-text' }, c.error),
		h('div', { class: 'notice-actions' },
			h('button', { type: 'button', class: 'button button-primary', onclick: () => filesPanel.setOpen(true) }, 'Open another file'),
			h('button', { type: 'button', class: 'button', onclick: () => run('close_case', { case_id: c.id }) }, 'Close')));
}

/** The notice an example shows while it loads and until its car is drawn: one per case, so it stays put meanwhile. */
const loadingCards = new WeakMap();
function loadingCard(c) {
	if (!loadingCards.has(c)) loadingCards.set(c, h('div', { class: 'notice', role: 'status' }, h('div', { class: 'notice-title' }, c.name), h('div', { class: 'progress-bar indeterminate' }, h('span', { class: 'progress-fill' }))));
	return loadingCards.get(c);
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
			if (center.firstChild !== loadingCard(c)) center.replaceChildren(loadingCard(c));
			strip.replaceChildren();
			return;
		case 'queued':
		case 'analysing':
			progress.render(c, session.device);
			placeProgress(carShown(c) ? strip : center);
			return;
		case 'failed':
			center.replaceChildren(errorCard(c));
			strip.replaceChildren();
			return;
		case 'ready':
			// A result whose car has not been drawn yet (an example still on its way to the GPU) keeps the loading
			// notice, so the car, its card and its plots arrive together.
			if (c.drawn) center.replaceChildren();
			else if (center.firstChild !== loadingCard(c)) center.replaceChildren(loadingCard(c));
			strip.replaceChildren();
			return;
		default:
			throw new Error(`Unknown status ${c.status}`);
	}
}

/** Whether the progress is travelling to the strip; it is moved once, when the transition takes its first picture. */
let travelling = false;

/**
 * Put the progress in the middle of the view or in the strip. When the car appears, the same steps travel down to the
 * strip, so the eye follows them there; under reduced motion, or without view transitions, they move at once.
 */
function placeProgress(place) {
	if (progress.el.parentElement === place || travelling) return;
	const move = () => {
		center.replaceChildren(...(place === center ? [progress.el] : []));
		strip.replaceChildren(...(place === strip ? [progress.el] : []));
	};
	if (place !== strip || progress.el.parentElement !== center || !document.startViewTransition || REDUCED_MOTION.matches) return move();
	travelling = true;
	document.startViewTransition(move).updateCallbackDone.finally(() => (travelling = false));
}

// ---- keeping everything in step ------------------------------------------------------------------------------------------

function renderLayout() {
	const c = active();
	body.dataset.screen = !c ? 'start' : c.status === 'failed' || c.status === 'loading' || (c.status === 'ready' && !c.drawn) ? 'notice' : 'case';
	body.dataset.mode = mode;
	modelToggle.setAttribute('aria-pressed', String(mode === 'model'));
	modelToggle.title = mode === 'model' ? 'Back to the tool' : 'How the model is built';
	$('[data-export]').hidden = !ready();
	modelHost.hidden = mode !== 'model';
	viewer.hoverPicking = ready() && !viewer.tool;
	body.classList.toggle('dock-closed', view.dockCollapsed);
}

/** While the Model view owns the car, the field picker and the section wait until Result is back. */
function renderToolbar() {
	const c = active();
	const enabled = Boolean(c?.result);
	toolbar.render({
		fields: fieldsOf(c),
		field: shownField(),
		maps: view.maps,
		enabled,
		projection: viewer.projection,
		display: view.display,
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
	// The one field the app supplies (fields.addField), for whichever pane draws one.
	const supplied = (c?.result && shown.map((pane) => FIELDS[pane.field].values?.(c)).find(Boolean)) || null;
	if (supplied !== viewer.custom) viewer.setCustomField(supplied);
	paneLabels.replaceChildren(...(shown.length > 1 ? shown.map((pane) => h('span', { class: 'pane-label' }, pane.label)) : []));
	// Two cars side by side take the whole view; the comparison table in the dock carries the card's numbers.
	body.classList.toggle('side-by-side', shown.length > 1);
	const s = view.section;
	viewer.setSection(c?.result && s.on ? { axis: s.axis, offset: s.offset, clip: s.clip, plane: Boolean(viewer.section?.plane) } : null);
	viewer.setKeptSections(c?.result && s.on ? section.keptLines() : []);
	viewer.setDisplay(view.display);
	// The selection's tint, on the car it was worked out for; a new array only when the selection or the car changes.
	const tint = ready() ? regions.tintOf(c) : null;
	if (tint !== viewer.selection) viewer.setSelection(tint);
	annotations.apply(c);
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
}

/**
 * While a case is analysed the view shows the work: until its car is on screen, a band crossing the lattice at the
 * share done; then the car itself, at the share predicted, until the field arrives on it.
 */
function showWork(c) {
	const analysing = c?.status === 'analysing' || c?.status === 'queued';
	const shown = carShown(c);
	viewer.setSweep(analysing && !shown ? progress.fraction(c) : null);
	viewer.predicting(analysing && shown ? progress.predicted(c) : null);
}

/** Put a case on screen, or bring the screen up to date with it: its surface, its result and everything drawn from it. */
function present(c) {
	const arriving = Boolean(c?.result && !c.arrived);
	if (arriving) {
		c.arrived = true;
		// Every car opens on its centreline section; the sections kept in the plot and how it is drawn stay. A variant
		// opens as its car was left, in the tab it was made from.
		if (!c.variant) {
			Object.assign(view.section, { on: true, axis: 1, offset: 0, clip: 0 });
			view.dock = 'section';
			view.dockCollapsed = false;
		}
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
	showWork(c);
}

session.addEventListener('change', ({ detail: { case: c, what } }) => {
	switch (what) {
		// The section plot can show every open car, and its All cars control needs two of them.
		case 'cases':
			update('tabs', 'dock');
			return;
		case 'active':
			present(c);
			return;
		case 'status':
			update('tabs', 'dock');
			if (c === active()) present(c);
			return;
		case 'stage':
			if (c !== active()) return;
			showWork(c);
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
			if (ready()) renderExportMenu();
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
	const c = active();
	// The first time a case's car is drawn, everything about it comes on screen with it.
	if (c && carShown(c) && !c.drawn) {
		c.drawn = true;
		present(c);
		return;
	}
	placeBackdrop(c);
	showWork(c);
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
	if (e.metaKey || e.ctrlKey || e.altKey || e.target.closest('input, textarea, select, [contenteditable]')) return;
	if (!active()?.surface) return;
	const key = e.key.toLowerCase();
	if (VIEW_BY_KEY[key] && VIEWS[VIEW_BY_KEY[key]]) run('set_camera_view', { view: VIEW_BY_KEY[key] });
	else if (key === 'f') run('fit_view', {});
	else if (key === 'p' || key === 'o') run('set_projection', { projection: key === 'p' ? 'perspective' : 'orthographic' });
	else if (key === 'e' || key === 'g' || key === 's') toggle({ e: 'edges', g: 'ground', s: 'section' }[key]);
});

acceptDrops((file) => addFiles([file], { workspace, run }));
present(null);

export const app = { engine, session, viewer, view, workspace, update, run, openPath: files.openPath };
