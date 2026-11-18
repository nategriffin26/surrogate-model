// The model's page: one diagram of how the model works, read top to bottom. At the top the whole model in one line;
// below it a section for each step, drawn on the open car with a point the reader moves. Everything comes from the
// model card (`engine.model`), the training cars' file, the open car's result and `engine.explain` at the point, so a
// retrained model changes the page without a change here.
//
// The page has its own viewer for the car it draws on, and takes its pictures of the car's fields from it.

import { h } from '../dom.js';
import { EXAMPLES } from '../examples.js';
import { defaultMaps } from '../fields.js';
import { Surface } from '../../viewer/surface.js';
import { createAccuracy } from './accuracy.js';
import { createDrag } from './drag.js';
import { arrowDefinition, silhouettes } from './drawings.js';
import { surfaceFlow } from './flow.js';
import { inputGroups } from './groups.js';
import { CAMERA, createInputs, requestAt } from './inputs.js';
import { measure } from './measures.js';
import { createNetworks } from './networks.js';
import { createOverview } from './overview.js';
import { takePictures, whenDrawn } from './pictures.js';
import { createPointwise } from './pointwise.js';
import { createSpread } from './spread.js';
import { caseStats } from './stats.js';
import { loadTrainingCars } from './training.js';

/**
 * Where the page starts on a car: the working-mesh vertex facing up nearest a point on the hood, three tenths of the
 * way out on the side the side pictures show.
 */
function hoodRequest({ mesh: { positions, normals }, geometry: { bounds }, lookups: { yCenter } }) {
	const [xmin, xmax, ymin, , , zmax] = bounds;
	const target = [xmin + 0.14 * (xmax - xmin), yCenter - 0.3 * (yCenter - ymin), zmax];
	let vertex = 0;
	let best = Infinity;
	for (let v = 0; v < positions.length / 3; v++) {
		if (normals[3 * v + 2] < 0.5) continue;
		const d = (positions[3 * v] - target[0]) ** 2 + (positions[3 * v + 1] - target[1]) ** 2 + (positions[3 * v + 2] - target[2]) ** 2;
		if (d < best) {
			best = d;
			vertex = v;
		}
	}
	return requestAt(positions.slice(3 * vertex, 3 * vertex + 3), vertex, normals);
}

/**
 * The colour maps the pictures use: Cp on the tool's shared scale, the difference from CFD on this car's, and the
 * spread and the error's size on one scale that holds both.
 */
function picturesMaps(mesh, shared) {
	const defaults = defaultMaps(mesh);
	const cp = shared?.cp ?? defaults.cp;
	const spread = { ...defaults.sigma, hi: Math.max(defaults.sigma.hi, defaults.error?.hi ?? 0) };
	return { cp, error: defaults.error ?? null, spread };
}

/** What the page shows for the open case, by its status. */
function caseState(c) {
	if (!c) return 'none';
	switch (c.status) {
		case 'queued':
		case 'loading':
		case 'analysing':
			return 'waiting';
		case 'ready':
			return 'ready';
		case 'failed':
			return 'failed';
		default:
			throw new Error(`Unknown status ${c.status}`);
	}
}

export function mountModelView(host, { engine, session }) {
	const notice = h('div', { class: 'case-notice', role: 'status' });
	const page = h('div', { class: 'model-page', 'data-case': 'none' },
		arrowDefinition(),
		h('header', { class: 'model-top' }, h('h1', {}, 'How the model predicts pressure')),
		notice);
	host.append(page);

	Promise.all([engine.model, loadTrainingCars()]).then(
		([card, training]) => build(card, training),
		(error) => page.append(h('p', { class: 'point-failure', role: 'alert' }, error.message)),
	);

	function build(card, training) {
		const groups = inputGroups(card);
		/** Per case: the point shown on it, and the numbers of the latest point asked about and of the latest shown. */
		const points = new WeakMap();
		const tickets = new WeakMap();
		const view = { case: null, caseView: null, surface: null, opening: null };

		function ask(request) {
			const c = view.case;
			if (c) explain(c, request);
		}

		/** Ask about a point on a case. Answers are shown in the order asked; one overtaken by a later answer is dropped. */
		async function explain(c, request) {
			const ticket = (tickets.get(c)?.asked ?? 0) + 1;
			tickets.set(c, { asked: ticket, shown: tickets.get(c)?.shown ?? 0 });
			const point = { explanation: null, marks: [], failure: null };
			try {
				point.explanation = await engine.explain(c.result, request.point, request.normal);
				point.marks = measure(card.columns, c.result.lookups, point.explanation);
			} catch (error) {
				point.failure = error.message;
			}
			const order = tickets.get(c);
			if (ticket < order.shown) return;
			order.shown = ticket;
			points.set(c, point);
			if (c === view.case && view.caseView) for (const part of parts) part.setPoint(point);
		}

		const inputs = createInputs({ card, groups, ask });
		const parts = [
			createOverview({ card, groups, training }),
			createPointwise({ card, groups, training }),
			inputs,
			createNetworks({ card, groups, setNetwork: (k) => inputs.setNetwork(k) }),
			createAccuracy({ card }),
			createSpread({ card }),
			createDrag({ card, groups, training }),
		];
		page.append(h('div', { class: 'model-body' }, parts.map((part) => part.el)));
		const viewer = inputs.viewer;

		/** Draw the page on a ready case: its car in the page's viewer, its pictures, its numbers, then its point. */
		async function open(c) {
			const result = c.result;
			// The page's own copy of the mesh's fields, so its pictures can put the error where the spread was.
			const mesh = { ...result.mesh };
			const surface = new Surface(mesh);
			view.surface?.dispose();
			view.surface = surface;
			viewer.show(surface);
			viewer.resize();
			viewer.moveTo(viewer.framing(CAMERA), false);
			inputs.stencil.setCar(result);
			const maps = picturesMaps(result.mesh, session.view.maps);
			const stats = result.mesh.cfd ? caseStats(engine.workingCells(result), result.geometry.bounds, result.lookups.yCenter) : null;
			if (!(await whenDrawn(viewer, surface))) return;
			const side = (field, map) => ({ field, map, view: 'side' });
			const below = (field, map) => ({ field, map, view: 'below' });
			const wanted = {
				claySide: side('none', null), fieldSide: side('cp', maps.cp),
				sigmaSide: side('sigma', maps.spread), sigmaBelow: below('sigma', maps.spread),
				...(stats ? {
					cfdSide: side('cfd', maps.cp), cpSide: side('cp', maps.cp), errorSide: side('error', maps.error),
					cfdBelow: below('cfd', maps.cp), cpBelow: below('cp', maps.cp), errorBelow: below('error', maps.error),
				} : {}),
			};
			const taken = takePictures(viewer, Object.values(wanted));
			const pictures = Object.fromEntries(Object.keys(wanted).map((id, i) => [id, taken[i]]));
			if (stats) {
				// The error's size, drawn through the spread's colour map so the two pictures compare directly.
				surface.working.setFields({ sigma: Float32Array.from(mesh.cp, (cp, v) => Math.abs(cp - mesh.cfd[v])) });
				if (!(await whenDrawn(viewer, surface))) return;
				[pictures.missSide, pictures.missBelow] = takePictures(viewer, [side('sigma', maps.spread), below('sigma', maps.spread)]);
			}
			const flow = card.columns.some((column) => column.kind === 'flow') ? await flowPictures(result, surface, maps) : null;
			if (view.case !== c) return;
			view.caseView = { result, pictures, stats, maps, flow, surface, outlines: silhouettes(result.mesh), cutter: inputs.stencil.car.cutter };
			for (const part of parts) part.setCase(view.caseView);
			if (points.has(c)) for (const part of parts) part.setPoint(points.get(c));
			if (!tickets.has(c)) explain(c, hoodRequest(result));
		}

		/**
		 * The potential flow over the car: its pressure on the page's car pictured from the side through the Cp map,
		 * beside CFD's from the same view when the file has it, and its velocity arrows. The car's own Cp is put back.
		 */
		async function flowPictures(result, surface, maps) {
			const flow = surfaceFlow(result.mesh, await engine.flowOnMesh(result));
			surface.working.setFields({ cp: flow.pressure });
			if (!(await whenDrawn(viewer, surface))) return null;
			const [picture, cfdPicture = null] = takePictures(viewer, [
				{ field: 'cp', map: maps.cp, view: 'side' },
				...(result.mesh.cfd ? [{ field: 'cfd', map: maps.cp, view: 'side' }] : []),
			]);
			surface.working.setFields({ cp: result.mesh.cp });
			return { ...flow, picture, cfdPicture, map: maps.cp };
		}

		/** Bring the page up to date with the open case while it is shown. */
		function sync() {
			const c = session.active;
			const state = caseState(c);
			page.dataset.case = state;
			showNotice(c, state);
			if (host.hidden || state !== 'ready' || c === view.case) return;
			view.case = c;
			view.caseView = null;
			open(c).catch((error) => {
				page.dataset.case = 'failed';
				notice.replaceChildren(h('p', { class: 'point-failure', role: 'alert' }, error.message));
			});
		}

		session.addEventListener('change', ({ detail: { what, case: c } }) => {
			// The exact numbers of an analysed file replace the sampled ones: same car, new forces.
			if (what === 'numbers' && c === view.case && view.caseView) {
				view.caseView = { ...view.caseView, result: c.result };
				for (const part of parts) part.setCase(view.caseView);
				return;
			}
			if (what === 'active' || what === 'status' || what === 'cases') sync();
		});
		new MutationObserver(sync).observe(host, { attributes: true, attributeFilter: ['hidden'] });
		sync();
	}

	function showNotice(c, state) {
		switch (state) {
			case 'none':
				notice.replaceChildren(h('p', {}, 'The figures are drawn on an open car.'),
					h('div', { class: 'notice-actions' }, EXAMPLES.map((example) => h('button', { type: 'button', class: 'button', onclick: () => session.openExample(example.id, example.label) }, `Open ${example.label}`))));
				return;
			case 'waiting':
				notice.replaceChildren(h('p', {}, `${c.name} is being read. The figures follow once it is ready.`));
				return;
			case 'failed':
				notice.replaceChildren(h('p', { class: 'point-failure', role: 'alert' }, c.error));
				return;
			case 'ready':
				notice.replaceChildren();
				return;
			default:
				throw new Error(`Unknown case state ${state}`);
		}
	}
}
