// The model's page: one diagram of how the model works, read top to bottom. At the top the whole model in one line;
// below it a section for each step, drawn on one reference car with a point the reader moves. Everything comes from
// the model card (`engine.model`), the training cars' file, the reference car's result and `engine.explain` at the
// point, so a retrained model changes the page without a change here.
//
// The page is a reference, not a view of the workbench: it loads its car, a bundled example, itself, whatever is open
// there, and keeps it for the session. It has its own viewer for that car and takes its pictures of the car from it.

import { h } from '../dom.js';
import { EXAMPLES } from '../examples.js';
import { defaultMaps } from '../fields.js';
import { loadResult } from '../../engine/result.js';
import { Surface } from '../../viewer/surface.js';
import { createAccuracy } from './accuracy.js';
import { countWord } from './charts.js';
import { copied, loadCopy, shareValues } from './copy.js';
import { arrowDefinition } from './drawings.js';
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

/** The car the page draws on: DrivAerML run 1, kept out of training, with its CFD. */
const REFERENCE = EXAMPLES.find((example) => example.id === 'run_1');

/**
 * Where the page starts on the car: the working-mesh vertex facing up nearest a point on the hood, three tenths of the
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
 * The colour maps the pictures use, from the car alone: Cp on the tool's default scale, the difference from CFD on the
 * car's own, and the spread and the error's size on one scale that holds both.
 */
function picturesMaps(mesh) {
	const defaults = defaultMaps(mesh);
	const spread = { ...defaults.sigma, hi: Math.max(defaults.sigma.hi, defaults.error.hi) };
	return { cp: defaults.cp, error: defaults.error, spread };
}

/** Resolves once `host` is shown: the page's viewer draws nothing while it is hidden. */
function whenShown(host) {
	return new Promise((resolve) => {
		if (!host.hidden) return resolve();
		const watch = new MutationObserver(() => {
			if (host.hidden) return;
			watch.disconnect();
			resolve();
		});
		watch.observe(host, { attributes: true, attributeFilter: ['hidden'] });
	});
}

/** Build the Model page in `host`; it draws once `host` is shown. */
export function mountModelPage(host, { engine }) {
	const notice = h('div', { class: 'page-notice', role: 'status' });
	const title = h('h1');
	const page = h('div', { class: 'model-page', 'data-state': 'loading' },
		arrowDefinition(),
		h('header', { class: 'model-top' }, title),
		notice);
	host.append(page);

	// Asked for at once and held as an outcome, so it may fail before the words that report it are read.
	const reference = loadResult(new URL(`../../examples/${REFERENCE.id}`, import.meta.url).href).then((result) => ({ result }), (error) => ({ error }));
	const fail = (message) => {
		page.dataset.state = 'failed';
		notice.replaceChildren(message);
	};
	Promise.all([engine.model, loadTrainingCars(), loadCopy()]).then(([card, training]) => {
		// Values any words may name: {inputs}, {networks}, {networksWord}, and {car}, the reference car.
		shareValues({ inputs: card.columns.length, networks: card.networks.length, networksWord: countWord(card.networks.length), car: REFERENCE.label });
		copied(title, 'page.title');
		notice.replaceChildren(copied(h('p'), 'page.notice.waiting'));
		reference.then(({ result, error }) => (error
			? fail(copied(h('p', { class: 'point-failure', role: 'alert' }), 'page.notice.failed', { error: error.message }))
			: build(card, training, result)));
	}, (error) => fail(h('p', { class: 'point-failure', role: 'alert' }, error.message)));

	function build(card, training, result) {
		const groups = inputGroups(card);
		/** The numbers of the latest point asked about and of the latest shown. */
		const tickets = { asked: 0, shown: 0 };

		/** Ask about a point on the car. Answers are shown in the order asked; one overtaken by a later answer is dropped. */
		async function explain(request) {
			const ticket = ++tickets.asked;
			const point = { explanation: null, marks: [], failure: null };
			try {
				point.explanation = await engine.explain(result, request.point, request.normal);
				point.marks = measure(card.columns, result.lookups, point.explanation);
			} catch (error) {
				point.failure = error.message;
			}
			if (ticket < tickets.shown) return;
			tickets.shown = ticket;
			for (const part of parts) part.setPoint(point);
		}

		const inputs = createInputs({ card, groups, engine, ask: explain });
		const parts = [
			createOverview({ card, groups, training }),
			createPointwise({ card, training }),
			inputs,
			createNetworks({ card, groups }),
			createAccuracy({ card }),
			createSpread(),
		];
		page.append(h('div', { class: 'model-body' }, parts.map((part) => part.el)));
		// The car is in hand: the figures show, so the page's viewer has its size when it draws the car's pictures.
		page.dataset.state = 'ready';
		notice.replaceChildren();
		whenShown(host).then(() => draw(parts, inputs, result)).then(
			() => explain(hoodRequest(result)),
			(error) => fail(h('p', { class: 'point-failure', role: 'alert' }, error.message)),
		);
	}

	/** Draw the car in the page's viewer, take its pictures, and give the parts the car, its pictures and its numbers. */
	async function draw(parts, inputs, result) {
		const viewer = inputs.viewer;
		// The page's own copy of the mesh's fields, so its pictures can put the error where the spread was.
		const mesh = { ...result.mesh };
		const surface = new Surface(mesh);
		viewer.show(surface);
		viewer.resize();
		viewer.moveTo(viewer.framing(CAMERA), false);
		inputs.stencil.setCar(result);
		const maps = picturesMaps(result.mesh);
		const stats = caseStats(engine.workingCells(result), result.geometry.bounds, result.lookups.yCenter);
		await whenDrawn(viewer, surface);
		const side = (field, map) => ({ field, map, view: 'side' });
		const below = (field, map) => ({ field, map, view: 'below' });
		const wanted = {
			claySide: side('none', null), fieldSide: side('cp', maps.cp),
			sigmaSide: side('sigma', maps.spread), sigmaBelow: below('sigma', maps.spread),
			cfdSide: side('cfd', maps.cp), cpSide: side('cp', maps.cp), errorSide: side('error', maps.error),
			cfdBelow: below('cfd', maps.cp), cpBelow: below('cp', maps.cp), errorBelow: below('error', maps.error),
		};
		const taken = takePictures(viewer, Object.values(wanted));
		const pictures = Object.fromEntries(Object.keys(wanted).map((id, i) => [id, taken[i]]));
		// The error's size, drawn through the spread's colour map so the two pictures compare directly.
		surface.working.setFields({ sigma: Float32Array.from(mesh.cp, (cp, v) => Math.abs(cp - mesh.cfd[v])) });
		await whenDrawn(viewer, surface);
		[pictures.missSide, pictures.missBelow] = takePictures(viewer, [side('sigma', maps.spread), below('sigma', maps.spread)]);
		const car = { result, pictures, stats, maps, cutter: inputs.stencil.car.cutter };
		for (const part of parts) part.setCar(car);
	}
}
