// The model view: whether the model can stand in for CFD, what it is told about a point on the car and how it turns
// that into pressure and drag, and where it applies. A click on the car picks the point and the pointer previews
// others; the stencil on the car and every value here follow. Everything comes from the model card, `engine.model`,
// and from `engine.explain` at the point, so a retrained model changes the view without a change here.

import { h, put } from '../dom.js';
import { icon } from '../icons.js';
import { appliesBlock, evidenceBlock } from './evidence.js';
import { inputGroups, inputUnits } from './inputs.js';
import { PIPELINE } from './pipeline.js';
import { Stencil, emphasized, stencilMarks } from './stencil.js';

/** How long the diagram shows a new point's values flowing through it. */
const FLOW_MS = 1100;

/**
 * Where the view starts on a car: the working-mesh vertex facing up nearest a point on the hood, three tenths of the
 * way out on the side the standard isometric view looks at.
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

/** A point to explain: where it is, the normal of the working-mesh vertex there, and that vertex. */
function requestAt(point, vertex, normals) {
	return { point: Array.from(point), normal: Array.from(normals.subarray(3 * vertex, 3 * vertex + 3)), vertex };
}

/** What the car shows in the model view: clay under the stencil, or prediction and CFD side by side. */
const CLAY = [{ field: 'none' }];
const SIDE_BY_SIDE = [{ field: 'cp', label: 'Predicted' }, { field: 'cfd', label: 'CFD' }];

export function mountModelView(host, { engine, session, viewer, showPanes }) {
	const stencil = new Stencil(viewer);
	const evidence = h('div');
	const applies = h('div');
	const tip = h('div', { class: 'model-tip', hidden: true, role: 'tooltip' });
	const parts = PIPELINE.map((part, i) => {
		const value = h('span', { class: 'part-value num' });
		const figure = h('div', { class: 'part-figure' });
		const detail = h('div', { class: 'part-detail-inner' });
		const head = part.detail
			? h('button', { type: 'button', class: 'part-head', 'aria-expanded': 'false', onclick: () => act.open(part.id) },
				h('span', { class: 'part-title' }, part.title), value, icon('chevron'))
			: h('div', { class: 'part-head' }, h('span', { class: 'part-title' }, part.title), value);
		const body = h('div', { class: 'part-body' }, figure, part.detail ? h('div', { class: 'part-detail' }, detail) : null);
		const el = h('section', { class: 'part', 'data-part': part.id, style: `--i: ${i}` },
			h('span', { class: 'part-rail', 'aria-hidden': 'true' }), h('span', { class: 'part-node', 'aria-hidden': 'true' }), head, body);
		return { part, el, head, body, value, figure, detail };
	});
	const pipeline = h('div', { class: 'pipeline' }, parts.map((p) => p.el));
	const panel = h('div', { class: 'model' }, evidence, pipeline, applies, tip);
	host.append(panel);

	const state = {
		card: null,
		groups: null,
		units: null,
		open: null,
		network: 0,
		selected: -1,
		focusedGroup: null,
		preview: null,
		previewAsked: 0,
		compare: false,
	};
	/** Per case: the point picked on it, what the engine said there, and the request it answers. */
	const points = new WeakMap();
	/** Per case: how many points have been picked on it, so only the latest pick's answer is kept. */
	const tickets = new WeakMap();

	const active = () => session.active;
	const showing = () => !host.hidden && state.card !== null && active()?.status === 'ready';
	const pinned = (c) => points.get(c) ?? null;
	const shownFor = (c) => state.preview ?? pinned(c);
	const sideBySide = (c) => state.compare && c.result.accuracy !== null;
	let carPanes = null;

	const act = {
		focus(group) {
			state.focusedGroup = group;
			draw();
		},
		open(id) {
			state.open = state.open === id ? null : id;
			render();
		},
		select(index) {
			state.selected = index;
			render();
			draw();
		},
		network(k) {
			state.network = k;
			state.open = 'networks';
			render();
		},
		compare(on) {
			state.compare = on;
			showCar();
			render();
			draw();
		},
		openExample(example) {
			session.openExample(example.id, example.label);
		},
		tip: {
			show(target, content) {
				put(tip, content);
				tip.hidden = false;
				const box = target.getBoundingClientRect();
				const frame = panel.getBoundingClientRect();
				const left = Math.min(frame.width - tip.offsetWidth - 4, Math.max(4, box.left + box.width / 2 - frame.left - tip.offsetWidth / 2));
				tip.style.transform = `translate(${left}px, ${box.top - frame.top - tip.offsetHeight - 6}px)`;
			},
			hide() {
				tip.hidden = true;
			},
		},
	};

	function context() {
		const c = active();
		const shown = shownFor(c);
		const explanation = shown?.explanation ?? null;
		const result = c.result;
		const vertex = explanation ? (explanation.at === 'vertex' ? explanation.index : shown.request.vertex) : -1;
		return {
			card: state.card,
			columns: state.card.columns,
			groups: state.groups,
			units: state.units,
			result,
			shown: explanation ? shown : null,
			failure: shown?.failure ?? null,
			cfdAt: explanation && result.mesh.cfd ? result.mesh.cfd[vertex] : null,
			given: new Set(state.card.comparison[state.selected].inputs),
			selected: state.selected,
			network: state.network,
			compare: sideBySide(c),
			width: parts[0].body.clientWidth,
			act,
		};
	}

	function render() {
		tip.hidden = true;
		if (!showing()) return;
		const ctx = context();
		put(evidence, evidenceBlock(ctx));
		for (const p of parts) {
			put(p.value, p.part.value(ctx));
			put(p.figure, p.part.figure?.(ctx) ?? null);
			if (!p.part.detail) continue;
			const open = state.open === p.part.id;
			if (open || p.el.dataset.open === 'true') put(p.detail, p.part.detail(ctx));
			p.el.dataset.open = String(open);
			p.head.setAttribute('aria-expanded', String(open));
		}
		put(applies, appliesBlock(ctx));
	}

	/** Clay under the stencil, or prediction beside CFD for a case that has it; the car is framed again for the panes. */
	function showCar() {
		const panes = sideBySide(active()) ? SIDE_BY_SIDE : CLAY;
		if (panes === carPanes) return;
		const reframe = carPanes !== null && panes.length !== carPanes.length;
		carPanes = panes;
		showPanes(panes);
		if (reframe) viewer.fit();
	}

	/**
	 * The stencil for the point shown, with the inputs the chosen model was given, the focused ones standing out. Beside
	 * CFD only the point and its normal are drawn, so the two fields can be compared.
	 */
	function draw() {
		const c = active();
		const explanation = showing() ? shownFor(c)?.explanation : null;
		if (!explanation) {
			stencil.clear();
			return;
		}
		const given = new Set(state.card.comparison[state.selected].inputs);
		const marks = stencilMarks(state.card.columns, state.groups, explanation, c.result.lookups.yCenter)
			.filter((m) => m.group === null || (!sideBySide(c) && given.has(m.kind)));
		stencil.draw(marks, (m) => emphasized(m, state.focusedGroup));
	}

	/** The point's values run down the diagram, once per point picked. */
	let flowTimer = 0;
	function flow() {
		pipeline.classList.remove('flowing');
		void pipeline.offsetWidth; // a layout read between the two restarts the animation if one is still running
		pipeline.classList.add('flowing');
		clearTimeout(flowTimer);
		flowTimer = setTimeout(() => pipeline.classList.remove('flowing'), FLOW_MS);
	}

	/** Explain a point picked on a case; the previous point stays shown until it is answered, and a later pick wins. */
	async function pin(c, request) {
		const ticket = (tickets.get(c) ?? 0) + 1;
		tickets.set(c, ticket);
		const entry = { request, explanation: null, failure: null };
		try {
			entry.explanation = await engine.explain(c.result, request.point, request.normal);
		} catch (error) {
			entry.failure = error.message;
		}
		if (tickets.get(c) !== ticket) return;
		points.set(c, entry);
		if (c !== active()) return;
		render();
		draw();
		if (entry.explanation) flow();
	}

	async function preview(c, request) {
		const asked = ++state.previewAsked;
		let explanation;
		try {
			explanation = await engine.explain(c.result, request.point, request.normal);
		} catch {
			// A point that cannot be explained is not previewed; picking it shows why.
			return;
		}
		if (asked !== state.previewAsked || c !== active() || !showing()) return;
		state.preview = { request, explanation, failure: null };
		render();
		draw();
	}

	function endPreview() {
		state.previewAsked++;
		if (!state.preview) return;
		state.preview = null;
		render();
		draw();
	}

	/** Bring the view up to date with the case shown, or take the stencil off the car when the view is hidden. */
	function sync() {
		const c = active();
		if (host.hidden || c?.status !== 'ready') {
			state.preview = null;
			stencil.clear();
			return;
		}
		showCar();
		if (!state.card) return;
		if (!tickets.has(c)) pin(c, hoodRequest(c.result));
		render();
		draw();
	}

	viewer.addEventListener('hover', ({ detail: { hit } }) => {
		if (!showing()) return;
		if (!hit) return endPreview();
		const c = active();
		preview(c, requestAt(hit.point, hit.vertex, c.result.mesh.normals));
	});
	viewer.addEventListener('tap', ({ detail: { hit } }) => {
		if (!showing() || !hit) return;
		const c = active();
		state.preview = null;
		state.previewAsked++;
		pin(c, requestAt(hit.point, hit.vertex, c.result.mesh.normals));
	});
	session.addEventListener('change', ({ detail: { what } }) => {
		if (what === 'active' || what === 'status' || what === 'numbers' || what === 'cases') sync();
	});
	new MutationObserver(sync).observe(host, { attributes: true, attributeFilter: ['hidden'] });
	// The figures are drawn to the panel's width; its height changes with every part opened and needs nothing.
	let width = 0;
	new ResizeObserver(([entry]) => {
		if (entry.contentRect.width === width) return;
		width = entry.contentRect.width;
		render();
	}).observe(panel);

	sync();
	engine.model.then(
		(card) => {
			state.card = card;
			state.groups = inputGroups(card.columns);
			state.units = inputUnits(card.comparison, state.groups);
			state.selected = card.comparison.findIndex((row) => row.learner === 'ensemble');
			sync();
		},
		(error) => put(panel, h('p', { class: 'point-failure', role: 'alert' }, error.message)),
	);
}
