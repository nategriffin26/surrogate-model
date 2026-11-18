// The model view: why the model is built as it is, told down a scrolling column beside the car, with what the
// networks were told about one point drawn on the car. The point is a handle: drag it across the car, or click
// anywhere on the car, and the drawing, the inputs, each network's working and the answers follow. Everything
// comes from the model card, `engine.model`, and from `engine.explain` at the point, so a retrained model changes
// the view without a change here.

import { h, put } from '../dom.js';
import { createCorner } from './corner.js';
import { columnValue, inputGroups, inputLayers } from './inputs.js';
import { LAYERS, measure } from './measures.js';
import { Stencil } from './stencil.js';
import { createStory } from './story.js';

/** What the car shows in the model view: clay under the drawing, or prediction and CFD side by side. */
const CLAY = [{ field: 'none' }];
const SIDE_BY_SIDE = [{ field: 'cp', label: 'Predicted' }, { field: 'cfd', label: 'CFD' }];
/** A press that moves farther than this, in CSS pixels, turns the camera instead of picking a point. */
const CLICK_SLOP = 4;

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

/** A point to explain: where it is, and the normal of the working-mesh vertex there. */
function requestAt(point, vertex, normals) {
	return { point: Array.from(point), normal: Array.from(normals.subarray(3 * vertex, 3 * vertex + 3)) };
}

/** A cut's name, from the offset its inputs are read at. */
function cutName(mark, columns) {
	const [, dy, dz] = columns[mark.columns[0]].offset;
	const cm = (v) => `${Math.round(Math.abs(v) * 100)} cm`;
	if (mark.layer === 'slice') return 'Streamwise cut through the point';
	if (mark.layer === 'neighbour') return `Streamwise cut ${cm(dy)} ${dy < 0 ? 'inboard' : 'outboard'}`;
	return dz === 0 ? 'Outline at the point’s height' : `Outline ${cm(dz)} ${dz < 0 ? 'lower' : 'higher'}`;
}

export function mountModelView(host, { engine, session, viewer, showPanes, corner }) {
	const stencil = new Stencil(viewer);
	const canvas = viewer.canvas;
	const tip = h('div', { class: 'model-tip', hidden: true, role: 'tooltip' });
	host.append(tip);

	const state = {
		card: null,
		groups: null,
		layers: null,
		network: 0,
		shown: new Set(LAYERS.filter((layer) => layer.shown).map((layer) => layer.id)),
		scaleShown: null,
		/** Inputs the reader is pointing at, in the column or on the car. */
		pointed: null,
		/** The drawing layer the reader is pointing at in the corner card. */
		pointedLayer: null,
		/** The drawing layers of the part of the story being read. */
		reading: null,
		compare: false,
	};
	/** Per case: the point shown on it, what the engine said there and the marks it makes. */
	const points = new WeakMap();
	/** Per case: the number of the latest point asked about, and of the latest answer shown. */
	const tickets = new WeakMap();
	let story = null;
	let cornerCard = null;
	let storyCase = null;
	let carPanes = null;

	const active = () => session.active;
	const showing = () => !host.hidden && state.card !== null && active()?.status === 'ready';
	const sideBySide = (c) => state.compare && c.result.accuracy !== null;

	const act = {
		/** The reader points at inputs in the column: one input is named with its value, a group with its count. */
		hoverColumns(columns, target) {
			state.pointed = columns ? new Set(columns) : null;
			const explanation = points.get(active())?.explanation;
			if (columns && target && explanation) {
				const column = state.card.columns[columns[0]];
				const box = target.getBoundingClientRect();
				act.tip(target, columns.length > 1
					? h('span', {}, h('b', {}, column.group), ` ${columns.length} inputs`)
					: h('span', {}, h('b', {}, column.name), ` ${columnValue(column, explanation.features[columns[0]])}`), box.left + box.width / 2, box.top);
			} else {
				act.untip();
			}
			story.highlight(columns);
			draw();
		},
		focusColumns(columns) {
			state.pointed = columns ? new Set(columns) : null;
			draw();
		},
		hoverLayer(id) {
			state.pointedLayer = id;
			draw();
		},
		reading(layers) {
			state.reading = layers;
			draw();
		},
		toggleLayer(id) {
			if (state.shown.has(id)) state.shown.delete(id);
			else state.shown.add(id);
			renderCorner();
			draw();
		},
		toggleScale(scale) {
			if (state.scaleShown.has(scale)) state.scaleShown.delete(scale);
			else state.scaleShown.add(scale);
			renderCorner();
			draw();
		},
		network(k) {
			state.network = k;
			show(false);
		},
		compare(on) {
			state.compare = on;
			showCar();
			renderCorner();
			storyCase = null;
			sync();
		},
		openExample(example) {
			session.openExample(example.id, example.label);
		},
		tip(target, content, x, y) {
			put(tip, content);
			tip.hidden = false;
			const left = Math.min(innerWidth - tip.offsetWidth - 8, Math.max(8, x - tip.offsetWidth / 2));
			tip.style.transform = `translate(${left}px, ${y - tip.offsetHeight - 12}px)`;
		},
		untip() {
			tip.hidden = true;
		},
	};

	/** Whether a mark stands out: the inputs pointed at, else the layer pointed at, else the part being read. */
	function strong(mark) {
		if (state.pointed) return mark.columns.some((i) => state.pointed.has(i));
		if (state.pointedLayer) return mark.layer === state.pointedLayer;
		if (state.reading) return state.reading.includes(mark.layer);
		return true;
	}

	/** The marks drawn: those of the layers and box sizes switched on; the boxes up- and downstream only when pointed at. */
	function drawn(marks) {
		return marks.filter((m) => state.shown.has(m.layer)
			&& (m.type !== 'block' || (state.scaleShown.has(m.scale) && (m.sample === 'here' || (state.pointed && strong(m))))));
	}

	function draw() {
		const c = active();
		const entry = showing() ? points.get(c) : null;
		if (!entry?.explanation) {
			stencil.clear();
			return;
		}
		stencil.draw(entry.explanation, drawn(entry.marks), strong, sideBySide(c));
	}

	/** Show the point of the active case everywhere: on the car and down the column. */
	function show(sweep) {
		const c = active();
		const entry = showing() ? points.get(c) : null;
		draw();
		if (!entry) return;
		const { explanation } = entry;
		story.update({
			explanation,
			failure: entry.failure,
			cfdAt: explanation ? explanation.cfd : null,
			network: state.network,
			sweep,
		});
	}

	/** Ask about a point on a case. Answers are shown in the order asked; one overtaken by a later answer is dropped. */
	async function explain(c, request, sweep) {
		const ticket = (tickets.get(c)?.asked ?? 0) + 1;
		tickets.set(c, { asked: ticket, shown: tickets.get(c)?.shown ?? 0 });
		const entry = { request, explanation: null, failure: null, marks: [] };
		try {
			entry.explanation = await engine.explain(c.result, request.point, request.normal);
			entry.marks = measure(state.card.columns, c.result.lookups, entry.explanation);
		} catch (error) {
			entry.failure = error.message;
		}
		const order = tickets.get(c);
		if (ticket < order.shown) return;
		order.shown = ticket;
		points.set(c, entry);
		if (c === active()) show(sweep);
	}

	/** Clay under the drawing, or prediction beside CFD for a case that has it; the car is framed again for the panes. */
	function showCar() {
		const panes = sideBySide(active()) ? SIDE_BY_SIDE : CLAY;
		if (panes === carPanes) return;
		const reframe = carPanes !== null && panes.length !== carPanes.length;
		carPanes = panes;
		showPanes(panes);
		if (reframe) viewer.fit();
	}

	function renderCorner() {
		cornerCard?.render({ shown: state.shown, scaleShown: state.scaleShown, hidden: !showing() || sideBySide(active()) });
	}

	/** Bring the view up to date with the case shown, or take the drawing off the car when the view is hidden. */
	function sync() {
		const c = active();
		renderCorner();
		if (host.hidden || c?.status !== 'ready') {
			stencil.clear();
			return;
		}
		showCar();
		if (!state.card) return;
		stencil.setCar(c.result);
		if (storyCase !== c) {
			storyCase = c;
			story.setCase(c.result, stencil.car.cutter, state.compare);
		}
		if (!tickets.has(c)) explain(c, hoodRequest(c.result), true);
		show(false);
	}

	// ---- the point on the car: click to move it, drag its handle, point at a mark to name it -------------------------

	let press = null;
	let dragging = null;
	let hovering = null;

	/** Pick under a client point and explain the surface there. */
	async function pickAndExplain(x, y, sweep) {
		const c = active();
		const hit = await viewer.pick(x, y);
		if (hit && c === active() && showing()) explain(c, requestAt(hit.point, hit.vertex, c.result.mesh.normals), sweep);
	}

	/**
	 * While the handle is dragged the point follows the pointer; moves that arrive while a pick runs collapse into
	 * the latest.
	 */
	async function follow(x, y) {
		dragging.next = [x, y];
		if (dragging.picking) return;
		dragging.picking = true;
		while (dragging?.next) {
			const [px, py] = dragging.next;
			dragging.next = null;
			await pickAndExplain(px, py, false);
		}
		if (dragging) dragging.picking = false;
	}

	function nameMark(mark, x, y) {
		const { columns } = state.card;
		const explanation = points.get(active())?.explanation;
		const value = (i) => h('span', { class: 'tip-row' }, h('span', {}, columns[i].name), h('span', { class: 'num' }, columnValue(columns[i], explanation.features[i])));
		let content;
		if (mark === 'handle') content = h('span', {}, h('b', {}, 'The point'), ' Drag it across the car');
		else if (mark.type === 'cut') content = h('span', {}, h('b', {}, cutName(mark, columns)), ` ${mark.columns.length} inputs read along it`);
		else if (mark.type === 'block') content = h('span', { class: 'tip-rows' }, h('b', {}, `${Math.round(mark.scale * 100)} cm box, ${mark.sample}`), mark.columns.map(value));
		else content = h('span', { class: 'tip-rows' }, mark.columns.map(value));
		act.tip(canvas, content, x, y);
	}

	/** What is under a still pointer: the handle (to grab), or a mark (named, and its inputs lit). */
	function hover(x, y) {
		if (!showing() || dragging) return;
		const onHandle = stencil.onHandle(x, y);
		canvas.style.cursor = onHandle ? 'grab' : '';
		const mark = onHandle ? 'handle' : sideBySide(active()) ? null : stencil.markAt(x, y);
		if (mark === hovering) return;
		hovering = mark;
		// A box lights its whole size, here, upstream and downstream, so all three are drawn.
		const columns = !mark || mark === 'handle' ? null
			: mark.type === 'block' ? state.groups.find((g) => g.scale === mark.scale).columns : mark.columns;
		if (mark) nameMark(mark, x, y);
		else act.untip();
		state.pointed = columns ? new Set(columns) : null;
		story.highlight(columns);
		draw();
	}

	/** Pointer moves are tested against the marks once a frame. */
	let hoverNext = null;
	function hoverAt(x, y) {
		if (!hoverNext) requestAnimationFrame(() => {
			const [px, py] = hoverNext;
			hoverNext = null;
			hover(px, py);
		});
		hoverNext = [x, y];
	}

	canvas.addEventListener('pointerdown', (e) => {
		if (!showing() || e.button !== 0) return;
		press = { x: e.clientX, y: e.clientY, handle: stencil.onHandle(e.clientX, e.clientY) };
		if (!press.handle) return;
		// The handle's press is the model view's: the camera never sees it.
		e.stopImmediatePropagation();
		e.preventDefault();
		canvas.setPointerCapture(e.pointerId);
		dragging = { next: null, picking: false };
		canvas.style.cursor = 'grabbing';
		act.untip();
	}, { capture: true });

	canvas.addEventListener('pointermove', (e) => {
		if (dragging) {
			e.stopImmediatePropagation();
			follow(e.clientX, e.clientY);
		} else if (e.buttons === 0) {
			hoverAt(e.clientX, e.clientY);
		}
	}, { capture: true });

	const release = (e) => {
		if (!dragging) return;
		e.stopImmediatePropagation();
		canvas.releasePointerCapture(e.pointerId);
		dragging = null;
		canvas.style.cursor = 'grab';
	};
	canvas.addEventListener('pointerup', release, { capture: true });
	canvas.addEventListener('pointercancel', release, { capture: true });

	// A click (a press that did not turn the camera) moves the point there at once. The click and double-click that
	// end a press on the handle belong to the handle, so grabbing the point just after placing it is no pivot.
	canvas.addEventListener('click', (e) => {
		if (press?.handle) e.stopImmediatePropagation();
		const clicked = press && !press.handle && Math.hypot(e.clientX - press.x, e.clientY - press.y) <= CLICK_SLOP;
		if (!showing() || !clicked || e.button !== 0) return;
		pickAndExplain(e.clientX, e.clientY, true);
	}, { capture: true });
	canvas.addEventListener('dblclick', (e) => {
		if (press?.handle) e.stopImmediatePropagation();
	}, { capture: true });

	canvas.addEventListener('pointerleave', () => {
		if (dragging || !hovering) return;
		hovering = null;
		act.untip();
		state.pointed = null;
		story?.highlight(null);
		draw();
	});

	// ---- the column ------------------------------------------------------------------------------------------------

	session.addEventListener('change', ({ detail: { what } }) => {
		if (what === 'numbers') storyCase = null;
		if (what === 'active' || what === 'status' || what === 'numbers' || what === 'cases') sync();
	});
	new MutationObserver(sync).observe(host, { attributes: true, attributeFilter: ['hidden'] });
	let width = 0;
	// A hidden column has no width; its figures are drawn again when it is shown at a new one.
	new ResizeObserver(([entry]) => {
		if (!story || entry.contentRect.width === 0 || entry.contentRect.width === width) return;
		width = entry.contentRect.width;
		story.resize();
	}).observe(host);

	/** The part of the story being read draws its layers strongly; sections fade in as they arrive. */
	function observeStory() {
		const middle = new IntersectionObserver((entries) => {
			for (const entry of entries) entry.target.dataset.reading = String(entry.isIntersecting);
			const read = story.sections.find((el) => el.dataset.reading === 'true');
			act.reading(read ? read.dataset.drawn.split(' ') : null);
		}, { root: host, rootMargin: '-35% 0px -45% 0px' });
		for (const el of story.sections) middle.observe(el);
		const arrival = new IntersectionObserver((entries) => {
			for (const entry of entries) {
				if (!entry.isIntersecting) continue;
				entry.target.classList.add('arrived');
				arrival.unobserve(entry.target);
				if (entry.target.dataset.section === 'network') show(true);
			}
		}, { root: host, threshold: 0.12 });
		for (const el of story.el.querySelectorAll('.story-section')) arrival.observe(el);
	}

	engine.model.then(
		(card) => {
			state.card = card;
			state.groups = inputGroups(card.columns);
			state.layers = inputLayers(card.comparison, state.groups);
			const scales = state.groups.filter((g) => g.kind === 'bend').map((g) => g.scale);
			state.scaleShown = new Set(scales);
			story = createStory({ card, groups: state.groups, layers: state.layers, act });
			host.prepend(story.el);
			cornerCard = createCorner(corner, scales, act);
			observeStory();
			sync();
		},
		(error) => host.prepend(h('p', { class: 'model point-failure', role: 'alert' }, error.message)),
	);
}
