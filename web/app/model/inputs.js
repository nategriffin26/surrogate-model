// What a point is told, group by group. On the left the open car, with the point as a handle the reader drags or
// moves with a click, the chosen group's inputs drawn on the car where they are read, and the networks' answer there.
// On the right a row per group: its numbers at the point on their ranges, its reason, and the accuracy it added as the
// staircase of compared networks added it. A group that reads the same things at several stations or sizes shows one
// of them at a time, on the car and as numbers.

import { h } from '../dom.js';
import { signed } from '../format.js';
import { Viewer } from '../../viewer/viewer.js';
import { section } from './charts.js';
import { drawGroup } from './numbers.js';
import { locator } from './overview.js';
import { Stencil } from './stencil.js';

/** A press that moves farther than this, in CSS pixels, turns the camera instead of picking a point. */
const CLICK_SLOP = 4;
/** The drawings' width at most, in CSS pixels. */
const DRAWING = 320;
/** Where the page's car is seen from: ahead, to the left (−y) side and above. */
export const CAMERA = [-1, -1.3, 0.75];
/**
 * While nearby surface is chosen the car is seen close to the point: from outside, along the point's facing tipped
 * by this toward the page's own view (and never from below), at CLOSE_VIEW of the distance that frames the car.
 */
const CLOSE_TILT = [-0.7, -0.9, 0.6];
const CLOSE_VIEW = 0.2;

/** A point to explain: where it is, and the normal of the working-mesh vertex there. */
export function requestAt(point, vertex, normals) {
	return { point: Array.from(point), normal: Array.from(normals.subarray(3 * vertex, 3 * vertex + 3)) };
}

/**
 * @param {{ card: import('../../engine/result.js').ModelCard, groups: import('./groups.js').InputGroup[],
 *   ask: (request: { point: number[], normal: number[] }) => void }} options
 */
export function createInputs({ card, groups, ask }) {
	const carHost = h('div', { class: 'car-view' });
	const viewer = new Viewer(carHost);
	// The page scrolls under the wheel; the car only turns.
	viewer.controls.enableZoom = false;
	viewer.controls.enablePan = false;
	const stencil = new Stencil(viewer);
	const answer = h('div', { class: 'point-answer' });
	// `close`: whether the camera is in nearby surface's close view of the point. `shown`: per repeating group, which of
	// its stations or sizes is shown; at first the point's own station, or the smallest size.
	const state = {
		group: groups[0], case: null, point: null, width: DRAWING, close: false,
		shown: Object.fromEntries(groups.filter((g) => g.repeat).map((g) => [g.id, Math.max(0, g.repeat.values.indexOf(0))])),
	};

	const rows = groups.map((group) => {
		const drawings = h('div', { class: 'group-drawings' });
		const step = group.step;
		const el = h('div', {
			class: 'group-row', role: 'radio', tabindex: 0, 'aria-checked': 'false', style: `--group: ${group.color}`,
			onclick: () => choose(group),
			onkeydown: (e) => {
				// Keys on the row's own buttons are theirs.
				if (e.target !== e.currentTarget || (e.key !== 'Enter' && e.key !== ' ')) return;
				e.preventDefault();
				choose(group);
			},
		},
		drawings,
		h('div', { class: 'group-text' },
			h('h3', {}, h('i', { class: 'swatch' }), group.name, h('span', { class: 'group-count num' }, `${group.columns.length} inputs`)),
			h('p', { class: 'group-reason' }, group.reason),
			step ? h('div', { class: 'group-step' },
				h('span', { class: 'step-track' },
					h('span', { class: 'step-before', style: `width: ${100 * Math.max(0, step.before)}%` }),
					h('span', { class: 'step-gain', style: `left: ${100 * Math.max(0, step.before)}%; width: ${100 * (step.after - Math.max(0, step.before))}%` })),
				h('span', { class: 'step-value num' }, h('b', {}, `+${(step.after - step.before).toFixed(2)}`), ` R² ${step.after.toFixed(2)}`)) : null));
		return { group, el, drawings };
	});
	const list = h('div', { class: 'group-list', role: 'radiogroup', 'aria-label': 'Groups of inputs' },
		h('div', { class: 'group-list-head' }, h('span'), h('span', { class: 'step-head' }, 'Cross-validated R² as each group is added')),
		rows.map((row) => row.el));

	const el = section('inputs', 'What a point is told', null,
		h('div', { class: 'inputs-layout' },
			h('div', { class: 'inputs-car' }, carHost, h('p', { class: 'hint' }, 'Drag the point, or click anywhere on the car'), answer),
			list));
	el.prepend(locator('inputs'));

	function choose(group) {
		state.group = group;
		for (const row of rows) row.el.setAttribute('aria-checked', String(row.group === group));
		drawMarks();
		frame();
	}

	function drawMarks() {
		const point = state.point;
		if (!point?.explanation) {
			stencil.clear();
			return;
		}
		const { repeat, id, kinds, color } = state.group;
		stencil.draw(point.explanation, point.marks.filter((m) => kinds.includes(m.layer)), color, repeat ? repeat.values[state.shown[id]] : null);
	}

	function drawRows() {
		const explanation = state.point?.explanation;
		if (!explanation || !state.case) return;
		for (const { group, drawings } of rows) {
			drawings.replaceChildren(...drawGroup(group, {
				width: state.width, explanation, columns: card.columns, shown: state.shown[group.id] ?? null,
				// The row's own click then chooses the group, which draws the shown one on the car. A stepper button that
				// had the keyboard keeps it in the new drawing.
				show: (k) => {
					const focused = drawings.contains(document.activeElement) ? document.activeElement.getAttribute('aria-label') : null;
					state.shown[group.id] = k;
					drawRows();
					if (focused) drawings.querySelector(`[aria-label="${focused}"]`)?.focus();
				},
			}));
		}
	}

	function showAnswer() {
		const point = state.point;
		if (!point) return;
		if (point.failure) {
			answer.replaceChildren(h('p', { class: 'point-failure', role: 'alert' }, point.failure));
			return;
		}
		answer.replaceChildren(answerFigure(point.explanation));
	}

	new ResizeObserver(([entry]) => {
		const width = Math.min(DRAWING, Math.floor(entry.contentRect.width));
		if (width <= 0 || width === state.width) return;
		state.width = width;
		drawRows();
	}).observe(rows[0].drawings);

	// ---- the point on the car: click to move it, drag its handle ------------------------------------------------------

	let press = null;
	let dragging = null;
	const canvas = viewer.canvas;

	async function pickAndAsk(x, y) {
		const c = state.case;
		const hit = await viewer.pick(x, y);
		if (hit && c === state.case) ask(requestAt(hit.point, hit.vertex, c.result.mesh.normals));
	}

	/** While the handle is dragged the point follows the pointer; moves that arrive while a pick runs collapse into the latest. */
	async function follow(x, y) {
		dragging.next = [x, y];
		if (dragging.picking) return;
		dragging.picking = true;
		while (dragging?.next) {
			const [px, py] = dragging.next;
			dragging.next = null;
			await pickAndAsk(px, py);
		}
		if (dragging) dragging.picking = false;
	}

	canvas.addEventListener('pointerdown', (e) => {
		if (!state.case || e.button !== 0) return;
		press = { x: e.clientX, y: e.clientY, handle: stencil.onHandle(e.clientX, e.clientY) };
		if (!press.handle) return;
		// The handle's press is the point's: the camera never sees it.
		e.stopImmediatePropagation();
		e.preventDefault();
		canvas.setPointerCapture(e.pointerId);
		dragging = { next: null, picking: false };
		canvas.style.cursor = 'grabbing';
	}, { capture: true });
	canvas.addEventListener('pointermove', (e) => {
		if (dragging) {
			e.stopImmediatePropagation();
			follow(e.clientX, e.clientY);
		} else if (e.buttons === 0) {
			canvas.style.cursor = stencil.onHandle(e.clientX, e.clientY) ? 'grab' : '';
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
	// A click (a press that did not turn the camera) moves the point there. The click and double-click that end a
	// press on the handle belong to the handle, so grabbing the point just after placing it is no pivot.
	canvas.addEventListener('click', (e) => {
		if (press?.handle) e.stopImmediatePropagation();
		const clicked = press && !press.handle && Math.hypot(e.clientX - press.x, e.clientY - press.y) <= CLICK_SLOP;
		if (clicked && e.button === 0 && state.case) pickAndAsk(e.clientX, e.clientY);
	}, { capture: true });
	canvas.addEventListener('dblclick', (e) => {
		if (press?.handle) e.stopImmediatePropagation();
	}, { capture: true });

	/**
	 * Choosing nearby surface takes the camera close to the point, and choosing it again takes it to where the point
	 * is now; any other group brings the whole car back. Dragging the point leaves the camera where it is.
	 */
	function frame() {
		const point = state.point?.explanation;
		if (state.group.id === 'surface' && point) {
			const n = point.normal;
			const whole = viewer.framing([n[0] + CLOSE_TILT[0], n[1] + CLOSE_TILT[1], Math.max(n[2], 0.2) + CLOSE_TILT[2]]);
			const offset = whole.position.clone().sub(whole.target).multiplyScalar(CLOSE_VIEW);
			const target = whole.target.clone().set(...point.position);
			viewer.moveTo({ position: target.clone().add(offset), target, zoom: viewer.camera.isOrthographicCamera ? whole.zoom / CLOSE_VIEW : 1 }, true);
			state.close = true;
		} else if (state.close) {
			viewer.moveTo(viewer.framing(CAMERA), true);
			state.close = false;
		}
	}

	choose(state.group);

	return {
		el,
		viewer,
		stencil,
		/** A newly opened car is framed whole by the page; nearby surface's close view follows its first point. */
		setCase(caseView) {
			if (caseView.surface !== state.case?.surface) state.close = false;
			state.case = caseView;
			// The car in clay: the pictures taken of it leave a field on.
			viewer.setPanes([{ field: 'none', map: null }]);
			drawRows();
		},
		setPoint(point) {
			state.point = point;
			drawMarks();
			drawRows();
			showAnswer();
			if (state.group.id === 'surface' && !state.close) frame();
		},
	};
}

/** The networks' answer at the point: each network's Cp as a tick, their mean, and the CFD there when the file has it. */
function answerFigure({ cpModels, cp, sigma, cfd }) {
	const values = [...cpModels, ...(cfd === null ? [] : [cfd])];
	const mid = (Math.min(...values) + Math.max(...values)) / 2;
	const half = Math.max(0.05, (Math.max(...values) - Math.min(...values)) * 0.65);
	const x = (v) => `${50 + (50 * (v - mid)) / half}%`;
	return h('div', { class: 'answer' },
		h('div', { class: 'answer-strip' },
			cpModels.map((v, k) => h('i', { class: 'answer-network', style: `left: ${x(v)}`, title: `Network ${k + 1}` })),
			h('i', { class: 'answer-mean', style: `left: ${x(cp)}` }),
			cfd === null ? null : h('i', { class: 'answer-cfd', style: `left: ${x(cfd)}` })),
		h('p', { class: 'answer-key' },
			h('span', {}, h('i', { class: 'key-network' }), 'each network'),
			h('span', {}, h('i', { class: 'key-mean' }), 'their mean'),
			cfd === null ? null : h('span', {}, h('i', { class: 'key-cfd' }), 'CFD')),
		h('dl', { class: 'answer-numbers' },
			h('div', {}, h('dt', {}, 'Cp'), h('dd', { class: 'num' }, signed(cp, 3))),
			h('div', {}, h('dt', {}, 'Spread'), h('dd', { class: 'num' }, sigma.toFixed(3))),
			cfd === null ? null : h('div', {}, h('dt', {}, 'CFD'), h('dd', { class: 'num' }, signed(cfd, 3)))));
}
