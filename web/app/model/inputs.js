// What a cell is told. First the invented body (`primer.js`), its groups of inputs as modes; then the open car, with
// its cell as a handle the reader drags or moves with a click, the shown group's inputs drawn on the car where they are
// read, the networks' answer there, and the car's own numbers in the body's grid.

import { h } from '../dom.js';
import { signed } from '../format.js';
import { Viewer } from '../../viewer/viewer.js';
import { COLOR } from './body-views.js';
import { carName, section } from './charts.js';
import { copied } from './copy.js';
import { locator } from './overview.js';
import { createPrimer } from './primer.js';
import { Stencil } from './stencil.js';

/**
 * A press that moves farther than this, in CSS pixels, turns the camera instead of picking a cell: a mouse's, and a
 * finger's, which wanders more in a tap.
 */
const TAP_SLOP = { mouse: 4, touch: 10 };
/** Where the page's car is seen from: ahead, to the left (−y) side and above. */
export const CAMERA = [-1, -1.3, 0.75];
/**
 * While nearby surface is shown the car is seen close to the cell: from outside, along the cell's facing tipped by
 * this toward the page's own view (and never from below), at CLOSE_VIEW of the distance that frames the car.
 */
const CLOSE_TILT = [-0.7, -0.9, 0.6];
const CLOSE_VIEW = 0.2;

/** A cell to explain: where it is, and the normal of the working-mesh vertex there. */
export function requestAt(point, vertex, normals) {
	return { point: Array.from(point), normal: Array.from(normals.subarray(3 * vertex, 3 * vertex + 3)) };
}

/**
 * @param {{ card: import('../../engine/result.js').ModelCard, groups: import('./groups.js').InputGroup[],
 *   engine: import('../../engine/result.js').Engine, ask: (request: { point: number[], normal: number[] }) => void }} options
 */
export function createInputs({ card, groups, engine, ask }) {
	const carHost = h('div', { class: 'car-view' });
	const viewer = new Viewer(carHost);
	// The page scrolls under the wheel; the car only turns.
	viewer.controls.enableZoom = false;
	viewer.controls.enablePan = false;
	const stencil = new Stencil(viewer);
	const answer = h('div', { class: 'point-answer' });
	const caption = h('p', { class: 'car-caption' });
	const carGrid = h('div', { class: 'primer-vector' });
	// `close`: whether the camera is in nearby surface's close view of the cell.
	const state = { case: null, point: null, close: false };
	const primer = createPrimer({ card, groups, engine, onShow: () => { drawMarks(); frame(); } });
	primer.carGrid(carGrid);

	const el = section('inputs', { inputs: card.columns.length },
		primer.el,
		h('div', { class: 'inputs-car' },
			h('div', { class: 'inputs-car-view' }, carHost, copied(h('p', { class: 'hint' }), 'inputs.car.hint')),
			h('div', { class: 'inputs-car-numbers' }, caption, answer, carGrid)));
	el.prepend(locator('inputs'));

	/** The shown group's marks on the car, at its shown size or station; only the handle while all are shown. */
	function drawMarks() {
		const point = state.point;
		if (!point?.explanation) {
			stencil.clear();
			return;
		}
		const shown = primer.shown;
		const group = groups.find((g) => g.words === shown.group);
		if (!group) {
			stencil.draw(point.explanation, [], COLOR.network, null);
			return;
		}
		const at = group.repeat ? { scale: shown.size, station: shown.station }[group.repeat] : null;
		stencil.draw(point.explanation, point.marks.filter((m) => group.kinds.includes(m.layer)), group.color, at);
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

	// ---- the cell on the car: click to move it, drag its handle ------------------------------------------------------

	let press = null;
	let dragging = null;
	const canvas = viewer.canvas;

	async function pickAndAsk(x, y) {
		const c = state.case;
		const hit = await viewer.pick(x, y);
		if (hit && c === state.case) ask(requestAt(hit.point, hit.vertex, c.result.mesh.normals));
	}

	/** While the handle is dragged the cell follows the pointer; moves that arrive while a pick runs collapse into the latest. */
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
		press = { x: e.clientX, y: e.clientY, slop: TAP_SLOP[e.pointerType] ?? TAP_SLOP.touch, handle: stencil.onHandle(e.clientX, e.clientY) };
		if (!press.handle) return;
		// The handle's press is the cell's: the camera never sees it.
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
	// A tap (a press that did not turn the camera) moves the cell where it lifts, for a mouse and a finger alike.
	canvas.addEventListener('pointerup', (e) => {
		if (dragging) {
			e.stopImmediatePropagation();
			canvas.releasePointerCapture(e.pointerId);
			dragging = null;
			canvas.style.cursor = 'grab';
			return;
		}
		const tapped = press && !press.handle && Math.hypot(e.clientX - press.x, e.clientY - press.y) <= press.slop;
		if (tapped && e.button === 0 && state.case) pickAndAsk(e.clientX, e.clientY);
	}, { capture: true });
	canvas.addEventListener('pointercancel', () => {
		dragging = null;
		press = null;
	}, { capture: true });
	// The click and double-click that end a press on the handle belong to the handle, so grabbing the cell just after
	// placing it is no pivot.
	canvas.addEventListener('click', (e) => {
		if (press?.handle) e.stopImmediatePropagation();
	}, { capture: true });
	canvas.addEventListener('dblclick', (e) => {
		if (press?.handle) e.stopImmediatePropagation();
	}, { capture: true });

	/**
	 * Showing nearby surface takes the camera close to the cell, and showing it again takes it to where the cell is
	 * now; any other group brings the whole car back. Dragging the cell leaves the camera where it is.
	 */
	function frame() {
		const point = state.point?.explanation;
		if (primer.shown.group === 'surface' && point) {
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

	return {
		el,
		viewer,
		stencil,
		/** A newly opened car is framed whole by the page; nearby surface's close view follows its first cell. */
		setCase(caseView) {
			if (caseView.surface !== state.case?.surface) state.close = false;
			state.case = caseView;
			// The car in clay: the pictures taken of it leave a field on.
			viewer.setPanes([{ field: 'none', map: null }]);
			copied(caption, 'inputs.car.caption', { car: () => carName(caseView.result, card) });
		},
		setPoint(point) {
			state.point = point;
			drawMarks();
			showAnswer();
			primer.setCar(point.explanation);
			if (primer.shown.group === 'surface' && !state.close) frame();
		},
	};
}

/** The networks' answer at the cell: each network's Cp as a tick, their mean, and the CFD there when the file has it. */
function answerFigure({ cpModels, cp, sigma, cfd }) {
	const values = [...cpModels, ...(cfd === null ? [] : [cfd])];
	const mid = (Math.min(...values) + Math.max(...values)) / 2;
	const half = Math.max(0.05, (Math.max(...values) - Math.min(...values)) * 0.65);
	const x = (v) => `${50 + (50 * (v - mid)) / half}%`;
	const item = (key, swatch) => h('span', { class: 'copy-block' }, h('i', { class: swatch }), copied(h('span'), key));
	const number = (key, value) => h('div', {}, copied(h('dt'), key), h('dd', { class: 'num' }, value));
	return h('div', { class: 'answer' },
		h('div', { class: 'answer-strip' },
			cpModels.map((v) => h('i', { class: 'answer-network', style: `left: ${x(v)}` })),
			h('i', { class: 'answer-mean', style: `left: ${x(cp)}` }),
			cfd === null ? null : h('i', { class: 'answer-cfd', style: `left: ${x(cfd)}` })),
		h('p', { class: 'answer-key copy-block' },
			item('inputs.car.network', 'key-network'),
			item('inputs.car.mean', 'key-mean'),
			cfd === null ? null : item('inputs.car.cfd', 'key-cfd')),
		h('dl', { class: 'answer-numbers' },
			number('inputs.car.cp', signed(cp, 3)),
			number('inputs.car.spread', sigma.toFixed(3)),
			cfd === null ? null : number('inputs.car.cfdValue', signed(cfd, 3))));
}
