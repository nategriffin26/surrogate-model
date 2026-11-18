// What a cell is told, taught on an invented body: one figure, the groups of inputs as its modes. The side view
// carries the shown group's measurements, the panel beside it shows them close (end-on, the surface close up, or the
// networks), and the numbers stay whole underneath with the shown group's squares lit and listed. The body is measured
// and predicted by the engine like any car given as arrays; every number here is the engine's explanation at the
// body's cell, read through the model's own columns.

import { h } from '../dom.js';
import { aroundLookup, bodySurface, sideLookup } from './body.js';
import {
	cm, endFrame, ladderHeight, networkMarkup, num, paintEnd, paintPatch, paintSide, patchFrame, sideFrame, vectorCell, vectorLayout,
	vectorSquares, view,
} from './body-views.js';
import { shade } from './charts.js';
import { copied } from './copy.js';
import { COLORS } from './groups.js';

/** The modes, in the figure's order: a kind of input each, then all of them. */
const MODES = ['position', 'surface', 'slice', 'all'];
/** The words of position's seven numbers, in column order. */
const POSITION_WORDS = ['fromNose', 'toTail', 'fromCentre', 'height', 'facingRearward', 'facingSideways', 'facingUp'];
/** The words of one part of nearby surface's four numbers, in column order. */
const SURFACE_WORDS = ['facingRearward', 'facingSideways', 'facingUp', 'amount'];
const PARTS = ['whole', 'ahead', 'behind'];
/** A slice's reads, in the grid's rows from the top. */
const READS = ['area', 'outboard', 'above', 'below'];
/** Where the side view starts the cell, in metres along and up the body. */
const START = [0.85, 0.5];
/** The side view's span along the body, and the room under it, in metres. */
const SIDE = { range: [-1.5, 4.5], below: 0.3 };
const detailHeight = (width) => (width < 420 ? 230 : 260);

/**
 * The model's columns arranged as the figure reads them: each column's group (its kind), column and row in the grid;
 * the sizes of nearby surface and the stations of the slices, ascending; and the columns of each.
 * @param {import('../../engine/result.js').ModelColumn[]} columns
 * @param {import('./groups.js').InputGroup[]} groups
 */
function arrange(columns, groups) {
	const of = (kind) => groups.find((g) => g.id === kind).columns;
	const scales = [...new Set(of('surface').map((i) => columns[i].scale))].sort((a, b) => a - b);
	const stations = [...new Set(of('slice').map((i) => columns[i].station))].sort((a, b) => a - b);
	const surface = scales.map((scale) => PARTS.map((part) => of('surface').filter((i) => columns[i].scale === scale && columns[i].part === part)));
	const slices = stations.map((station) => READS.map((read) => of('slice').find((i) => columns[i].station === station && columns[i].read === read)));
	if (surface.flat().some((part) => part.length !== SURFACE_WORDS.length)) throw new Error('The model reads nearby surface in a way the page cannot draw.');
	const places = columns.map((column) => ({ group: column.kind }));
	// Position: its distances (measured along a direction) in one column, its facing in the next.
	of('position').filter((i) => columns[i].along).forEach((i, row) => Object.assign(places[i], { col: 0, row }));
	of('position').filter((i) => !columns[i].along).forEach((i, row) => Object.assign(places[i], { col: 1, row }));
	surface.forEach((parts, row) => parts.forEach((part, p) => part.forEach((i, r) => Object.assign(places[i], { col: SURFACE_WORDS.length * p + r, row }))));
	slices.forEach((reads, col) => reads.forEach((i, row) => Object.assign(places[i], { col, row })));
	const blocks = [
		{ group: 'position', cols: 2, count: of('position').length },
		{ group: 'surface', cols: SURFACE_WORDS.length * PARTS.length, count: of('surface').length },
		{ group: 'slice', cols: stations.length, count: of('slice').length },
	];
	return { position: of('position'), scales, stations, surface, slices, places, blocks, sigma: (k) => columns[surface[k][0][0]].sigma };
}

/**
 * A cell as the views draw it, from the engine's explanation there: where it is and its distances, the surface
 * around it per size and part (reaching `truncate` standard deviations), each station's reads, the numbers as the
 * first network reads them, and the answers.
 */
function cellState(arrangement, truncate, { position, normal, features, activations, cpModels, cp }) {
	const [fromNose, toTail, fromCentre, height] = arrangement.position.map((i) => features[i]);
	return {
		position: { x: position[0], y: fromCentre, z: position[2], normal, fromNose, toTail, height, nose: position[0] - fromNose, tail: position[0] + toTail },
		surface: arrangement.scales.map((scale, k) => ({
			scale, sigma: arrangement.sigma(k), reach: truncate * arrangement.sigma(k),
			parts: arrangement.surface[k].map((part) => ({ facing: part.slice(0, 3).map((i) => features[i]), amount: features[part[3]] })),
		})),
		stations: arrangement.stations.map((dx, s) => {
			const [area, outboard, above, below] = arrangement.slices[s].map((i) => features[i]);
			return { dx, x: position[0] + dx, area, outboard, above, below };
		}),
		values: features, standard: activations[0][0], cps: cpModels, cp,
	};
}

/** A number with its unit, as the readout lists it. */
const valueText = (column, v) => (column.unit ? `${num(v)} ${column.unit}` : num(v));

/**
 * The figure, and a grid of the reference car's numbers laid out as the body's. `onShow` hears when the reader shows
 * another group, size or station, to draw the same on the car.
 * @param {{ card: import('../../engine/result.js').ModelCard, groups: import('./groups.js').InputGroup[],
 *   engine: import('../../engine/result.js').Engine, onShow: () => void }} options
 */
export function createPrimer({ card, groups, engine, onShow }) {
	const arrangement = arrange(card.columns, groups);
	const { scales, stations } = arrangement;
	// The figure was designed to start on the second size and the fourth station ahead of the cell's own.
	const ui = { mode: 0, size: 1, part: 0, station: stations.indexOf(0) - 4 };
	const body = { result: null, side: null, vertex: -1, state: null, asking: false, next: -1 };
	const modeButtons = MODES.map((mode, i) => h('button', {
		type: 'button', role: 'tab', style: `--group: ${mode === 'all' ? COLORS.network : COLORS[mode]}`, onclick: () => { ui.mode = i; repaint(); },
	},
	mode === 'all' ? null : h('i', { class: 'swatch' }),
	copied(h('span'), `inputs.modes.${mode}`),
	mode === 'all' ? null : h('b', { class: 'num' }, arrangement.blocks[i].count)));

	const choice = (labels, onPick) => {
		const buttons = labels.map((label, i) => h('button', { type: 'button', onclick: () => { onPick(i); repaint(); } }, label));
		return { el: h('div', { class: 'choice', role: 'group' }, buttons), press: (index) => buttons.forEach((b, i) => b.setAttribute('aria-pressed', String(i === index))) };
	};
	const sizes = choice(scales.map(cm), (i) => { ui.size = i; });
	const parts = choice(PARTS.map((part) => copied(h('span'), `inputs.parts.${part}`)), (i) => { ui.part = i; });
	const stationName = h('output', { class: 'station-name' });
	const step = (by, label) => h('button', {
		type: 'button', 'aria-label': label, onclick: () => { ui.station = Math.min(Math.max(ui.station + by, 0), stations.length - 1); repaint(); },
	}, by < 0 ? '◂' : '▸');
	const ahead = step(-1, 'Station further ahead');
	const behind = step(1, 'Station further behind');
	const note = (mode, values) => copied(h('p', { class: 'primer-note' }), `inputs.notes.${mode}`, values);
	const controls = {
		position: h('div', { class: 'primer-control' }, note('position')),
		surface: h('div', { class: 'primer-control' }, sizes.el, parts.el, note('surface', { sizes: scales.length })),
		slice: h('div', { class: 'primer-control' }, h('div', { class: 'stepper' }, ahead, stationName, behind),
			note('slice', { ahead: stations.filter((dx) => dx < 0).length, behind: stations.filter((dx) => dx > 0).length })),
		all: h('div', { class: 'primer-control' }, note('all')),
	};

	const sideHost = h('div', { class: 'primer-figure' });
	const detailHost = h('div', { class: 'primer-figure primer-detail' });
	const vectorHost = h('div', { class: 'primer-vector' });
	const carHost = h('div', { class: 'primer-vector' });
	const readout = h('dl', { class: 'primer-readout' });
	const failure = h('p', { class: 'point-failure', role: 'alert', hidden: true });
	const el = h('div', { class: 'inputs-primer' },
		h('div', { class: 'primer-modes', role: 'tablist' }, modeButtons),
		h('div', { class: 'primer-controls' }, Object.values(controls)),
		failure,
		h('div', { class: 'primer-stage' }, sideHost, detailHost),
		h('div', { class: 'primer-bottom' },
			h('div', {}, vectorHost, h('p', { class: 'colour-key copy-block' }, h('i', { class: 'ramp' }), copied(h('span'), 'inputs.key'))),
			readout));

	const pickSide = (x, z) => move(body.side(x, z));
	const side = view(sideHost, {
		label: 'The invented body from the side, and the cell',
		frame: (w) => sideFrame(w, { ...SIDE, ladder: ladderHeight(w, stations.length) }),
		paint: (f, s) => {
			const mode = MODES[ui.mode];
			return paintSide(f, s, {
				position: mode === 'position' || mode === 'all',
				size: mode === 'surface' || mode === 'all' ? ui.size : null,
				station: mode === 'slice' || mode === 'all' ? ui.station : null,
				ladder: mode === 'slice',
			});
		},
		pick: (x, z) => { if (body.side) pickSide(x, z); },
	});
	side.svg.addEventListener('click', (e) => {
		const slot = e.target.closest('.slot');
		if (slot) { ui.station = Number(slot.dataset.station); repaint(); }
	});
	const detail = view(detailHost, {
		label: 'The shown inputs, close',
		frame: (w) => ({ ...endFrame(w, detailHeight(w)), height: detailHeight(w), patch: patchFrame(w, detailHeight(w)) }),
		paint: (f, s) => {
			const mode = MODES[ui.mode];
			if (mode === 'position') return paintEnd(f, s, { position: true });
			if (mode === 'surface') return paintPatch(f.patch, s, ui.size, ui.part);
			if (mode === 'slice') return paintEnd(f, s, { station: ui.station });
			const net = networkMarkup(f.width, s, card.networks[0].layers);
			return `<g transform="translate(0 ${((f.height - net.height) / 2).toFixed(1)})">${net.markup}</g>`;
		},
		pick: (y, z) => {
			if (MODES[ui.mode] === 'position' || MODES[ui.mode] === 'slice') move(aroundLookup(body.result.mesh.positions, body.vertex, y, z));
		},
	});
	const bodyGrid = numbersGrid(vectorHost, 'The invented body\'s numbers');
	const carGrid = numbersGrid(carHost, 'The reference car\'s numbers');

	/** A grid of the numbers in `node`, the shown group lit; a tap on a square shows its group. */
	function numbersGrid(node, label) {
		const grid = view(node, {
			label,
			frame: (w) => {
				const layout = vectorLayout(arrangement.places, arrangement.blocks, vectorCell(w, arrangement.blocks));
				return { width: w, height: layout.height + 2, layout };
			},
			paint: (f, s) => `<g transform="translate(${((f.width - f.layout.width) / 2).toFixed(1)} 0)" data-mode="${MODES[ui.mode]}">${vectorSquares(f.layout, s.standard, new Set(listed().map(([i]) => i)))}</g>`,
		});
		grid.svg.addEventListener('click', (e) => {
			const square = e.target.closest('[data-col]');
			if (square) show(Number(square.dataset.col));
		});
		return grid;
	}

	/** The numbers the readout lists, with their words: the shown group's at the shown size and part, or station. */
	function listed() {
		const mode = MODES[ui.mode];
		if (mode === 'position') return arrangement.position.map((i, k) => [i, POSITION_WORDS[k]]);
		if (mode === 'surface') return arrangement.surface[ui.size][ui.part].map((i, r) => [i, SURFACE_WORDS[r]]);
		if (mode === 'slice') return arrangement.slices[ui.station].map((i, r) => [i, READS[r]]);
		return [];
	}

	/** The readout: the body's listed numbers, or for all of them each group's count and the answer. */
	function paintReadout() {
		const rows = listed();
		if (!rows.length) {
			readout.replaceChildren(
				...arrangement.blocks.map((block, i) => h('div', {},
					h('dt', {}, h('i', { class: 'swatch', style: `--group: ${COLORS[block.group]}` }), copied(h('span'), `inputs.modes.${MODES[i]}`)),
					h('dd', { class: 'num' }, block.count))),
				h('div', {}, copied(h('dt'), 'inputs.readout.cp'), h('dd', { class: 'num' }, body.state ? num(body.state.cp) : '')));
			return;
		}
		const value = (i) => (body.state
			? h('dd', { class: 'num' }, h('i', { class: 'swatch', style: `--group: ${shade(body.state.standard[i], 1.5)}` }), valueText(card.columns[i], body.state.values[i]))
			: h('dd'));
		readout.replaceChildren(...rows.map(([i, words]) => h('div', {}, copied(h('dt'), `inputs.names.${words}`), value(i))));
	}

	/** Draw the figure for what is shown; `told` tells the page too, for the car. */
	function repaint(told = true) {
		const mode = MODES[ui.mode];
		el.dataset.mode = mode;
		modeButtons.forEach((b, i) => b.setAttribute('aria-selected', String(i === ui.mode)));
		// The controls of every mode share one place, the others inert and unseen, so the figure keeps the height of the
		// tallest at any width and with any words.
		for (const [id, control] of Object.entries(controls)) control.inert = id !== mode;
		sizes.press(ui.size);
		parts.press(ui.part);
		ahead.disabled = ui.station === 0;
		behind.disabled = ui.station === stations.length - 1;
		const dx = stations[ui.station];
		copied(stationName, dx === 0 ? 'inputs.station.own' : dx < 0 ? 'inputs.station.ahead' : 'inputs.station.behind', { distance: cm(dx) });
		side.redraw();
		detail.redraw();
		bodyGrid.redraw();
		carGrid.redraw();
		paintReadout();
		if (told) onShow();
	}

	/** Show the group of a tapped square, at its size and part, or its station. */
	function show(i) {
		const place = arrangement.places[i];
		ui.mode = MODES.indexOf(place.group);
		if (place.group === 'surface') { ui.size = place.row; ui.part = Math.floor(place.col / SURFACE_WORDS.length); }
		if (place.group === 'slice') ui.station = place.col;
		repaint();
	}

	function fail(error) {
		failure.textContent = error.message;
		failure.hidden = false;
	}

	/** Move the body's cell to vertex `v`; moves that arrive while the engine answers collapse into the latest. */
	async function move(v) {
		if (v < 0 || v === body.vertex) return;
		body.next = v;
		if (body.asking) return;
		body.asking = true;
		try {
			while (body.next >= 0) {
				const vertex = body.next;
				body.next = -1;
				const { positions, normals } = body.result.mesh;
				const at = (array) => Array.from(array.subarray(3 * vertex, 3 * vertex + 3));
				const explanation = await engine.explain(body.result, at(positions), at(normals));
				body.vertex = vertex;
				body.state = cellState(arrangement, body.result.lookups.definition.gaussianTruncate, explanation);
				side.update(body.state);
				detail.update(body.state);
				bodyGrid.update(body.state);
				paintReadout();
			}
		} catch (error) {
			fail(error);
		}
		body.asking = false;
	}

	repaint(false);
	// The engine measures and predicts the body once, after any car it is busy with.
	engine.predictQuick(bodySurface()).then((result) => {
		body.result = result;
		body.side = sideLookup(result.mesh.positions);
		pickSide(...START);
	}, fail);

	return {
		el,
		/** The shown kind of input (null for all of them), and its shown size and station, in metres. */
		get shown() {
			const mode = MODES[ui.mode];
			return { group: mode === 'all' ? null : mode, size: scales[ui.size], station: stations[ui.station] };
		},
		/** The grid of the reference car's numbers, laid out as the body's; `setCar` fills it. */
		carGrid: carHost,
		/** The reference car's numbers at its cell, for the car's grid. */
		setCar(explanation) {
			if (!explanation) return;
			carGrid.update(cellState(arrangement, 0, explanation));
		},
	};
}
