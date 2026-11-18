// The flat views of the invented body: the body from the side, one slice end-on, the surface around the cell close
// up, the networks and their answer, and the 107 numbers as one grid. Each view owns an <svg>, sizes itself to its
// width in pixels (so text keeps one size) and redraws from the cell's state (`primer.js`). Lengths in metres.

import { signed } from '../format.js';
import { LENGTH, outline, profile, sideOutline } from './body.js';
import { shade, sizeShades } from './charts.js';
import { markup } from './copy.js';

const NS = 'http://www.w3.org/2000/svg';
export const COLOR = { position: '#2f739c', surface: '#a35b1a', slices: '#197b80', network: '#0b6fbf' };

export const num = (v, digits = 2) => signed(v, digits);
const metres = (v) => `${num(v)} m`;
export const cm = (dx) => `${Math.round(Math.abs(dx) * 100)} cm`;
const f1 = (v) => v.toFixed(1);
const pathOf = (points, close = false) => points.map(([x, y], i) => `${i ? 'L' : 'M'}${f1(x)} ${f1(y)}`).join('') + (close ? 'Z' : '');

/** An arrow from (x1, y1) to (x2, y2) with an open head; nothing when it is shorter than its head. */
function arrow(x1, y1, x2, y2, cls, head = 5) {
	const length = Math.hypot(x2 - x1, y2 - y1);
	if (length < 0.5) return '';
	const ux = (x2 - x1) / length;
	const uy = (y2 - y1) / length;
	const h = Math.min(head, length);
	const a = [x2 - ux * h - uy * h * 0.55, y2 - uy * h + ux * h * 0.55];
	const b = [x2 - ux * h + uy * h * 0.55, y2 - uy * h - ux * h * 0.55];
	return `<path class="${cls}" d="M${f1(x1)} ${f1(y1)}L${f1(x2)} ${f1(y2)}M${f1(a[0])} ${f1(a[1])}L${f1(x2)} ${f1(y2)}L${f1(b[0])} ${f1(b[1])}"/>`;
}

/** A dimension: a line with end ticks between two points, its text beside the middle. */
function dimension(x1, y1, x2, y2, text, cls, where = 'above') {
	const horizontal = Math.abs(y2 - y1) < Math.abs(x2 - x1);
	const t = 4;
	const ticks = horizontal ? `M${f1(x1)} ${f1(y1 - t)}V${f1(y1 + t)}M${f1(x2)} ${f1(y2 - t)}V${f1(y2 + t)}` : `M${f1(x1 - t)} ${f1(y1)}H${f1(x1 + t)}M${f1(x2 - t)} ${f1(y2)}H${f1(x2 + t)}`;
	const mx = (x1 + x2) / 2;
	const my = (y1 + y2) / 2;
	const label = horizontal
		? `<text class="dim-text" x="${f1(mx)}" y="${f1(where === 'above' ? my - 6 : my + 14)}" text-anchor="middle">${text}</text>`
		: `<text class="dim-text" x="${f1(where === 'left' ? mx - 7 : mx + 7)}" y="${f1(my + 4)}" text-anchor="${where === 'left' ? 'end' : 'start'}">${text}</text>`;
	return `<g class="${cls}"><path d="M${f1(x1)} ${f1(y1)}L${f1(x2)} ${f1(y2)}${ticks}"/>${label}</g>`;
}

/**
 * Watch an element's width; `draw(width)` runs on every change and once now. A redraw changes the element's height,
 * so it runs on the next frame, outside the observer's delivery.
 */
function sized(node, draw) {
	let last = -1;
	const run = () => {
		const width = Math.floor(node.getBoundingClientRect().width);
		if (width > 0 && width !== last) { last = width; draw(width); }
	};
	new ResizeObserver(() => requestAnimationFrame(run)).observe(node);
	run();
}

/**
 * Pointer handling for a view the cell can be moved in: a press on the handle drags it, a tap elsewhere puts it
 * there. `toWorld` maps a pointer to the view's coordinates; `pick` moves the cell.
 */
function movable(svg, toWorld, pick) {
	let dragging = false;
	const at = (e) => {
		const box = svg.getBoundingClientRect();
		return toWorld(e.clientX - box.left, e.clientY - box.top);
	};
	// A finger on the handle drags the cell rather than the page. Browsers ignore touch-action inside an svg, so the
	// touch itself is held; elsewhere on the figure a finger still scrolls the page.
	svg.addEventListener('touchstart', (e) => { if (e.target.closest('.handle')) e.preventDefault(); }, { passive: false });
	svg.addEventListener('pointerdown', (e) => {
		if (!e.target.closest('.handle')) return;
		dragging = true;
		svg.setPointerCapture(e.pointerId);
		svg.classList.add('dragging');
		e.preventDefault();
	});
	svg.addEventListener('pointermove', (e) => { if (dragging) pick(...at(e)); });
	const stop = () => { dragging = false; svg.classList.remove('dragging'); };
	svg.addEventListener('pointerup', stop);
	svg.addEventListener('pointercancel', stop);
	svg.addEventListener('click', (e) => { if (!e.target.closest('.handle') && e.target.closest('.pickable')) pick(...at(e)); });
}

/**
 * A view in `node`: an svg sized by `frame(width)` and drawn by `paint(frame, state)`. With `pick`, the cell's handle
 * drags and a tap on anything `.pickable` moves the cell, through `pick(...frame.toWorld(px, py))`.
 */
export function view(node, { frame, paint, pick = null, label }) {
	const svg = document.createElementNS(NS, 'svg');
	svg.setAttribute('role', 'img');
	svg.setAttribute('aria-label', label);
	node.append(svg);
	let current = null;
	let state = null;
	const draw = () => { if (current && state) svg.innerHTML = paint(current, state); };
	sized(node, (width) => {
		current = frame(width);
		svg.setAttribute('width', current.width);
		svg.setAttribute('height', current.height);
		svg.setAttribute('viewBox', `0 0 ${current.width} ${current.height}`);
		draw();
	});
	if (pick) movable(svg, (px, py) => current.toWorld(px, py), pick);
	return { svg, update(s) { state = s; draw(); }, redraw: draw };
}

// ---- the body from the side ----
const SIDE = sideOutline();
const NOSE = SIDE[0][0];
const TAIL = NOSE + LENGTH;
const LOWEST = Math.min(...SIDE.map(([, z]) => z));
const HIGHEST = Math.max(...SIDE.map(([, z]) => z));

/** A side view of x in `range` (metres), nose on the left, `below` metres of room under the lowest point and `ladder` pixels under that. */
export function sideFrame(width, { range, below, above = 0.12, ladder = 0, pad = 12 }) {
	const k = (width - 2 * pad) / (range[1] - range[0]);
	const z0 = LOWEST - below;
	const bodyHeight = (HIGHEST + above - z0) * k;
	return {
		width, height: Math.round(bodyHeight + 2 * pad + ladder), k, range, pad,
		X: (x) => pad + (x - range[0]) * k,
		Y: (z) => pad + (HIGHEST + above - z) * k,
		toWorld: (px, py) => [range[0] + (px - pad) / k, HIGHEST + above - (py - pad) / k],
		ladderTop: pad + bodyHeight,
	};
}

function sideBody({ X, Y }) {
	return `<path class="body pickable" d="${pathOf(SIDE.map(([x, z]) => [X(x), Y(z)]), true)}"/>`
		+ markup('text', 'inputs.side.nose', {}, `class="end-label" x="${f1(X(NOSE) - 4)}" y="${f1(Y(LOWEST + 0.42) + 4)}" text-anchor="end"`)
		+ markup('text', 'inputs.side.tail', {}, `class="end-label" x="${f1(X(TAIL) + 4)}" y="${f1(Y(LOWEST + 0.62) + 4)}"`);
}

/** The cell and its facing, as seen from the side; the handle round it takes the pointer. */
function sideCell({ X, Y }, { position: { x, z, normal } }, length = 34) {
	const cx = X(x);
	const cy = Y(z);
	return `${arrow(cx, cy, cx + normal[0] * length, cy - normal[2] * length, 'facing')}<g class="handle"><circle class="handle-area" cx="${f1(cx)}" cy="${f1(cy)}" r="22"/><circle class="cell-dot" cx="${f1(cx)}" cy="${f1(cy)}" r="5"/></g>`;
}

/** Three distances of position, drawn where they are measured: along the length below the body, and up from the lowest point. */
function sidePosition({ X, Y, range }, { position: p }) {
	const level = Y(LOWEST) + 16;
	let out = `<line class="lowest" x1="${f1(X(range[0]))}" x2="${f1(X(range[1]))}" y1="${f1(Y(LOWEST))}" y2="${f1(Y(LOWEST))}"/>`;
	out += `<line class="guide" x1="${f1(X(p.x))}" x2="${f1(X(p.x))}" y1="${f1(Y(p.z))}" y2="${f1(level)}"/>`;
	out += dimension(X(p.nose), level, X(p.x), level, metres(p.fromNose), 'dim dim-position', 'below');
	out += dimension(X(p.x), level, X(p.tail), level, metres(p.toTail), 'dim dim-position', 'below');
	const left = p.x - NOSE > LENGTH / 2;
	const dx = left ? -12 : 12;
	out += dimension(X(p.x) + dx, Y(LOWEST), X(p.x) + dx, Y(p.z), metres(p.height), 'dim dim-position', left ? 'left' : 'right');
	return out;
}

/** The surface reach of each size round the cell, the shown one strong. */
function sideRings({ X, Y, k }, { position: { x, z }, surface }, size) {
	const shades = sizeShades(COLOR.surface, surface.length);
	return surface.map((s, si) => `<circle class="ring${si === size ? ' on' : ''}" style="--shade: ${shades[si]}" cx="${f1(X(x))}" cy="${f1(Y(z))}" r="${f1(Math.max(1.5, s.reach * k))}"/>`).join('');
}

/** The stations as lines across the body, the shown one strong with its reads above and below the cell. */
function sideStations({ X, Y, range }, { position: p, stations }, shown) {
	let out = '';
	stations.forEach(({ x }, st) => {
		if (x <= range[0] || x >= range[1]) return;
		const s = x - NOSE;
		const top = s > 0 && s < LENGTH ? profile(s).top : LOWEST + 0.15;
		const bottom = s > 0 && s < LENGTH ? profile(s).bottom : LOWEST - 0.05;
		out += `<line class="station${st === shown ? ' on' : ''}" x1="${f1(X(x))}" x2="${f1(X(x))}" y1="${f1(Y(top) - 6)}" y2="${f1(Y(bottom) + 6)}"/>`;
	});
	const read = stations[shown];
	const sx = X(read.x);
	if (sx > X(range[0]) && sx < X(range[1])) {
		out += `<circle class="station-cell" cx="${f1(sx)}" cy="${f1(Y(p.z))}" r="3.5"/>`;
		out += arrow(sx, Y(p.z), sx, Y(p.z + read.above), 'read-arrow', 4) + arrow(sx, Y(p.z), sx, Y(p.z - read.below), 'read-arrow', 4);
	}
	return out;
}

/**
 * The ladder under a side view: the stations at even steps, each joined to where it is on the body and drawn as its
 * slice, the cell's place on it marked. Taps on a slot carry `data-station`.
 */
function sideLadder({ X, width, pad, ladderTop, range }, { position: p, stations }, shown) {
	const slot = (width - 2 * pad) / stations.length;
	const thumb = Math.min(slot - 3, 40);
	const top = ladderTop + 26;
	let out = '';
	stations.forEach(({ x, dx }, st) => {
		const cx = pad + slot * (st + 0.5);
		const bodyX = Math.min(Math.max(X(x), X(range[0])), X(range[1]));
		out += `<path class="fan${st === shown ? ' on' : ''}" d="M${f1(bodyX)} ${f1(ladderTop - 2)}C${f1(bodyX)} ${f1(ladderTop + 14)} ${f1(cx)} ${f1(top - 16)} ${f1(cx)} ${f1(top - 3)}"/>`;
		const ring = outline(x, 96).points;
		const scale = thumb / 2.2;
		const mid = (LOWEST + HIGHEST) / 2;
		const ty = (z) => top + thumb * 0.36 - (z - mid) * scale;
		const shape = ring.length ? `<path class="thumb" d="${pathOf(ring.map(([y, z]) => [cx + y * scale, ty(z)]), true)}"/>` : `<line class="thumb-empty" x1="${f1(cx - thumb * 0.3)}" x2="${f1(cx + thumb * 0.3)}" y1="${f1(ty(LOWEST))}" y2="${f1(ty(LOWEST))}"/>`;
		const label = String(Math.round(Math.abs(dx) * 100));
		out += `<g class="slot${st === shown ? ' on' : ''}" data-station="${st}"><rect class="slot-hit" x="${f1(cx - slot / 2)}" y="${f1(top - 4)}" width="${f1(slot)}" height="${f1(thumb * 0.72 + 30)}" rx="4"/>${shape}<circle class="thumb-cell" cx="${f1(cx + p.y * scale)}" cy="${f1(ty(p.z))}" r="1.8"/><text class="slot-label" x="${f1(cx)}" y="${f1(top + thumb * 0.72 + 20)}" text-anchor="middle">${label}</text></g>`;
	});
	const labelY = top + thumb * 0.72 + 34;
	out += markup('text', 'inputs.ladder.ahead', {}, `class="ladder-end" x="${pad}" y="${f1(labelY)}"`);
	out += markup('text', 'inputs.ladder.behind', {}, `class="ladder-end" x="${f1(width - pad)}" y="${f1(labelY)}" text-anchor="end"`);
	return out;
}
/** The ladder's height in pixels for a side view `width` wide with `count` stations. */
export const ladderHeight = (width, count) => Math.min((width - 24) / count - 3, 40) * 0.72 + 62;

/** The side view with the overlays asked for: `position`, the rings of `size`, the stations with `station` shown and its ladder. */
export function paintSide(frame, state, { position = false, size = null, station = null, ladder = false }) {
	let out = sideBody(frame);
	if (position) out += sidePosition(frame, state);
	if (size !== null) out += sideRings(frame, state, size);
	if (station !== null) out += sideStations(frame, state, station);
	if (ladder) out += sideLadder(frame, state, station);
	return out + sideCell(frame, state);
}

// ---- one slice, end-on ----
/** An end-on view seen from the front (+y to the right), `maxHeight` pixels at most. */
export function endFrame(width, maxHeight, pad = 14) {
	const yHalf = 1.15;
	const z0 = LOWEST - 0.12;
	const z1 = HIGHEST + 0.12;
	const k = Math.min((width - 2 * pad) / (2 * yHalf), (maxHeight - 2 * pad) / (z1 - z0));
	const cx = width / 2;
	return {
		width, height: Math.round((z1 - z0) * k + 2 * pad), k,
		X: (y) => cx + y * k,
		Y: (z) => pad + (z1 - z) * k,
		toWorld: (px, py) => [(px - cx) / k, z1 - (py - pad) / k],
	};
}

function endSlice({ X, Y }, x, cls) {
	const ring = outline(x, 240).points;
	return ring.length ? `<path class="${cls}" d="${pathOf(ring.map(([y, z]) => [X(y), Y(z)]), true)}"/>` : '';
}

/** The four reads of one station: its area, and how far it reaches outboard of, above and below the cell's place. */
function endReads({ X, Y }, { position: { y, z }, stations }, station) {
	const read = stations[station];
	const out = Math.sign(y || 1);
	const reach = Math.abs(y) + read.outboard;
	const side = out > 0 ? 'start' : 'end';
	const nudge = out > 0 ? 6 : -6;
	let s = `<circle class="station-cell" cx="${f1(X(y))}" cy="${f1(Y(z))}" r="4"/>`;
	s += arrow(X(y), Y(z), X(out * reach), Y(z), 'read-arrow') + arrow(X(y), Y(z), X(y), Y(z + read.above), 'read-arrow') + arrow(X(y), Y(z), X(y), Y(z - read.below), 'read-arrow');
	s += `<text class="read-text" x="${f1((X(y) + X(out * reach)) / 2)}" y="${f1(Y(z) - 7)}" text-anchor="middle">${num(read.outboard)}</text>`;
	s += `<text class="read-text" x="${f1(X(y) + nudge)}" y="${f1((Y(z) + Y(z + read.above)) / 2 + 4)}" text-anchor="${side}">${num(read.above)}</text>`;
	s += `<text class="read-text" x="${f1(X(y) + nudge)}" y="${f1((Y(z) + Y(z - read.below)) / 2 + 4)}" text-anchor="${side}">${num(read.below)}</text>`;
	s += `<text class="read-text area-text" x="${f1(X(-out * 0.45))}" y="${f1(Y(LOWEST + 0.5))}" text-anchor="middle">${num(read.area)} m²</text>`;
	return s;
}

/** The end-on view: the cell's own slice, and with `station` that station's slice and its reads; with `position`, its distance from the centre plane. */
export function paintEnd(frame, state, { position = false, station = null }) {
	const { X, Y, width } = frame;
	const { x, y, z, normal } = state.position;
	let out = `<line class="centre-plane" x1="${f1(X(0))}" x2="${f1(X(0))}" y1="${f1(Y(HIGHEST + 0.1))}" y2="${f1(Y(LOWEST - 0.1))}"/><line class="lowest" x1="8" x2="${width - 8}" y1="${f1(Y(LOWEST))}" y2="${f1(Y(LOWEST))}"/>`;
	if (station === null) out += endSlice(frame, x, 'body pickable');
	else out += endSlice(frame, x, 'own-slice') + endSlice(frame, state.stations[station].x, 'slice') + endSlice(frame, x, 'pick-area pickable');
	if (position) out += dimension(X(0), Y(z), X(y), Y(z), metres(y), 'dim dim-position', 'above');
	if (station !== null) out += endReads(frame, state, station);
	return `${out}${arrow(X(y), Y(z), X(y) + normal[1] * 30, Y(z) - normal[2] * 30, 'facing')}<g class="handle"><circle class="handle-area" cx="${f1(X(y))}" cy="${f1(Y(z))}" r="22"/><circle class="cell-dot" cx="${f1(X(y))}" cy="${f1(Y(z))}" r="5"/></g>`;
}

// ---- the surface round the cell, close up ----
const FINE_X = 49;
const ROUND = 720;
const ELEVATION = (32 * Math.PI) / 180;

/**
 * The surface within one size's reach of the cell, seen close with the cell's facing straight up and the body's
 * length across: a mesh of lines faded by the Gaussian's weight, the part ahead of the cell (smaller x) on the left.
 * Returns the projection and the line work in four layers: front and back faces, ahead and behind.
 */
function patch({ position: p, surface }, size, radius) {
	const n = p.normal;
	const { reach, sigma } = surface[size];
	// e1 along the body in the tangent plane, e2 across it; the view looks down on the tangent plane.
	let e1 = [1 - n[0] * n[0], -n[0] * n[1], -n[0] * n[2]];
	if (Math.hypot(...e1) < 0.2) e1 = [-n[0] * n[2], -n[1] * n[2], 1 - n[2] * n[2]];
	const l1 = Math.hypot(...e1);
	e1 = e1.map((v) => v / l1);
	const e2 = [n[1] * e1[2] - n[2] * e1[1], n[2] * e1[0] - n[0] * e1[2], n[0] * e1[1] - n[1] * e1[0]];
	const up = n.map((v, i) => v * Math.cos(ELEVATION) + e2[i] * Math.sin(ELEVATION));
	const into = n.map((v, i) => e2[i] * Math.cos(ELEVATION) - v * Math.sin(ELEVATION));
	const scale = radius / reach;
	const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
	const project = (d) => [dot(d, e1) * scale, -dot(d, up) * scale];
	const centre = [p.x, p.y, p.z];

	const xs = Array.from({ length: FINE_X }, (_, j) => p.x - reach + (2 * reach * j) / (FINE_X - 1));
	const rings = xs.map((x) => outline(x, ROUND).points.map(([y, z]) => [x, y, z]));
	// Where round the outline the cell is, from the outline at the cell's own x, or just inside the body for a cell at
	// its very nose or tail, where the outline closes to a point.
	const own = outline(Math.min(Math.max(p.x, NOSE + 1e-4), TAIL - 1e-4), ROUND).points;
	let start = 0;
	own.forEach(([y, z], m) => { if (Math.hypot(y - p.y, z - p.z) < Math.hypot(own[start][0] - p.y, own[start][1] - p.z)) start = m; });
	const perimeter = own.reduce((sum, q, m) => sum + Math.hypot(q[0] - own[(m + 1) % ROUND][0], q[1] - own[(m + 1) % ROUND][1]), 0);
	const roundStep = Math.max(1, Math.round(((2 * reach) / 12 / perimeter) * ROUND));

	const layers = { aheadFront: [], aheadBack: [], behindFront: [], behindBack: [] };
	const distance2 = (q) => (q[0] - centre[0]) ** 2 + (q[1] - centre[1]) ** 2 + (q[2] - centre[2]) ** 2;
	const normalAt = (j, m) => {
		const ring = rings[j];
		if (!ring.length) return [0, 0, 0];
		const a = rings[Math.min(j + 1, FINE_X - 1)][m] ?? ring[m];
		const b = rings[Math.max(j - 1, 0)][m] ?? ring[m];
		const c = ring[(m + 1) % ROUND];
		const d = ring[(m + ROUND - 1) % ROUND];
		const ts = a.map((v, i) => v - b[i]);
		const tf = c.map((v, i) => v - d[i]);
		return [tf[1] * ts[2] - tf[2] * ts[1], tf[2] * ts[0] - tf[0] * ts[2], tf[0] * ts[1] - tf[1] * ts[0]];
	};
	const add = (q1, q2, j, m) => {
		if (distance2(q1) > reach * reach || distance2(q2) > reach * reach) return;
		const mid = q1.map((v, i) => (v + q2[i]) / 2);
		const w = Math.exp(-distance2(mid) / (2 * sigma * sigma));
		const front = dot(normalAt(j, m), into) < 0;
		layers[`${mid[0] < p.x ? 'ahead' : 'behind'}${front ? 'Front' : 'Back'}`].push({
			a: project(q1.map((v, i) => v - centre[i])), b: project(q2.map((v, i) => v - centre[i])), w,
		});
	};
	// Lines across the body (one x each) and along it (one place round the outline each).
	for (let j = 0; j < FINE_X; j += 4) {
		const ring = rings[j];
		for (let m = 0; m < ring.length; m++) add(ring[m], ring[(m + 1) % ROUND], j, m);
	}
	for (let k = -Math.floor(ROUND / 2 / roundStep); k <= Math.floor(ROUND / 2 / roundStep); k++) {
		const m = (((start + k * roundStep) % ROUND) + ROUND) % ROUND;
		for (let j = 0; j < FINE_X - 1; j++) if (rings[j].length && rings[j + 1].length) add(rings[j][m], rings[j + 1][m], j, m);
	}
	return { project: (v) => [dot(v, e1), -dot(v, up)], layers };
}

/** Line work of a patch layer as paths, one per band of weight, so the faint and the strong draw apart. */
function patchPaths(segments, cx, cy, cls) {
	const bands = [[], [], [], [], []];
	for (const { a, b, w } of segments) bands[Math.min(4, Math.floor(w * 5))].push(`M${f1(cx + a[0])} ${f1(cy + a[1])}L${f1(cx + b[0])} ${f1(cy + b[1])}`);
	return bands.map((d, i) => (d.length ? `<path class="${cls}" style="opacity: ${(0.14 + 0.86 * ((i + 0.5) / 5)).toFixed(2)}" d="${d.join('')}"/>` : '')).join('');
}

export function patchFrame(width, height) {
	return { width, height, radius: Math.min(width * 0.42, height * 0.42), cx: width / 2, cy: height * 0.54 };
}

/** The surface within reach of `size` close up, with the cell's facing and the mean facing of `part` (0 whole, 1 ahead, 2 behind). */
export function paintPatch(frame, state, size, part) {
	const { radius, cx, cy, height } = frame;
	const { project, layers } = patch(state, size, radius);
	const dimAhead = part === 2 ? ' dim' : '';
	const dimBehind = part === 1 ? ' dim' : '';
	let out = `<circle class="reach" cx="${f1(cx)}" cy="${f1(cy)}" r="${f1(radius)}"/>`;
	out += patchPaths(layers.aheadBack, cx, cy, `patch-line back${dimAhead}`) + patchPaths(layers.behindBack, cx, cy, `patch-line back${dimBehind}`);
	out += patchPaths(layers.aheadFront, cx, cy, `patch-line${dimAhead}`) + patchPaths(layers.behindFront, cx, cy, `patch-line${dimBehind}`);
	const length = radius * 0.72;
	const [fx, fy] = project(state.position.normal);
	const [mx, my] = project(state.surface[size].parts[part].facing);
	out += arrow(cx, cy, cx + fx * length, cy + fy * length, 'facing', 6);
	out += arrow(cx, cy, cx + mx * length, cy + my * length, 'mean-facing', 7);
	out += `<circle class="cell-dot" cx="${f1(cx)}" cy="${f1(cy)}" r="4.5"/>`;
	out += markup('text', 'inputs.patch.ahead', {}, `class="patch-side" x="${f1(cx - radius)}" y="${f1(height - 4)}"`);
	out += markup('text', 'inputs.patch.behind', {}, `class="patch-side" x="${f1(cx + radius)}" y="${f1(height - 4)}" text-anchor="end"`);
	out += `<text class="patch-size" x="${f1(cx)}" y="${f1(height - 4)}" text-anchor="middle">${cm(state.surface[size].scale)}</text>`;
	return out;
}

// ---- the networks ----
/**
 * The networks: cards of their layers, each giving the cell one pressure coefficient, and their answers on one line
 * with their mean. `layers` are a network's widths, inputs first.
 */
export function networkMarkup(width, { cps, cp }, layers) {
	const pad = 14;
	const cw = Math.min(width - 2 * pad - 60, 280);
	const ch = 64;
	const x0 = (width - cw) / 2 - 14;
	const top = 16;
	let out = '';
	// The cards, the front one drawn with its layers, the others behind it.
	for (let k = cps.length - 1; k >= 0; k--) out += `<rect class="net-card" x="${f1(x0 + k * 5)}" y="${f1(top - k * 4)}" width="${f1(cw)}" height="${ch}" rx="6"/>`;
	const cols = layers.map((_, i) => x0 + 22 + (i * (cw - 44)) / (layers.length - 1));
	const dots = (n) => Math.min(n, 9);
	const dy = (j, count) => top + ch / 2 + (j - (count - 1) / 2) * 5.6;
	layers.forEach((n, i) => {
		if (i === layers.length - 1) return;
		for (let j = 0; j < dots(n); j++) for (let m = 0; m < dots(layers[i + 1]); m++) {
			out += `<line class="net-edge" x1="${f1(cols[i])}" y1="${f1(dy(j, dots(n)))}" x2="${f1(cols[i + 1])}" y2="${f1(dy(m, dots(layers[i + 1])))}"/>`;
		}
	});
	layers.forEach((n, i) => {
		for (let j = 0; j < dots(n); j++) out += `<circle class="net-node" cx="${f1(cols[i])}" cy="${f1(dy(j, dots(n)))}" r="1.9"/>`;
		out += `<text class="net-count" x="${f1(cols[i])}" y="${top + ch + 15}" text-anchor="middle">${n}</text>`;
	});
	out += `<text class="net-times" x="${f1(x0 + cw + 28)}" y="${f1(top + ch / 2 - 4)}">×${cps.length}</text>`;
	// The answer: −1 … 1 with each network's value as a tick and their mean as a dot.
	const lo = -1.2;
	const hi = 1.2;
	const ax = (v) => pad + ((Math.min(Math.max(v, lo), hi) - lo) / (hi - lo)) * (width - 2 * pad);
	const line = top + ch + 64;
	out += `<line class="answer-axis" x1="${pad}" x2="${width - pad}" y1="${line}" y2="${line}"/>`;
	for (const v of [-1, 0, 1]) out += `<line class="answer-grid" x1="${f1(ax(v))}" x2="${f1(ax(v))}" y1="${line - 3}" y2="${line + 3}"/><text class="answer-tick" x="${f1(ax(v))}" y="${line + 17}" text-anchor="middle">${num(v, 0)}</text>`;
	for (const v of cps) out += `<line class="answer-network" x1="${f1(ax(v))}" x2="${f1(ax(v))}" y1="${line - 9}" y2="${line + 9}"/>`;
	out += `<circle class="answer-mean" cx="${f1(ax(cp))}" cy="${line}" r="6"/>`;
	out += markup('text', 'inputs.answer.value', { cp: num(cp) }, `class="answer-value" x="${f1(Math.min(Math.max(ax(cp), pad + 30), width - pad - 30))}" y="${line - 15}" text-anchor="middle"`);
	const key = line + 40;
	const kx = width / 2 - 92;
	out += `<line class="answer-network" x1="${f1(kx)}" x2="${f1(kx)}" y1="${key - 10}" y2="${key + 2}"/>${markup('text', 'inputs.answer.network', {}, `class="answer-key" x="${f1(kx + 7)}" y="${key}"`)}`;
	out += `<circle class="answer-mean" cx="${f1(kx + 104)}" cy="${key - 4}" r="5"/>${markup('text', 'inputs.answer.mean', {}, `class="answer-key" x="${f1(kx + 114)}" y="${key}"`)}`;
	return { markup: out, height: key + 8 };
}

// ---- the 107 numbers ----
/**
 * The numbers as one grid of `cell`-pixel squares, a block per group: position (two columns: distances, facing),
 * nearby surface (a row per size, a column per read), slices (a row per read, a column per station, ahead on the
 * left). `places` gives each column's block, column and row; `blocks` each block's group and width in columns.
 */
export function vectorLayout(places, blocks, cell, gap = 2, blockGap = 10) {
	const top = 16;
	const step = cell + gap;
	let x = 0;
	const origin = {};
	for (const block of blocks) { origin[block.group] = x; x += block.cols * step - gap + blockGap; }
	return {
		width: x - blockGap, height: top + 4 * step - gap, origin, blocks,
		squares: places.map((c) => ({ x: origin[c.group] + c.col * step, y: top + c.row * step, size: cell, group: c.group })),
	};
}

/** The squares of `layout`, each shaded by the number as the first network reads it (`standard`), the shown ones outlined. */
export function vectorSquares(layout, standard, shown) {
	let out = '';
	for (const block of layout.blocks) out += `<text class="block-name" style="fill: ${COLOR[block.group]}" x="${f1(layout.origin[block.group])}" y="11">${block.count}</text>`;
	layout.squares.forEach((sq, i) => {
		out += `<rect class="square${shown.has(i) ? ' selected' : ''}" data-col="${i}" data-group="${sq.group}" x="${f1(sq.x)}" y="${f1(sq.y)}" width="${f1(sq.size)}" height="${f1(sq.size)}" rx="1.5" style="fill: ${shade(standard[i], 1.5)}"/>`;
	});
	return out;
}
