// The section tool: a plane normal to x, y or z, placed by typing or by dragging its line on the car; its line drawn
// on the car, the car optionally clipped at it, and the shown field plotted along the outer surface it cuts, with a
// cursor linked to the car.

import { h } from './dom.js';
import { metres, signed } from './format.js';
import { drawCursor, drawPlot, valueAlong } from './plots.js';
import { CUT_AXES, branches, cut } from './section.js';

const AXIS_NAMES = ['x', 'y', 'z'];
const GRAB_PX = 8;

/** What the plot shows for each field at a cut point. Pressure is drawn with negative values up, as aerodynamicists do. */
const QUANTITY = {
	cp: { label: 'Cp', key: 'cp', invert: true, value: (q) => q.cp },
	drag: { label: '−Cp nₓ', key: 'drag', invert: false, value: (q) => -q.cp * q.nx },
	sigma: { label: 'σ', key: 'sigma', invert: false, value: (q) => q.sigma },
};

/** The cut of a case's working mesh at a section, cached until the section moves. */
function sectionOf(c, { axis, offset }) {
	if (c.cut?.axis === axis && c.cut.offset === offset) return c.cut;
	const lines = cut(c.result.mesh, axis, offset);
	c.cut = { axis, offset, lines, branches: branches(lines, axis) };
	return c.cut;
}

/** Plot rows of a section: per branch, rows {along, value, p} in path order. */
function sectionRows(c, section, field) {
	const { along } = CUT_AXES[section.axis];
	const q = QUANTITY[field];
	return sectionOf(c, section).branches.map((paths) => paths.map((path) => path.map((p) => ({ along: p.p[along], value: q.value(p), p: p.p }))));
}

export function sectionCsv(c, view) {
	const names = CUT_AXES[view.section.axis].branches;
	const lines = [['branch', 'x_m', 'y_m', 'z_m', QUANTITY[view.field].key].join(',')];
	sectionRows(c, view.section, view.field).forEach((paths, b) => paths.forEach((path) => {
		for (const r of path) lines.push([names[b], ...r.p.map((v) => v.toFixed(6)), r.value.toFixed(6)].join(','));
	}));
	return lines.join('\n') + '\n';
}

/**
 * host: the overlay over the 3D view (markers for the plot cursor); viewport: the element the car is dragged in;
 * view: the shared view, whose section and field this tool reads. changed(patch) applies a change to the section.
 */
export function createSectionTool({ host, viewport, viewer, active, view, changed }) {
	const markers = [0, 1].map((b) => h('div', { class: `cut-marker b${b}`, hidden: true }));
	host.append(...markers);
	let plot = null;
	let cursorAt = null;

	const enabled = () => Boolean(active()?.result && view.section.on);

	/** Distance in pixels from a client point to the cut line on screen, and the cut point nearest to it. */
	function nearestOnScreen(clientX, clientY) {
		const rect = viewport.getBoundingClientRect();
		const x = clientX - rect.left, y = clientY - rect.top;
		let best = { d: Infinity, point: null };
		for (const line of sectionOf(active(), view.section).lines) {
			let previous = null;
			for (const q of line.points) {
				const s = viewer.project(q.p);
				if (previous && s.visible) {
					const dx = s.x - previous.x, dy = s.y - previous.y;
					const t = Math.max(0, Math.min(1, ((x - previous.x) * dx + (y - previous.y) * dy) / (dx * dx + dy * dy || 1)));
					const d = Math.hypot(previous.x + t * dx - x, previous.y + t * dy - y);
					if (d < best.d) best = { d, point: q.p };
				}
				previous = s;
			}
		}
		return best;
	}

	/** Screen pixels the plane moves per metre along its normal, measured at a point on it. */
	function screenNormal(point) {
		const a = viewer.project(point);
		const moved = [...point];
		moved[view.section.axis] += 0.01;
		const b = viewer.project(moved);
		return { x: (b.x - a.x) / 0.01, y: (b.y - a.y) / 0.01 };
	}

	let dragging = false;
	viewport.addEventListener('pointerdown', (e) => {
		if (e.button !== 0 || !enabled()) return;
		const near = nearestOnScreen(e.clientX, e.clientY);
		if (near.d > GRAB_PX) return;
		const n = screenNormal(near.point);
		const perMetre = n.x * n.x + n.y * n.y;
		if (perMetre < 400) return;
		e.stopPropagation();
		e.preventDefault();
		dragging = true;
		viewport.setPointerCapture(e.pointerId);
		const start = { x: e.clientX, y: e.clientY, offset: view.section.offset };
		viewer.setSection({ ...view.section, plane: true });
		const move = (ev) => {
			const offset = start.offset + ((ev.clientX - start.x) * n.x + (ev.clientY - start.y) * n.y) / perMetre;
			changed({ offset: Number(offset.toFixed(3)) });
		};
		const up = () => {
			dragging = false;
			viewport.removeEventListener('pointermove', move);
			viewport.removeEventListener('pointerup', up);
			changed({});
		};
		viewport.addEventListener('pointermove', move);
		viewport.addEventListener('pointerup', up);
	}, { capture: true });

	viewport.addEventListener('pointermove', (e) => {
		if (dragging || e.buttons || !enabled()) return;
		const near = nearestOnScreen(e.clientX, e.clientY).d <= GRAB_PX;
		viewport.classList.toggle('over-cut', near);
		if (Boolean(viewer.section?.plane) !== near) viewer.setSection({ ...view.section, plane: near });
	});

	/** Markers on the car where the plot cursor meets each branch. */
	function placeMarkers() {
		const rows = cursorAt === null || !enabled() ? null : sectionRows(active(), view.section, view.field);
		markers.forEach((marker, b) => {
			const point = rows ? pointAlong(rows[b], cursorAt) : null;
			marker.hidden = point === null;
			if (point === null) return;
			const p = viewer.project(point);
			marker.hidden = !p.visible;
			marker.style.transform = `translate(${p.x}px, ${p.y}px)`;
		});
	}
	viewer.addEventListener('camera', placeMarkers);

	/** The section plot into an SVG of a given size, with its readout element. */
	function drawSection(svg, readout, width, height) {
		const c = active();
		const q = QUANTITY[view.field];
		const { along, branches: names } = CUT_AXES[view.section.axis];
		const rows = sectionRows(c, view.section, view.field);
		const series = rows.flatMap((paths, b) => paths.map((path) => ({ kind: 'line', cls: `b${b}`, points: path.map((r) => [r.along, r.value]) })));
		const bounds = c.result.geometry.bounds;
		const domain = [bounds[2 * along] - 0.05, bounds[2 * along + 1] + 0.05];
		let lo = Infinity, hi = -Infinity;
		for (const paths of rows) for (const path of paths) for (const r of path) {
			lo = Math.min(lo, r.value);
			hi = Math.max(hi, r.value);
		}
		if (!Number.isFinite(lo)) [lo, hi] = [-1, 1];
		// Pressure keeps a steady scale while the plane is dragged; other fields follow their values.
		if (q.invert) [lo, hi] = [Math.min(-1.2, Math.floor(lo * 2) / 2), Math.max(1, Math.ceil(hi * 2) / 2)];
		const pad = (hi - lo) * 0.06 || 1;
		plot = { frame: drawPlot(svg, { width, height, x: { domain, label: `${AXIS_NAMES[along]}, m` }, y: { domain: [lo - pad, hi + pad], label: q.label, invert: q.invert }, series }), rows, readout, along, names };
		setCursor(cursorAt);
		svg.onpointermove = (e) => {
			const rect = svg.getBoundingClientRect();
			const f = plot.frame;
			const x = e.clientX - rect.left;
			setCursor(x < f.m.l || x > f.width - f.m.r ? null : f.x.invert(x));
		};
		svg.onpointerleave = () => setCursor(null);
	}

	/** Put the plot cursor at a coordinate along the plot axis (or remove it), with the car markers and readout. */
	function setCursor(at) {
		cursorAt = at;
		placeMarkers();
		if (!plot) return;
		const dots = [];
		const parts = [];
		if (at !== null) {
			plot.rows.forEach((paths, b) => {
				const value = valueOn(paths, at);
				if (value === null) return;
				dots.push({ value, cls: `b${b}` });
				parts.push(h('span', { class: `readout-b${b}` }, `${plot.names[b]} `, h('b', {}, signed(value, 3))));
			});
		}
		drawCursor(plot.frame, dots.length ? at : null, dots);
		plot.readout.replaceChildren(...(dots.length ? [h('span', {}, `${AXIS_NAMES[plot.along]} ${metres(at)}`), ...parts] : []));
	}

	/** The cursor follows the car: hovering the surface moves it to the hovered point's coordinate along the plot. */
	function followCar(hit) {
		if (!plot || !enabled()) return;
		setCursor(hit ? hit.point[plot.along] : null);
	}

	/** Header controls for the dock: normal axis, position, clip. */
	function controls(c) {
		const s = view.section;
		const b = c.result.geometry.bounds;
		const [min, max] = [b[2 * s.axis], b[2 * s.axis + 1]];
		const position = h('input', { class: 'number', type: 'text', inputmode: 'decimal', value: signed(s.offset, 3), 'aria-label': 'Position, m', spellcheck: 'false' });
		position.addEventListener('change', () => {
			const value = Number(position.value.replace('\u2212', '-'));
			if (Number.isFinite(value)) changed({ offset: value });
			else position.value = signed(s.offset, 3);
		});
		position.addEventListener('keydown', (e) => e.key === 'Enter' && position.blur());
		const slider = h('input', { type: 'range', class: 'slider', min, max, step: 0.001, value: s.offset, 'aria-label': 'Position' });
		slider.addEventListener('input', () => changed({ offset: Number(slider.value) }));
		const segmented = (label, options, current, pick) => h('div', { class: 'segmented', role: 'radiogroup', 'aria-label': label },
			options.map(([value, text, title]) => h('button', { type: 'button', role: 'radio', class: 'segment', title, 'aria-checked': String(value === current), onclick: () => pick(value) }, text)));
		const name = AXIS_NAMES[s.axis];
		return h('div', { class: 'cut-controls' },
			h('span', { class: 'control-label' }, 'Normal'),
			segmented('Normal', [[0, 'x'], [1, 'y'], [2, 'z']], s.axis, (axis) => changed({ axis, offset: axis === 1 ? 0 : Number(((b[2 * axis] + b[2 * axis + 1]) / 2).toFixed(3)) })),
			h('label', { class: 'position' }, h('span', { class: 'control-label' }, name), position, h('span', { class: 'unit' }, 'm')),
			slider,
			h('span', { class: 'control-label' }, 'Clip'),
			segmented('Clip', [[0, 'Off', 'No clipping'], [1, `−${name}`, `Keep ${name} below the plane`], [-1, `+${name}`, `Keep ${name} above the plane`]], s.clip, (clip) => changed({ clip })));
	}

	function key() {
		const names = CUT_AXES[view.section.axis].branches;
		return names.map((name, b) => h('span', { class: `key-item b${b}` }, h('i'), name));
	}

	return { drawSection, controls, key, followCar, setCursor, forget: () => (plot = null) };
}

/** Value at a coordinate on the first path of a branch that spans it. */
function valueOn(paths, at) {
	for (const path of paths) {
		const value = valueAlong(path.map((r) => [r.along, r.value]), at);
		if (value !== null) return value;
	}
	return null;
}

/** Surface point at a coordinate on the first path of a branch that spans it. */
function pointAlong(paths, at) {
	for (const path of paths) {
		const pts = path.map((r) => [r.along, ...r.p]);
		const x = valueAlong(pts, at, 1);
		if (x === null) continue;
		return [x, valueAlong(pts, at, 2), valueAlong(pts, at, 3)];
	}
	return null;
}
