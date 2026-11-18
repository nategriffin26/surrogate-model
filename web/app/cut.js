// The section tool: a plane normal to x, y or z, placed by typing or by dragging its line on the car; its line drawn
// on the car, the car optionally clipped at it, and the shown field plotted along the outer surface it cuts, with a
// cursor linked to the car.

import { finite, oneOf, round } from './checks.js';
import { h } from './dom.js';
import { baseName, defineExport } from './export.js';
import { metres, niceTicks, signed } from './format.js';
import { defineOperation } from './operations.js';
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

const plotRanges = new WeakMap();

/**
 * A field's value axis for a car: its lowest and highest value over the whole working mesh, rounded outward to tidy
 * ticks, so the axis holds still wherever the plane goes.
 */
function plotRange(c, field) {
	const ranges = plotRanges.get(c) ?? plotRanges.set(c, {}).get(c);
	if (ranges[field]) return ranges[field];
	const { cp, sigma, normals } = c.result.mesh;
	let lo = Infinity, hi = -Infinity;
	for (let v = 0; v < cp.length; v++) {
		const value = field === 'cp' ? cp[v] : field === 'sigma' ? sigma[v] : -cp[v] * normals[3 * v];
		if (value < lo) lo = value;
		if (value > hi) hi = value;
	}
	const { step } = niceTicks(lo, hi, 5);
	ranges[field] = [Math.floor(lo / step) * step, Math.ceil(hi / step) * step];
	return ranges[field];
}

export function sectionCsv(c, view) {
	const names = CUT_AXES[view.section.axis].branches;
	const lines = [['branch', 'x_m', 'y_m', 'z_m', QUANTITY[view.field].key].join(',')];
	sectionRows(c, view.section, view.field).forEach((paths, b) => paths.forEach((path) => {
		for (const r of path) lines.push([names[b], ...r.p.map((v) => v.toFixed(6)), r.value.toFixed(6)].join(','));
	}));
	return lines.join('\n') + '\n';
}

const CLIPS = { off: 0, keep_below: 1, keep_above: -1 };

/** app: { session, update }. The section as operations, and its plot's points as an export. */
export function defineSectionOperations({ session, update }) {
	const view = session.view;
	const shown = () => {
		const c = session.active;
		if (!c?.result) throw new Error('No case with a result is on screen. Open a file or an example first.');
		return c;
	};
	const summary = () => ({ on: view.section.on, axis: AXIS_NAMES[view.section.axis], position_m: view.section.offset, clip: Object.keys(CLIPS).find((k) => CLIPS[k] === view.section.clip) });

	defineOperation({
		name: 'set_section',
		description: 'Place the section: a plane normal to x, y or z at a position in metres, its line drawn on the car and the shown field plotted along the outer surface it cuts (Cp with suction, negative Cp, upward). y = 0 is the car\'s centreline. The car can be clipped at the plane to see inside. Turning it on brings its plot up in the dock. Every car is seen through the same section.',
		parameters: {
			type: 'object',
			properties: {
				on: { type: 'boolean', description: 'Whether the section is drawn and plotted.' },
				axis: { type: 'string', enum: AXIS_NAMES, description: 'Axis the plane is normal to. Changing it without a position puts the plane at the car\'s middle along that axis (y = 0 for y).' },
				position_m: { type: 'number', description: 'Where the plane crosses its axis, in metres.' },
				clip: { type: 'string', enum: Object.keys(CLIPS), description: 'Hide the car on one side of the plane: keep_below keeps the part where the coordinate is below the position.' },
			},
			additionalProperties: false,
		},
		run({ on, axis, position_m, clip }) {
			const c = shown();
			const s = view.section;
			if (axis !== undefined) {
				s.axis = AXIS_NAMES.indexOf(oneOf('axis', axis, AXIS_NAMES));
				const b = c.result.geometry.bounds;
				if (position_m === undefined) s.offset = s.axis === 1 ? 0 : Number(((b[2 * s.axis] + b[2 * s.axis + 1]) / 2).toFixed(3));
			}
			if (position_m !== undefined) s.offset = finite('position_m', position_m);
			if (clip !== undefined) s.clip = CLIPS[oneOf('clip', clip, Object.keys(CLIPS))];
			if (on !== undefined && on !== s.on) {
				s.on = Boolean(on);
				// Turning the section on brings its plot up; turning it off leaves the probe table in its place.
				if (s.on) {
					view.dock = 'section';
					view.dockCollapsed = false;
				} else if (view.dock === 'section') {
					view.dock = 'probes';
				}
			}
			update('toolbar', 'viewer', 'dock', 'layout');
			return summary();
		},
	});

	defineOperation({
		name: 'read_section',
		description: 'Numbers along the section on a case: for each branch of the outer surface it cuts (Upper and Lower, or y > 0 and y < 0 for a z section), the range it spans along the plot axis and the lowest and highest value of a field with where they are. Optionally the field at given positions along the plot axis (x for y and z sections, y for x sections). Values interpolate linearly along the working mesh\'s edges, as drawn.',
		parameters: {
			type: 'object',
			properties: {
				case_id: { type: 'integer', description: 'Default the case on screen.' },
				field: { type: 'string', enum: Object.keys(QUANTITY), description: 'cp, drag (−Cp·nₓ) or sigma; default cp.' },
				at_m: { type: 'array', items: { type: 'number' }, description: 'Positions along the plot axis, in metres, to read the field at.' },
			},
			additionalProperties: false,
		},
		run({ case_id, field = 'cp', at_m = [] }) {
			const c = case_id === undefined ? shown() : session.byId(case_id);
			if (!c.result) throw new Error(`Case ${c.id} has no result yet (${c.status}).`);
			if (!view.section.on) throw new Error('The section is off. Turn it on with set_section first.');
			oneOf('field', field, Object.keys(QUANTITY));
			at_m.forEach((x, i) => finite(`at_m[${i}]`, x));
			const { along, branches: names } = CUT_AXES[view.section.axis];
			const point = (r) => r.p.map((v) => round(v, 4));
			return {
				...summary(),
				plot_axis: AXIS_NAMES[along],
				branches: sectionRows(c, view.section, field).map((paths, b) => {
					const rows = paths.flat();
					if (rows.length === 0) return { name: names[b], points: 0 };
					const low = rows.reduce((m, r) => (r.value < m.value ? r : m));
					const high = rows.reduce((m, r) => (r.value > m.value ? r : m));
					return {
						name: names[b],
						points: rows.length,
						from_m: round(Math.min(...rows.map((r) => r.along)), 4),
						to_m: round(Math.max(...rows.map((r) => r.along)), 4),
						min: { value: round(low.value), at_m: point(low) },
						max: { value: round(high.value), at_m: point(high) },
						...(at_m.length ? { at: at_m.map((x) => ({ position_m: x, value: valueOn(paths, x) === null ? null : round(valueOn(paths, x)) })) } : {}),
					};
				}),
			};
		},
	});

	defineExport({
		name: 'section',
		label: 'Section',
		format: 'CSV',
		title: 'The section plot\'s points: branch, x, y, z and the field shown',
		unavailable: () => (view.section.on ? null : 'The section is off. Turn it on to export its plot.'),
		make: async (c) => new Blob([sectionCsv(c, view)], { type: 'text/csv' }),
		fileName: (c) => `${baseName(c)}-section.csv`,
	});
}

/**
 * host: the overlay over the 3D view (markers for the plot cursor); viewport: the element the car is dragged in;
 * view: the shared view, whose section and field this tool reads. run(name, args) runs an operation: the controls
 * change the section through `set_section`.
 */
export function createSectionTool({ host, viewport, viewer, active, view, run }) {
	const markers = [0, 1].map((b) => h('div', { class: `cut-marker b${b}`, hidden: true }));
	host.append(...markers);
	let plot = null;
	let cursorAt = null;

	// The cut line can be grabbed only while it is drawn: not with the section off, nor while the Model view owns the car.
	const enabled = () => Boolean(active()?.result && viewer.section);
	// While a tool owns the left button (painting, a lasso), a stroke across the cut line is the tool's, not a drag of it.
	const grabbable = () => enabled() && !viewer.tool;

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
		if (e.button !== 0 || !grabbable()) return;
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
			run('set_section', { position_m: Number(offset.toFixed(3)) });
		};
		const up = () => {
			dragging = false;
			viewport.removeEventListener('pointermove', move);
			viewport.removeEventListener('pointerup', up);
			run('set_section', {});
		};
		viewport.addEventListener('pointermove', move);
		viewport.addEventListener('pointerup', up);
	}, { capture: true });

	viewport.addEventListener('pointermove', (e) => {
		if (dragging || e.buttons) return;
		const near = grabbable() && nearestOnScreen(e.clientX, e.clientY).d <= GRAB_PX;
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
		const series = rows.flatMap((paths, b) => paths.map((path) => ({ cls: `b${b}`, points: path.map((r) => [r.along, r.value]) })));
		const bounds = c.result.geometry.bounds;
		const domain = [bounds[2 * along] - 0.05, bounds[2 * along + 1] + 0.05];
		const [lo, hi] = plotRange(c, view.field);
		plot = { frame: drawPlot(svg, { width, height, x: { domain, label: `${AXIS_NAMES[along]}, m` }, y: { domain: [lo, hi], label: q.label, invert: q.invert }, series }), rows, readout, along, names };
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

	// The header controls are built once and updated in place: replacing them while the slider is held would end
	// the drag after its first step.
	const position = h('input', { class: 'number', type: 'text', inputmode: 'decimal', 'aria-label': 'Position, m', spellcheck: 'false' });
	position.addEventListener('change', () => {
		const value = Number(position.value.replace('\u2212', '-'));
		if (Number.isFinite(value)) run('set_section', { position_m: value });
		else position.value = signed(view.section.offset, 3);
	});
	position.addEventListener('keydown', (e) => e.key === 'Enter' && position.blur());
	const slider = h('input', { type: 'range', class: 'slider', step: 0.001, 'aria-label': 'Position' });
	slider.addEventListener('input', () => run('set_section', { position_m: Number(slider.value) }));
	const segments = (label, values, pick) => {
		const buttons = values.map((value) => h('button', { type: 'button', role: 'radio', class: 'segment', onclick: () => pick(value) }));
		return { el: h('div', { class: 'segmented', role: 'radiogroup', 'aria-label': label }, buttons), buttons };
	};
	const normal = segments('Normal', [0, 1, 2], (axis) => run('set_section', { axis: AXIS_NAMES[axis] }));
	normal.buttons.forEach((b, axis) => (b.textContent = AXIS_NAMES[axis]));
	const clip = segments('Clip', [0, 1, -1], (value) => run('set_section', { clip: Object.keys(CLIPS).find((k) => CLIPS[k] === value) }));
	const axisLabel = h('span', { class: 'control-label' });
	const controlsEl = h('div', { class: 'cut-controls' },
		h('span', { class: 'control-label' }, 'Normal'), normal.el,
		h('label', { class: 'position' }, axisLabel, position, h('span', { class: 'unit' }, 'm')),
		slider,
		h('span', { class: 'control-label' }, 'Clip'), clip.el);

	/** Header controls for the dock, brought up to date with the section and the car: normal axis, position, clip. */
	function controls(c) {
		const s = view.section;
		const b = c.result.geometry.bounds;
		const name = AXIS_NAMES[s.axis];
		slider.min = b[2 * s.axis];
		slider.max = b[2 * s.axis + 1];
		slider.value = s.offset;
		if (document.activeElement !== position) position.value = signed(s.offset, 3);
		axisLabel.textContent = name;
		normal.buttons.forEach((button, axis) => button.setAttribute('aria-checked', String(axis === s.axis)));
		const clipLabels = [['Off', 'No clipping'], [`−${name}`, `Keep ${name} below the plane`], [`+${name}`, `Keep ${name} above the plane`]];
		clip.buttons.forEach((button, i) => {
			[button.textContent, button.title] = clipLabels[i];
			button.setAttribute('aria-checked', String([0, 1, -1][i] === s.clip));
		});
		return controlsEl;
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
