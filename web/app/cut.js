// The section tool: a plane normal to x, y or z, placed by typing, by its slider or by dragging its line on the car;
// its line drawn on the car, and the shown field plotted along the outer surface it cuts. Beside the predicted Cp the
// plot draws the car it is compared with, or else the CFD when the file carries CFD. A cursor on the plot, moved by
// the pointer or a finger, reads every curve and marks the place on the car; hovering the car moves it there.

import { finite, oneOf, round } from './checks.js';
import { COMPARED, counterpart } from './compare.js';
import { h } from './dom.js';
import { baseName, defineExport } from './export.js';
import { FIELDS } from './fields.js';
import { niceTicks, signed } from './format.js';
import { defineOperation } from './operations.js';
import { drawCursor, drawPlot, valueAlong } from './plots.js';
import { CUT_AXES, branches, cut, interpolated } from './section.js';

const AXIS_NAMES = ['x', 'y', 'z'];
/** How near the cut line, in px, a press grabs it: a fingertip covers far more of the screen than a cursor's tip. */
const GRAB_PX = { mouse: 8, pen: 10, touch: 22 };
/**
 * How the curves are drawn: the car's own in the ink its line is drawn in on the car; what it is set against as a
 * wide pale band under it, so where they agree the ink runs inside the band and where they part it leaves it.
 */
const OWN = { colour: '#0d2233', width: 1.5, opacity: 1 };
const AGAINST = { colour: '#0b6fbf', width: 5, opacity: 0.32 };
/** Cuts remembered per case before the oldest is dropped. */
const CUTS_KEPT = 16;

/**
 * What the plot shows for each of the model's fields, and for the CFD, at a cut point: read(c) gives the reader of a
 * point on case c, or null when c lacks it. Pressure is drawn with negative values up, as aerodynamicists do.
 */
const QUANTITY = {
	cp: { label: 'Cp', key: 'cp', invert: true, read: () => (q) => q.cp },
	drag: { label: '−Cp nₓ', key: 'drag', invert: false, read: () => (q) => -q.cp * q.nx },
	sigma: { label: 'σ', key: 'sigma', invert: false, read: () => (q) => q.sigma },
	cfd: { label: 'Cp', key: 'cp_cfd', invert: true, read: (c) => (c.result.mesh.cfd ? (q) => interpolated(c.result.mesh.cfd, q) : null) },
};

/**
 * What the plot shows for a field: one above, or one added at run time (a comparison's, an annotation's), read from
 * its values at the working-mesh vertices; its reader is null on a case without it.
 */
function quantityOf(field) {
	if (QUANTITY[field]) return QUANTITY[field];
	const added = FIELDS[field];
	return {
		label: added.symbol,
		key: field,
		invert: added.map === 'cp',
		read: (c) => {
			const values = added.values?.(c);
			return values ? (q) => interpolated(values, q) : null;
		},
	};
}

/** Whether case c has a field the section can read. */
const readable = (field, c) => Boolean((QUANTITY[field] || FIELDS[field]) && quantityOf(field).read(c));

/** The field the section plot shows: the one on the car, or Cp where the car shown lacks it. */
const plottedField = (field, c) => (FIELDS[field] && readable(field, c) ? field : 'cp');

/** The cut of a case's working mesh at a plane, remembered for the case. */
function sectionOf(c, { axis, offset }) {
	c.cuts ??= new Map();
	const key = `${axis}:${offset}`;
	const known = c.cuts.get(key);
	if (known) return known;
	const lines = cut(c.result.mesh, axis, offset);
	const made = { lines, branches: branches(lines, axis) };
	c.cuts.set(key, made);
	if (c.cuts.size > CUTS_KEPT) c.cuts.delete(c.cuts.keys().next().value);
	return made;
}

/** Plot rows of a section on a case: per branch, rows {along, value, p} in path order, `along` in metres. */
function sectionRows(c, plane, field) {
	const { along } = CUT_AXES[plane.axis];
	const read = quantityOf(field).read(c);
	const cuts = sectionOf(c, plane).branches;
	if (!read) return cuts.map(() => []);
	return cuts.map((paths) => paths.map((path) => path.map((p) => ({ along: p.p[along], value: read(p), p: p.p }))));
}

/**
 * `count` points spaced evenly along the plot axis on one branch of a section's outer surface (0 the upper, 1 the
 * lower), each in the middle of its share of the branch's span: the stations of a row of probes.
 */
export function sectionStations(c, plane, count, branch) {
	const paths = sectionRows(c, plane, 'cp')[branch];
	const along = paths.flat().map((r) => r.along);
	if (along.length === 0) return [];
	const from = Math.min(...along), to = Math.max(...along);
	return Array.from({ length: count }, (_, i) => pointAlong(paths, from + ((i + 0.5) * (to - from)) / count)).filter(Boolean);
}

/** A value range rounded outward to tidy ticks. */
function tidy(lo, hi) {
	const { step } = niceTicks(lo, hi, 5);
	return [Math.floor(lo / step) * step, Math.ceil(hi / step) * step];
}

/**
 * The value axis for the curves plotted: the span of the middle 98 % of their values, taking in zero, padded and
 * rounded out to tidy ticks. A spike at a sharp edge (the nose, a mirror) reaches past it and is cut at the frame, so
 * it does not flatten every other curve; the cursor still reads its value.
 */
function valueRange(lines) {
	const values = Float64Array.from(lines.flatMap((line) => line.points.map((point) => point[1]))).sort();
	if (values.length === 0) return [-1, 1];
	const at = (q) => values[Math.round(q * (values.length - 1))];
	const lo = Math.min(0, at(0.01)), hi = Math.max(0, at(0.99));
	const pad = 0.08 * (hi - lo || 1);
	return tidy(lo - pad, hi + pad);
}

const planeLabel = ({ axis, offset }) => `${AXIS_NAMES[axis]} = ${signed(offset, 3)} m`;

/**
 * What the section plot shows on the car shown: the field on it, and beside its Cp what it is set against: the car
 * or CFD it is compared with, read on its surface along the same cut, or else its file's CFD. Series carry their
 * field, how they are drawn, the name of what they show (`source`) and their label in the key; `against` marks the
 * second.
 */
function plotSeries(view, shown, cases) {
	const field = plottedField(view.field, shown);
	const own = { case: shown, field, source: shown.name, ...OWN, label: null };
	if (field !== 'cp') return [own];
	const pair = counterpart(view, cases, shown);
	if (pair) return [{ ...own, label: pair.mine }, { case: shown, field: COMPARED, against: true, source: pair.other?.name ?? pair.label, ...AGAINST, label: pair.label }];
	if (shown.result.mesh.cfd) return [{ ...own, label: 'Predicted' }, { case: shown, field: 'cfd', against: true, source: `${shown.name} CFD`, ...AGAINST, label: 'CFD' }];
	return [own];
}

/** The plot's points for every series: what it shows, the section, branch, the point and the value. */
function sectionCsv(view, shown, cases) {
	const names = CUT_AXES[view.section.axis].branches;
	const plane = `${AXIS_NAMES[view.section.axis]}=${view.section.offset}`;
	const lines = [['case', 'section', 'branch', 'x_m', 'y_m', 'z_m', quantityOf(plottedField(view.field, shown)).key].join(',')];
	for (const series of plotSeries(view, shown, cases)) {
		sectionRows(series.case, view.section, series.field).forEach((paths, branch) => paths.forEach((path) => {
			for (const r of path) lines.push([`"${series.source.replaceAll('"', '""')}"`, plane, names[branch], ...r.p.map((v) => v.toFixed(6)), r.value.toFixed(6)].join(','));
		}));
	}
	return lines.join('\n') + '\n';
}

/** app: { session, update }. The section as operations, and its plot's points as an export. */
export function defineSectionOperations({ session, update }) {
	const view = session.view;
	const shown = () => {
		const c = session.active;
		if (!c?.result) throw new Error('No case with a result is on screen. Open a file or an example first.');
		return c;
	};
	const summary = () => ({ on: view.section.on, axis: AXIS_NAMES[view.section.axis], position_m: view.section.offset });

	defineOperation({
		name: 'set_section',
		description: 'Place the section: a plane normal to x, y or z at a position in metres, its line drawn on the car and the shown field plotted along the outer surface it cuts (Cp with negative values up). Beside the predicted Cp the plot draws the car or CFD it is compared with (set_comparison), or else the CFD when the file carries CFD. y = 0 is the car\'s centreline. Turning it on brings its plot up in the dock. Every car is seen through the same section.',
		parameters: {
			type: 'object',
			properties: {
				on: { type: 'boolean', description: 'Whether the section is drawn and plotted.' },
				axis: { type: 'string', enum: AXIS_NAMES, description: 'Axis the plane is normal to. Changing it without a position puts the plane at the car\'s middle along that axis (y = 0 for y).' },
				position_m: { type: 'number', description: 'Where the plane crosses its axis, in metres.' },
			},
			additionalProperties: false,
		},
		run({ on, axis, position_m }) {
			const c = shown();
			const s = view.section;
			if (axis !== undefined) {
				const index = AXIS_NAMES.indexOf(oneOf('axis', axis, AXIS_NAMES));
				const b = c.result.geometry.bounds;
				s.axis = index;
				s.offset = position_m === undefined ? (index === 1 ? 0 : Number(((b[2 * index] + b[2 * index + 1]) / 2).toFixed(3))) : finite('position_m', position_m);
			} else if (position_m !== undefined) {
				s.offset = finite('position_m', position_m);
			}
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
		description: 'Numbers along the section on a case. For each branch of the outer surface it cuts (Upper and Lower, or y > 0 and y < 0 for a z section), the range it spans along the plot axis and the lowest and highest value of a field with where they are. Optionally the field at given positions along the plot axis (x for y and z sections, y for x sections). Values interpolate linearly along the working mesh\'s edges, as drawn.',
		parameters: {
			type: 'object',
			properties: {
				case_id: { type: 'integer', description: 'Default the case on screen.' },
				field: { type: 'string', description: 'cp, drag (−Cp·nₓ), sigma, cfd (the CFD\'s Cp, on a file that carries CFD), or a field added to the case, such as a comparison\'s compared and difference; default cp.' },
				at_m: { type: 'array', items: { type: 'number' }, description: 'Positions along the plot axis, in metres, to read the field at.' },
			},
			additionalProperties: false,
		},
		run({ case_id, field = 'cp', at_m = [] }) {
			const c = case_id === undefined ? shown() : session.byId(case_id);
			if (!c.result) throw new Error(`Case ${c.id} has no result yet (${c.status}).`);
			if (!readable(field, c)) throw new Error(`Case ${c.id} has no field ${JSON.stringify(field)}. It has ${[...Object.keys(QUANTITY), ...Object.keys(FIELDS).filter((key) => !QUANTITY[key])].filter((key) => readable(key, c)).join(', ')}.`);
			at_m.forEach((x, i) => finite(`at_m[${i}]`, x));
			if (!view.section.on) throw new Error('The section is off. Turn it on with set_section first.');
			const plane = view.section;
			const { along, branches: names } = CUT_AXES[plane.axis];
			const point = (r) => r.p.map((v) => round(v, 4));
			return {
				case_id: c.id,
				axis: AXIS_NAMES[plane.axis],
				position_m: plane.offset,
				plot_axis: AXIS_NAMES[along],
				branches: sectionRows(c, plane, field).map((paths, b) => {
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
		title: 'The section plot\'s points: what each curve shows, the section, branch, x, y, z and the value',
		unavailable: () => (view.section.on ? null : 'The section is off. Turn it on to export its plot.'),
		make: async (c) => new Blob([sectionCsv(view, c, session.cases)], { type: 'text/csv' }),
		fileName: (c) => `${baseName(c)}-section.csv`,
	});
}

/**
 * host: the overlay over the 3D view (markers for the plot cursor); viewport: the element the car is dragged in;
 * view: the shared view, whose section and field this tool reads; cases(): every open case. run(name, args) runs an
 * operation: the controls change the section through `set_section`.
 */
export function createSectionTool({ host, viewport, viewer, active, cases, view, run }) {
	let markers = [];
	let plot = null;
	let cursorAt = null;

	// The cut line can be grabbed only while it is drawn: not with the section off, nor while the Model view owns the car.
	const enabled = () => Boolean(active()?.result && viewer.section);

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
		if (near.d > (GRAB_PX[e.pointerType] ?? GRAB_PX.mouse)) return;
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
		const end = () => {
			dragging = false;
			viewport.removeEventListener('pointermove', move);
			viewport.removeEventListener('pointerup', end);
			viewport.removeEventListener('pointercancel', end);
			run('set_section', {});
		};
		viewport.addEventListener('pointermove', move);
		viewport.addEventListener('pointerup', end);
		viewport.addEventListener('pointercancel', end);
	}, { capture: true });

	viewport.addEventListener('pointermove', (e) => {
		if (dragging || e.buttons) return;
		const near = enabled() && nearestOnScreen(e.clientX, e.clientY).d <= GRAB_PX.mouse;
		viewport.classList.toggle('over-cut', near);
		if (Boolean(viewer.section?.plane) !== near) viewer.setSection({ ...view.section, plane: near });
	});

	const seriesNow = () => plotSeries(view, active(), cases());

	/** Markers on the car where the plot cursor meets each branch of the section, the lower one dotted. */
	function placeMarkers() {
		const spots = [];
		if (cursorAt !== null && enabled() && plot) {
			plot.series[0].rows.forEach((paths, b) => {
				const point = pointAlong(paths, cursorAt);
				if (point) spots.push({ point, lower: b === 1 });
			});
		}
		while (markers.length < spots.length) {
			const marker = h('div', { class: 'cut-marker' });
			host.append(marker);
			markers.push(marker);
		}
		markers.forEach((marker, i) => {
			const spot = spots[i];
			marker.hidden = !spot;
			if (!spot) return;
			const p = viewer.project(spot.point);
			marker.hidden = !p.visible;
			marker.classList.toggle('lower', spot.lower);
			marker.style.transform = `translate(${p.x}px, ${p.y}px)`;
		});
	}
	viewer.addEventListener('camera', placeMarkers);

	/** Draw the section plot into an SVG of a given size: its frame, the series drawn and their rows. */
	function plotInto(svg, width, height) {
		const c = active();
		const { along, branches: names } = CUT_AXES[view.section.axis];
		const series = seriesNow().map((one) => ({ ...one, rows: sectionRows(one.case, view.section, one.field) }));
		// What the car is set against is drawn first, under its own curve.
		const lines = [...series].reverse().flatMap((one) => one.rows.flatMap((paths, b) => paths.map((path) => ({ colour: one.colour, width: one.width, opacity: one.opacity, dashed: b === 1, points: path.map((r) => [r.along, r.value]) }))));
		const bounds = c.result.geometry.bounds;
		const q = quantityOf(series[0].field);
		const frame = drawPlot(svg, {
			width, height,
			x: { domain: [bounds[2 * along] - 0.05, bounds[2 * along + 1] + 0.05], label: `${AXIS_NAMES[along]}, m` },
			// The value axis says which way it runs where it is turned over, as pressure is.
			y: { domain: valueRange(lines), label: q.invert ? `${q.label}, negative up` : q.label, invert: q.invert },
			series: lines,
		});
		return { frame, series, along, names };
	}

	/**
	 * The section plot into an SVG of a given size, with its readout element. A press, or a finger moving over it,
	 * puts the cursor there; a finger's cursor stays when it lifts, so its numbers can be read.
	 */
	function drawSection(svg, readout, width, height) {
		plot = { ...plotInto(svg, width, height), readout };
		setCursor(cursorAt);
		const follow = (e) => {
			const f = plot.frame;
			const x = e.clientX - svg.getBoundingClientRect().left;
			setCursor(x < f.m.l || x > f.width - f.m.r ? null : f.x.invert(x));
		};
		svg.onpointerdown = follow;
		svg.onpointermove = (e) => (e.pointerType === 'mouse' || e.buttons) && follow(e);
		svg.onpointerleave = (e) => e.pointerType === 'mouse' && setCursor(null);
	}

	/** Put the plot cursor at a position along the plot axis (or remove it), with the car markers and readout. */
	function setCursor(at) {
		cursorAt = at;
		placeMarkers();
		if (!plot) return;
		const dots = [];
		const parts = [];
		if (at !== null) {
			for (const one of plot.series) {
				const values = one.rows.map((paths) => valueOn(paths, at));
				values.forEach((value) => value !== null && dots.push({ value, colour: one.colour }));
				if (values.every((v) => v === null)) continue;
				// One curve names its branches; two name what each shows, upper then lower.
				const text = plot.series.length === 1
					? values.map((v, b) => (v === null ? null : `${plot.names[b]} ${signed(v, 3)}`)).filter(Boolean).join('  ')
					: `${one.label} ${values.map((v) => (v === null ? '–' : signed(v, 3))).join(' / ')}`;
				parts.push(h('b', { style: `color: ${one.colour}` }, text));
			}
		}
		drawCursor(plot.frame, dots.length ? at : null, dots);
		plot.readout.replaceChildren(...(dots.length ? [h('span', {}, `${AXIS_NAMES[plot.along]} ${signed(at, 3)} m`), ...parts] : []));
	}

	/** The cursor follows the car: hovering the surface moves it to the hovered point's position along the plot. */
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
	const normals = AXIS_NAMES.map((name) => h('button', { type: 'button', role: 'radio', class: 'segment', onclick: () => run('set_section', { axis: name }) }, name));
	const axisLabel = h('span', { class: 'control-label' });
	const controlsEl = h('div', { class: 'cut-controls' },
		h('span', { class: 'control-label' }, 'Normal'), h('div', { class: 'segmented', role: 'radiogroup', 'aria-label': 'Normal' }, normals),
		h('label', { class: 'position' }, axisLabel, position, h('span', { class: 'unit' }, 'm')),
		slider);

	/** Header controls for the dock, brought up to date with the section and the car: normal axis and position. */
	function controls(c) {
		const s = view.section;
		const b = c.result.geometry.bounds;
		slider.min = b[2 * s.axis];
		slider.max = b[2 * s.axis + 1];
		slider.value = s.offset;
		if (document.activeElement !== position) position.value = signed(s.offset, 3);
		axisLabel.textContent = AXIS_NAMES[s.axis];
		normals.forEach((button, axis) => button.setAttribute('aria-checked', String(axis === s.axis)));
		return controlsEl;
	}

	/**
	 * The plot's key, as {label, colour, width, opacity, dashed}: what each curve shows when there are two, then how
	 * the branches are drawn.
	 */
	function keyItems() {
		const series = seriesNow();
		const names = CUT_AXES[view.section.axis].branches;
		const branch = series.length === 1 ? series[0] : { colour: 'var(--muted)', width: 2, opacity: 1 };
		const style = ({ colour, width, opacity }) => ({ colour, width: Math.max(2, width), opacity });
		return [
			...(series.length === 1 ? [] : series.map((one) => ({ label: one.label, ...style(one), dashed: false }))),
			{ label: names[0], ...style(branch), dashed: false },
			{ label: names[1], ...style(branch), dashed: true },
		];
	}

	/** The key as the dock shows it. */
	function key() {
		return keyItems().map(({ label, colour, width, opacity, dashed }) => h('span', { class: 'key-item' }, h('i', { style: `border-top: ${width}px ${dashed ? 'dashed' : 'solid'} ${colour}; opacity: ${opacity}` }), label));
	}

	/** The plot as a report shows it: drawn into a new SVG of the given size, with its key and the section's plane. */
	function picture(width, height) {
		const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
		plotInto(svg, width, height);
		return { svg, key: keyItems(), plane: planeLabel(view.section) };
	}

	return { drawSection, picture, controls, key, followCar, setCursor, forget: () => (plot = null) };
}

/** Value at a position on the first path of a branch that spans it. */
function valueOn(paths, at) {
	for (const path of paths) {
		const value = valueAlong(path.map((r) => [r.along, r.value]), at);
		if (value !== null) return value;
	}
	return null;
}

/** Surface point at a position on the first path of a branch that spans it. */
function pointAlong(paths, at) {
	for (const path of paths) {
		const pts = path.map((r) => [r.along, ...r.p]);
		const x = valueAlong(pts, at, 1);
		if (x === null) continue;
		return [x, valueAlong(pts, at, 2), valueAlong(pts, at, 3)];
	}
	return null;
}
