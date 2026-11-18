// The section tool: a plane normal to x, y or z, placed by typing or by dragging its line on the car; its line drawn
// on the car, the car optionally clipped at it, and the shown field plotted along the outer surface it cuts, with a
// cursor linked to the car. Sections can be kept in the plot beside the live one, each drawn on the car in its plot
// colour, and the plot can show every open car, against position in metres or as a fraction of the car's size, so
// cuts and cars overlay on one axis.

import { finite, oneOf, round } from './checks.js';
import { COMPARED, counterpart } from './compare.js';
import { h } from './dom.js';
import { baseName, defineExport } from './export.js';
import { FIELDS, valueAt } from './fields.js';
import { niceTicks, signed } from './format.js';
import { icon } from './icons.js';
import { defineOperation } from './operations.js';
import { drawCursor, drawPlot, valueAlong } from './plots.js';
import { CUT_AXES, branches, cut, interpolated } from './section.js';
import { MAX_KEPT } from '../viewer/shaders.js';

const AXIS_NAMES = ['x', 'y', 'z'];
const GRAB_PX = 8;
const CLIPS = { off: 0, keep_below: 1, keep_above: -1 };
const ABSCISSAE = ['metres', 'fraction'];
/**
 * A line per section and car, in this order; the branch below is dashed. The first is the ink the live section's
 * line is drawn in on the car; the rest are taken from the theme.
 */
const COLOURS = ['#0d2233', '#0b6fbf', '#a35b1a', '#197b80', '#9a55e8', '#a83258', '#237f50', '#7a6a12'];
/** A kept section that runs along another axis than the live one is drawn on the car but not plotted. */
const UNPLOTTED = '#58727f';
/** Cuts remembered per case, beyond those plotted, before the oldest is dropped. */
const CUTS_KEPT = 16;

/**
 * What the plot shows for each of the model's fields at a cut point: read(c) gives the reader of a point on case c.
 * Pressure is drawn with negative values up, as aerodynamicists do.
 */
const QUANTITY = {
	cp: { label: 'Cp', key: 'cp', invert: true, read: () => (q) => q.cp },
	drag: { label: '−Cp nₓ', key: 'drag', invert: false, read: () => (q) => -q.cp * q.nx },
	sigma: { label: 'σ', key: 'sigma', invert: false, read: () => (q) => q.sigma },
};

/**
 * What the plot shows for a field: one of the model's, or one added at run time (a comparison's, an annotation's),
 * read from its values at the working-mesh vertices; its reader is null on a case without it.
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

/** The field the section plot shows: the one on the car, or Cp where the car shown lacks it. */
const plottedField = (field, c) => (FIELDS[field] && quantityOf(field).read(c) ? field : 'cp');

/**
 * Position as a fraction of the car: along x, x/L from the nose; along y, 2y/W, 0 on the centreline and ±1 at the
 * widest point. Cars of different sizes line up on it.
 */
const FRACTIONS = [
	{ label: 'x/L', column: 'x_over_l', domain: [-0.02, 1.02], of: (b, v) => (v - b[0]) / (b[1] - b[0]), metres: (b, f) => b[0] + f * (b[1] - b[0]) },
	{ label: '2y/W', column: 'two_y_over_w', domain: [-1.05, 1.05], of: (b, v) => (2 * v) / (b[3] - b[2]), metres: (b, f) => (f * (b[3] - b[2])) / 2 },
];

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

/** Plot rows of a section on a case: per branch, rows {along, value, p} in path order, `along` on the plot's axis. */
function sectionRows(c, plane, field, abscissa) {
	const { along } = CUT_AXES[plane.axis];
	const read = quantityOf(field).read(c);
	const cuts = sectionOf(c, plane).branches;
	if (!read) return cuts.map(() => []);
	const b = c.result.geometry.bounds;
	const position = abscissa === 'fraction' ? (v) => FRACTIONS[along].of(b, v) : (v) => v;
	return cuts.map((paths) => paths.map((path) => path.map((p) => ({ along: position(p.p[along]), value: read(p), p: p.p }))));
}

/**
 * `count` points spaced evenly along the plot axis on one branch of a section's outer surface (0 the upper, 1 the
 * lower), each in the middle of its share of the branch's span: the stations of a row of probes.
 */
export function sectionStations(c, plane, count, branch) {
	const paths = sectionRows(c, plane, 'cp', 'metres')[branch];
	const along = paths.flat().map((r) => r.along);
	if (along.length === 0) return [];
	const from = Math.min(...along), to = Math.max(...along);
	return Array.from({ length: count }, (_, i) => pointAlong(paths, from + ((i + 0.5) * (to - from)) / count)).filter(Boolean);
}

const plotRanges = new WeakMap();

/** A value range rounded outward to tidy ticks. */
function tidy(lo, hi) {
	const { step } = niceTicks(lo, hi, 5);
	return [Math.floor(lo / step) * step, Math.ceil(hi / step) * step];
}

/**
 * A field's value axis for a car, held still wherever the plane goes: for the model's fields their lowest and highest
 * value over the whole working mesh; for a field added at run time, whose extremes can sit far out on a few cells, its
 * colour scale (`map`), as the car shows it.
 */
function plotRange(c, field, map) {
	if (!QUANTITY[field]) return tidy(map.lo, map.hi);
	const ranges = plotRanges.get(c) ?? plotRanges.set(c, {}).get(c);
	if (ranges[field]) return ranges[field];
	const { mesh } = c.result;
	let lo = Infinity, hi = -Infinity;
	for (let v = 0; v < mesh.positions.length / 3; v++) {
		const value = valueAt(mesh, field, v);
		if (value < lo) lo = value;
		if (value > hi) hi = value;
	}
	ranges[field] = tidy(lo, hi);
	return ranges[field];
}

const planeLabel = ({ axis, offset }) => `${AXIS_NAMES[axis]} = ${signed(offset, 3)} m`;

/**
 * What the section plot shows: the live section and the kept ones that run along the same axis, each on the case
 * shown or on every case with a result, and, beside Cp, what the car shown is compared with, read on its surface
 * along the same cut (`alongside`). Series are section-major, each with its field, its colour, the name of what it
 * shows (`source`) and its label in the key.
 */
export function plotSeries(view, shown, cases) {
	const s = view.section;
	const along = CUT_AXES[s.axis].along;
	const field = plottedField(view.field, shown);
	const planes = [{ id: null, axis: s.axis, offset: s.offset }, ...s.kept.filter((k) => CUT_AXES[k.axis].along === along)];
	// The car shown comes first, so its live section is plotted in the ink its line is drawn in.
	const cars = s.cars === 'all' ? [shown, ...cases.filter((c) => c.result && c !== shown)] : [shown];
	const pair = field === 'cp' ? counterpart(view, cases, shown) : null;
	// A car plotted already is not plotted again as what the car shown is compared with.
	const alongside = pair && !cars.includes(pair.other);
	const sectionName = (plane) => (planes.length > 1 ? planeLabel(plane) : null);
	const series = planes.flatMap((plane) => [
		...cars.map((c) => ({ plane, case: c, field, source: c.name, label: [sectionName(plane), cars.length > 1 ? c.name : alongside ? pair.mine : null].filter(Boolean).join(' · ') })),
		...(alongside ? [{ plane, case: shown, field: COMPARED, alongside: true, source: pair.other?.name ?? pair.label, label: [sectionName(plane), pair.label].filter(Boolean).join(' · ') }] : []),
	]);
	return series.map((one, i) => ({ ...one, colour: COLOURS[i % COLOURS.length] }));
}

/** The plot's points for every series: what it shows, section, branch, the point, its position as a fraction, the field. */
export function sectionCsv(view, shown, cases) {
	const along = CUT_AXES[view.section.axis].along;
	const names = CUT_AXES[view.section.axis].branches;
	const fraction = FRACTIONS[along];
	const lines = [['case', 'section', 'branch', 'x_m', 'y_m', 'z_m', fraction.column, quantityOf(plottedField(view.field, shown)).key].join(',')];
	for (const series of plotSeries(view, shown, cases)) {
		const b = series.case.result.geometry.bounds;
		const label = `${AXIS_NAMES[series.plane.axis]}=${series.plane.offset}`;
		sectionRows(series.case, series.plane, series.field, 'metres').forEach((paths, branch) => paths.forEach((path) => {
			for (const r of path) lines.push([`"${series.source.replaceAll('"', '""')}"`, label, names[branch], ...r.p.map((v) => v.toFixed(6)), fraction.of(b, r.along).toFixed(6), r.value.toFixed(6)].join(','));
		}));
	}
	return lines.join('\n') + '\n';
}

/** app: { session, update }. The section as operations, and its plot's points as an export. */
export function defineSectionOperations({ session, update }) {
	const view = session.view;
	let nextId = 1;
	const shown = () => {
		const c = session.active;
		if (!c?.result) throw new Error('No case with a result is on screen. Open a file or an example first.');
		return c;
	};
	const keptOut = (k) => ({ section_id: k.id, axis: AXIS_NAMES[k.axis], position_m: k.offset });
	const summary = () => {
		const s = view.section;
		return { on: s.on, axis: AXIS_NAMES[s.axis], position_m: s.offset, clip: Object.keys(CLIPS).find((k) => CLIPS[k] === s.clip), kept: s.kept.map(keptOut), plotted_against: s.abscissa, cars: s.cars };
	};
	const planeArgs = (c, axis, position_m) => {
		const index = AXIS_NAMES.indexOf(oneOf('axis', axis, AXIS_NAMES));
		const b = c.result.geometry.bounds;
		return { axis: index, offset: position_m === undefined ? (index === 1 ? 0 : Number(((b[2 * index] + b[2 * index + 1]) / 2).toFixed(3))) : finite('position_m', position_m) };
	};

	defineOperation({
		name: 'set_section',
		description: 'Place the live section: a plane normal to x, y or z at a position in metres, its line drawn on the car and the shown field plotted along the outer surface it cuts (Cp with suction, negative Cp, upward). y = 0 is the car\'s centreline. The car can be clipped at the plane to see inside. Turning it on brings its plot up in the dock. The plot runs against position in metres or as a fraction of the car (x/L from the nose for y and z sections, 2y/W for x sections), and shows the car on screen or every open car. Every car is seen through the same sections.',
		parameters: {
			type: 'object',
			properties: {
				on: { type: 'boolean', description: 'Whether the sections are drawn and plotted.' },
				axis: { type: 'string', enum: AXIS_NAMES, description: 'Axis the plane is normal to. Changing it without a position puts the plane at the car\'s middle along that axis (y = 0 for y).' },
				position_m: { type: 'number', description: 'Where the plane crosses its axis, in metres.' },
				clip: { type: 'string', enum: Object.keys(CLIPS), description: 'Hide the car on one side of the plane: keep_below keeps the part where the coordinate is below the position.' },
				plotted_against: { type: 'string', enum: ABSCISSAE, description: 'metres, or fraction: x/L from the nose (y and z sections) or 2y/W (x sections), so cars of different sizes line up.' },
				cars: { type: 'string', enum: ['shown', 'all'], description: 'Plot the car on screen, or every open car with a result over it.' },
			},
			additionalProperties: false,
		},
		run({ on, axis, position_m, clip, plotted_against, cars }) {
			const c = shown();
			const s = view.section;
			if (axis !== undefined) Object.assign(s, planeArgs(c, axis, position_m));
			else if (position_m !== undefined) s.offset = finite('position_m', position_m);
			if (clip !== undefined) s.clip = CLIPS[oneOf('clip', clip, Object.keys(CLIPS))];
			if (plotted_against !== undefined) s.abscissa = oneOf('plotted_against', plotted_against, ABSCISSAE);
			if (cars !== undefined) s.cars = oneOf('cars', cars, ['shown', 'all']);
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
		name: 'keep_section',
		description: `Keep a section in the plot beside the live one, drawn on the car in its plot colour: the live section by default, or a plane given by axis and position. Kept sections that run along the same axis as the live one (y and z sections run along x, x sections along y) are plotted with it. At most ${MAX_KEPT}. Returns its section_id.`,
		parameters: {
			type: 'object',
			properties: {
				axis: { type: 'string', enum: AXIS_NAMES, description: 'Axis the plane is normal to; default the live section\'s.' },
				position_m: { type: 'number', description: 'Where the plane crosses its axis, in metres; default the live section\'s, or the car\'s middle for another axis.' },
			},
			additionalProperties: false,
		},
		run({ axis, position_m }) {
			const c = shown();
			const s = view.section;
			const plane = axis === undefined ? { axis: s.axis, offset: position_m === undefined ? s.offset : finite('position_m', position_m) } : planeArgs(c, axis, position_m);
			const same = s.kept.find((k) => k.axis === plane.axis && k.offset === plane.offset);
			if (same) return { section_id: same.id, ...summary() };
			if (s.kept.length >= MAX_KEPT) throw new Error(`${MAX_KEPT} sections are kept already. Remove one with remove_section first.`);
			const kept = { id: nextId++, ...plane };
			s.kept.push(kept);
			if (!s.on) s.on = true;
			update('viewer', 'dock', 'toolbar');
			return { section_id: kept.id, ...summary() };
		},
	});

	defineOperation({
		name: 'remove_section',
		description: 'Remove a kept section from the plot and the car.',
		parameters: { type: 'object', properties: { section_id: { type: 'integer' } }, required: ['section_id'], additionalProperties: false },
		run({ section_id }) {
			const s = view.section;
			const i = s.kept.findIndex((k) => k.id === section_id);
			if (i < 0) throw new Error(`There is no kept section ${section_id}. Kept: ${s.kept.map((k) => `${k.id} (${planeLabel(k)})`).join(', ') || 'none'}.`);
			s.kept.splice(i, 1);
			update('viewer', 'dock');
			return summary();
		},
	});

	defineOperation({
		name: 'read_section',
		description: 'Numbers along a section on a case: the live section, or a kept one by section_id. For each branch of the outer surface it cuts (Upper and Lower, or y > 0 and y < 0 for a z section), the range it spans along the plot axis and the lowest and highest value of a field with where they are. Optionally the field at given positions along the plot axis (x for y and z sections, y for x sections). Values interpolate linearly along the working mesh\'s edges, as drawn.',
		parameters: {
			type: 'object',
			properties: {
				case_id: { type: 'integer', description: 'Default the case on screen.' },
				section_id: { type: 'integer', description: 'A kept section; default the live one.' },
				field: { type: 'string', description: 'cp, drag (−Cp·nₓ) or sigma, or a field added to the case, such as a comparison\'s compared and difference; default cp.' },
				at_m: { type: 'array', items: { type: 'number' }, description: 'Positions along the plot axis, in metres, to read the field at.' },
			},
			additionalProperties: false,
		},
		run({ case_id, section_id, field = 'cp', at_m = [] }) {
			const c = case_id === undefined ? shown() : session.byId(case_id);
			if (!c.result) throw new Error(`Case ${c.id} has no result yet (${c.status}).`);
			if (!FIELDS[field] || !quantityOf(field).read(c)) throw new Error(`Case ${c.id} has no field ${JSON.stringify(field)}. It has cp, drag, sigma${Object.keys(FIELDS).filter((key) => !QUANTITY[key] && FIELDS[key].values?.(c)).map((key) => `, ${key}`).join('')}.`);
			at_m.forEach((x, i) => finite(`at_m[${i}]`, x));
			let plane = view.section;
			if (section_id !== undefined) {
				plane = view.section.kept.find((k) => k.id === section_id);
				if (!plane) throw new Error(`There is no kept section ${section_id}. Kept: ${view.section.kept.map((k) => `${k.id} (${planeLabel(k)})`).join(', ') || 'none'}.`);
			} else if (!view.section.on) {
				throw new Error('The section is off. Turn it on with set_section first, or read a kept section by section_id.');
			}
			const { along, branches: names } = CUT_AXES[plane.axis];
			const point = (r) => r.p.map((v) => round(v, 4));
			return {
				case_id: c.id,
				axis: AXIS_NAMES[plane.axis],
				position_m: plane.offset,
				plot_axis: AXIS_NAMES[along],
				branches: sectionRows(c, plane, field, 'metres').map((paths, b) => {
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
		title: 'The section plot\'s points: case, section, branch, x, y, z, position as a fraction of the car, and the field shown',
		unavailable: () => (view.section.on ? null : 'The section is off. Turn it on to export its plot.'),
		make: async (c) => new Blob([sectionCsv(view, c, session.cases)], { type: 'text/csv' }),
		fileName: (c) => `${baseName(c)}-section.csv`,
	});
}

/**
 * host: the overlay over the 3D view (markers for the plot cursor); viewport: the element the car is dragged in;
 * view: the shared view, whose section and field this tool reads; cases(): every open case. run(name, args) runs an
 * operation: the controls change the section through `set_section`, `keep_section` and `remove_section`.
 */
export function createSectionTool({ host, viewport, viewer, active, cases, view, run }) {
	let markers = [];
	let plot = null;
	let cursorAt = null;

	// The cut line can be grabbed only while it is drawn: not with the section off, nor while the Model view owns the car.
	const enabled = () => Boolean(active()?.result && viewer.section);
	// While a tool owns the left button (painting, a lasso), a stroke across the cut line is the tool's, not a drag of it.
	const grabbable = () => enabled() && !viewer.tool;

	/** Distance in pixels from a client point to the live cut line on screen, and the cut point nearest to it. */
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

	const seriesNow = () => plotSeries(view, active(), cases());

	/** The kept sections drawn on the car shown, each in the colour of its line for that car; grey when not plotted. */
	function keptLines() {
		const c = active();
		const series = seriesNow().filter((s) => s.case === c);
		return view.section.kept.map((k) => ({ axis: k.axis, offset: k.offset, colour: series.find((s) => s.plane === k)?.colour ?? UNPLOTTED }));
	}

	/** Markers on the car shown where the plot cursor meets each branch of each of its sections, in their colours. */
	function placeMarkers() {
		const c = active();
		const spots = [];
		if (cursorAt !== null && enabled() && plot) {
			for (const s of plot.series) {
				if (s.case !== c || s.alongside) continue;
				s.rows.forEach((paths, b) => {
					const point = pointAlong(paths, cursorAt);
					if (point) spots.push({ point, colour: s.colour, lower: b === 1 });
				});
			}
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
			marker.style.borderColor = spot.colour;
			marker.classList.toggle('lower', spot.lower);
			marker.style.transform = `translate(${p.x}px, ${p.y}px)`;
		});
	}
	viewer.addEventListener('camera', placeMarkers);

	/** The section plot into an SVG of a given size, with its readout element. */
	function drawSection(svg, readout, width, height) {
		const s = view.section;
		const field = plottedField(view.field, active());
		const q = quantityOf(field);
		const { along, branches: names } = CUT_AXES[s.axis];
		const series = seriesNow().map((one) => ({ ...one, rows: sectionRows(one.case, one.plane, one.field, s.abscissa) }));
		const lines = series.flatMap((one) => one.rows.flatMap((paths, b) => paths.map((path) => ({ cls: `b${b}`, colour: one.colour, dashed: b === 1, points: path.map((r) => [r.along, r.value]) }))));
		const cars = [...new Set(series.map((one) => one.case))];
		const fraction = FRACTIONS[along];
		const domain = s.abscissa === 'fraction'
			? fraction.domain
			: [Math.min(...cars.map((c) => c.result.geometry.bounds[2 * along])) - 0.05, Math.max(...cars.map((c) => c.result.geometry.bounds[2 * along + 1])) + 0.05];
		// One range per car whose field the plot reads; what the car shown is compared with is drawn on its axis.
		const read = series.filter((one) => !one.alongside && quantityOf(one.field).read(one.case));
		const ranges = [...new Set(read.map((one) => one.case))].map((c) => plotRange(c, field, view.maps?.[FIELDS[field].map]));
		const label = s.abscissa === 'fraction' ? fraction.label : `${AXIS_NAMES[along]}, m`;
		const frame = drawPlot(svg, { width, height, x: { domain, label }, y: { domain: [Math.min(...ranges.map((r) => r[0])), Math.max(...ranges.map((r) => r[1]))], label: q.label, invert: q.invert }, series: lines });
		plot = { frame, series, readout, along, names, label };
		setCursor(cursorAt);
		svg.onpointermove = (e) => {
			const rect = svg.getBoundingClientRect();
			const f = plot.frame;
			const x = e.clientX - rect.left;
			setCursor(x < f.m.l || x > f.width - f.m.r ? null : f.x.invert(x));
		};
		svg.onpointerleave = () => setCursor(null);
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
				values.forEach((value, b) => value !== null && dots.push({ value, cls: `b${b}`, colour: one.colour }));
				if (values.every((v) => v === null)) continue;
				// One line of the plot names its branches; several name their section or car.
				const text = values.map((v, b) => (v === null ? null : plot.series.length === 1 ? `${plot.names[b]} ${signed(v, 3)}` : signed(v, 3))).filter(Boolean).join(plot.series.length === 1 ? '  ' : ' / ');
				parts.push(h('b', { style: `color: ${one.colour}`, title: one.label || null }, text));
			}
		}
		drawCursor(plot.frame, dots.length ? at : null, dots);
		const position = view.section.abscissa === 'fraction' ? `${plot.label} ${at?.toFixed(3)}` : `${AXIS_NAMES[plot.along]} ${signed(at ?? 0, 3)} m`;
		plot.readout.replaceChildren(...(dots.length ? [h('span', {}, position), ...parts] : []));
	}

	/** The cursor follows the car: hovering the surface moves it to the hovered point's position along the plot. */
	function followCar(hit) {
		if (!plot || !enabled()) return;
		if (!hit) return setCursor(null);
		const v = hit.point[plot.along];
		setCursor(view.section.abscissa === 'fraction' ? FRACTIONS[plot.along].of(active().result.geometry.bounds, v) : v);
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
	const keep = h('button', { type: 'button', class: 'tool', title: 'Keep this section in the plot and on the car', onclick: () => run('keep_section', {}).catch(() => {}) }, icon('pin'), 'Keep');
	const abscissa = segments('Plotted against', ABSCISSAE, (value) => run('set_section', { plotted_against: value }));
	const allCars = h('button', { type: 'button', class: 'tool', title: 'Plot every open car over this one', onclick: () => run('set_section', { cars: view.section.cars === 'all' ? 'shown' : 'all' }) }, 'All cars');
	const controlsEl = h('div', { class: 'cut-controls' },
		h('span', { class: 'control-label' }, 'Normal'), normal.el,
		h('label', { class: 'position' }, axisLabel, position, h('span', { class: 'unit' }, 'm')),
		slider,
		h('span', { class: 'control-label' }, 'Clip'), clip.el,
		keep, abscissa.el, allCars);

	/** Header controls for the dock, brought up to date with the section and the car: normal axis, position, clip, plot. */
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
		keep.disabled = s.kept.length >= MAX_KEPT || s.kept.some((k) => k.axis === s.axis && k.offset === s.offset);
		const fraction = FRACTIONS[CUT_AXES[s.axis].along];
		abscissa.buttons.forEach((button, i) => {
			[button.textContent, button.title] = [['m', 'Plot against position in metres'], [fraction.label, `Plot against ${fraction.label}, so cars of different sizes line up`]][i];
			button.setAttribute('aria-checked', String(ABSCISSAE[i] === s.abscissa));
		});
		allCars.hidden = cases().filter((x) => x.result).length < 2;
		allCars.setAttribute('aria-pressed', String(s.cars === 'all'));
		return controlsEl;
	}

	/**
	 * The plot's key: a line per section and car when there are several, a kept section's with a button that removes
	 * it; then how the two branches are drawn. Kept sections along another axis are listed, dimmed, as not plotted.
	 */
	function key() {
		const s = view.section;
		const names = CUT_AXES[s.axis].branches;
		const series = seriesNow();
		const sample = (colour, dashed) => h('i', { style: `border-top: 2px ${dashed ? 'dashed' : 'solid'} ${colour}` });
		const items = series.length === 1 ? [] : series.map((one) => h('span', { class: 'key-item' }, sample(one.colour, false), one.label,
			one.plane.id === null ? null : h('button', { type: 'button', class: 'key-remove', title: `Remove ${planeLabel(one.plane)}`, onclick: () => run('remove_section', { section_id: one.plane.id }) }, icon('close'))));
		const unplotted = s.kept.filter((k) => !series.some((one) => one.plane === k)).map((k) => h('span', { class: 'key-item unplotted', title: `Runs along ${AXIS_NAMES[CUT_AXES[k.axis].along]}; plotted with sections that do` }, planeLabel(k),
			h('button', { type: 'button', class: 'key-remove', title: `Remove ${planeLabel(k)}`, onclick: () => run('remove_section', { section_id: k.id }) }, icon('close'))));
		const colour = series.length === 1 ? series[0].colour : 'var(--muted)';
		return [...items, ...unplotted, h('span', { class: 'key-item' }, sample(colour, false), names[0]), h('span', { class: 'key-item' }, sample(colour, true), names[1])];
	}

	return { drawSection, controls, key, keptLines, followCar, setCursor, forget: () => (plot = null) };
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
