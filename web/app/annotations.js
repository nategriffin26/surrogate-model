// Annotations: what the assistant lays on a car to show something. A field of their own colours the car as Pressure
// does, with its legend, range and probe readout; paint colours chosen faces; labels, points, lines, arrows, boxes and
// planes mark places in 3D. Everything is drawn in named layers that belong to the case, kept while other cases are
// shown; the strip above the view lists the case's layers, where each is shown, hidden or removed (toolbar.js).
// Every action is an operation, so a call from the assistant, a line of Python and a control do the same thing;
// Python sends its arrays as typed arrays, never as JSON.
//
// Faces are the working mesh's triangles (`engine.workingCells`); the viewer colours per vertex, so a value given per
// face reaches it as the area-weighted mean of the faces around each vertex, as the model's own fields do.

import { Annotations, rgbOf } from '../viewer/annotations.js';
import { finite, oneOf, round, triple } from './checks.js';
import { evaluate } from './expression.js';
import { MAPS, addField, faceMean, removeField } from './fields.js';
import { count, niceTicks } from './format.js';
import { defineOperation } from './operations.js';
import { RAMPS } from './ramps.js';

/** Colours by name: the theme's ink and accents, with orange bright enough to stand out on the car. */
const PALETTE = {
	ink: '#0d2233',
	blue: '#0b6fbf',
	orange: '#e2741d',
	teal: '#197b80',
	rose: '#a83258',
	violet: '#9a55e8',
	green: '#237f50',
	grey: '#58727f',
	white: '#ffffff',
};
const RAMP_NAMES = Object.keys(RAMPS);
/** Line width, CSS px, of a new layer. */
const LINE_PX = 1.5;
const POINT_PX = 7;
/** A label's text is cut at this length: a label names a place, it does not explain it. */
const LABEL_CHARS = 80;
/** Arrows given no scale are drawn with the longest this share of the car's length. */
const ARROW_SHARE = 0.06;
/** A layer called by no name takes its kind's. */
const DEFAULT_NAMES = {
	labels: 'Labels',
	points: 'Points',
	lines: 'Lines',
	arrows: 'Arrows',
	box: 'Boxes',
	plane: 'Planes',
	paint: 'Paint',
};

/**
 * @typedef {Object} Layer
 * @property {string} name     unique on its case; how calls refer to it
 * @property {'field' | 'drawing'} kind  a field of its own, or paint and marks
 * @property {'engineer' | 'assistant'} by  who made it; the assistant asks before removing the engineer's
 * @property {boolean} visible  a drawing layer drawn; a field layer is shown by choosing its field
 * @property {string} colour   '#rrggbb': its marks', where they carry none of their own
 * @property {number} width    line width, CSS px
 * @property {import('../viewer/annotations.js').Mark[]} marks
 * @property {Uint8Array | null} paint  RGBA per working-mesh face; alpha 0 leaves a face unpainted
 * @property {{ key: string, unit: string, vertices: Float32Array, source: string } | null} field
 */

/** Per case: its layers in drawing order, later on top, and a number that changes whenever they do. */
const stores = new WeakMap();
const storeOf = (c) => stores.get(c) ?? stores.set(c, { layers: [], version: 0, paint: null }).get(c);

// ---- checking arguments -------------------------------------------------------------------------------------------------

/** Points as a flat (N, 3) array: a typed array from Python, a list of [x, y, z], or one [x, y, z]. */
function pointsOf(name, value) {
	if (ArrayBuffer.isView(value)) {
		if (value.length === 0 || value.length % 3)
			throw new Error(`${name} must hold three numbers per point; got ${value.length} numbers.`);
		return Float32Array.from(value);
	}
	if (Array.isArray(value) && value.length === 3 && value.every((v) => typeof v === 'number'))
		return Float32Array.from(triple(name, value));
	if (!Array.isArray(value) || value.length === 0)
		throw new Error(
			`${name} must be a list of points [x, y, z] in metres; got ${JSON.stringify(value)?.slice(0, 80)}.`,
		);
	const out = new Float32Array(3 * value.length);
	value.forEach((p, i) => out.set(triple(`${name}[${i}]`, p), 3 * i));
	return out;
}

function colourOf(name, value) {
	if (typeof value === 'string') {
		const named = PALETTE[value.trim().toLowerCase()];
		if (named) return named;
		if (/^#[0-9a-f]{6}$/i.test(value)) return value.toLowerCase();
		if (/^#[0-9a-f]{3}$/i.test(value)) return `#${[...value.slice(1)].map((d) => d + d).join('')}`.toLowerCase();
	}
	throw new Error(
		`${name} must be a colour name (${Object.keys(PALETTE).join(', ')}) or "#rrggbb"; got ${JSON.stringify(value)}.`,
	);
}

function widthOf(name, value) {
	const width = finite(name, value);
	if (width <= 0 || width > 12) throw new Error(`${name} must be above 0 and at most 12 px; got ${width}.`);
	return width;
}

/** RGBA bytes per item, from Python: `count` items. */
function itemColours(name, value, items) {
	if (!(value instanceof Uint8Array) || value.length !== 4 * items)
		throw new Error(
			`${name} must be RGBA bytes, four per item (${4 * items}); got ${value?.length ?? JSON.stringify(value)}.`,
		);
	return value;
}

const nameOf = (value, fallback) => {
	if (value === undefined) return fallback;
	const name = String(value).trim();
	if (!name) throw new Error('layer must not be empty.');
	return name;
};

// ---- per face and per vertex ----------------------------------------------------------------------------------------------

/** Per vertex, the area-weighted mean of a value over the faces around it. */
function toVertices(mesh, cells, faces) {
	const { indices } = mesh;
	const vertices = new Float32Array(mesh.positions.length / 3);
	const around = new Float32Array(vertices.length);
	for (let f = 0; f < faces.length; f++) {
		const a = cells.areas[f];
		for (let k = 0; k < 3; k++) {
			const v = indices[3 * f + k];
			vertices[v] += a * faces[f];
			around[v] += a;
		}
	}
	for (let v = 0; v < vertices.length; v++) vertices[v] = around[v] > 0 ? vertices[v] / around[v] : 0;
	return vertices;
}

/** Per face, the mean of its corners: the value the viewer draws at its centre. */
function toFaces(mesh, vertices) {
	return Float32Array.from({ length: mesh.indices.length / 3 }, (_, f) => faceMean(vertices, mesh.indices, f));
}

/** Every face a layer paints in one colour, each keeping its opacity. */
function repaint(paint, colour) {
	const [r, g, b] = rgbOf(colour);
	for (let i = 0; i < paint.length; i += 4) if (paint[i + 3]) paint.set([r, g, b], i);
}

/**
 * The paint of a case's visible layers, per vertex as the viewer takes it: each face the colour of the topmost layer
 * that paints it; each vertex the area-weighted colour and strength of the painted faces around it, and the share of
 * its area they cover.
 */
function paintOn(mesh, cells, layers) {
	const faces = cells.areas.length;
	const top = new Int32Array(faces).fill(-1);
	layers.forEach((layer, l) => {
		if (!layer.visible || !layer.paint) return;
		for (let f = 0; f < faces; f++) if (layer.paint[4 * f + 3] > 0) top[f] = l;
	});
	if (top.every((l) => l < 0)) return null;
	const vertices = mesh.positions.length / 3;
	const sums = new Float32Array(5 * vertices);
	const around = new Float32Array(vertices);
	for (let f = 0; f < faces; f++) {
		const area = cells.areas[f];
		const paint = top[f] >= 0 ? layers[top[f]].paint : null;
		for (let k = 0; k < 3; k++) {
			const v = mesh.indices[3 * f + k];
			around[v] += area;
			if (!paint) continue;
			for (let c = 0; c < 4; c++) sums[5 * v + c] += area * paint[4 * f + c];
			sums[5 * v + 4] += area;
		}
	}
	const colour = new Uint8Array(4 * vertices);
	const cover = new Uint8Array(vertices);
	for (let v = 0; v < vertices; v++) {
		const painted = sums[5 * v + 4];
		if (painted === 0) continue;
		for (let c = 0; c < 4; c++) colour[4 * v + c] = Math.round(sums[5 * v + c] / painted);
		cover[v] = Math.round((255 * painted) / around[v]);
	}
	return { colour, cover };
}

/** A starting scale for values: symmetric about zero when they straddle it, else their 1st to 99th percentile. */
function scaleFor(values) {
	const step = Math.max(1, Math.floor(values.length / 20000));
	const sample = new Float32Array(Math.ceil(values.length / step));
	for (let i = 0, k = 0; i < values.length; i += step) sample[k++] = values[i];
	sample.sort();
	const lo = sample[Math.floor(0.01 * (sample.length - 1))];
	const hi = sample[Math.floor(0.99 * (sample.length - 1))];
	const round = (x, up) => {
		const { step: tick } = niceTicks(0, Math.abs(x) || 1, 4);
		return (up ? Math.ceil(x / tick - 1e-9) : Math.floor(x / tick + 1e-9)) * tick;
	};
	if (lo < 0 && hi > 0 && Math.min(-lo, hi) > 0.2 * Math.max(-lo, hi)) {
		const m = round(Math.max(-lo, hi), true);
		return { lo: -m, hi: m, diverging: true };
	}
	if (hi <= lo) return { lo: lo - 1, hi: hi + 1, diverging: false };
	return { lo: round(lo, false), hi: round(hi, true), diverging: false };
}

/** Area-weighted mean, lowest and highest of per-face values. */
function statistics(values, areas) {
	let sum = 0,
		area = 0,
		lo = Infinity,
		hi = -Infinity;
	for (let f = 0; f < values.length; f++) {
		sum += values[f] * areas[f];
		area += areas[f];
		if (values[f] < lo) lo = values[f];
		if (values[f] > hi) hi = values[f];
	}
	return { min: round(lo, 6), max: round(hi, 6), mean: round(sum / area, 6) };
}

// ---- the annotations as the app holds them ---------------------------------------------------------------------------------

/**
 * context: { session, engine, viewer, view, overlay, update }. Defines the annotation operations and returns what the
 * rest of the app reads: `apply(c)` lays case c's annotations on the viewer, `fields(c)` its fields' keys, `layersOf(c)`
 * its layers in drawing order, and `paintLabels(ctx, scale)` the labels for the picture. A field's values reach the
 * viewer, the probes and the plots through its entry in fields.js.
 */
export function createAnnotations({ session, engine, viewer, view, overlay, update }) {
	const drawing = new Annotations(viewer, overlay);
	const cellsKept = new WeakMap();
	const cellsOf = (result) => cellsKept.get(result) ?? cellsKept.set(result, engine.workingCells(result)).get(result);
	/** What was last handed to the viewer, so an update that changes nothing hands it nothing. */
	let pushed = {};

	const layerOn = (store, name) => store.layers.find((layer) => layer.name === name);
	const needLayer = (c, name) => {
		const layer = layerOn(storeOf(c), name);
		if (!layer)
			throw new Error(
				`Case ${c.id} has no layer "${name}". Its layers: ${
					storeOf(c)
						.layers.map((l) => `"${l.name}"`)
						.join(', ') || 'none'
				}.`,
			);
		return layer;
	};
	const fieldLayer = (c, key) => storeOf(c).layers.find((layer) => layer.field?.key === key) ?? null;
	const casesWith = (key) => session.cases.filter((c) => stores.has(c) && fieldLayer(c, key));

	/** Something on case c changed: the viewer, the dock and the toolbar, with its fields and layers, catch up. */
	function changed(c, ...parts) {
		const store = storeOf(c);
		store.version++;
		store.paint = null;
		if (c === session.active) update('viewer', 'dock', 'toolbar', ...parts);
	}

	/** The drawing layer called `name` on case c, made now if there is none, restyled by `style`. */
	function drawingLayer(c, name, by, style) {
		const store = storeOf(c);
		let layer = layerOn(store, name);
		if (layer?.kind === 'field')
			throw new Error(`"${name}" is a field layer on case ${c.id}; draw in a layer of another name.`);
		if (!layer) {
			layer = {
				name,
				kind: 'drawing',
				by,
				visible: true,
				colour: PALETTE.ink,
				width: LINE_PX,
				marks: [],
				paint: null,
				field: null,
			};
			store.layers.push(layer);
		}
		if (style.colour !== undefined) layer.colour = colourOf('colour', style.colour);
		if (style.width_px !== undefined) layer.width = widthOf('width_px', style.width_px);
		return layer;
	}

	/** Add a mark to a layer of case c and show it. */
	function draw(c, args, by, kind, mark) {
		const layer = drawingLayer(c, nameOf(args.layer, DEFAULT_NAMES[kind]), by, {
			colour: args.colour,
			width_px: args.width_px,
		});
		layer.marks.push(mark);
		layer.visible = true;
		changed(c);
		return layer;
	}

	// ---- the field shown, through its key ----

	const keyOf = (name) => `custom:${name}`;

	/** Make sure the field `key` is offered while any case has it, at the scale given. */
	function offerField(key, name, unit, source, start, diverging) {
		const values = (c) => (c?.result && stores.has(c) ? (fieldLayer(c, key)?.field.vertices ?? null) : null);
		addField(
			key,
			{ label: name, title: unit ? `${name} (${unit})` : name, symbol: name, tip: source, values },
			{ ramps: [start.ramp, ...RAMP_NAMES.filter((r) => r !== start.ramp)], diverging, start },
		);
		view.maps ??= {};
		view.maps[key] = { ...start };
	}

	/** A field no case has any longer stops being offered; a case showing it goes back to pressure. */
	function withdrawField(key) {
		if (casesWith(key).length) return;
		removeField(key);
		if (view.maps) delete view.maps[key];
		if (view.field === key) view.field = 'cp';
	}

	function showField(c, { name, expression, values, unit = '', ramp, min, max }, by) {
		const label = nameOf(name, undefined);
		if (label === undefined) throw new Error("name is needed: the field's name, as the toolbar shows it.");
		if ((expression === undefined) === (values === undefined))
			throw new Error('Give expression or values, one of them.');
		const mesh = c.result.mesh;
		const cells = cellsOf(c.result);
		const faceCount = cells.areas.length,
			vertexCount = mesh.positions.length / 3;
		let faces, vertices;
		if (expression !== undefined) {
			faces = evaluate(expression, cells);
			vertices = toVertices(mesh, cells, faces);
		} else {
			if (!ArrayBuffer.isView(values) && !Array.isArray(values))
				throw new Error('values must be an array with one number per face or per vertex of the working mesh.');
			const given = Float32Array.from(values);
			if (given.length === faceCount) [faces, vertices] = [given, toVertices(mesh, cells, given)];
			else if (given.length === vertexCount) [faces, vertices] = [toFaces(mesh, given), given];
			else
				throw new Error(
					`values must hold one number per face (${faceCount}) or per vertex (${vertexCount}) of case ${c.id}'s working mesh; got ${given.length}.`,
				);
		}
		const bad = faces.reduce((n, v) => n + !Number.isFinite(v), 0);
		if (bad)
			throw new Error(
				`${count(bad)} of ${count(faceCount)} faces have no finite value (a division by zero, a log of a negative). Guard the expression or fill those values.`,
			);
		const auto = scaleFor(faces);
		const lo = min === undefined ? auto.lo : finite('min', min);
		const hi = max === undefined ? auto.hi : finite('max', max);
		if (!(lo < hi)) throw new Error(`min must be below max; got ${lo} and ${hi}.`);
		const diverging = lo < 0 && hi > 0;
		const chosen = ramp === undefined ? (diverging ? 'coolwarm' : 'viridis') : oneOf('ramp', ramp, RAMP_NAMES);
		const store = storeOf(c);
		const key = keyOf(label);
		let layer = layerOn(store, label);
		if (layer && layer.kind !== 'field')
			throw new Error(`"${label}" is a drawing layer on case ${c.id}; give the field another name.`);
		if (!layer) {
			layer = {
				name: label,
				kind: 'field',
				by,
				visible: true,
				colour: PALETTE.ink,
				width: LINE_PX,
				marks: [],
				paint: null,
				field: null,
			};
			store.layers.push(layer);
		}
		const source = expression !== undefined ? String(expression) : 'From Python';
		layer.field = { key, unit: String(unit), vertices, source };
		offerField(key, label, layer.field.unit, source, { ramp: chosen, lo, hi, bands: 0 }, diverging);
		view.field = key;
		changed(c, 'legend', 'markers');
		return {
			layer: label,
			field: key,
			faces: faceCount,
			values: statistics(faces, cells.areas),
			scale: { min: lo, max: hi, ramp: chosen },
		};
	}

	// ---- paint ----

	function paintFaces(c, args, by) {
		const { where, mask, colours } = args;
		const sources = [where, mask, colours].filter((v) => v !== undefined).length;
		if (sources !== 1) throw new Error('Say which faces with exactly one of where, mask or colours.');
		const cells = cellsOf(c.result);
		const faces = cells.areas.length;
		const rgba = new Uint8Array(4 * faces);
		const colour = colours === undefined ? colourOf('colour', args.colour ?? 'orange') : undefined;
		if (colours !== undefined) {
			rgba.set(itemColours('colours', colours, faces));
		} else {
			let picked;
			if (where !== undefined) picked = evaluate(where, cells);
			else {
				if (!ArrayBuffer.isView(mask) || mask.length !== faces)
					throw new Error(
						`mask must hold one value per face of the working mesh (${faces}); got ${mask?.length ?? JSON.stringify(mask)}.`,
					);
				picked = mask;
			}
			const [r, g, b] = rgbOf(colour);
			const opacity = args.opacity === undefined ? 1 : finite('opacity', args.opacity);
			if (opacity <= 0 || opacity > 1) throw new Error(`opacity must be above 0 and at most 1; got ${opacity}.`);
			for (let f = 0; f < faces; f++) if (picked[f]) rgba.set([r, g, b, Math.round(255 * opacity)], 4 * f);
		}
		const layer = drawingLayer(c, nameOf(args.layer, DEFAULT_NAMES.paint), by, { colour });
		if (!layer.paint) layer.paint = rgba;
		else for (let f = 0; f < faces; f++) if (rgba[4 * f + 3]) layer.paint.set(rgba.subarray(4 * f, 4 * f + 4), 4 * f);
		layer.visible = true;
		let painted = 0,
			area = 0,
			whole = 0;
		for (let f = 0; f < faces; f++) {
			whole += cells.areas[f];
			if (!rgba[4 * f + 3]) continue;
			painted++;
			area += cells.areas[f];
		}
		changed(c);
		return {
			layer: layer.name,
			faces: painted,
			area_m2: round(area, 5),
			share_of_area: round(area / whole, 5),
			summed_over: `the ${count(faces)} faces of the working mesh`,
		};
	}

	// ---- what the viewer draws ----

	function paintOf(c) {
		const store = storeOf(c);
		store.paint ??= { value: paintOn(c.result.mesh, cellsOf(c.result), store.layers) };
		return store.paint.value;
	}

	/** Lay case c's annotations on the viewer, or none for null; called whenever the view is brought up to date. */
	function apply(c) {
		const shown = c?.result ? c : null;
		const store = shown && stores.get(shown);
		if (pushed.surface !== viewer.surface) pushed = { surface: viewer.surface };
		const paint = store ? paintOf(shown) : null;
		if (paint !== pushed.paint) viewer.setPaint((pushed.paint = paint));
		const marks = store ? `${shown.id}:${store.version}` : null;
		if (marks !== pushed.marks) {
			pushed.marks = marks;
			const layers = store ? store.layers.filter((layer) => layer.kind === 'drawing' && layer.visible) : [];
			drawing.show(
				layers.map(({ colour, width, marks }) => ({ colour, width, marks })),
				shown?.result.mesh ?? null,
			);
		}
	}

	// ---- operations ----

	const caseId = { type: 'integer', description: 'Default the case on screen.' };
	const layerArg = (kind) => ({
		type: 'string',
		description: `Layer to draw in: made if new, added to if it exists. Default "${DEFAULT_NAMES[kind]}".`,
	});
	const colourArg = {
		type: 'string',
		description: `A colour name (${Object.keys(PALETTE).join(', ')}) or "#rrggbb". Sets the layer's colour.`,
	};
	const widthArg = { type: 'number', description: `Line width in px (default ${LINE_PX}). Sets the layer's.` };
	const pointsArg = (what) => ({
		type: 'array',
		items: { type: 'array', items: { type: 'number' }, minItems: 3, maxItems: 3 },
		description: `${what}, each [x, y, z] in metres.`,
	});
	const fromPython = (what) => ({
		description: `${what}, sent from Python as a typed array (pressure.py); not for typing out.`,
	});
	const layerSummary = (layer) => {
		const counts = {};
		for (const mark of layer.marks) {
			const n =
				mark.kind === 'lines'
					? mark.starts.length - 1
					: mark.kind === 'box' || mark.kind === 'plane'
						? 1
						: mark.points.length / 3;
			counts[mark.kind] = (counts[mark.kind] ?? 0) + n;
		}
		const painted = layer.paint ? layer.paint.reduce((n, v, i) => n + (i % 4 === 3 && v > 0), 0) : 0;
		return {
			name: layer.name,
			kind: layer.kind,
			made_by: layer.by,
			visible: layer.kind === 'field' ? view.field === layer.field.key : layer.visible,
			...(layer.kind === 'field'
				? { field: layer.field.key, unit: layer.field.unit, source: layer.field.source }
				: {
						colour: layer.colour,
						width_px: layer.width,
						marks: counts,
						...(painted ? { painted_faces: painted } : {}),
					}),
		};
	};

	defineOperation({
		name: 'show_field',
		description:
			"Colour the car by a field of your own, shown and offered beside Pressure, Drag and Spread with its own legend, colour scale and probe readout; computed from an expression over the faces of the working mesh, or from an array sent from Python (case.show). Names: cp, sigma, cfd (the reference field), drag (−Cp·nx), x, y, z (face centre, m), nx, ny, nz (outward unit normal), area (m²); functions abs, sqrt, exp, log, log10, sign, min, max, clip; comparisons give 1 or 0. A field of the same name on another car shares its colour scale. Returns its key, the values' lowest, highest and area-weighted mean, and the scale.",
		parameters: {
			type: 'object',
			properties: {
				name: { type: 'string', description: 'Short name, as the toolbar and legend show it, e.g. "Spread / |Cp|".' },
				expression: { type: 'string', description: 'Per face, e.g. "sigma / max(abs(cp), 0.05)".' },
				values: {
					type: 'array',
					items: { type: 'number' },
					...fromPython('One number per face or per vertex of the working mesh'),
				},
				unit: { type: 'string', description: 'Unit shown in the legend title; empty for none.' },
				ramp: {
					type: 'string',
					enum: RAMP_NAMES,
					description: 'Colour ramp; default coolwarm for values either side of zero, else viridis.',
				},
				min: {
					type: 'number',
					description:
						'Value at the low end of the scale; default from the values (1st percentile, or symmetric about zero).',
				},
				max: { type: 'number', description: 'Value at the high end of the scale.' },
				case_id: caseId,
			},
			required: ['name'],
			additionalProperties: false,
		},
		run: (args, { by }) => showField(session.withResult(args.case_id), args, by),
	});

	defineOperation({
		name: 'paint_faces',
		description:
			"Paint faces of the working mesh in a colour over the field shown: those where an expression is true (show_field's names), or a mask or colours per face sent from Python (case.paint). Painting adds to a layer's paint. Returns the faces painted, their area and their share of the car's area.",
		parameters: {
			type: 'object',
			properties: {
				where: { type: 'string', description: 'Faces where this is true, e.g. "cp < -1".' },
				mask: fromPython('One 0 or 1 per face'),
				colours: fromPython('RGBA bytes per face; alpha 0 leaves a face unpainted'),
				colour: {
					type: 'string',
					description: `Colour name (${Object.keys(PALETTE).join(', ')}) or "#rrggbb"; default orange.`,
				},
				opacity: { type: 'number', description: 'How strongly the colour covers the field, above 0 to 1 (default 1).' },
				layer: layerArg('paint'),
				case_id: caseId,
			},
			additionalProperties: false,
		},
		run: (args, { by }) => paintFaces(session.withResult(args.case_id), args, by),
	});

	defineOperation({
		name: 'draw_labels',
		description:
			'Label places on the car: text in a small box beside a dot at each point, kept in place as the camera moves, moved clear of other labels with a leader where needed, and faded where the car hides the point. Batches: many points and texts in one call. Returns the layer and the count.',
		parameters: {
			type: 'object',
			properties: {
				points: pointsArg('Where each label points'),
				texts: {
					type: 'array',
					items: { type: 'string' },
					description: `One text per point, at most ${LABEL_CHARS} characters, e.g. "Cp −2.73".`,
				},
				offset_px: {
					type: 'array',
					items: { type: 'number' },
					minItems: 2,
					maxItems: 2,
					description:
						'Where to put the boxes first, [right, down] in px from their points, with a leader; default beside them.',
				},
				layer: layerArg('labels'),
				colour: colourArg,
				case_id: caseId,
			},
			required: ['points', 'texts'],
			additionalProperties: false,
		},
		run(args, { by }) {
			const c = session.withResult(args.case_id);
			const points = pointsOf('points', args.points);
			if (!Array.isArray(args.texts) || args.texts.length !== points.length / 3)
				throw new Error(
					`texts must hold one text per point (${points.length / 3}); got ${Array.isArray(args.texts) ? args.texts.length : JSON.stringify(args.texts)}.`,
				);
			const texts = args.texts.map((t) => String(t).slice(0, LABEL_CHARS));
			const offset =
				args.offset_px === undefined
					? null
					: [finite('offset_px[0]', args.offset_px[0]), finite('offset_px[1]', args.offset_px[1])];
			const layer = draw(c, args, by, 'labels', { kind: 'labels', points, texts, offset });
			return { layer: layer.name, labels: texts.length };
		},
	});

	defineOperation({
		name: 'draw_points',
		description: 'Mark points on or around the car with dots, in batches of any size. Returns the layer and the count.',
		parameters: {
			type: 'object',
			properties: {
				points: pointsArg('The points'),
				size_px: { type: 'number', description: `Dot diameter in px (default ${POINT_PX}).` },
				colours: fromPython('RGBA bytes per point'),
				layer: layerArg('points'),
				colour: colourArg,
				case_id: caseId,
			},
			required: ['points'],
			additionalProperties: false,
		},
		run(args, { by }) {
			const c = session.withResult(args.case_id);
			const points = pointsOf('points', args.points);
			const colours = args.colours === undefined ? null : itemColours('colours', args.colours, points.length / 3);
			const layer = draw(c, args, by, 'points', {
				kind: 'points',
				points,
				size: args.size_px === undefined ? POINT_PX : widthOf('size_px', args.size_px),
				colours,
			});
			return { layer: layer.name, points: points.length / 3 };
		},
	});

	defineOperation({
		name: 'draw_lines',
		description:
			'Draw lines through points, on or off the surface: one polyline or many in one call, straight between consecutive points. Returns the layer and the count.',
		parameters: {
			type: 'object',
			properties: {
				lines: {
					type: 'array',
					items: pointsArg("A polyline's points"),
					description: 'Polylines, each a list of [x, y, z] in metres (from Python, each may be a typed array).',
				},
				closed: { type: 'boolean', description: "Join each polyline's last point to its first." },
				layer: layerArg('lines'),
				colour: colourArg,
				width_px: widthArg,
				case_id: caseId,
			},
			required: ['lines'],
			additionalProperties: false,
		},
		run(args, { by }) {
			const c = session.withResult(args.case_id);
			if (!Array.isArray(args.lines) || args.lines.length === 0)
				throw new Error('lines must be a list of polylines, each a list of points [x, y, z].');
			const polylines = args.lines.map((line, i) => pointsOf(`lines[${i}]`, line));
			const starts = new Uint32Array(polylines.length + 1);
			polylines.forEach((line, i) => (starts[i + 1] = starts[i] + line.length / 3));
			const points = new Float32Array(3 * starts.at(-1));
			polylines.forEach((line, i) => points.set(line, 3 * starts[i]));
			const layer = draw(c, args, by, 'lines', { kind: 'lines', points, starts, closed: Boolean(args.closed) });
			return { layer: layer.name, lines: polylines.length };
		},
	});

	defineOperation({
		name: 'draw_arrows',
		description:
			"Draw arrows from points along vectors, vector × scale long in metres: a vector plot of thousands in one call. With no scale the longest arrow is drawn 6 % of the car's length. Returns the layer, the count, the scale used and the longest arrow in metres.",
		parameters: {
			type: 'object',
			properties: {
				points: pointsArg('Where each arrow starts'),
				vectors: pointsArg("Each arrow's direction and size"),
				scale: { type: 'number', description: 'Metres drawn per unit of vector.' },
				colours: fromPython('RGBA bytes per arrow'),
				layer: layerArg('arrows'),
				colour: colourArg,
				width_px: widthArg,
				case_id: caseId,
			},
			required: ['points', 'vectors'],
			additionalProperties: false,
		},
		run(args, { by }) {
			const c = session.withResult(args.case_id);
			const points = pointsOf('points', args.points);
			const vectors = pointsOf('vectors', args.vectors);
			if (vectors.length !== points.length)
				throw new Error(`vectors must hold one vector per point (${points.length / 3}); got ${vectors.length / 3}.`);
			let longest = 0;
			for (let i = 0; i < vectors.length; i += 3)
				longest = Math.max(longest, Math.hypot(vectors[i], vectors[i + 1], vectors[i + 2]));
			if (!(longest > 0)) throw new Error('Every vector is zero; there is nothing to draw.');
			const scale =
				args.scale === undefined ? (ARROW_SHARE * c.result.geometry.length) / longest : finite('scale', args.scale);
			const colours = args.colours === undefined ? null : itemColours('colours', args.colours, points.length / 3);
			const layer = draw(c, args, by, 'arrows', {
				kind: 'arrows',
				points,
				vectors: vectors.map((v) => v * scale),
				colours,
			});
			return {
				layer: layer.name,
				arrows: points.length / 3,
				scale: round(scale, 8),
				longest_m: round(longest * scale, 4),
			};
		},
	});

	defineOperation({
		name: 'draw_box',
		description:
			'Draw a box with its edges along the axes, from its lowest to its highest corner, faces lightly washed: for a reference volume or a part of the car. Returns the layer.',
		parameters: {
			type: 'object',
			properties: {
				min: {
					type: 'array',
					items: { type: 'number' },
					minItems: 3,
					maxItems: 3,
					description: 'Lowest corner [x, y, z] in metres.',
				},
				max: {
					type: 'array',
					items: { type: 'number' },
					minItems: 3,
					maxItems: 3,
					description: 'Highest corner [x, y, z] in metres.',
				},
				layer: layerArg('box'),
				colour: colourArg,
				width_px: widthArg,
				case_id: caseId,
			},
			required: ['min', 'max'],
			additionalProperties: false,
		},
		run(args, { by }) {
			const c = session.withResult(args.case_id);
			const lo = triple('min', args.min),
				hi = triple('max', args.max);
			if (!lo.every((v, k) => v < hi[k]))
				throw new Error(`min must be below max on every axis; got ${JSON.stringify(lo)} and ${JSON.stringify(hi)}.`);
			return { layer: draw(c, args, by, 'box', { kind: 'box', min: [...lo], max: [...hi] }).name };
		},
	});

	defineOperation({
		name: 'draw_plane',
		description:
			'Draw a rectangle of a plane through a point, square to a normal, lightly washed: for a reference plane, a cut, a ground line. Returns the layer.',
		parameters: {
			type: 'object',
			properties: {
				point: {
					type: 'array',
					items: { type: 'number' },
					minItems: 3,
					maxItems: 3,
					description: 'Its centre [x, y, z] in metres.',
				},
				normal: {
					type: 'array',
					items: { type: 'number' },
					minItems: 3,
					maxItems: 3,
					description: 'Its normal [x, y, z].',
				},
				size_m: {
					type: 'array',
					items: { type: 'number' },
					minItems: 2,
					maxItems: 2,
					description:
						"Its sides in metres, the first along the frame's axis the normal points least along; default the car's length both ways.",
				},
				layer: layerArg('plane'),
				colour: colourArg,
				width_px: widthArg,
				case_id: caseId,
			},
			required: ['point', 'normal'],
			additionalProperties: false,
		},
		run(args, { by }) {
			const c = session.withResult(args.case_id);
			const normal = triple('normal', args.normal);
			if (Math.hypot(...normal) === 0) throw new Error('normal must not be zero.');
			const length = c.result.geometry.length;
			const size =
				args.size_m === undefined
					? [length, length]
					: [finite('size_m[0]', args.size_m[0]), finite('size_m[1]', args.size_m[1])];
			return {
				layer: draw(c, args, by, 'plane', {
					kind: 'plane',
					point: [...triple('point', args.point)],
					normal: [...normal],
					size,
				}).name,
			};
		},
	});

	defineOperation({
		name: 'list_layers',
		description:
			"The layers on a car, in drawing order: each one's name, kind (field or drawing), who made it, whether it shows, and its colour, width, marks and painted faces or its field.",
		parameters: { type: 'object', properties: { case_id: caseId }, additionalProperties: false },
		run({ case_id }) {
			const c = session.withResult(case_id);
			return { case_id: c.id, layers: storeOf(c).layers.map(layerSummary) };
		},
	});

	defineOperation({
		name: 'set_layer',
		description:
			'Show, hide, rename or restyle a layer. Showing a field layer colours the car by its field; hiding it goes back to pressure. A new colour repaints the faces the layer paints.',
		parameters: {
			type: 'object',
			properties: {
				layer: { type: 'string', description: "The layer's name." },
				visible: { type: 'boolean' },
				name: { type: 'string', description: 'A new name.' },
				colour: colourArg,
				width_px: widthArg,
				case_id: caseId,
			},
			required: ['layer'],
			additionalProperties: false,
		},
		run({ layer: name, visible, name: rename, colour, width_px, case_id }) {
			const c = session.withResult(case_id);
			const layer = needLayer(c, nameOf(name));
			const store = storeOf(c);
			if (rename !== undefined) {
				const next = nameOf(rename);
				if (next !== layer.name && layerOn(store, next)) throw new Error(`Case ${c.id} already has a layer "${next}".`);
				if (layer.kind === 'field' && next !== layer.name) {
					const old = layer.field.key;
					const key = keyOf(next);
					const map = view.maps?.[old];
					layer.name = next;
					layer.field.key = key;
					offerField(key, next, layer.field.unit, layer.field.source, MAPS[old].start, MAPS[old].diverging);
					if (map) view.maps[key] = map;
					if (view.field === old) view.field = key;
					withdrawField(old);
				}
				layer.name = next;
			}
			if (colour !== undefined || width_px !== undefined) {
				if (layer.kind === 'field')
					throw new Error(`"${layer.name}" is a field; change its colours with set_colour_scale.`);
				if (colour !== undefined) {
					layer.colour = colourOf('colour', colour);
					if (layer.paint) repaint(layer.paint, layer.colour);
				}
				if (width_px !== undefined) layer.width = widthOf('width_px', width_px);
			}
			if (visible !== undefined) {
				if (layer.kind === 'field') {
					if (visible) view.field = layer.field.key;
					else if (view.field === layer.field.key) view.field = 'cp';
				} else {
					layer.visible = Boolean(visible);
				}
			}
			changed(c, 'legend', 'markers');
			return layerSummary(layer);
		},
	});

	/** The layers a removal takes: the one named, or with none named every layer the assistant made. */
	const removing = (c, name) => {
		const layers = storeOf(c).layers;
		return name === undefined
			? layers.filter((layer) => layer.by === 'assistant')
			: layers.filter((layer) => layer.name === name);
	};

	defineOperation({
		name: 'remove_layer',
		description:
			'Remove a layer from a car, or with no layer named every layer the assistant made on it. Removing a layer the engineer made asks them first. Returns the names removed.',
		parameters: {
			type: 'object',
			properties: {
				layer: { type: 'string', description: "The layer's name; default every layer the assistant made." },
				case_id: caseId,
			},
			additionalProperties: false,
		},
		confirm({ layer, case_id }) {
			const c = case_id === undefined ? session.active : session.cases.find((each) => each.id === case_id);
			return Boolean(c && stores.has(c) && removing(c, layer).some((each) => each.by === 'engineer'));
		},
		run({ layer: name, case_id }) {
			const c = session.withResult(case_id);
			if (name !== undefined) needLayer(c, nameOf(name));
			const gone = removing(c, name);
			const store = storeOf(c);
			store.layers = store.layers.filter((layer) => !gone.includes(layer));
			for (const layer of gone) if (layer.kind === 'field') withdrawField(layer.field.key);
			changed(c, 'legend', 'markers');
			return { case_id: c.id, removed: gone.map((layer) => layer.name) };
		},
	});

	return {
		apply,
		fields: (c) =>
			c && stores.has(c)
				? storeOf(c)
						.layers.filter((layer) => layer.kind === 'field')
						.map((layer) => layer.field.key)
				: [],
		layersOf: (c) => (c && stores.has(c) ? storeOf(c).layers : []),
		paintLabels: (ctx, scale) => drawing.paintLabels(ctx, scale),
	};
}
