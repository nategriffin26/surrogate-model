// The view of the car as operations: which field colours it and on what scale, where the camera looks from, and
// whether mesh edges and the ground are drawn. The toolbar, the legend and the keys run these, as the assistant does.

import { VIEWS } from '../viewer/viewer.js';
import { finite, oneOf, round, triple } from './checks.js';
import { FIELDS, MAPS, RESULT_FIELDS, defaultMaps } from './fields.js';
import { bandCount } from './legend.js';
import { defineOperation } from './operations.js';

const VIEW_NAMES = Object.keys(VIEWS);

/** A colour map as an operation reports it. */
const mapResult = (field, map) => ({ field, min: round(map.lo), max: round(map.hi), ramp: map.ramp, banded: Boolean(map.banded), bands: bandCount(map) });

/** The case whose view is changed: the one on screen, with its result. */
function shown(session) {
	const c = session.active;
	if (!c?.result) throw new Error('No case with a result is on screen. Open a file or an example, or wait for its analysis.');
	return c;
}

/** The camera moves as soon as a car is on screen, while its pressure is still being predicted too. */
function carOnScreen(session) {
	if (!session.active?.surface) throw new Error('No car is on screen. Open a file or an example first.');
}

/** app: { session, viewer, update } */
export function defineDisplayOperations({ session, viewer, update }) {
	const view = session.view;

	defineOperation({
		name: 'set_field',
		description: 'Colour the car by a field. `cp`: the predicted pressure coefficient (mean of the model\'s networks). `drag`: the local pressure drag density −Cp·nₓ, Cp times the x component of the outward unit normal, negated: positive where pressure pushes the car rearward (x runs downstream). `sigma`: the spread σ(Cp) between the networks, scaled by the model\'s calibration. Section plots and probe readouts follow the field. Returns the field and its colour scale.',
		parameters: {
			type: 'object',
			properties: { field: { type: 'string', enum: RESULT_FIELDS, description: 'Field to colour the car by.' } },
			required: ['field'],
			additionalProperties: false,
		},
		run({ field }) {
			shown(session);
			view.field = oneOf('field', field, RESULT_FIELDS);
			update('toolbar', 'legend', 'viewer', 'dock', 'markers');
			return mapResult(field, view.maps[FIELDS[field].map]);
		},
	});

	defineOperation({
		name: 'set_colour_scale',
		description: 'Change the colour scale of a field: its range, its colour ramp, whether it is split into discrete bands, or reset it to the default range. Pressure starts at a fixed −0.9 to 0.9 so cars compare on one scale; the other fields start at 98 % of the surface\'s values. Every open case is drawn on the same scale. Returns the scale as it now is.',
		parameters: {
			type: 'object',
			properties: {
				field: { type: 'string', enum: RESULT_FIELDS, description: 'Field whose scale to change; default the field shown.' },
				min: { type: 'number', description: 'Value at the low end of the ramp, in the field\'s units (Cp is dimensionless).' },
				max: { type: 'number', description: 'Value at the high end of the ramp.' },
				ramp: { type: 'string', description: 'Colour ramp: for cp coolwarm, turbo or viridis; for drag tealorange, coolwarm or brownteal; for sigma purples, viridis or inferno.' },
				banded: { type: 'boolean', description: 'Split the ramp into discrete bands at round values.' },
				reset: { type: 'boolean', description: 'Back to the default range, before min and max are applied.' },
			},
			additionalProperties: false,
		},
		run({ field = view.field, min, max, ramp, banded, reset }) {
			const c = shown(session);
			oneOf('field', field, RESULT_FIELDS);
			const name = FIELDS[field].map;
			if (reset) view.maps[name] = { ...defaultMaps(c.result.mesh)[name], ramp: view.maps[name].ramp, banded: view.maps[name].banded };
			const map = view.maps[name];
			const lo = min === undefined ? map.lo : finite('min', min);
			const hi = max === undefined ? map.hi : finite('max', max);
			if (!(lo < hi)) throw new Error(`min must be below max; got min ${lo} and max ${hi}.`);
			if (ramp !== undefined) oneOf('ramp', ramp, MAPS[name].ramps);
			Object.assign(map, { lo, hi }, ramp === undefined ? {} : { ramp }, banded === undefined ? {} : { banded: Boolean(banded) });
			update('legend', 'toolbar', 'viewer');
			return mapResult(field, map);
		},
	});

	defineOperation({
		name: 'set_camera_view',
		description: 'Move the camera to a standard view of the whole car, framed to fit: isometric (from front left, above), front, rear, left (the −y side), right (the +y side), top or bottom. Every open case is seen through the same camera.',
		parameters: {
			type: 'object',
			properties: { view: { type: 'string', enum: VIEW_NAMES, description: 'Standard view.' } },
			required: ['view'],
			additionalProperties: false,
		},
		run({ view: name }) {
			carOnScreen(session);
			viewer.view(oneOf('view', name, VIEW_NAMES));
			return { view: name };
		},
	});

	defineOperation({
		name: 'fit_view',
		description: 'Frame the whole car without changing the direction the camera looks from.',
		parameters: { type: 'object', properties: {}, additionalProperties: false },
		run() {
			carOnScreen(session);
			viewer.fit();
			return {};
		},
	});

	defineOperation({
		name: 'set_projection',
		description: 'Perspective or orthographic projection. Orthographic keeps sizes true across the car, as for a side elevation; the car keeps its size at the orbit centre.',
		parameters: {
			type: 'object',
			properties: { projection: { type: 'string', enum: ['perspective', 'orthographic'] } },
			required: ['projection'],
			additionalProperties: false,
		},
		run({ projection }) {
			carOnScreen(session);
			viewer.setProjection(oneOf('projection', projection, ['perspective', 'orthographic']));
			update('toolbar');
			return { projection };
		},
	});

	defineOperation({
		name: 'fly_to',
		description: 'Fly the camera close to a surface point and look at it from outside, along the surface normal there (tipped to stay above the floor). Use it to show the engineer a place on the car: an extreme, a probe, a region. Points are in metres along the file\'s axes (x downstream, z up).',
		parameters: {
			type: 'object',
			properties: {
				point: { type: 'array', items: { type: 'number' }, minItems: 3, maxItems: 3, description: '[x, y, z] in metres.' },
				normal: { type: 'array', items: { type: 'number' }, minItems: 3, maxItems: 3, description: 'Outward direction to look from; default from the side the camera is on now.' },
			},
			required: ['point'],
			additionalProperties: false,
		},
		run({ point, normal }) {
			carOnScreen(session);
			triple('point', point);
			const from = normal ? triple('normal', normal) : viewer.camera.position.toArray().map((v, k) => v - viewer.target.getComponent(k));
			viewer.flyTo(point, from);
			return { point };
		},
	});

	defineOperation({
		name: 'set_display',
		description: 'Draw or hide the mesh edges on the car and the ground under it.',
		parameters: {
			type: 'object',
			properties: {
				edges: { type: 'boolean', description: 'Mesh edges, faded where they would crowd.' },
				ground: { type: 'boolean', description: 'The floor with the car\'s contact shadow.' },
			},
			additionalProperties: false,
		},
		run({ edges, ground }) {
			if (edges !== undefined) view.display.edges = Boolean(edges);
			if (ground !== undefined) view.display.ground = Boolean(ground);
			update('toolbar', 'viewer');
			return { ...view.display };
		},
	});

	defineOperation({
		name: 'get_view',
		description: 'What is on screen: the case shown, the field and its colour scale, the camera (projection, position and the point it orbits, in metres), edges and ground, the section and the dock tab.',
		parameters: { type: 'object', properties: {}, additionalProperties: false },
		run() {
			const c = session.active;
			const camera = viewer.cameraState();
			return {
				case_id: c?.id ?? null,
				field: view.field,
				colour_scale: view.maps ? mapResult(view.field, view.maps[FIELDS[view.field].map]) : null,
				camera: { projection: camera.projection, position_m: camera.position.map((v) => round(v, 4)), target_m: camera.target.map((v) => round(v, 4)) },
				display: { ...view.display },
				section: { ...view.section },
				dock: view.dockCollapsed ? null : view.dock,
			};
		},
	});
}
