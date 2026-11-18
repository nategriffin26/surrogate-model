// The open cases as operations: what is open, opening a bundled example, showing a case and closing one. Files
// are opened from the workspace (`open_file` in files.js).

import { oneOf } from './checks.js';
import { EXAMPLES } from './examples.js';
import { defineOperation } from './operations.js';

/** A case as operations report it. */
export function caseSummary(c, active) {
	const r = c.result;
	return {
		case_id: c.id,
		name: c.name,
		kind: c.kind,
		status: c.status,
		shown: c === active,
		...(c.path ? { path: c.path } : {}),
		...(c.variant ? { variant_of: c.variant.root.id, changes: c.variant.changes } : {}),
		...(c.error ? { error: c.error } : {}),
		...(r ? resultSummary(r, sampled(c)) : {}),
	};
}

/** Whether a case's whole-car totals still come from an even area sample of a dropped file's polygons. */
export const sampled = (c) => c.kind === 'file' && c.result.form === 'full' && !c.exact;

/** What whole-car totals are summed over, as operations report it. */
export const totalsOver = (fromSample) => (fromSample ? 'an even area sample of the polygons; every polygon replaces it when ready' : 'every polygon');

/** The dimensions of a Result's car outside the training cars' range. */
export const outsideTraining = (r) => r.coverage.dimensions.filter((d) => d.value < d.trainMin || d.value > d.trainMax);

/**
 * A Result's size and whole-car numbers, as the result card shows them and list_cases reports them; `fromSample`
 * while every polygon is still to come. The assistant's kernel reports a predicted variant with it too.
 */
export function resultSummary(r, fromSample) {
	const t = r.totals;
	const point = (x) => x.map((v) => +v.toFixed(4));
	const extreme = (e) => ({ value: e.value, center_m: point(e.center), normal: point(e.normal) });
	return {
		cells: r.source.cells,
		length_m: r.geometry.length,
		width_m: r.geometry.width,
		height_m: r.geometry.height,
		reference_area_m2: r.geometry.frontalArea,
		totals_over: totalsOver(fromSample),
		area_m2: t.area,
		mean_cp: t.meanCp,
		cx: t.coefficients[0],
		cy: t.coefficients[1],
		cz: t.coefficients[2],
		cx_by_network: r.forces.cdPModels,
		cp_min: extreme(t.cpMin),
		cp_max: extreme(t.cpMax),
		outside_training: outsideTraining(r).map((d) => ({ dimension: d.name, value_m: d.value, training_min_m: d.trainMin, training_max_m: d.trainMax })),
		reference: r.reference ? {
			file: r.reference.file, array: r.reference.array, mapping: r.reference.mapping,
			mean_cp: r.referenceTotals.meanCp, cx: r.referenceTotals.coefficients[0], cy: r.referenceTotals.coefficients[1], cz: r.referenceTotals.coefficients[2],
			r2: r.accuracy?.r2 ?? null,
		} : null,
	};
}

/** app: { session } */
export function defineCaseOperations({ session }) {
	defineOperation({
		name: 'list_cases',
		description: 'The open cases, one per tab: each with its case_id (what other operations take), name, kind (file: a .vtp from the workspace the model ran on; example: a bundled DrivAerML car with its result precomputed; variant: a changed copy of an open car, predicted on its working mesh), status (queued, analysing, loading, ready or failed) and whether it is on screen. Once ready, the numbers the result card shows: polygon count, size, reference (frontal) area, and summed over the polygons named in totals_over the surface area, area-weighted mean Cp, the pressure force coefficients cx (pressure drag), cy and cz on the reference area (no wall shear), cx from each network, the lowest and highest Cp with the centre and outward normal of their polygon, the dimensions outside the training cars\' range, and the reference field when there is one (its file, array, how it was mapped, its own mean Cp, cx, cy and cz summed over every polygon, and the area-weighted r2 against it). Field data of a case is reached by its case_id.',
		parameters: { type: 'object', properties: {}, additionalProperties: false },
		run: () => ({ cases: session.cases.map((c) => caseSummary(c, session.active)) }),
	});

	defineOperation({
		name: 'open_example',
		description: `Open a bundled example as a new case: a DrivAerML car from outside the training set whose prediction is precomputed on its 600k-triangle working mesh, so it is ready in a moment. Examples: ${EXAMPLES.map((e) => `${e.id} (${e.label})`).join(', ')}. Returns the new case_id; the case is shown.`,
		parameters: {
			type: 'object',
			properties: { example: { type: 'string', enum: EXAMPLES.map((e) => e.id) } },
			required: ['example'],
			additionalProperties: false,
		},
		async run({ example }) {
			const { id, label } = EXAMPLES.find((e) => e.id === oneOf('example', example, EXAMPLES.map((e) => e.id)));
			const opening = session.openExample(id, label);
			const c = session.active;
			await opening;
			if (c.status === 'failed') throw new Error(c.error);
			return caseSummary(c, session.active);
		},
	});

	defineOperation({
		name: 'show_case',
		description: 'Put an open case on screen, as clicking its tab does. The camera, field, colour scale and section stay as they are, so cars compare from the same view.',
		parameters: {
			type: 'object',
			properties: { case_id: { type: 'integer', description: 'From list_cases.' } },
			required: ['case_id'],
			additionalProperties: false,
		},
		run({ case_id }) {
			session.activate(session.byId(case_id));
			return caseSummary(session.active, session.active);
		},
	});

	defineOperation({
		name: 'close_case',
		description: 'Close an open case: its tab goes, its analysis stops if it is still running, and its memory is released. Its file stays in the workspace and can be opened again.',
		parameters: {
			type: 'object',
			properties: { case_id: { type: 'integer', description: 'From list_cases.' } },
			required: ['case_id'],
			additionalProperties: false,
		},
		confirm: true,
		run({ case_id }) {
			const c = session.byId(case_id);
			session.close(c);
			return { closed: case_id };
		},
	});
}
