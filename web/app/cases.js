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
		...(c.error ? { error: c.error } : {}),
		...(r ? {
			cells: r.source.cells,
			length_m: r.geometry.length,
			width_m: r.geometry.width,
			height_m: r.geometry.height,
			reference_area_m2: r.geometry.frontalArea,
		} : {}),
	};
}

/** app: { session } */
export function defineCaseOperations({ session }) {
	defineOperation({
		name: 'list_cases',
		description: 'The open cases, one per tab: each with its case_id (what other operations take), name, kind (file: a .vtp from the workspace the model ran on; example: a bundled DrivAerML car with its result precomputed), status (queued, analysing, loading, ready or failed), whether it is on screen, and once ready its polygon count, size and reference (frontal) area. Field data of a case is reached by its case_id.',
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
