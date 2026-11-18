// Every analysis action the tool offers is one operation. The engineer's controls run it and the assistant
// runs it, through this one registry, so the two can never disagree about what an action does or which number
// it returns. An operation changes what is on screen (a cut appears, the camera moves) and returns the numbers.
//
// Operations act on the cases, the view of the car and the workspace's files. Arranging the interface (choosing or
// folding a dock tab, opening a panel, resizing the dock) and the engineer's own uploads are not operations.

/**
 * @typedef {Object} Operation
 * @property {string} name         snake_case, verb first, as the assistant will call it: `set_section`, `read_comparison`.
 * @property {string} description  For the assistant: what it does, when to use it, what it returns. One to three sentences.
 * @property {object} parameters   JSON Schema (type "object") of the arguments, each property with a description and unit.
 * @property {boolean | ((args: object, context: OperationContext) => boolean)} [confirm]
 *   Whether the assistant asks the engineer before `run`, for calls by the assistant only: true for closing a case,
 *   downloading anything and deleting a file; a function of the arguments and the context where that depends on
 *   what it acts on (removing a layer the engineer made, moving a file the engineer wrote).
 * @property {(args: object, context: OperationContext) => object | Promise<object>} run
 *   Does it and returns a small JSON-serialisable result: the numbers asked for, with units in the key names
 *   (`area_m2`, `mean_cp`), and the ids of anything it created (`set_id`, `case_id`). Never whole arrays:
 *   field data is reached by case id (the plots, the exports, the assistant's Python kernel). Throws an Error
 *   whose message says what was wrong with the request and what would work.
 *
 * @typedef {Object} OperationContext
 * @property {import('./session.js').Session} session  The open cases and the shared view state.
 * @property {object} viewer                           The 3D viewer.
 * @property {import('../engine/result.js').Engine} engine
 * @property {'engineer' | 'assistant'} by             Who asked, for the step list.
 */

/** @type {Map<string, Operation>} */
export const operations = new Map();

/** Add an operation to the registry. Names are unique; the schema is what the assistant is shown. */
export function defineOperation(operation) {
	if (operations.has(operation.name)) throw new Error(`Operation ${operation.name} is defined twice.`);
	if (operation.parameters?.type !== 'object') throw new Error(`Operation ${operation.name} needs an object schema.`);
	operations.set(operation.name, operation);
}

/** Subscribers hear of every operation run, by anyone: the assistant's step list is one. */
const listeners = new Set();
export const onOperation = (listener) => (listeners.add(listener), () => listeners.delete(listener));

/** Run an operation by name. Rejects with the operation's own error, or for an unknown name. */
export async function runOperation(name, args, context) {
	const operation = operations.get(name);
	if (!operation) throw new Error(`There is no operation named ${name}.`);
	const result = await operation.run(args ?? {}, context);
	for (const listener of listeners) listener({ name, args, result, by: context.by });
	return result;
}
