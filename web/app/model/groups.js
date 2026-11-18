// The model's inputs as the page tells them: groups of columns that the models compared in training were given
// or not given together, each with its colour and its words. Everything else comes from the model card, so a model
// trained on other groups brings its own, and a group with nothing here stops the page with an error naming it.

/** The learners of this model's own kind: one network on a cell's inputs, or the five of them. */
const OWN = new Set(['mlp', 'ensemble']);

/**
 * Each group, by the kind of its first input (the engine's column kind, the one place the page names it): the stem of
 * its keys in copy.json (its legend at `overview.legend.<words>`) and its colour; and, for a group that reads the same things
 * over and over, the column field it repeats over, of which the page shows one value at a time (one size of nearby
 * surface, one station of the slices).
 */
const WORDS = {
	position: { words: 'position', color: '#2f739c' },
	surface: { words: 'surface', color: '#a35b1a', repeat: 'scale' },
	slice: { words: 'slices', color: '#197b80', repeat: 'station' },
};

/**
 * @typedef {Object} InputGroup
 * @property {string} id         the kind of its first input
 * @property {string} words      the stem of its keys in copy.json; its legend is at `overview.legend.<words>`
 * @property {string} color
 * @property {string[]} kinds    the column kinds in it
 * @property {number[]} columns  indices into `ModelCard.columns`
 * @property {string | null} repeat  for a group that reads the same things at each of several values of a column
 *   field, that field
 */

/**
 * The groups, in the order a staircase of compared networks adds them: from the constant baseline, each step adds
 * the one group that, given with those before it, scored best. Groups no step adds follow in column order.
 * @param {import('../../engine/result.js').ModelCard} card
 * @returns {InputGroup[]}
 */
export function inputGroups(card) {
	const { columns, comparison } = card;
	// Kinds that every compared model was given or not given together form one group.
	const byPattern = new Map();
	columns.forEach((column, index) => {
		const pattern = comparison.map((row) => row.inputs.includes(column.kind)).join();
		if (!byPattern.has(pattern)) byPattern.set(pattern, { kinds: [], columns: [] });
		const group = byPattern.get(pattern);
		if (!group.kinds.includes(column.kind)) group.kinds.push(column.kind);
		group.columns.push(index);
	});
	const groups = [...byPattern.values()].map(({ kinds, columns: indices }) => {
		const id = columns[indices[0]].kind;
		if (!(id in WORDS)) throw new Error(`The model has a group of inputs the page has no words for: “${id}”.`);
		return { repeat: null, ...WORDS[id], id, kinds, columns: indices };
	});

	const own = comparison.filter((row) => OWN.has(row.learner));
	const given = (row, group) => group.kinds.every((kind) => row.inputs.includes(kind));
	const exactly = (row, set) => groups.every((group) => given(row, group) === set.includes(group));
	const ordered = [];
	let before = comparison.find((row) => row.learner === 'constant')?.crossValidationR2 ?? null;
	while (before !== null && ordered.length < groups.length) {
		const next = groups
			.filter((group) => !ordered.includes(group))
			.map((group) => ({ group, row: own.filter((row) => exactly(row, [...ordered, group])).sort((a, b) => b.crossValidationR2 - a.crossValidationR2)[0] }))
			.filter(({ row }) => row)
			.sort((a, b) => b.row.crossValidationR2 - a.row.crossValidationR2)[0];
		if (!next) break;
		ordered.push(next.group);
		before = next.row.crossValidationR2;
	}
	return [...ordered, ...groups.filter((group) => !ordered.includes(group))];
}

/** The group a column belongs to. */
export function groupOf(groups, column) {
	return groups.find((group) => group.columns.includes(column));
}
