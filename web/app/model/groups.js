// The model's inputs as the page tells them: groups of columns that the models compared in training were given
// or not given together, each with its colour, its words and the accuracy it added; and the compared models
// themselves, in those groups' terms. Everything else comes from the model card, so a model trained on other groups
// brings its own, and a group with no words here stops the page with an error naming it.

/** The learners of this model's own kind: one network on a point's inputs, or the five of them. */
const OWN = new Set(['mlp', 'ensemble']);

/**
 * Words for each group, by the kind of its first input: its name, its colour and the physical reason it is there; and,
 * for a group that reads the same things over and over, the column field it repeats over (one station of the
 * sections, one size of nearby surface) and, if not the columns' own order, the order its reads are listed in.
 */
const WORDS = {
	position: {
		name: 'Position and facing',
		color: '#2f739c',
		reason: 'Surface facing the wind takes high pressure; surface facing away sits in the low pressure of the wake.',
	},
	surface: {
		name: 'Nearby surface',
		color: '#a35b1a',
		reason: 'How the surface curves around the point: flow speeds up over a crest, and pressure falls.',
		repeat: 'scale',
	},
	section: {
		name: 'Slices across the car',
		color: '#197b80',
		reason: 'Air speeds up where the car grows across and slows where it narrows: how big each slice is, and how far it reaches out, up and down from the point.',
		repeat: 'station',
		reads: ['outboard', 'above', 'below', 'area'],
	},
};

/**
 * @typedef {Object} InputGroup
 * @property {string} id         the kind of its first input
 * @property {string} name
 * @property {string} reason
 * @property {string} color
 * @property {string[]} kinds    the column kinds in it
 * @property {number[]} columns  indices into `ModelCard.columns`
 * @property {{ field: string, values: number[], columns: number[][] } | null} repeat  for a group that reads the same
 *   things at each of several values of a column field: the values, ascending, and the columns read at each, in the
 *   order they are listed
 * @property {{ before: number, after: number } | null} step  cross-validation R² of this model's kind of network
 *   given the groups before it, and given this one too; null when no such pair was compared
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
		const { repeat: field, reads, ...words } = WORDS[id];
		return { id, ...words, kinds, columns: indices, repeat: field ? repeatOf(columns, indices, field, reads) : null, step: null };
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
		next.group.step = { before, after: next.row.crossValidationR2 };
		ordered.push(next.group);
		before = next.row.crossValidationR2;
	}
	return [...ordered, ...groups.filter((group) => !ordered.includes(group))];
}

/** A group's columns by the value of `field` they are read at, each in `reads` order when given, else in column order. */
function repeatOf(columns, indices, field, reads) {
	const values = [...new Set(indices.map((i) => columns[i][field]))].sort((a, b) => a - b);
	const place = (i) => (reads ? reads.indexOf(columns[i].read) : i);
	return {
		field,
		values,
		columns: values.map((value) => indices.filter((i) => columns[i][field] === value).sort((a, b) => place(a) - place(b))),
	};
}

/** The group a column belongs to. */
export function groupOf(groups, column) {
	return groups.find((group) => group.columns.includes(column));
}

/** The groups a compared model was given. */
export function groupsGiven(groups, row) {
	return groups.filter((group) => group.kinds.every((kind) => row.inputs.includes(kind)));
}

/**
 * A compared model in plain words. Single networks differ only in their inputs, so each is named by them: the first
 * group alone, or the groups added to it.
 */
export function learnerName(row, groups) {
	switch (row.learner) {
		case 'constant':
			return 'Guess the training average';
		case 'pointnet':
			return 'Learned whole-car summary';
		case 'mlp': {
			const [first, ...added] = groupsGiven(groups, row);
			return added.length === 0
				? `One network, ${first.name.toLowerCase()}`
				: `One network + ${added.map((g) => g.name.toLowerCase().replace(/^the /, '')).join(' + ')}`;
		}
		case 'trees':
			return 'Boosted trees';
		case 'ensemble':
			return 'This model';
		default:
			throw new Error(`The model card compares a learner the page does not know: “${row.learner}”.`);
	}
}
