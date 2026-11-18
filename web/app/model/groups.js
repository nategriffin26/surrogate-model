// The model's inputs as the page tells them: groups of columns that the models compared in training were given
// or not given together, each with its colour, its words and the accuracy it added; and the compared models
// themselves, in those groups' terms. Everything comes from the model card, so a model trained on other groups
// brings its own, and a group with no words here stops the page with an error naming it.

/** Group colours, in the order the groups are added. */
const PALETTE = ['#2f739c', '#9a55e8', '#a35b1a', '#197b80', '#237f50', '#a83258'];

/** The learners of this model's own kind: one network on a point's inputs, or the five of them. */
const OWN = new Set(['mlp', 'ensemble']);

/** Words for each group, by the kind of its first input: its name and the physical reason it is there. */
const WORDS = {
	position: {
		name: 'Position and facing',
		reason: 'Surface facing the wind takes high pressure; surface facing away sits in the low pressure of the wake.',
	},
	bend: {
		name: 'Nearby surface',
		reason: 'How the surface curves around the point: flow speeds up over a crest, and pressure falls.',
	},
	slice: {
		name: 'The car ahead and behind',
		reason: 'What the flow met before the point and meets after it: a long hood or a sharp drop changes the pressure here.',
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
		return { id, ...WORDS[id], kinds, columns: indices, color: '', step: null };
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
	const all = [...ordered, ...groups.filter((group) => !ordered.includes(group))];
	all.forEach((group, i) => (group.color = PALETTE[i % PALETTE.length]));
	return all;
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
