// The model's inputs as the page tells them: one group per kind of input column (`ModelColumn.kind`), in column
// order, each with its colour. The page is drawn for the kinds below; a model with an input of any other kind stops
// the page with an error naming it.

/** The page's colours: each kind of input's, and the networks' (the theme's --steel, --orange, --teal and --link). */
export const COLORS = { position: '#2f739c', surface: '#a35b1a', slice: '#197b80', network: '#0b6fbf' };

/**
 * For a kind that reads the same things over and over, the column field it repeats over, of which the page shows one
 * value at a time (one size of nearby surface, one station of the slices).
 */
const REPEATS = { position: null, surface: 'scale', slice: 'station' };

/**
 * @typedef {Object} InputGroup
 * @property {string} id         the column kind; also the stem of its keys in copy.json (`overview.legend.<id>`)
 * @property {string} color
 * @property {number[]} columns  indices into `ModelCard.columns`
 * @property {string | null} repeat  for a group that reads the same things at each of several values of a column
 *   field, that field
 */

/**
 * The groups, in column order.
 * @param {import('../../engine/result.js').ModelCard} card
 * @returns {InputGroup[]}
 */
export function inputGroups({ columns }) {
	const groups = [];
	columns.forEach((column, index) => {
		let group = groups.find((g) => g.id === column.kind);
		if (!group) {
			if (!(column.kind in REPEATS))
				throw new Error(`The model has a group of inputs the page has no words for: “${column.kind}”.`);
			group = { id: column.kind, color: COLORS[column.kind], columns: [], repeat: REPEATS[column.kind] };
			groups.push(group);
		}
		group.columns.push(index);
	});
	return groups;
}

/** The group a column belongs to. */
export function groupOf(groups, column) {
	return groups.find((group) => group.columns.includes(column));
}
