// The Model page's words. Every sentence, label and name the page shows is a value in copy.json, keyed by where it
// sits, and the element that shows it carries data-copy="<key>". A value may hold placeholders, {name}, that the page
// fills from data (counts, R², a car's name); {Name} fills in the same value with its first letter capitalised. An
// empty value hides its element, as do words whose placeholder fills with nothing (a car whose name was emptied), and
// a `.copy-block` whose words are all hidden hides with them (model.css).

let words = null;
/** Values every placeholder may use, anywhere on the page: the model's counts. */
const shared = {};

/** Values any words may use, such as {inputs} or {networksWord}, beside each element's own. */
export function shareValues(values) {
	Object.assign(shared, values);
}

/** Read copy.json once; everything else here needs it read. */
export async function loadCopy() {
	if (words) return;
	const response = await fetch(new URL('./copy.json', import.meta.url));
	if (!response.ok) throw new Error(`The page's words could not be read: copy.json answered ${response.status}.`);
	words = await response.json();
}

/**
 * The words at `key` with their placeholders filled from `own` and the shared values: as a list of strings and nodes,
 * since a placeholder may stand for a node (a symbol with a subscript).
 * Null when the words are empty or a placeholder fills with an empty string: words naming nothing are not shown.
 */
function fill(key, own) {
	if (!(key in words)) throw new Error(`The Model page has no words for “${key}” in copy.json.`);
	if (words[key] === '') return null;
	const values = { ...shared, ...own };
	const parts = [];
	let last = 0;
	for (const match of words[key].matchAll(/\{(\w+)\}/g)) {
		const name = match[1];
		const lower = name[0].toLowerCase() + name.slice(1);
		const exact = name in values;
		if (!exact && !(lower in values)) continue;
		parts.push(words[key].slice(last, match.index));
		const value = exact ? values[name] : values[lower];
		if (value === '') return null;
		if (value instanceof Node) parts.push(value.cloneNode(true));
		else parts.push(exact ? String(value) : String(value).replace(/^./, (c) => c.toUpperCase()));
		last = match.index + match[0].length;
	}
	parts.push(words[key].slice(last));
	return parts;
}

/** Whether copy.json holds words at `key`. */
export const hasCopy = (key) => key in words;

/** The words at `key` as one string, placeholders filled, or '' when not shown: for text drawn where no element can carry them. */
export function text(key, values = {}) {
	const parts = fill(key, values);
	return parts ? parts.map((part) => (part instanceof Node ? part.textContent : part)).join('') : '';
}

/** Give `element` the words at `key`, and mark it as theirs. Returns the element. */
export function copied(element, key, values = {}) {
	element.dataset.copy = key;
	const parts = fill(key, values);
	element.replaceChildren(...(parts ?? []));
	element.toggleAttribute('hidden', !parts);
	return element;
}

const ESCAPES = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' };

/** The words at `key` as markup: a `tag` element (SVG or HTML) holding them, with `attributes`, marked as theirs. */
export function markup(tag, key, values = {}, attributes = '') {
	const filled = text(key, values).replace(/[&<>"]/g, (c) => ESCAPES[c]);
	return `<${tag} data-copy="${key}"${attributes ? ` ${attributes}` : ''}${filled === '' ? ' hidden' : ''}>${filled}</${tag}>`;
}
