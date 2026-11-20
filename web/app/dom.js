// Element builders. `h('div', {class: 'x', onclick}, ...children)`; null, false and undefined children are skipped.

const SVG_NS = 'http://www.w3.org/2000/svg';
/** A screen used by touch: a tap places what a click would, and nothing can be dropped onto the page. */
export const TOUCH = matchMedia('(pointer: coarse)');

function build(element, attrs, children) {
	for (const [k, v] of Object.entries(attrs)) {
		if (v === false || v === null || v === undefined) continue;
		if (k.startsWith('on')) element.addEventListener(k.slice(2), v);
		else if (k === 'class') element.setAttribute('class', v);
		else element.setAttribute(k, v === true ? '' : v);
	}
	for (const c of children.flat(Infinity)) {
		if (c === null || c === undefined || c === false) continue;
		element.append(c instanceof Node ? c : document.createTextNode(String(c)));
	}
	return element;
}

export const h = (tag, attrs = {}, ...children) => build(document.createElement(tag), attrs, children);
export const s = (tag, attrs = {}, ...children) => build(document.createElementNS(SVG_NS, tag), attrs, children);

/** A force coefficient's name as it is written on screen, C with its axis subscripted, after a prefix such as Δ. */
export const coefficient = (axis, prefix = '') => [prefix, 'C', h('sub', {}, axis)];

/** Replace an element's children, skipping null, false and undefined as `h` does; `replaceChildren` would write them as text. */
export function put(element, ...children) {
	element.replaceChildren();
	return build(element, {}, children);
}

/**
 * A row of radio buttons, one per option [value, text, title?]: a click runs `choose(value)`, and `show(value)` marks
 * the option chosen. `buttons` are the buttons in option order.
 */
export function segmented(label, options, choose) {
	const buttons = options.map(([value, text, title]) =>
		h('button', { type: 'button', role: 'radio', class: 'segment', title, onclick: () => choose(value) }, text),
	);
	const el = h('div', { class: 'segmented', role: 'radiogroup', 'aria-label': label }, buttons);
	const show = (chosen) =>
		options.forEach(([value], k) => buttons[k].setAttribute('aria-checked', String(value === chosen)));
	return { el, buttons, show };
}

/**
 * A note that shows why the operation a control ran failed, until that control or another runs again. `run(promise)`
 * resolves to the operation's result, or to undefined when it failed.
 */
export function failureNote() {
	const el = h('p', { class: 'failure-note', role: 'alert', hidden: true });
	async function run(promise) {
		el.hidden = true;
		try {
			return await promise;
		} catch (error) {
			el.textContent = error.message;
			el.hidden = false;
		}
	}
	return { el, run, hide: () => (el.hidden = true) };
}

/** Download a Blob under a file name. */
export function download(blob, name) {
	const a = h('a', { href: URL.createObjectURL(blob), download: name });
	document.body.append(a);
	a.click();
	a.remove();
	setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
