// Element builders. `h('div', {class: 'x', onclick}, ...children)`; null, false and undefined children are skipped.

const SVG_NS = 'http://www.w3.org/2000/svg';

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

/** Replace an element's children. */
export function put(element, ...children) {
	element.replaceChildren();
	return build(element, {}, children);
}

/** Download a Blob under a file name. */
export function download(blob, name) {
	const a = h('a', { href: URL.createObjectURL(blob), download: name });
	document.body.append(a);
	a.click();
	a.remove();
	setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
