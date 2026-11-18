// The assistant's replies as the Markdown it writes: paragraphs, headings, lists, tables, code, bold, italic and
// inline code, built as DOM nodes (never as HTML, so nothing in a reply can run or load anything). Links stay text.

import { h } from '../app/dom.js';

/** Block elements for `text`. Safe to call again on a longer text as it streams in. */
export function renderMarkdown(text) {
	const lines = text.replaceAll('\r', '').split('\n');
	const blocks = [];
	for (let i = 0; i < lines.length; ) {
		const line = lines[i];
		if (line.trim() === '') {
			i++;
		} else if (line.startsWith('```')) {
			const code = [];
			for (i++; i < lines.length && !lines[i].startsWith('```'); i++) code.push(lines[i]);
			i++;
			blocks.push(h('pre', { class: 'md-code' }, code.join('\n')));
		} else if (/^#{1,6}\s/.test(line)) {
			blocks.push(h('p', { class: 'md-heading' }, inline(line.replace(/^#+\s+/, ''))));
			i++;
		} else if (isTableRow(line) && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1])) {
			const rows = [];
			for (; i < lines.length && isTableRow(lines[i]); i++) rows.push(cells(lines[i]));
			const [head, , ...body] = rows;
			blocks.push(h('div', { class: 'md-table' }, h('table', {},
				h('thead', {}, h('tr', {}, head.map((cell) => h('th', {}, inline(cell))))),
				h('tbody', {}, body.map((row) => h('tr', {}, row.map((cell) => h('td', { class: numeric(cell) ? 'num' : null }, inline(cell)))))))));
		} else if (/^\s*([-*•]|\d+[.)])\s+/.test(line)) {
			const ordered = /^\s*\d/.test(line);
			const items = [];
			for (; i < lines.length && /^\s*([-*•]|\d+[.)])\s+/.test(lines[i]); i++) items.push(h('li', {}, inline(lines[i].replace(/^\s*([-*•]|\d+[.)])\s+/, ''))));
			blocks.push(h(ordered ? 'ol' : 'ul', {}, items));
		} else {
			const paragraph = [];
			for (; i < lines.length && lines[i].trim() !== '' && !startsBlock(lines[i], lines[i + 1]); i++) paragraph.push(lines[i]);
			blocks.push(h('p', {}, paragraph.flatMap((part, k) => (k ? [h('br'), ...inline(part)] : inline(part)))));
		}
	}
	return blocks;
}

function startsBlock(line, next) {
	return line.startsWith('```') || /^#{1,6}\s/.test(line) || /^\s*([-*•]|\d+[.)])\s+/.test(line) || (isTableRow(line) && /^\s*\|?\s*:?-{2,}/.test(next ?? ''));
}

const isTableRow = (line) => /^\s*\|.*\|\s*$/.test(line);
const cells = (line) => line.trim().replace(/^\||\|$/g, '').split('|').map((cell) => cell.trim());
const numeric = (cell) => /^[−+-]?[\d.,]+\s*(%|m²|m|s|counts?)?$/.test(cell);

/** Inline spans: `code`, **bold**, *italic*; everything else as text. */
function inline(text) {
	const parts = [];
	const pattern = /`([^`]+)`|\*\*([^*]+)\*\*|\*([^*\s][^*]*)\*/g;
	let last = 0;
	for (const match of text.matchAll(pattern)) {
		if (match.index > last) parts.push(text.slice(last, match.index));
		if (match[1] !== undefined) parts.push(h('code', {}, match[1]));
		else if (match[2] !== undefined) parts.push(h('strong', {}, match[2]));
		else parts.push(h('em', {}, match[3]));
		last = match.index + match[0].length;
	}
	if (last < text.length) parts.push(text.slice(last));
	return parts;
}
