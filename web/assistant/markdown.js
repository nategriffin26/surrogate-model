// The assistant's replies as the Markdown it writes: paragraphs, headings, lists, tables, code, bold, italic and
// inline code, built as DOM nodes (never as HTML, so nothing in a reply can run or load anything). Links stay text.

import { h } from '../app/dom.js';

/**
 * The blocks of `text`, each `{ source, element }`. A block whose source is the same as the block in its place in
 * `previous` keeps that element, so a reply drawn again as it streams rebuilds only the block still being written.
 * While `streaming`, only what is settled is drawn (see `settled`), so half-written markup never shows and nothing
 * already drawn changes shape when the rest arrives.
 */
export function renderMarkdown(text, { streaming = false, previous = [] } = {}) {
	let lines = text.replaceAll('\r', '').split('\n');
	if (streaming) lines = settled(lines);
	const blocks = [];
	for (let i = 0; i < lines.length;) {
		const start = i;
		const line = lines[i];
		let build = null;
		if (line.trim() === '') {
			i++;
		} else if (line.startsWith('```')) {
			const code = [];
			for (i++; i < lines.length && !lines[i].startsWith('```'); i++) code.push(lines[i]);
			i++;
			build = () => h('pre', { class: 'md-code' }, code.join('\n'));
		} else if (HEADING.test(line)) {
			i++;
			build = () => h('p', { class: 'md-heading' }, inline(line.replace(/^#+\s+/, '')));
		} else if (isTableRow(line) && i + 1 < lines.length && TABLE_RULE.test(lines[i + 1])) {
			const rows = [];
			for (; i < lines.length && isTableRow(lines[i]); i++) rows.push(tableCells(lines[i]));
			const [head, , ...body] = rows;
			build = () =>
				h(
					'div',
					{ class: 'md-table' },
					h(
						'table',
						{},
						h(
							'thead',
							{},
							h(
								'tr',
								{},
								head.map((cell) => h('th', {}, inline(cell))),
							),
						),
						h(
							'tbody',
							{},
							body.map((row) =>
								h(
									'tr',
									{},
									row.map((cell) => h('td', { class: numeric(cell) ? 'num' : null }, inline(cell))),
								),
							),
						),
					),
				);
		} else if (LIST_ITEM.test(line)) {
			const ordered = /^\s*\d/.test(line);
			const items = [];
			for (; i < lines.length && LIST_ITEM.test(lines[i]); i++) items.push(lines[i].replace(LIST_ITEM, ''));
			build = () =>
				h(
					ordered ? 'ol' : 'ul',
					{},
					items.map((item) => h('li', {}, inline(item))),
				);
		} else {
			const paragraph = [];
			for (; i < lines.length && lines[i].trim() !== '' && !startsBlock(lines[i], lines[i + 1]); i++)
				paragraph.push(lines[i]);
			build = () =>
				h(
					'p',
					{},
					paragraph.flatMap((part, k) => (k ? [h('br'), ...inline(part)] : inline(part))),
				);
		}
		if (!build) continue;
		const source = lines.slice(start, i).join('\n');
		const kept = previous[blocks.length];
		blocks.push(kept?.source === source ? kept : { source, element: build() });
	}
	return blocks;
}

const HEADING = /^#{1,6}\s/;
const LIST_ITEM = /^\s*([-*•]|\d+[.)])\s+/;
/** The line under a table's header: dashes between pipes, optionally with alignment colons. */
const TABLE_RULE = /^\s*\|?\s*:?-{2,}/;

/**
 * The lines of a reply still streaming that are safe to draw. The last line has no newline after it yet. A table is
 * held back until the line after it has begun, so it appears whole instead of re-flowing its columns row by row; an
 * unfinished line that may yet open a table, a list or code is held back; and on the line being written a ** or `
 * that is not yet closed is left out rather than shown as a stray mark.
 */
function settled(lines) {
	const unfinished = lines.at(-1) !== '';
	let end = unfinished ? lines.length : lines.length - 1;
	if (unfinished && /^\s*(\||`|([-*•]|\d+[.)]?)\s*$)/.test(lines[end - 1])) end--;
	while (end > 0 && isTableLike(lines[end - 1])) end--;
	const kept = lines.slice(0, end);
	if (unfinished && end === lines.length) kept[end - 1] = withoutOpenMarks(kept[end - 1]);
	return kept;
}

/** `line` without its last ** or ` when that mark opens a span not yet closed. */
function withoutOpenMarks(line) {
	for (const mark of ['**', '`']) {
		if (line.split(mark).length % 2 === 0) {
			const at = line.lastIndexOf(mark);
			line = line.slice(0, at) + line.slice(at + mark.length);
		}
	}
	return line;
}

function startsBlock(line, next) {
	return (
		line.startsWith('```') ||
		HEADING.test(line) ||
		LIST_ITEM.test(line) ||
		(isTableRow(line) && TABLE_RULE.test(next ?? ''))
	);
}

const isTableRow = (line) => /^\s*\|.*\|\s*$/.test(line);
const isTableLike = (line) => /^\s*\|/.test(line);
/** A table row's cells, split at its pipes; an escaped pipe (\|, as in \|ΔCp\|) is a literal | inside its cell. */
export const tableCells = (line) =>
	line
		.trim()
		.replace(/^\|/, '')
		.replace(/(?<!\\)\|$/, '')
		.split(/(?<!\\)\|/)
		.map((cell) => cell.trim().replaceAll('\\|', '|'));
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
