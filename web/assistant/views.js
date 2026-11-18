// The parts of a conversation in the assistant's panel, as DOM: the model's thinking and its reply streaming in as
// Markdown, each tool call as a step the engineer can open, and the figures and tables a Python run shows.

import { download, h, put, s } from '../app/dom.js';
import { icon } from '../app/icons.js';
import { renderMarkdown } from './markdown.js';

/**
 * The model's thinking for one request, from the moment its first piece arrives: open while it streams, then
 * folded to "Thought for 4 s" (timed from `requested`) unless the engineer opened or closed it meanwhile. When the
 * thinking came without text, the row says only that it thought and for how long, and does not open.
 */
export function thoughtView(requested) {
	const row = disclosure('thought', 'Thinking');
	const text = markdownView('thought-text');
	row.content.append(text.el);
	row.setOpen(true);
	let touched = false;
	row.toggle.addEventListener('click', () => (touched = true));
	return {
		el: row.el,
		add(piece) {
			if (piece) text.add(piece, () => (text.el.scrollTop = text.el.scrollHeight));
		},
		finish() {
			const said = text.finish().trim();
			row.finish('done', `Thought for ${Math.max(1, Math.round((performance.now() - requested) / 1000))} s`);
			if (!said) {
				row.toggle.disabled = true;
				row.setOpen(false);
				return;
			}
			row.setDetail(said.split('\n')[0].replaceAll('**', ''));
			if (!touched) row.setOpen(false);
		},
	};
}

/** Streamed Markdown, drawn once a frame; returns its whole text when finished. */
export function markdownView(className = 'answer') {
	const el = h('div', { class: className });
	let text = '';
	let blocks = [];
	let frame = 0;
	let drawn = null;
	const draw = (streaming) => {
		blocks = renderMarkdown(text, { streaming, previous: blocks });
		blocks.forEach(({ element }, i) => {
			if (el.children[i] !== element) el.insertBefore(element, el.children[i] ?? null);
		});
		while (el.children.length > blocks.length) el.lastElementChild.remove();
		drawn?.();
	};
	return {
		el,
		add(delta, then) {
			text += delta;
			drawn = then;
			frame ||= requestAnimationFrame(() => {
				frame = 0;
				draw(true);
			});
		},
		finish() {
			cancelAnimationFrame(frame);
			frame = 0;
			draw(false);
			return text;
		},
	};
}

/**
 * A tool call as a row the engineer can open to see what was asked and what came back. `python` marks the kernel's
 * tool; `schemaOf(name)` gives an operation's argument schema, for the units in the one-line summary.
 */
export function stepView(name, { python, schemaOf }) {
	const row = disclosure('step', python ? 'Python' : sentence(name));
	const subs = h('div', { class: 'step-subs' });
	const outputs = h('div', { class: 'step-outputs' });
	row.el.insertBefore(subs, row.body);
	row.el.append(outputs);
	let printed = null;
	return {
		el: row.el,
		begin(args) {
			row.setDetail(python ? firstLine(args.code) : summary(args, schemaOf(name)));
			row.content.append(
				h('p', { class: 'step-label' }, 'Asked'),
				h('pre', {}, python ? args.code : JSON.stringify(args, null, 2)),
			);
		},
		output(item) {
			if (item.kind === 'text') {
				if (!printed) row.content.append(h('p', { class: 'step-label' }, 'Printed'), (printed = h('pre')));
				printed.append(item.stream === 'stderr' ? h('span', { class: 'stderr' }, item.text) : item.text);
			} else if (item.kind === 'figure') {
				outputs.append(figureView(item));
			} else if (item.kind === 'table') {
				outputs.append(tableView(item));
			}
		},
		/** An operation the Python run called, as a line under the step. */
		sub(subName, subArgs) {
			const sub = h(
				'div',
				{ class: 'step-sub running' },
				h('span', { class: 'row-state' }),
				h('span', { class: 'row-name step-name' }, sentence(subName)),
				h('span', { class: 'row-detail' }, summary(subArgs, schemaOf(subName))),
			);
			subs.append(sub);
			return { finish: (outcome) => sub.classList.replace('running', endState(outcome)) };
		},
		finish(outcome, error, result) {
			row.finish(outcome);
			if (result !== undefined)
				row.content.append(h('p', { class: 'step-label' }, 'Returned'), h('pre', {}, JSON.stringify(result, null, 2)));
			if (error)
				row.content.append(
					h('p', { class: 'step-label' }, python ? 'Error' : 'Refused'),
					h('pre', { class: 'step-error' }, error),
				);
		},
	};
}

/**
 * The row a thinking block and a tool step share: state mark, name, a one-line summary, how it ended when not
 * well, and a chevron; it opens a body below. Running, the mark is blue and the name shimmers.
 */
function disclosure(kind, name) {
	const nameEl = h('span', { class: kind === 'step' ? 'row-name step-name' : 'row-name' }, name);
	const detail = h('span', { class: 'row-detail' });
	const ending = h('span', { class: 'row-ending' });
	const toggle = h(
		'button',
		{ type: 'button', class: 'row', 'aria-expanded': 'false' },
		h('span', { class: 'row-state' }),
		nameEl,
		detail,
		ending,
		h('span', { class: 'row-chevron' }, icon('chevron')),
	);
	const content = h('div', { class: 'row-content' });
	const body = h('div', { class: 'row-body' }, h('div', { class: 'row-clip' }, content));
	const el = h('div', { class: `${kind} running` }, toggle, body);
	const setOpen = (open) => {
		el.classList.toggle('open', open);
		toggle.setAttribute('aria-expanded', String(open));
	};
	toggle.addEventListener('click', () => setOpen(!el.classList.contains('open')));
	return {
		el,
		toggle,
		body,
		content,
		setOpen,
		setDetail: (text) => put(detail, text),
		/** `outcome` 'done', or how it ended otherwise: 'failed', 'timed out', 'declined' or 'stopped'. */
		finish(outcome, newName) {
			el.classList.remove('running');
			el.classList.add(endState(outcome));
			if (outcome !== 'done') put(ending, sentence(outcome));
			if (newName) put(nameEl, newName);
		},
	};
}

/** The class a row or line keeps once it has ended: 'done', 'ended' (declined or stopped) or 'failed'. */
function endState(outcome) {
	if (outcome === 'done') return 'done';
	return ['declined', 'stopped'].includes(outcome) ? 'ended' : 'failed';
}

function figureView({ png, title }) {
	const blob = new Blob([png], { type: 'image/png' });
	const url = URL.createObjectURL(blob);
	const name = `${
		title
			.replace(/[^\w\- ]+/g, '')
			.trim()
			.slice(0, 60) || 'figure'
	}.png`;
	// The size from the PNG header, so the figure's space is kept before it decodes and nothing below it jumps.
	const header = new DataView(png.buffer, png.byteOffset, 24);
	return h(
		'figure',
		{ class: 'output-figure' },
		h('img', { src: url, alt: title, width: header.getUint32(16), height: header.getUint32(20) }),
		h(
			'button',
			{
				type: 'button',
				class: 'icon-button figure-save',
				title: 'Download',
				'aria-label': `Download ${name}`,
				onclick: () => download(blob, name),
			},
			icon('download'),
		),
	);
}

function tableView({ columns, rows, total }) {
	return h(
		'div',
		{ class: 'output-table' },
		h(
			'div',
			{ class: 'output-table-scroll' },
			h(
				'table',
				{},
				h(
					'thead',
					{},
					h(
						'tr',
						{},
						columns.map((column) => h('th', {}, column)),
					),
				),
				h(
					'tbody',
					{},
					rows.map((row) =>
						h(
							'tr',
							{},
							row.map((cell) => h('td', { class: /^[−+-]?[\d.]+(e[−+-]?\d+)?$/i.test(cell) ? 'num' : null }, cell)),
						),
					),
				),
			),
		),
		total > rows.length
			? h('p', { class: 'output-table-more' }, `${rows.length} of ${total.toLocaleString()} rows`)
			: null,
	);
}

/** `delete_file` → "Delete file". */
export const sentence = (name) => name.charAt(0).toUpperCase() + name.slice(1).replaceAll('_', ' ');

/** Units by the suffix of an argument's name, as the operations name them, or by its schema's "in metres". */
const UNITS = { m: 'm', m2: 'm²', mm: 'mm', deg: '°', s: 's', pct: '%' };
const UNIT_WORDS = { metres: 'm', millimetres: 'mm', degrees: '°', seconds: 's' };

/**
 * Arguments in a line, numbers to three decimals: `case 1 · point [-0.031, 0.679, -0.313] m`. The opened step shows
 * them exactly.
 */
function summary(args, schema = {}) {
	return Object.entries(args ?? {})
		.map(([key, value]) => {
			if (key === 'case_id') return `case ${value}`;
			const suffix = key.match(/_(m|m2|mm|deg|s|pct)$/);
			const word = schema[key]?.description?.match(/\bin (metres|millimetres|degrees|seconds)\b/);
			const unit = suffix ? UNITS[suffix[1]] : word ? UNIT_WORDS[word[1]] : null;
			const label = (suffix ? key.slice(0, -suffix[0].length) : key).replaceAll('_', ' ');
			return `${label} ${rounded(value)}${unit ? ` ${unit}` : ''}`;
		})
		.join(' · ');
}

function rounded(value) {
	if (typeof value === 'number') return String(Math.round(value * 1000) / 1000);
	// Arrays sent from Python arrive whole; their size says enough in one line.
	if (ArrayBuffer.isView(value)) return `${value.length.toLocaleString('en')} values`;
	if (Array.isArray(value)) return `[${value.map(rounded).join(', ')}]`;
	if (value !== null && typeof value === 'object')
		return `{${Object.entries(value)
			.map(([key, part]) => `${key} ${rounded(part)}`)
			.join(', ')}}`;
	return String(value);
}

/** A Python run's one-line summary: its first comment, else its first line. */
function firstLine(code = '') {
	const lines = String(code)
		.split('\n')
		.map((line) => line.trim())
		.filter(Boolean);
	const comment = lines.find((line) => line.startsWith('#'));
	return (comment ? comment.replace(/^#+\s*/, '') : (lines[0] ?? '')).slice(0, 80);
}

export function arrowIcon() {
	return s(
		'svg',
		{ class: 'icon', viewBox: '0 0 16 16', 'aria-hidden': 'true' },
		s('path', { d: 'M8 13V3M3.5 7.5 8 3l4.5 4.5' }),
	);
}

export function forwardIcon() {
	return s(
		'svg',
		{ class: 'icon', viewBox: '0 0 16 16', 'aria-hidden': 'true' },
		s('path', { d: 'M3 8h10M8.5 3.5 13 8l-4.5 4.5' }),
	);
}

export function downIcon() {
	return s(
		'svg',
		{ class: 'icon', viewBox: '0 0 16 16', 'aria-hidden': 'true' },
		s('path', { d: 'M8 3v10M3.5 8.5 8 13l4.5-4.5' }),
	);
}
