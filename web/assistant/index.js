// The assistant: one agent the engineer talks to in a panel beside the car. Its tools are the tool's own operations
// (web/app/operations.js, read afresh for every request) and a Python kernel; each call shows in the conversation as
// a step the engineer can open, and operations change the view as the engineer's controls do.

import { download, h, put, s } from '../app/dom.js';
import { icon } from '../app/icons.js';
import { availability, respond } from './conversation.js';
import { createKernel } from './kernel.js';
import { renderMarkdown } from './markdown.js';
import { describeState, systemPrompt } from './prompt.js';

const PYTHON = 'run_python';
/** Characters of a tool's result the model is sent; more is cut from the middle. */
const RESULT_LIMIT = 16000;
const SUGGESTIONS = [
	'Open run 1 and show me the lowest pressure on the car',
	'Plot Cp along the centreline of the roof',
	'Study ride height at −20, 0 and +20 mm and tabulate the pressure drag',
];

const PYTHON_TOOL = {
	type: 'function',
	function: {
		name: PYTHON,
		description: 'Run Python in the persistent kernel (numpy, scipy, pandas, matplotlib, and the `pressure` module that reaches the open cases, every operation and the model). Variables persist between calls; top-level await works. Returns what the code printed, the tables shown, the titles of the figures shown to the engineer, the workspace files written, and the traceback if it failed.',
		parameters: {
			type: 'object',
			properties: { code: { type: 'string', description: 'Python source to run as one cell; the value of its last line is shown.' } },
			required: ['code'],
			additionalProperties: false,
		},
	},
};

/**
 * @param {HTMLElement} host  The panel's element; mounted once, kept while hidden.
 * @param {{ session: import('../app/session.js').Session, engine: import('../engine/result.js').Engine,
 *   operations: Map<string, import('../app/operations.js').Operation>, runOperation: (name: string, args: object) => Promise<object>,
 *   onOperation: Function, workspace: import('../app/workspace.js').Workspace, close: () => void }} tool
 */
export async function mountAssistant(host, { session, engine, operations, runOperation, onOperation, workspace, close }) {
	await stylesheet();

	/** 'checking' | 'unavailable' | 'unreachable' | 'idle' | 'running' */
	let status = 'checking';
	let messages = [];
	let controller = null;
	let turn = null;
	/** Approvals waiting on the engineer, each a function that settles it. */
	const waiting = new Set();
	/** What the engineer did since the last message, the latest of each operation (a drag runs one many times). */
	const actions = new Map();

	const log = h('div', { class: 'assistant-log', role: 'log', 'aria-live': 'polite' });
	const input = h('textarea', { class: 'assistant-input', rows: 1, placeholder: 'Ask about the cars, or describe a study', 'aria-label': 'Message to the assistant' });
	const sendButton = h('button', { type: 'submit', class: 'assistant-send', 'aria-label': 'Send' });
	const form = h('form', { class: 'assistant-composer' }, input, sendButton);
	const newButton = h('button', { type: 'button', class: 'icon-button', title: 'New conversation', 'aria-label': 'New conversation', onclick: restart }, icon('plus'));
	const closeButton = h('button', { type: 'button', class: 'icon-button', title: 'Close', 'aria-label': 'Close the assistant', onclick: close }, icon('close'));
	host.append(h('div', { class: 'assistant' }, h('header', { class: 'assistant-head' }, h('h2', { class: 'assistant-title' }, 'Assistant'), newButton, closeButton), log, form));

	const kernel = createKernel({ workspace, engine, session, approve, operate: (name, args) => operate(name, args, turn?.python) });

	onOperation(({ name, args, by }) => {
		if (by !== 'engineer') return;
		actions.delete(name);
		actions.set(name, args);
	});

	form.addEventListener('submit', (event) => {
		event.preventDefault();
		if (status === 'running') stop();
		else send(input.value);
	});
	input.addEventListener('keydown', (event) => {
		if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
			event.preventDefault();
			form.requestSubmit();
		}
	});
	input.addEventListener('input', fitInput);

	/** The message box grows with its text, up to a few lines. */
	function fitInput() {
		input.style.height = 'auto';
		input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
	}

	function setStatus(next) {
		status = next;
		host.dataset.status = next;
		const running = next === 'running';
		put(sendButton, running ? icon('cancel') : arrowIcon());
		sendButton.setAttribute('aria-label', running ? 'Stop' : 'Send');
		sendButton.classList.toggle('stop', running);
		newButton.disabled = running || messages.length === 0;
		form.hidden = !['idle', 'running'].includes(next);
		if (next === 'unavailable') {
			put(log, h('div', { class: 'assistant-note' },
				h('p', { class: 'assistant-note-title' }, 'The assistant is not available in this copy.'),
				h('p', {}, 'To use it here, set ', h('code', {}, 'OPENROUTER_API_KEY'), ' in ', h('code', {}, 'local/secrets/assistant.env'), ' and restart ', h('code', {}, 'uv run surrogate-app'), '.')));
		} else if (next === 'unreachable') {
			put(log, h('div', { class: 'assistant-note' },
				h('p', { class: 'assistant-note-title' }, 'The assistant could not be reached.'),
				h('button', { type: 'button', class: 'button', onclick: check }, 'Try again')));
		} else if (next === 'idle' && messages.length === 0) {
			put(log, h('div', { class: 'assistant-start' }, SUGGESTIONS.map((text) => h('button', { type: 'button', class: 'suggestion', onclick: () => send(text) }, text))));
		}
	}

	async function check() {
		setStatus('checking');
		const found = await availability();
		setStatus(found === 'available' ? 'idle' : found);
		if (found === 'available') input.focus();
	}

	async function send(text) {
		text = text.trim();
		if (status !== 'idle' || !text) return;
		if (messages.length === 0) {
			messages = [{ role: 'system', content: systemPrompt(await engine.model) }];
			log.replaceChildren();
		}
		input.value = '';
		fitInput();
		// The cases as list_cases reports them, so the model starts from the numbers the engineer sees.
		const { cases } = await runOperation('list_cases', {});
		messages.push({ role: 'user', content: `${describeState({ cases, workspace, actions: [...actions].map(([name, args]) => ({ name, args })) })}\n\n${text}` });
		actions.clear();
		turn = createTurn(text);
		setStatus('running');
		controller = new AbortController();
		try {
			await respond(messages, {
				tools: toolList,
				execute: (call) => execute(call, turn),
				onStart: () => turn.waiting(true),
				onText: (delta) => turn.text(delta),
				signal: controller.signal,
			});
		} catch (error) {
			if (error.name === 'AbortError') turn.note('Stopped.');
			else turn.error(error.message);
		} finally {
			turn.end();
			turn = null;
			controller = null;
			setStatus('idle');
		}
	}

	/** End the run now: the request, the code computing and any question waiting on the engineer. */
	function stop() {
		controller?.abort();
		for (const settle of waiting) settle(false);
	}

	function restart() {
		if (status === 'running') return;
		messages = [];
		kernel.close();
		setStatus('idle');
		input.focus();
	}

	function toolList() {
		const defined = [...operations.values()].map(({ name, description, parameters }) => ({ type: 'function', function: { name, description, parameters } }));
		return [...defined, PYTHON_TOOL];
	}

	/** Run one tool call; returns what the model is told. */
	async function execute(call, current) {
		const { name } = call.function;
		let args;
		try {
			args = JSON.parse(call.function.arguments || '{}');
		} catch {
			return JSON.stringify({ error: 'The arguments were not valid JSON. Send one JSON object.' });
		}
		current.waiting(false);
		const step = current.step(name, args);
		if (name === PYTHON) {
			const items = [];
			current.python = step;
			try {
				const outcome = await kernel.run(String(args.code ?? ''), (item) => {
					items.push(item);
					step.output(item);
				}, controller.signal);
				step.finish(outcome.status === 'finished' ? null : outcome.error);
				return JSON.stringify(report(outcome, items));
			} catch (error) {
				step.finish(error.message);
				return JSON.stringify({ error: error.message });
			} finally {
				current.python = null;
			}
		}
		// An operation cannot be called back once started; a stop ends the run without waiting for it.
		const { result, error } = await Promise.race([operate(name, args), whenStopped(controller.signal)]);
		step.finish(error ?? null, result);
		return clip(JSON.stringify(error ? { error } : result));
	}

	/** Run an operation as the assistant, asking the engineer first when it opens, writes, downloads or deletes. */
	async function operate(name, args, within) {
		const operation = operations.get(name);
		if (!operation) return { error: `There is no operation named ${name}.` };
		const sub = within?.sub(name, args);
		if (operation.confirm && !(await approve(`${sentence(name)}${subject(args)}?`, sentence(name)))) {
			sub?.finish('Declined');
			return { error: 'The engineer declined.' };
		}
		try {
			const result = await runOperation(name, args);
			sub?.finish(null);
			return { result };
		} catch (error) {
			sub?.finish(error.message);
			return { error: error.message };
		}
	}

	/** Ask the engineer a yes-or-no question in the conversation; `action` names the yes. */
	function approve(question, action) {
		return new Promise((resolve) => {
			const yes = h('button', { type: 'button', class: 'button button-primary' }, action);
			const no = h('button', { type: 'button', class: 'button' }, 'Decline');
			const row = h('div', { class: 'approval', role: 'group', 'aria-label': question }, h('p', { class: 'approval-question' }, question), h('div', { class: 'approval-actions' }, yes, no));
			const settle = (approved) => {
				waiting.delete(settle);
				row.replaceWith(h('p', { class: 'approval-decided' }, `${question.replace(/\?$/, '')}: ${approved ? 'approved' : 'declined'}`));
				resolve(approved);
			};
			yes.addEventListener('click', () => settle(true));
			no.addEventListener('click', () => settle(false));
			waiting.add(settle);
			(turn ?? createTurn(null)).append(row);
			// The run waits on this answer, so it is brought into view wherever the engineer has scrolled.
			row.scrollIntoView({ block: 'nearest' });
			yes.focus({ preventScroll: true });
		});
	}

	/** One exchange in the log: the engineer's message, then the steps and text of the reply as they come. */
	function createTurn(text) {
		const reply = h('div', { class: 'reply' });
		const pending = h('div', { class: 'pending', hidden: true }, h('span', { class: 'spinner' }), 'Thinking');
		log.append(h('section', { class: 'exchange' }, text ? h('p', { class: 'question' }, text) : null, reply, pending));
		log.scrollTop = log.scrollHeight;
		let answer = null;
		let answerText = '';
		let frame = 0;
		const follow = () => {
			const near = log.scrollHeight - log.scrollTop - log.clientHeight < 80;
			return () => near && (log.scrollTop = log.scrollHeight);
		};
		const add = (element) => {
			const keep = follow();
			reply.append(element);
			keep();
		};
		return {
			python: null,
			append: add,
			waiting(on) {
				pending.hidden = !on;
				if (on) follow()();
			},
			text(delta) {
				pending.hidden = true;
				if (!answer) {
					answer = h('div', { class: 'answer' });
					answerText = '';
					add(answer);
				}
				answerText += delta;
				const element = answer;
				const shown = answerText;
				cancelAnimationFrame(frame);
				frame = requestAnimationFrame(() => {
					const keep = follow();
					put(element, renderMarkdown(shown));
					keep();
				});
			},
			step(name, args) {
				answer = null;
				const view = stepView(name, args);
				add(view.el);
				return view;
			},
			note: (message) => add(h('p', { class: 'reply-note' }, message)),
			error: (message) => add(h('p', { class: 'reply-error', role: 'alert' }, message)),
			end() {
				pending.hidden = true;
			},
		};
	}

	/** A tool call as a row the engineer can open to see what was asked and what came back. */
	function stepView(name, args) {
		const python = name === PYTHON;
		const body = h('div', { class: 'step-body', hidden: true });
		const toggle = h('button', { type: 'button', class: 'step-row', 'aria-expanded': 'false' },
			h('span', { class: 'step-state' }),
			h('span', { class: 'step-name' }, python ? 'Python' : sentence(name)),
			h('span', { class: 'step-detail' }, python ? firstLine(args.code) : summary(args, operations.get(name)?.parameters.properties)),
			h('span', { class: 'step-chevron' }, icon('chevron')));
		const subs = h('div', { class: 'step-subs' });
		const outputs = h('div', { class: 'step-outputs' });
		const el = h('div', { class: 'step running' }, toggle, subs, body, outputs);
		let printed = null;
		toggle.addEventListener('click', () => {
			body.hidden = !body.hidden;
			toggle.setAttribute('aria-expanded', String(!body.hidden));
		});
		body.append(h('p', { class: 'step-label' }, 'Asked'), h('pre', { class: python ? 'step-code' : 'step-json' }, python ? args.code : JSON.stringify(args, null, 2)));
		return {
			el,
			output(item) {
				if (item.kind === 'text') {
					if (!printed) body.append(h('p', { class: 'step-label' }, 'Printed'), (printed = h('pre', { class: 'step-text' })));
					printed.append(item.stream === 'stderr' ? h('span', { class: 'stderr' }, item.text) : item.text);
				} else if (item.kind === 'figure') {
					outputs.append(figureView(item));
				} else if (item.kind === 'table') {
					outputs.append(tableView(item));
				}
			},
			sub(subName, subArgs) {
				const row = h('div', { class: 'step-sub running' }, h('span', { class: 'step-state' }), h('span', { class: 'step-name' }, sentence(subName)), h('span', { class: 'step-detail' }, summary(subArgs, operations.get(subName)?.parameters.properties)));
				subs.append(row);
				return { finish: (error) => row.classList.replace('running', error ? 'failed' : 'done') };
			},
			finish(error, result) {
				el.classList.replace('running', error ? 'failed' : 'done');
				if (result !== undefined) body.append(h('p', { class: 'step-label' }, 'Returned'), h('pre', { class: 'step-json' }, JSON.stringify(result, null, 2)));
				if (error) body.append(h('p', { class: 'step-label' }, python ? 'Error' : 'Refused'), h('pre', { class: 'step-error' }, error));
			},
		};
	}

	await check();
}

/** Settles with an error when `signal` aborts: what an operation still running is reported as after a stop. */
function whenStopped(signal) {
	return new Promise((resolve) => {
		const stopped = () => resolve({ error: 'The engineer stopped the run while this operation ran; it may still finish on screen.' });
		if (signal.aborted) stopped();
		else signal.addEventListener('abort', stopped, { once: true });
	});
}

/** What the model is told of a Python run: its printed output and tables, the figures shown, the files and decisions. */
function report(outcome, items) {
	const printed = items.map((item) => (item.kind === 'text' ? item.text : item.kind === 'table' ? `${item.text}\n` : '')).join('');
	const figures = items.filter((item) => item.kind === 'figure').map((item) => item.title);
	return {
		status: outcome.status,
		...(printed && { output: clip(printed) }),
		...(figures.length && { figures_shown: figures }),
		...(outcome.written.length && { files_written: outcome.written }),
		...(outcome.removed.length && { files_removed: outcome.removed }),
		...(outcome.requests.length && { engineer_decisions: outcome.requests }),
		...(outcome.error && { error: clip(outcome.error) }),
		seconds: Math.round(outcome.seconds * 10) / 10,
	};
}

function figureView({ png, title }) {
	const blob = new Blob([png], { type: 'image/png' });
	const url = URL.createObjectURL(blob);
	const name = `${title.replace(/[^\w\- ]+/g, '').trim().slice(0, 60) || 'figure'}.png`;
	return h('figure', { class: 'output-figure' },
		h('img', { src: url, alt: title }),
		h('button', { type: 'button', class: 'icon-button figure-save', title: 'Download', 'aria-label': `Download ${name}`, onclick: () => download(blob, name) }, icon('download')));
}

function tableView({ columns, rows, total }) {
	return h('div', { class: 'output-table' },
		h('div', { class: 'output-table-scroll' }, h('table', {},
			h('thead', {}, h('tr', {}, columns.map((column) => h('th', {}, column)))),
			h('tbody', {}, rows.map((row) => h('tr', {}, row.map((cell) => h('td', { class: /^[−+-]?[\d.]+(e[−+-]?\d+)?$/i.test(cell) ? 'num' : null }, cell))))))),
		total > rows.length ? h('p', { class: 'output-table-more' }, `${rows.length} of ${total.toLocaleString()} rows`) : null);
}

/** `delete_file` → "Delete file". */
const sentence = (name) => name.charAt(0).toUpperCase() + name.slice(1).replaceAll('_', ' ');

/** The thing an operation acts on, for its approval question. */
function subject(args) {
	const value = args.path ?? args.from ?? args.export ?? args.example ?? (args.case_id != null ? `case ${args.case_id}` : null);
	return value != null ? ` ${value}` : '';
}

/** Units by the suffix of an argument's name, as the operations name them, or by its schema's "in metres". */
const UNITS = { m: 'm', m2: 'm²', mm: 'mm', deg: '°', s: 's', pct: '%' };
const UNIT_WORDS = { metres: 'm', millimetres: 'mm', degrees: '°', seconds: 's' };

/**
 * Arguments in a line, numbers to three decimals: `case 1 · point [-0.031, 0.679, -0.313] m`. The opened step shows
 * them exactly.
 */
function summary(args, schema = {}) {
	return Object.entries(args ?? {}).map(([key, value]) => {
		if (key === 'case_id') return `case ${value}`;
		const suffix = key.match(/_(m|m2|mm|deg|s|pct)$/);
		const word = schema[key]?.description?.match(/\bin (metres|millimetres|degrees|seconds)\b/);
		const unit = suffix ? UNITS[suffix[1]] : word ? UNIT_WORDS[word[1]] : null;
		const label = (suffix ? key.slice(0, -suffix[0].length) : key).replaceAll('_', ' ');
		return `${label} ${rounded(value)}${unit ? ` ${unit}` : ''}`;
	}).join(' · ');
}

function rounded(value) {
	if (typeof value === 'number') return String(Math.round(value * 1000) / 1000);
	if (Array.isArray(value)) return `[${value.map(rounded).join(', ')}]`;
	if (value !== null && typeof value === 'object') return `{${Object.entries(value).map(([key, part]) => `${key} ${rounded(part)}`).join(', ')}}`;
	return String(value);
}

function firstLine(code = '') {
	const lines = String(code).split('\n').map((line) => line.trim()).filter(Boolean);
	const comment = lines.find((line) => line.startsWith('#'));
	return (comment ? comment.replace(/^#+\s*/, '') : lines[0] ?? '').slice(0, 80);
}

function clip(text, limit = RESULT_LIMIT) {
	if (text.length <= limit) return text;
	const half = Math.floor(limit / 2);
	return `${text.slice(0, half)}\n… ${text.length - limit} characters cut …\n${text.slice(-half)}`;
}

function arrowIcon() {
	return s('svg', { class: 'icon', viewBox: '0 0 16 16', 'aria-hidden': 'true' }, s('path', { d: 'M8 13V3M3.5 7.5 8 3l4.5 4.5' }));
}

function stylesheet() {
	const link = h('link', { rel: 'stylesheet', href: new URL('./assistant.css', import.meta.url).href });
	document.head.append(link);
	return new Promise((resolve, reject) => {
		link.onload = resolve;
		link.onerror = () => reject(new Error('The assistant panel could not load its styles.'));
	});
}
