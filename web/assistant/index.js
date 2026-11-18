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
/** What the panel says it does, before the first message. */
const PURPOSE = 'Works this tool for you: reads the pressure, cuts sections, places probes, compares cars and makes variants.';

/**
 * The requests offered before the first message, about the car on screen (`car`, a case, or none), each one run and
 * its answer checked on runs 1 and 500. Comparing with the CFD is offered only for a car that carries it (the bundled
 * examples always do, so theirs is offered while they load too).
 */
function offers(car) {
	if (!car || car.status === 'failed') {
		return [
			'Open DrivAerML run 1 and show me where the pressure is lowest',
			'Open DrivAerML run 1, raise it 20 mm and show me where the pressure changes most',
		];
	}
	return [
		'Where is the pressure lowest on this car?',
		car.kind === 'example' || car.result?.reference ? 'How close is the prediction to the CFD for this car?' : null,
		'Raise this car 20 mm and show me where the pressure changes most',
	].filter(Boolean);
}

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
 * @param {{ session: import('../app/session.js').Session, engine: import('../engine/result.js').Engine, viewer: object,
 *   operations: Map<string, import('../app/operations.js').Operation>, runOperation: (name: string, args: object) => Promise<object>,
 *   onOperation: Function, workspace: import('../app/workspace.js').Workspace, close: () => void }} tool
 */
export async function mountAssistant(host, { session, engine, viewer, operations, runOperation, onOperation, workspace, close }) {
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

	const thread = h('div', { class: 'assistant-thread' });
	// A log for navigation, but not read out as it streams: the announcer below says each turn's outcome once.
	const log = h('div', { class: 'assistant-log', role: 'log', 'aria-live': 'off', 'aria-label': 'Conversation' }, thread);
	const latest = h('button', { type: 'button', class: 'assistant-latest', hidden: true, title: 'Scroll to the latest', 'aria-label': 'Scroll to the latest' }, downIcon());
	const announcer = h('p', { class: 'assistant-announcer', role: 'status' });
	const input = h('textarea', { class: 'assistant-input', rows: 1, placeholder: 'Ask about the cars, or describe a study', 'aria-label': 'Message to the assistant' });
	const sendButton = h('button', { type: 'submit', class: 'assistant-send', 'aria-label': 'Send' });
	const form = h('form', { class: 'assistant-composer' }, input, sendButton);
	const newButton = h('button', { type: 'button', class: 'icon-button', title: 'New conversation', 'aria-label': 'New conversation', onclick: restart }, icon('plus'));
	const closeButton = h('button', { type: 'button', class: 'icon-button', title: 'Close', 'aria-label': 'Close the assistant', onclick: close }, icon('close'));
	host.append(h('div', { class: 'assistant' },
		h('header', { class: 'assistant-head' }, h('h2', { class: 'assistant-title' }, 'Assistant'), newButton, closeButton),
		h('div', { class: 'assistant-view' }, log, latest),
		form, announcer));

	const kernel = createKernel({ workspace, engine, session, approve, operate: (name, args) => operate(name, args, turn?.python) });

	onOperation(({ name, args, by }) => {
		if (by !== 'engineer') return;
		actions.delete(name);
		actions.set(name, args);
	});

	// The log keeps the latest in view while the reader is at the bottom. Scrolling up stops that, and the button
	// below brings them back; reaching the bottom again resumes it.
	let following = true;
	let lastTop = 0;
	log.addEventListener('scroll', () => {
		if (log.scrollHeight - log.scrollTop - log.clientHeight < 24) following = true;
		else if (log.scrollTop < lastTop) following = false;
		lastTop = log.scrollTop;
		latest.hidden = following;
	}, { passive: true });
	new ResizeObserver(() => {
		if (following) log.scrollTop = log.scrollHeight;
	}).observe(thread);
	latest.addEventListener('click', () => {
		following = true;
		latest.hidden = true;
		log.scrollTo({ top: log.scrollHeight, behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
	});

	form.addEventListener('submit', (event) => {
		event.preventDefault();
		if (status === 'running') stop();
		else send(input.value);
	});
	// Enter sends; while a turn runs it does nothing, so the next message can be written meanwhile (Stop is the button).
	input.addEventListener('keydown', (event) => {
		if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
			event.preventDefault();
			if (status !== 'running') form.requestSubmit();
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
			put(thread, h('div', { class: 'assistant-note' },
				h('p', { class: 'assistant-note-title' }, 'The assistant is not available in this copy.'),
				h('p', {}, 'To use it here, set ', h('code', {}, 'OPENROUTER_API_KEY'), ' in ', h('code', {}, 'local/secrets/assistant.env'), ' and restart ', h('code', {}, 'uv run surrogate app'), '.')));
		} else if (next === 'unreachable') {
			put(thread, h('div', { class: 'assistant-note' },
				h('p', { class: 'assistant-note-title' }, 'The assistant could not be reached.'),
				h('button', { type: 'button', class: 'button', onclick: check }, 'Try again')));
		} else if (next === 'idle' && messages.length === 0) {
			showStart();
		}
	}

	/**
	 * Before the first message: what the assistant does here and requests about the car on screen. A tap sends the
	 * request, since each is complete as written; while the car is still being analysed they wait, disabled.
	 */
	let shownStart = '';
	function showStart() {
		const car = session.active;
		const waiting = car && !['ready', 'failed'].includes(car.status);
		const requests = offers(car);
		const key = JSON.stringify([requests, waiting]);
		if (key === shownStart && thread.querySelector('.assistant-start')) return;
		shownStart = key;
		put(thread, h('div', { class: 'assistant-start' },
			h('p', { class: 'assistant-purpose' }, PURPOSE),
			h('div', { class: 'assistant-offers' }, requests.map((text) => h('button', { type: 'button', class: 'offer', disabled: waiting, onclick: () => send(text) },
				h('span', {}, text), forwardIcon())))));
	}
	session.addEventListener('change', () => {
		if (status === 'idle' && messages.length === 0) showStart();
	});

	async function check() {
		setStatus('checking');
		const found = await availability();
		setStatus(found === 'available' ? 'idle' : found);
		// Not on touch screens, where focusing the box opens the keyboard over the panel before anything is asked.
		if (found === 'available' && matchMedia('(pointer: fine)').matches) input.focus();
	}

	async function send(text) {
		text = text.trim();
		if (status !== 'idle' || !text) return;
		if (messages.length === 0) {
			messages = [{ role: 'system', content: systemPrompt(await engine.model) }];
			thread.replaceChildren();
		}
		input.value = '';
		fitInput();
		// Only the latest failed turn can be tried again.
		for (const retry of thread.querySelectorAll('.reply-retry')) retry.remove();
		// The cases as list_cases reports them, so the model starts from the numbers the engineer sees.
		const { cases } = await runOperation('list_cases', {});
		const before = messages.length;
		const extents = session.cases.filter((c) => c.result).map((c) => ({ case_id: c.id, bounds_m: c.result.geometry.bounds.map((v) => Math.round(v * 1000) / 1000) }));
		messages.push({ role: 'user', content: `${describeState({ cases, extents, workspace, actions: [...actions].map(([name, args]) => ({ name, args })) })}\n\n${text}` });
		actions.clear();
		following = true;
		turn = createTurn(text);
		setStatus('running');
		controller = new AbortController();
		try {
			await respond(messages, {
				tools: toolList,
				execute: (call) => execute(call, turn),
				signal: controller.signal,
				onRequest: () => turn.request(),
				onReasoning: (piece) => turn.reasoning(piece),
				onText: (delta) => turn.text(delta),
				onToolCall: (call) => turn.toolCall(call),
				onRequestEnd: () => turn.requestEnd(),
			});
			turn.finish('answered');
		} catch (error) {
			if (error.name === 'AbortError') {
				turn.finish('stopped');
			} else {
				// Trying again sends the same message from the conversation as it was before it.
				turn.finish('failed', error.message, () => {
					messages.length = before;
					send(text);
				});
			}
		} finally {
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
		const step = current.step(call);
		let args;
		try {
			args = JSON.parse(call.function.arguments || '{}');
		} catch {
			const error = 'The arguments were not valid JSON. Send one JSON object.';
			step.finish('failed', error);
			return JSON.stringify({ error });
		}
		step.begin(args);
		if (name === PYTHON) {
			const items = [];
			current.python = step;
			try {
				const outcome = await kernel.run(String(args.code ?? ''), (item) => {
					items.push(item);
					step.output(item);
				}, controller.signal);
				step.finish(outcome.status === 'finished' ? 'done' : outcome.status, outcome.error);
				return JSON.stringify(report(outcome, items));
			} catch (error) {
				step.finish('failed', error.message);
				return JSON.stringify({ error: error.message });
			} finally {
				current.python = null;
			}
		}
		// An operation cannot be called back once started; a stop ends the run without waiting for it.
		const { result, error, outcome } = await Promise.race([operate(name, args), whenStopped(controller.signal)]);
		step.finish(outcome, error, result);
		return clip(JSON.stringify(error ? { error } : result));
	}

	/**
	 * Run an operation as the assistant, asking the engineer first when it opens, writes, downloads or deletes.
	 * `outcome` is how its step ends: 'done', 'declined' or 'failed'.
	 */
	async function operate(name, args, within) {
		const operation = operations.get(name);
		if (!operation) return { error: `There is no operation named ${name}.`, outcome: 'failed' };
		const sub = within?.sub(name, args);
		const asks = typeof operation.confirm === 'function' ? operation.confirm(args, { session, viewer, engine, by: 'assistant' }) : operation.confirm;
		if (asks && !(await approve(`${sentence(name)}${subject(args)}?`, sentence(name)))) {
			sub?.finish('declined');
			return { error: 'The engineer declined.', outcome: 'declined' };
		}
		try {
			const result = await runOperation(name, args);
			sub?.finish('done');
			return { result, outcome: 'done' };
		} catch (error) {
			sub?.finish('failed');
			return { error: error.message, outcome: 'failed' };
		}
	}

	/** Ask the engineer a yes-or-no question in the conversation; `action` names the yes. */
	function approve(question, action) {
		return new Promise((resolve) => {
			const yes = h('button', { type: 'button', class: 'button button-primary' }, action);
			const no = h('button', { type: 'button', class: 'button' }, 'Decline');
			const row = h('div', { class: 'approval', role: 'group', 'aria-label': question }, h('p', { class: 'approval-question' }, question), h('div', { class: 'approval-actions' }, yes, no));
			const exchange = turn ?? createTurn(null);
			const settle = (approved) => {
				waiting.delete(settle);
				exchange.asking(false);
				row.replaceWith(h('p', { class: 'approval-decided' }, `${question.replace(/\?$/, '')}: ${approved ? 'approved' : 'declined'}`));
				resolve(approved);
			};
			yes.addEventListener('click', () => settle(true));
			no.addEventListener('click', () => settle(false));
			waiting.add(settle);
			exchange.asking(true);
			exchange.append(row);
			// The run waits on this answer, so it is brought into view wherever the engineer has scrolled.
			row.scrollIntoView({ block: 'nearest' });
			yes.focus({ preventScroll: true });
		});
	}

	/**
	 * One exchange in the log: the engineer's message, then the parts of the reply in the order they come. Whatever is
	 * in progress says so with its name shimmering: "Thinking" while the model's thinking streams in, a step while
	 * its call is written and run, and "Working" at the end of the exchange while nothing else is moving.
	 */
	function createTurn(text) {
		const reply = h('div', { class: 'reply' });
		const working = h('p', { class: 'working', hidden: true }, h('span', { class: 'shimmer' }, 'Working'));
		thread.append(h('section', { class: 'exchange' }, text ? h('p', { class: 'question' }, text) : null, reply, working));
		let thought = null;
		let answer = null;
		let lastAnswer = '';
		let requested = 0;
		/** Each tool call's step, shown from the moment its name arrives. */
		const steps = new Map();
		const settle = () => {
			thought?.finish();
			thought = null;
			if (answer) lastAnswer = answer.finish();
			answer = null;
		};
		return {
			python: null,
			append: (element) => reply.append(element),
			request() {
				requested = performance.now();
				working.hidden = false;
			},
			reasoning(piece) {
				working.hidden = true;
				if (!thought) reply.append((thought = thoughtView(requested)).el);
				thought.add(piece);
			},
			text(delta) {
				thought?.finish();
				thought = null;
				working.hidden = true;
				if (!answer) reply.append((answer = answerView()).el);
				answer.add(delta);
			},
			toolCall(call) {
				settle();
				working.hidden = true;
				const view = stepView(call.function.name);
				steps.set(call, view);
				reply.append(view.el);
			},
			requestEnd: settle,
			step(call) {
				if (!steps.has(call)) this.toolCall(call);
				return steps.get(call);
			},
			/** While a question waits on the engineer, nothing shimmers: the run is waiting on them, not working. */
			asking(on) {
				reply.classList.toggle('asking', on);
			},
			/** The turn is over: 'answered', 'stopped' (what arrived stays) or 'failed' with what to do next. */
			finish(outcome, message, retry) {
				settle();
				working.hidden = true;
				if (outcome === 'stopped') {
					reply.append(h('p', { class: 'reply-note' }, 'Stopped.'));
					announcer.textContent = 'Stopped.';
				} else if (outcome === 'failed') {
					reply.append(h('div', { class: 'reply-failed', role: 'alert' },
						h('p', { class: 'reply-error' }, message),
						h('button', { type: 'button', class: 'button reply-retry', onclick: retry }, 'Try again')));
				} else {
					announcer.textContent = lastAnswer;
				}
			},
		};
	}

	/**
	 * The model's thinking for one request, from the moment its first piece arrives: open while it streams, then
	 * folded to "Thought for 4 s" (timed from the request) unless the engineer opened or closed it meanwhile. When the
	 * thinking came without text, the row says only that it thought and for how long, and does not open.
	 */
	function thoughtView(requested) {
		const row = disclosure('thought', 'Thinking');
		const text = answerView('thought-text');
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
	function answerView(className = 'answer') {
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
	 * The row a thinking block and a tool step share: state mark, name, a one-line summary, how it ended when not
	 * well, and a chevron; it opens a body below. Running, the mark is blue and the name shimmers.
	 */
	function disclosure(kind, name) {
		const nameEl = h('span', { class: kind === 'step' ? 'row-name step-name' : 'row-name' }, name);
		const detail = h('span', { class: 'row-detail' });
		const ending = h('span', { class: 'row-ending' });
		const toggle = h('button', { type: 'button', class: 'row', 'aria-expanded': 'false' },
			h('span', { class: 'row-state' }), nameEl, detail, ending, h('span', { class: 'row-chevron' }, icon('chevron')));
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
				el.classList.add(outcome === 'done' ? 'done' : ['declined', 'stopped'].includes(outcome) ? 'ended' : 'failed');
				if (outcome !== 'done') put(ending, sentence(outcome));
				if (newName) put(nameEl, newName);
			},
		};
	}

	/** A tool call as a row the engineer can open to see what was asked and what came back. */
	function stepView(name) {
		const python = name === PYTHON;
		const row = disclosure('step', python ? 'Python' : sentence(name));
		const subs = h('div', { class: 'step-subs' });
		const outputs = h('div', { class: 'step-outputs' });
		row.el.insertBefore(subs, row.body);
		row.el.append(outputs);
		let printed = null;
		return {
			el: row.el,
			begin(args) {
				row.setDetail(python ? firstLine(args.code) : summary(args, operations.get(name)?.parameters.properties));
				row.content.append(h('p', { class: 'step-label' }, 'Asked'), h('pre', { class: python ? 'step-code' : 'step-json' }, python ? args.code : JSON.stringify(args, null, 2)));
			},
			output(item) {
				if (item.kind === 'text') {
					if (!printed) row.content.append(h('p', { class: 'step-label' }, 'Printed'), (printed = h('pre', { class: 'step-text' })));
					printed.append(item.stream === 'stderr' ? h('span', { class: 'stderr' }, item.text) : item.text);
				} else if (item.kind === 'figure') {
					outputs.append(figureView(item));
				} else if (item.kind === 'table') {
					outputs.append(tableView(item));
				}
			},
			sub(subName, subArgs) {
				const sub = h('div', { class: 'step-sub running' }, h('span', { class: 'row-state' }), h('span', { class: 'row-name step-name' }, sentence(subName)), h('span', { class: 'row-detail' }, summary(subArgs, operations.get(subName)?.parameters.properties)));
				subs.append(sub);
				return { finish: (outcome) => sub.classList.replace('running', outcome === 'done' ? 'done' : outcome === 'declined' ? 'ended' : 'failed') };
			},
			finish(outcome, error, result) {
				row.finish(outcome);
				if (result !== undefined) row.content.append(h('p', { class: 'step-label' }, 'Returned'), h('pre', { class: 'step-json' }, JSON.stringify(result, null, 2)));
				if (error) row.content.append(h('p', { class: 'step-label' }, python ? 'Error' : 'Refused'), h('pre', { class: 'step-error' }, error));
			},
		};
	}

	await check();
}

/** Settles with an error when `signal` aborts: what an operation still running is reported as after a stop. */
function whenStopped(signal) {
	return new Promise((resolve) => {
		const stopped = () => resolve({ error: 'The engineer stopped the run while this operation ran; it may still finish on screen.', outcome: 'stopped' });
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
	// The size from the PNG header, so the figure's space is kept before it decodes and nothing below it jumps.
	const header = new DataView(png.buffer, png.byteOffset, 24);
	return h('figure', { class: 'output-figure' },
		h('img', { src: url, alt: title, width: header.getUint32(16), height: header.getUint32(20) }),
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
	const value = args.path ?? args.from ?? args.export ?? args.example ?? args.layer ?? (args.case_id != null ? `case ${args.case_id}` : null);
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
	// Arrays sent from Python arrive whole; their size says enough in one line.
	if (ArrayBuffer.isView(value)) return `${value.length.toLocaleString('en')} values`;
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

function forwardIcon() {
	return s('svg', { class: 'icon', viewBox: '0 0 16 16', 'aria-hidden': 'true' }, s('path', { d: 'M3 8h10M8.5 3.5 13 8l-4.5 4.5' }));
}

function downIcon() {
	return s('svg', { class: 'icon', viewBox: '0 0 16 16', 'aria-hidden': 'true' }, s('path', { d: 'M8 3v10M3.5 8.5 8 13l4.5-4.5' }));
}

function stylesheet() {
	const link = h('link', { rel: 'stylesheet', href: new URL('./assistant.css', import.meta.url).href });
	document.head.append(link);
	return new Promise((resolve, reject) => {
		link.onload = resolve;
		link.onerror = () => reject(new Error('The assistant panel could not load its styles.'));
	});
}
