// The page's side of the assistant's Python kernel (kernel-worker.js): starts it on first use, runs code in it,
// answers what its `pressure` module asks of the tool, and keeps its working directory and the workspace the same
// files. A run that computes for longer than RUN_LIMIT seconds, or that the engineer stops, ends by restarting the
// kernel; the next run starts a fresh one with the workspace's files.

import { resultSummary } from '../app/cases.js';
import { writeVtp } from '../engine/vtp.js';

const RUN_LIMIT = 60;
const WORKER = new URL('./kernel-worker.js', import.meta.url);
const MODULES = ['pressure.py', 'runner.py'];

/**
 * @typedef {Object} RunOutcome
 * @property {'finished' | 'failed' | 'timed out' | 'stopped'} status  'failed' when the code raised or the kernel
 *   broke; 'timed out' and 'stopped' restarted the kernel, losing its variables.
 * @property {string | null} error      the traceback, or what ended the run
 * @property {string[]} written          workspace files the code wrote
 * @property {string[]} removed          files of its own the code removed
 * @property {{ kind: 'remove' | 'replace' | 'move', path: string, to?: string, approved: boolean }[]} requests
 *   changes the code made to the engineer's files, each approved or declined by the engineer
 * @property {number} seconds
 */

/**
 * @param {{ workspace: import('../app/workspace.js').Workspace, engine: import('../engine/result.js').Engine,
 *   session: import('../app/session.js').Session,
 *   operate: (name: string, args: object) => Promise<{ result?: object, error?: string }>,
 *   approve: (question: string, action: string) => Promise<boolean> }} tool
 *   `operate` runs an operation as the assistant, asking the engineer first when it is marked `confirm`;
 *   `approve` asks the engineer a yes-or-no question, `action` naming the yes.
 */
export function createKernel({ workspace, engine, session, operate, approve }) {
	/** @type {Worker | null} */
	let worker = null;
	/** Settles once the kernel is ready to run code, or could not start. */
	let ready = null;
	let starting = null;
	/** The run in progress: { id, onOutput, clock, signal, phase: 'computing' | 'deciding', finish }. */
	let current = null;
	let nextRun = 1;
	/** True while the workspace is told of files the kernel already holds, so they are not sent back to it. */
	let fromKernel = false;
	/** A number per case result and its reference, so the kernel knows when a case's arrays have changed. */
	const versions = new WeakMap();
	let nextVersion = 1;

	workspace.onChange(({ kind, path, from, by }) => {
		if (!worker || fromKernel || kind === 'link') return;
		if (kind === 'write')
			worker.postMessage({ type: 'file', kind: 'place', path, by, blob: workspace.file(path).blob });
		else if (kind === 'remove') worker.postMessage({ type: 'file', kind: 'remove', path });
		else if (kind === 'move') worker.postMessage({ type: 'file', kind: 'move', path: from, to: path });
	});

	function start() {
		ready ??= (async () => {
			// The kernel's isolation comes from the policy its script is served with; without it, it does not start.
			const served = await fetch(WORKER, { method: 'HEAD', cache: 'no-store' });
			if (!served.headers.get('Content-Security-Policy')) {
				throw new Error(
					'This server does not send the Python kernel its network policy, so the kernel is not started. Run the tool with uv run surrogate app.',
				);
			}
			const modules = Object.fromEntries(
				await Promise.all(
					MODULES.map(async (name) => [name, await (await fetch(new URL(`./${name}`, import.meta.url))).text()]),
				),
			);
			await new Promise((resolve, reject) => {
				starting = { resolve, reject };
				worker = new Worker(WORKER, { type: 'module' });
				worker.onmessage = ({ data }) => receive(data);
				worker.onerror = (event) => fail(event.message || 'its script did not load');
				worker.postMessage({
					type: 'start',
					modules,
					files: workspace.files().map(({ path, blob, by }) => ({ path, blob, by })),
				});
			});
			starting = null;
		})();
		ready.catch(shutDown);
		return ready;
	}

	/** Terminate the kernel's worker; the next run starts a fresh one. */
	function shutDown() {
		worker?.terminate();
		worker = null;
		ready = null;
	}

	/** The kernel broke, starting or running: say so to whoever waits on it, and start afresh next time. */
	function fail(message) {
		shutDown();
		if (starting) starting.reject(new Error(`The Python kernel could not start: ${message}`));
		else
			current?.finish({
				status: 'failed',
				error: `The Python kernel failed and was restarted, losing its variables: ${message}`,
			});
	}

	function cases() {
		return {
			active: session.active?.id ?? null,
			cases: session.cases.map((c) => {
				const reference = c.result?.reference ?? null;
				// A case's arrays change with its result and with the reference attached to it.
				if (c.result && versions.get(c.result)?.reference !== reference)
					versions.set(c.result, { reference, number: nextVersion++ });
				return {
					id: c.id,
					name: c.name,
					kind: c.kind,
					status: c.status,
					path: c.path ?? null,
					reference,
					variant_of: c.variant?.root.id ?? null,
					version: c.result ? versions.get(c.result).number : 0,
				};
			}),
		};
	}

	/**
	 * Run code in the kernel, starting it if needed. `onOutput` hears each item as it appears: { kind: 'text', text,
	 * stream? }, { kind: 'table', columns, rows, total, text } or { kind: 'figure', png, title }. Aborting `signal`
	 * ends the run at once: before the code starts, the kernel goes on loading and the code never runs; while it
	 * computes, the kernel is restarted, since Python cannot be interrupted without shared memory.
	 * @returns {Promise<RunOutcome>}
	 */
	async function run(code, onOutput, signal) {
		const begun = performance.now();
		const outcome = (fields) => ({
			written: [],
			removed: [],
			requests: [],
			...fields,
			seconds: (performance.now() - begun) / 1000,
		});
		const stopped = new Promise((resolve) => signal.addEventListener('abort', resolve, { once: true }));
		if (!signal.aborted) await Promise.race([start(), stopped]);
		if (signal.aborted)
			return outcome({ status: 'stopped', error: 'The engineer stopped the run before the code started.' });
		const id = nextRun++;
		return new Promise((resolve) => {
			const clock = pausableClock(RUN_LIMIT * 1000, () => end('timed out'));
			current = { id, onOutput, clock, signal, phase: 'computing' };
			current.finish = (fields) => {
				clock.stop();
				current = null;
				resolve(outcome(fields));
			};
			stopped.then(() => current?.id === id && end('stopped'));
			worker.postMessage({ type: 'run', id, code, cases: cases() });
		});
	}

	/** End the run in progress by restarting the kernel, if it is computing (not once it has ended or is asking). */
	function end(status) {
		if (current?.phase !== 'computing') return;
		shutDown();
		const error =
			status === 'timed out'
				? `The run computed for more than ${RUN_LIMIT} s and was stopped by restarting the kernel.`
				: 'The engineer stopped the run, which restarted the kernel.';
		current.finish({
			status,
			error: `${error} Its variables and imports are gone; the workspace files are kept, except any this run was still writing.`,
		});
	}

	async function receive(message) {
		switch (message.type) {
			case 'ready':
				starting.resolve();
				return;
			case 'failed':
				fail(message.message);
				return;
			case 'output':
				if (current?.id === message.run) current.onOutput(message.item);
				return;
			case 'request': {
				const asker = worker;
				current?.clock.pause();
				const { error = null, value = null, transfer = [] } = await answer(message.kind, message.payload);
				current?.clock.resume();
				// A kernel restarted meanwhile (stopped, timed out) is not answered.
				if (asker === worker)
					worker.postMessage({ type: 'reply', id: message.id, error, value, cases: cases() }, transfer);
				return;
			}
			case 'done': {
				const run = current;
				if (run?.id !== message.id) return;
				run.clock.stop();
				run.phase = 'deciding';
				const { written, removed, requests } = message.files;
				fromKernel = true;
				try {
					for (const { path, bytes } of written) workspace.write(path, new Blob([bytes]), 'assistant');
					for (const path of removed) workspace.remove(path, 'assistant');
				} finally {
					fromKernel = false;
				}
				// A stop declines the question on screen and every one after it, unasked.
				const decided = [];
				for (const change of requests)
					decided.push({
						kind: change.kind,
						path: change.path,
						to: change.to,
						approved: !run.signal.aborted && (await decide(change)),
					});
				run.finish({
					status: message.error ? 'failed' : 'finished',
					error: message.error,
					written: written.map((file) => file.path),
					removed,
					requests: decided,
				});
				return;
			}
		}
	}

	/** Ask the engineer about a change the code made to one of their files; apply it if approved. */
	async function decide({ kind, path, to, bytes }) {
		const questions = {
			remove: [`Delete ${path}?`, 'Delete'],
			replace: [`Replace ${path} with the assistant's version?`, 'Replace'],
			move: [`Move ${path} to ${to}?`, 'Move'],
		};
		if (!(await approve(...questions[kind]))) return false;
		if (kind === 'remove') workspace.remove(path, 'assistant');
		else if (kind === 'replace') workspace.write(path, new Blob([bytes]), 'assistant');
		else workspace.move(path, to, 'assistant');
		return true;
	}

	/** What the kernel's `pressure` module asked for: { error } or { value, transfer }. */
	async function answer(kind, payload) {
		try {
			switch (kind) {
				case 'operation': {
					const { result, error } = await operate(payload.name, payload.args);
					return error ? { error } : { value: result };
				}
				case 'working':
					// Shared with the viewer, so copied on the way rather than moved.
					return { value: engine.workingCells(readyCase(payload.case_id).result) };
				case 'unchanged': {
					// The car a variant was made from, predicted unchanged on the variant's own cells (the variant's
					// baseline), so subtracting it cell by cell leaves the change alone.
					const baseline = readyCase(payload.case_id).variant?.root.baseline;
					return { value: baseline ? engine.workingCells(baseline) : null };
				}
				case 'file': {
					const cells = await engine.fileCells(readyCase(payload.case_id).result);
					return { value: cells, transfer: cells ? buffersOf(cells) : [] };
				}
				case 'predict': {
					const { name, points, connectivity, offsets } = payload;
					const result = await engine.predictQuick({ name, points, connectivity, offsets });
					const seconds = result.timings.reduce((sum, stage) => sum + stage.seconds, 0);
					return {
						value: { summary: resultSummary(result, false), cells: result.cells, seconds },
						transfer: buffersOf(result.cells),
					};
				}
				case 'write_vtp': {
					const { path, points, connectivity, offsets, cellData, pointData } = payload;
					if (
						workspace.file(path)?.by === 'engineer' &&
						!(await approve(`Replace ${path} with the assistant's version?`, 'Replace'))
					) {
						return { error: `The engineer declined replacing ${path}. Write to another path.` };
					}
					const name = path.slice(path.lastIndexOf('/') + 1);
					workspace.write(path, writeVtp({ name, points, connectivity, offsets }, cellData, pointData), 'assistant');
					return { value: path };
				}
			}
			return { error: `The kernel asked for ${kind}, which the page does not answer.` };
		} catch (error) {
			return { error: error.message };
		}
	}

	function readyCase(id) {
		const c = session.cases.find((each) => each.id === id);
		if (!c)
			throw new Error(
				`There is no open case ${id}. Open cases: ${session.cases.map((each) => `${each.id} (${each.name})`).join(', ') || 'none'}.`,
			);
		if (c.status !== 'ready')
			throw new Error(`Case ${id} (${c.name}) is ${c.status}; its arrays are there once it is ready.`);
		return c;
	}

	return { run, close: shutDown };
}

const buffersOf = (cells) =>
	Object.values(cells)
		.filter(ArrayBuffer.isView)
		.map((array) => array.buffer);

/**
 * A timer that stops while the kernel waits on the page (an operation, a prediction, the engineer), so only the
 * kernel's own computing counts. Waits may overlap.
 */
function pausableClock(limit, onExpiry) {
	let left = limit;
	let since = performance.now();
	let waits = 0;
	let timer = setTimeout(onExpiry, left);
	return {
		pause() {
			if (waits++ > 0) return;
			clearTimeout(timer);
			left -= performance.now() - since;
		},
		resume() {
			if (--waits > 0) return;
			since = performance.now();
			timer = setTimeout(onExpiry, left);
		},
		stop: () => clearTimeout(timer),
	};
}
