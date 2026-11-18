// The assistant's Python kernel: Pyodide with numpy, scipy, pandas and matplotlib in a worker of its own, started on
// first use and kept for the conversation.
//
// No network. The server sends this script with a Content-Security-Policy (KERNEL_POLICY in src/surrogate/server.py
// and deploy/assistant-worker/worker.js) under which it may load Pyodide from its CDN and reach nothing else: not
// another address, not the page's own server, not another script or worker. Once Pyodide and its packages are in,
// the functions that could load more from the CDN are removed as well, so code run here has no network at all; it
// reaches the page only through the messages below, and the page decides what each may do.
//
// Files. The working directory, /work, mirrors the tool's workspace (web/app/workspace.js). A file the engineer
// added is a read-only link to its Blob, read in place when code opens it, so a 650 MB surface costs nothing until
// read. A file the assistant wrote is a plain file. After each run the directory is compared with what it held:
// files written are sent to the page; an engineer's file removed, replaced or renamed is put back as it was and the
// change is sent as a request the engineer approves or declines.
//
// Messages from the page: start { modules, files }, run { id, code, cases }, reply { id, error, value, cases },
// file { kind: 'place' | 'remove' | 'move', path, to?, blob?, by? }. To the page: ready, failed { message },
// output { run, item }, request { id, kind, payload }, done { id, error, files }.

import { loadPyodide } from 'https://cdn.jsdelivr.net/pyodide/v314.0.8/full/pyodide.mjs';

const PACKAGES = ['numpy', 'scipy', 'pandas', 'matplotlib'];
const WORK = '/work';
const BLOBS = '/blobs';

let pyodide;
let FS;
let runner;
/** The run in progress, whose output the page is shown. */
let running = null;
/** Requests to the page awaiting their reply, by id. */
const pending = new Map();
let nextRequest = 1;
/** The engineer's files in /work: path → the WORKERFS directory holding its Blob. */
const engineerFiles = new Map();
let nextBlob = 1;
/** What /work held after the last comparison: path → { link } or { size, modified }. */
let held = new Map();

/** Settles once Pyodide, its packages and the workspace's files are in; every later message waits for it. */
let started;

onmessage = async ({ data }) => {
	if (data.type === 'start') {
		started = start(data).catch((error) => postMessage({ type: 'failed', message: error.message }));
		return;
	}
	await started;
	switch (data.type) {
		case 'run':
			run(data);
			return;
		case 'reply': {
			setCases(data.cases);
			const { resolve } = pending.get(data.id);
			pending.delete(data.id);
			resolve({ error: data.error, value: data.value });
			return;
		}
		case 'file':
			mirror(data);
			return;
	}
};

async function start({ modules, files }) {
	pyodide = await loadPyodide({ packages: PACKAGES, env: { MPLBACKEND: 'Agg', MPLCONFIGDIR: '/tmp/matplotlib' } });
	removeNetwork();
	FS = pyodide.FS;
	FS.mkdirTree('/lib');
	for (const [name, text] of Object.entries(modules)) FS.writeFile(`/lib/${name}`, text);
	FS.mkdirTree(WORK);
	FS.mkdirTree(BLOBS);
	for (const file of files) await place(file);
	held = scan();
	pyodide.setStdout({ write: printed.writer('stdout') });
	pyodide.setStderr({ write: printed.writer('stderr') });
	pyodide.registerJsModule('_page', { request, emit });
	pyodide.runPython(`import os, sys\nsys.path.insert(0, '/lib')\nos.chdir('${WORK}')`);
	runner = pyodide.pyimport('runner');
	postMessage({ type: 'ready' });
}

/** Take away every way to make a request: the policy already confines them to the CDN, and now not even that. */
function removeNetwork() {
	for (const name of ['fetch', 'XMLHttpRequest', 'importScripts', 'WebSocket', 'EventSource', 'WebTransport', 'Worker']) {
		for (let scope = self; scope; scope = Object.getPrototypeOf(scope)) delete scope[name];
	}
}

async function run({ id, code, cases }) {
	running = id;
	printed.reset();
	try {
		setCases(cases);
		const error = await runner.run(code);
		printed.flush();
		const files = compare();
		postMessage({ type: 'done', id, error: error ?? null, files }, transfers(files));
	} catch (error) {
		postMessage({ type: 'failed', message: error.message });
	} finally {
		running = null;
	}
}

function setCases(cases) {
	runner.set_cases(JSON.stringify(cases));
}

/** Ask the page for something (an operation, a case's cells, a prediction) and wait for its reply. */
function request(kind, payload) {
	const id = nextRequest++;
	const plain = payload.toJs ? payload.toJs({ dict_converter: Object.fromEntries }) : payload;
	postMessage({ type: 'request', id, kind, payload: plain }, transfers(plain));
	return new Promise((resolve) => pending.set(id, { resolve }));
}

/** Something to show in the conversation: printed text, a table or a figure. */
function emit(item) {
	const plain = item.toJs ? item.toJs({ dict_converter: Object.fromEntries }) : item;
	postMessage({ type: 'output', run: running, item: plain }, transfers(plain));
}

/**
 * What a run prints, passed on a line at a time and kept up to OUTPUT_LIMIT characters a run: a print of a whole
 * array would otherwise send the page hundreds of megabytes, which nobody reads.
 */
const OUTPUT_LIMIT = 100_000;
const printed = {
	kept: 0,
	full: false,
	pending: { stdout: '', stderr: '' },
	reset() {
		this.kept = 0;
		this.full = false;
	},
	writer(stream) {
		const decoder = new TextDecoder();
		return (bytes) => {
			if (this.full) return bytes.length;
			this.pending[stream] += decoder.decode(bytes, { stream: true });
			const end = this.pending[stream].lastIndexOf('\n');
			if (this.kept + this.pending[stream].length > OUTPUT_LIMIT) this.send(stream, this.pending[stream]);
			else if (end >= 0) this.send(stream, this.pending[stream].slice(0, end + 1));
			return bytes.length;
		};
	},
	/** Pass on `text`, the start of what `stream` holds, as far as the limit allows. */
	send(stream, text) {
		this.pending[stream] = this.pending[stream].slice(text.length);
		const room = OUTPUT_LIMIT - this.kept;
		if (text.length > room) {
			text = `${text.slice(0, room)}\n… the rest of this run's output was not kept (over ${OUTPUT_LIMIT.toLocaleString('en')} characters).\n`;
			this.full = true;
			this.pending = { stdout: '', stderr: '' };
		}
		this.kept += text.length;
		emit({ kind: 'text', text, ...(stream === 'stderr' && { stream }) });
	},
	flush() {
		for (const stream of ['stdout', 'stderr']) if (this.pending[stream]) this.send(stream, this.pending[stream]);
	},
};

/** The buffers of the typed arrays in a message, to move rather than copy; never Pyodide's own memory. */
function transfers(value, found = new Set()) {
	if (ArrayBuffer.isView(value)) {
		if (value.buffer !== pyodide._module.HEAPU8.buffer) found.add(value.buffer);
	} else if (value !== null && typeof value === 'object') {
		for (const part of Object.values(value)) transfers(part, found);
	}
	return [...found];
}

// ---- the working directory ------------------------------------------------------------------------------------------

/** Put a workspace file in /work: the engineer's as a read-only link to its Blob, the assistant's as a plain file. */
async function place({ path, blob, by }) {
	removeEntry(path);
	const target = `${WORK}/${path}`;
	FS.mkdirTree(target.slice(0, target.lastIndexOf('/')));
	if (by === 'engineer') {
		const dir = `${BLOBS}/${nextBlob++}`;
		FS.mkdirTree(dir);
		FS.mount(FS.filesystems.WORKERFS, { blobs: [{ name: basename(path), data: blob }] }, dir);
		FS.symlink(`${dir}/${basename(path)}`, target);
		engineerFiles.set(path, dir);
	} else {
		FS.writeFile(target, new Uint8Array(await blob.arrayBuffer()));
	}
}

function removeEntry(path) {
	const target = `${WORK}/${path}`;
	if (FS.analyzePath(target).exists || isLink(target)) FS.unlink(target);
	const dir = engineerFiles.get(path);
	if (dir) {
		FS.unmount(dir);
		FS.rmdir(dir);
		engineerFiles.delete(path);
	}
}

/** Apply a change the workspace made (the engineer, an operation, or an approved request) to /work. */
async function mirror({ kind, path, to, blob, by }) {
	if (kind === 'place') await place({ path, blob, by });
	else if (kind === 'remove') removeEntry(path);
	else if (kind === 'move') {
		const target = `${WORK}/${to}`;
		FS.mkdirTree(target.slice(0, target.lastIndexOf('/')));
		FS.rename(`${WORK}/${path}`, target);
		if (engineerFiles.has(path)) {
			engineerFiles.set(to, engineerFiles.get(path));
			engineerFiles.delete(path);
		}
	}
	held = scan();
}

/**
 * What changed in /work since it was last compared: files the assistant wrote (with their bytes) and removed, and
 * requests to change the engineer's files, each of which is undone here until the engineer approves it.
 */
function compare() {
	const now = scan();
	const written = [];
	const removed = [];
	const requests = [];
	const linkPaths = new Map([...now].filter(([, entry]) => entry.link).map(([path, entry]) => [entry.link, path]));
	for (const [path, dir] of engineerFiles) {
		const original = `${dir}/${basename(path)}`;
		const entry = now.get(path);
		if (entry?.link === original) continue;
		const movedTo = linkPaths.get(original);
		if (movedTo) {
			requests.push({ kind: 'move', path, to: movedTo });
			FS.unlink(`${WORK}/${movedTo}`);
			now.delete(movedTo);
		} else if (entry) {
			requests.push({ kind: 'replace', path, bytes: FS.readFile(`${WORK}/${path}`) });
			FS.unlink(`${WORK}/${path}`);
		} else {
			requests.push({ kind: 'remove', path });
		}
		FS.mkdirTree(`${WORK}/${path}`.slice(0, `${WORK}/${path}`.lastIndexOf('/')));
		FS.symlink(original, `${WORK}/${path}`);
		now.set(path, { link: original });
	}
	for (const [path, entry] of now) {
		if (entry.link) continue;
		const before = held.get(path);
		if (!before || before.size !== entry.size || before.modified !== entry.modified) written.push({ path, bytes: FS.readFile(`${WORK}/${path}`) });
	}
	for (const [path, entry] of held) if (!entry.link && !now.has(path)) removed.push(path);
	held = now;
	return { written, removed, requests };
}

/** Every file and link under /work, by path relative to it. */
function scan(dir = WORK, prefix = '', found = new Map()) {
	for (const name of FS.readdir(dir)) {
		if (name === '.' || name === '..') continue;
		const full = `${dir}/${name}`;
		const path = prefix + name;
		if (isLink(full)) found.set(path, { link: FS.readlink(full) });
		else {
			const stat = FS.stat(full);
			if (FS.isDir(stat.mode)) scan(full, `${path}/`, found);
			else found.set(path, { size: stat.size, modified: stat.mtime.getTime() });
		}
	}
	return found;
}

function isLink(path) {
	try {
		return FS.isLink(FS.lstat(path).mode);
	} catch {
		return false;
	}
}

const basename = (path) => path.slice(path.lastIndexOf('/') + 1);
