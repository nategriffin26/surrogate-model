// The workspace in a panel that slides in from the left edge: every file with what it is, who made it and when;
// files come in by dropping or choosing them and go out by download; a geometry runs the model and opens its
// result, and shows when it has one. The bundled examples sit under the files, so everything that can be opened is
// in one place, and the tabs are what is open. The file operations are defined here too, so the panel and the
// assistant act on the workspace alike.

import { caseSummary } from './cases.js';
import { download, failureNote, h, s } from './dom.js';
import { EXAMPLES, exampleFacts } from './examples.js';
import { bytes, millions, seconds } from './format.js';
import { icon } from './icons.js';
import { menu } from './menu.js';
import { defineOperation } from './operations.js';
import { normalPath } from './workspace.js';

/** What a file is, by its extension: a label, an icon, and whether it reads as text. */
const KINDS = [
	{ test: /\.vtp$/i, label: 'Geometry', icon: 'geometry', text: false },
	{ test: /\.(csv|tsv)$/i, label: 'Table', icon: 'table', text: true },
	{ test: /\.(png|jpe?g|webp|gif)$/i, label: 'Image', icon: 'image', text: false },
	{ test: /\.svg$/i, label: 'Image', icon: 'image', text: true },
	{ test: /\.html?$/i, label: 'Report', icon: 'report', text: true },
	{ test: /\.(txt|md|json|py|log|ya?ml)$/i, label: 'Text', icon: 'text', text: true },
	{ test: /\.(npy|npz)$/i, label: 'Array', icon: 'file', text: false },
];
const OTHER = { label: 'File', icon: 'file', text: false };
const kindOf = (path) => KINDS.find((k) => k.test.test(path)) ?? OTHER;
const isGeometry = (path) => kindOf(path).icon === 'geometry';
const baseOf = (path) => path.slice(path.lastIndexOf('/') + 1);
const folderOf = (path) => (path.includes('/') ? path.slice(0, path.lastIndexOf('/')) : '');

/** Text read back to the assistant is capped: a table or log longer than this is read in part. */
const READ_LIMIT = 200_000;

/** A case's analysis settled: ready or failed. Resolves at once if it already has. */
function settled(session, c) {
	return new Promise((resolve) => {
		const check = () => {
			if (c.status === 'ready' || c.status === 'failed' || c.closed) {
				session.removeEventListener('change', check);
				resolve(c);
			}
		};
		session.addEventListener('change', check);
		check();
	});
}

/** The engine reads a File; a workspace blob becomes one named for its path, without copying its bytes. */
const fileFor = (path, blob) =>
	blob instanceof File && blob.name === baseOf(path) ? blob : new File([blob], baseOf(path));

/** context: { session, workspace, engine, update }. */
export function defineFileOperations({ session, workspace, engine, update }) {
	const fileAt = (path) => {
		const file = workspace.file(normalPath(path));
		if (!file) {
			const listed = workspace
				.files()
				.map((f) => f.path)
				.slice(0, 20);
			throw new Error(`There is no file "${path}". Files: ${listed.join(', ') || 'none'}.`);
		}
		return file;
	};
	/** The files at a normalised path, or under it when it names a folder. */
	const filesAt = (path) => workspace.files().filter((f) => f.path === path || f.path.startsWith(`${path}/`));

	/** The case showing a geometry, made and analysed if it has none. */
	function openPath(path) {
		const file = fileAt(path);
		if (!isGeometry(file.path))
			throw new Error(`${file.path} is not a geometry; only .vtp files open as cases. Read text files with read_file.`);
		const open = session.cases.find((c) => c.id === file.caseId);
		if (open) {
			session.activate(open);
			return open;
		}
		const c = session.openFile(fileFor(file.path, file.blob), file.path);
		workspace.link(file.path, c.id);
		return c;
	}

	// A closed case lets go of its file; a moved file keeps its case.
	session.addEventListener('change', ({ detail: { case: c, what } }) => {
		if (what === 'cases' && c?.closed && c.path) workspace.link(c.path, null);
	});
	workspace.onChange(({ kind, path, from }) => {
		const c = session.cases.find((x) => x.path === (kind === 'move' ? from : path));
		if (!c) return;
		if (kind === 'move') c.path = path;
		if (kind === 'remove' || (kind === 'write' && !workspace.file(path)?.caseId)) c.path = null;
	});

	const pathParameter = { type: 'string', description: 'Workspace path, as list_files gives it.' };

	defineOperation({
		name: 'list_files',
		description:
			'The files in the workspace, which the engineer sees in the file panel and the Python kernel sees as its working directory: geometry (.vtp) the engineer dropped or that was made in a study, tables, pictures, reports and anything else written there. Each with its path, kind, size, when and by whom (engineer or assistant) it was last written, and for a geometry the case_id of the open case showing its result.',
		parameters: {
			type: 'object',
			properties: { folder: { type: 'string', description: 'Only files under this folder; default all.' } },
			additionalProperties: false,
		},
		run({ folder }) {
			const prefix = folder ? `${normalPath(folder)}/` : '';
			return {
				files: workspace
					.files()
					.filter((f) => f.path.startsWith(prefix))
					.map((f) => ({
						path: f.path,
						kind: kindOf(f.path).label.toLowerCase(),
						size_bytes: f.size,
						modified: new Date(f.modified).toISOString(),
						by: f.by,
						...(f.caseId ? { case_id: f.caseId } : {}),
					})),
			};
		},
	});

	defineOperation({
		name: 'read_file',
		description: `Read a text file from the workspace (CSV, text, JSON, Python, HTML), up to ${READ_LIMIT / 1000} kB from its start. For large tables and arrays use the Python kernel, which reads the same files.`,
		parameters: {
			type: 'object',
			properties: {
				path: pathParameter,
				max_bytes: { type: 'integer', description: `Read at most this many bytes; default and limit ${READ_LIMIT}.` },
			},
			required: ['path'],
			additionalProperties: false,
		},
		async run({ path, max_bytes = READ_LIMIT }) {
			const file = fileAt(path);
			if (!kindOf(file.path).text)
				throw new Error(
					`${file.path} is a ${kindOf(file.path).label.toLowerCase()} file, not text. Read it in the Python kernel.`,
				);
			const limit = Math.min(READ_LIMIT, Math.max(1, max_bytes));
			return {
				path: file.path,
				size_bytes: file.size,
				text: await file.blob.slice(0, limit).text(),
				truncated: file.size > limit,
			};
		},
	});

	defineOperation({
		name: 'open_file',
		description:
			'Open a geometry (.vtp) from the workspace as a case: the model runs on it in the browser (reading, measuring, meshing, predicting, a few seconds for a DrivAerML car) and its result is shown in a new tab. A geometry already open is shown again instead. Resolves once the result is ready, with the case_id; the background prediction of every polygon continues after that.',
		parameters: {
			type: 'object',
			properties: { path: pathParameter },
			required: ['path'],
			additionalProperties: false,
		},
		async run({ path }) {
			const c = await settled(session, openPath(path));
			if (c.status === 'failed') throw new Error(c.error);
			return caseSummary(c, session.active);
		},
	});

	defineOperation({
		name: 'attach_reference',
		description:
			"Give a dropped car a reference field from a .vtp in the workspace: one of its cell arrays (default CpMeanTrim, the CFD pressure DrivAerML files carry), put on the car's polygons. The car can then be compared with it (set_comparison with \"cfd\"), and its result reports the CFD's whole-car numbers. The file must describe the same surface, its polygons within 5 mm of the car's: the CFD of another car is refused. The engine holds the car analysed last only, so another car is first analysed again (about as long as its first analysis); a car whose every polygon this device could not predict cannot take one. Reading a DrivAerML file takes a few seconds.",
		parameters: {
			type: 'object',
			properties: {
				path: pathParameter,
				case_id: { type: 'integer', description: 'The car to give it to; default the case on screen.' },
				array: { type: 'string', description: 'The cell array to read; default CpMeanTrim.' },
			},
			required: ['path'],
			additionalProperties: false,
		},
		async run({ path, case_id, array }) {
			const c = case_id === undefined ? session.active : session.byId(case_id);
			if (c?.kind !== 'file' || !c.result)
				throw new Error("A reference is attached to a dropped file's result. Open the geometry first.");
			const file = fileAt(path);
			if (!isGeometry(file.path)) throw new Error(`${file.path} is not a .vtp file.`);
			if (!engine.canExportVtp(c.result)) {
				if (!c.exact)
					throw new Error(`${c.name} cannot take a reference: this device did not predict its every polygon.`);
				if (!c.path || !workspace.file(c.path))
					throw new Error(
						`${c.name}'s file is no longer in the workspace, so it cannot be analysed again to take a reference.`,
					);
				await session.analyseAgain(c, fileFor(c.path, workspace.file(c.path).blob));
			}
			const { cfd, fullCfd } = await engine.attachReference(c.result, fileFor(file.path, file.blob), { array });
			// The surface drawn takes the field too, for the views that draw CFD from the GPU.
			c.surface.working.setFields({ cfd });
			if (fullCfd && c.surface.full) c.surface.full.setFields({ cfd: fullCfd });
			update('card', 'toolbar', 'dock', 'viewer');
			return {
				case_id: c.id,
				reference: c.result.reference,
				cx_cfd: c.result.referenceTotals.coefficients[0],
				cx_predicted: c.result.forces.cdP,
			};
		},
	});

	defineOperation({
		name: 'move_file',
		description:
			'Rename or move a file or folder in the workspace: from one path to another, e.g. "boundary_1.vtp" to "baseline/boundary_1.vtp". Fails if the destination exists. Moving a file the engineer wrote asks the engineer first. An open case keeps showing a moved geometry.',
		parameters: {
			type: 'object',
			properties: { from: pathParameter, to: { type: 'string', description: 'New path.' } },
			required: ['from', 'to'],
			additionalProperties: false,
		},
		/** Asks when anything it moves was written by the engineer. A path the workspace refuses asks nothing: `run` fails on it. */
		confirm({ from }) {
			let path;
			try {
				path = normalPath(from);
			} catch {
				return false;
			}
			return filesAt(path).some((f) => f.by === 'engineer');
		},
		run({ from, to }, { by }) {
			const moved = workspace.move(from, to, by);
			return { moved: moved.map((f) => f.path) };
		},
	});

	defineOperation({
		name: 'delete_file',
		description:
			'Delete a file or folder from the workspace. An open case showing a deleted geometry stays open. Ask the engineer first.',
		parameters: {
			type: 'object',
			properties: { path: pathParameter },
			required: ['path'],
			additionalProperties: false,
		},
		confirm: true,
		run({ path }, { by }) {
			const target = normalPath(path);
			const count = filesAt(target).length;
			workspace.remove(target, by);
			return { deleted: target, files: count };
		},
	});

	defineOperation({
		name: 'download_file',
		description: "Download a file from the workspace to the engineer's computer.",
		parameters: {
			type: 'object',
			properties: { path: pathParameter },
			required: ['path'],
			additionalProperties: false,
		},
		confirm: true,
		run({ path }) {
			const file = fileAt(path);
			download(file.blob, baseOf(file.path));
			return { path: file.path, size_bytes: file.size };
		},
	});
}

/**
 * Run the model on a geometry from a control. A failed analysis is shown on its case's own card; the operation's
 * rejection repeats it for a caller that waits on the result, which a control does not.
 */
function openGeometry(run, path) {
	run('open_file', { path }).catch(() => {});
}

/** Add files the engineer dropped or chose; a geometry starts its analysis at once. */
export function addFiles(files, { workspace, run }) {
	for (const file of files) {
		const { path } = workspace.write(workspace.freePath(file.name), file, 'engineer');
		if (isGeometry(path)) openGeometry(run, path);
	}
}

const clock = (ms) => new Date(ms).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
/** What an analysis held and took: the file's cells, the time from reading to the last prediction, and the device. */
const runFacts = (r) => [
	`${millions(r.source.cells)} cells`,
	seconds(r.timings.reduce((sum, t) => sum + t.seconds, 0)),
	r.device,
];

/**
 * context: { session, workspace, engine, run, progress }; progress(c) is the share of a case's analysis done. A geometry's
 * row offers its CFD to the cars without CFD of their own; the engine holds the car analysed last only, so another is
 * analysed again first (attach_reference).
 */
export function createFilesPanel({ session, workspace, engine, run, progress }) {
	const input = h('input', {
		type: 'file',
		multiple: true,
		hidden: true,
		onchange: () => {
			addFiles([...input.files], { workspace, run });
			input.value = '';
		},
	});
	const list = h('div', { class: 'files-list', role: 'tree', 'aria-label': 'Files' });
	const examples = h('div', { class: 'files-examples' });
	const close = h(
		'button',
		{ type: 'button', class: 'icon-button', title: 'Close the Files panel', onclick: () => setOpen(false) },
		icon('close'),
	);
	const el = h(
		'aside',
		{ class: 'files', id: 'files', 'aria-label': 'Files' },
		h(
			'div',
			{ class: 'files-head' },
			h('span', { class: 'files-title' }, 'Files'),
			h(
				'button',
				{ type: 'button', class: 'button files-upload', onclick: () => input.click() },
				icon('upload'),
				'Upload',
			),
			close,
			input,
		),
		h('div', { class: 'files-body' }, list, h('div', { class: 'files-group' }, 'Examples'), examples),
	);
	const toggle = h(
		'button',
		{
			type: 'button',
			class: 'button files-button',
			'aria-pressed': 'false',
			'aria-controls': 'files',
			onclick: () => setOpen(!open),
		},
		icon('folder'),
		'Files',
	);

	/** Folders the engineer folded shut, by path. */
	const folded = new Set();
	/** The file being renamed or moved in place: { path, mode: 'rename' | 'move' }; the list holds still around it. */
	let editing = null;
	/** The file whose CFD is being read onto a car, and that car: { path, taker }, or null. */
	let attaching = null;

	/** Whether the panel is open: its column opens beside the stage, which narrows to make room. */
	let open = false;
	function setOpen(next) {
		open = next;
		toggle.setAttribute('aria-pressed', String(open));
		document.body.classList.toggle('files-open', open);
		if (open) render();
	}

	/** Run an operation from a control; a failure shows on the panel until the next action. */
	const note = failureNote();
	el.querySelector('.files-head').after(note.el);
	const act = (name, args) => note.run(run(name, args));

	/** A geometry's state: its case and how far its analysis has got. */
	function geometryState(file) {
		const c = session.cases.find((x) => x.id === file.caseId);
		if (!c) return { c: null, label: null };
		switch (c.status) {
			case 'queued':
				return { c, label: 'Queued', busy: true };
			case 'analysing':
				return { c, label: `${Math.round(progress(c) * 100)} %`, busy: true, fraction: progress(c) };
			case 'loading':
				return { c, label: null, busy: true };
			case 'ready':
				return { c, label: 'Result', ready: true };
			case 'failed':
				return { c, label: 'Failed', failed: true };
			default:
				throw new Error(`Unknown status ${c.status}`);
		}
	}

	/** Open a row: a geometry runs or shows its result; any other file opens in a browser tab to be read. */
	function openRow(file) {
		if (isGeometry(file.path)) openGeometry(run, file.path);
		else
			window.open(
				URL.createObjectURL(
					kindOf(file.path).text && !/\.html?$/i.test(file.path)
						? new Blob([file.blob], { type: 'text/plain' })
						: file.blob,
				),
				'_blank',
				'noopener',
			);
	}

	/**
	 * The cars a geometry's CFD can go to: every other dropped car without CFD of its own whose every polygon this
	 * device has predicted. None when the tool knows the file carries no CFD: its own analysis found none (a car
	 * that took CFD from another file counts as one without).
	 */
	function referenceTakers(path) {
		if (!isGeometry(path)) return [];
		const own = session.cases.find((c) => c.id === workspace.file(path)?.caseId);
		if (own?.result && own.result.reference?.mapping !== 'own') return [];
		return session.cases.filter(
			(c) =>
				c.kind === 'file' &&
				c.status === 'ready' &&
				c.path &&
				c.path !== path &&
				c.exact &&
				c.result.reference?.mapping !== 'own',
		);
	}

	/** The state an attaching row shows: the car analysed again while the engine no longer holds it, then the CFD read. */
	const attachingLabel = () =>
		engine.canExportVtp(attaching.taker.result) ? 'Reading CFD' : 'Analysing the car again';

	function rowMenu(file) {
		let m = null;
		const item = (label, onclick, cls = 'menu-item') =>
			h('button', { type: 'button', class: cls, onclick }, h('span', {}, label));
		const edit = (mode) => () => {
			m.close();
			editing = { path: file.path, mode };
			render();
			// Focused at once, so the first key typed lands in the field.
			const field = list.querySelector('.file-edit');
			field.focus();
			const dot = field.value.lastIndexOf('.');
			if (mode === 'rename') field.setSelectionRange(0, dot > 0 ? dot : field.value.length);
		};
		// Deleting takes a second click on the same item, so a slip of the pointer removes nothing.
		const remove = item(
			'Delete',
			() => {
				if (remove.dataset.armed !== 'true') {
					remove.dataset.armed = 'true';
					remove.firstChild.textContent = 'Delete, click to confirm';
					return;
				}
				m.close();
				act('delete_file', { path: file.path });
			},
			'menu-item danger',
		);
		const attach = referenceTakers(file.path).map((taker) =>
			item(`Use as CFD for ${taker.name}`, async () => {
				m.close();
				attaching = { path: file.path, taker };
				render();
				await act('attach_reference', { path: file.path, case_id: taker.id });
				attaching = null;
				render();
			}),
		);
		const items = h(
			'div',
			{ class: 'menu-list', role: 'menu' },
			item('Download', () => {
				m.close();
				act('download_file', { path: file.path });
			}),
			item('Rename', edit('rename')),
			item('Move to folder', edit('move')),
			attach,
			h('hr', { class: 'menu-rule' }),
			remove,
		);
		m = menu(
			h(
				'button',
				{ type: 'button', class: 'icon-button file-more', title: `More for ${baseOf(file.path)}` },
				icon('more'),
			),
			items,
			{ align: 'right' },
		);
		return m.el;
	}

	/** The inline field a rename or move is typed into. Enter commits, Escape or leaving it cancels. */
	function editor(file) {
		const rename = editing.mode === 'rename';
		const field = h('input', {
			class: 'file-edit',
			spellcheck: 'false',
			'aria-label': rename ? 'New name' : 'Folder',
			placeholder: rename ? '' : 'Folder, empty for the top',
		});
		field.value = rename ? baseOf(file.path) : folderOf(file.path);
		const finish = async (commit) => {
			if (!editing) return;
			const value = field.value.trim();
			editing = null;
			if (commit) {
				const to = rename
					? [folderOf(file.path), value].filter(Boolean).join('/')
					: [value, baseOf(file.path)].filter(Boolean).join('/');
				if (value || !rename) await act('move_file', { from: file.path, to });
			}
			render();
		};
		field.addEventListener('keydown', (e) => {
			if (e.key === 'Enter') finish(true);
			if (e.key === 'Escape') finish(false);
		});
		field.addEventListener('blur', () => finish(false));
		return field;
	}

	function fileRow(file, depth) {
		const kind = kindOf(file.path);
		const geometry = isGeometry(file.path) ? geometryState(file) : null;
		const shown = geometry?.c && geometry.c === session.active;
		// Once analysed, what the file held and how long it took on this device replaces who added it and when.
		const meta = (
			geometry?.ready
				? runFacts(geometry.c.result)
				: [kind.label, bytes(file.size), file.by === 'assistant' ? 'Assistant' : 'You', clock(file.modified)]
		).join(' · ');
		const name = editing?.path === file.path ? editor(file) : h('span', { class: 'file-name' }, baseOf(file.path));
		const state =
			attaching?.path === file.path
				? h('span', { class: 'file-state' }, attachingLabel())
				: geometry?.label
					? h(
							'span',
							{ class: `file-state${geometry.ready ? ' ready' : ''}${geometry.failed ? ' failed' : ''}` },
							geometry.label,
						)
					: null;
		const action =
			isGeometry(file.path) && !geometry.busy && !shown
				? h(
						'button',
						{
							type: 'button',
							class: 'file-action',
							title: geometry.c ? `Show the result of ${baseOf(file.path)}` : `Run the model on ${baseOf(file.path)}`,
							onclick: (e) => {
								e.stopPropagation();
								openRow(file);
							},
						},
						geometry.c ? 'Open' : h('span', { class: 'file-run' }, icon('run'), 'Run'),
					)
				: null;
		const row = h(
			'div',
			{
				class: `file-row${shown ? ' current' : ''}`,
				role: 'treeitem',
				tabindex: '0',
				draggable: 'true',
				'data-path': file.path,
				style: `--depth:${depth}`,
				title: file.path,
			},
			h('span', { class: 'file-icon' }, icon(kind.icon)),
			h(
				'span',
				{ class: 'file-text' },
				h('span', { class: 'file-line' }, name, state),
				h('span', { class: 'file-meta' }, meta),
			),
			h('span', { class: 'file-actions' }, action, rowMenu(file)),
			geometry?.fraction !== undefined
				? h('span', { class: 'file-progress', style: `transform:scaleX(${geometry.fraction})` })
				: null,
		);
		row.addEventListener('click', (e) => {
			if (editing || e.target.closest('.menu, .file-edit')) return;
			openRow(file);
		});
		row.addEventListener('keydown', (e) => {
			if (e.target === row && e.key === 'Enter') openRow(file);
		});
		row.addEventListener('dragstart', (e) => e.dataTransfer.setData('application/x-workspace-path', file.path));
		return row;
	}

	/** A folder row, which folds its contents and takes files dragged onto it. */
	function folderRow(path, depth, count) {
		const open = !folded.has(path);
		const row = h(
			'div',
			{
				class: 'file-row folder',
				role: 'treeitem',
				'aria-expanded': String(open),
				tabindex: '0',
				style: `--depth:${depth}`,
				'data-folder': path,
			},
			h('span', { class: 'file-icon' }, icon('folder')),
			h(
				'span',
				{ class: 'file-text' },
				h('span', { class: 'file-line' }, h('span', { class: 'file-name' }, baseOf(path))),
				h('span', { class: 'file-meta' }, `${count} ${count === 1 ? 'file' : 'files'}`),
			),
			h('span', { class: 'folder-chevron' }, icon('chevron')),
		);
		const flip = () => {
			if (open) folded.add(path);
			else folded.delete(path);
			render();
		};
		row.addEventListener('click', flip);
		row.addEventListener('keydown', (e) => e.key === 'Enter' && flip());
		acceptMoves(row, path);
		return row;
	}

	/** Files dragged within the panel move into the folder they are dropped on; the list itself is the top. */
	function acceptMoves(target, folder) {
		const moving = (e) => [...e.dataTransfer.types].includes('application/x-workspace-path');
		target.addEventListener('dragover', (e) => {
			if (!moving(e)) return;
			e.preventDefault();
			e.stopPropagation();
			target.classList.add('drop-target');
		});
		target.addEventListener('dragleave', () => target.classList.remove('drop-target'));
		target.addEventListener('drop', (e) => {
			if (!moving(e)) return;
			e.preventDefault();
			e.stopPropagation();
			target.classList.remove('drop-target');
			const from = e.dataTransfer.getData('application/x-workspace-path');
			const to = [folder, baseOf(from)].filter(Boolean).join('/');
			if (to !== from) act('move_file', { from, to });
		});
	}
	acceptMoves(list, '');

	/**
	 * Rows built so far, by key, with the state each was built from. A row whose state has not changed is kept as it
	 * is, so an open menu, the hover and the focus survive the frequent updates of an analysis elsewhere in the list.
	 */
	const built = new Map();
	function keyed(key, signature, build) {
		const known = built.get(key);
		if (known?.signature === signature) return known.el;
		const row = build();
		built.set(key, { signature, el: row });
		return row;
	}
	const empty = h(
		'label',
		{ class: 'files-empty' },
		h('span', {}, 'Drop files here or '),
		h('span', { class: 'link', onclick: () => input.click() }, 'choose them'),
	);

	function render() {
		if (!open) return;
		// A rename being typed keeps its row: rebuilding would take its field away mid-word.
		if (editing && list.querySelector('.file-edit') === document.activeElement) {
			refreshStates();
			return;
		}
		const files = workspace.files();
		const rows = [];
		const seen = new Set();
		const used = new Set();
		for (const file of files) {
			const parts = file.path.split('/');
			let hiddenUnder = false;
			for (let d = 1; d < parts.length; d++) {
				const folder = parts.slice(0, d).join('/');
				if (!seen.has(folder)) {
					seen.add(folder);
					const count = files.filter((f) => f.path.startsWith(`${folder}/`)).length;
					const key = `folder:${folder}`;
					used.add(key);
					if (!hiddenUnder)
						rows.push(keyed(key, `${d}|${count}|${folded.has(folder)}`, () => folderRow(folder, d - 1, count)));
				}
				if (folded.has(folder)) hiddenUnder = true;
			}
			if (hiddenUnder) continue;
			const c = session.cases.find((x) => x.id === file.caseId);
			const key = `file:${file.path}`;
			used.add(key);
			const signature = [
				file.size,
				file.by,
				file.modified,
				file.caseId,
				c?.status,
				c === session.active,
				editing?.path === file.path && editing.mode,
				referenceTakers(file.path)
					.map((taker) => taker.id)
					.join(','),
				attaching?.path === file.path && attachingLabel(),
			].join('|');
			rows.push(keyed(key, signature, () => fileRow(file, parts.length - 1)));
		}
		for (const key of built.keys()) if (!used.has(key)) built.delete(key);
		const wanted = rows.length ? rows : [empty];
		if (wanted.length !== list.children.length || wanted.some((row, i) => row !== list.children[i]))
			list.replaceChildren(...wanted);
		renderExamples();
	}

	/** Progress moves often; it updates the rows' states without rebuilding them. */
	function refreshStates() {
		for (const row of list.querySelectorAll('.file-row[data-path]')) {
			const file = workspace.file(row.dataset.path);
			if (!file || !isGeometry(file.path)) continue;
			const state = geometryState(file);
			const label = row.querySelector('.file-state');
			if (label && state.label) label.textContent = state.label;
			const bar = row.querySelector('.file-progress');
			if (bar && state.fraction !== undefined) bar.style.transform = `scaleX(${state.fraction})`;
		}
	}

	/** Each example's size, read from its own summary once. */
	let facts = [];
	exampleFacts().then((texts) => {
		facts = texts;
		renderExamples();
	});

	let examplesShown = null;
	function renderExamples() {
		const states = EXAMPLES.map((example) => session.cases.find((x) => x.example === example.id));
		const signature = JSON.stringify([facts, states.map((c) => c && [c.id, c.status, c === session.active])]);
		if (signature === examplesShown) return;
		examplesShown = signature;
		examples.replaceChildren(
			...EXAMPLES.map((example, i) => {
				const c = states[i];
				const shown = c && c === session.active;
				return h(
					'button',
					{
						type: 'button',
						class: `example-row${shown ? ' current' : ''}`,
						'data-example-row': example.id,
						onclick: () => act(c ? 'show_case' : 'open_example', c ? { case_id: c.id } : { example: example.id }),
					},
					s(
						'svg',
						{ class: 'example-mark', viewBox: '0 0 132 40', 'aria-hidden': 'true' },
						s('path', { d: example.profile }),
					),
					h(
						'span',
						{ class: 'file-text' },
						h(
							'span',
							{ class: 'file-line' },
							h('span', { class: 'file-name' }, example.label),
							c?.status === 'ready' ? h('span', { class: 'file-state ready' }, 'Result') : null,
						),
						h('span', { class: 'file-meta' }, ['Example', facts[i]].filter(Boolean).join(' · ')),
					),
				);
			}),
		);
	}

	workspace.onChange(() => render());
	// The list shows which cases exist, their status and which is on screen, and once a car's every polygon is predicted
	// (`numbers`) which can take another file's CFD; an analysis's progress moves only its row's state.
	session.addEventListener('change', ({ detail: { what } }) => {
		if (what === 'stage') refreshStates();
		else if (what === 'cases' || what === 'status' || what === 'active' || what === 'numbers') render();
	});

	return { el, toggle, setOpen };
}
