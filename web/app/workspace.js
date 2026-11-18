// The workspace is the one set of files in the tool: geometry the engineer dropped, results and exports, and
// everything the assistant's Python kernel reads and writes. The file panel shows it, the kernel's working
// directory mirrors it, and file operations in the registry act on it, so the engineer and the assistant
// always see the same files.
//
// A file is held as a Blob. A dropped 650 MB `.vtp` stays the browser's handle to the file on disk and is
// never copied into memory here.

/**
 * @typedef {Object} WorkspaceFile
 * @property {string} path                    Forward slashes, no leading slash: `boundary_1.vtp`, `study/ride_height.csv`.
 * @property {Blob} blob
 * @property {number} size                    Bytes.
 * @property {number} modified                Milliseconds since the epoch.
 * @property {'engineer' | 'assistant'} by    Who wrote it last.
 * @property {number} [caseId]                Set on a geometry once it is analysed: the open case showing it.
 */

/**
 * @typedef {Object} Workspace
 * @property {() => WorkspaceFile[]} files                       Every file, sorted by path.
 * @property {(path: string) => WorkspaceFile | undefined} file
 * @property {(path: string, blob: Blob, by: 'engineer' | 'assistant') => WorkspaceFile} write
 *   Creates or replaces. Replacing a file the engineer wrote, when `by` is the assistant, needs approval first.
 *   A replaced geometry loses its case: the case shows what the file held before.
 * @property {(from: string, to: string, by: 'engineer' | 'assistant') => WorkspaceFile[]} move
 *   Moves a file, or a folder with everything in it. Throws when the destination is taken.
 * @property {(path: string, by: 'engineer' | 'assistant') => void} remove
 *   Removes a file, or a folder with everything in it.
 *   Nothing the assistant does removes a file without the engineer's approval: not an operation, and not code
 *   in the kernel deleting from its working directory. The caller obtains the approval; this records the act.
 * @property {(path: string, caseId: number | null) => void} link  Record the case showing a geometry, or none.
 * @property {(path: string) => string} freePath  `path`, or the first of `name (2).ext`, `name (3).ext`… not taken.
 * @property {(listener: (change: { kind: 'write' | 'move' | 'remove' | 'link', path: string, from?: string, by?: string }) => void) => () => void} onChange
 */

/** A path as the workspace keeps it, or an error saying what is wrong with it. */
export function normalPath(path) {
	if (typeof path !== 'string') throw new Error('A path is a string such as "study/variants.csv".');
	const parts = path
		.replaceAll('\\', '/')
		.split('/')
		.filter((part) => part !== '' && part !== '.');
	if (parts.length === 0) throw new Error('The path is empty. Give a file name such as "notes.txt".');
	if (parts.includes('..')) throw new Error(`"${path}" climbs out of the workspace with "..". Give a path inside it.`);
	return parts.join('/');
}

/** @returns {Workspace} */
export function createWorkspace() {
	/** @type {Map<string, WorkspaceFile>} */
	const entries = new Map();
	const listeners = new Set();
	const emit = (change) => {
		for (const listener of listeners) listener(change);
	};
	/** The files at `path` itself, or under it when it names a folder. */
	const under = (path) => (entries.has(path) ? [path] : [...entries.keys()].filter((p) => p.startsWith(`${path}/`)));

	function write(path, blob, by) {
		path = normalPath(path);
		if ([...entries.keys()].some((p) => p.startsWith(`${path}/`)))
			throw new Error(`"${path}" is a folder. Give a file name inside it.`);
		const file = { path, blob, size: blob.size, modified: Date.now(), by };
		entries.set(path, file);
		emit({ kind: 'write', path, by });
		return file;
	}

	function move(from, to, by) {
		from = normalPath(from);
		to = normalPath(to);
		const paths = under(from);
		if (paths.length === 0) throw new Error(`There is no file or folder "${from}".`);
		if (to === from) return paths.map((p) => entries.get(p));
		if (to.startsWith(`${from}/`)) throw new Error(`"${from}" cannot move into itself.`);
		const target = (p) => to + p.slice(from.length);
		const taken = paths.map(target).find((p) => entries.has(p) || under(p).length > 0);
		if (taken) throw new Error(`"${taken}" already exists. Choose another name, or delete it first.`);
		return paths.map((p) => {
			const file = { ...entries.get(p), path: target(p) };
			entries.delete(p);
			entries.set(file.path, file);
			emit({ kind: 'move', path: file.path, from: p, by });
			return file;
		});
	}

	function remove(path, by) {
		path = normalPath(path);
		const paths = under(path);
		if (paths.length === 0) throw new Error(`There is no file or folder "${path}".`);
		for (const p of paths) {
			entries.delete(p);
			emit({ kind: 'remove', path: p, by });
		}
	}

	function link(path, caseId) {
		const file = entries.get(path);
		if (!file || (file.caseId ?? null) === caseId) return;
		entries.set(path, { ...file, caseId: caseId ?? undefined });
		emit({ kind: 'link', path });
	}

	function freePath(path) {
		path = normalPath(path);
		// The extension starts at the name's last dot, unless the name starts there (".env").
		const dot = path.lastIndexOf('.');
		const [stem, ext] = dot > path.lastIndexOf('/') + 1 ? [path.slice(0, dot), path.slice(dot)] : [path, ''];
		let candidate = path;
		for (let n = 2; entries.has(candidate) || under(candidate).length > 0; n++) candidate = `${stem} (${n})${ext}`;
		return candidate;
	}

	return {
		files: () => [...entries.values()].sort((a, b) => a.path.localeCompare(b.path)),
		file: (path) => entries.get(path),
		write,
		move,
		remove,
		link,
		freePath,
		onChange: (listener) => (listeners.add(listener), () => listeners.delete(listener)),
	};
}
