// What can be taken away from the tool. Each export is a file made on demand from the case on screen; the Export
// menu offers every one that is available, to download or to save into the workspace, and so do the operations.
// The picture is the view with its legend and the file's name; the tools add their own tables.

import { download, failureNote, h } from './dom.js';
import { FIELDS } from './fields.js';
import { icon } from './icons.js';
import { paintLegend } from './legend.js';
import { menu } from './menu.js';
import { defineOperation } from './operations.js';

/** Download names start from the file analysed, without its extension. */
export const baseName = (c) => (c.result?.source.name ?? c.name).replace(/\.[^.]+$/, '');

/**
 * @typedef {Object} Export
 * @property {string} name      snake_case id, as the operations take it
 * @property {string} label     the menu's word for it
 * @property {string} format    file type, shown beside the label: PNG, CSV, VTP, HTML
 * @property {string} [title]   tooltip
 * @property {(c: object) => string | null} unavailable  why it cannot be made now, or null when it can
 * @property {(c: object) => Promise<Blob>} make
 * @property {(c: object) => string} fileName
 */

/** @type {Map<string, Export>} */
export const EXPORTS = new Map();

export function defineExport(spec) {
	EXPORTS.set(spec.name, spec);
}

/**
 * The view as drawn, with the legend in its corner and the file's name in the other. `marks(ctx, scale)`, when given,
 * paints what the page shows over the 3D view (labels on the car), `scale` image pixels to a CSS pixel.
 */
export function pictureOf(c, viewer, { field, map, marks }) {
	const shot = viewer.snapshot();
	const image = document.createElement('canvas');
	image.width = shot.width;
	image.height = shot.height;
	const ctx = image.getContext('2d');
	const ground = ctx.createLinearGradient(0, 0, 0, image.height);
	ground.addColorStop(0, '#ffffff');
	ground.addColorStop(0.34, '#ffffff');
	ground.addColorStop(1, '#eef4fb');
	ctx.fillStyle = ground;
	ctx.fillRect(0, 0, image.width, image.height);
	ctx.drawImage(shot, 0, 0);
	const scale = image.width / viewer.host.clientWidth;
	if (marks) {
		ctx.save();
		marks(ctx, scale);
		ctx.restore();
	}
	ctx.textBaseline = 'top';
	ctx.textAlign = 'right';
	ctx.font = `600 ${13 * scale}px "Hanken Grotesk", system-ui, sans-serif`;
	ctx.fillStyle = '#0d2233';
	ctx.fillText(c.name, image.width - 16 * scale, 14 * scale);
	if (FIELDS[field].map) paintLegend(ctx, 16 * scale, image.height - 80 * scale, scale, FIELDS[field].title, map);
	return new Promise((resolve) => image.toBlob(resolve, 'image/png'));
}

/** The export's file for the case on screen, or an error saying why it cannot be made now. */
async function makeExport(name, session) {
	const spec = EXPORTS.get(name);
	if (!spec) throw new Error(`There is no export "${name}". Exports: ${[...EXPORTS.keys()].join(', ')}.`);
	const c = session.withResult();
	const why = spec.unavailable(c);
	if (why) throw new Error(why);
	return { blob: await spec.make(c), fileName: spec.fileName(c) };
}

/**
 * Write a file into the workspace for whoever asked. The assistant never replaces a file the engineer wrote; with
 * no path given the file takes its own name, numbered if that is taken.
 */
function saveFile(workspace, path, fileName, blob, by) {
	const target = path ?? workspace.freePath(fileName);
	const existing = workspace.file(target);
	if (existing && existing.by === 'engineer' && by === 'assistant') {
		throw new Error(
			`${target} was written by the engineer. Save under another path, or ask the engineer to delete it first.`,
		);
	}
	return workspace.write(target, blob, by);
}

/** context: { session, workspace, engine }. The surface export, and the operations that make any export. */
export function defineExportOperations({ session, workspace, engine }) {
	defineExport({
		name: 'surface',
		label: 'Surface',
		format: 'VTP',
		title: 'The analysed file with CpPredicted and CpSpread cell arrays added',
		// The engine holds the whole predicted surface of the file analysed last only, from its exact numbers on. A
		// variant's surface was written with its prediction when it was made.
		unavailable: (c) =>
			engine.canExportVtp(c.result)
				? null
				: c.variant
					? `The variant's surface, with CpPredicted and CpSpread, is in the workspace at ${c.path}.`
					: 'The surface file exists for the file analysed last, once every polygon is predicted; open or reopen it to export it.',
		make: (c) => engine.exportVtp(c.result),
		fileName: (c) => `${baseName(c)}_prediction.vtp`,
	});

	const describe = () => [...EXPORTS.values()].map((e) => `${e.name} (${e.label}, ${e.format})`).join(', ');
	const exportParameter = () => ({
		type: 'string',
		enum: [...EXPORTS.keys()],
		description: `What to export: ${describe()}.`,
	});

	defineOperation({
		name: 'save_export',
		description:
			"Make an export of the case on screen and save it into the workspace as a file, where the engineer sees it and the Python kernel can read it. picture: the 3D view as drawn, with the colour legend. surface: the analysed .vtp with CpPredicted and CpSpread cell arrays added (a dropped file once every polygon is predicted). section, probes, comparison and study: the section plot's points, the probe table, the Compare table and the Variants table, as CSV. report: one HTML page with the picture, the whole-car numbers, and the section, probes, comparison and study the car has. Returns the path and size.",
		get parameters() {
			return {
				type: 'object',
				properties: {
					export: exportParameter(),
					path: {
						type: 'string',
						description: 'Workspace path to save to, e.g. "figures/rear.png"; default the export\'s own name.',
					},
				},
				required: ['export'],
				additionalProperties: false,
			};
		},
		async run({ export: name, path }, { by }) {
			const { blob, fileName } = await makeExport(name, session);
			const file = saveFile(workspace, path, fileName, blob, by);
			return { path: file.path, size_bytes: file.size };
		},
	});

	defineOperation({
		name: 'download_export',
		description:
			"Make an export of the case on screen and download it to the engineer's computer. The same exports as save_export.",
		get parameters() {
			return {
				type: 'object',
				properties: { export: exportParameter() },
				required: ['export'],
				additionalProperties: false,
			};
		},
		confirm: true,
		async run({ export: name }) {
			const { blob, fileName } = await makeExport(name, session);
			download(blob, fileName);
			return { file_name: fileName, size_bytes: blob.size };
		},
	});
}

/**
 * The Export menu: every export, downloaded by its row or saved into the workspace by the folder beside it. Those
 * that cannot be made now are disabled with the reason as their tooltip; the surface file is offered only while the
 * engine holds the car's whole predicted surface. `render()` brings the rows up to date with the case on screen.
 */
export function createExportMenu({ run, active }) {
	const note = failureNote();
	const list = h('div', { class: 'menu-list export-list' });
	const button = h('button', { type: 'button', class: 'button export-button' }, icon('download'), 'Export');
	const exportMenu = menu(button, h('div', {}, list, note.el), { align: 'right' });

	async function exportTo(row, name, to) {
		row.classList.add('busy');
		const saved = await note.run(run(to === 'download' ? 'download_export' : 'save_export', { export: name }));
		row.classList.remove('busy');
		if (saved) exportMenu.close();
	}

	function render() {
		const c = active();
		list.replaceChildren(
			...[...EXPORTS.values()].flatMap((e) => {
				const why = e.unavailable(c);
				if (e.name === 'surface' && why) return [];
				const row = h(
					'div',
					{ class: 'export-row', 'data-export': e.name },
					h(
						'button',
						{
							type: 'button',
							class: 'menu-item',
							title: why ?? e.title,
							disabled: Boolean(why),
							onclick: () => exportTo(row, e.name, 'download'),
						},
						h('span', {}, e.label),
						h('kbd', {}, e.format),
					),
					h(
						'button',
						{
							type: 'button',
							class: 'icon-button export-keep',
							title: `Save ${e.label.toLowerCase()} to files`,
							disabled: Boolean(why),
							onclick: () => exportTo(row, e.name, 'files'),
						},
						icon('keep'),
					),
				);
				return [row];
			}),
		);
	}
	button.addEventListener('click', () => {
		note.hide();
		render();
	});

	return { el: exportMenu.el, render };
}
