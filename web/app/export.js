// Taking results away: a picture of the 3D view with its legend and the file's name and pressure drag. The plots
// and probes save their own CSV files.

import { download } from './dom.js';
import { FIELDS } from './fields.js';
import { paintLegend } from './legend.js';

/** Download names start from the file analysed, without its extension. */
export const baseName = (c) => (c.result?.source.name ?? c.name).replace(/\.[^.]+$/, '');

/** The view as drawn, with the legend in its corner and the file and its pressure drag in the other. */
export function savePicture(c, viewer, { field, map }) {
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
	const font = (size, weight = 400) => `${weight} ${size * scale}px "Hanken Grotesk", system-ui, sans-serif`;
	ctx.textBaseline = 'top';
	ctx.textAlign = 'right';
	ctx.font = font(13, 600);
	ctx.fillStyle = '#0d2233';
	ctx.fillText(c.name, image.width - 16 * scale, 14 * scale);
	ctx.textAlign = 'left';
	ctx.fillStyle = '#416473';
	// C with the subscript D,p, then the value, set right-aligned as one line.
	const parts = [['C', font(12), 0], ['D,p', font(9), 5 * scale], [` ${c.result.forces.cdP.toFixed(3)}`, font(12), 0]];
	const widths = parts.map(([text, f]) => ((ctx.font = f), ctx.measureText(text).width));
	let x = image.width - 16 * scale - widths.reduce((a, b) => a + b, 0);
	parts.forEach(([text, f, drop], i) => {
		ctx.font = f;
		ctx.fillText(text, x, 34 * scale + drop);
		x += widths[i];
	});
	if (FIELDS[field].map) paintLegend(ctx, 16 * scale, image.height - 80 * scale, scale, FIELDS[field].title, map);
	image.toBlob((blob) => download(blob, `${baseName(c)}-${field}.png`), 'image/png');
}
