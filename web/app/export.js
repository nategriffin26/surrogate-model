// Taking results away: a picture of the 3D view with its legend, and CSV files of the current plot and the probes.

import { download } from './dom.js';
import { FIELDS } from './fields.js';
import { paintLegend } from './legend.js';

/** Download names start from the file analysed, without its extension. */
export const baseName = (c) => (c.result?.source.name ?? c.name).replace(/\.[^.]+$/, '');

/** The view as drawn, with the legend in its corner, the pane names in compare, and the file and field. */
export function savePicture(c, viewer, panes) {
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
	ctx.fillStyle = '#0d2233';
	const paneWidth = image.width / panes.length;
	if (panes.length > 1) {
		ctx.font = font(13, 600);
		panes.forEach((pane, i) => ctx.fillText(pane.label, i * paneWidth + 16 * scale, 14 * scale));
	}
	ctx.font = font(12);
	ctx.fillStyle = '#416473';
	ctx.textAlign = 'right';
	ctx.fillText(c.name, image.width - 16 * scale, 14 * scale);
	ctx.textAlign = 'left';
	const field = panes[0].field;
	const title = panes.length > 1 ? 'Cp' : FIELDS[field].title;
	if (FIELDS[field].map) paintLegend(ctx, 16 * scale, image.height - 80 * scale, scale, title, panes[0].map);
	image.toBlob((blob) => download(blob, `${baseName(c)}-${panes.length > 1 ? 'compare' : field}.png`), 'image/png');
}
