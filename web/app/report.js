// The report: the analysis on screen as one HTML page, for someone without the tool. It holds the view as drawn, the
// whole car, the selection and the saved regions, the section plot, the probe set in use, the comparison and the
// study, each read from what the operations give, so the page and the tool agree. The picture is inlined as a data
// URL, the plot as SVG, and the styles travel in the page: it loads nothing.

import { caseSummary } from './cases.js';
import { EXPORTS, baseName, defineExport } from './export.js';
import { percent, signed } from './format.js';
import { readSet } from './probes.js';

const PLOT = { width: 900, height: 300 };

const escape = (text) => String(text).replace(/[&<>"]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[ch]);
const number = (v, digits = 3) => (v === null || v === undefined ? '' : signed(v, digits));
/** A table; with a null head, rows of a label and its value. */
const table = (head, rows) => `<table>${head ? `<thead><tr>${head.map((text) => `<th>${escape(text)}</th>`).join('')}</tr></thead>` : ''}<tbody>${rows.map((row) => `<tr>${row.map((cell) => `<td>${escape(cell)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
const note = (text) => `<p class="basis">${escape(text)}</p>`;

const STYLE = `
body { margin: 0; background: #fff; color: var(--ink); font: 400 14px/1.45 var(--body); }
main { max-width: 980px; margin: 0 auto; padding: 40px 32px 64px; }
h1 { margin: 0; font: 600 26px/1.2 var(--head); color: var(--heading); }
h2 { margin: 36px 0 10px; font: 600 17px/1.3 var(--head); color: var(--heading); }
.meta { margin: 6px 0 20px; color: var(--muted); font-size: 13px; }
.basis { margin: 8px 0 0; color: var(--muted); font-size: 12.5px; }
img { display: block; width: 100%; border: 1px solid var(--hair); border-radius: 6px; }
table { border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
th { text-align: right; font-weight: 500; color: var(--muted); padding: 6px 10px; border-bottom: 1px solid var(--rule); white-space: nowrap; }
td { text-align: right; padding: 5px 10px; border-bottom: 1px solid var(--hair); color: var(--heading); white-space: nowrap; }
th:first-child, td:first-child { text-align: left; padding-left: 0; }
.key { display: flex; flex-wrap: wrap; gap: 6px 18px; margin-top: 6px; color: var(--muted); font-size: 12.5px; }
.key i { display: inline-block; width: 18px; margin-right: 6px; vertical-align: middle; }
`;

/**
 * The theme's colours and fonts, and the plot's styles, as the page has them: one source for the tool and the report.
 * Only the app's own sheets are read; the font service's, from another origin, cannot be.
 */
function carriedStyles() {
	const own = [...document.styleSheets].filter((sheet) => !sheet.href || new URL(sheet.href).origin === location.origin);
	const rules = own.flatMap((sheet) => [...sheet.cssRules]);
	return rules.filter((rule) => rule.selectorText === ':root' || rule.selectorText?.startsWith('svg ')).map((rule) => rule.cssText).join('\n');
}

const dataUrl = (blob) => new Promise((resolve, reject) => {
	const reader = new FileReader();
	reader.onload = () => resolve(reader.result);
	reader.onerror = () => reject(reader.error);
	reader.readAsDataURL(blob);
});

/** app: { session, engine, regions, section, comparison, variants }. Defines the report export. */
export function defineReport({ session, engine, regions, section, comparison, variants }) {
	const view = session.view;

	function wholeCar(c) {
		const s = caseSummary(c, null);
		const point = (p) => `(${p.map((v) => signed(v, 2)).join(', ')}) m`;
		const rows = [
			['Cx', number(s.cx)], ['Cy', number(s.cy)], ['Cz', number(s.cz)], ['Mean Cp', number(s.mean_cp)],
			['Area', `${s.area_m2.toFixed(2)} m²`], ['Frontal area', `${s.reference_area_m2.toFixed(2)} m²`],
			['Length × width × height', `${[s.length_m, s.width_m, s.height_m].map((v) => v.toFixed(2)).join(' × ')} m`],
			['Cx by network', `${number(Math.min(...s.cx_by_network))} to ${number(Math.max(...s.cx_by_network))}`],
			['Cp max', `${number(s.cp_max.value, 2)} at ${point(s.cp_max.center_m)}`],
			['Cp min', `${number(s.cp_min.value, 2)} at ${point(s.cp_min.center_m)}`],
			...(s.reference ? [['Cx of the CFD', number(s.reference.cx)], ['R² against the CFD', number(s.reference.r2)]] : []),
		];
		const outside = s.outside_training.map((d) => `${d.dimension} ${d.value_m.toFixed(3)} m (training ${d.training_min_m.toFixed(3)} to ${d.training_max_m.toFixed(3)} m)`);
		return `<h2>Whole car</h2>${table(null, rows)}${note(`Summed over ${s.totals_over}.`)}${outside.length ? note(`Outside the training range: ${outside.join('; ')}.`) : ''}`;
	}

	function regionsPart(c) {
		const picked = [
			...(view.selection ? [{ name: 'Selection', terms: view.selection.terms }] : []),
			...session.regions.map((r) => ({ name: r.name, terms: r.terms })),
		];
		if (!picked.length) return '';
		const read = picked.map((r) => ({ name: r.name, ...regions.readoutOf(c, r.terms) }));
		const rows = read.map(({ name, readout: m }) => [name, m.area_m2.toFixed(3), number(m.cx), number(m.cz), number(m.mean_cp), percent(m.share_of_car.cx)]);
		return `<h2>Regions</h2>${table(['Region', 'Area, m²', 'Cx', 'Cz', 'Mean Cp', 'Share of Cx'], rows)}${note(`Summed over ${read[0].summed_over}.`)}`;
	}

	function sectionPart() {
		if (!view.section.on) return '';
		const { svg, key, plane } = section.picture(PLOT.width, PLOT.height);
		const items = key.map(({ label, colour, dashed }) => `<span><i style="border-top: 2px ${dashed ? 'dashed' : 'solid'} ${escape(colour)}"></i>${escape(label)}</span>`);
		return `<h2>Section ${escape(plane)}</h2>${new XMLSerializer().serializeToString(svg)}<div class="key">${items.join('')}</div>`;
	}

	async function probesPart(c) {
		const set = session.probeSets.find((x) => x.id === view.probeSet);
		if (!set?.probes.length) return '';
		const sites = await readSet(engine, c, set);
		const labels = set.probes.some((p) => p.label);
		const rows = set.probes.map((p, i) => [p.n, ...(labels ? [p.label ?? ''] : []), ...sites[i].point.map((v) => signed(v, 3)), number(sites[i].cp), number(sites[i].sigma)]);
		return `<h2>Probes: ${escape(set.name)}</h2>${table(['Probe', ...(labels ? ['Label'] : []), 'x, m', 'y, m', 'z, m', 'Cp', 'σ'], rows)}`;
	}

	function comparisonPart(c) {
		const out = comparison.read(c);
		if (!out) return '';
		const other = out.compared_with === 'cfd' ? 'CFD' : out.compared_with.name;
		const mine = out.compared_with === 'cfd' ? 'Predicted' : c.name;
		const columns = [['cx', 'Cx'], ['cz', 'Cz'], ['mean_cp', 'Mean Cp']];
		const head = ['Region', ...columns.flatMap(([, label]) => [`${label}, ${mine}`, `${label}, ${other}`, `Δ${label}`]), 'RMS ΔCp'];
		const rows = out.rows.map((row) => [row.name, ...columns.flatMap(([key]) => [number(row.this_car[key]), number(row.other[key]), number(row.difference[key])]), number(row.rms_dcp)]);
		return `<h2>Compared with ${escape(other)}</h2>${table(head, rows)}${out.like_for_like ? note(out.like_for_like) : ''}${note(`RMS ΔCp is taken over ${out.rms_dcp_over}.`)}`;
	}

	function studyPart(c) {
		const study = variants.study(c);
		if (!study) return '';
		const unchanged = study.car.predicted_unchanged;
		const rows = [
			[`${study.car.name}, unchanged`, number(unchanged.cx), '', number(unchanged.cz), '', number(unchanged.mean_cp), '', unchanged.frontal_area_m2.toFixed(3)],
			...study.variants.map((v) => [v.name, number(v.cx), number(v.difference.cx), number(v.cz), number(v.difference.cz), number(v.mean_cp), number(v.difference.mean_cp), v.frontal_area_m2.toFixed(3)]),
		];
		return `<h2>Study</h2>${table(['Variant', 'Cx', 'ΔCx', 'Cz', 'ΔCz', 'Mean Cp', 'ΔMean Cp', 'Frontal area, m²'], rows)}${note(study.predicted_on)}`;
	}

	async function reportOf(c) {
		const picture = await dataUrl(await EXPORTS.get('picture').make(c));
		const r = c.result;
		const made = new Date().toLocaleString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' });
		const meta = [
			c.variant ? `Variant of ${c.variant.root.name}` : 'Predicted surface pressure',
			made,
			`${r.source.cells.toLocaleString('en-US')} polygons`,
			`model ${r.model.slice(0, 8)}`,
		].join(' · ');
		const parts = [wholeCar(c), regionsPart(c), sectionPart(), await probesPart(c), comparisonPart(c), studyPart(c)];
		return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${escape(c.name)}, report</title>
<style>${carriedStyles()}${STYLE}</style>
</head>
<body>
<main>
<h1>${escape(c.name)}</h1>
<p class="meta">${escape(meta)}</p>
<img alt="${escape(`${c.name} as drawn`)}" src="${picture}">
${parts.join('\n')}
</main>
</body>
</html>
`;
	}

	defineExport({
		name: 'report',
		label: 'Report',
		format: 'HTML',
		title: 'The analysis on screen as one page: the view, the whole car, regions, the section, probes, the comparison and the study',
		unavailable: () => null,
		make: async (c) => new Blob([await reportOf(c)], { type: 'text/html' }),
		fileName: (c) => `${baseName(c)}-report.html`,
	});
}
