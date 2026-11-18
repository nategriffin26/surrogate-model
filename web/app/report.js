// The report: the analysis on screen as one HTML page, for someone without the tool. It holds the view as drawn, the
// whole car, the section plot, the probe set in use, the comparison and the study, each read from what the
// operations give, so the page and the tool agree. The picture is inlined as a data URL, the plot as SVG, and the
// styles travel in the page: it loads nothing.

import { caseSummary } from './cases.js';
import { EXPORTS, baseName, defineExport } from './export.js';
import { signed } from './format.js';
import { readSet } from './probes.js';

const PLOT = { width: 900, height: 300 };

const escapeHtml = (text) =>
	String(text).replace(/[&<>"]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[ch]);
const number = (v, digits = 3) => (v === null || v === undefined ? '' : signed(v, digits));
/** Markup set into a table as it is, for a coefficient's name; every other cell is text. */
const markup = (html) => ({ html });
/** A force coefficient's name as the screen writes it, C with its axis subscripted. */
const coefficient = (axis) => `C<sub>${axis}</sub>`;
const cell = (content) => content.html ?? escapeHtml(content);
/** A table; with a null head, rows of a label and its value. */
const table = (head, rows) =>
	`<table>${head ? `<thead><tr>${head.map((content) => `<th>${cell(content)}</th>`).join('')}</tr></thead>` : ''}<tbody>${rows.map((row) => `<tr>${row.map((content) => `<td>${cell(content)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
const note = (text) => `<p class="basis">${escapeHtml(text)}</p>`;

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

/** The app's own stylesheets; the report reads nothing it could not reach offline. */
const ownSheets = () =>
	[...document.styleSheets].filter((sheet) => !sheet.href || new URL(sheet.href).origin === location.origin);

/** The theme's colours and fonts, and the plot's styles, as the page has them: one source for the tool and the report. */
function carriedStyles() {
	const rules = ownSheets().flatMap((sheet) => [...sheet.cssRules]);
	return rules
		.filter((rule) => rule.selectorText === ':root' || rule.selectorText?.startsWith('svg '))
		.map((rule) => rule.cssText)
		.join('\n');
}

const dataUrl = (blob) =>
	new Promise((resolve, reject) => {
		const reader = new FileReader();
		reader.onload = () => resolve(reader.result);
		reader.onerror = () => reject(reader.error);
		reader.readAsDataURL(blob);
	});

/** The theme's @font-face rules with each vendored file inlined as a data URL, so the report keeps its faces offline. */
async function carriedFaces() {
	const faces = ownSheets().flatMap((sheet) =>
		[...sheet.cssRules].filter((rule) => rule instanceof CSSFontFaceRule).map((rule) => ({ rule, base: sheet.href })),
	);
	const inlined = await Promise.all(
		faces.map(async ({ rule, base }) => {
			const [, file] = rule.style.getPropertyValue('src').match(/url\("?([^")]+)"?\)/);
			const data = await dataUrl(await (await fetch(new URL(file, base))).blob());
			return rule.cssText.replace(file, data);
		}),
	);
	return inlined.join('\n');
}

/** context: { session, engine, section, comparison, variants }. Defines the report export. */
export function defineReport({ session, engine, section, comparison, variants }) {
	const view = session.view;

	function wholeCar(c) {
		const s = caseSummary(c, null);
		const point = (p) => `(${p.map((v) => signed(v, 2)).join(', ')}) m`;
		const rows = [
			[markup(coefficient('x')), number(s.cx)],
			[markup(coefficient('y')), number(s.cy)],
			[markup(coefficient('z')), number(s.cz)],
			['Mean Cp', number(s.mean_cp)],
			['Area', `${s.area_m2.toFixed(2)} m²`],
			['Frontal area', `${s.reference_area_m2.toFixed(2)} m²`],
			['Length × width × height', `${[s.length_m, s.width_m, s.height_m].map((v) => v.toFixed(2)).join(' × ')} m`],
			[
				markup(`${coefficient('x')} by network`),
				`${number(Math.min(...s.cx_by_network))} to ${number(Math.max(...s.cx_by_network))}`,
			],
			['Cp max', `${number(s.cp_max.value, 2)} at ${point(s.cp_max.center_m)}`],
			['Cp min', `${number(s.cp_min.value, 2)} at ${point(s.cp_min.center_m)}`],
			...(s.reference
				? [
						[markup(`${coefficient('x')} of the CFD`), number(s.reference.cx)],
						['R² against the CFD', number(s.reference.r2)],
					]
				: []),
		];
		const outside = s.outside_training.map(
			(d) =>
				`${d.dimension} ${d.value_m.toFixed(3)} m (training ${d.training_min_m.toFixed(3)} to ${d.training_max_m.toFixed(3)} m)`,
		);
		return `<h2>Whole car</h2>${table(null, rows)}${note(`Summed over ${s.totals_over}.`)}${outside.length ? note(`Outside the training range: ${outside.join('; ')}.`) : ''}`;
	}

	function sectionPart() {
		if (!view.section.on) return '';
		const { svg, key, plane } = section.picture(PLOT.width, PLOT.height);
		const items = key.map(
			({ label, colour, width, opacity, dashed }) =>
				`<span><i style="border-top: ${width}px ${dashed ? 'dashed' : 'solid'} ${escapeHtml(colour)}; opacity: ${opacity}"></i>${escapeHtml(label)}</span>`,
		);
		return `<h2>Section ${escapeHtml(plane)}</h2>${new XMLSerializer().serializeToString(svg)}<div class="key">${items.join('')}</div>`;
	}

	async function probesPart(c) {
		const set = session.probeSets.find((x) => x.id === view.probeSet);
		if (!set?.probes.length) return '';
		const sites = await readSet(engine, c, set);
		const rows = set.probes.map((p, i) => [
			p.n,
			...sites[i].point.map((v) => signed(v, 3)),
			number(sites[i].cp),
			number(sites[i].sigma),
		]);
		return `<h2>Probes: ${escapeHtml(set.name)}</h2>${table(['Probe', 'x, m', 'y, m', 'z, m', 'Cp', 'σ'], rows)}`;
	}

	function comparisonPart(c) {
		const out = comparison.read(c);
		if (!out) return '';
		const other = out.compared_with === 'cfd' ? 'CFD' : out.compared_with.name;
		const mine = out.compared_with === 'cfd' ? 'Predicted' : c.name;
		const columns = [
			['cx', coefficient('x')],
			['cz', coefficient('z')],
			['mean_cp', 'Mean Cp'],
		];
		const head = [
			'',
			...columns.flatMap(([, name]) => [
				markup(`${name}, ${escapeHtml(mine)}`),
				markup(`${name}, ${escapeHtml(other)}`),
				markup(`Δ${name}`),
			]),
			'RMS ΔCp',
		];
		const whole = out.whole_car;
		const rows = [
			[
				'Whole car',
				...columns.flatMap(([key]) => [
					number(whole.this_car[key]),
					number(whole.other[key]),
					number(whole.difference[key]),
				]),
				number(whole.rms_dcp),
			],
		];
		return `<h2>Compared with ${escapeHtml(other)}</h2>${table(head, rows)}${out.like_for_like ? note(out.like_for_like) : ''}${note(`RMS ΔCp is taken over ${out.rms_dcp_over}.`)}`;
	}

	function studyPart(c) {
		const study = variants.study(c);
		if (!study) return '';
		const unchanged = study.car.predicted_unchanged;
		const rows = [
			[
				`${study.car.name}, unchanged`,
				number(unchanged.cx),
				'',
				number(unchanged.cz),
				'',
				number(unchanged.mean_cp),
				'',
				unchanged.frontal_area_m2.toFixed(3),
			],
			...study.variants.map((v) => [
				v.name,
				number(v.cx),
				number(v.difference.cx),
				number(v.cz),
				number(v.difference.cz),
				number(v.mean_cp),
				number(v.difference.mean_cp),
				v.frontal_area_m2.toFixed(3),
			]),
		];
		return `<h2>Study</h2>${table(['Variant', markup(coefficient('x')), markup(`Δ${coefficient('x')}`), markup(coefficient('z')), markup(`Δ${coefficient('z')}`), 'Mean Cp', 'ΔMean Cp', 'Frontal area, m²'], rows)}${note(study.predicted_on)}`;
	}

	async function reportOf(c) {
		const picture = await dataUrl(await EXPORTS.get('picture').make(c));
		const r = c.result;
		const made = new Date().toLocaleString('en-GB', {
			day: 'numeric',
			month: 'short',
			year: 'numeric',
			hour: '2-digit',
			minute: '2-digit',
		});
		const meta = [
			c.variant ? `Variant of ${c.variant.root.name}` : 'Predicted surface pressure',
			made,
			`${r.source.cells.toLocaleString('en-US')} cells`,
			`model ${r.model.slice(0, 8)}`,
		].join(' · ');
		const parts = [wholeCar(c), sectionPart(), await probesPart(c), comparisonPart(c), studyPart(c)];
		return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${escapeHtml(c.name)}, report</title>
<style>${await carriedFaces()}${carriedStyles()}${STYLE}</style>
</head>
<body>
<main>
<h1>${escapeHtml(c.name)}</h1>
<p class="meta">${escapeHtml(meta)}</p>
<img alt="${escapeHtml(`${c.name} as drawn`)}" src="${picture}">
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
		title:
			'The analysis on screen as one page: the view, the whole car, the section, probes, the comparison and the study',
		unavailable: () => null,
		make: async (c) => new Blob([await reportOf(c)], { type: 'text/html' }),
		fileName: (c) => `${baseName(c)}-report.html`,
	});
}
