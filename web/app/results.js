// The numbers for one case: pressure drag and each network's value, agreement with CFD, the car's size against the
// ten training cars, and the facts of the run.

import { h, s } from './dom.js';
import { bytes, count, counts, seconds, signed } from './format.js';
import { STAGES } from './session.js';

const STAGE_LABELS = { reading: 'Reading', measuring: 'Measuring', meshing: 'Meshing', predicting: 'Predicting' };
const DIMENSION_LABELS = { length: 'Length', width: 'Width', height: 'Height' };

function row(label, value, attrs = {}) {
	return h('div', { class: 'row', ...attrs }, h('span', { class: 'row-label' }, label), h('span', { class: 'row-value num' }, value));
}

function section(title, ...body) {
	return h('section', { class: 'block' }, h('h3', {}, title), ...body);
}

/** Each network's pressure drag as a tick on one short scale, their mean marked, and CFD when the file has it. */
function networkStrip({ cdP, cdPModels, cdPCfd }) {
	const values = [...cdPModels, cdP, ...(cdPCfd === null ? [] : [cdPCfd])];
	const lo = Math.min(...values), hi = Math.max(...values);
	const pad = (hi - lo) * 0.12 || 0.005;
	const W = 300, H = 42;
	const x = (v) => 10 + ((v - (lo - pad)) / (hi - lo + 2 * pad)) * (W - 20);
	return s('svg', { class: 'networks', viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Each network’s pressure drag' },
		s('line', { x1: 10, x2: W - 10, y1: 18, y2: 18, class: 'track' }),
		cdPModels.map((v, i) => s('line', { x1: x(v), x2: x(v), y1: 11, y2: 25, class: 'network' }, s('title', {}, `Network ${i + 1}: ${v.toFixed(4)}`))),
		s('line', { x1: x(cdP), x2: x(cdP), y1: 5, y2: 31, class: 'mean' }, s('title', {}, `Mean ${cdP.toFixed(4)}`)),
		s('text', { x: x(cdP), y: 41, 'text-anchor': 'middle' }, 'mean'),
		cdPCfd === null ? null : [
			s('path', { d: `M${x(cdPCfd)} 12l6 6-6 6-6-6z`, class: 'cfd-mark' }, s('title', {}, `CFD ${cdPCfd.toFixed(4)}`)),
			s('text', { x: x(cdPCfd), y: 7, 'text-anchor': 'middle' }, 'CFD'),
		]);
}

/** A dimension on the training cars' range: the range as a band, this car as a tick, outside values marked. */
function rangeRow({ name, value, trainMin, trainMax }) {
	const outside = value < trainMin || value > trainMax;
	const lo = Math.min(trainMin, value), hi = Math.max(trainMax, value);
	const pad = (hi - lo) * 0.15;
	const W = 120;
	const x = (v) => 2 + ((v - (lo - pad)) / (hi - lo + 2 * pad)) * (W - 4);
	return h('div', { class: `row range${outside ? ' outside' : ''}`, 'data-dimension': name },
		h('span', { class: 'row-label' }, DIMENSION_LABELS[name]),
		s('svg', { class: 'range-bar', viewBox: `0 0 ${W} 14`, 'aria-hidden': 'true' },
			s('rect', { x: x(trainMin), y: 4, width: Math.max(1, x(trainMax) - x(trainMin)), height: 6, rx: 1.5, class: 'train' }),
			s('line', { x1: x(value), x2: x(value), y1: 0, y2: 14, class: 'tick' })),
		h('span', { class: 'row-value num' }, `${value.toFixed(3)} m`),
		h('span', { class: 'row-note num' }, `training ${trainMin.toFixed(2)} to ${trainMax.toFixed(2)} m`));
}

export function renderResults(c, { device, drawn }) {
	const r = c.result;
	const blocks = [];
	if (r) {
		const f = r.forces;
		blocks.push(section('Pressure drag',
			h('div', { class: 'headline' },
				h('span', { class: 'symbol' }, 'C', h('sub', {}, 'D,p')),
				h('span', { class: 'big num' }, f.cdP.toFixed(3))),
			networkStrip(f),
			h('div', { class: 'networks-values', style: `grid-template-columns: repeat(${f.cdPModels.length}, 1fr)` },
				f.cdPModels.map((v, i) => h('span', { class: 'num', title: `Network ${i + 1}` }, h('i', {}, i + 1), v.toFixed(3)))),
			row(`Spread of ${f.cdPModels.length} networks`, f.cdPStd.toFixed(3), { title: 'Standard deviation of the networks’ values' }),
			f.cdPCfd === null ? null : h('div', { class: 'cfd-line' },
				row('CFD', f.cdPCfd.toFixed(3)),
				row('Predicted − CFD', `${signed(counts(f.cdP, f.cdPCfd), 0)} counts`))));
		if (r.accuracy) {
			const a = r.accuracy;
			blocks.push(section('Agreement with CFD',
				h('div', { class: 'metrics' },
					[['R²', a.r2.toFixed(3)], ['MAE', a.mae.toFixed(4)], ['RMSE', a.rmse.toFixed(4)], ['Bias', signed(a.bias, 4)]]
						.map(([k, v]) => h('div', { class: 'metric', title: 'Cp, area-weighted over the surface' }, h('span', { class: 'metric-label' }, k), h('span', { class: 'metric-value num' }, v))))));
		}
		const g = r.geometry;
		const cov = r.coverage;
		blocks.push(section('Geometry',
			cov.dimensions.map(rangeRow),
			row('Reference area', `${g.frontalArea.toFixed(3)} m²`, { title: 'Frontal area, the reference area for C_D' }),
			row('Surface area', `${g.surfaceArea.toFixed(2)} m²`),
			row('Section area outside training', `${(cov.sectionOutsideFraction * 100).toFixed(1)} % of length`)));
		blocks.push(section('Model spread σ(Cp)',
			row('Mean', cov.spread.mean.toFixed(3)),
			row('95th percentile', cov.spread.p95.toFixed(3)),
			row('Mean on the training cars', cov.spread.trainMean.toFixed(3), { title: 'In cross-validation' })));
	}
	blocks.push(section('Run', runFacts(c, { device, drawn })));
	return h('div', { class: 'results' }, blocks);
}

/** What the viewer last drew, as the run facts show it. */
export const drawnText = (drawn) => `${count(drawn.triangles)} · ${drawn.full ? 'full mesh' : 'working mesh'}`;

function runFacts(c, { device, drawn }) {
	const r = c.result;
	const rows = [row('File', r?.source.name ?? c.name)];
	if (c.bytes !== null) rows.push(row('Size', bytes(c.bytes)));
	else if (r?.source.bytes) rows.push(row('Size', bytes(r.source.bytes)));
	if (r) {
		rows.push(row('Polygons', count(r.source.cells)), row('Points', count(r.source.points)));
		rows.push(row('Triangles drawn', drawn ? drawnText(drawn) : '—', { 'data-fact': 'drawn' }));
		rows.push(row('Working mesh', `${count(r.mesh.indices.length / 3)} triangles`));
		if (c.full) rows.push(row('Full mesh', `${count(r.mesh.sourceTriangles)} triangles${c.full.fraction < 1 ? ` · ${Math.round(c.full.fraction * 100)} % predicted` : ''}`));
		rows.push(row('Networks ran on', r.device));
		for (const t of r.timings) rows.push(row(STAGE_LABELS[t.stage], seconds(t.seconds)));
		rows.push(row('Total', seconds(r.timings.reduce((sum, t) => sum + t.seconds, 0))));
	} else {
		if (device) rows.push(row('Networks run on', device));
		for (const stage of STAGES) {
			const st = c.stages[stage];
			if (st) rows.push(row(STAGE_LABELS[stage], st.end ? seconds((st.end - st.start) / 1000) : `${Math.round(st.fraction * 100)} %`));
		}
	}
	return rows;
}
