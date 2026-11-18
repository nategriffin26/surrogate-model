// The distribution of the field shown, beside the selection in the Regions tab: the area of the car's faces in each
// band of values, as bars coloured as the car is, across the colour scale, with the area beyond either end of it
// beside the bars. Over the selection when there is one, with the whole car's outline behind. A range brushed across
// the bars tints its faces on the car and, with Select, joins the selection as a field rule, combined as the tab's
// mode says. Sums run over the working mesh's faces, as the regions' do.

import { finite, oneOf, round } from './checks.js';
import { h, s } from './dom.js';
import { FIELDS } from './fields.js';
import { niceTicks, signed, percent } from './format.js';
import { defineOperation } from './operations.js';
import { scale } from './plots.js';
import { bandCount } from './legend.js';
import { rampPosition, sample } from './ramps.js';
import { faceValues } from './regions.js';

const HISTOGRAM_FIELDS = ['cp', 'drag', 'sigma'];
const BINS = 48;
const MOST_BINS = 400;

/**
 * Area per bin of `values` over faces of `areas` a mask picks (every face without one), across [from, to] in `bins`
 * equal bins, with the area below and above that span, and the area-weighted 5th, 25th, 50th, 75th and 95th
 * percentiles.
 */
export function histogramOf(values, areas, mask, from, to, bins) {
	const area = new Float64Array(bins);
	let below = 0, above = 0, total = 0;
	const picked = [];
	const width = (to - from) / bins;
	for (let c = 0; c < areas.length; c++) {
		if (mask && !mask[c]) continue;
		const v = values[c], a = areas[c];
		total += a;
		picked.push(c);
		if (v < from) below += a;
		else if (v > to) above += a;
		else area[Math.min(bins - 1, Math.floor((v - from) / width))] += a;
	}
	picked.sort((p, q) => values[p] - values[q]);
	const percentiles = {};
	let running = 0, next = 0;
	const wanted = [5, 25, 50, 75, 95];
	for (const c of picked) {
		running += areas[c];
		while (next < wanted.length && running >= (wanted[next] / 100) * total) percentiles[`p${wanted[next++]}`] = values[c];
	}
	return { area, below, above, total, width, percentiles };
}

/** app: { session, regions }. The histogram as an operation. */
export function defineHistogramOperations({ session, regions }) {
	const view = session.view;

	defineOperation({
		name: 'field_histogram',
		description: `Area-weighted histogram of a field over the faces of a case's working mesh: the whole car, a saved region (region_id) or steps of rules (terms, as select_region takes). Bins span from..to (default the field's colour scale); the area below and above that span is given apart, with the area-weighted percentiles. To make a range of values a region, use select_region with a field rule.`,
		parameters: {
			type: 'object',
			properties: {
				case_id: { type: 'integer', description: 'Default the case on screen.' },
				field: { type: 'string', enum: HISTOGRAM_FIELDS, description: 'cp, drag (−Cp·nₓ) or sigma; default the field shown.' },
				bins: { type: 'integer', description: `How many bins; default ${BINS}, at most ${MOST_BINS}.` },
				from: { type: 'number', description: 'Low end of the bins; default the colour scale\'s.' },
				to: { type: 'number', description: 'High end of the bins; default the colour scale\'s.' },
				region_id: { type: 'integer' },
				terms: { type: 'array', description: 'Steps of rules, as select_region takes.' },
			},
			additionalProperties: false,
		},
		run({ case_id, field = view.field, bins = BINS, from, to, region_id, terms }) {
			const c = case_id === undefined ? session.active : session.byId(case_id);
			if (!c?.result) throw new Error('No case with a result is on screen. Open a file or an example first.');
			oneOf('field', field, HISTOGRAM_FIELDS);
			if (!Number.isInteger(bins) || bins < 1 || bins > MOST_BINS) throw new Error(`bins must be a whole number from 1 to ${MOST_BINS}; got ${bins}.`);
			const map = view.maps[FIELDS[field].map];
			const lo = from === undefined ? map.lo : finite('from', from);
			const hi = to === undefined ? map.hi : finite('to', to);
			if (!(lo < hi)) throw new Error(`from must be below to; got ${lo} and ${hi}.`);
			const steps = regions.namedSteps(region_id, terms);
			const cells = regions.cellsOf(c.result);
			const hist = histogramOf(faceValues(cells, field), cells.areas, steps ? regions.maskOf(c, steps) : null, lo, hi, bins);
			return {
				case_id: c.id,
				field,
				over: steps ? 'the faces the steps pick' : 'the whole car',
				from: lo,
				to: hi,
				bin_width: round(hist.width, 6),
				area_m2: [...hist.area].map((a) => round(a, 5)),
				below_m2: round(hist.below, 5),
				above_m2: round(hist.above, 5),
				total_m2: round(hist.total, 5),
				percentiles: Object.fromEntries(Object.entries(hist.percentiles).map(([k, v]) => [k, round(v, 4)])),
				summed_over: `${c.result.mesh.indices.length / 3} faces of the working mesh`,
			};
		},
	});
}

/**
 * The histogram in the Regions tab: bars of the field shown, a range brushed across them, and Select, which combines
 * the range with the selection as the tab's mode says. regions: the selection and the faces; combine: the Regions
 * panel's apply(rule) and combined(rule).
 */
export function createHistogramPanel({ session, regions, active, combine }) {
	const view = session.view;
	const svg = s('svg', { class: 'histogram-plot' });
	/** The range brushed across the bars, in the field's units, for the field it was brushed on; or null. */
	let range = null;
	let frame = null;
	const rangeRule = () => ({ kind: 'field', field: range.field, min: round(range.min, 4), max: round(range.max, 4) });

	const readout = h('span', { class: 'histogram-readout num' });
	const select = h('button', { type: 'button', class: 'tool', onclick: () => {
		const rule = rangeRule();
		clearRange();
		combine.apply(rule);
	} }, 'Select');
	const clear = h('button', { type: 'button', class: 'tool', onclick: () => clearRange() }, 'Clear range');
	// Over the bars, and only while a range is brushed: the actions belong to the range, not to the tab.
	const actions = h('div', { class: 'histogram-actions', hidden: true }, readout, select, clear);
	const el = h('div', { class: 'histogram' }, svg, actions);

	function clearRange() {
		range = null;
		regions.preview(null);
		render();
	}

	/** The faces the brushed range would leave selected, tinted on the car until it is selected or cleared. */
	function previewRange() {
		regions.preview(range ? combine.combined(rangeRule()) : null);
	}

	function render() {
		const c = active();
		const field = view.field;
		if (range && range.field !== field) clearRange();
		const map = view.maps?.[FIELDS[field]?.map];
		if (!c?.result || !map || !HISTOGRAM_FIELDS.includes(field)) {
			svg.replaceChildren();
			actions.hidden = true;
			return;
		}
		const cells = regions.cellsOf(c.result);
		const values = faceValues(cells, field);
		const whole = histogramOf(values, cells.areas, null, map.lo, map.hi, BINS);
		const selection = regions.selectionOn(c);
		const shown = selection ? histogramOf(values, cells.areas, selection.mask, map.lo, map.hi, BINS) : whole;
		draw(field, map, whole, selection ? shown : null);
		const inRange = range ? histogramOf(values, cells.areas, selection?.mask ?? null, range.min, range.max, 1).area[0] : 0;
		readout.textContent = range ? `${FIELDS[field].symbol} ${signed(range.min, 2)} to ${signed(range.max, 2)} · ${inRange.toFixed(2)} m² · ${percent(inRange / shown.total)}` : '';
		actions.hidden = !range;
	}

	/** Bars into the SVG at its box's size: the whole car's, or the selection's over the whole car's outline. */
	function draw(field, map, whole, part) {
		const width = el.clientWidth, height = el.clientHeight;
		if (width < 80 || height < 60) return;
		const m = { l: 52, r: 16, t: 22, b: 30 };
		svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
		svg.setAttribute('width', width);
		svg.setAttribute('height', height);
		const shown = part ?? whole;
		const top = Math.max(...whole.area) || 1;
		const sx = scale([map.lo, map.hi], [m.l, width - m.r]);
		const sy = scale([0, top], [height - m.b, m.t]);
		const bands = bandCount(map);
		const axis = s('g', { class: 'axis' });
		const yt = niceTicks(0, top, Math.max(2, Math.floor((height - m.t - m.b) / 28)));
		for (const t of yt.ticks) {
			axis.append(s('line', { x1: m.l, x2: width - m.r, y1: sy(t), y2: sy(t), class: t === 0 ? 'zero' : 'grid' }));
			axis.append(s('text', { x: m.l - 8, y: sy(t) + 3.5, 'text-anchor': 'end' }, t.toFixed(yt.digits)));
		}
		const xt = niceTicks(map.lo, map.hi, Math.max(3, Math.floor((width - m.l - m.r) / 90)));
		for (const t of xt.ticks) {
			axis.append(s('line', { x1: sx(t), x2: sx(t), y1: height - m.b, y2: height - m.b + 4, class: 'tick' }));
			axis.append(s('text', { x: sx(t), y: height - m.b + 16, 'text-anchor': 'middle' }, signed(t, xt.digits)));
		}
		axis.append(s('text', { x: width - m.r, y: height - 3, 'text-anchor': 'end', class: 'unit' }, FIELDS[field].symbol));
		axis.append(s('text', { x: m.l - 8, y: 10, 'text-anchor': 'end', class: 'unit' }, 'm²'));
		const bars = s('g', { class: 'bars' });
		shown.area.forEach((a, i) => {
			const v0 = map.lo + i * shown.width;
			const [r, g, b] = sample(map.ramp, rampPosition(v0 + shown.width / 2, map.lo, map.hi, bands));
			bars.append(s('rect', { x: sx(v0) + 0.5, y: sy(a), width: Math.max(0.5, sx(v0 + shown.width) - sx(v0) - 1), height: sy(0) - sy(a), fill: `rgb(${r} ${g} ${b})` }));
		});
		// The whole car behind a selection's bars, as a step outline.
		const outline = part ? s('path', { class: 'histogram-outline', d: whole.area.reduce((d, a, i) => `${d}H${sx(map.lo + i * whole.width).toFixed(1)}V${sy(a).toFixed(1)}`, `M${sx(map.lo).toFixed(1)},${sy(0).toFixed(1)}`) + `H${sx(map.hi).toFixed(1)}V${sy(0).toFixed(1)}` }) : null;
		// What lies beyond the colour scale, over its two ends.
		const beyond = s('g', { class: 'histogram-beyond' },
			s('text', { x: m.l + 4, y: 10, 'text-anchor': 'start' }, shown.below > 0 ? `${shown.below.toFixed(2)} m² below` : ''),
			s('text', { x: width - m.r, y: 10, 'text-anchor': 'end' }, shown.above > 0 ? `${shown.above.toFixed(2)} m² above` : ''));
		const brush = range ? s('rect', { class: 'histogram-range', x: sx(Math.max(map.lo, range.min)), y: m.t, width: Math.max(1, sx(Math.min(map.hi, range.max)) - sx(Math.max(map.lo, range.min))), height: height - m.t - m.b }) : null;
		svg.replaceChildren(...[axis, outline, bars, brush, beyond].filter(Boolean));
		frame = { sx, m, width, map, field };
	}

	// Brushing: press on the bars and drag across the values to take.
	svg.addEventListener('pointerdown', (e) => {
		if (!frame || e.button !== 0) return;
		const rect = svg.getBoundingClientRect();
		const at = (x) => Math.min(frame.map.hi, Math.max(frame.map.lo, frame.sx.invert(x - rect.left)));
		const start = at(e.clientX);
		svg.setPointerCapture(e.pointerId);
		const move = (ev) => {
			const end = at(ev.clientX);
			range = { field: frame.field, min: Math.min(start, end), max: Math.max(start, end) };
			previewRange();
			render();
		};
		const up = () => {
			svg.removeEventListener('pointermove', move);
			svg.removeEventListener('pointerup', up);
			if (range && range.max - range.min < 1e-9) clearRange();
		};
		svg.addEventListener('pointermove', move);
		svg.addEventListener('pointerup', up);
	});

	/** Leaving the tab takes the range's tint off the car. */
	function stop() {
		if (range) clearRange();
	}

	return { el, render, stop };
}
