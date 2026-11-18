// The result card over the view: the readout of what is selected, or of the whole car when nothing is. Its area, its
// area-weighted mean Cp and its pressure force coefficients (pressure only: the model predicts no wall shear), with a
// selection's share of the car's; where Cp is lowest and highest, on a side view of the car and flown to on request;
// then, for the whole car, its reference area and size, the dimensions outside the training range, the spread of Cx
// across the networks and one line about the run.
//
// The whole car's numbers are the Result's totals, summed over every polygon of the file. A selection's are summed
// over the working mesh's faces it picks (regions.js), and its shares over the same faces, so their bias cancels.

import { h, s } from './dom.js';
import { millions, percent, seconds, signed } from './format.js';

const DIMENSION_LABELS = { length: 'Length', width: 'Width', height: 'Height' };
/** The card's content width, which the drawings fill: 264 px less its padding and border. */
const WIDTH = 234;
const PROFILE_BINS = 96;

/** The car's side outline (x against z) as an SVG path inside a 4 px margin of the card's width, and the mapping into it. */
function profileOf(mesh, bounds) {
	const [x0, x1, , , z0, z1] = bounds;
	const scale = (WIDTH - 8) / (x1 - x0);
	const top = new Float32Array(PROFILE_BINS).fill(-Infinity);
	const bottom = new Float32Array(PROFILE_BINS).fill(Infinity);
	const P = mesh.positions;
	const perMetre = PROFILE_BINS / (x1 - x0);
	for (let i = 0; i < P.length; i += 3) {
		const bin = Math.min(PROFILE_BINS - 1, ((P[i] - x0) * perMetre) | 0);
		const z = P[i + 2];
		if (z > top[bin]) top[bin] = z;
		if (z < bottom[bin]) bottom[bin] = z;
	}
	const height = (z1 - z0) * scale;
	const map = ([x, , z]) => [(x - x0) * scale, height - (z - z0) * scale];
	const bins = [...top.keys()].filter((b) => Number.isFinite(top[b]));
	const along = (b) => x0 + ((b + 0.5) / PROFILE_BINS) * (x1 - x0);
	const points = [...bins.map((b) => map([along(b), 0, top[b]])), ...bins.reverse().map((b) => map([along(b), 0, bottom[b]]))];
	return { path: `M${points.map(([x, y]) => `${x.toFixed(1)} ${y.toFixed(1)}`).join('L')}Z`, height, map };
}

/**
 * What the card reads out for a case: the selection on it, with the whole car summed over the same faces for shares,
 * or the whole car from its totals. Extremes are {cp, point, normal}.
 */
function subjectOf(c, regions) {
	const at = (e) => e && { cp: e.value, point: e.center, normal: e.normal };
	const selection = regions.selectionOn(c);
	if (selection) {
		const { part, whole } = selection;
		return { name: 'Selection', area: part.area, meanCp: part.meanCp, coefficients: part.coefficients, whole, extremes: { max: at(part.cpMax), min: at(part.cpMin) } };
	}
	const t = c.result.totals;
	return { name: 'Whole car', area: t.area, meanCp: t.meanCp, coefficients: t.coefficients, whole: null, extremes: { max: at(t.cpMax), min: at(t.cpMin) } };
}

/**
 * host: where the card goes; overlay: over the 3D view, for the extremes' markers; regions: the selection's readout;
 * run(name, args) runs an operation.
 */
export function createResultCard({ host, overlay, viewer, active, regions, run }) {
	const outlines = new WeakMap();
	const markers = { max: h('div', { class: 'extreme-marker max', hidden: true }, h('span', {}, 'max')), min: h('div', { class: 'extreme-marker min', hidden: true }, h('span', {}, 'min')) };
	overlay.append(markers.max, markers.min);
	let shown = null;
	let subject = null;
	/** The extremes marked on the car: the one hovered in the card, and the one last flown to. */
	const marked = { hovered: null, pinned: null };
	const outline = (c) => outlines.get(c) ?? outlines.set(c, profileOf(c.result.mesh, c.result.geometry.bounds)).get(c);

	function placeMarkers() {
		for (const which of ['max', 'min']) {
			const marker = markers[which];
			const spot = subject?.extremes[which];
			const on = Boolean(spot) && (marked.hovered === which || marked.pinned === which);
			marker.hidden = !on;
			if (!on) continue;
			const p = viewer.project(spot.point);
			marker.hidden = !p.visible || viewer.clipped(spot.point);
			marker.style.transform = `translate(${p.x}px, ${p.y}px)`;
		}
	}
	viewer.addEventListener('camera', placeMarkers);
	// The pinned spot stays marked until the car is pressed again.
	viewer.canvas.addEventListener('pointerdown', () => {
		marked.pinned = null;
		placeMarkers();
	});

	/**
	 * The pressure force coefficients and the rest of the readout. A selection's share of the car's, where a share
	 * means something, stands in a third column so the values stay aligned.
	 */
	function readout(of) {
		const share = (part, whole) => h('dd', { class: 'result-share num', title: 'Share of the whole car\'s, summed over the same faces' }, whole == null || Math.abs(whole) < 1e-9 ? '' : percent(part / whole));
		const row = (label, title, text, part, whole) => [h('dt', { title }, ...label), h('dd', { class: 'num' }, text), of.whole ? share(part, whole) : null];
		const [cx, cy, cz] = of.coefficients;
		const [wx, , wz] = of.whole?.coefficients ?? [];
		return h('dl', { class: `result-forces${of.whole ? ' shares' : ''}` },
			row(['C', h('sub', {}, 'x')], 'Pressure force along x on the frontal area: the pressure drag coefficient', signed(cx, 3), cx, wx),
			row(['C', h('sub', {}, 'y')], 'Pressure force along y on the frontal area: side force', signed(cy, 3)),
			row(['C', h('sub', {}, 'z')], 'Pressure force along z on the frontal area: lift', signed(cz, 3), cz, wz),
			row(['Area'], 'Surface area', `${of.area.toFixed(2)} m²`, of.area, of.whole?.area),
			row(['Mean Cp'], 'Area-weighted mean pressure coefficient', of.meanCp === null ? '–' : signed(of.meanCp, 3)));
	}

	function extremes(c, of) {
		const ex = of.extremes;
		if (!ex.max) return null;
		const profile = outline(c);
		const fly = (which) => {
			marked.pinned = which;
			run('fly_to', { point: ex[which].point, normal: ex[which].normal });
			placeMarkers();
		};
		const hover = (el, which) => {
			el.addEventListener('pointerenter', () => {
				marked.hovered = which;
				placeMarkers();
			});
			el.addEventListener('pointerleave', () => {
				marked.hovered = null;
				placeMarkers();
			});
			return el;
		};
		const spot = (which) => {
			const [x, y] = profile.map(ex[which].point);
			return hover(s('circle', { class: `spot ${which}`, cx: x, cy: y, r: 3.5, onclick: () => fly(which) }, s('title', {}, `Cp ${which} ${signed(ex[which].cp, 2)}`)), which);
		};
		const button = (which, label) => hover(h('button', { type: 'button', class: `extreme ${which}`, title: `Fly to the ${which === 'max' ? 'highest' : 'lowest'} Cp`, onclick: () => fly(which) },
			h('i', { 'aria-hidden': 'true' }), label, h('b', { class: 'num' }, signed(ex[which].cp, 2))), which);
		return h('div', { class: 'extremes' },
			s('svg', { class: 'profile', viewBox: `-4 -4 ${WIDTH} ${profile.height + 8}`, width: WIDTH, height: profile.height + 8, role: 'img', 'aria-label': 'Side view with the highest and lowest Cp' },
				s('path', { class: 'outline', d: profile.path }), spot('max'), spot('min')),
			h('div', { class: 'extreme-buttons' }, button('max', 'Cp max'), button('min', 'Cp min')));
	}

	/** The whole car's facts: its reference area and size, what lies outside training, the networks' spread, the run. */
	function facts(r) {
		const g = r.geometry;
		const f = r.forces;
		const outside = r.coverage.dimensions.filter((d) => d.value < d.trainMin || d.value > d.trainMax);
		const took = r.timings.reduce((sum, t) => sum + t.seconds, 0);
		return [
			h('dl', { class: 'result-facts' },
				h('dt', {}, 'Frontal area'), h('dd', { class: 'num' }, `${g.frontalArea.toFixed(2)} m²`),
				h('dt', {}, 'L × W × H'), h('dd', { class: 'num' }, `${g.length.toFixed(2)} × ${g.width.toFixed(2)} × ${g.height.toFixed(2)} m`),
				h('dt', { title: 'Lowest and highest Cx among the networks; the prediction is their mean' }, 'Cx by network'), h('dd', { class: 'num' }, `${Math.min(...f.cdPModels).toFixed(3)} to ${Math.max(...f.cdPModels).toFixed(3)}`)),
			outside.length === 0 ? null : h('div', { class: 'result-outside', role: 'note' },
				h('div', {}, 'Outside the training range'),
				outside.map((d) => h('div', { class: 'num' }, `${DIMENSION_LABELS[d.name]} ${d.value.toFixed(2)} m, trained ${d.trainMin.toFixed(2)} to ${d.trainMax.toFixed(2)} m`))),
			h('div', { class: 'result-run num' }, `${millions(r.source.cells)} cells · ${seconds(took)} · ${r.device}`),
		];
	}

	function render() {
		const c = active();
		// The card is rebuilt, so nothing in it is hovered any more; a pinned spot belongs to the car it was on.
		marked.hovered = null;
		if (c !== shown) marked.pinned = null;
		subject = c?.result ? subjectOf(c, regions) : null;
		placeMarkers();
		if (!subject) {
			host.replaceChildren();
			shown = c;
			return;
		}
		const card = h('div', { class: `result-card${c === shown ? ' settled' : ''}` },
			h('div', { class: 'result-label' }, subject.name),
			readout(subject),
			extremes(c, subject),
			subject.whole ? null : facts(c.result));
		host.replaceChildren(card);
		shown = c;
	}

	return { render };
}
