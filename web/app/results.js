// The result card over the view: the readout of what is selected, or of the whole car when nothing is.
//
// For the whole car: its pressure force coefficients (pressure only: the model predicts no wall shear), beside the
// CFD's when the file carries CFD, with the spread of Cx across the networks; where Cp is highest and lowest, on a
// side view of the car and flown to on request; then its reference area and size, and the dimensions outside the
// training range. For a selection: its coefficients, area and area-weighted mean Cp, with its share of the car's.
//
// The whole car's numbers are the Result's totals, summed over every polygon of the file, and the CFD's are summed
// over the same polygons. A selection's are summed over the file's own polygons it picks, or on a bundled example
// over its simplified mesh's faces (regions.js), with its shares over the same faces.

import { coefficient, h, s } from './dom.js';
import { percent, signed } from './format.js';

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
 * or the whole car from its totals, with the CFD's when the file carries CFD. Extremes are {cp, point, normal}.
 */
function subjectOf(c, regions) {
	const at = (e) => e && { cp: e.value, point: e.center, normal: e.normal };
	const selection = regions.selectionOn(c);
	if (selection) {
		const { part, whole } = selection;
		return { name: 'Selection', area: part.area, meanCp: part.meanCp, coefficients: part.coefficients, whole, extremes: { max: at(part.cpMax), min: at(part.cpMin) } };
	}
	const r = c.result;
	const t = r.totals;
	return { name: 'Whole car', coefficients: t.coefficients, reference: r.referenceTotals?.coefficients ?? null, byNetwork: r.forces.cdPModels, whole: null, extremes: { max: at(t.cpMax), min: at(t.cpMin) } };
}

const COEFFICIENTS = [
	['x', 'Pressure force along x on the frontal area: the pressure drag coefficient'],
	['y', 'Pressure force along y on the frontal area: side force'],
	['z', 'Pressure force along z on the frontal area: lift'],
];

/** The whole car's coefficients, with the CFD's beside them and the networks' spread of Cx under it. */
function wholeCarRows(of, row) {
	const cfd = of.reference;
	const byNetwork = `${Math.min(...of.byNetwork).toFixed(3)}–${Math.max(...of.byNetwork).toFixed(3)}`;
	return COEFFICIENTS.flatMap(([axis, title], i) => [
		row(coefficient(axis), title, signed(of.coefficients[i], 3), cfd && signed(cfd[i], 3)),
		i === 0 ? row([coefficient('x'), ' by network'], 'Lowest and highest Cx among the networks; the prediction is their mean', byNetwork, null, 'num quiet') : null,
	]);
}

/** A selection's coefficients, area and mean Cp, with its share of the car's where a share means something. */
function selectionRows(of, row) {
	const share = (part, whole) => (Math.abs(whole) < 1e-9 ? '' : percent(part / whole));
	const [cx, cy, cz] = of.coefficients;
	const [wx, , wz] = of.whole.coefficients;
	return [
		row(coefficient('x'), COEFFICIENTS[0][1], signed(cx, 3), share(cx, wx)),
		row(coefficient('y'), COEFFICIENTS[1][1], signed(cy, 3)),
		row(coefficient('z'), COEFFICIENTS[2][1], signed(cz, 3), share(cz, wz)),
		row(['Area'], 'Surface area', `${of.area.toFixed(2)} m²`, share(of.area, of.whole.area)),
		row(['Mean Cp'], 'Area-weighted mean pressure coefficient', of.meanCp === null ? '–' : signed(of.meanCp, 3)),
	];
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
			marker.hidden = !p.visible;
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
	 * The readout as a grid whose third column holds what the values are set against: a selection's share of the
	 * car's, or the CFD's for the whole car of a file that carries CFD. That column's head shares the title's row, so
	 * it costs no line.
	 */
	function readout(of) {
		const against = of.whole ? 'Share' : of.reference ? 'CFD' : null;
		const row = (label, title, text, aside, tone = 'num') => [h('dt', { title }, ...label), h('dd', { class: tone }, text), against ? h('dd', { class: 'result-aside num' }, aside ?? '') : null];
		const rows = of.whole ? selectionRows(of, row) : wholeCarRows(of, row);
		if (!against) return [h('div', { class: 'result-label' }, of.name), h('dl', { class: 'result-forces' }, rows)];
		return h('dl', { class: 'result-forces against' },
			h('dt', { class: 'result-label' }, of.name),
			h('dd', { class: 'result-head' }, of.whole ? '' : 'Predicted'),
			h('dd', { class: 'result-head', title: of.whole ? 'Share of the whole car\'s, summed over the same faces' : 'Summed from the CFD in the file, over the same polygons' }, against),
			rows);
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

	/** The whole car's reference area and size, and which of its dimensions lie outside the training range. */
	function facts(r) {
		const g = r.geometry;
		const outside = r.coverage.dimensions.filter((d) => d.value < d.trainMin || d.value > d.trainMax);
		return [
			h('dl', { class: 'result-facts' },
				h('dt', {}, 'Frontal area'), h('dd', { class: 'num' }, `${g.frontalArea.toFixed(2)} m²`),
				h('dt', {}, 'L × W × H'), h('dd', { class: 'num' }, `${g.length.toFixed(2)} × ${g.width.toFixed(2)} × ${g.height.toFixed(2)} m`)),
			outside.length === 0 ? null : h('div', { class: 'result-outside', role: 'note' },
				h('div', {}, 'Outside the training range'),
				outside.map((d) => h('div', { class: 'num' }, `${DIMENSION_LABELS[d.name]} ${d.value.toFixed(2)} m, trained ${d.trainMin.toFixed(2)} to ${d.trainMax.toFixed(2)} m`))),
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
			readout(subject),
			extremes(c, subject),
			subject.whole ? null : facts(c.result));
		host.replaceChildren(card);
		shown = c;
	}

	return { render };
}
