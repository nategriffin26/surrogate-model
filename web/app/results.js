// The result card over the view: what an engineer reads off a surface-pressure solution. The pressure drag, where
// it comes from (forward- against rearward-facing surfaces), and the pressure extremes on a side view of the car;
// then the car's reference area and size, the dimensions outside the training range (only when there are any), and
// one line about the run. Hovering a part of the drag picks out its faces on the car; an extreme flies the camera to
// its spot and marks it.

import { h, s } from './dom.js';
import { millions, seconds, signed } from './format.js';

const DIMENSION_LABELS = { length: 'Length', width: 'Width', height: 'Height' };
/** The card's content width, which the drawings fill: 264 px less its padding and border. */
const WIDTH = 234;
const PROFILE_BINS = 96;

/** The polygons of highest and lowest Cp over the whole car, from the Result's totals, as the card marks them. */
function extremesOf({ cpMax, cpMin }) {
	const at = (e) => ({ cp: e.value, point: e.center, normal: e.normal });
	return { max: at(cpMax), min: at(cpMin) };
}

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

/** host: where the card goes; overlay: over the 3D view, for the extremes' markers; run(name, args) runs an operation. */
export function createResultCard({ host, overlay, viewer, active, run }) {
	const derived = new WeakMap();
	const markers = { max: h('div', { class: 'extreme-marker max', hidden: true }, h('span', {}, 'max')), min: h('div', { class: 'extreme-marker min', hidden: true }, h('span', {}, 'min')) };
	overlay.append(markers.max, markers.min);
	let shown = null;
	/** The extremes marked on the car: the one hovered in the card, and the one last flown to. */
	const marked = { hovered: null, pinned: null };

	// The outline is drawn once per car; the extremes follow the totals, which every polygon replaces once summed.
	const outline = (c) => derived.get(c) ?? derived.set(c, profileOf(c.result.mesh, c.result.geometry.bounds)).get(c);
	const of = (c) => ({ extremes: extremesOf(c.result.totals), profile: outline(c) });

	function placeMarkers() {
		const c = active();
		for (const which of ['max', 'min']) {
			const marker = markers[which];
			const on = c?.result && (marked.hovered === which || marked.pinned === which);
			marker.hidden = !on;
			if (!on) continue;
			const { point } = of(c).extremes[which];
			const p = viewer.project(point);
			marker.hidden = !p.visible || viewer.clipped(point);
			marker.style.transform = `translate(${p.x}px, ${p.y}px)`;
		}
	}
	viewer.addEventListener('camera', placeMarkers);
	// The pinned spot stays marked until the car is pressed again.
	viewer.canvas.addEventListener('pointerdown', () => {
		marked.pinned = null;
		placeMarkers();
	});

	/** Hovering an element picks out faces on the car while the pointer is on it. */
	const emphasis = (el, faces) => {
		el.addEventListener('pointerenter', () => viewer.setEmphasis(faces));
		el.addEventListener('pointerleave', () => viewer.setEmphasis(null));
		return el;
	};

	/**
	 * The pressure drag split by facing, as signed bars about zero on one scale: the forward-facing part (negative,
	 * thrust, when suction over the hood and windscreen outweighs the stagnation on the nose) and the rearward-facing
	 * part, each from zero on its own row, with their sum, C_D,p, marked across both.
	 */
	function split({ forward, rearward }) {
		const net = forward + rearward;
		const lo = Math.min(0, forward, rearward), hi = Math.max(0, forward, rearward);
		const x = (v) => ((v - lo) / (hi - lo || 1)) * WIDTH;
		const bar = (faces, value, y) => emphasis(s('rect', { class: `split-part ${faces}`, x: x(Math.min(0, value)), y, width: Math.max(1, Math.abs(x(value) - x(0))), height: 6, rx: 1 }), faces);
		const withSign = (v) => `${v > 0 ? '+' : ''}${signed(v, 3)}`;
		const part = (faces, label, value, title) => emphasis(h('div', { class: `split-item ${faces}`, title }, h('span', {}, label), h('b', { class: 'num' }, withSign(value))), faces);
		return h('div', { class: 'split' },
			s('svg', { class: 'split-bar', viewBox: `0 0 ${WIDTH} 16`, width: WIDTH, height: 16, role: 'img', 'aria-label': `Forward-facing ${withSign(forward)}, rearward-facing ${withSign(rearward)}, net ${net.toFixed(3)}` },
				s('line', { class: 'zero', x1: x(0), x2: x(0), y1: 0, y2: 16 }),
				bar('forward', forward, 1),
				bar('rearward', rearward, 9),
				s('line', { class: 'net', x1: x(net), x2: x(net), y1: 0, y2: 16 }, s('title', {}, `Net ${net.toFixed(3)}`))),
			h('div', { class: 'split-legend' },
				part('forward', 'Forward-facing', forward, 'Pressure drag of the surfaces facing upstream (outward nₓ < 0); negative is thrust, where suction on them pulls the car forward'),
				part('rearward', 'Rearward-facing', rearward, 'Pressure drag of the surfaces facing downstream (outward nₓ > 0)')));
	}

	function extremes(c) {
		const { extremes: ex, profile } = of(c);
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

	function render() {
		const c = active();
		// The card is rebuilt, so nothing in it is hovered any more; a pinned spot belongs to the car it was on.
		marked.hovered = null;
		viewer.setEmphasis(null);
		if (c !== shown) marked.pinned = null;
		placeMarkers();
		if (!c?.result) {
			host.replaceChildren();
			shown = c;
			return;
		}
		const r = c.result;
		const f = r.forces;
		const g = r.geometry;
		const outside = r.coverage.dimensions.filter((d) => d.value < d.trainMin || d.value > d.trainMax);
		const took = r.timings.reduce((sum, t) => sum + t.seconds, 0);
		const card = h('div', { class: `result-card${c === shown ? ' settled' : ''}` },
			h('div', { class: 'result-label' }, 'Pressure drag'),
			h('div', { class: 'headline' },
				h('span', { class: 'symbol' }, 'C', h('sub', {}, 'D,p')),
				h('span', { class: 'big num' }, f.cdP.toFixed(3))),
			h('div', { class: 'result-range num', title: 'Lowest and highest pressure drag among the networks; the prediction is their mean' },
				`${Math.min(...f.cdPModels).toFixed(3)} to ${Math.max(...f.cdPModels).toFixed(3)} across ${f.cdPModels.length} networks`),
			split(f.byFacing),
			extremes(c),
			h('dl', { class: 'result-facts' },
				h('dt', {}, 'Frontal area'), h('dd', { class: 'num' }, `${g.frontalArea.toFixed(2)} m²`),
				h('dt', {}, 'L × W × H'), h('dd', { class: 'num' }, `${g.length.toFixed(2)} × ${g.width.toFixed(2)} × ${g.height.toFixed(2)} m`)),
			outside.length === 0 ? null : h('div', { class: 'result-outside', role: 'note' },
				h('div', {}, 'Outside the training range'),
				outside.map((d) => h('div', { class: 'num' }, `${DIMENSION_LABELS[d.name]} ${d.value.toFixed(2)} m, trained ${d.trainMin.toFixed(2)} to ${d.trainMax.toFixed(2)} m`))),
			h('div', { class: 'result-run num' }, `${millions(r.source.cells)} cells · ${seconds(took)} · ${r.device}`));
		host.replaceChildren(card);
		shown = c;
	}

	return { render };
}
