// The invented body the inputs are taught on: a smooth car-like shape defined by its side profile and half-width
// along its length, meshed into quadrilaterals no longer than 14 mm and handed to the engine like any car given as
// arrays, which measures and predicts it as it does a file. Its profile follows the centre-plane outline of DrivAerML
// run 1 (`web/app/examples.js`) scaled to its length: bonnet, windscreen, roof, rear window and boot deck, without
// wheels. Axes as in the car files: x back from the nose, z up, y across from the centre plane. Lengths in metres.

/** Length, and where the nose and the lowest point sit, inside the box the model's slices are read in. */
export const LENGTH = 4.4;
const NOSE_X = -0.7;
const LOW_Z = -0.3;

/** A monotone cubic through knots (Fritsch–Carlson): no overshoot between them. */
function monotone(knots) {
	const xs = knots.map((k) => k[0]);
	const ys = knots.map((k) => k[1]);
	const n = xs.length;
	const d = xs.slice(1).map((x, i) => (ys[i + 1] - ys[i]) / (x - xs[i]));
	const m = xs.map((_, i) =>
		i === 0 ? d[0] : i === n - 1 ? d[n - 2] : d[i - 1] * d[i] <= 0 ? 0 : (d[i - 1] + d[i]) / 2,
	);
	for (let i = 0; i < n - 1; i++) {
		if (d[i] === 0) {
			m[i] = 0;
			m[i + 1] = 0;
			continue;
		}
		const a = m[i] / d[i];
		const b = m[i + 1] / d[i];
		const q = a * a + b * b;
		if (q > 9) {
			m[i] = (3 * a * d[i]) / Math.sqrt(q);
			m[i + 1] = (3 * b * d[i]) / Math.sqrt(q);
		}
	}
	return (x) => {
		const t0 = Math.min(Math.max(x, xs[0]), xs[n - 1]);
		let i = 0;
		while (i < n - 2 && t0 > xs[i + 1]) i++;
		const hh = xs[i + 1] - xs[i];
		const t = (t0 - xs[i]) / hh;
		const t2 = t * t;
		const t3 = t2 * t;
		return (
			(2 * t3 - 3 * t2 + 1) * ys[i] +
			(t3 - 2 * t2 + t) * hh * m[i] +
			(-2 * t3 + 3 * t2) * ys[i + 1] +
			(t3 - t2) * hh * m[i + 1]
		);
	};
}

// Side profile (top and bottom above the lowest point) and half-width, by distance s back from the nose.
const TOP = monotone([
	[0, 0.5],
	[0.15, 0.6],
	[0.4, 0.68],
	[0.8, 0.75],
	[1.15, 0.79],
	[1.3, 0.82],
	[1.5, 0.93],
	[1.8, 1.08],
	[2.1, 1.19],
	[2.4, 1.22],
	[2.8, 1.22],
	[3.1, 1.2],
	[3.45, 1.1],
	[3.8, 0.95],
	[3.95, 0.91],
	[4.2, 0.87],
	[4.4, 0.84],
]);
const BOTTOM = monotone([
	[0, 0.08],
	[0.25, 0.04],
	[0.6, 0.02],
	[0.9, 0],
	[3.5, 0],
	[3.9, 0.03],
	[4.2, 0.09],
	[4.4, 0.18],
]);
const HALF_WIDTH = monotone([
	[0, 0.8],
	[0.6, 0.89],
	[1.2, 0.92],
	[3.4, 0.92],
	[4.4, 0.86],
]);
const NOSE_ROUND = 0.2;
const TAIL_ROUND = 0.1;
/** The beltline, above the lowest point: the cabin above it narrows to the roof by up to CABIN_LEAN of the width. */
const BELT = 0.82;
const CABIN_LEAN = 0.3;
/** The heights the nose and the tail close to: the middle of the body's end there. */
const NOSE_MID = (TOP(0) + BOTTOM(0)) / 2;
const TAIL_MID = (TOP(LENGTH) + BOTTOM(LENGTH)) / 2;
const EXPONENT = 4;

/** The rounding that closes the body at the nose and the tail: 1 between, 0 at the ends. */
function closure(s) {
	if (s <= 0 || s >= LENGTH) return 0;
	if (s < NOSE_ROUND) return Math.sqrt(1 - (1 - s / NOSE_ROUND) ** 2);
	if (s > LENGTH - TAIL_ROUND) return Math.sqrt(1 - ((s - (LENGTH - TAIL_ROUND)) / TAIL_ROUND) ** 2);
	return 1;
}

/** The body's station at s from the nose: its top and bottom (z), half-width, beltline (z) and the cabin's lean. */
export function profile(s) {
	const c = closure(s);
	const mid = s < LENGTH / 2 ? NOSE_MID : TAIL_MID;
	const top = mid + (TOP(s) - mid) * c;
	const bottom = mid + (BOTTOM(s) - mid) * c;
	// A station that rises above the beltline carries a cabin, leaning in more the taller it is.
	const lean = CABIN_LEAN * Math.min(Math.max((TOP(s) - BELT) / 0.3, 0), 1);
	return { top: LOW_Z + top, bottom: LOW_Z + bottom, half: HALF_WIDTH(s) * c, belt: LOW_Z + BELT, lean };
}

/** A point of the station's outline at angle θ from the bottom (θ = 0), round through +y (θ = π/2). */
function outlinePoint(p, theta) {
	const u = Math.sin(theta);
	const v = -Math.cos(theta);
	const Y = Math.sign(u) * Math.abs(u) ** (2 / EXPONENT);
	const T = Math.sign(v) * Math.abs(v) ** (2 / EXPONENT);
	const z = (p.top + p.bottom) / 2 + ((p.top - p.bottom) / 2) * T;
	const cabin = p.lean ? Math.min(Math.max((z - p.belt) / (p.top - p.belt), 0), 1) : 0;
	return [p.half * Y * (1 - p.lean * cabin), z];
}

/** Points taken round an outline before they are spaced evenly along it. */
const FINE = 1024;

/**
 * The outline of the body across x, as `count` points (y, z) evenly spaced along it from the bottom on the centre
 * plane round through +y, and its length. Empty beyond the ends.
 */
export function outline(x, count) {
	const s = x - NOSE_X;
	if (s <= 0 || s >= LENGTH) return { points: [], length: 0 };
	const p = profile(s);
	const fine = [];
	const along = [0];
	for (let i = 0; i <= FINE; i++) fine.push(outlinePoint(p, (2 * Math.PI * i) / FINE));
	for (let i = 1; i <= FINE; i++)
		along.push(along[i - 1] + Math.hypot(fine[i][0] - fine[i - 1][0], fine[i][1] - fine[i - 1][1]));
	const length = along[FINE];
	const points = [];
	let j = 0;
	for (let k = 0; k < count; k++) {
		const target = (length * k) / count;
		while (j < FINE - 1 && along[j + 1] < target) j++;
		const t = along[j + 1] > along[j] ? (target - along[j]) / (along[j + 1] - along[j]) : 0;
		points.push([fine[j][0] + t * (fine[j + 1][0] - fine[j][0]), fine[j][1] + t * (fine[j + 1][1] - fine[j][1])]);
	}
	return { points, length };
}

/** The body from the side: its top and bottom at each x, as one closed polygon of (x, z). */
export function sideOutline(step = 0.01) {
	const top = [];
	const bottom = [];
	for (let s = 0; s <= LENGTH + 1e-9; s += step) {
		const p = profile(Math.min(s, LENGTH));
		top.push([NOSE_X + s, p.top]);
		bottom.push([NOSE_X + s, p.bottom]);
	}
	return [...top, [NOSE_X + LENGTH, profile(LENGTH).top], ...bottom.reverse()];
}

/** The longest edge of a cell; the slices' 2 cm raster needs every cell smaller than its step. */
const CELL = 0.014;
/** Points round the outline at which neighbouring rings are compared when the step between them is chosen. */
const MATCH = 64;

/** `count` points round the outline at x; at an end of the body, the point the outlines close to, repeated. */
function ring(x, count) {
	const { points } = outline(x, count);
	if (points.length) return points;
	const p = profile(x - NOSE_X < LENGTH / 2 ? 0 : LENGTH);
	return Array.from({ length: count }, () => [0, (p.top + p.bottom) / 2]);
}

/**
 * The body as the engine takes a car given as arrays (`ParsedSurface`): bands of quadrilaterals along x, each band
 * its own two rings of points, no edge longer than `CELL`; a quadrilateral at a closed end, where a ring is one point,
 * is a triangle.
 * @returns {import('../../engine/result.js').ParsedSurface}
 */
export function bodySurface() {
	const end = NOSE_X + LENGTH;
	const ringXs = [NOSE_X];
	for (let x = NOSE_X; x < end - 1e-9; ringXs.push(x)) {
		const here = ring(x, MATCH);
		let dx = Math.min(CELL, end - x);
		while (
			dx > 1e-5 &&
			Math.max(...ring(x + dx, MATCH).map((p, k) => Math.hypot(dx, p[0] - here[k][0], p[1] - here[k][1]))) > CELL
		)
			dx /= 2;
		x = Math.min(x + dx, end);
	}
	const lengths = ringXs.map((rx) => outline(rx, 4).length);
	const points = [];
	const connectivity = [];
	const offsets = [0];
	for (let i = 0; i < ringXs.length - 1; i++) {
		let m = Math.ceil(Math.max(lengths[i], lengths[i + 1]) / CELL);
		m += m % 2;
		const first = points.length / 3;
		for (const [x, around] of [
			[ringXs[i], ring(ringXs[i], m)],
			[ringXs[i + 1], ring(ringXs[i + 1], m)],
		]) {
			for (const [y, z] of around) points.push(x, y, z);
		}
		const same = (a, b) =>
			points[3 * a + 1] === points[3 * b + 1] &&
			points[3 * a + 2] === points[3 * b + 2] &&
			points[3 * a] === points[3 * b];
		for (let k = 0; k < m; k++) {
			// Round the outline, then back along x: the order that faces out of the body.
			const corners = [first + k, first + ((k + 1) % m), first + m + ((k + 1) % m), first + m + k];
			const kept = corners.filter((c, n) => !same(c, corners[(n + 1) % 4]));
			if (kept.length < 3) continue;
			connectivity.push(...kept);
			offsets.push(connectivity.length);
		}
	}
	return {
		name: 'invented body',
		points: Float32Array.from(points),
		connectivity: Uint32Array.from(connectivity),
		offsets: Uint32Array.from(offsets),
	};
}

/**
 * The vertex a point of the side view (x, z) lands on: of the vertices on the near side (y ≥ 0), the outermost in a
 * 1 cm square, nearest the point. Built once from the body's drawn vertices.
 */
export function sideLookup(positions) {
	const step = 0.01;
	let [xmin, zmin] = [Infinity, Infinity];
	for (let v = 0; v < positions.length; v += 3) {
		xmin = Math.min(xmin, positions[v]);
		zmin = Math.min(zmin, positions[v + 2]);
	}
	const cols = Math.ceil(LENGTH / step) + 2;
	const rows = Math.ceil(2 / step);
	const best = new Int32Array(cols * rows).fill(-1);
	for (let v = 0; v < positions.length / 3; v++) {
		const y = positions[3 * v + 1];
		if (y < 0) continue;
		const i = Math.floor((positions[3 * v] - xmin) / step);
		const j = Math.floor((positions[3 * v + 2] - zmin) / step);
		if (i < 0 || j < 0 || i >= cols || j >= rows) continue;
		const k = i * rows + j;
		if (best[k] < 0 || positions[3 * best[k] + 1] < y) best[k] = v;
	}
	return (x, z) => {
		const i0 = Math.floor((x - xmin) / step);
		const j0 = Math.floor((z - zmin) / step);
		let found = -1;
		let bestD = Infinity;
		for (let radius = 0; radius < 40 && found < 0; radius++) {
			for (let i = i0 - radius; i <= i0 + radius; i++)
				for (let j = j0 - radius; j <= j0 + radius; j++) {
					if (i < 0 || j < 0 || i >= cols || j >= rows) continue;
					const v = best[i * rows + j];
					if (v < 0) continue;
					const d = Math.hypot(positions[3 * v] - x, positions[3 * v + 2] - z);
					if (d < bestD) {
						bestD = d;
						found = v;
					}
				}
		}
		return found;
	};
}

/** The vertex nearest (y, z) among those at x, the station of vertex `v`: a cell moved round its own station. */
export function aroundLookup(positions, v, y, z) {
	const x = positions[3 * v];
	let found = v;
	let bestD = Infinity;
	for (let k = 0; k < positions.length / 3; k++) {
		if (positions[3 * k] !== x) continue;
		const d = Math.hypot(positions[3 * k + 1] - y, positions[3 * k + 2] - z);
		if (d < bestD) {
			bestD = d;
			found = k;
		}
	}
	return found;
}
