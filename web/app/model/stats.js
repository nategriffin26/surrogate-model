// What the page measures on the reference car, over its working mesh's triangles weighted by area: where the
// prediction misses the CFD (by region and cell by cell) and how the misses compare with the spread σ(Cp).

/**
 * Regions of the surface, in the order the chart lists them, each named at `accuracy.regions.<id>` in copy.json.
 * Wheels first: every other region leaves them out.
 */
export const REGIONS = [{ id: 'wheels' }, { id: 'underbody' }, { id: 'base' }, { id: 'front' }, { id: 'sides' }, { id: 'top' }];
/** A face belongs to a facing region when its normal is within about 45° of the axis. */
const FACING = 0.7;
/** Wheels and arches: within this distance of an axle in x, this far out from the centre plane, this high off the ground. */
const WHEEL = { reach: 0.42, outboard: 0.55, height: 0.75 };
/** Cells within this height of the lowest point are the tyres' contact with the ground. */
const CONTACT = 0.01;
const CALIBRATION_BINS = 10;
const SCATTER_BINS = 120;

/** The region of each triangle, as an index into REGIONS, or −1 for faces that are none of them. */
function regionOf({ centers, normals }, bounds, yCenter) {
	const count = centers.length / 3;
	const zmin = bounds[4];
	// The axles: the middle of the front and the rear tyres' contact patches.
	const contact = [];
	for (let t = 0; t < count; t++) if (centers[3 * t + 2] < zmin + CONTACT) contact.push(centers[3 * t]);
	if (contact.length < 2) throw new Error('The car has no tyres touching the ground, so its wheels cannot be found.');
	const split = (Math.min(...contact) + Math.max(...contact)) / 2;
	const median = (values) => values.sort((a, b) => a - b)[values.length >> 1];
	const axles = [median(contact.filter((x) => x < split)), median(contact.filter((x) => x >= split))];
	const region = new Int8Array(count);
	const index = Object.fromEntries(REGIONS.map((r, i) => [r.id, i]));
	for (let t = 0; t < count; t++) {
		const [x, y, z] = [centers[3 * t], centers[3 * t + 1] - yCenter, centers[3 * t + 2]];
		const [nx, ny, nz] = [normals[3 * t], normals[3 * t + 1], normals[3 * t + 2]];
		const wheel = axles.some((a) => Math.abs(x - a) < WHEEL.reach) && Math.abs(y) > WHEEL.outboard && z < zmin + WHEEL.height;
		region[t] = wheel ? index.wheels
			: nz < -FACING ? index.underbody
				: nx > FACING ? index.base
					: nx < -FACING ? index.front
						: Math.abs(ny) > FACING ? index.sides
							: nz > FACING ? index.top
								: -1;
	}
	return region;
}

/** The values below which shares `qs` of the area lie, read from a fine histogram over the values' range. */
function areaQuantiles(values, areas, qs) {
	let min = Infinity;
	let max = -Infinity;
	for (const v of values) {
		min = Math.min(min, v);
		max = Math.max(max, v);
	}
	const fine = 8192;
	const width = (max - min) / fine || 1;
	const histogram = new Float64Array(fine);
	let total = 0;
	for (let i = 0; i < values.length; i++) {
		histogram[Math.min(fine - 1, Math.floor((values[i] - min) / width))] += areas[i];
		total += areas[i];
	}
	let k = 0;
	let below = histogram[0];
	return qs.map((q) => {
		while (below < q * total && k < fine - 1) below += histogram[++k];
		return min + (k + 1) * width;
	});
}

/**
 * @param {import('../../engine/result.js').CellSet} cells  the working mesh's triangles, with CFD
 * @param {number[]} bounds
 * @param {number} yCenter  y of the car's centre plane
 */
export function caseStats(cells, bounds, yCenter) {
	const { cp, cfd, sigma, areas } = cells;
	const count = areas.length;
	const region = regionOf(cells, bounds, yCenter);

	const regionArea = new Float64Array(REGIONS.length);
	const regionError = new Float64Array(REGIONS.length);
	let total = 0;
	for (let t = 0; t < count; t++) {
		const error = Math.abs(cp[t] - cfd[t]);
		total += areas[t];
		if (region[t] < 0) continue;
		regionArea[region[t]] += areas[t];
		regionError[region[t]] += areas[t] * error;
	}
	const regions = REGIONS.map((r, i) => ({ ...r, share: regionArea[i] / total, error: regionError[i] / regionArea[i] }));

	// Spread against error: cells in bins of equal area by σ, each bin's mean σ and root-mean-square error.
	const edges = areaQuantiles(sigma, areas, Array.from({ length: CALIBRATION_BINS - 1 }, (_, k) => (k + 1) / CALIBRATION_BINS));
	const bins = Array.from({ length: CALIBRATION_BINS }, () => ({ area: 0, sigma: 0, squared: 0 }));
	for (let t = 0; t < count; t++) {
		let b = 0;
		while (b < edges.length && sigma[t] >= edges[b]) b++;
		const bin = bins[b];
		bin.area += areas[t];
		bin.sigma += areas[t] * sigma[t];
		bin.squared += areas[t] * (cp[t] - cfd[t]) ** 2;
	}
	const calibration = bins.filter((b) => b.area > 0).map((b) => ({ sigma: b.sigma / b.area, error: Math.sqrt(b.squared / b.area) }));

	// Predicted against CFD: area per cell of a square grid over the range that holds 99 % of the surface's CFD.
	const [low, high] = areaQuantiles(cfd, areas, [0.005, 0.995]);
	const lo = Math.floor(low * 10) / 10;
	const hi = Math.ceil(high * 10) / 10;
	const density = new Float64Array(SCATTER_BINS * SCATTER_BINS);
	const cell = (v) => Math.floor(((v - lo) / (hi - lo)) * SCATTER_BINS);
	for (let t = 0; t < count; t++) {
		const i = cell(cfd[t]);
		const j = cell(cp[t]);
		if (i < 0 || j < 0 || i >= SCATTER_BINS || j >= SCATTER_BINS) continue;
		density[j * SCATTER_BINS + i] += areas[t];
	}

	return { regions, calibration, scatter: { lo, hi, bins: SCATTER_BINS, density } };
}
