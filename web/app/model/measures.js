// What the networks were told at a point, as places on and around the car, in world coordinates: each distance at
// the place `engine/features.js` measures it to, each size of nearby surface with the mean normal it finds, and the
// slices' reads at each station with a hairline from the station to where each ends. The page draws these on the
// car.

/**
 * @typedef {{ type: 'reading', layer: string, at: number[], columns: number[], hollow: boolean, station?: number }} Reading
 *   a place an input measured to; `hollow` for a slice read where the slice holds no body near the point's place
 *   (outboard it then reads the centre plane, above and below the ground); `station` a slice read's
 * @typedef {{ type: 'guide', layer: string, from: number[], to: number[], columns: number[] }} Guide
 *   a distance measured from the point itself: to a face of the car's bounding box, or to the centre plane
 * @typedef {{ type: 'hairline', layer: 'slice', station: number, from: number[], to: number[], ground: boolean,
 *   columns: number[] }} Hairline
 *   a slice's read from its station on the point's fore-aft line to where it ends; `ground` for a read above or
 *   below where the slice holds no body, which runs to the ground
 * @typedef {{ type: 'stations', layer: 'slice', from: number[], to: number[] }} Stations
 *   the point's fore-aft line from the first station to the last
 * @typedef {{ type: 'neighbourhood', layer: 'surface', at: number[], scale: number, normal: number[],
 *   columns: number[] }} Neighbourhood
 *   one size of nearby surface: the surface within about `scale` of the point (weighted by a Gaussian that fades
 *   with distance), and its mean normal
 * @typedef {Reading | Guide | Hairline | Stations | Neighbourhood} Mark
 */

/** Two reads closer than this, in metres, are the same number. */
const SAME = 1e-5;

/**
 * Every mark for an explanation. A mark's `layer` is the kind of the inputs it shows.
 * @param {import('../../engine/result.js').ModelColumn[]} columns
 * @param {import('../../engine/result.js').SurfaceLookups} lookups
 * @param {import('../../engine/result.js').Explanation} explanation
 * @returns {Mark[]}
 */
export function measure(columns, { yCenter, bounds }, { position, features }) {
	const sy = position[1] - yCenter;
	const ground = bounds[4];
	// Offsets and directions are in the car's axes, except a slice's reach outboard, which runs away from the centre
	// plane on the point's own side.
	const side = sy >= 0 ? 1 : -1;
	const direction = (column) => (column.read === 'outboard' ? [0, side * column.along[1], 0] : column.along);
	// A slice with no body near the point's place across the car reads the ground both above and below. Reads are
	// blended between raster bins, so a read near the body's edge is a blend and only one with no body near is exact.
	const reads = new Map();
	columns.forEach((column, i) => {
		if (column.kind !== 'slice') return;
		if (!reads.has(column.station)) reads.set(column.station, {});
		reads.get(column.station)[column.read] = i;
	});
	const empty = (column) => {
		const { above, below } = reads.get(column.station);
		return Math.abs(features[above] + features[below]) < SAME && Math.abs(position[2] + features[above] - ground) < SAME;
	};
	const marks = [];

	// Readings: a distance where it ends; readings at one place are one mark. A slice's read gets a hairline from its
	// station; a distance from the point itself gets a guide.
	const readings = new Map();
	columns.forEach((column, i) => {
		if (!column.along) return;
		const place = position.map((p, k) => p + column.offset[k]);
		const at = place.map((p, k) => p + features[i] * direction(column)[k]);
		const key = `${column.kind}:${at.join()}`;
		const slice = column.kind === 'slice';
		const hollow = slice && (column.read === 'outboard' ? Math.abs(features[i] + Math.abs(sy)) < SAME : empty(column));
		if (!readings.has(key)) readings.set(key, { type: 'reading', layer: column.kind, at, columns: [], hollow, ...(slice ? { station: column.station } : {}) });
		readings.get(key).columns.push(i);
		if (slice) {
			marks.push({ type: 'hairline', layer: 'slice', station: column.station, from: place, to: at, ground: hollow && column.read !== 'outboard', columns: [i] });
		} else if (!column.offset.some(Boolean)) {
			marks.push({ type: 'guide', layer: column.kind, from: position, to: at, columns: [i] });
		}
	});
	marks.push(...readings.values());
	if (reads.size) {
		const stations = [...reads.keys()];
		const along = (dx) => [position[0] + dx, position[1], position[2]];
		marks.push({ type: 'stations', layer: 'slice', from: along(Math.min(...stations)), to: along(Math.max(...stations)) });
	}

	// Nearby surface: per size, the mean normal of the surface about the point, its first three inputs.
	const sizes = new Map();
	columns.forEach((column, i) => {
		if (column.kind !== 'surface') return;
		if (!sizes.has(column.scale)) sizes.set(column.scale, { type: 'neighbourhood', layer: 'surface', at: position, scale: column.scale, normal: [], columns: [] });
		const size = sizes.get(column.scale);
		if (column.part === 'whole' && size.normal.length < 3) size.normal.push(features[i]);
		size.columns.push(i);
	});
	marks.push(...sizes.values());
	return marks;
}
