// What the networks were told at a point, as places on and around the car, in world coordinates: each reading at
// the place `engine/features.js` reads it, each regional box at the voxels it averages, each slice as the car's
// own cut along the plane the raster column lies in. The stencil draws these; the view names them on hover.

const f32 = Math.fround;

/**
 * The drawing layers, in the order the view lists them; each input belongs to one. `shown` is whether a layer is
 * drawn when the view opens: the cuts beside the point, drawn with the rest, crowd it.
 */
export const LAYERS = [
	{ id: 'position', label: 'Point and facing', kind: 'position', shown: true },
	{ id: 'bend', label: 'Boxes', kind: 'bend', shown: true },
	{ id: 'slice', label: 'Streamwise cut', kind: 'slice', shown: true },
	{ id: 'neighbour', label: 'Cuts beside it', kind: 'neighbour', shown: false },
	{ id: 'outline', label: 'Outline at its height', kind: 'outline', shown: true },
];

/** The layer an input is drawn in: its kind's, except a body-present flag, which belongs to the cut it is read in. */
export function layerOf(column) {
	if (column.kind === 'flag') return column.offset[1] === 0 ? 'slice' : 'neighbour';
	return column.kind;
}

/**
 * @typedef {{ type: 'reading', layer: string, kind: string, at: number[], columns: number[], hollow: boolean }} Reading
 *   a place an input was read, or measured to; `hollow` for a body-present flag that found no body
 * @typedef {{ type: 'cut', layer: string, axis: 1 | 2, value: number, own: boolean, columns: number[],
 *   span: number[], side: number }} Cut
 *   the car's outline in the plane where coordinate `axis` equals `value`, and the inputs read along it: `span` is
 *   the stretch of x they are read over, `side` the side of the centre plane (+1 or −1) a plan-view outline is read
 *   on; `own` for the plane through the point
 * @typedef {{ type: 'block', layer: 'bend', scale: number, sample: 'here' | 'upstream' | 'downstream',
 *   min: number[], max: number[], cell: number, weights: number[], columns: number[] }} Block
 *   the voxels a regional input averages: from `min`, cells of size `cell`, two (or one at the grid's edge)
 *   along each axis, the first weighted 1 − weights[a] along axis a and the second weights[a]
 * @typedef {{ type: 'guide', layer: string, kind: string, from: number[], to: number[], columns: number[] }} Guide
 *   a distance measured from the point itself: to a face of the car's bounding box, or along its own slice
 * @typedef {Reading | Cut | Block | Guide} Mark
 */

/**
 * Every mark for an explanation.
 * @param {import('../../engine/result.js').ModelColumn[]} columns
 * @param {import('../../engine/result.js').SurfaceLookups} lookups
 * @param {import('../../engine/result.js').Explanation} explanation
 * @returns {Mark[]}
 */
export function measure(columns, lookups, { position, features }) {
	const { yCenter, definition } = lookups;
	const h = definition.rasterStep;
	const yRelative = f32(position[1] - f32(yCenter));
	const side = yRelative >= 0 ? 1 : -1;
	const world = ([x, y, z]) => [x, side * y, z];
	const marks = [];

	// Readings: a distance where it ends. Readings at one place are one mark. A distance measured from the point
	// itself also gets a guide from the point to where it ends.
	const readings = new Map();
	const stations = new Map();
	const station = (layer, offset) => `${layer}:${offset.join()}`;
	columns.forEach((column, i) => {
		if (!column.along) return;
		const place = position.map((p, k) => p + world(column.offset)[k]);
		const at = place.map((p, k) => p + features[i] * world(column.along)[k]);
		const layer = layerOf(column);
		const key = `${column.kind}:${at.join()}`;
		if (!readings.has(key)) readings.set(key, { type: 'reading', layer, kind: column.kind, at, columns: [], hollow: false });
		const reading = readings.get(key);
		reading.columns.push(i);
		const here = station(layer, column.offset);
		if (!stations.has(here)) stations.set(here, new Set());
		stations.get(here).add(reading);
		if (!column.offset.some(Boolean)) marks.push({ type: 'guide', layer, kind: column.kind, from: position, to: at, columns: [i] });
	});
	// A flag that found body there belongs with the body's top and bottom read at its station; one that found none
	// is drawn at the station, hollow.
	columns.forEach((column, i) => {
		if (column.kind !== 'flag') return;
		const found = features[i] >= 0.5 && stations.get(station(layerOf(column), column.offset));
		if (found) {
			for (const reading of found) reading.columns.push(i);
			return;
		}
		const at = position.map((p, k) => p + world(column.offset)[k]);
		readings.set(`flag:${i}`, { type: 'reading', layer: layerOf(column), kind: 'flag', at, columns: [i], hollow: features[i] < 0.5 });
	});
	marks.push(...readings.values());

	// Cuts: the raster column a slice reads is a bin of |y − yCenter| (or of z) one raster step wide; the car is
	// cut along that bin's middle, on the side of the centre plane the reading lies.
	const sliceY = (offsetY) => {
		const across = f32(Math.abs(yRelative) + offsetY);
		const bin = Math.trunc(f32(Math.abs(across) / h));
		return yCenter + side * Math.sign(across || 1) * (bin + 0.5) * h;
	};
	const heightZ = (offsetZ) => {
		const z0 = f32(definition.domainZ[0]);
		const bin = Math.trunc(f32(f32(f32(position[2] + offsetZ) - z0) / f32(h)));
		return z0 + (bin + 0.5) * h;
	};
	const cuts = new Map();
	columns.forEach((column, i) => {
		const layer = layerOf(column);
		let cut = null;
		if (layer === 'slice') cut = { axis: 1, value: sliceY(0), own: true };
		else if (layer === 'neighbour') cut = { axis: 1, value: sliceY(column.offset[1]), own: false };
		else if (layer === 'outline') cut = { axis: 2, value: heightZ(column.offset[2]), own: column.offset[2] === 0 };
		if (!cut) return;
		const key = `${cut.axis}:${cut.value}`;
		if (!cuts.has(key)) cuts.set(key, { type: 'cut', layer, ...cut, columns: [] });
		cuts.get(key).columns.push(i);
	});
	for (const cut of cuts.values()) {
		const dx = cut.columns.map((i) => columns[i].offset[0]);
		cut.span = [position[0] + Math.min(...dx), position[0] + Math.max(...dx)];
		cut.side = side;
	}
	marks.push(...cuts.values());

	// Regional boxes: the voxels each scale's grid blends at the point and one box up- and downstream.
	const blocks = new Map();
	columns.forEach((column, i) => {
		if (column.kind !== 'bend') return;
		const dx = column.offset[0];
		const scale = Math.max(...columns.filter((c) => c.group === column.group).map((c) => Math.abs(c.offset[0])));
		const key = `${scale}:${dx}`;
		if (!blocks.has(key)) {
			const grid = lookups.voxels.find((g) => g.scale === scale);
			const x = dx === 0 ? position[0] : f32(position[0] + f32(dx));
			const sample = dx === 0 ? 'here' : dx < 0 ? 'upstream' : 'downstream';
			blocks.set(key, { type: 'block', layer: 'bend', scale, sample, ...voxelBlock(grid, [x, yRelative, position[2]], yCenter), columns: [] });
		}
		blocks.get(key).columns.push(i);
	});
	marks.push(...blocks.values());
	return marks;
}

/** The voxels `sampleGrid` in `engine/features.js` blends at grid point p (y relative to the centre plane). */
function voxelBlock({ scale, origin, shape }, p, yCenter) {
	const min = [];
	const max = [];
	const weights = [];
	for (let a = 0; a < 3; a++) {
		const c = Math.min(Math.max((p[a] - origin[a]) / scale - 0.5, 0), shape[a] - 1);
		const first = Math.floor(c);
		const second = Math.min(first + 1, shape[a] - 1);
		const shift = a === 1 ? f32(yCenter) : 0;
		min.push(origin[a] + first * scale + shift);
		max.push(origin[a] + (second + 1) * scale + shift);
		weights.push(second === first ? 0 : c - first);
	}
	return { min, max, cell: scale, weights };
}
