// What the networks were told at a point, as places on and around the car, in world coordinates: each distance at
// the place `engine/features.js` measures it to, the plan-view outline's heights as the car's own cuts there, each
// size of nearby surface as the Gaussian it is weighted by, and the potential flow as its velocity along the surface.
// The page draws these on the car and in its flat drawings.

import { featureLayout } from '../../engine/features.js';

const f32 = Math.fround;
const clamp = (index, size) => Math.min(Math.max(index, 0), size - 1);
/** The raster layout of each feature definition, worked out once. */
const layouts = new WeakMap();
const layoutOf = (definition) => {
	if (!layouts.has(definition)) layouts.set(definition, featureLayout(definition));
	return layouts.get(definition);
};

/**
 * @typedef {{ type: 'reading', layer: string, at: number[], columns: number[], hollow: boolean }} Reading
 *   a place an input measured to; `hollow` for a plan-view outline read where no body is (its value is then minus
 *   the point's own distance from the centre plane)
 * @typedef {{ type: 'cut', layer: 'outline', axis: 2, value: number, own: boolean, columns: number[], span: number[],
 *   side: number }} Cut
 *   the car's outline in the plane z = `value`, and the inputs read along it: `span` is the stretch of x they are
 *   read over, `side` the side of the centre plane (+1 or −1) they are read on; `own` for the plane through the point
 * @typedef {{ type: 'guide', layer: string, from: number[], to: number[], columns: number[] }} Guide
 *   a distance measured from the point itself: to a face of the car's bounding box, or outboard to the outline
 * @typedef {{ type: 'neighbourhood', layer: 'surface', at: number[], scale: number, sigma: number, normal: number[],
 *   columns: number[] }} Neighbourhood
 *   one size of nearby surface: the Gaussian of standard deviation `sigma` about the point that weights the surface,
 *   and the mean normal of the surface under it
 * @typedef {{ type: 'velocity', layer: 'flow', at: number[], velocity: number[], columns: number[] }} Velocity
 *   the potential flow's velocity along the surface at the point, in units of the stream's speed
 * @typedef {Reading | Cut | Guide | Neighbourhood | Velocity} Mark
 */

/**
 * Every mark for an explanation. A mark's `layer` is the kind of the inputs it shows.
 * @param {import('../../engine/result.js').ModelColumn[]} columns
 * @param {import('../../engine/result.js').SurfaceLookups} lookups
 * @param {import('../../engine/result.js').Explanation} explanation
 * @returns {Mark[]}
 */
export function measure(columns, lookups, { position, normal, features }) {
	const { yCenter, definition } = lookups;
	const { h, z0, nz } = layoutOf(definition);
	const sy = f32(position[1] - f32(yCenter));
	// Offsets and directions are in the car's axes, except the plan-view outline's reach, which runs outboard on the
	// point's own side.
	const side = sy >= 0 ? 1 : -1;
	const direction = (column) => (column.kind === 'outline' ? [column.along[0], side * column.along[1], column.along[2]] : column.along);
	const marks = [];

	// Readings: a distance where it ends; readings at one place are one mark. A distance measured from the point
	// itself also gets a guide from the point to where it ends.
	const readings = new Map();
	columns.forEach((column, i) => {
		if (!column.along) return;
		const place = position.map((p, k) => p + column.offset[k]);
		const at = place.map((p, k) => p + features[i] * direction(column)[k]);
		const key = `${column.kind}:${at.join()}`;
		const hollow = column.kind === 'outline' && Math.abs(features[i] + Math.abs(sy)) < 1e-5;
		if (!readings.has(key)) readings.set(key, { type: 'reading', layer: column.kind, at, columns: [], hollow });
		readings.get(key).columns.push(i);
		if (!column.offset.some(Boolean)) marks.push({ type: 'guide', layer: column.kind, from: position, to: at, columns: [i] });
	});
	marks.push(...readings.values());

	// Cuts: the plan-view outline reads one layer of its raster in z, a raster step thick; the car is cut along the
	// layer's middle, rounded as the engine rounds.
	const heightZ = (offsetZ) => z0 + (clamp(Math.trunc(f32(f32(f32(position[2] + f32(offsetZ)) - z0) / h)), nz) + 0.5) * h;
	const cuts = new Map();
	columns.forEach((column, i) => {
		if (column.kind !== 'outline') return;
		const value = heightZ(column.offset[2]);
		if (!cuts.has(value)) cuts.set(value, { type: 'cut', layer: 'outline', axis: 2, value, own: column.offset[2] === 0, columns: [], span: [], side });
		cuts.get(value).columns.push(i);
	});
	for (const cut of cuts.values()) {
		const dx = cut.columns.map((i) => columns[i].offset[0]);
		cut.span = [position[0] + Math.min(...dx), position[0] + Math.max(...dx)];
	}
	marks.push(...cuts.values());

	// Nearby surface: per size, the Gaussian about the point and the mean normal under it, its first three inputs.
	const sizes = new Map();
	columns.forEach((column, i) => {
		if (column.kind !== 'surface') return;
		if (!sizes.has(column.scale)) sizes.set(column.scale, { type: 'neighbourhood', layer: 'surface', at: position, scale: column.scale, sigma: column.sigma, normal: [], columns: [] });
		const size = sizes.get(column.scale);
		if (column.part === 'whole' && size.normal.length < 3) size.normal.push(features[i]);
		size.columns.push(i);
	});
	marks.push(...sizes.values());

	// The potential flow: the velocity the car induces (its first three inputs) added to the stream, along the surface.
	const flow = columns.flatMap((column, i) => (column.kind === 'flow' ? [i] : []));
	if (flow.length) {
		const velocity = [1 + features[flow[0]], features[flow[1]], features[flow[2]]];
		const across = velocity[0] * normal[0] + velocity[1] * normal[1] + velocity[2] * normal[2];
		marks.push({ type: 'velocity', layer: 'flow', at: position, velocity: velocity.map((v, k) => v - across * normal[k]), columns: flow });
	}
	return marks;
}
