// The potential flow over the open car as the page draws it, from the flow inputs at every vertex of the working mesh
// (`engine.flowOnMesh`): the pressure an ideal flow would put on the surface, and arrows of its velocity along the
// surface on the side the side pictures show.

/** Spacing of the velocity arrows on the side of the car, in metres. */
const ARROW_SPACING = 0.25;

/**
 * @param {import('../../engine/result.js').DisplayMesh} mesh
 * @param {{ induced: Float32Array, speedSquared: Float32Array }} flow  the flow inputs at the mesh's vertices
 * @returns {{ pressure: Float32Array, arrows: { at: number[], velocity: number[] }[] }}  `pressure` per vertex is the
 *   potential flow's Cp, 1 − |V_t|²; `velocity` its velocity along the surface, in units of the stream's speed
 */
export function surfaceFlow({ positions, normals, bounds }, { induced, speedSquared }) {
	const pressure = Float32Array.from(speedSquared, (s) => 1 - s);
	// On a grid over the side, the vertex nearest the −y camera in each cell: the one the side picture shows.
	const [x0, x1, , , z0, z1] = bounds;
	const columns = Math.ceil((x1 - x0) / ARROW_SPACING);
	const rows = Math.ceil((z1 - z0) / ARROW_SPACING);
	const shown = new Int32Array(columns * rows).fill(-1);
	for (let v = 0; v < positions.length / 3; v++) {
		const i = Math.min(columns - 1, Math.floor((positions[3 * v] - x0) / ARROW_SPACING));
		const k = Math.min(rows - 1, Math.floor((positions[3 * v + 2] - z0) / ARROW_SPACING));
		const cell = i * rows + k;
		if (shown[cell] < 0 || positions[3 * v + 1] < positions[3 * shown[cell] + 1]) shown[cell] = v;
	}
	const arrows = [];
	for (const v of shown) {
		if (v < 0) continue;
		const n = normals.subarray(3 * v, 3 * v + 3);
		const velocity = [1 + induced[3 * v], induced[3 * v + 1], induced[3 * v + 2]];
		const across = velocity[0] * n[0] + velocity[1] * n[1] + velocity[2] * n[2];
		arrows.push({ at: Array.from(positions.subarray(3 * v, 3 * v + 3)), velocity: velocity.map((c, k) => c - across * n[k]) });
	}
	return { pressure, arrows };
}
