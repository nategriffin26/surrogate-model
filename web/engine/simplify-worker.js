/**
 * Simplifies one piece of the surface by edge collapse for `meshes.js`. Edges on the piece's border
 * stay where they are, so the piece still meets its neighbours exactly.
 *
 * The surface normals at the points ride along as attributes, so the simplifier keeps the points
 * that shape a crease or a panel gap and the piece still shades like the file.
 *
 * meshoptimizer rejects a collapse that turns a triangle over by more than about 75 degrees, but a
 * run of smaller turns can still leave a triangle folded over (facing into the car, or doubled back
 * on a neighbour), and collapsing two close sheets can pinch them together along an edge. Most
 * folds are undone by flipping an edge they share with a neighbour; for any defect that remains the
 * piece is simplified again with the points around it pinned in place.
 */
import { MeshoptSimplifier } from '../vendor/meshoptimizer/meshopt_simplifier.js';

/**
 * `RegularizeLight` keeps triangles of even size and shape: without it flat panels end up as long
 * slivers, across which the pressure fields interpolate badly, and many more triangles fold over.
 */
const OPTIONS = ['LockBorder', 'ErrorAbsolute', 'RegularizeLight'];

self.onmessage = ({ data: { job, positions, normals, indices, tolerance, normalWeight } }) => {
	MeshoptSimplifier.ready
		.then(() => {
			const kept = simplifyPiece(positions, normals, indices, tolerance, normalWeight);
			self.postMessage({ job, indices: kept }, [kept.buffer]);
		})
		.catch(reportError);
};

/**
 * @param {number} tolerance     how far, in metres, a collapse may move the surface
 * @param {number} normalWeight  metres of movement one unit of change in the normal counts as
 */
function simplifyPiece(positions, normals, indices, tolerance, normalWeight) {
	// meshoptimizer measures attributes against positions scaled to the piece's extent.
	const weight = normalWeight / extent(positions);
	const weights = [weight, weight, weight];
	const pinned = new Uint8Array(positions.length / 3);
	const border = borderPoints(indices, pinned.length);
	// A fold the file itself has is part of the surface, not something simplification made
	// (`pinchedTriangles` does the same for pinched edges).
	const ownFolds = new Set(foldedTriangles(indices, positions, normals).map((t) => triangleKey(indices, t)));
	for (;;) {
		const [kept] = MeshoptSimplifier.simplifyWithAttributes(
			indices,
			positions,
			3,
			normals,
			3,
			weights,
			pinned,
			0,
			tolerance,
			OPTIONS,
		);
		unfold(kept, positions, normals, border);
		const defects = [
			...foldedTriangles(kept, positions, normals).filter((t) => !ownFolds.has(triangleKey(kept, t))),
			...pinchedTriangles(kept, indices, pinned.length),
		];
		if (defects.length === 0) return kept;
		if (!pinAround(defects, kept, positions, pinned, tolerance))
			throw new Error('A fully pinned piece came back folded.');
	}
}

/**
 * Offsets into `triangles` of the triangles on an edge that more than two triangles share, where
 * the piece as it came in (`original`) had no such edge.
 */
function pinchedTriangles(triangles, original, pointCount) {
	const edgeKey = (a, b) => (a < b ? a * pointCount + b : b * pointCount + a);
	const sharing = new Map();
	for (let t = 0; t < triangles.length; t += 3) {
		for (let corner = 0; corner < 3; corner++) {
			const key = edgeKey(triangles[t + corner], triangles[t + ((corner + 1) % 3)]);
			sharing.set(key, (sharing.get(key) ?? 0) + 1);
		}
	}
	// Each edge more than two triangles share, with how many shared it in the original.
	const pinchedEdges = new Map();
	for (const [key, count] of sharing) if (count > 2) pinchedEdges.set(key, 0);
	if (pinchedEdges.size === 0) return [];
	for (let t = 0; t < original.length; t += 3) {
		for (let corner = 0; corner < 3; corner++) {
			const key = edgeKey(original[t + corner], original[t + ((corner + 1) % 3)]);
			if (pinchedEdges.has(key)) pinchedEdges.set(key, pinchedEdges.get(key) + 1);
		}
	}
	const pinched = [];
	for (let t = 0; t < triangles.length; t += 3) {
		for (let corner = 0; corner < 3; corner++) {
			const before = pinchedEdges.get(edgeKey(triangles[t + corner], triangles[t + ((corner + 1) % 3)]));
			if (before !== undefined && before <= 2) {
				pinched.push(t);
				break;
			}
		}
	}
	return pinched;
}

/**
 * Which points lie on the piece's border. Around an inner point the triangles close into a ring,
 * so summing (next neighbour - previous neighbour) over them telescopes to zero; around a border
 * point the ring is open and the sum is the difference of its two ends.
 */
function borderPoints(triangles, pointCount) {
	const balance = new Float64Array(pointCount);
	for (let t = 0; t < triangles.length; t += 3) {
		const a = triangles[t],
			b = triangles[t + 1],
			c = triangles[t + 2];
		balance[a] += b - c;
		balance[b] += c - a;
		balance[c] += a - b;
	}
	return balance.map((sum) => (sum === 0 ? 0 : 1));
}

function extent(positions) {
	const low = [Infinity, Infinity, Infinity],
		high = [-Infinity, -Infinity, -Infinity];
	for (let i = 0; i < positions.length; i += 3) {
		for (let axis = 0; axis < 3; axis++) {
			low[axis] = Math.min(low[axis], positions[i + axis]);
			high[axis] = Math.max(high[axis], positions[i + axis]);
		}
	}
	return Math.max(high[0] - low[0], high[1] - low[1], high[2] - low[2]);
}

/** How far triangle (a, b, c) faces along the surface normals at its corners; zero or less is a fold. */
function facing(positions, normals, a, b, c) {
	a *= 3;
	b *= 3;
	c *= 3;
	const ux = positions[b] - positions[a],
		uy = positions[b + 1] - positions[a + 1],
		uz = positions[b + 2] - positions[a + 2];
	const vx = positions[c] - positions[a],
		vy = positions[c + 1] - positions[a + 1],
		vz = positions[c + 2] - positions[a + 2];
	return (
		(uy * vz - uz * vy) * (normals[a] + normals[b] + normals[c]) +
		(uz * vx - ux * vz) * (normals[a + 1] + normals[b + 1] + normals[c + 1]) +
		(ux * vy - uy * vx) * (normals[a + 2] + normals[b + 2] + normals[c + 2])
	);
}

/** Offsets into `triangles` of the folded triangles. */
function foldedTriangles(triangles, positions, normals) {
	const folded = [];
	for (let t = 0; t < triangles.length; t += 3) {
		if (facing(positions, normals, triangles[t], triangles[t + 1], triangles[t + 2]) <= 0) folded.push(t);
	}
	return folded;
}

function triangleKey(triangles, t) {
	return [triangles[t], triangles[t + 1], triangles[t + 2]].sort((a, b) => a - b).join();
}

/**
 * Flips edges of folded triangles while that helps. A flip is taken only when both triangles it
 * makes face outward and their new shared edge is not in the mesh yet, so every flip removes a
 * fold and the loop ends. The new edge never joins two border points: the neighbouring piece may
 * join them too. Folds are few, so neighbours are found by scanning the triangles.
 */
function unfold(triangles, positions, normals, border) {
	const isFolded = (t) => facing(positions, normals, triangles[t], triangles[t + 1], triangles[t + 2]) <= 0;
	const onEdge = (a, b) => {
		const found = [];
		for (let t = 0; t < triangles.length; t += 3) {
			const hasA = triangles[t] === a || triangles[t + 1] === a || triangles[t + 2] === a;
			if (hasA && (triangles[t] === b || triangles[t + 1] === b || triangles[t + 2] === b)) found.push(t);
		}
		return found;
	};

	// Triangle t = (u, v, w) and its neighbour s = (v, u, o) across edge uv become (u, o, w) and (o, v, w).
	const flipAcross = (t) => {
		for (let corner = 0; corner < 3; corner++) {
			const u = triangles[t + corner],
				v = triangles[t + ((corner + 1) % 3)],
				w = triangles[t + ((corner + 2) % 3)];
			const pair = onEdge(u, v);
			if (pair.length !== 2) continue;
			const s = pair[0] === t ? pair[1] : pair[0];
			const at = triangles[s] === v ? 0 : triangles[s + 1] === v ? 1 : 2;
			if (triangles[s + ((at + 1) % 3)] !== u) continue;
			const o = triangles[s + ((at + 2) % 3)];
			if (o === w || (border[w] && border[o])) continue;
			if (facing(positions, normals, u, o, w) <= 0 || facing(positions, normals, o, v, w) <= 0) continue;
			if (onEdge(w, o).length > 0) continue;
			triangles[t] = u;
			triangles[t + 1] = o;
			triangles[t + 2] = w;
			triangles[s] = o;
			triangles[s + 1] = v;
			triangles[s + 2] = w;
			return true;
		}
		return false;
	};

	const folded = foldedTriangles(triangles, positions, normals);
	let flipped = folded.length > 0;
	while (flipped) {
		flipped = false;
		for (const t of folded) {
			if (isFolded(t) && flipAcross(t)) flipped = true;
		}
	}
}

/**
 * Pins every point within reach of each fold: the ball around the fold that reaches its farthest
 * corner (at least `tolerance`), doubled until it takes in a point not pinned yet. Pinning only
 * grows, so the retries end: at worst the whole piece is pinned and comes back as it went in.
 * Returns false when every point was pinned already.
 */
function pinAround(folds, triangles, positions, pinned, tolerance) {
	if (pinned.every(Boolean)) return false;
	for (let scale = 1; ; scale *= 2) {
		let added = false;
		for (const t of folds) {
			const corners = [3 * triangles[t], 3 * triangles[t + 1], 3 * triangles[t + 2]];
			const center = [0, 1, 2].map((axis) => corners.reduce((sum, c) => sum + positions[c + axis], 0) / 3);
			const distance2 = (p) =>
				(positions[p] - center[0]) ** 2 + (positions[p + 1] - center[1]) ** 2 + (positions[p + 2] - center[2]) ** 2;
			const reach2 = scale * scale * Math.max(tolerance * tolerance, ...corners.map(distance2));
			for (let v = 0; v < pinned.length; v++) {
				if (!pinned[v] && distance2(3 * v) <= reach2) {
					pinned[v] = 1;
					added = true;
				}
			}
		}
		if (added) return true;
	}
}
