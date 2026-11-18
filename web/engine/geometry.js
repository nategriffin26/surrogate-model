/**
 * What the reference pipeline (`surface.py`) measures on a surface, computed in typed-array loops:
 * each polygon's centre, outward normal and area, the bounds, the frontal area, and whether the
 * file's polygons are wound inward. Centres, normals and areas follow VTK's definitions, since the
 * networks were trained on features built from VTK's values. Also checks a surface handed over as
 * arrays (`checkSurface`).
 */

/**
 * @typedef {Object} CellGeometry
 * @property {Float32Array} centers   (C, 3) VTK cell centres
 * @property {Float32Array} normals   (C, 3) unit, pointing out of the car
 * @property {Float64Array} areas     (C)
 * @property {number[]} bounds        [xmin, xmax, ymin, ymax, zmin, zmax] of the centres
 * @property {boolean} flip           the file's polygons are wound inward
 */

/**
 * A surface handed over as arrays (`predictQuick`), checked as the reader checks a file's, so a
 * wrong array is named where it comes in rather than failing somewhere in the prediction.
 * @param {import('./result.js').ParsedSurface} surface
 * @returns {import('./result.js').ParsedSurface} the same arrays, with `cfd` null when left out
 */
export function checkSurface({ name, points, connectivity, offsets, cfd = null }) {
	const count = (n) => n.toLocaleString('en-US');
	if (typeof name !== 'string') throw new Error('The surface needs a name (a string).');
	if (!(points instanceof Float32Array) || points.length % 3 !== 0) {
		throw new Error("The surface's points must be a Float32Array of x, y, z for each point.");
	}
	for (let i = 0; i < points.length; i++) {
		if (!Number.isFinite(points[i]))
			throw new Error(`The surface's point ${count(Math.floor(i / 3))} has a coordinate that is not a finite number.`);
	}
	if (!(connectivity instanceof Uint32Array))
		throw new Error("The surface's connectivity must be a Uint32Array of point indices.");
	if (!(offsets instanceof Uint32Array) || offsets.length < 2 || offsets[0] !== 0) {
		throw new Error(
			"The surface's offsets must be a Uint32Array that starts at 0, with one entry more than there are polygons.",
		);
	}
	for (let c = 1; c < offsets.length; c++) {
		if (offsets[c] < offsets[c - 1] + 3)
			throw new Error(`Polygon ${count(c - 1)} of the surface has fewer than three corners.`);
	}
	const corners = offsets[offsets.length - 1];
	if (corners !== connectivity.length) {
		throw new Error(
			`The surface's offsets end at ${count(corners)}, but its connectivity holds ${count(connectivity.length)} indices.`,
		);
	}
	const pointCount = points.length / 3;
	for (let i = 0; i < corners; i++) {
		if (connectivity[i] >= pointCount) {
			throw new Error(
				`The surface's connectivity refers to point ${count(connectivity[i])}, but it has ${count(pointCount)} points.`,
			);
		}
	}
	if (cfd !== null && (!(cfd instanceof Float32Array) || cfd.length !== offsets.length - 1)) {
		throw new Error(
			`The surface's cfd must be a Float32Array with one value per polygon (${count(offsets.length - 1)}).`,
		);
	}
	return { name, points, connectivity, offsets, cfd };
}

/**
 * @param {import('./result.js').ParsedSurface} surface
 * @returns {CellGeometry}
 */
export function measureCells(surface) {
	const { points, connectivity, offsets } = surface;
	const cellCount = offsets.length - 1;
	const centers = new Float32Array(3 * cellCount);
	const normals = new Float32Array(3 * cellCount);
	const areas = new Float64Array(cellCount);
	let signedVolume = 0;

	for (let c = 0; c < cellCount; c++) {
		const begin = offsets[c];
		const corners = offsets[c + 1] - begin;
		const a = 3 * connectivity[begin];
		const ax = points[a],
			ay = points[a + 1],
			az = points[a + 2];

		// VTK's definitions: the normal is the polygon's vector area (Newell's method, which the fan of
		// triangles from the first corner adds up to); the area is that vector's length for a quad and the
		// sum of the fan's triangle areas for any other polygon (vtkCellSizeFilter).
		let fanArea = 0;
		let nx = 0,
			ny = 0,
			nz = 0;
		let sx = ax,
			sy = ay,
			sz = az;
		for (let k = 1; k < corners; k++) {
			const b = 3 * connectivity[begin + k];
			sx += points[b];
			sy += points[b + 1];
			sz += points[b + 2];
			if (k === corners - 1) break;
			const d = 3 * connectivity[begin + k + 1];
			const ux = points[b] - ax,
				uy = points[b + 1] - ay,
				uz = points[b + 2] - az;
			const vx = points[d] - ax,
				vy = points[d + 1] - ay,
				vz = points[d + 2] - az;
			const cx = uy * vz - uz * vy,
				cy = uz * vx - ux * vz,
				cz = ux * vy - uy * vx;
			nx += cx;
			ny += cy;
			nz += cz;
			fanArea += 0.5 * Math.sqrt(cx * cx + cy * cy + cz * cz);
		}
		const length = Math.sqrt(nx * nx + ny * ny + nz * nz);
		const area = corners === 4 ? 0.5 * length : fanArea;
		if (length > 0) {
			nx /= length;
			ny /= length;
			nz /= length;
		}

		let centerX, centerY, centerZ;
		if (corners <= 4) {
			centerX = sx / corners;
			centerY = sy / corners;
			centerZ = sz / corners;
		} else {
			// vtkPolygon's parametric centre: the middle of the corners' bounding rectangle in the frame
			// of the first edge and the normal.
			const b = 3 * connectivity[begin + 1];
			const ex = points[b] - ax,
				ey = points[b + 1] - ay,
				ez = points[b + 2] - az;
			const fx = ny * ez - nz * ey,
				fy = nz * ex - nx * ez,
				fz = nx * ey - ny * ex;
			const ee = ex * ex + ey * ey + ez * ez;
			const ff = fx * fx + fy * fy + fz * fz;
			let sMin = 0,
				sMax = 0,
				tMin = 0,
				tMax = 0;
			for (let k = 1; k < corners; k++) {
				const p = 3 * connectivity[begin + k];
				const px = points[p] - ax,
					py = points[p + 1] - ay,
					pz = points[p + 2] - az;
				const s = (px * ex + py * ey + pz * ez) / ee;
				const t = (px * fx + py * fy + pz * fz) / ff;
				if (s < sMin) sMin = s;
				if (s > sMax) sMax = s;
				if (t < tMin) tMin = t;
				if (t > tMax) tMax = t;
			}
			const s = 0.5 * (sMin + sMax);
			const t = 0.5 * (tMin + tMax);
			centerX = ax + s * ex + t * fx;
			centerY = ay + s * ey + t * fy;
			centerZ = az + s * ez + t * fz;
		}

		const i = 3 * c;
		centers[i] = centerX;
		centers[i + 1] = centerY;
		centers[i + 2] = centerZ;
		normals[i] = nx;
		normals[i + 1] = ny;
		normals[i + 2] = nz;
		areas[c] = area;
		// The enclosed volume by the divergence theorem, with the stored (float32) values as the
		// reference sums them: its sign says which way the polygons are wound.
		signedVolume +=
			(centers[i] * normals[i] + centers[i + 1] * normals[i + 1] + centers[i + 2] * normals[i + 2]) * area;
	}

	const flip = signedVolume < 0;
	if (flip) for (let i = 0; i < normals.length; i++) normals[i] = -normals[i];
	return { centers, normals, areas, bounds: boundsOf(centers), flip };
}

/** [xmin, xmax, ymin, ymax, zmin, zmax] of points stored x, y, z, x, y, z, ... */
export function boundsOf(xyz) {
	const bounds = [Infinity, -Infinity, Infinity, -Infinity, Infinity, -Infinity];
	for (let i = 0; i < xyz.length; i += 3) {
		for (let k = 0; k < 3; k++) {
			const value = xyz[i + k];
			if (value < bounds[2 * k]) bounds[2 * k] = value;
			if (value > bounds[2 * k + 1]) bounds[2 * k + 1] = value;
		}
	}
	return bounds;
}

const SILHOUETTE_PIXEL = 0.005;

/**
 * Projected frontal area, the reference area for the coefficients: the car's outline seen from ahead,
 * drawn on a 5 mm grid by the pixels its surface falls in, with enclosed gaps filled. Pixels are
 * marked by points on the surface: every point of it, and on polygons wider than half a pixel a
 * lattice of points half a pixel apart, so a coarse mesh draws its outline as fully as a fine one
 * (run 1's 600,000-triangle working mesh and its 17.7 million-triangle file agree within 0.01%;
 * from points alone the working mesh came out 0.5% small). The reference (`estimate_frontal_area`)
 * marks pixels with 1.5 million random samples instead, which miss some of the pixels the outline
 * only grazes: on DrivAerML cars this one is 0.2–0.3% larger, and gives the same answer every time.
 * @param {import('./result.js').ParsedSurface} surface
 */
export function frontalArea({ points, connectivity, offsets }) {
	const f32 = Math.fround;
	let ymin = Infinity,
		ymax = -Infinity,
		zmin = Infinity,
		zmax = -Infinity;
	for (let i = 0; i < points.length; i += 3) {
		const y = points[i + 1];
		const z = points[i + 2];
		if (y < ymin) ymin = y;
		if (y > ymax) ymax = y;
		if (z < zmin) zmin = z;
		if (z > zmax) zmax = z;
	}
	// One empty pixel on every side, so the outside of the outline is connected all round.
	const ny = Math.ceil((ymax - ymin) / SILHOUETTE_PIXEL) + 3;
	const nz = Math.ceil((zmax - zmin) / SILHOUETTE_PIXEL) + 3;
	const pixel = f32(SILHOUETTE_PIXEL);
	const covered = new Uint8Array(ny * nz);
	const mark = (y, z) => {
		const iy = Math.min(Math.trunc(f32(f32(y - ymin) / pixel)) + 1, ny - 1);
		const iz = Math.min(Math.trunc(f32(f32(z - zmin) / pixel)) + 1, nz - 1);
		covered[iy * nz + iz] = 1;
	};
	for (let i = 0; i < points.length; i += 3) mark(points[i + 1], points[i + 2]);

	const spacing = SILHOUETTE_PIXEL / 2;
	for (let c = 0; c + 1 < offsets.length; c++) {
		const apex = 3 * connectivity[offsets[c]];
		const ay = points[apex + 1],
			az = points[apex + 2];
		for (let k = offsets[c] + 1; k + 1 < offsets[c + 1]; k++) {
			const b = 3 * connectivity[k],
				d = 3 * connectivity[k + 1];
			const uy = points[b + 1] - ay,
				uz = points[b + 2] - az;
			const vy = points[d + 1] - ay,
				vz = points[d + 2] - az;
			const longest = Math.sqrt(Math.max(uy * uy + uz * uz, vy * vy + vz * vz, (vy - uy) ** 2 + (vz - uz) ** 2));
			if (longest < spacing) continue;
			const n = Math.ceil(longest / spacing);
			for (let i = 0; i <= n; i++) {
				for (let j = 0; i + j <= n; j++) mark(ay + (i * uy + j * vy) / n, az + (i * uz + j * vz) / n);
			}
		}
	}

	// Flood the outside from the border through empty pixels (4-connected); everything else is the car.
	const outside = new Uint8Array(ny * nz);
	const queue = new Int32Array(ny * nz);
	let head = 0;
	let tail = 0;
	const visit = (y, z) => {
		const p = y * nz + z;
		if (!covered[p] && !outside[p]) {
			outside[p] = 1;
			queue[tail++] = p;
		}
	};
	for (let y = 0; y < ny; y++) {
		visit(y, 0);
		visit(y, nz - 1);
	}
	for (let z = 0; z < nz; z++) {
		visit(0, z);
		visit(ny - 1, z);
	}
	while (head < tail) {
		const p = queue[head++];
		const y = Math.floor(p / nz);
		const z = p - y * nz;
		if (y > 0) visit(y - 1, z);
		if (y < ny - 1) visit(y + 1, z);
		if (z > 0) visit(y, z - 1);
		if (z < nz - 1) visit(y, z + 1);
	}
	return (ny * nz - tail) * SILHOUETTE_PIXEL * SILHOUETTE_PIXEL;
}
