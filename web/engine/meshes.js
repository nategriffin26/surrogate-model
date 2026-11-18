/**
 * The meshes a parsed surface is drawn and inspected with. Runs inside the engine's worker.
 *
 * The full mesh is the file itself: every polygon fan-triangulated and wound outward, its vertices
 * the file's own points. The working mesh is that surface simplified by edge collapse
 * (meshoptimizer, in `simplify-worker.js`), every vertex one of the file's points. Clustering
 * simplifiers are avoided on purpose: they weld nearby points across thin parts and tear seams
 * around the wheels. Values the file or the networks give per polygon reach the vertices through
 * `cellToPoint`.
 *
 * Simplification runs in parallel workers on pieces of the surface cut by a grid of cubes. Each
 * piece keeps its cut edges where they are (`LockBorder`), so neighbouring pieces still meet point
 * for point; a second round on the grid shifted by half a cube then simplifies the bands of fine
 * triangles the first round had to keep along the cuts.
 *
 * @typedef {import('./result.js').ParsedSurface} ParsedSurface
 * @typedef {import('./result.js').DisplayMesh} DisplayMesh
 */

import { boundsOf } from './geometry.js';
import { helperCount } from './pool.js';

/** Cubes along the longest side of the surface's bounding box in the grid that cuts it into jobs. */
const GRID_DIVISIONS = 48;
/** Most triangles one simplification job is given. */
const JOB_TRIANGLES = 200_000;
/**
 * Most simplification workers. Each holds about 60 MB once it has run a job, and past eight they
 * barely help: on an 18-core M5 Pro, 8 workers took 2.2 s and 16 took 2.1 s.
 */
const MAX_WORKERS = 8;
/**
 * How far an edge collapse may move the surface, as a fraction of the longest side: 0.9 mm on a
 * DrivAerML car, which leaves about 600,000 of its 17.7 million triangles. Picked from renders
 * beside the full mesh at the wheel well, mirror and grille: half of it tripled the triangles for
 * no visible gain, and more of it made the tyre grooves waver.
 */
const WORKING_TOLERANCE = 2e-4;
/**
 * How much a change in the surface normal costs a collapse, as movement in fractions of the longest
 * side per unit of change. It keeps the points that make a crease or a panel gap look sharp.
 */
const NORMAL_WEIGHT = 0.2;

/**
 * @param {ParsedSurface} surface
 * @param {{ flip: boolean, average: CellToPoint, signal?: AbortSignal }} options
 *   flip = the file's polygons are wound inward; `average` puts the file's CFD on the vertices.
 *   Aborting stops the workers and rejects with the signal's reason; no callback fires after that.
 * @param {{ preview?: (mesh: DisplayMesh) => void,
 *           full?: (mesh: DisplayMesh) => void,
 *           working?: (mesh: DisplayMesh & { sourcePoint: Uint32Array }) => void,
 *           progress?: (fraction: number) => void }} on
 *   `full` fires first, when the full mesh fits in memory. Otherwise `preview` fires with the working
 *   mesh, as the first mesh there is to draw, just before `working` does.
 * @returns {Promise<void>} once every mesh that will be built has been delivered
 *
 * `full.positions` is `surface.points` itself; it, `full.normals` and `full.cfd` are read until the
 * promise settles, `full.indices` never again.
 */
export async function buildMeshes(surface, { flip, average, signal }, on) {
  signal?.throwIfAborted();
  const workers = startWorkers();
  try {
    const { points, connectivity, offsets } = surface;
    const sourceTriangles = connectivity.length - 2 * (offsets.length - 1);
    const normals = pointNormals(surface, flip);
    const cfd = surface.cfd && average.all(surface.cfd);
    const indices = triangulate(surface, flip, sourceTriangles);
    const hasFull = indices !== null;
    if (hasFull) {
      signal?.throwIfAborted();
      on.full?.({ positions: points, normals, indices, ...(cfd && { cfd }) });
    }

    const bounds = boundsOf(points);
    const longest = Math.max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]);
    const cube = longest / GRID_DIVISIONS;
    const limits = { tolerance: WORKING_TOLERANCE * longest, normalWeight: NORMAL_WEIGHT * longest };
    const localOf = new Int32Array(points.length / 3).fill(-1);

    let simplified = 0;
    const firstRound = await simplifyRound(
      { points, normals, connectivity, offsets, flip }, { bounds, cube, shift: 0 }, limits, workers, localOf, signal,
      (triangles) => on.progress?.(0.9 * (simplified += triangles) / sourceTriangles),
    );
    const firstTriangles = firstRound.length / 3;
    simplified = 0;
    const secondRound = await simplifyRound(
      { points, normals, connectivity: firstRound, offsets: null, flip: false }, { bounds, cube, shift: cube / 2 },
      limits, workers, localOf, signal,
      (triangles) => on.progress?.(0.9 + 0.1 * (simplified += triangles) / firstTriangles),
    );

    const working = workingMesh(secondRound, points, normals, cfd, localOf);
    if (!hasFull) on.preview?.(working);
    on.working?.(working);
  } finally {
    for (const worker of workers) worker.terminate();
  }
}

/**
 * A surface drawn as it is, without simplifying: every polygon fan-triangulated and wound outward,
 * with area-weighted normals at its points. For surfaces small enough to draw whole, such as a
 * working mesh handed to `predictQuick`.
 * @param {ParsedSurface} surface
 * @param {boolean} flip  the polygons are wound inward
 * @returns {{ positions: Float32Array, normals: Float32Array, indices: Uint32Array }}
 */
export function surfaceMesh(surface, flip) {
  const indices = triangulate(surface, flip, surface.connectivity.length - 2 * (surface.offsets.length - 1));
  if (!indices) throw new RangeError('Out of memory for the surface\'s triangles.');
  return { positions: surface.points, normals: pointNormals(surface, flip), indices };
}

/**
 * @typedef {Object} CellToPoint  Per-polygon values put on points, as the meshes' `cfd` is made.
 * @property {(values: ArrayLike<number>) => Float32Array} all
 *   (C) per polygon in, (P) out: the value at every file point, which are the full mesh's vertices.
 * @property {(sourcePoint: Uint32Array) => { cells: Uint32Array, average: (values: ArrayLike<number>) => Float32Array }} around
 *   For vertices sitting on the file points `sourcePoint` (the working mesh's): `cells`, every
 *   polygon touching one of them in ascending order, and `average`, which takes `values[k]` for
 *   `cells[k]` and returns the value at each vertex.
 */

/**
 * Each point's value is the area-weighted mean of the values of the polygons around it. Build it
 * once per surface; it keeps each point's total polygon area (P floats).
 *
 * @param {ParsedSurface} surface
 * @param {ArrayLike<number>} areas  (C) polygon areas, as `measureCells` gives them
 * @returns {CellToPoint}
 */
export function cellToPoint({ points, connectivity, offsets }, areas) {
  const pointCount = points.length / 3;
  const cellCount = offsets.length - 1;
  const pointArea = new Float32Array(pointCount);
  for (let cell = 0; cell < cellCount; cell++) {
    for (let i = offsets[cell]; i < offsets[cell + 1]; i++) pointArea[connectivity[i]] += areas[cell];
  }

  const all = (values) => {
    const out = new Float32Array(pointCount);
    for (let cell = 0; cell < cellCount; cell++) {
      const weighted = areas[cell] * values[cell];
      for (let i = offsets[cell]; i < offsets[cell + 1]; i++) out[connectivity[i]] += weighted;
    }
    // A point no polygon uses keeps zero.
    for (let p = 0; p < pointCount; p++) if (pointArea[p] > 0) out[p] /= pointArea[p];
    return out;
  };

  const around = (sourcePoint) => {
    const vertexOf = new Int32Array(pointCount).fill(-1);
    sourcePoint.forEach((p, v) => { vertexOf[p] = v; });
    // Two passes over the polygons: count, then fill each (polygon, vertex) pair.
    let cellTotal = 0, pairTotal = 0;
    for (let cell = 0; cell < cellCount; cell++) {
      let touching = 0;
      for (let i = offsets[cell]; i < offsets[cell + 1]; i++) if (vertexOf[connectivity[i]] >= 0) touching++;
      if (touching > 0) cellTotal++;
      pairTotal += touching;
    }
    const cells = new Uint32Array(cellTotal);
    const pairCell = new Uint32Array(pairTotal);
    const pairVertex = new Uint32Array(pairTotal);
    let k = 0, pair = 0;
    for (let cell = 0; cell < cellCount; cell++) {
      let touching = false;
      for (let i = offsets[cell]; i < offsets[cell + 1]; i++) {
        const v = vertexOf[connectivity[i]];
        if (v < 0) continue;
        touching = true;
        pairCell[pair] = k;
        pairVertex[pair++] = v;
      }
      if (touching) cells[k++] = cell;
    }
    const average = (values) => {
      const out = new Float32Array(sourcePoint.length);
      for (let i = 0; i < pairTotal; i++) out[pairVertex[i]] += areas[cells[pairCell[i]]] * values[pairCell[i]];
      for (let v = 0; v < out.length; v++) out[v] /= pointArea[sourcePoint[v]];
      return out;
    };
    return { cells, average };
  };

  return { all, around };
}

/**
 * Unit normals at every file point, area-weighted from the polygons around it: the sum of their
 * vector areas, each the sum of its fan's cross products. Points no polygon covers keep zero.
 */
function pointNormals({ points, connectivity, offsets }, flip) {
  const normals = new Float32Array(points.length);
  const sign = flip ? -1 : 1;
  for (let cell = 0; cell + 1 < offsets.length; cell++) {
    const start = offsets[cell], end = offsets[cell + 1];
    const a = 3 * connectivity[start];
    const ax = points[a], ay = points[a + 1], az = points[a + 2];
    let b = 3 * connectivity[start + 1];
    let ux = points[b] - ax, uy = points[b + 1] - ay, uz = points[b + 2] - az;
    let nx = 0, ny = 0, nz = 0;
    for (let i = start + 2; i < end; i++) {
      b = 3 * connectivity[i];
      const vx = points[b] - ax, vy = points[b + 1] - ay, vz = points[b + 2] - az;
      nx += uy * vz - uz * vy;
      ny += uz * vx - ux * vz;
      nz += ux * vy - uy * vx;
      ux = vx; uy = vy; uz = vz;
    }
    for (let i = start; i < end; i++) {
      const p = 3 * connectivity[i];
      normals[p] += sign * nx;
      normals[p + 1] += sign * ny;
      normals[p + 2] += sign * nz;
    }
  }
  for (let p = 0; p < normals.length; p += 3) {
    const length = Math.sqrt(normals[p] ** 2 + normals[p + 1] ** 2 + normals[p + 2] ** 2);
    if (length > 0) {
      normals[p] /= length;
      normals[p + 1] /= length;
      normals[p + 2] /= length;
    }
  }
  return normals;
}

/** Every polygon as a triangle fan wound outward, or null when the device cannot hold them. */
function triangulate({ connectivity, offsets }, flip, triangleCount) {
  let indices;
  try {
    indices = new Uint32Array(3 * triangleCount);
  } catch (error) {
    if (error instanceof RangeError) return null;
    throw error;
  }
  let k = 0;
  for (let cell = 0; cell + 1 < offsets.length; cell++) {
    const start = offsets[cell], end = offsets[cell + 1];
    const apex = connectivity[start];
    for (let i = start + 1; i + 1 < end; i++) {
      indices[k++] = apex;
      indices[k++] = connectivity[flip ? i + 1 : i];
      indices[k++] = connectivity[flip ? i : i + 1];
    }
  }
  return indices;
}

/**
 * Workers for the simplification jobs, as many as the engine's helpers up to `MAX_WORKERS`. This
 * worker only hands out jobs meanwhile.
 */
function startWorkers() {
  const count = Math.min(MAX_WORKERS, helperCount());
  return Array.from({ length: count }, () =>
    new Worker(new URL('./simplify-worker.js', import.meta.url), { type: 'module' }));
}

/**
 * One round of simplification. Faces are cut into jobs by the grid cube their first point falls
 * in, every job is simplified in a worker with its cut edges kept, and the surviving triangles come
 * back in the file's point ids, job after job in a fixed order so the result never depends on
 * which worker finished first.
 *
 * @param {{ points: Float32Array, normals: Float32Array, connectivity: Uint32Array, offsets: Uint32Array | null,
 *           flip: boolean }} faces
 *   Polygons `connectivity[offsets[f] .. offsets[f + 1])`, or triangles when `offsets` is null, over
 *   the file's points and their surface normals.
 * @param {{ bounds: number[], cube: number, shift: number }} grid  cubes of edge `cube`, origin at the
 *   bounding box's minimum corner moved back by `shift` along every axis
 * @param {{ tolerance: number, normalWeight: number }} limits  in metres, for `simplify-worker.js`
 * @param {Int32Array} localOf  -1 for every point; returned that way
 * @param {AbortSignal | undefined} signal  rejects the round with its reason
 */
async function simplifyRound(faces, grid, limits, workers, localOf, signal, onJobDone) {
  const jobs = cutIntoJobs(faces, grid);
  const byTriangles = jobs.triangles.map((_, job) => job).sort((a, b) => jobs.triangles[b] - jobs.triangles[a]);
  const results = new Array(jobs.triangles.length);
  const pointIds = new Array(jobs.triangles.length);
  let sent = 0;
  let received = 0;

  await new Promise((resolve, reject) => {
    signal?.throwIfAborted();
    signal?.addEventListener('abort', () => reject(signal.reason), { once: true });
    const send = (worker) => {
      if (sent === byTriangles.length) return;
      const job = byTriangles[sent++];
      const input = jobInput(faces, jobs.order, jobs.starts[job], jobs.starts[job + 1], jobs.triangles[job], localOf);
      pointIds[job] = input.pointIds;
      const { positions, normals, indices } = input;
      worker.postMessage({ job, positions, normals, indices, ...limits }, [positions.buffer, normals.buffer, indices.buffer]);
    };
    if (byTriangles.length === 0) resolve();
    for (const worker of workers) {
      worker.onmessage = ({ data: { job, indices } }) => {
        const ids = pointIds[job];
        for (let i = 0; i < indices.length; i++) indices[i] = ids[indices[i]];
        results[job] = indices;
        pointIds[job] = null;
        onJobDone(jobs.triangles[job]);
        if (++received === byTriangles.length) resolve();
        else send(worker);
      };
      worker.onerror = (event) => {
        event.preventDefault();
        reject(new Error(`Simplifying the surface failed: ${event.message}`));
      };
      // Two jobs each, so a worker never waits on this thread for its next one.
      send(worker);
      send(worker);
    }
  });

  let length = 0;
  for (const triangles of results) length += triangles.length;
  const merged = new Uint32Array(length);
  let at = 0;
  for (const triangles of results) {
    merged.set(triangles, at);
    at += triangles.length;
  }
  return merged;
}

/**
 * Groups faces into jobs: each face goes to the grid cube of its first point, and cubes are taken
 * in Morton order (so a job stays compact in space) until a job would pass `JOB_TRIANGLES`.
 * A cube denser than that is a job on its own.
 *
 * @returns {{ order: Uint32Array, starts: Uint32Array, triangles: number[] }}  job j owns the faces
 *   `order[starts[j] .. starts[j + 1])`, `triangles[j]` triangles in all
 */
function cutIntoJobs({ points, connectivity, offsets }, { bounds, cube, shift }) {
  const faceCount = offsets ? offsets.length - 1 : connectivity.length / 3;
  const origin = [bounds[0] - shift, bounds[2] - shift, bounds[4] - shift];
  const size = [0, 1, 2].map((axis) => Math.floor((bounds[2 * axis + 1] - origin[axis]) / cube) + 1);
  const cubeOf = (face) => {
    const p = 3 * connectivity[offsets ? offsets[face] : 3 * face];
    const i = Math.min(size[0] - 1, Math.floor((points[p] - origin[0]) / cube));
    const j = Math.min(size[1] - 1, Math.floor((points[p + 1] - origin[1]) / cube));
    const k = Math.min(size[2] - 1, Math.floor((points[p + 2] - origin[2]) / cube));
    return (k * size[1] + j) * size[0] + i;
  };

  const cubeTriangles = new Float64Array(size[0] * size[1] * size[2]);
  for (let face = 0; face < faceCount; face++) {
    cubeTriangles[cubeOf(face)] += offsets ? offsets[face + 1] - offsets[face] - 2 : 1;
  }

  const occupied = [];
  for (let c = 0; c < cubeTriangles.length; c++) if (cubeTriangles[c] > 0) occupied.push(c);
  const morton = (c) => {
    const i = c % size[0], j = Math.floor(c / size[0]) % size[1], k = Math.floor(c / (size[0] * size[1]));
    return spread(i) | (spread(j) << 1) | (spread(k) << 2);
  };
  occupied.sort((a, b) => morton(a) - morton(b));

  const jobOfCube = new Int32Array(cubeTriangles.length);
  const triangles = [];
  for (const c of occupied) {
    const last = triangles.length - 1;
    if (last >= 0 && triangles[last] + cubeTriangles[c] <= JOB_TRIANGLES) {
      triangles[last] += cubeTriangles[c];
    } else {
      triangles.push(cubeTriangles[c]);
    }
    jobOfCube[c] = triangles.length - 1;
  }

  const starts = new Uint32Array(triangles.length + 1);
  const jobOf = (face) => jobOfCube[cubeOf(face)];
  for (let face = 0; face < faceCount; face++) starts[jobOf(face) + 1]++;
  for (let job = 0; job < triangles.length; job++) starts[job + 1] += starts[job];
  const next = starts.slice(0, triangles.length);
  const order = new Uint32Array(faceCount);
  for (let face = 0; face < faceCount; face++) order[next[jobOf(face)]++] = face;
  return { order, starts, triangles };
}

/** The low 10 bits of `value` spread out to every third bit, for Morton codes. */
function spread(value) {
  let x = value & 0x3ff;
  x = (x | (x << 16)) & 0x030000ff;
  x = (x | (x << 8)) & 0x0300f00f;
  x = (x | (x << 4)) & 0x030c30c3;
  x = (x | (x << 2)) & 0x09249249;
  return x;
}

/**
 * One job's faces as a compact triangle mesh: its own positions, surface normals and triangles
 * over local vertex ids, and the file point id of each local vertex.
 */
function jobInput({ points, normals: pointNormals, connectivity, offsets, flip }, order, from, to, triangleCount, localOf) {
  const indices = new Uint32Array(3 * triangleCount);
  const pointIds = new Uint32Array(triangleCount + 2 * (to - from));
  let vertexCount = 0;
  const local = (p) => {
    let v = localOf[p];
    if (v < 0) {
      v = localOf[p] = vertexCount++;
      pointIds[v] = p;
    }
    return v;
  };
  let k = 0;
  for (let n = from; n < to; n++) {
    const face = order[n];
    const start = offsets ? offsets[face] : 3 * face;
    const end = offsets ? offsets[face + 1] : start + 3;
    const apex = local(connectivity[start]);
    let previous = local(connectivity[start + 1]);
    for (let i = start + 2; i < end; i++) {
      const current = local(connectivity[i]);
      indices[k++] = apex;
      indices[k++] = flip ? current : previous;
      indices[k++] = flip ? previous : current;
      previous = current;
    }
  }
  const positions = new Float32Array(3 * vertexCount);
  const normals = new Float32Array(3 * vertexCount);
  for (let v = 0; v < vertexCount; v++) {
    const p = pointIds[v];
    for (let axis = 0; axis < 3; axis++) {
      positions[3 * v + axis] = points[3 * p + axis];
      normals[3 * v + axis] = pointNormals[3 * p + axis];
    }
    localOf[p] = -1;
  }
  return { positions, normals, indices, pointIds: pointIds.subarray(0, vertexCount) };
}

/**
 * The simplified triangles as a mesh of their own: vertices in order of first use, each carrying
 * the position, normal and CFD pressure of the file point it sits on.
 */
function workingMesh(triangles, points, normals, cfd, localOf) {
  const sourcePoint = new Uint32Array(Math.min(localOf.length, triangles.length));
  let vertexCount = 0;
  for (let i = 0; i < triangles.length; i++) {
    const p = triangles[i];
    let v = localOf[p];
    if (v < 0) {
      v = localOf[p] = vertexCount++;
      sourcePoint[v] = p;
    }
    triangles[i] = v;
  }
  const mesh = {
    positions: new Float32Array(3 * vertexCount),
    normals: new Float32Array(3 * vertexCount),
    indices: triangles,
    sourcePoint: sourcePoint.slice(0, vertexCount),
  };
  for (let v = 0; v < vertexCount; v++) {
    const p = mesh.sourcePoint[v];
    localOf[p] = -1;
    for (let axis = 0; axis < 3; axis++) {
      mesh.positions[3 * v + axis] = points[3 * p + axis];
      mesh.normals[3 * v + axis] = normals[3 * p + axis];
    }
  }
  if (cfd) mesh.cfd = Float32Array.from(mesh.sourcePoint, (p) => cfd[p]);
  return mesh;
}
