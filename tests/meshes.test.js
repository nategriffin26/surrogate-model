// Tests for web/engine/meshes.js on a small synthetic car: an ellipsoid body and a torus wheel beside it.
// Run with: deno test --allow-read tests/meshes.test.js
import { buildMeshes, cellToPoint } from '../web/engine/meshes.js';

/** Quads over a closed grid surface: `rows` x `columns` points wrapping in both directions. */
function wrappedGrid(rows, columns, pointAt, firstPoint) {
  const points = [], quads = [];
  for (let i = 0; i < rows; i++) for (let j = 0; j < columns; j++) points.push(...pointAt(i / rows, j / columns));
  const id = (i, j) => firstPoint + (i % rows) * columns + (j % columns);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < columns; j++) quads.push([id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)]);
  }
  return { points, quads };
}

/**
 * A closed ellipsoid body 4.4 m long (a cube's faces subdivided into `n` x `n` quads and pushed out
 * onto it) and a torus wheel of minor radius 0.1 m beside it, every quad wound outward.
 */
function syntheticCar(n) {
  const points = [], quads = [];
  const ids = new Map();
  const point = (x, y, z) => {
    const key = `${x},${y},${z}`;
    if (!ids.has(key)) {
      const length = Math.hypot(x, y, z);
      ids.set(key, points.length / 3);
      points.push((2.2 * x) / length, (1.0 * y) / length, (0.7 * z) / length);
    }
    return ids.get(key);
  };
  // Each face of the cube [-n/2, n/2]^3, its corners listed counter-clockwise seen from outside.
  const h = n / 2;
  for (const [axis, sign] of [[0, 1], [0, -1], [1, 1], [1, -1], [2, 1], [2, -1]]) {
    const u = (axis + 1) % 3, v = (axis + 2) % 3;
    const at = (a, b) => {
      const c = [0, 0, 0];
      c[axis] = sign * h;
      c[u] = a - h;
      c[v] = b - h;
      return point(...c);
    };
    for (let a = 0; a < n; a++) {
      for (let b = 0; b < n; b++) {
        const quad = [at(a, b), at(a + 1, b), at(a + 1, b + 1), at(a, b + 1)];
        quads.push(sign > 0 ? quad : quad.reverse());
      }
    }
  }
  const bodyPoints = points.length / 3;
  const wheel = wrappedGrid(300, 100, (s, t) => {
    const around = 2 * Math.PI * s, tube = 2 * Math.PI * t;
    const r = 0.3 + 0.1 * Math.cos(tube);
    return [1.0 + r * Math.cos(around), -1.25 - 0.1 * Math.sin(tube), -0.4 + r * Math.sin(around)];
  }, bodyPoints);
  points.push(...wheel.points);
  quads.push(...wheel.quads);

  const connectivity = Uint32Array.from(quads.flat());
  const offsets = Uint32Array.from({ length: quads.length + 1 }, (_, c) => 4 * c);
  const vertices = new Float32Array(points);
  // A smooth stand-in for CFD pressure: the height of each polygon's centre.
  const cfd = Float32Array.from(quads, (q) => q.reduce((sum, p) => sum + vertices[3 * p + 2], 0) / 4);
  return { surface: { name: 'synthetic', points: vertices, connectivity, offsets, cfd }, bodyPoints };
}

function polygonAreas({ points, connectivity, offsets }) {
  const areas = new Float64Array(offsets.length - 1);
  for (let c = 0; c < areas.length; c++) {
    const corner = (k) => [0, 1, 2].map((axis) => points[3 * connectivity[offsets[c] + k] + axis]);
    const [a, b, , d] = [0, 1, 2, 3].map(corner);
    const e = corner(2);
    const u = e.map((x, i) => x - a[i]), w = d.map((x, i) => x - b[i]);
    areas[c] = 0.5 * Math.hypot(u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]);
  }
  return areas;
}

async function build(surface, flip = false) {
  const average = cellToPoint(surface, polygonAreas(surface));
  const delivered = { preview: [], full: [], working: [] };
  await buildMeshes(surface, { flip, average }, {
    preview: (mesh) => delivered.preview.push(mesh),
    full: (mesh) => delivered.full.push(mesh),
    working: (mesh) => delivered.working.push(mesh),
  });
  return { ...delivered, average };
}

function signedVolume(positions, indices) {
  let volume = 0;
  for (let t = 0; t < indices.length; t += 3) {
    const [a, b, c] = [0, 1, 2].map((k) => [0, 1, 2].map((axis) => positions[3 * indices[t + k] + axis]));
    volume += (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6;
  }
  return volume;
}

function edgeUse(indices) {
  const use = new Map();
  for (let t = 0; t < indices.length; t += 3) {
    for (let k = 0; k < 3; k++) {
      const a = indices[t + k], b = indices[t + (k + 1) % 3];
      const key = a < b ? `${a},${b}` : `${b},${a}`;
      use.set(key, (use.get(key) ?? 0) + 1);
    }
  }
  return use;
}

/** Connected components of the triangles, each as the set of its vertices. */
function components(indices, vertexCount) {
  const parent = Int32Array.from({ length: vertexCount }, (_, v) => v);
  const root = (v) => (parent[v] === v ? v : (parent[v] = root(parent[v])));
  for (let t = 0; t < indices.length; t += 3) {
    parent[root(indices[t + 1])] = root(indices[t]);
    parent[root(indices[t + 2])] = root(indices[t]);
  }
  const groups = new Map();
  for (let t = 0; t < indices.length; t++) {
    const r = root(indices[t]);
    if (!groups.has(r)) groups.set(r, new Set());
    groups.get(r).add(indices[t]);
  }
  return [...groups.values()];
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

const { surface, bodyPoints } = syntheticCar(220);
const triangleCount = surface.connectivity.length - 2 * (surface.offsets.length - 1);

Deno.test('the full mesh is every polygon, wound outward, with unit normals', async () => {
  const { full, preview } = await build(surface);
  assert(full.length === 1 && preview.length === 0, 'the full mesh is delivered once, with no separate preview');
  const { positions, normals, indices } = full[0];
  assert(indices.length === 3 * triangleCount, `expected ${triangleCount} triangles, got ${indices.length / 3}`);
  assert(positions === surface.points, 'the full mesh shares the parsed points');
  const wheelStart = 3 * (triangleCount - 2 * 300 * 100);
  const body = signedVolume(positions, indices.subarray(0, wheelStart));
  const wheel = signedVolume(positions, indices.subarray(wheelStart));
  assert(body > 0 && wheel > 0, `signed volumes ${body} (body) and ${wheel} (wheel) should be positive`);
  for (let p = 0; p < normals.length; p += 3) {
    assert(Math.abs(Math.hypot(normals[p], normals[p + 1], normals[p + 2]) - 1) < 1e-5, `normal ${p / 3} is not unit length`);
  }
});

Deno.test('inward polygons with flip give the same outward full mesh', async () => {
  const inward = { ...surface, connectivity: surface.connectivity.slice() };
  for (let c = 0; c + 1 < inward.offsets.length; c++) inward.connectivity.subarray(inward.offsets[c], inward.offsets[c + 1]).reverse();
  const { full } = await build(inward, true);
  const { full: reference } = await build(surface);
  for (let i = 0; i < full[0].normals.length; i++) {
    assert(Math.abs(full[0].normals[i] - reference[0].normals[i]) < 1e-6, 'flipped normals differ');
  }
});

Deno.test('the working mesh keeps the wheel and opens no seam', async () => {
  const { working, full } = await build(surface);
  const mesh = working[0];
  const { positions, indices, sourcePoint } = mesh;
  // Simplified enough that the pieces' cut edges were collapsed against each other (this surface makes six jobs).
  assert(indices.length < 0.5 * full[0].indices.length, `only simplified to ${indices.length / 3} triangles`);

  // A closed surface stays closed: an edge used by one triangle is a crack between pieces.
  const open = [...edgeUse(indices).entries()].filter(([, count]) => count !== 2);
  assert(open.length === 0, `${open.length} edges are not shared by exactly two triangles, e.g. ${open.slice(0, 3)}`);

  // The body and the wheel are still the two separate parts, and the wheel keeps its size.
  const parts = components(indices, positions.length / 3);
  assert(parts.length === 2, `expected body and wheel, got ${parts.length} parts`);
  const wheel = parts.find((part) => sourcePoint[part.values().next().value] >= bodyPoints);
  assert(wheel, 'the wheel is gone');
  const extent = [0, 1, 2].map((axis) => {
    const values = [...wheel].map((v) => positions[3 * v + axis]);
    return Math.max(...values) - Math.min(...values);
  });
  for (const [axis, expected] of [[0, 0.8], [1, 0.2], [2, 0.8]]) {
    assert(Math.abs(extent[axis] - expected) < 0.002, `wheel extent on axis ${axis} is ${extent[axis]}, not ${expected}`);
  }
  // Still a ring with its hole (Euler characteristic 0) and the volume of the fine torus.
  const wheelTriangles = [];
  for (let t = 0; t < indices.length; t += 3) if (wheel.has(indices[t])) wheelTriangles.push(indices[t], indices[t + 1], indices[t + 2]);
  const euler = wheel.size - edgeUse(wheelTriangles).size + wheelTriangles.length / 3;
  assert(euler === 0, `the wheel's Euler characteristic is ${euler}, so its hole closed or it tore`);
  const fineWheel = signedVolume(surface.points, full[0].indices.subarray(3 * (triangleCount - 2 * 300 * 100)));
  const wheelVolume = signedVolume(positions, wheelTriangles);
  assert(Math.abs(wheelVolume / fineWheel - 1) < 0.01, `the wheel's volume is ${wheelVolume}, the file's ${fineWheel}`);

  for (let v = 0; v < sourcePoint.length; v++) {
    for (let axis = 0; axis < 3; axis++) {
      assert(positions[3 * v + axis] === surface.points[3 * sourcePoint[v] + axis], `vertex ${v} is not on its file point`);
    }
  }
  const { normals } = mesh;
  for (let t = 0; t < indices.length; t += 3) {
    const [a, b, c] = [0, 1, 2].map((k) => 3 * indices[t + k]);
    const u = [0, 1, 2].map((i) => positions[b + i] - positions[a + i]);
    const w = [0, 1, 2].map((i) => positions[c + i] - positions[a + i]);
    const n = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
    const facing = [0, 1, 2].reduce((sum, i) => sum + n[i] * (normals[a + i] + normals[b + i] + normals[c + i]), 0);
    assert(facing > 0, `triangle ${t / 3} is folded over`);
  }
});

Deno.test('the working mesh is the same whatever the number of workers', async () => {
  const { working: many } = await build(surface);
  Object.defineProperty(navigator, 'hardwareConcurrency', { value: 2, configurable: true });
  try {
    const { working: one } = await build(surface);
    assert(one[0].indices.length === many[0].indices.length && one[0].indices.every((v, i) => v === many[0].indices[i]),
      'one worker and many gave different working meshes');
  } finally {
    delete navigator.hardwareConcurrency;
  }
});

Deno.test('a device that cannot hold the full mesh gets the working mesh as its preview', async () => {
  const Native = globalThis.Uint32Array;
  globalThis.Uint32Array = class extends Native {
    constructor(...args) {
      if (typeof args[0] === 'number' && args[0] >= 3 * triangleCount) throw new RangeError('Array buffer allocation failed');
      super(...args);
    }
  };
  try {
    const { preview, full, working } = await build(surface);
    assert(full.length === 0, 'the full mesh was delivered');
    assert(working.length === 1 && preview.length === 1 && preview[0] === working[0], 'the working mesh is not the preview');
  } finally {
    globalThis.Uint32Array = Native;
  }
});

Deno.test('polygon values reach points as the area-weighted mean of the polygons around them', async () => {
  const { full, working, average } = await build(surface);
  const cfd = full[0].cfd;
  // The stand-in pressure is the height of each polygon's centre, so at a point it is close to the point's height.
  for (let p = 0; p < bodyPoints; p += 97) {
    assert(Math.abs(cfd[p] - surface.points[3 * p + 2]) < 0.01, `cfd at point ${p} is ${cfd[p]}, height ${surface.points[3 * p + 2]}`);
  }
  const { sourcePoint } = working[0];
  const near = average.around(sourcePoint);
  const atVertices = near.average(Float32Array.from(near.cells, (c) => surface.cfd[c]));
  for (let v = 0; v < sourcePoint.length; v++) {
    assert(Math.abs(atVertices[v] - cfd[sourcePoint[v]]) < 1e-6, `around() and all() disagree at vertex ${v}`);
    assert(working[0].cfd[v] === cfd[sourcePoint[v]], `working cfd at vertex ${v} is not the full mesh's`);
  }
});

Deno.test('aborting stops the build at once: it rejects with the reason and reports nothing more', async () => {
  const controller = new AbortController();
  const reason = new DOMException('Another file was dropped.', 'AbortError');
  const delivered = [];
  const building = buildMeshes(surface, { flip: false, average: cellToPoint(surface, polygonAreas(surface)), signal: controller.signal }, {
    full: () => {
      delivered.push('full');
      setTimeout(() => controller.abort(reason));
    },
    working: () => delivered.push('working'),
    progress: () => {
      if (controller.signal.aborted) delivered.push('progress');
    },
  });
  const outcome = await building.then(() => 'resolved', (error) => error);
  assert(outcome === reason, `expected the abort reason, got ${outcome}`);
  assert(delivered.join() === 'full', `delivered ${delivered.join()}`);
});
