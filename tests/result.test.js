// Round trip of web/engine/result.js's pack format on a synthetic mesh.
// Run with: deno test --allow-read tests/result.test.js
import { packResult, unpackResult } from '../web/engine/result.js';

/** A wavy 3 m x 1 m sheet on a 1 cm grid, with smooth pressure-like fields on its vertices. */
function wavySheet(withCfd) {
  const columns = 301, rows = 101;
  const positions = new Float32Array(3 * columns * rows);
  const normals = new Float32Array(3 * columns * rows);
  const cp = new Float32Array(columns * rows), sigma = new Float32Array(columns * rows), cfd = new Float32Array(columns * rows);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < columns; j++) {
      const v = i * columns + j, x = j / 100, y = i / 100;
      const z = 0.1 * Math.sin(3 * x) * Math.cos(2 * y);
      positions.set([x - 0.7, y - 0.5, z], 3 * v);
      const dx = 0.3 * Math.cos(3 * x) * Math.cos(2 * y), dy = -0.2 * Math.sin(3 * x) * Math.sin(2 * y);
      const length = Math.hypot(dx, dy, 1);
      normals.set([-dx / length, -dy / length, 1 / length], 3 * v);
      cp[v] = 1.0 - 3.5 * Math.sin(x) ** 2 * Math.cos(y);
      sigma[v] = 0.2 + 0.2 * Math.sin(5 * x + y);
      cfd[v] = cp[v] + 0.05 * Math.cos(7 * y);
    }
  }
  const indices = [];
  for (let i = 0; i + 1 < rows; i++) {
    for (let j = 0; j + 1 < columns; j++) {
      const a = i * columns + j;
      indices.push(a, a + 1, a + columns + 1, a, a + columns + 1, a + columns);
    }
  }
  const mesh = { sourceTriangles: 1234567, positions, normals, indices: Uint32Array.from(indices), cp, sigma };
  if (withCfd) mesh.cfd = cfd;
  return { source: { name: 'sheet.vtp', hasCfd: withCfd }, forces: { cdP: 0.27 }, mesh };
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

/** Packs and unpacks `result`; returns the copy and, for each of its vertices, the original vertex. */
async function roundTrip(result) {
  const { json, block } = await packResult(result);
  const back = await unpackResult(JSON.parse(json), block.buffer);
  // Vertices sit on a 1 cm grid in x and y, so x and y to the nearest millimetre name each one.
  const key = (positions, v) => [0, 1].map((axis) => Math.round(positions[3 * v + axis] * 1000)).join();
  const original = new Map();
  for (let v = 0; v < result.mesh.positions.length / 3; v++) original.set(key(result.mesh.positions, v), v);
  const match = Array.from({ length: back.mesh.positions.length / 3 }, (_, v) => original.get(key(back.mesh.positions, v)));
  return { back, match, bytes: block.length };
}

Deno.test('a packed result comes back within the stated precision', async () => {
  const result = wavySheet(true);
  const { back, match, bytes } = await roundTrip(result);
  const raw = ['positions', 'normals', 'indices', 'cp', 'sigma', 'cfd'].reduce((sum, k) => sum + result.mesh[k].byteLength, 0);
  assert(bytes < raw / 4, `packed to ${bytes} bytes from ${raw}`);
  assert(back.forces.cdP === 0.27 && back.source.name === 'sheet.vtp' && back.mesh.sourceTriangles === 1234567, 'summary changed');
  assert(new Set(match).size === match.length && !match.includes(undefined), 'vertices do not map one to one');

  for (let v = 0; v < match.length; v++) {
    const o = match[v];
    for (let axis = 0; axis < 3; axis++) {
      const error = Math.abs(back.mesh.positions[3 * v + axis] - result.mesh.positions[3 * o + axis]);
      assert(error < 1e-4, `vertex ${v} moved ${error} m`);
    }
    const dot = [0, 1, 2].reduce((sum, axis) => sum + back.mesh.normals[3 * v + axis] * result.mesh.normals[3 * o + axis], 0);
    assert(dot > Math.cos((0.1 * Math.PI) / 180), `normal ${v} turned ${(Math.acos(Math.min(dot, 1)) * 180) / Math.PI} degrees`);
    for (const field of ['cp', 'sigma', 'cfd']) {
      const error = Math.abs(back.mesh[field][v] - result.mesh[field][o]);
      assert(error < 0.002, `${field} at vertex ${v} is off by ${error}`);
    }
  }

  // The same triangles, each still wound the same way round.
  const canonical = (a, b, c) => {
    const turn = a < b && a < c ? [a, b, c] : b < c ? [b, c, a] : [c, a, b];
    return turn.join();
  };
  const triangles = new Set();
  for (let t = 0; t < result.mesh.indices.length; t += 3) triangles.add(canonical(...result.mesh.indices.subarray(t, t + 3)));
  assert(back.mesh.indices.length === result.mesh.indices.length, 'triangle count changed');
  for (let t = 0; t < back.mesh.indices.length; t += 3) {
    const [a, b, c] = back.mesh.indices.subarray(t, t + 3);
    assert(triangles.has(canonical(match[a], match[b], match[c])), `triangle ${t / 3} is not one of the original's`);
  }
});

Deno.test('a result without CFD comes back without CFD', async () => {
  const { back } = await roundTrip(wavySheet(false));
  assert(!('cfd' in back.mesh) && back.mesh.cp && back.mesh.sigma, 'fields changed');
});
