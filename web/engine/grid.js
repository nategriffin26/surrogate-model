/**
 * Fields on regular grids around a car, and the operations the model's surface inputs are made of,
 * on the CPU: spreading what the surface's cells carry onto the nodes, and Gaussian blurs along an
 * axis. `grid-gpu.js` does the same on WebGPU. Each step follows `Grid` and `_surface_scale` in
 * `features.py` one for one.
 */

/**
 * @typedef {Object} Grid  Nodes at origin + step × (i, j, k), fields stored [i][j][k], k fastest.
 * @property {number[]} origin
 * @property {number} step
 * @property {number[]} nodes  (nx, ny, nz)
 */

/**
 * Each point's weight shared among the eight nodes around it by trilinear weights, summed in float64
 * as the reference sums it (it casts the sums to float32 before blurring). A point beyond the grid
 * goes to the nearest cube, its weights clamped.
 * @param {Grid} grid
 * @param {Float32Array} positions  (M, 3)
 * @param {ArrayLike<number>} weights  (M)
 * @returns {Float64Array}
 */
export function spreadOnGrid(grid, positions, weights) {
  const [nx, ny, nz] = grid.nodes;
  const sums = new Float64Array(nx * ny * nz);
  const cube = new Float64Array(6);
  for (let p = 0; p < weights.length; p++) {
    locate(cube, grid, positions[3 * p], positions[3 * p + 1], positions[3 * p + 2]);
    const ix = cube[0], fx = cube[1], iy = cube[2], fy = cube[3], iz = cube[4], fz = cube[5];
    const weight = weights[p];
    for (let dx = 0; dx < 2; dx++) {
      const wx = dx ? fx : 1 - fx;
      for (let dy = 0; dy < 2; dy++) {
        const wxy = wx * (dy ? fy : 1 - fy);
        const row = ((ix + dx) * ny + iy + dy) * nz + iz;
        sums[row] += weight * (wxy * (1 - fz));
        sums[row + 1] += weight * (wxy * fz);
      }
    }
  }
  return sums;
}

/**
 * The cube of nodes (x, y, z) falls in and where across it, per axis, into `cube` as [i, f, j, g,
 * k, h]. A point beyond the grid takes the nearest cube at its edge, which is how `features.py`
 * spreads and reads (base clipped, fraction clipped).
 * @param {Float64Array} cube
 * @param {Grid} grid
 * @param {number} x
 * @param {number} y
 * @param {number} z
 */
export function locate(cube, { origin, step, nodes }, x, y, z) {
  place(cube, 0, (x - origin[0]) / step, nodes[0]);
  place(cube, 1, (y - origin[1]) / step, nodes[1]);
  place(cube, 2, (z - origin[2]) / step, nodes[2]);
}

function place(cube, axis, q, n) {
  const clamped = Math.min(Math.max(q, 0), n - 1);
  const i = Math.min(Math.floor(clamped), n - 2);
  cube[2 * axis] = i;
  cube[2 * axis + 1] = clamped - i;
}

/**
 * The normalised Gaussian the surface scales blur with, as `features.py` `_gaussian_weights` and
 * SciPy's `gaussian_filter1d` make it: exp(−(k / σ)² / 2) for k from −r to r, r = ⌊truncate σ + ½⌋,
 * summing to 1. `sigma` is in nodes.
 * @param {number} sigma
 * @param {number} truncate
 * @returns {Float64Array}
 */
export function gaussianWeights(sigma, truncate) {
  const radius = Math.floor(truncate * sigma + 0.5);
  const weights = new Float64Array(2 * radius + 1);
  for (let k = -radius; k <= radius; k++) weights[k + radius] = Math.exp(-0.5 * (k / sigma) ** 2);
  const total = weights.reduce((sum, w) => sum + w, 0);
  return weights.map((w) => w / total);
}

/**
 * The Gaussian kept on its upstream side: nodes at smaller x count fully, the node itself by half,
 * nodes downstream not at all (`_upstream_half`).
 * @param {Float64Array} weights  from `gaussianWeights`
 * @returns {Float64Array}
 */
export function upstreamHalf(weights) {
  const radius = (weights.length - 1) / 2;
  return weights.map((w, index) => (index < radius ? w : index === radius ? 0.5 * w : 0));
}

/**
 * out[i] = Σ_k weights[k + r] field[i + k] along one axis, SciPy's `correlate1d` with the field
 * reflected at its ends (d c b a | a b c d), summed in float64.
 * @param {Float32Array} field
 * @param {number[]} nodes
 * @param {number} axis
 * @param {Float64Array} weights  2r + 1 of them
 * @returns {Float32Array}
 */
export function correlateAlong(field, nodes, axis, weights) {
  const n = nodes[axis];
  const stride = axis === 0 ? nodes[1] * nodes[2] : axis === 1 ? nodes[2] : 1;
  const radius = (weights.length - 1) / 2;
  const out = new Float32Array(field.length);
  const reflected = (index) => {
    const period = 2 * n;
    const m = ((index % period) + period) % period;
    return m < n ? m : period - 1 - m;
  };
  for (let node = 0; node < field.length; node++) {
    const index = Math.floor(node / stride) % n;
    const base = node - index * stride;
    let sum = 0;
    for (let k = -radius; k <= radius; k++) sum += weights[k + radius] * field[base + reflected(index + k) * stride];
    out[node] = sum;
  }
  return out;
}
