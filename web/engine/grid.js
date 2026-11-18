/**
 * Fields on regular grids around a car, and the operations the model's surface inputs are made of,
 * on the CPU: spreading what the surface's samples carry onto the nodes, and Gaussian blurs along an
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
 * Each point's weight shared among the 27 nodes around it by quadratic B-spline weights, added to
 * `sums` in float64 as the reference sums it (it casts the sums to float32 before blurring).
 * @param {Grid} grid
 * @param {Float32Array} positions  (M, 3)
 * @param {ArrayLike<number>} weights  (M)
 * @param {Float64Array} sums  (nx · ny · nz)
 */
export function spreadOnGrid(grid, positions, weights, sums) {
  const [, ny, nz] = grid.nodes;
  const spline = new Float64Array(12);
  for (let p = 0; p < weights.length; p++) {
    splineAround(spline, grid, positions[3 * p], positions[3 * p + 1], positions[3 * p + 2]);
    const weight = weights[p];
    for (let dx = 0; dx < 3; dx++) {
      for (let dy = 0; dy < 3; dy++) {
        const wxy = spline[3 + dx] * spline[6 + dy];
        const row = ((spline[0] + dx - 1) * ny + spline[1] + dy - 1) * nz + spline[2] - 1;
        for (let dz = 0; dz < 3; dz++) sums[row + dz] += weight * (wxy * spline[9 + dz]);
      }
    }
  }
}

/**
 * The node nearest (x, y, z) and the quadratic B-spline weights of it and its two neighbours, per
 * axis, into `spline` as [i, j, k, then x's weights for i − 1, i, i + 1, then y's, then z's], as
 * `features.py` `Grid.neighbours` takes them. A trilinear spread and read of a Gaussian about one
 * node wide reads a panel 16 to 33% differently by where it falls between nodes, which draws
 * contours along a sloping panel; quadratic weights cut that to 1.5 to 5%. An exact half step
 * rounds up, and the outer node is never the nearest: the grid's margin keeps it clear of the car.
 * @param {Float64Array} spline  (12)
 * @param {Grid} grid
 * @param {number} x
 * @param {number} y
 * @param {number} z
 */
export function splineAround(spline, { origin, step, nodes }, x, y, z) {
  place(spline, 0, (x - origin[0]) / step, nodes[0]);
  place(spline, 1, (y - origin[1]) / step, nodes[1]);
  place(spline, 2, (z - origin[2]) / step, nodes[2]);
}

function place(spline, axis, q, n) {
  const i = Math.min(Math.max(Math.floor(q + 0.5), 1), n - 2);
  const t = q - i;
  spline[axis] = i;
  spline[3 + 3 * axis] = 0.5 * (0.5 - t) * (0.5 - t);
  spline[4 + 3 * axis] = 0.75 - t * t;
  spline[5 + 3 * axis] = 0.5 * (0.5 + t) * (0.5 + t);
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
 * @param {Float32Array} [out]    where to write, not `field` itself; a new array if not given
 * @returns {Float32Array} out
 */
export function correlateAlong(field, nodes, axis, weights, out = new Float32Array(field.length)) {
  const n = nodes[axis];
  const stride = axis === 0 ? nodes[1] * nodes[2] : axis === 1 ? nodes[2] : 1;
  const radius = (weights.length - 1) / 2;
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
