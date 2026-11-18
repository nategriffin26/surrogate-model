/**
 * Fields on regular grids around a car, and the operations the model's flow and surface inputs are
 * made of, on the CPU: spreading what the surface's cells carry onto the nodes, the potential of
 * sources on a grid and their mirror image under the ground (an FFT convolution), Gaussian blurs
 * along an axis, and reading a field or its gradient back at points. `grid-gpu.js` does the same on
 * WebGPU. Each step follows `Grid`, `_potential` and `_surface_scale` in `features.py` one for one.
 */

import { linePlan, transformLine } from './fft.js';

/**
 * @typedef {Object} Grid  Nodes at origin + step × (i, j, k), fields stored [i][j][k], k fastest.
 * @property {number[]} origin
 * @property {number} step
 * @property {number[]} nodes  (nx, ny, nz)
 */

/**
 * Each point's weight shared among the eight nodes around it by trilinear weights, summed in float64
 * as the reference sums it (the surface casts the sums to float32; the flow adds the mirror image
 * first). A point beyond the grid goes to the nearest cube, its weights clamped.
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
 * k, h]. A point beyond the grid takes the nearest cube at its edge, which is both how `features.py`
 * spreads (base clipped, fraction clipped) and how SciPy's `map_coordinates(mode='nearest')` reads.
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

const CUBE = new Float64Array(6);

/**
 * The gradient of `field` at (x, y, z) into out[at .. at + 3]: NumPy's `gradient(field, step)` in
 * float32, central differences inside and one-sided at the faces, each component read trilinearly as
 * SciPy's `map_coordinates(order=1, mode='nearest')` reads a float32 field.
 * @param {Grid} grid
 * @param {Float32Array} field
 * @param {number} x
 * @param {number} y
 * @param {number} z
 * @param {Float32Array | Float64Array} out
 * @param {number} at
 */
export function gradientAt(grid, field, x, y, z, out, at) {
  const [nx, ny, nz] = grid.nodes;
  const step = Math.fround(grid.step);
  const twoSteps = Math.fround(2 * grid.step);
  locate(CUBE, grid, x, y, z);
  const ix = CUBE[0], fx = CUBE[1], iy = CUBE[2], fy = CUBE[3], iz = CUBE[4], fz = CUBE[5];
  let gx = 0, gy = 0, gz = 0;
  for (let dx = 0; dx < 2; dx++) {
    for (let dy = 0; dy < 2; dy++) {
      for (let dz = 0; dz < 2; dz++) {
        const w = (dx ? fx : 1 - fx) * (dy ? fy : 1 - fy) * (dz ? fz : 1 - fz);
        const i = ix + dx, j = iy + dy, k = iz + dz;
        const node = (i * ny + j) * nz + k;
        gx += w * derivative(field, node, i, nx, ny * nz, step, twoSteps);
        gy += w * derivative(field, node, j, ny, nz, step, twoSteps);
        gz += w * derivative(field, node, k, nz, 1, step, twoSteps);
      }
    }
  }
  out[at] = Math.fround(gx);
  out[at + 1] = Math.fround(gy);
  out[at + 2] = Math.fround(gz);
}

/** One component of NumPy's float32 `gradient` at a node with `index` of `n` along an axis of `stride`. */
function derivative(field, node, index, n, stride, step, twoSteps) {
  if (index === 0) return Math.fround(Math.fround(field[node + stride] - field[node]) / step);
  if (index === n - 1) return Math.fround(Math.fround(field[node] - field[node - stride]) / step);
  return Math.fround(Math.fround(field[node + stride] - field[node - stride]) / twoSteps);
}

/**
 * `gradientAt` at each point.
 * @param {Grid} grid
 * @param {Float32Array} field
 * @param {Float32Array} points  (M, 3)
 * @returns {Float32Array} (M, 3)
 */
export function readGradient(grid, field, points) {
  const out = new Float32Array(points.length);
  for (let p = 0; p < points.length; p += 3) gradientAt(grid, field, points[p], points[p + 1], points[p + 2], out, p);
  return out;
}

/**
 * The potential of a unit source, −1/(4π r), on the grid padded to twice its nodes so the FFT's
 * circular convolution does not wrap, transformed: NumPy's `rfftn(kernel).real` in float32. The
 * kernel is even along every axis, so its spectrum is real and even too, and only the octant of
 * indices up to half of each padded size is kept: (P/2 + 1, Q/2 + 1, R/2 + 1).
 * @param {number[]} nodes
 * @param {number} step
 * @param {number} cubeMeanInverseDistance  the mean of 1/r over a cube of unit side about its centre:
 *   the potential of a source spread over its own node, so the kernel stays finite at r = 0
 * @returns {Float32Array}
 */
export function potentialSpectrum(nodes, step, cubeMeanInverseDistance) {
  const [a, b, c] = nodes.map((n) => n + 1);
  const spectrum = new Float32Array(a * b * c);
  for (let i = 0; i < a; i++) {
    for (let j = 0; j < b; j++) {
      for (let k = 0; k < c; k++) {
        const dx = i * step, dy = j * step, dz = k * step;
        const r = i || j || k ? Math.sqrt(dx * dx + dy * dy + dz * dz) : step / cubeMeanInverseDistance;
        spectrum[(i * b + j) * c + k] = -1 / (4 * Math.PI * r);
      }
    }
  }
  // An even line of the octant, extended to its full length, transforms to a real even line, so two
  // lines go through one complex transform as its real and imaginary parts.
  const transformAxis = (length, count, start, stride) => {
    const plan = linePlan(2 * (length - 1));
    const n = plan.length;
    const lines = [new Float64Array(n), new Float64Array(n), new Float64Array(n), new Float64Array(n)];
    for (let line = 0; line < count; line += 2) {
      const first = start(line), second = line + 1 < count ? start(line + 1) : -1;
      for (let m = 0; m < n; m++) {
        const at = Math.min(m, n - m) * stride;
        lines[0][m] = spectrum[first + at];
        lines[1][m] = second < 0 ? 0 : spectrum[second + at];
      }
      const [re, im] = transformLine(plan, lines[0], lines[1], lines[2], lines[3]);
      for (let f = 0; f < length; f++) {
        spectrum[first + f * stride] = re[f];
        if (second >= 0) spectrum[second + f * stride] = im[f];
      }
    }
  };
  transformAxis(c, a * b, (line) => line * c, 1);
  transformAxis(b, a * c, (line) => Math.floor(line / c) * b * c + (line % c), c);
  transformAxis(a, b * c, (line) => line, b * c);
  return spectrum;
}

/**
 * The potential at the grid's nodes of sources on them and of their mirror image under the ground,
 * the grid's nodes being symmetric about it in z: `_potential` in `features.py`. The sources are
 * zero-padded to twice the nodes, transformed, multiplied by the kernel's spectrum and transformed
 * back, and the nodes' part of the result kept. Lines that hold only padding going in, or only
 * values thrown away coming out, are not transformed.
 * @param {number[]} nodes
 * @param {Float32Array} spectrum  from `potentialSpectrum`
 * @param {Float64Array | Float32Array} sources  (nx, ny, nz); the mirror image is added before rounding to float32
 * @returns {Float32Array} (nx, ny, nz)
 */
export function potential(nodes, spectrum, sources) {
  const [nx, ny, nz] = nodes;
  const [P, Q, R] = nodes.map((n) => 2 * n);
  const H = nz + 1;
  const half = ny + 1;
  // (nx, Q, H) complex: the z transform of every line, then the y transform of every column, of
  // the nodes' x range; the x transforms run line by line through the kernel and back.
  const work = new Float32Array(2 * nx * Q * H);
  const at = (i, j, f) => 2 * ((i * Q + j) * H + f);
  const buffers = (n) => [new Float64Array(n), new Float64Array(n), new Float64Array(n), new Float64Array(n)];

  // z: two real lines as one complex line, split after.
  let plan = linePlan(R);
  let [re, im, spareRe, spareIm] = buffers(R);
  const lines = nx * ny;
  for (let line = 0; line < lines; line += 2) {
    re.fill(0);
    im.fill(0);
    for (let k = 0; k < nz; k++) {
      re[k] = Math.fround(sources[line * nz + k] + sources[line * nz + nz - 1 - k]);
      if (line + 1 < lines) im[k] = Math.fround(sources[(line + 1) * nz + k] + sources[(line + 1) * nz + nz - 1 - k]);
    }
    const [cr, ci] = transformLine(plan, re, im, spareRe, spareIm);
    for (let f = 0; f < H; f++) {
      const g = (R - f) % R;
      const first = at(Math.floor(line / ny), line % ny, f);
      work[first] = 0.5 * (cr[f] + cr[g]);
      work[first + 1] = 0.5 * (ci[f] - ci[g]);
      if (line + 1 < lines) {
        const second = at(Math.floor((line + 1) / ny), (line + 1) % ny, f);
        work[second] = 0.5 * (ci[f] + ci[g]);
        work[second + 1] = -0.5 * (cr[f] - cr[g]);
      }
    }
  }

  // y, forward: only the nodes' rows hold anything.
  plan = linePlan(Q);
  [re, im, spareRe, spareIm] = buffers(Q);
  for (let i = 0; i < nx; i++) {
    for (let f = 0; f < H; f++) {
      re.fill(0);
      im.fill(0);
      for (let j = 0; j < ny; j++) {
        re[j] = work[at(i, j, f)];
        im[j] = work[at(i, j, f) + 1];
      }
      const [yr, yi] = transformLine(plan, re, im, spareRe, spareIm);
      for (let j = 0; j < Q; j++) {
        work[at(i, j, f)] = yr[j];
        work[at(i, j, f) + 1] = yi[j];
      }
    }
  }

  // x: forward, times the kernel's spectrum, back; only the nodes' planes in and out.
  plan = linePlan(P);
  [re, im, spareRe, spareIm] = buffers(P);
  for (let j = 0; j < Q; j++) {
    const kernelRow = Math.min(j, Q - j) * H;
    for (let f = 0; f < H; f++) {
      re.fill(0);
      im.fill(0);
      for (let i = 0; i < nx; i++) {
        re[i] = work[at(i, j, f)];
        im[i] = work[at(i, j, f) + 1];
      }
      const [xr, xi] = transformLine(plan, re, im, spareRe, spareIm);
      // Into the buffers the transform did not leave its result in, conjugated for the inverse.
      const [inRe, inIm, outRe, outIm] = xr === re ? [spareRe, spareIm, re, im] : [re, im, spareRe, spareIm];
      for (let a = 0; a < P; a++) {
        const kernel = spectrum[(Math.min(a, P - a) * half) * H + kernelRow + f];
        inRe[a] = xr[a] * kernel;
        inIm[a] = -xi[a] * kernel;
      }
      const [br, bi] = transformLine(plan, inRe, inIm, outRe, outIm);
      for (let i = 0; i < nx; i++) {
        work[at(i, j, f)] = br[i];
        work[at(i, j, f) + 1] = -bi[i];
      }
    }
  }

  // y, back: only the nodes' rows kept.
  plan = linePlan(Q);
  [re, im, spareRe, spareIm] = buffers(Q);
  for (let i = 0; i < nx; i++) {
    for (let f = 0; f < H; f++) {
      for (let j = 0; j < Q; j++) {
        re[j] = work[at(i, j, f)];
        im[j] = -work[at(i, j, f) + 1];
      }
      const [yr, yi] = transformLine(plan, re, im, spareRe, spareIm);
      for (let j = 0; j < ny; j++) {
        work[at(i, j, f)] = yr[j];
        work[at(i, j, f) + 1] = -yi[j];
      }
    }
  }

  // z, back to real: two lines' Hermitian spectra as one complex line, the imaginary parts at the
  // zero and Nyquist frequencies dropped as NumPy's `irfftn` drops them.
  plan = linePlan(R);
  [re, im, spareRe, spareIm] = buffers(R);
  const potentialOut = new Float32Array(nx * ny * nz);
  const scale = 1 / (P * Q * R);
  const spectrumOf = (line, k) => {
    if (line >= lines) return [0, 0];
    const f = k <= nz ? k : R - k;
    const index = at(Math.floor(line / ny), line % ny, f);
    const imaginary = f === 0 || f === nz ? 0 : work[index + 1];
    return [work[index], k <= nz ? imaginary : -imaginary];
  };
  for (let line = 0; line < lines; line += 2) {
    for (let k = 0; k < R; k++) {
      const [ar, ai] = spectrumOf(line, k);
      const [br, bi] = spectrumOf(line + 1, k);
      re[k] = ar - bi;
      im[k] = -(ai + br);
    }
    const [zr, zi] = transformLine(plan, re, im, spareRe, spareIm);
    for (let k = 0; k < nz; k++) {
      potentialOut[line * nz + k] = zr[k] * scale;
      if (line + 1 < lines) potentialOut[(line + 1) * nz + k] = -zi[k] * scale;
    }
  }
  return potentialOut;
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
