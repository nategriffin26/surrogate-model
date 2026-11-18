/**
 * Fast Fourier transforms of lines of complex numbers whose length has no prime factor but 2, 3 and
 * 5, as the potential flow's padded grids are sized: the plan the CPU and the GPU both follow, the
 * CPU's transform, and the WGSL of the GPU's.
 *
 * Both run Stockham's self-sorting form. Each stage reads one half of a ping-pong buffer and writes
 * the other in natural order, so there is no bit-reversal pass, and a GPU workgroup keeps its whole
 * line in shared memory. Only the forward transform exists: the inverse is the forward transform of
 * the conjugate, conjugated, unscaled as NumPy's before its division by the size.
 */

/**
 * @typedef {Object} LinePlan
 * @property {number} length
 * @property {number[]} radices   one per stage, their product the length
 * @property {Float64Array} twiddles  (length, 2) exp(−2πi m / length) for m < length
 */

/**
 * @param {number} length
 * @returns {LinePlan}
 */
export function linePlan(length) {
  const radices = [];
  let rest = length;
  // Radix 4 first: a stage of it does the work of two radix-2 stages with fewer multiplications.
  for (const radix of [4, 2, 3, 5]) {
    while (rest % radix === 0) {
      radices.push(radix);
      rest /= radix;
    }
  }
  if (rest !== 1 || length < 2) throw new Error(`A line of ${length} cannot be transformed: its length must be a product of 2, 3 and 5.`);
  const twiddles = new Float64Array(2 * length);
  for (let m = 0; m < length; m++) {
    twiddles[2 * m] = Math.cos((-2 * Math.PI * m) / length);
    twiddles[2 * m + 1] = Math.sin((-2 * Math.PI * m) / length);
  }
  return { length, radices, twiddles };
}

const SIN_60 = Math.sqrt(3) / 2;
const COS_72 = Math.cos((2 * Math.PI) / 5);
const COS_144 = Math.cos((4 * Math.PI) / 5);
const SIN_72 = Math.sin((2 * Math.PI) / 5);
const SIN_144 = Math.sin((4 * Math.PI) / 5);

/**
 * The forward transform of the line held in `re`, `im`, unscaled: X[f] = Σ x[n] exp(−2πi f n / N).
 * `spareRe`, `spareIm` are arrays of the same length the stages alternate with; the result is in
 * whichever pair the last stage wrote, which is returned.
 * @param {LinePlan} plan
 * @param {Float64Array} re
 * @param {Float64Array} im
 * @param {Float64Array} spareRe
 * @param {Float64Array} spareIm
 * @returns {[Float64Array, Float64Array]}
 */
export function transformLine({ length: n, radices, twiddles: w }, re, im, spareRe, spareIm) {
  let inRe = re, inIm = im, outRe = spareRe, outIm = spareIm;
  let span = 1;
  for (const radix of radices) {
    const m = n / radix;
    const step = n / (span * radix);
    for (let j = 0; j < m; j++) {
      const k = j % span;
      const d = (j - k) * radix + k;
      const t = k * step;
      const a0r = inRe[j], a0i = inIm[j];
      const b1r = inRe[j + m], b1i = inIm[j + m];
      const a1r = b1r * w[2 * t] - b1i * w[2 * t + 1], a1i = b1r * w[2 * t + 1] + b1i * w[2 * t];
      if (radix === 2) {
        outRe[d] = a0r + a1r; outIm[d] = a0i + a1i;
        outRe[d + span] = a0r - a1r; outIm[d + span] = a0i - a1i;
        continue;
      }
      const b2r = inRe[j + 2 * m], b2i = inIm[j + 2 * m];
      const a2r = b2r * w[4 * t] - b2i * w[4 * t + 1], a2i = b2r * w[4 * t + 1] + b2i * w[4 * t];
      if (radix === 3) {
        const sr = a1r + a2r, si = a1i + a2i;
        const mr = a0r - 0.5 * sr, mi = a0i - 0.5 * si;
        // −i sin 60° (a1 − a2)
        const qr = SIN_60 * (a1i - a2i), qi = -SIN_60 * (a1r - a2r);
        outRe[d] = a0r + sr; outIm[d] = a0i + si;
        outRe[d + span] = mr + qr; outIm[d + span] = mi + qi;
        outRe[d + 2 * span] = mr - qr; outIm[d + 2 * span] = mi - qi;
        continue;
      }
      const b3r = inRe[j + 3 * m], b3i = inIm[j + 3 * m];
      const a3r = b3r * w[6 * t] - b3i * w[6 * t + 1], a3i = b3r * w[6 * t + 1] + b3i * w[6 * t];
      if (radix === 4) {
        const pr = a0r + a2r, pi = a0i + a2i, mr = a0r - a2r, mi = a0i - a2i;
        const sr = a1r + a3r, si = a1i + a3i;
        // −i (a1 − a3)
        const qr = a1i - a3i, qi = -(a1r - a3r);
        outRe[d] = pr + sr; outIm[d] = pi + si;
        outRe[d + span] = mr + qr; outIm[d + span] = mi + qi;
        outRe[d + 2 * span] = pr - sr; outIm[d + 2 * span] = pi - si;
        outRe[d + 3 * span] = mr - qr; outIm[d + 3 * span] = mi - qi;
        continue;
      }
      const b4r = inRe[j + 4 * m], b4i = inIm[j + 4 * m];
      const a4r = b4r * w[8 * t] - b4i * w[8 * t + 1], a4i = b4r * w[8 * t + 1] + b4i * w[8 * t];
      const s1r = a1r + a4r, s1i = a1i + a4i, d1r = a1r - a4r, d1i = a1i - a4i;
      const s2r = a2r + a3r, s2i = a2i + a3i, d2r = a2r - a3r, d2i = a2i - a3i;
      const c1r = a0r + COS_72 * s1r + COS_144 * s2r, c1i = a0i + COS_72 * s1i + COS_144 * s2i;
      const c2r = a0r + COS_144 * s1r + COS_72 * s2r, c2i = a0i + COS_144 * s1i + COS_72 * s2i;
      // −i (sin 72° d1 + sin 144° d2) and −i (sin 144° d1 − sin 72° d2)
      const q1r = SIN_72 * d1i + SIN_144 * d2i, q1i = -(SIN_72 * d1r + SIN_144 * d2r);
      const q2r = SIN_144 * d1i - SIN_72 * d2i, q2i = -(SIN_144 * d1r - SIN_72 * d2r);
      outRe[d] = a0r + s1r + s2r; outIm[d] = a0i + s1i + s2i;
      outRe[d + span] = c1r + q1r; outIm[d + span] = c1i + q1i;
      outRe[d + 2 * span] = c2r + q2r; outIm[d + 2 * span] = c2i + q2i;
      outRe[d + 3 * span] = c2r - q2r; outIm[d + 3 * span] = c2i - q2i;
      outRe[d + 4 * span] = c1r - q1r; outIm[d + 4 * span] = c1i - q1i;
    }
    [inRe, inIm, outRe, outIm] = [outRe, outIm, inRe, inIm];
    span *= radix;
  }
  return [inRe, inIm];
}

/**
 * WGSL for one workgroup transforming the line it holds in `line[0 .. length)`, a workgroup array of
 * 2 × length `vec2f` the caller declares, with the plan's twiddles bound as `twiddles`. Stages are
 * unrolled with their sizes as constants. `transform(thread)` leaves the result in
 * `line[TRANSFORMED .. TRANSFORMED + length)` and ends on a barrier.
 * @param {LinePlan} plan
 * @param {number} threads  the workgroup's size
 * @returns {string}
 */
export function wgslLineTransform({ length: n, radices }, threads) {
  const stages = [];
  let span = 1;
  let from = 0;
  for (const radix of radices) {
    const m = n / radix;
    const to = n - from;
    const step = n / (span * radix);
    const load = (r) => (r === 0 ? `line[${from}u + j]` : `mul(line[${from + r * m}u + j], twiddles[${r * step}u * k])`);
    const lines = [];
    for (let r = 0; r < radix; r++) lines.push(`let a${r} = ${load(r)};`);
    lines.push(`let d = ${to}u + (j - k) * ${radix}u + k;`);
    lines.push(...BUTTERFLIES[radix].map((store) => store.replaceAll('SPAN', `${span}u`)));
    stages.push(`  for (var j = thread; j < ${m}u; j += ${threads}u) {
    let k = j % ${span}u;
    ${lines.join('\n    ')}
  }
  workgroupBarrier();`);
    from = to;
    span *= radix;
  }
  return /* wgsl */ `
const TRANSFORMED = ${from}u;

fn mul(a: vec2f, b: vec2f) -> vec2f {
  return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// −i a
fn turn(a: vec2f) -> vec2f {
  return vec2f(a.y, -a.x);
}

fn transform(thread: u32) {
${stages.join('\n')}
}
`;
}

/** Each radix's butterfly on a0 … a(radix − 1), stored at line[d + r × SPAN]. */
const BUTTERFLIES = {
  2: [
    'line[d] = a0 + a1;',
    'line[d + SPAN] = a0 - a1;',
  ],
  3: [
    'let s = a1 + a2;',
    'let m = a0 - 0.5 * s;',
    `let q = turn(${SIN_60} * (a1 - a2));`,
    'line[d] = a0 + s;',
    'line[d + SPAN] = m + q;',
    'line[d + 2u * SPAN] = m - q;',
  ],
  4: [
    'let p = a0 + a2;',
    'let m = a0 - a2;',
    'let s = a1 + a3;',
    'let q = turn(a1 - a3);',
    'line[d] = p + s;',
    'line[d + SPAN] = m + q;',
    'line[d + 2u * SPAN] = p - s;',
    'line[d + 3u * SPAN] = m - q;',
  ],
  5: [
    'let s1 = a1 + a4;',
    'let d1 = a1 - a4;',
    'let s2 = a2 + a3;',
    'let d2 = a2 - a3;',
    `let c1 = a0 + ${COS_72} * s1 + ${COS_144} * s2;`,
    `let c2 = a0 + ${COS_144} * s1 + ${COS_72} * s2;`,
    `let q1 = turn(${SIN_72} * d1 + ${SIN_144} * d2);`,
    `let q2 = turn(${SIN_144} * d1 - ${SIN_72} * d2);`,
    'line[d] = a0 + s1 + s2;',
    'line[d + SPAN] = c1 + q1;',
    'line[d + 2u * SPAN] = c2 + q2;',
    'line[d + 3u * SPAN] = c2 - q2;',
    'line[d + 4u * SPAN] = c1 - q1;',
  ],
};
