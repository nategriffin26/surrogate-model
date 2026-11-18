/**
 * The grid operations of `grid.js` as WebGPU passes, recorded into the caller's command encoder:
 * spreading what points carry onto a grid's nodes, the potential of sources on a grid and of their
 * mirror image under the ground, the wall condition's pass at every cell, correlation along an axis,
 * marking the nodes around points, and gathering a field's values at listed nodes.
 *
 * Fields are float32 storage buffers laid out as `grid.js` lays them out. A spread sums into 64-bit
 * fixed point with integer atomics, 2⁻⁴⁰ apart: integer sums do not depend on the order the GPU's
 * threads add in, so a spread gives the same field on every run and every GPU, finer than float32.
 * Grid coordinates are worked out in float32, which places a point to within about 1 µm on a car.
 */

import { linePlan, wgslLineTransform } from './fft.js';

/** Threads of a workgroup transforming one line. */
const LINE_THREADS = 64;
/** Threads of a workgroup working point by point or node by node. */
const THREADS = 256;
/** Workgroups along x of a dispatch; more go on along y. WebGPU guarantees 65535. */
const ROW_GROUPS = 65535;

/**
 * The grid's origin and step, node counts and the points a pass spreads or reads, with the cube of
 * nodes each point falls in: a point beyond the grid takes the nearest cube at its edge, which is both
 * how `features.py` spreads and how SciPy's `map_coordinates(mode='nearest')` reads.
 */
const POINTS_ON_GRID = /* wgsl */ `
struct Grid {
  origin: vec3f,
  step: f32,
  nodes: vec3u,
  count: u32,
}

@group(0) @binding(0) var<uniform> grid: Grid;
@group(0) @binding(1) var<storage, read> positions: array<f32>;

fn positionOf(point: u32) -> vec3f {
  return vec3f(positions[3u * point], positions[3u * point + 1u], positions[3u * point + 2u]);
}

// The cube's first node and the point's place across it, per axis.
fn cubeOf(point: u32) -> array<vec3f, 2> {
  let q = clamp((positionOf(point) - grid.origin) / grid.step, vec3f(0.0), vec3f(grid.nodes - 1u));
  let base = min(floor(q), vec3f(grid.nodes - 2u));
  return array<vec3f, 2>(base, q - base);
}

fn pointIndex(group: vec3u, thread: u32) -> u32 {
  return (group.x + group.y * ${ROW_GROUPS}u) * ${THREADS}u + thread;
}
`;

const SPREAD = /* wgsl */ `${POINTS_ON_GRID}
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> sums: array<atomic<u32>>;

const FIXED = 1099511627776.0;

// value × 2^40 as a two's-complement 64-bit integer, low word then high, added with a carry.
fn add(node: u32, value: f32) {
  let scaled = abs(value) * FIXED;
  let high = floor(scaled / 4294967296.0);
  var low = u32(scaled - high * 4294967296.0);
  var top = u32(high);
  if (value < 0.0) {
    low = ~low + 1u;
    top = ~top + select(0u, 1u, low == 0u);
  }
  let before = atomicAdd(&sums[2u * node], low);
  atomicAdd(&sums[2u * node + 1u], top + select(0u, 1u, before + low < before));
}

@compute @workgroup_size(${THREADS})
fn main(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) thread: u32) {
  let point = pointIndex(group, thread);
  if (point >= grid.count) {
    return;
  }
  let cube = cubeOf(point);
  let at = vec3u(cube[0]);
  let f = cube[1];
  let weight = weights[point];
  for (var dx = 0u; dx < 2u; dx++) {
    let wx = weight * select(1.0 - f.x, f.x, dx == 1u);
    for (var dy = 0u; dy < 2u; dy++) {
      let wxy = wx * select(1.0 - f.y, f.y, dy == 1u);
      let row = ((at.x + dx) * grid.nodes.y + at.y + dy) * grid.nodes.z + at.z;
      add(row, wxy * (1.0 - f.z));
      add(row + 1u, wxy * f.z);
    }
  }
}
`;

const FIXED_TO_FLOAT = /* wgsl */ `
@group(0) @binding(0) var<uniform> count: u32;
@group(0) @binding(1) var<storage, read> sums: array<vec2u>;
@group(0) @binding(2) var<storage, read_write> field: array<f32>;

@compute @workgroup_size(${THREADS})
fn main(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) thread: u32) {
  let node = (group.x + group.y * ${ROW_GROUPS}u) * ${THREADS}u + thread;
  if (node >= count) {
    return;
  }
  var low = sums[node].x;
  var top = sums[node].y;
  let negative = top >= 0x80000000u;
  // The magnitude first, so a small negative sum does not cancel between its two words.
  if (negative) {
    low = ~low + 1u;
    top = ~top + select(0u, 1u, low == 0u);
  }
  let magnitude = (f32(top) * 4294967296.0 + f32(low)) / 1099511627776.0;
  field[node] = select(magnitude, -magnitude, negative);
}
`;

/**
 * One pass of the wall condition at every cell: the velocity the sources induce, the gradient of the
 * potential read as `grid.js` `readGradient` reads it, moves each cell's source density σ part of the
 * way to what keeps the stream out of the wall, σ + relaxation (−2 n_x − 2 u·n − σ); the next spread
 * carries σ × area.
 */
const WALL = /* wgsl */ `${POINTS_ON_GRID}
@group(0) @binding(2) var<storage, read> potential: array<f32>;
@group(0) @binding(3) var<storage, read> normals: array<f32>;
@group(0) @binding(4) var<storage, read> areas: array<f32>;
@group(0) @binding(5) var<storage, read_write> densities: array<f32>;
@group(0) @binding(6) var<storage, read_write> strengths: array<f32>;
@group(0) @binding(7) var<uniform> relaxation: f32;

// NumPy's gradient: central differences inside, one-sided at the faces.
fn derivative(node: u32, index: u32, n: u32, stride: u32) -> f32 {
  if (index == 0u) {
    return (potential[node + stride] - potential[node]) / grid.step;
  }
  if (index == n - 1u) {
    return (potential[node] - potential[node - stride]) / grid.step;
  }
  return (potential[node + stride] - potential[node - stride]) / (2.0 * grid.step);
}

@compute @workgroup_size(${THREADS})
fn main(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) thread: u32) {
  let point = pointIndex(group, thread);
  if (point >= grid.count) {
    return;
  }
  let cube = cubeOf(point);
  let at = vec3u(cube[0]);
  let f = cube[1];
  let n = grid.nodes;
  var u = vec3f(0.0);
  for (var dx = 0u; dx < 2u; dx++) {
    for (var dy = 0u; dy < 2u; dy++) {
      for (var dz = 0u; dz < 2u; dz++) {
        let i = at + vec3u(dx, dy, dz);
        let node = (i.x * n.y + i.y) * n.z + i.z;
        let w = select(1.0 - f.x, f.x, dx == 1u) * select(1.0 - f.y, f.y, dy == 1u) * select(1.0 - f.z, f.z, dz == 1u);
        u += w * vec3f(derivative(node, i.x, n.x, n.y * n.z), derivative(node, i.y, n.y, n.z), derivative(node, i.z, n.z, 1u));
      }
    }
  }
  let normal = vec3f(normals[3u * point], normals[3u * point + 1u], normals[3u * point + 2u]);
  let density = densities[point];
  let next = density + relaxation * (-2.0 * normal.x - 2.0 * dot(u, normal) - density);
  densities[point] = next;
  strengths[point] = next * areas[point];
}
`;

/**
 * The nodes around every point within `reach` of a given one: each node of every cube that the box
 * of half-side `reach` about the point overlaps, set in a bitset of the grid's nodes.
 */
const MARK = /* wgsl */ `${POINTS_ON_GRID}
@group(0) @binding(2) var<storage, read_write> bits: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> reach: f32;

@compute @workgroup_size(${THREADS})
fn main(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) thread: u32) {
  let point = pointIndex(group, thread);
  if (point >= grid.count) {
    return;
  }
  let q = (positionOf(point) - grid.origin) / grid.step;
  let last = vec3f(grid.nodes - 2u);
  let low = vec3u(clamp(floor(q - reach / grid.step), vec3f(0.0), last));
  let high = vec3u(clamp(floor(q + reach / grid.step), vec3f(0.0), last));
  for (var i = low.x; i <= high.x + 1u; i++) {
    for (var j = low.y; j <= high.y + 1u; j++) {
      for (var k = low.z; k <= high.z + 1u; k++) {
        let node = (i * grid.nodes.y + j) * grid.nodes.z + k;
        atomicOr(&bits[node / 32u], 1u << (node % 32u));
      }
    }
  }
}
`;

/** A field's values at listed nodes, into one lane of rows of `lanes` floats. */
const GATHER = /* wgsl */ `
struct Rows {
  count: u32,
  lane: u32,
  lanes: u32,
}

@group(0) @binding(0) var<uniform> rows: Rows;
@group(0) @binding(1) var<storage, read> field: array<f32>;
@group(0) @binding(2) var<storage, read> nodes: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(${THREADS})
fn main(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) thread: u32) {
  let row = (group.x + group.y * ${ROW_GROUPS}u) * ${THREADS}u + thread;
  if (row >= rows.count) {
    return;
  }
  out[row * rows.lanes + rows.lane] = field[nodes[row]];
}
`;

const CORRELATE = /* wgsl */ `
struct Axis {
  count: u32,
  length: u32,
  stride: u32,
  radius: u32,
}

@group(0) @binding(0) var<uniform> axis: Axis;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> field: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

// SciPy's 'reflect' edge: d c b a | a b c d | d c b a.
fn reflected(index: i32, n: i32) -> i32 {
  let period = 2 * n;
  let m = ((index % period) + period) % period;
  return select(period - 1 - m, m, m < n);
}

@compute @workgroup_size(${THREADS})
fn main(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) thread: u32) {
  let node = (group.x + group.y * ${ROW_GROUPS}u) * ${THREADS}u + thread;
  if (node >= axis.count) {
    return;
  }
  let index = i32((node / axis.stride) % axis.length);
  let base = node - u32(index) * axis.stride;
  let radius = i32(axis.radius);
  var sum = 0.0;
  for (var k = -radius; k <= radius; k++) {
    sum += weights[k + radius] * field[base + u32(reflected(index + k, i32(axis.length))) * axis.stride];
  }
  out[node] = sum;
}
`;

/**
 * WGSL of the potential's passes for a grid of `nodes`, padded to (P, Q, R) = 2 × nodes. The work
 * buffer holds the nodes' x range of the transform, (nx, Q, R/2 + 1) complex; the kernel's spectrum
 * is the octant `grid.js` `potentialSpectrum` keeps, (P/2 + 1, Q/2 + 1, R/2 + 1).
 */
function potentialShaders([nx, ny, nz], step, cubeMeanInverseDistance) {
  const [P, Q, R] = [2 * nx, 2 * ny, 2 * nz];
  const H = nz + 1;
  const B = ny + 1;
  const constants = /* wgsl */ `
const NX = ${nx}u;
const NY = ${ny}u;
const NZ = ${nz}u;
const P = ${P}u;
const Q = ${Q}u;
const R = ${R}u;
const H = ${H}u;
const B = ${B}u;
const LINES = ${nx * ny}u;

fn lineIndex(group: vec3u) -> u32 {
  return group.x + group.y * ${ROW_GROUPS}u;
}
`;
  const module = (length, body) => /* wgsl */ `${constants}
@group(0) @binding(0) var<storage, read> twiddles: array<vec2f>;
var<workgroup> line: array<vec2f, ${2 * length}>;
${wgslLineTransform(linePlan(length), LINE_THREADS)}
${body}`;
  const entry = `@compute @workgroup_size(${LINE_THREADS})
fn main(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) thread: u32)`;

  // The z transform of two real lines at once, as the real and imaginary parts of one complex line.
  const zRow = /* wgsl */ `
fn row(source: u32) -> u32 {
  return ((source / NY) * Q + source % NY) * H;
}
`;
  const forwardZ = module(R, /* wgsl */ `
@group(0) @binding(1) var<storage, read> sources: array<f32>;
@group(0) @binding(2) var<storage, read_write> work: array<vec2f>;
${zRow}
// The sources and their mirror image under the ground: node k and node NZ - 1 - k mirror each other.
fn mirrored(source: u32, k: u32) -> f32 {
  if (source >= LINES || k >= NZ) {
    return 0.0;
  }
  return sources[source * NZ + k] + sources[source * NZ + NZ - 1u - k];
}

${entry} {
  let first = 2u * lineIndex(group);
  if (first >= LINES) {
    return;
  }
  for (var k = thread; k < R; k += ${LINE_THREADS}u) {
    line[k] = vec2f(mirrored(first, k), mirrored(first + 1u, k));
  }
  workgroupBarrier();
  transform(thread);
  for (var f = thread; f < H; f += ${LINE_THREADS}u) {
    let c = line[TRANSFORMED + f];
    let g = line[TRANSFORMED + (R - f) % R];
    work[row(first) + f] = 0.5 * vec2f(c.x + g.x, c.y - g.y);
    if (first + 1u < LINES) {
      work[row(first + 1u) + f] = 0.5 * vec2f(c.y + g.y, g.x - c.x);
    }
  }
}
`);

  // y lines: in the transform's direction only the nodes' rows hold anything; back, only they are kept.
  const alongY = (inverse) => module(Q, /* wgsl */ `
@group(0) @binding(1) var<storage, read_write> work: array<vec2f>;

${entry} {
  let index = lineIndex(group);
  if (index >= NX * H) {
    return;
  }
  let base = (index / H) * Q * H + index % H;
  for (var j = thread; j < Q; j += ${LINE_THREADS}u) {
    ${inverse
      ? 'let v = work[base + j * H];\n    line[j] = vec2f(v.x, -v.y);'
      : 'line[j] = vec2f(0.0);\n    if (j < NY) {\n      line[j] = work[base + j * H];\n    }'}
  }
  workgroupBarrier();
  transform(thread);
  for (var j = thread; j < ${inverse ? 'NY' : 'Q'}; j += ${LINE_THREADS}u) {
    let v = line[TRANSFORMED + j];
    work[base + j * H] = ${inverse ? 'vec2f(v.x, -v.y)' : 'v'};
  }
}
`);

  // x lines: forward, times the kernel's spectrum, and back in one workgroup, so the padded x range
  // never goes to memory.
  const convolveX = module(P, /* wgsl */ `
@group(0) @binding(1) var<storage, read_write> work: array<vec2f>;
@group(0) @binding(2) var<storage, read> kernel: array<f32>;

${entry} {
  let index = lineIndex(group);
  if (index >= Q * H) {
    return;
  }
  let j = index / H;
  let f = index % H;
  let base = j * H + f;
  for (var i = thread; i < P; i += ${LINE_THREADS}u) {
    line[i] = vec2f(0.0);
    if (i < NX) {
      line[i] = work[base + i * Q * H];
    }
  }
  workgroupBarrier();
  transform(thread);
  let kernelRow = min(j, Q - j) * H + f;
  for (var a = thread; a < P; a += ${LINE_THREADS}u) {
    let v = line[TRANSFORMED + a] * kernel[min(a, P - a) * B * H + kernelRow];
    line[a] = vec2f(v.x, -v.y);
  }
  workgroupBarrier();
  transform(thread);
  for (var i = thread; i < NX; i += ${LINE_THREADS}u) {
    let v = line[TRANSFORMED + i];
    work[base + i * Q * H] = vec2f(v.x, -v.y);
  }
}
`);

  // Back to real along z: two lines' Hermitian spectra as one complex line, the imaginary parts at
  // the zero and Nyquist frequencies dropped as NumPy's irfftn drops them.
  const inverseZ = module(R, /* wgsl */ `
@group(0) @binding(1) var<storage, read> work: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> potential: array<f32>;
${zRow}
fn hermitian(source: u32, k: u32) -> vec2f {
  if (source >= LINES) {
    return vec2f(0.0);
  }
  let f = select(R - k, k, k <= NZ);
  var v = work[row(source) + f];
  if (f == 0u || f == NZ) {
    v.y = 0.0;
  }
  if (k > NZ) {
    v.y = -v.y;
  }
  return v;
}

${entry} {
  let first = 2u * lineIndex(group);
  if (first >= LINES) {
    return;
  }
  for (var k = thread; k < R; k += ${LINE_THREADS}u) {
    let a = hermitian(first, k);
    let b = hermitian(first + 1u, k);
    line[k] = vec2f(a.x - b.y, -(a.y + b.x));
  }
  workgroupBarrier();
  transform(thread);
  for (var k = thread; k < NZ; k += ${LINE_THREADS}u) {
    let v = line[TRANSFORMED + k];
    potential[first * NZ + k] = v.x * ${1 / (P * Q * R)};
    if (first + 1u < LINES) {
      potential[(first + 1u) * NZ + k] = -v.y * ${1 / (P * Q * R)};
    }
  }
}
`);

  // The kernel −1/(4π r) on the octant; r = 0 holds the mean of 1/r over its node's cube.
  const kernelValues = /* wgsl */ `${constants}
@group(0) @binding(0) var<storage, read_write> kernel: array<f32>;

@compute @workgroup_size(${THREADS})
fn main(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) thread: u32) {
  let index = lineIndex(group) * ${THREADS}u + thread;
  if (index >= (NX + 1u) * B * H) {
    return;
  }
  let a = index / (B * H);
  let b = (index / H) % B;
  let c = index % H;
  let squared = a * a + b * b + c * c;
  kernel[index] = select(${-cubeMeanInverseDistance / (4 * Math.PI * step)}, ${-1 / (4 * Math.PI * step)} / sqrt(f32(squared)), squared > 0u);
}
`;

  // The kernel's transform along each axis: an even line of the octant, extended to its full length,
  // transforms to a real even line, so two lines go through one complex transform.
  const kernelAxis = (length, count, start, stride) => module(length, /* wgsl */ `
@group(0) @binding(1) var<storage, read_write> kernel: array<f32>;

fn start(index: u32) -> u32 {
  return ${start};
}

${entry} {
  let first = 2u * lineIndex(group);
  if (first >= ${count}u) {
    return;
  }
  let paired = first + 1u < ${count}u;
  for (var m = thread; m < ${length}u; m += ${LINE_THREADS}u) {
    let at = min(m, ${length}u - m) * ${stride}u;
    line[m] = vec2f(kernel[start(first) + at], 0.0);
    if (paired) {
      line[m].y = kernel[start(first + 1u) + at];
    }
  }
  workgroupBarrier();
  transform(thread);
  for (var f = thread; f <= ${length / 2}u; f += ${LINE_THREADS}u) {
    let v = line[TRANSFORMED + f];
    kernel[start(first) + f * ${stride}u] = v.x;
    if (paired) {
      kernel[start(first + 1u) + f * ${stride}u] = v.y;
    }
  }
}
`);
  const A = nx + 1;
  return {
    forwardZ, forwardY: alongY(false), convolveX, inverseY: alongY(true), inverseZ, kernelValues,
    kernelZ: kernelAxis(R, A * B, 'index * H', 1),
    kernelY: kernelAxis(Q, A * H, '(index / H) * B * H + index % H', H),
    kernelX: kernelAxis(P, B * H, 'index', B * H),
  };
}

/** Workgroups for `count` items, `perGroup` to a workgroup, laid out as the shaders index them. */
function groupsFor(count, perGroup = 1) {
  const groups = Math.ceil(count / perGroup);
  return [Math.min(groups, ROW_GROUPS), Math.ceil(groups / ROW_GROUPS)];
}

/**
 * @typedef {Object} GridOnGpu  A grid as the passes take it.
 * @property {number[]} origin
 * @property {number} step
 * @property {number[]} nodes
 *
 * @typedef {Object} Potential  The potential's passes for one grid shape, with the buffers they hold.
 * @property {number[]} nodes
 * @property {(encoder: GPUCommandEncoder, sources: GPUBuffer, potential: GPUBuffer) => void} encode
 *   The potential of `sources` and their mirror image at the nodes, into `potential` (both
 *   nx × ny × nz float32; they may be the same buffer).
 * @property {GPUBuffer} kernel  the kernel's spectrum, the octant `potentialSpectrum` gives
 * @property {() => void} destroy
 */

/**
 * @param {GPUDevice} device
 */
export function createGridOperations(device) {
  const pipelines = new Map();
  const pipeline = (code) => {
    if (!pipelines.has(code)) {
      pipelines.set(code, device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } }));
    }
    return pipelines.get(code);
  };
  const twiddleBuffers = new Map();
  const twiddles = (length) => {
    if (!twiddleBuffers.has(length)) {
      const values = Float32Array.from(linePlan(length).twiddles);
      const buffer = device.createBuffer({ size: values.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(buffer, 0, values);
      twiddleBuffers.set(length, buffer);
    }
    return twiddleBuffers.get(length);
  };
  const uniform = (bytes) => {
    const buffer = device.createBuffer({ size: Math.max(16, bytes.byteLength), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buffer, 0, bytes);
    return buffer;
  };
  const gridUniform = ({ origin, step, nodes }, count) => {
    const bytes = new ArrayBuffer(32);
    new Float32Array(bytes, 0, 4).set([...origin, step]);
    new Uint32Array(bytes, 16, 4).set([...nodes, count]);
    return uniform(bytes);
  };
  const dispatch = (encoder, code, buffers, groups) => {
    const compute = pipeline(code);
    const bindGroup = device.createBindGroup({
      layout: compute.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(compute);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(...groups);
    pass.end();
  };
  const storage = (size) => device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

  return {
    device,

    /**
     * Each point's weight shared among the eight nodes around it, added to the fixed-point `sums`
     * (8 bytes a node), which the caller clears before the first spread into them.
     * @param {GPUCommandEncoder} encoder
     * @param {GridOnGpu} grid
     * @param {{ positions: GPUBuffer, weights: GPUBuffer, count: number, sums: GPUBuffer }} buffers
     */
    spread(encoder, grid, { positions, weights, count, sums }) {
      dispatch(encoder, SPREAD, [gridUniform(grid, count), positions, weights, sums], groupsFor(count, THREADS));
    },

    /**
     * The fixed-point sums of `count` nodes as a float32 field.
     * @param {GPUCommandEncoder} encoder
     * @param {{ sums: GPUBuffer, field: GPUBuffer, count: number }} buffers
     */
    sumsToField(encoder, { sums, field, count }) {
      dispatch(encoder, FIXED_TO_FLOAT, [uniform(new Uint32Array([count])), sums, field], groupsFor(count, THREADS));
    },

    /**
     * One pass of the wall condition at `count` cells: their source densities move toward what the
     * flow of `potential` leaves them, and `strengths` becomes density × area for the next spread.
     * @param {GPUCommandEncoder} encoder
     * @param {GridOnGpu} grid
     * @param {{ positions: GPUBuffer, normals: GPUBuffer, areas: GPUBuffer, count: number, potential: GPUBuffer,
     *   densities: GPUBuffer, strengths: GPUBuffer, relaxation: number }} buffers
     */
    wall(encoder, grid, { positions, normals, areas, count, potential, densities, strengths, relaxation }) {
      const buffers = [gridUniform(grid, count), positions, potential, normals, areas, densities, strengths, uniform(new Float32Array([relaxation]))];
      dispatch(encoder, WALL, buffers, groupsFor(count, THREADS));
    },

    /**
     * The nodes around every point within `reach` (metres) of each of `count` points, set in `bits`
     * (one bit a node, 32 to a word; cleared by the caller).
     * @param {GPUCommandEncoder} encoder
     * @param {GridOnGpu} grid
     * @param {{ positions: GPUBuffer, count: number, reach: number, bits: GPUBuffer }} buffers
     */
    mark(encoder, grid, { positions, count, reach, bits }) {
      dispatch(encoder, MARK, [gridUniform(grid, count), positions, bits, uniform(new Float32Array([reach]))], groupsFor(count, THREADS));
    },

    /**
     * `field` at `count` listed nodes, into lane `lane` of rows of `lanes` floats.
     * @param {GPUCommandEncoder} encoder
     * @param {{ field: GPUBuffer, nodes: GPUBuffer, count: number, out: GPUBuffer, lane: number, lanes: number }} buffers
     */
    gather(encoder, { field, nodes, count, out, lane, lanes }) {
      dispatch(encoder, GATHER, [uniform(new Uint32Array([count, lane, lanes])), field, nodes, out], groupsFor(count, THREADS));
    },

    /**
     * `field` correlated along `axis` with `weights` (2r + 1 float32 in a storage buffer), into `out`,
     * as `grid.js` `correlateAlong`.
     * @param {GPUCommandEncoder} encoder
     * @param {number[]} nodes
     * @param {number} axis
     * @param {{ weights: GPUBuffer, radius: number, field: GPUBuffer, out: GPUBuffer }} buffers
     */
    correlate(encoder, nodes, axis, { weights, radius, field, out }) {
      const count = nodes[0] * nodes[1] * nodes[2];
      const stride = axis === 0 ? nodes[1] * nodes[2] : axis === 1 ? nodes[2] : 1;
      dispatch(encoder, CORRELATE, [uniform(new Uint32Array([count, nodes[axis], stride, radius])), weights, field, out], groupsFor(count, THREADS));
    },

    /**
     * The potential's passes for grids of `nodes` and `step`, with the kernel's spectrum made now.
     * @param {number[]} nodes
     * @param {number} step
     * @param {number} cubeMeanInverseDistance  the mean of 1/r over a unit cube, which stands for 1/r at r = 0
     * @returns {Potential}
     */
    potential(nodes, step, cubeMeanInverseDistance) {
      const [nx, ny, nz] = nodes;
      const shaders = potentialShaders(nodes, step, cubeMeanInverseDistance);
      const [P, Q, R] = nodes.map((n) => 2 * n);
      const H = nz + 1, B = ny + 1, A = nx + 1;
      const work = storage(8 * nx * Q * H);
      const kernel = storage(4 * A * B * H);
      const encoder = device.createCommandEncoder();
      dispatch(encoder, shaders.kernelValues, [kernel], groupsFor(A * B * H, THREADS));
      dispatch(encoder, shaders.kernelZ, [twiddles(R), kernel], groupsFor(A * B, 2));
      dispatch(encoder, shaders.kernelY, [twiddles(Q), kernel], groupsFor(A * H, 2));
      dispatch(encoder, shaders.kernelX, [twiddles(P), kernel], groupsFor(B * H, 2));
      device.queue.submit([encoder.finish()]);
      return {
        nodes,
        kernel,
        encode(encoder, sources, potential) {
          dispatch(encoder, shaders.forwardZ, [twiddles(R), sources, work], groupsFor(nx * ny, 2));
          dispatch(encoder, shaders.forwardY, [twiddles(Q), work], groupsFor(nx * H));
          dispatch(encoder, shaders.convolveX, [twiddles(P), work, kernel], groupsFor(Q * H));
          dispatch(encoder, shaders.inverseY, [twiddles(Q), work], groupsFor(nx * H));
          dispatch(encoder, shaders.inverseZ, [twiddles(R), work, potential], groupsFor(nx * ny, 2));
        },
        destroy() {
          work.destroy();
          kernel.destroy();
        },
      };
    },

    /** Let go of the buffers every grid shares (the transforms' twiddles). */
    destroy() {
      for (const buffer of twiddleBuffers.values()) buffer.destroy();
      twiddleBuffers.clear();
    },
  };
}
