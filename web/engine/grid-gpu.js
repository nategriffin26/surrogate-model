/**
 * The grid operations of `grid.js` as WebGPU passes, recorded into the caller's command encoder:
 * spreading what points carry onto a grid's nodes, correlation along an axis, marking the nodes
 * around points, and gathering a field's values at listed nodes.
 *
 * Fields are float32 storage buffers laid out as `grid.js` lays them out. A spread sums into 64-bit
 * fixed point with integer atomics, 2⁻⁴⁰ apart: integer sums do not depend on the order the GPU's
 * threads add in, so a spread gives the same field on every run and every GPU, finer than float32.
 * Grid coordinates are worked out in float32, which places a point to within about 1 µm on a car.
 */

/** Threads of a workgroup, each working on one point or one node. */
const THREADS = 256;
/** Workgroups along x of a dispatch; more go on along y. WebGPU guarantees 65535. */
const ROW_GROUPS = 65535;

/**
 * The grid's origin and step, node counts and the points a pass spreads, with the cube of nodes each
 * point falls in: a point beyond the grid takes the nearest cube at its edge, as `features.py` spreads.
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

/** Workgroups for `count` items, one thread each, laid out as the shaders index them. */
function groupsFor(count) {
  const groups = Math.ceil(count / THREADS);
  return [Math.min(groups, ROW_GROUPS), Math.ceil(groups / ROW_GROUPS)];
}

/**
 * @typedef {Object} GridOnGpu  A grid as the passes take it.
 * @property {number[]} origin
 * @property {number} step
 * @property {number[]} nodes
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
      dispatch(encoder, SPREAD, [gridUniform(grid, count), positions, weights, sums], groupsFor(count));
    },

    /**
     * The fixed-point sums of `count` nodes as a float32 field.
     * @param {GPUCommandEncoder} encoder
     * @param {{ sums: GPUBuffer, field: GPUBuffer, count: number }} buffers
     */
    sumsToField(encoder, { sums, field, count }) {
      dispatch(encoder, FIXED_TO_FLOAT, [uniform(new Uint32Array([count])), sums, field], groupsFor(count));
    },

    /**
     * The nodes around every point within `reach` (metres) of each of `count` points, set in `bits`
     * (one bit a node, 32 to a word; cleared by the caller).
     * @param {GPUCommandEncoder} encoder
     * @param {GridOnGpu} grid
     * @param {{ positions: GPUBuffer, count: number, reach: number, bits: GPUBuffer }} buffers
     */
    mark(encoder, grid, { positions, count, reach, bits }) {
      dispatch(encoder, MARK, [gridUniform(grid, count), positions, bits, uniform(new Float32Array([reach]))], groupsFor(count));
    },

    /**
     * `field` at `count` listed nodes, into lane `lane` of rows of `lanes` floats.
     * @param {GPUCommandEncoder} encoder
     * @param {{ field: GPUBuffer, nodes: GPUBuffer, count: number, out: GPUBuffer, lane: number, lanes: number }} buffers
     */
    gather(encoder, { field, nodes, count, out, lane, lanes }) {
      dispatch(encoder, GATHER, [uniform(new Uint32Array([count, lane, lanes])), field, nodes, out], groupsFor(count));
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
      dispatch(encoder, CORRELATE, [uniform(new Uint32Array([count, nodes[axis], stride, radius])), weights, field, out], groupsFor(count));
    },
  };
}
