/**
 * The ensemble's networks on WebGPU. Each hidden layer is one tiled matrix-multiply dispatch: 128
 * points by 128 neurons per workgroup, 8 by 8 per thread, the input standardised and clipped on the
 * way in for the first layer, GELU on the way out. The output neuron is a dot product per point.
 * Points go through in chunks so the buffers stay small and other GPU work (drawing the page) gets turns.
 */

const TILE = 128;

const SHADER = /* wgsl */ `
struct Layer {
  inputs: u32,
  outputs: u32,
  weight: u32,
  bias: u32,
  mean: u32,
  deviation: u32,
  standardize: u32,
  clip: f32,
}

struct Output {
  inputs: u32,
  weight: u32,
  bias: u32,
  destination: u32,
  scale: f32,
  shift: f32,
}

// Every array in the weights starts on a multiple of four floats (export_web.py), so all reads are vec4.
@group(0) @binding(0) var<storage, read> weights: array<vec4f>;
@group(0) @binding(1) var<storage, read> x: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> y: array<vec4f>;
@group(0) @binding(3) var<uniform> layer: Layer;
@group(0) @binding(4) var<uniform> output: Output;
@group(0) @binding(5) var<storage, read_write> models: array<f32>;

// Sixteen steps of k for 128 rows of the input and 128 neurons of the weights.
var<workgroup> xTile: array<array<f32, 128>, 16>;
var<workgroup> wTile: array<array<f32, 128>, 16>;

fn tileQuad(tile: ptr<workgroup, array<array<f32, 128>, 16>>, k: u32, first: u32) -> vec4f {
  return vec4f((*tile)[k][first], (*tile)[k][first + 1u], (*tile)[k][first + 2u], (*tile)[k][first + 3u]);
}

// x Phi(x) with erf from Abramowitz and Stegun 7.1.26, as network.js computes it.
fn gelu(v: vec4f) -> vec4f {
  let z = abs(v) * 0.70710678118654752;
  let t = 1.0 / (1.0 + 0.3275911 * z);
  let e = 1.0 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-z * z);
  return 0.5 * v * (1.0 + select(e, -e, v < vec4f(0.0)));
}

@compute @workgroup_size(16, 16)
fn dense(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_id) local: vec3u,
         @builtin(local_invocation_index) thread: u32) {
  let row0 = group.x * 128u;
  let col0 = group.y * 128u;
  let quads = layer.inputs / 4u;
  // acc[i][h]: row (i < 4 ? 4 ty + i : 64 + 4 ty + i - 4), neurons 4 (tx + 16 h) .. + 3.
  var acc: array<array<vec4f, 2>, 8>;
  for (var k0 = 0u; k0 < quads; k0 += 4u) {
    // Each thread loads two groups of four k for one row of inputs and one neuron of weights, and
    // stores them transposed one float at a time (no two threads write the same word).
    for (var e = thread; e < 512u; e += 256u) {
      let r = e >> 2u;
      let q = e & 3u;
      var v = x[(row0 + r) * quads + k0 + q];
      if (layer.standardize == 1u) {
        let mean = weights[layer.mean / 4u + k0 + q];
        let deviation = weights[layer.deviation / 4u + k0 + q];
        v = clamp((v - mean) / deviation, vec4f(-layer.clip), vec4f(layer.clip));
      }
      let w = weights[layer.weight / 4u + (col0 + r) * quads + k0 + q];
      for (var c = 0u; c < 4u; c++) {
        xTile[4u * q + c][r] = v[c];
        wTile[4u * q + c][r] = w[c];
      }
    }
    workgroupBarrier();
    for (var k = 0u; k < 16u; k++) {
      let a0 = tileQuad(&xTile, k, 4u * local.y);
      let a1 = tileQuad(&xTile, k, 64u + 4u * local.y);
      let b0 = tileQuad(&wTile, k, 4u * local.x);
      let b1 = tileQuad(&wTile, k, 64u + 4u * local.x);
      for (var i = 0u; i < 4u; i++) {
        acc[i][0] += a0[i] * b0;
        acc[i][1] += a0[i] * b1;
        acc[i + 4u][0] += a1[i] * b0;
        acc[i + 4u][1] += a1[i] * b1;
      }
    }
    workgroupBarrier();
  }
  let outQuads = layer.outputs / 4u;
  for (var i = 0u; i < 8u; i++) {
    let row = row0 + select(4u * local.y + i, 60u + 4u * local.y + i, i >= 4u);
    for (var h = 0u; h < 2u; h++) {
      let quad = col0 / 4u + local.x + 16u * h;
      y[row * outQuads + quad] = gelu(acc[i][h] + weights[layer.bias / 4u + quad]);
    }
  }
}

@compute @workgroup_size(64)
fn last(@builtin(global_invocation_id) id: vec3u) {
  let row = id.x;
  let quads = output.inputs / 4u;
  var sum = vec4f(0.0);
  for (var q = 0u; q < quads; q++) {
    sum += x[row * quads + q] * weights[output.weight / 4u + q];
  }
  let bias = weights[output.bias / 4u].x;
  models[output.destination + row] = (sum.x + sum.y + sum.z + sum.w + bias) * output.scale + output.shift;
}
`;

/**
 * @typedef {Object} GpuNetworks
 * @property {string} name   the adapter as the browser names it
 * @property {number} stride floats per row of the features `predict` takes: the feature count rounded
 *   up to a tile, the rest of each row zero
 * @property {(features: Float32Array, count: number, out: Float32Array) => Promise<void>} predict
 *   Each network's Cp at `count` points from their features (count, stride), into `out` (networks, count).
 * @property {boolean} lost  the browser took the device away; new networks are needed to go on
 */

/**
 * The networks on this machine's GPU, or null when the browser offers no hardware WebGPU adapter.
 * @param {import('./network.js').Ensemble} ensemble
 * @returns {Promise<GpuNetworks | null>}
 */
export async function createGpuNetworks(ensemble) {
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
  // A software adapter (SwiftShader, where the browser has no usable GPU) runs the kernels on the
  // CPU far slower than the helpers' own path.
  if (!adapter || adapter.info.isFallbackAdapter) return null;
  const packed = packNetworks(ensemble);
  const { stride, width } = packed;
  const networks = ensemble.folds.length;
  // A chunk's activations stay within 64 MiB, inside WebGPU's default binding limit.
  const chunk = Math.max(TILE, Math.floor((64 << 20) / (4 * Math.max(width, stride)) / TILE) * TILE);
  const device = await adapter.requestDevice();
  // The browser can reset the GPU under it (a driver reset, its graphics process restarting); the
  // device is then gone for good and the engine makes a new one for the next file.
  let lost = false;
  device.lost.then(() => {
    lost = true;
  });

  device.pushErrorScope('validation');
  const module = device.createShaderModule({ code: SHADER });
  const densePipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'dense' } });
  const lastPipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'last' } });
  const invalid = await device.popErrorScope();
  if (invalid) throw new Error(`The WebGPU networks failed to compile: ${invalid.message}`);

  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const weights = device.createBuffer({ size: packed.weights.byteLength, usage: storage });
  device.queue.writeBuffer(weights, 0, packed.weights);
  const features = device.createBuffer({ size: chunk * stride * 4, usage: storage });
  const hidden = [0, 1].map(() => device.createBuffer({ size: chunk * width * 4, usage: storage }));
  const models = device.createBuffer({ size: networks * chunk * 4, usage: storage });
  const readback = device.createBuffer({ size: networks * chunk * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

  const uniform = (words) => {
    const buffer = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const data = new ArrayBuffer(32);
    const u32 = new Uint32Array(data);
    const f32 = new Float32Array(data);
    words.forEach(([kind, value], i) => {
      if (kind === 'u32') u32[i] = value;
      else f32[i] = value;
    });
    device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  };

  // Every dispatch's bindings are fixed, so they are made once: per network, each hidden layer and the output.
  const passes = packed.folds.map((fold, f) => {
    const hiddenLayers = fold.layers.slice(0, -1);
    const steps = hiddenLayers.map((layer, l) => {
      const params = uniform([
        ['u32', layer.inputs], ['u32', layer.outputs], ['u32', layer.weight], ['u32', layer.bias],
        ['u32', fold.xMean], ['u32', fold.xStd], ['u32', l === 0 ? 1 : 0], ['f32', ensemble.zClip],
      ]);
      return {
        outputs: layer.outputs,
        bindGroup: device.createBindGroup({
          layout: densePipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: weights } },
            { binding: 1, resource: { buffer: l === 0 ? features : hidden[(l - 1) % 2] } },
            { binding: 2, resource: { buffer: hidden[l % 2] } },
            { binding: 3, resource: { buffer: params } },
          ],
        }),
      };
    });
    const last = fold.layers.at(-1);
    const params = uniform([
      ['u32', last.inputs], ['u32', last.weight], ['u32', last.bias], ['u32', f * chunk],
      ['f32', fold.yStd], ['f32', fold.yMean],
    ]);
    const lastBindGroup = device.createBindGroup({
      layout: lastPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: weights } },
        { binding: 1, resource: { buffer: hidden[(hiddenLayers.length - 1) % 2] } },
        { binding: 4, resource: { buffer: params } },
        { binding: 5, resource: { buffer: models } },
      ],
    });
    return { steps, lastBindGroup };
  });

  async function run(input, count, out) {
    for (let start = 0; start < count; start += chunk) {
      const rows = Math.min(chunk, count - start);
      const tiles = Math.ceil(rows / TILE);
      device.queue.writeBuffer(features, 0, input, start * stride, rows * stride);
      const encoder = device.createCommandEncoder();
      for (const { steps, lastBindGroup } of passes) {
        for (const step of steps) {
          const pass = encoder.beginComputePass();
          pass.setPipeline(densePipeline);
          pass.setBindGroup(0, step.bindGroup);
          pass.dispatchWorkgroups(tiles, step.outputs / TILE);
          pass.end();
        }
        const pass = encoder.beginComputePass();
        pass.setPipeline(lastPipeline);
        pass.setBindGroup(0, lastBindGroup);
        pass.dispatchWorkgroups(Math.ceil(rows / 64));
        pass.end();
      }
      encoder.copyBufferToBuffer(models, 0, readback, 0, networks * chunk * 4);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const result = new Float32Array(readback.getMappedRange());
      for (let f = 0; f < networks; f++) out.set(result.subarray(f * chunk, f * chunk + rows), f * count + start);
      readback.unmap();
    }
  }

  // Calls share the device buffers, so each waits for the one before it.
  let previous = Promise.resolve();
  function predict(input, count, out) {
    const call = previous.then(() => run(input, count, out)).catch(async (error) => {
      // Work cut off by a reset fails before or after `lost` settles; give it a moment to say which.
      await Promise.race([device.lost, new Promise((resolve) => setTimeout(resolve, 100))]);
      throw lost ? new Error('The browser reset the graphics processor during the analysis. Drop the file again to restart it.') : error;
    });
    previous = call.catch(() => {});
    return call;
  }

  // One pass over a tile of zeros compiles the pipelines now rather than on the first car.
  await predict(new Float32Array(TILE * stride), TILE, new Float32Array(networks * TILE));

  const info = adapter.info;
  return {
    name: info.description || vendorName(info.vendor),
    stride,
    predict,
    get lost() {
      return lost;
    },
  };
}

/**
 * The networks laid out for the kernels: every hidden layer's inputs padded to a multiple of 16 and
 * its neurons to a multiple of 128 with zero weights (a padded neuron outputs GELU(0) = 0 and the next
 * layer ignores it), the input mean and deviation padded with 0 and 1. Offsets are in floats.
 * @param {import('./network.js').Ensemble} ensemble
 */
function packNetworks(ensemble) {
  const pad = (n, to) => Math.ceil(n / to) * to;
  const inputs = ensemble.features.count;
  const stride = pad(inputs, 16);
  const blocks = [];
  let offset = 0;
  const place = (array) => {
    blocks.push(array);
    const at = offset;
    offset += array.length;
    return at;
  };
  let width = 0;
  const folds = ensemble.folds.map((fold) => {
    const xMean = new Float32Array(stride);
    const xStd = new Float32Array(stride).fill(1);
    xMean.set(fold.xMean);
    xStd.set(fold.xStd);
    let previous = stride;
    const layers = fold.layers.map((layer, l) => {
      const isLast = l === fold.layers.length - 1;
      const outputs = isLast ? 1 : pad(layer.outputs, TILE);
      const weight = new Float32Array(Math.max(outputs, 4) * previous);
      for (let o = 0; o < layer.outputs; o++) {
        weight.set(layer.weight.subarray(o * layer.inputs, (o + 1) * layer.inputs), o * previous);
      }
      const bias = new Float32Array(Math.max(outputs, 4));
      bias.set(layer.bias);
      const packedLayer = { inputs: previous, outputs, weight: place(weight), bias: place(bias) };
      if (!isLast) width = Math.max(width, outputs);
      previous = outputs;
      return packedLayer;
    });
    return { xMean: place(xMean), xStd: place(xStd), yMean: fold.yMean, yStd: fold.yStd, layers };
  });
  const weights = new Float32Array(offset);
  let at = 0;
  for (const block of blocks) {
    weights.set(block, at);
    at += block.length;
  }
  return { stride, width, folds, weights };
}

function vendorName(vendor) {
  const names = { apple: 'Apple', nvidia: 'NVIDIA', amd: 'AMD', intel: 'Intel', qualcomm: 'Qualcomm', arm: 'Arm' };
  return names[vendor] ?? vendor;
}
