/**
 * A car's lookups: what the features read, built once from all of its polygons as `SurfaceContext`
 * in `features.py` builds them. The plan-view outline is a raster of how far the body reaches from
 * the centre plane; the flow is the potential of the car's source sheet after the wall condition's
 * passes; each surface scale is the area and area-weighted normal spread on a grid and blurred,
 * whole and cut to its upstream half. The grids are built on the GPU when there is one
 * (`grid-gpu.js`), else on the CPU (`grid.js`), and the same lookups come out of both.
 *
 * A surface scale's fields are kept only at the nodes around the polygon centres and the file's
 * points (a few percent of its grid): every read at a polygon, and at a working-mesh vertex, which
 * sits on a file point.
 */

import { correlateAlong, potential, potentialSpectrum, readGradient, spreadOnGrid } from './grid.js';

/**
 * How far from a polygon centre or a file point a read must still find its nodes kept: beyond the
 * float32 rounding of the GPU's grid coordinates (1 µm), and beyond a bundled example's rounding of
 * its vertex positions (`packResult`, half of 1/65535 of the car: 0.04 mm on a 5 m car).
 */
const KEPT_REACH = 5e-5;

/**
 * @typedef {Object} FlowLookup  The potential of the car's sources and their image under the
 *   ground, on the flow grid (`FeatureLayout.flow` gives its step and nodes).
 * @property {number[]} origin
 * @property {Float32Array} potential  (nx, ny, nz)
 *
 * @typedef {Object} ScaleLookup  The surface around the car at one scale, kept at the nodes around it.
 * @property {number[]} origin
 * @property {number[]} nodes
 * @property {Int32Array} kept       (K) flat numbers of the nodes kept, ascending
 * @property {Float32Array} whole    (K, 4) blurred area, then area × each normal component
 * @property {Float32Array} ahead    (K, 4) the same with the Gaussian cut to its upstream half
 *
 * @typedef {Object} SurfaceLookups
 * @property {number[]} bounds   [xmin, xmax, ymin, ymax, zmin, zmax] of the polygon centres
 * @property {number} yCenter
 * @property {Float32Array} yOut (2, nx, nz) of the raster: how far the body reaches from the centre
 *   plane in each (x, z) column, [0] toward −y, [1] toward +y
 * @property {FlowLookup} flow
 * @property {ScaleLookup[]} scales  in `FeatureLayout.scales` order
 */

/**
 * @param {{ centers: Float32Array, normals: Float32Array, areas: Float64Array, bounds: number[] }} cells
 * @param {Float32Array} points  the file's points (P, 3)
 * @param {import('./features.js').FeatureLayout} layout
 * @param {ReturnType<import('./grid-gpu.js').createGridOperations> | null} gpu  the GPU's grid operations, or none
 * @returns {Promise<SurfaceLookups>}
 */
export async function buildLookups(cells, points, layout, gpu) {
  const { bounds } = cells;
  const yCenter = 0.5 * (bounds[2] + bounds[3]);
  const flowGrid = { origin: flowOrigin(layout.flow, yCenter, bounds[4]), step: layout.flow.step, nodes: layout.flow.nodes };
  const scaleGrids = layout.scales.map((scale) => scaleGrid(scale, bounds));
  const built = gpu ? await gridsOnGpu(gpu, cells, points, layout, flowGrid, scaleGrids) : gridsOnCpu(cells, points, layout, flowGrid, scaleGrids);
  return {
    bounds,
    yCenter,
    yOut: outlineRaster(cells, layout, yCenter),
    flow: { origin: flowGrid.origin, potential: built.potential },
    scales: scaleGrids.map(({ origin, nodes }, s) => ({ origin, nodes, ...built.scales[s] })),
  };
}

/** Nodes symmetric about the ground (the lowest polygon centre) in z, so the mirror image is the grid flipped in z. */
function flowOrigin({ x0, step, nodes }, yCenter, zGround) {
  return [x0, yCenter - 0.5 * (nodes[1] - 1) * step, zGround - (nodes[2] / 2 - 0.5) * step];
}

/** A scale's grid: the polygon centres' bounds and a margin of the Gaussian's reach and two nodes. */
function scaleGrid({ step, radius }, bounds) {
  const margin = (radius + 2) * step;
  const origin = [bounds[0] - margin, bounds[2] - margin, bounds[4] - margin];
  const nodes = [1, 3, 5].map((k, axis) => Math.ceil((bounds[k] + margin - origin[axis]) / step) + 1);
  return { origin, step, nodes };
}

/**
 * The plan-view outline: for each (x, z) column of the raster and each side of the centre plane, the
 * farthest polygon centre from it, in the reference's float32 arithmetic.
 */
function outlineRaster({ centers, areas }, { h, x0, z0, nx, nz }, yCenter) {
  const f32 = Math.fround;
  const yCenter32 = f32(yCenter);
  const yOut = new Float32Array(2 * nx * nz);
  for (let c = 0; c < areas.length; c++) {
    const yRelative = f32(centers[3 * c + 1] - yCenter32);
    const ix = clamp(Math.trunc(f32(f32(centers[3 * c] - x0) / h)), nx);
    const iz = clamp(Math.trunc(f32(f32(centers[3 * c + 2] - z0) / h)), nz);
    const reach = ((yRelative >= 0 ? nx : 0) + ix) * nz + iz;
    if (Math.abs(yRelative) > yOut[reach]) yOut[reach] = Math.abs(yRelative);
  }
  return yOut;
}

function clamp(index, size) {
  return index < 0 ? 0 : index >= size ? size - 1 : index;
}

/** The surface's four carried quantities at each polygon: area, then area × each normal component. */
function carried({ normals, areas }, quantity, out) {
  for (let c = 0; c < areas.length; c++) out[c] = quantity === 0 ? areas[c] : areas[c] * normals[3 * c + quantity - 1];
  return out;
}

function gridsOnCpu(cells, points, layout, flowGrid, scaleGrids) {
  const { centers, normals, areas } = cells;
  const count = areas.length;
  const { flow } = layout;

  // The source density σ starts as the sheet's own answer to the stream, −2 n_x; each pass moves it
  // part of the way to what the wall condition asks given the velocity the sheet induces.
  const spectrum = potentialSpectrum(flow.nodes, flow.step, flow.cubeMeanInverseDistance);
  const density = new Float64Array(count);
  for (let c = 0; c < count; c++) density[c] = -2 * normals[3 * c];
  const strengths = new Float64Array(count);
  const flowPotential = () => {
    for (let c = 0; c < count; c++) strengths[c] = density[c] * areas[c];
    return potential(flow.nodes, spectrum, spreadOnGrid(flowGrid, centers, strengths));
  };
  for (let pass = 0; pass < flow.passes; pass++) {
    const u = readGradient(flowGrid, flowPotential(), centers);
    for (let c = 0; c < count; c++) {
      const normal = 3 * c;
      const inward = u[normal] * normals[normal] + u[normal + 1] * normals[normal + 1] + u[normal + 2] * normals[normal + 2];
      density[c] += flow.relaxation * (-2 * normals[normal] - 2 * inward - density[c]);
    }
  }

  const quantity = new Float64Array(count);
  const scales = layout.scales.map((scale, s) => {
    const grid = scaleGrids[s];
    const kept = keptNodes(grid, centers, points);
    const whole = new Float32Array(4 * kept.length);
    const ahead = new Float32Array(4 * kept.length);
    for (let q = 0; q < 4; q++) {
      const spread = Float32Array.from(spreadOnGrid(grid, centers, carried(cells, q, quantity)));
      const yz = correlateAlong(correlateAlong(spread, grid.nodes, 1, scale.weights), grid.nodes, 2, scale.weights);
      gather(correlateAlong(yz, grid.nodes, 0, scale.weights), kept, whole, q);
      gather(correlateAlong(yz, grid.nodes, 0, scale.upstream), kept, ahead, q);
    }
    return { kept, whole, ahead };
  });
  return { potential: flowPotential(), scales };
}

/** The nodes around every polygon centre and file point, within `KEPT_REACH`, ascending. */
function keptNodes({ origin, step, nodes }, ...pointSets) {
  const [, ny, nz] = nodes;
  const marked = new Uint8Array(nodes[0] * ny * nz);
  const low = [0, 0, 0], high = [0, 0, 0];
  for (const points of pointSets) {
    for (let p = 0; p < points.length; p += 3) {
      for (let axis = 0; axis < 3; axis++) {
        const q = (points[p + axis] - origin[axis]) / step;
        const reach = KEPT_REACH / step;
        low[axis] = Math.min(Math.max(Math.floor(q - reach), 0), nodes[axis] - 2);
        high[axis] = Math.min(Math.max(Math.floor(q + reach), 0), nodes[axis] - 2) + 1;
      }
      for (let i = low[0]; i <= high[0]; i++) {
        for (let j = low[1]; j <= high[1]; j++) {
          for (let k = low[2]; k <= high[2]; k++) marked[(i * ny + j) * nz + k] = 1;
        }
      }
    }
  }
  let count = 0;
  for (let n = 0; n < marked.length; n++) count += marked[n];
  const kept = new Int32Array(count);
  for (let n = 0, i = 0; n < marked.length; n++) if (marked[n]) kept[i++] = n;
  return kept;
}

/** `field` at the kept nodes into lane `lane` of rows of four. */
function gather(field, kept, out, lane) {
  for (let i = 0; i < kept.length; i++) out[4 * i + lane] = field[kept[i]];
}

async function gridsOnGpu(gpu, cells, points, layout, flowGrid, scaleGrids) {
  const { device } = gpu;
  const { centers, normals, areas } = cells;
  const count = areas.length;
  const { flow } = layout;
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const created = [];
  const buffer = (bytes) => {
    const made = device.createBuffer({ size: Math.max(16, bytes), usage });
    created.push(made);
    return made;
  };
  const upload = (array) => {
    const made = buffer(array.byteLength);
    device.queue.writeBuffer(made, 0, array);
    return made;
  };
  const submit = (record) => {
    const encoder = device.createCommandEncoder();
    record(encoder);
    device.queue.submit([encoder.finish()]);
  };
  const download = async (source, bytes) => {
    const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    submit((encoder) => encoder.copyBufferToBuffer(source, 0, staging, 0, bytes));
    await staging.mapAsync(GPUMapMode.READ);
    const copy = staging.getMappedRange().slice(0);
    staging.destroy();
    return copy;
  };

  const build = async () => {
    const positions = upload(centers);
    const nodeCount = ({ nodes }) => nodes[0] * nodes[1] * nodes[2];

    // Which nodes each scale keeps.
    const filePoints = upload(points);
    const bits = scaleGrids.map((grid) => buffer(4 * Math.ceil(nodeCount(grid) / 32)));
    submit((encoder) => scaleGrids.forEach((grid, s) => {
      gpu.mark(encoder, grid, { positions, count, reach: KEPT_REACH, bits: bits[s] });
      gpu.mark(encoder, grid, { positions: filePoints, count: points.length / 3, reach: KEPT_REACH, bits: bits[s] });
    }));
    filePoints.destroy();
    const kept = await Promise.all(bits.map(async (set, s) => nodesOf(new Uint32Array(await download(set, set.size)), nodeCount(scaleGrids[s]))));

    // The flow: the wall condition's passes, then the potential of the final sheet.
    const flowNodes = nodeCount(flowGrid);
    const density = new Float32Array(count);
    const strength = new Float32Array(count);
    for (let c = 0; c < count; c++) {
      density[c] = -2 * normals[3 * c];
      strength[c] = density[c] * areas[c];
    }
    const flowBuffers = {
      normals: upload(normals), areas: upload(Float32Array.from(areas)), densities: upload(density), strengths: upload(strength),
      sums: buffer(8 * flowNodes), potential: buffer(4 * flowNodes),
    };
    const flowPotential = gpu.potential(flow.nodes, flow.step, flow.cubeMeanInverseDistance);
    const sheetPotential = (encoder) => {
      encoder.clearBuffer(flowBuffers.sums);
      gpu.spread(encoder, flowGrid, { positions, weights: flowBuffers.strengths, count, sums: flowBuffers.sums });
      gpu.sumsToField(encoder, { sums: flowBuffers.sums, field: flowBuffers.potential, count: flowNodes });
      flowPotential.encode(encoder, flowBuffers.potential, flowBuffers.potential);
    };
    submit((encoder) => {
      for (let pass = 0; pass < flow.passes; pass++) {
        sheetPotential(encoder);
        gpu.wall(encoder, flowGrid, { positions, count, relaxation: flow.relaxation, ...flowBuffers });
      }
      sheetPotential(encoder);
    });
    const flowField = new Float32Array(await download(flowBuffers.potential, 4 * flowNodes));
    flowPotential.destroy();
    for (const name of ['normals', 'areas', 'densities', 'sums', 'potential']) flowBuffers[name].destroy();

    // Each scale, one carried quantity at a time, so only one quantity's grids are held at once.
    const quantity = new Float32Array(count);
    const scales = [];
    for (let s = 0; s < scaleGrids.length; s++) {
      const grid = scaleGrids[s];
      const { weights, upstream, radius } = layout.scales[s];
      const n = nodeCount(grid);
      const keptBuffer = upload(kept[s]);
      const [sums, spread, yz, blurred] = [buffer(8 * n), buffer(4 * n), buffer(4 * n), buffer(4 * n)];
      const whole = buffer(16 * kept[s].length);
      const ahead = buffer(16 * kept[s].length);
      const gaussian = upload(Float32Array.from(weights));
      const upstreamHalf = upload(Float32Array.from(upstream));
      for (let q = 0; q < 4; q++) {
        device.queue.writeBuffer(flowBuffers.strengths, 0, carried(cells, q, quantity));
        submit((encoder) => {
          encoder.clearBuffer(sums);
          gpu.spread(encoder, grid, { positions, weights: flowBuffers.strengths, count, sums });
          gpu.sumsToField(encoder, { sums, field: spread, count: n });
          gpu.correlate(encoder, grid.nodes, 1, { weights: gaussian, radius, field: spread, out: blurred });
          gpu.correlate(encoder, grid.nodes, 2, { weights: gaussian, radius, field: blurred, out: yz });
          gpu.correlate(encoder, grid.nodes, 0, { weights: gaussian, radius, field: yz, out: blurred });
          gpu.gather(encoder, { field: blurred, nodes: keptBuffer, count: kept[s].length, out: whole, lane: q, lanes: 4 });
          gpu.correlate(encoder, grid.nodes, 0, { weights: upstreamHalf, radius, field: yz, out: blurred });
          gpu.gather(encoder, { field: blurred, nodes: keptBuffer, count: kept[s].length, out: ahead, lane: q, lanes: 4 });
        });
      }
      scales.push({
        kept: kept[s],
        whole: new Float32Array(await download(whole, whole.size)),
        ahead: new Float32Array(await download(ahead, ahead.size)),
      });
      for (const made of [keptBuffer, sums, spread, yz, blurred, whole, ahead, gaussian, upstreamHalf]) made.destroy();
    }

    return { potential: flowField, scales };
  };

  device.pushErrorScope('out-of-memory');
  device.pushErrorScope('validation');
  const built = await build().then((value) => ({ value }), (error) => ({ error })).finally(() => {
    for (const made of created) made.destroy();
  });
  const invalid = await device.popErrorScope();
  const outOfMemory = await device.popErrorScope();
  if (built.error) throw built.error;
  if (outOfMemory) throw new Error(`The GPU ran out of memory building the car's lookups: ${outOfMemory.message}`);
  if (invalid) throw new Error(`The GPU could not build the car's lookups: ${invalid.message}`);
  return built.value;
}

/** The set bits of a bitset of `count` nodes, as ascending node numbers. */
function nodesOf(words, count) {
  let total = 0;
  for (let w = 0; w < words.length; w++) {
    let word = words[w];
    while (word) {
      word &= word - 1;
      total++;
    }
  }
  const kept = new Int32Array(total);
  let i = 0;
  for (let n = 0; n < count; n++) if (words[n >>> 5] & (1 << (n & 31))) kept[i++] = n;
  return kept;
}
