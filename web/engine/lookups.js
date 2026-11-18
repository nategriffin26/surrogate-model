/**
 * A car's lookups: what the features read, built once from all of its polygons as `SurfaceContext`
 * in `features.py` builds them. The raster is a 2 cm picture of the car that the sections across it
 * are read from: how far the body reaches from the centre plane at each (x, z), the body's top and
 * bottom over each (x, y) column, and each section's area S(x). Each surface scale is the area and
 * area-weighted normal spread on a grid and blurred, whole and cut to its upstream half; its grids
 * are built on the GPU when there is one (`grid-gpu.js`), else on the CPU (`grid.js`), and the same
 * lookups come out of both.
 *
 * A surface scale's fields are kept only at the nodes around the polygon centres and the file's
 * points (a few percent of its grid): every read at a polygon, and at a working-mesh vertex, which
 * sits on a file point.
 */

import { correlateAlong, spreadOnGrid } from './grid.js';

/**
 * How far from a polygon centre or a file point a read must still find its nodes kept: beyond the
 * float32 rounding of the GPU's grid coordinates (1 µm), and beyond a bundled example's rounding of
 * its vertex positions (`packResult`, half of 1/65535 of the car: 0.04 mm on a 5 m car).
 */
const KEPT_REACH = 5e-5;

/**
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
 * @property {Float32Array} yOut         (2, nx, nz) of the raster: how far the body reaches from the
 *   centre plane in each (x, z) bin, [0] toward −y, [1] toward +y; 0 where it does not reach the bin
 * @property {Float32Array} zTop         (nx, ny) the highest polygon centre over each (x, y) column,
 *   with y measured from the centre plane; the ground (the lowest centre) where the column holds none
 * @property {Float32Array} zBottom      (nx, ny) the lowest, the ground where the column holds none
 * @property {Float32Array} sectionArea  (nx) the section across the car at each x, S(x) in m²: the
 *   columns' heights summed across y, times the raster step
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
  const scaleGrids = layout.scales.map((scale) => scaleGrid(scale, bounds));
  const scales = gpu ? await scalesOnGpu(gpu, cells, points, layout, scaleGrids) : scalesOnCpu(cells, points, layout, scaleGrids);
  return {
    bounds,
    yCenter,
    ...carRaster(cells, layout, yCenter, bounds[4]),
    scales: scaleGrids.map(({ origin, nodes }, s) => ({ origin, nodes, ...scales[s] })),
  };
}

/** A scale's grid: the polygon centres' bounds and a margin of the Gaussian's reach and two nodes. */
function scaleGrid({ step, radius }, bounds) {
  const margin = (radius + 2) * step;
  const origin = [bounds[0] - margin, bounds[2] - margin, bounds[4] - margin];
  const nodes = [1, 3, 5].map((k, axis) => Math.ceil((bounds[k] + margin - origin[axis]) / step) + 1);
  return { origin, step, nodes };
}

/**
 * The raster, from every polygon centre in the reference's float32 arithmetic, so each centre lands
 * in the bin it lands in there; S(x) sums each row in NumPy's order and so is the reference's to the bit.
 */
function carRaster({ centers, areas }, { h, x0, yHalf, z0, nx, ny, nz }, yCenter, zGround) {
  const f32 = Math.fround;
  const yCenter32 = f32(yCenter);
  const yOut = new Float32Array(2 * nx * nz);
  const zTop = new Float32Array(nx * ny).fill(-Infinity);
  const zBottom = new Float32Array(nx * ny).fill(Infinity);
  for (let c = 0; c < areas.length; c++) {
    const z = centers[3 * c + 2];
    const yRelative = f32(centers[3 * c + 1] - yCenter32);
    const ix = clamp(Math.trunc(f32(f32(centers[3 * c] - x0) / h)), nx);
    const iy = clamp(Math.trunc(f32(f32(yRelative + yHalf) / h)), ny);
    const iz = clamp(Math.trunc(f32(f32(z - z0) / h)), nz);
    const reach = ((yRelative >= 0 ? nx : 0) + ix) * nz + iz;
    if (Math.abs(yRelative) > yOut[reach]) yOut[reach] = Math.abs(yRelative);
    const column = ix * ny + iy;
    if (z > zTop[column]) zTop[column] = z;
    if (z < zBottom[column]) zBottom[column] = z;
  }
  const heights = new Float32Array(nx * ny);
  for (let column = 0; column < heights.length; column++) {
    if (zTop[column] === -Infinity) {
      zTop[column] = zGround;
      zBottom[column] = zGround;
    }
    heights[column] = zTop[column] - zBottom[column];
  }
  const sectionArea = new Float32Array(nx);
  for (let ix = 0; ix < nx; ix++) sectionArea[ix] = f32(numpySum32(heights, ix * ny, ny) * h);
  return { yOut, zTop, zBottom, sectionArea };
}

function clamp(index, size) {
  return index < 0 ? 0 : index >= size ? size - 1 : index;
}

/**
 * values[start, start + n) summed in float32 in NumPy's order (`pairwise_sum`): eight running sums
 * for up to 128 values, joined in pairs, and halves summed apart beyond that.
 */
function numpySum32(values, start, n) {
  const f32 = Math.fround;
  if (n < 8) {
    let sum = 0;
    for (let i = 0; i < n; i++) sum = f32(sum + values[start + i]);
    return sum;
  }
  if (n <= 128) {
    const r = Array.from(values.subarray(start, start + 8));
    let i = 8;
    for (; i < n - (n % 8); i += 8) {
      for (let j = 0; j < 8; j++) r[j] = f32(r[j] + values[start + i + j]);
    }
    let sum = f32(f32(f32(r[0] + r[1]) + f32(r[2] + r[3])) + f32(f32(r[4] + r[5]) + f32(r[6] + r[7])));
    for (; i < n; i++) sum = f32(sum + values[start + i]);
    return sum;
  }
  let half = Math.floor(n / 2);
  half -= half % 8;
  return f32(numpySum32(values, start, half) + numpySum32(values, start + half, n - half));
}

/** The surface's four carried quantities at each polygon: area, then area × each normal component. */
function carried({ normals, areas }, quantity, out) {
  for (let c = 0; c < areas.length; c++) out[c] = quantity === 0 ? areas[c] : areas[c] * normals[3 * c + quantity - 1];
  return out;
}

function scalesOnCpu(cells, points, layout, scaleGrids) {
  const { centers, areas } = cells;
  const quantity = new Float64Array(areas.length);
  return layout.scales.map((scale, s) => {
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

async function scalesOnGpu(gpu, cells, points, layout, scaleGrids) {
  const { device } = gpu;
  const { centers, areas } = cells;
  const count = areas.length;
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

    // Each scale, one carried quantity at a time, so only one quantity's grids are held at once.
    const quantity = new Float32Array(count);
    const weights = buffer(4 * count);
    const scales = [];
    for (let s = 0; s < scaleGrids.length; s++) {
      const grid = scaleGrids[s];
      const { radius } = layout.scales[s];
      const n = nodeCount(grid);
      const keptBuffer = upload(kept[s]);
      const [sums, spread, yz, blurred] = [buffer(8 * n), buffer(4 * n), buffer(4 * n), buffer(4 * n)];
      const whole = buffer(16 * kept[s].length);
      const ahead = buffer(16 * kept[s].length);
      const gaussian = upload(Float32Array.from(layout.scales[s].weights));
      const upstreamHalf = upload(Float32Array.from(layout.scales[s].upstream));
      for (let q = 0; q < 4; q++) {
        device.queue.writeBuffer(weights, 0, carried(cells, q, quantity));
        submit((encoder) => {
          encoder.clearBuffer(sums);
          gpu.spread(encoder, grid, { positions, weights, count, sums });
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
    return scales;
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
