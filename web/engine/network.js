/**
 * The ensemble's networks (`PerCellMLP`: the features in, hidden layers with GELU, one Cp out), loaded
 * from `web/model/` with whatever shape the model file gives, and their evaluation on the CPU. `gpu.js`
 * evaluates the same networks on WebGPU.
 */

/**
 * @typedef {{ inputs: number, outputs: number, weight: Float32Array, bias: Float32Array }} Layer
 *   `weight` is outputs x inputs, row-major.
 * @typedef {{ xMean: Float32Array, xStd: Float32Array, yMean: number, yStd: number, layers: Layer[] }} Fold
 * @typedef {{ features: import('./features.js').FeatureDefinition, zClip: number, activation: string, spreadScale: number,
 *             folds: Fold[], card: object, weightsSha256: string }} Ensemble
 *   As `export_web.py` writes `ensemble.json`.
 */

/** @returns {Promise<Ensemble>} */
export async function loadEnsemble(baseUrl) {
  const [spec, buffer] = await Promise.all([
    fetch(new URL('ensemble.json', baseUrl)).then(expectOk).then((response) => response.json()),
    fetch(new URL('weights.bin', baseUrl)).then(expectOk).then((response) => response.arrayBuffer()),
  ]);
  const weights = new Float32Array(buffer);
  const take = (offset, length) => weights.subarray(offset, offset + length);
  const inputs = spec.features.count;
  // The kernels here and in gpu.js compute GELU in its erf form, PyTorch's default.
  if (spec.activation !== 'GELU') throw new Error(`The model's networks use ${spec.activation} between layers; this page computes GELU.`);
  if (!(spec.spreadScale > 0)) throw new Error('The model file has no spread scale: export it again from a current checkpoint.');
  const folds = spec.folds.map((fold) => {
    if (fold.layers[0].inputs !== inputs || fold.layers.at(-1).outputs !== 1) {
      throw new Error(`The model file's networks take ${fold.layers[0].inputs} inputs; its features number ${inputs}.`);
    }
    return {
      xMean: take(fold.xMean, inputs),
      xStd: take(fold.xStd, inputs),
      yMean: fold.yMean,
      yStd: fold.yStd,
      layers: fold.layers.map((layer) => ({
        inputs: layer.inputs,
        outputs: layer.outputs,
        weight: take(layer.weight, layer.inputs * layer.outputs),
        bias: take(layer.bias, layer.outputs),
      })),
    };
  });
  const { features, zClip, activation, spreadScale, card, weightsSha256 } = spec;
  return { features, zClip, activation, spreadScale, folds, card, weightsSha256 };
}

function expectOk(response) {
  if (!response.ok) throw new Error(`The model files could not be loaded (${response.status} ${response.url}). Reload the page.`);
  return response;
}

/**
 * GELU as PyTorch's default computes it, x Φ(x), with erf from Abramowitz and Stegun 7.1.26
 * (absolute error below 1.5e-7). `gpu.js` uses the same formula.
 */
function gelu(x) {
  const z = Math.abs(x) * Math.SQRT1_2;
  const t = 1 / (1 + 0.3275911 * z);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-z * z);
  return 0.5 * x * (1 + (x < 0 ? -y : y));
}

const BLOCK = 32;

/**
 * Each network's Cp at `count` points from their features.
 * @param {Ensemble} ensemble
 * @param {Float32Array} features (count, inputs)
 * @param {Float32Array} out      (networks, count) network by network
 */
export function predictOnCpu(ensemble, features, count, out) {
  const { zClip } = ensemble;
  const inputs = ensemble.features.count;
  const width = Math.max(...ensemble.folds.flatMap((fold) => fold.layers.map((layer) => layer.outputs)));
  const input = new Float32Array(BLOCK * inputs);
  const hidden = [new Float32Array(BLOCK * width), new Float32Array(BLOCK * width)];

  for (let start = 0; start < count; start += BLOCK) {
    const rows = Math.min(BLOCK, count - start);
    for (const [f, fold] of ensemble.folds.entries()) {
      for (let r = 0; r < BLOCK; r++) {
        for (let i = 0; i < inputs; i++) {
          const value = r < rows ? Math.fround((features[(start + r) * inputs + i] - fold.xMean[i]) / fold.xStd[i]) : 0;
          input[r * inputs + i] = value < -zClip ? -zClip : value > zClip ? zClip : value;
        }
      }
      let source = input;
      let target = 0;
      const last = fold.layers.length - 1;
      for (let l = 0; l < last; l++) {
        dense(source, fold.layers[l], hidden[target]);
        source = hidden[target];
        target = 1 - target;
      }
      const { weight, bias, inputs: hiddenWidth } = fold.layers[last];
      for (let r = 0; r < rows; r++) {
        let sum = bias[0];
        for (let i = 0; i < hiddenWidth; i++) sum += weight[i] * source[r * hiddenWidth + i];
        out[f * count + start + r] = Math.fround(sum) * fold.yStd + fold.yMean;
      }
    }
  }
}

/** One hidden layer for a block of rows, four rows by four outputs at a time, then GELU. */
function dense(input, { inputs, outputs, weight, bias }, output) {
  const whole = outputs - (outputs % 4);
  for (let o = 0; o < whole; o += 4) {
    const w0 = o * inputs, w1 = w0 + inputs, w2 = w1 + inputs, w3 = w2 + inputs;
    for (let r = 0; r < BLOCK; r += 4) {
      const x0 = r * inputs, x1 = x0 + inputs, x2 = x1 + inputs, x3 = x2 + inputs;
      let a00 = 0, a01 = 0, a02 = 0, a03 = 0, a10 = 0, a11 = 0, a12 = 0, a13 = 0;
      let a20 = 0, a21 = 0, a22 = 0, a23 = 0, a30 = 0, a31 = 0, a32 = 0, a33 = 0;
      for (let i = 0; i < inputs; i++) {
        const v0 = weight[w0 + i], v1 = weight[w1 + i], v2 = weight[w2 + i], v3 = weight[w3 + i];
        const u0 = input[x0 + i], u1 = input[x1 + i], u2 = input[x2 + i], u3 = input[x3 + i];
        a00 += u0 * v0; a01 += u0 * v1; a02 += u0 * v2; a03 += u0 * v3;
        a10 += u1 * v0; a11 += u1 * v1; a12 += u1 * v2; a13 += u1 * v3;
        a20 += u2 * v0; a21 += u2 * v1; a22 += u2 * v2; a23 += u2 * v3;
        a30 += u3 * v0; a31 += u3 * v1; a32 += u3 * v2; a33 += u3 * v3;
      }
      const b0 = bias[o], b1 = bias[o + 1], b2 = bias[o + 2], b3 = bias[o + 3];
      let y = r * outputs + o;
      output[y] = gelu(a00 + b0); output[y + 1] = gelu(a01 + b1); output[y + 2] = gelu(a02 + b2); output[y + 3] = gelu(a03 + b3);
      y += outputs;
      output[y] = gelu(a10 + b0); output[y + 1] = gelu(a11 + b1); output[y + 2] = gelu(a12 + b2); output[y + 3] = gelu(a13 + b3);
      y += outputs;
      output[y] = gelu(a20 + b0); output[y + 1] = gelu(a21 + b1); output[y + 2] = gelu(a22 + b2); output[y + 3] = gelu(a23 + b3);
      y += outputs;
      output[y] = gelu(a30 + b0); output[y + 1] = gelu(a31 + b1); output[y + 2] = gelu(a32 + b2); output[y + 3] = gelu(a33 + b3);
    }
  }
  // A layer whose width is not a multiple of four: the last few neurons one at a time, summed in the
  // same order as above.
  for (let o = whole; o < outputs; o++) {
    for (let r = 0; r < BLOCK; r++) {
      let sum = 0;
      for (let i = 0; i < inputs; i++) sum += input[r * inputs + i] * weight[o * inputs + i];
      output[r * outputs + o] = gelu(sum + bias[o]);
    }
  }
}

/**
 * Each network at one point, layer by layer: what its first layer reads (the features standardized
 * and clipped by its own scaling), each hidden layer's output after GELU, and its Cp. The arithmetic
 * is `predictOnCpu`'s in the same order, so each Cp is the same to the bit.
 * @param {Ensemble} ensemble
 * @param {Float32Array} features (inputs)
 * @returns {{ cpModels: Float32Array, activations: Float32Array[][] }}
 */
export function networkActivations(ensemble, features) {
  const { zClip } = ensemble;
  const cpModels = new Float32Array(ensemble.folds.length);
  const activations = ensemble.folds.map((fold, f) => {
    const input = new Float32Array(features.length);
    for (let i = 0; i < input.length; i++) {
      const value = Math.fround((features[i] - fold.xMean[i]) / fold.xStd[i]);
      input[i] = value < -zClip ? -zClip : value > zClip ? zClip : value;
    }
    const layers = [input];
    for (const { inputs, outputs, weight, bias } of fold.layers.slice(0, -1)) {
      const source = layers.at(-1);
      const output = new Float32Array(outputs);
      for (let o = 0; o < outputs; o++) {
        let sum = 0;
        for (let i = 0; i < inputs; i++) sum += source[i] * weight[o * inputs + i];
        output[o] = gelu(sum + bias[o]);
      }
      layers.push(output);
    }
    const { inputs, weight, bias } = fold.layers.at(-1);
    const source = layers.at(-1);
    let sum = bias[0];
    for (let i = 0; i < inputs; i++) sum += weight[i] * source[i];
    cpModels[f] = Math.fround(sum) * fold.yStd + fold.yMean;
    return layers;
  });
  return { cpModels, activations };
}

/**
 * Mean Cp over the networks at each point, and the spread σ(Cp): their standard deviation
 * (population, as NumPy's `std`) times the scale fitted in training.
 * @param {Float32Array} models (networks, count)
 * @param {number} count
 * @param {number} spreadScale
 */
export function ensembleStatistics(models, count, spreadScale) {
  const networks = models.length / count;
  const cp = new Float32Array(count);
  const sigma = new Float32Array(count);
  for (let p = 0; p < count; p++) {
    let sum = 0;
    for (let f = 0; f < networks; f++) sum += models[f * count + p];
    const mean = sum / networks;
    let squares = 0;
    for (let f = 0; f < networks; f++) {
      const d = models[f * count + p] - mean;
      squares += d * d;
    }
    cp[p] = mean;
    sigma[p] = Math.sqrt(squares / networks) * spreadScale;
  }
  return { cp, sigma };
}
