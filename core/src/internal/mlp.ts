// Small dense feed-forward neural network with hand-coded forward + analytic
// backward pass and Adam optimizer. Domain-agnostic.
//
// Sized for the residual-correction use case (input ~20, hidden 2x32, output
// ~6). At that scale, hand-coded pure-TS is fast enough for browser training
// and avoids the heavy TF.js / WebGPU dependency.
//
// All weights stored in `Float64Array` for stable training (Float32 was noisy
// at small batch sizes). Activations: ReLU for hidden layers, linear for
// output (residual targets are real-valued, not bounded).

export interface MLPConfig {
  inputDim: number;
  hiddenDims: number[];
  outputDim: number;
}

interface DenseLayer {
  weights: Float64Array; // shape [out, in] flattened row-major
  biases: Float64Array;  // shape [out]
  outDim: number;
  inDim: number;
}

export interface MLP {
  config: MLPConfig;
  layers: DenseLayer[];
}

/** Random seeded normal sampler (Box-Muller). */
function makeRng(seed: number): () => number {
  let state = (seed | 0) || 1;
  // Mulberry32 — small, fast, decent for synthetic ML weight init.
  const u01 = (): number => {
    state = (state + 0x6d2b79f5) | 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return (): number => {
    // Box-Muller, return one normal per call (waste second sample — simpler).
    const u = Math.max(1e-12, u01());
    const v = u01();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
}

/** He-style initialization for ReLU networks. Final layer uses smaller std. */
export function createMLP(config: MLPConfig, seed = 1): MLP {
  const rng = makeRng(seed);
  const layers: DenseLayer[] = [];
  let prev = config.inputDim;
  const dims = [...config.hiddenDims, config.outputDim];
  for (let li = 0; li < dims.length; li++) {
    const out = dims[li]!;
    const isLast = li === dims.length - 1;
    const std = isLast
      ? 0.01 // small final init so initial residual ≈ 0 (parametric backbone dominates at start)
      : Math.sqrt(2 / prev);
    const w = new Float64Array(out * prev);
    for (let i = 0; i < w.length; i++) w[i] = rng() * std;
    const b = new Float64Array(out); // zero
    layers.push({ weights: w, biases: b, outDim: out, inDim: prev });
    prev = out;
  }
  return { config, layers };
}

/** Forward pass. Returns the output vector and the per-layer pre/post
 *  activations needed for backprop. */
export interface ForwardCache {
  inputs: Float64Array;
  preActs: Float64Array[];  // z = Wx + b per hidden layer
  postActs: Float64Array[]; // a = relu(z) per hidden layer
  output: Float64Array;
}

export function forward(mlp: MLP, input: ReadonlyArray<number> | Float64Array): ForwardCache {
  const inputs = input instanceof Float64Array ? input : Float64Array.from(input);
  const preActs: Float64Array[] = [];
  const postActs: Float64Array[] = [];
  let prev: Float64Array = inputs;
  for (let li = 0; li < mlp.layers.length; li++) {
    const layer = mlp.layers[li]!;
    const isLast = li === mlp.layers.length - 1;
    const z = new Float64Array(layer.outDim);
    for (let o = 0; o < layer.outDim; o++) {
      let s = layer.biases[o]!;
      const row = o * layer.inDim;
      for (let i = 0; i < layer.inDim; i++) s += layer.weights[row + i]! * prev[i]!;
      z[o] = s;
    }
    if (isLast) {
      preActs.push(z);
      postActs.push(z); // linear output
      return { inputs, preActs, postActs, output: z };
    }
    const a = new Float64Array(layer.outDim);
    for (let o = 0; o < layer.outDim; o++) a[o] = z[o]! > 0 ? z[o]! : 0;
    preActs.push(z);
    postActs.push(a);
    prev = a;
  }
  // Unreachable: handled in loop.
  return { inputs, preActs, postActs, output: postActs[postActs.length - 1]! };
}

/**
 * Allocation-free inference closure for hot loops (MPPI rollouts call the
 * forward model thousands of times per solve; `forward()`'s per-call cache
 * arrays are backprop bookkeeping that inference never reads and the GC
 * pays for). Returns a function writing the output into `out`.
 */
export function makeMLPInfer(
  mlp: MLP,
): (input: ArrayLike<number>, out: Float64Array) => void {
  let maxDim = mlp.config.inputDim;
  for (const l of mlp.layers) maxDim = Math.max(maxDim, l.outDim);
  const bufA = new Float64Array(maxDim);
  const bufB = new Float64Array(maxDim);
  const layers = mlp.layers;
  const last = layers.length - 1;
  return (input: ArrayLike<number>, out: Float64Array): void => {
    let src: ArrayLike<number> = input;
    let scratch = bufA;
    for (let li = 0; li < layers.length; li++) {
      const layer = layers[li]!;
      const isLast = li === last;
      const dst = isLast ? out : scratch;
      const weights = layer.weights;
      const biases = layer.biases;
      const inDim = layer.inDim;
      for (let o = 0; o < layer.outDim; o++) {
        let s = biases[o]!;
        const row = o * inDim;
        for (let i = 0; i < inDim; i++) s += weights[row + i]! * (src[i] as number);
        dst[o] = isLast ? s : s > 0 ? s : 0;
      }
      src = dst;
      scratch = scratch === bufA ? bufB : bufA;
    }
  };
}

/** MSE gradient at the output, propagated back through the layers. Returns
 *  per-layer weight and bias gradients matching the layer shape. */
export interface Gradients {
  weights: Float64Array[];
  biases: Float64Array[];
}

export function backward(
  mlp: MLP,
  cache: ForwardCache,
  target: ReadonlyArray<number> | Float64Array,
): Gradients {
  const tgt = target instanceof Float64Array ? target : Float64Array.from(target);
  const L = mlp.layers.length;
  const gradW: Float64Array[] = new Array(L);
  const gradB: Float64Array[] = new Array(L);
  // delta at output: dL/dz_out = (a_out - target) for MSE with linear out.
  let delta = new Float64Array(mlp.layers[L - 1]!.outDim);
  for (let o = 0; o < delta.length; o++) delta[o] = cache.output[o]! - tgt[o]!;
  for (let li = L - 1; li >= 0; li--) {
    const layer = mlp.layers[li]!;
    const prevAct = li === 0 ? cache.inputs : cache.postActs[li - 1]!;
    // Apply ReLU mask for hidden layers (output layer is linear; we already
    // have delta directly at z).
    if (li < L - 1) {
      const z = cache.preActs[li]!;
      for (let o = 0; o < delta.length; o++) {
        if (z[o]! <= 0) delta[o] = 0;
      }
    }
    const gw = new Float64Array(layer.outDim * layer.inDim);
    const gb = new Float64Array(layer.outDim);
    for (let o = 0; o < layer.outDim; o++) {
      gb[o] = delta[o]!;
      const row = o * layer.inDim;
      const dlt = delta[o]!;
      for (let i = 0; i < layer.inDim; i++) {
        gw[row + i] = dlt * prevAct[i]!;
      }
    }
    gradW[li] = gw;
    gradB[li] = gb;
    if (li > 0) {
      // Propagate delta to previous layer's z: delta_prev = (W^T * delta).
      const prevLayer = mlp.layers[li - 1]!;
      const next = new Float64Array(prevLayer.outDim);
      for (let i = 0; i < layer.inDim; i++) {
        let s = 0;
        for (let o = 0; o < layer.outDim; o++) {
          s += layer.weights[o * layer.inDim + i]! * delta[o]!;
        }
        next[i] = s;
      }
      delta = next;
    }
  }
  return { weights: gradW, biases: gradB };
}

/** Adam optimizer state for in-place parameter updates. */
export interface AdamState {
  mW: Float64Array[];
  vW: Float64Array[];
  mB: Float64Array[];
  vB: Float64Array[];
  step: number;
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
}

export function createAdam(mlp: MLP, lr = 1e-3): AdamState {
  const mW: Float64Array[] = [];
  const vW: Float64Array[] = [];
  const mB: Float64Array[] = [];
  const vB: Float64Array[] = [];
  for (const layer of mlp.layers) {
    mW.push(new Float64Array(layer.weights.length));
    vW.push(new Float64Array(layer.weights.length));
    mB.push(new Float64Array(layer.biases.length));
    vB.push(new Float64Array(layer.biases.length));
  }
  return { mW, vW, mB, vB, step: 0, lr, beta1: 0.9, beta2: 0.999, eps: 1e-8 };
}

/** Apply Adam update step in place. `grads` is the accumulated batch
 *  gradient — caller is responsible for averaging across the batch. */
export function adamStep(mlp: MLP, grads: Gradients, state: AdamState): void {
  state.step++;
  const t = state.step;
  const bc1 = 1 - Math.pow(state.beta1, t);
  const bc2 = 1 - Math.pow(state.beta2, t);
  for (let li = 0; li < mlp.layers.length; li++) {
    const layer = mlp.layers[li]!;
    const gw = grads.weights[li]!;
    const gb = grads.biases[li]!;
    const mW = state.mW[li]!;
    const vW = state.vW[li]!;
    const mB = state.mB[li]!;
    const vB = state.vB[li]!;
    for (let i = 0; i < layer.weights.length; i++) {
      mW[i] = state.beta1 * mW[i]! + (1 - state.beta1) * gw[i]!;
      vW[i] = state.beta2 * vW[i]! + (1 - state.beta2) * gw[i]! * gw[i]!;
      const mh = mW[i]! / bc1;
      const vh = vW[i]! / bc2;
      layer.weights[i]! -= (state.lr * mh) / (Math.sqrt(vh) + state.eps);
    }
    for (let o = 0; o < layer.biases.length; o++) {
      mB[o] = state.beta1 * mB[o]! + (1 - state.beta1) * gb[o]!;
      vB[o] = state.beta2 * vB[o]! + (1 - state.beta2) * gb[o]! * gb[o]!;
      const mh = mB[o]! / bc1;
      const vh = vB[o]! / bc2;
      layer.biases[o]! -= (state.lr * mh) / (Math.sqrt(vh) + state.eps);
    }
  }
}

/** Convenience: one forward+backward call returning per-example loss and
 *  gradients. MSE loss. */
export function lossAndGradients(
  mlp: MLP,
  input: ReadonlyArray<number>,
  target: ReadonlyArray<number>,
): { loss: number; grads: Gradients; output: Float64Array } {
  const cache = forward(mlp, input);
  let loss = 0;
  for (let o = 0; o < cache.output.length; o++) {
    const d = cache.output[o]! - target[o]!;
    loss += 0.5 * d * d;
  }
  const grads = backward(mlp, cache, target);
  return { loss, grads, output: cache.output };
}

/** Average gradients across a batch in place into `acc`. */
export function accumulateGradients(acc: Gradients, add: Gradients, scale: number): void {
  for (let li = 0; li < acc.weights.length; li++) {
    const aw = acc.weights[li]!;
    const bw = add.weights[li]!;
    for (let i = 0; i < aw.length; i++) aw[i]! += bw[i]! * scale;
    const ab = acc.biases[li]!;
    const bb = add.biases[li]!;
    for (let o = 0; o < ab.length; o++) ab[o]! += bb[o]! * scale;
  }
}

export function zeroGradients(mlp: MLP): Gradients {
  const w: Float64Array[] = [];
  const b: Float64Array[] = [];
  for (const layer of mlp.layers) {
    w.push(new Float64Array(layer.weights.length));
    b.push(new Float64Array(layer.biases.length));
  }
  return { weights: w, biases: b };
}

/** Serialize / deserialize an MLP for persistence. */
export function serializeMLP(mlp: MLP): string {
  return JSON.stringify({
    config: mlp.config,
    layers: mlp.layers.map((l) => ({
      outDim: l.outDim,
      inDim: l.inDim,
      weights: Array.from(l.weights),
      biases: Array.from(l.biases),
    })),
  });
}

export function deserializeMLP(json: string): MLP {
  const obj = JSON.parse(json) as {
    config: MLPConfig;
    layers: { outDim: number; inDim: number; weights: number[]; biases: number[] }[];
  };
  return {
    config: obj.config,
    layers: obj.layers.map((l) => ({
      outDim: l.outDim,
      inDim: l.inDim,
      weights: Float64Array.from(l.weights),
      biases: Float64Array.from(l.biases),
    })),
  };
}
