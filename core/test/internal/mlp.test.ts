// Verify MLP forward + analytic backward pass against finite differences.
// Also verifies Adam optimizer reduces loss on a tiny synthetic regression.

import { describe, expect, it } from 'vitest';
import {
  createMLP,
  forward,
  backward,
  createAdam,
  adamStep,
  lossAndGradients,
  serializeMLP,
  deserializeMLP,
} from '../../src/internal/mlp';

function l2(pred: Float64Array, tgt: number[]): number {
  let s = 0;
  for (let i = 0; i < pred.length; i++) {
    const d = pred[i]! - tgt[i]!;
    s += 0.5 * d * d;
  }
  return s;
}

describe('MLP — forward + backward gradient correctness', () => {
  it('analytic gradients match finite differences within 1e-4 relative', () => {
    const mlp = createMLP({ inputDim: 4, hiddenDims: [6, 6], outputDim: 3 }, 7);
    const input = [0.1, -0.5, 0.7, 0.2];
    const target = [1.0, -0.3, 0.4];
    const cache = forward(mlp, input);
    const grads = backward(mlp, cache, target);

    // Finite difference on first hidden-layer weights.
    const eps = 1e-5;
    const layer = mlp.layers[0]!;
    const w = layer.weights;
    const gw = grads.weights[0]!;
    let maxRel = 0;
    for (let i = 0; i < Math.min(8, w.length); i++) {
      const orig = w[i]!;
      w[i] = orig + eps;
      const lp = l2(forward(mlp, input).output, target);
      w[i] = orig - eps;
      const lm = l2(forward(mlp, input).output, target);
      w[i] = orig;
      const fd = (lp - lm) / (2 * eps);
      const an = gw[i]!;
      const rel = Math.abs(an - fd) / Math.max(1e-3, Math.abs(an) + Math.abs(fd));
      maxRel = Math.max(maxRel, rel);
    }
    expect(maxRel).toBeLessThan(1e-4);
  });

  it('Adam reduces loss on a synthetic regression problem', () => {
    const mlp = createMLP({ inputDim: 3, hiddenDims: [8], outputDim: 2 }, 1);
    const adam = createAdam(mlp, 5e-2);
    // Synthetic target: y = [x0 - x1, x1 * x2]
    const dataset: { x: number[]; y: number[] }[] = [];
    for (let i = 0; i < 64; i++) {
      const x = [Math.sin(i), Math.cos(i * 0.7), Math.sin(i * 0.3)];
      dataset.push({ x, y: [x[0]! - x[1]!, x[1]! * x[2]!] });
    }
    function evalAvgLoss(): number {
      let s = 0;
      for (const d of dataset) s += lossAndGradients(mlp, d.x, d.y).loss;
      return s / dataset.length;
    }
    const before = evalAvgLoss();
    for (let ep = 0; ep < 60; ep++) {
      for (const d of dataset) {
        const r = lossAndGradients(mlp, d.x, d.y);
        adamStep(mlp, r.grads, adam);
      }
    }
    const after = evalAvgLoss();
    expect(after).toBeLessThan(before * 0.4);
  });

  it('serialize / deserialize preserves output bit-for-bit', () => {
    const mlp = createMLP({ inputDim: 5, hiddenDims: [7, 4], outputDim: 2 }, 11);
    const input = [0.1, 0.2, 0.3, 0.4, 0.5];
    const orig = forward(mlp, input).output;
    const restored = deserializeMLP(serializeMLP(mlp));
    const round = forward(restored, input).output;
    for (let i = 0; i < orig.length; i++) {
      expect(round[i]).toBe(orig[i]);
    }
  });
});
