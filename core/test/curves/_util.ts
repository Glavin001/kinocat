import type { Pose } from '../../src/curves/types';
import { wrapAngle } from '../../src/internal/math';

/** Deterministic PRNG (mulberry32) so fuzz tests are reproducible. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function poseClose(a: Pose, b: Pose, posEps = 1e-5, angEps = 1e-5): boolean {
  return (
    Math.abs(a.x - b.x) <= posEps &&
    Math.abs(a.y - b.y) <= posEps &&
    Math.abs(wrapAngle(a.theta - b.theta)) <= angEps
  );
}
