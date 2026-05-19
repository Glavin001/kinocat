// Dubins curves (Dubins 1957): shortest forward-only path for a car with a
// minimum turning radius. Ported from OMPL's DubinsStateSpace.

import type { CurvePath, Pose } from './types';
import { buildPath, toLocal, TWO_PI, type SegType } from './internal';

const DUBINS_EPS = 1e-6;
const DUBINS_ZERO = -1e-9;

/** Dubins uses [0, 2pi) so segment lengths stay non-negative (forward-only),
 *  unlike Reeds-Shepp's (-pi, pi] wrap. */
function mod2pi(x: number): number {
  if (x < 0 && x > -1e-12) return 0;
  return x - TWO_PI * Math.floor(x / TWO_PI);
}

const DUBINS_TYPES: readonly (readonly SegType[])[] = [
  ['L', 'S', 'L'], // 0
  ['R', 'S', 'R'], // 1
  ['R', 'S', 'L'], // 2
  ['L', 'S', 'R'], // 3
  ['R', 'L', 'R'], // 4
  ['L', 'R', 'L'], // 5
];

interface Word {
  typeIndex: number;
  t: number;
  p: number;
  q: number;
}

function len(w: Word | null): number {
  return w ? w.t + w.p + w.q : Infinity;
}

// CSC words. `tmp >= 0` for all finite (d,a,b): minimizing over d gives
// 2-2cos(a-b)-(sa-sb)^2 >= 0 since (sa-sb)^2 <= 2(1-cos(a-b)). So LSL/RSR are
// always valid — no nullable guard needed (Math.max clamps float dust).
function dubinsLSL(d: number, a: number, b: number): Word {
  const ca = Math.cos(a), sa = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b);
  const tmp = 2 + d * d - 2 * (ca * cb + sa * sb - d * (sa - sb));
  const theta = Math.atan2(cb - ca, d + sa - sb);
  return {
    typeIndex: 0,
    t: mod2pi(-a + theta),
    p: Math.sqrt(Math.max(tmp, 0)),
    q: mod2pi(b - theta),
  };
}

function dubinsRSR(d: number, a: number, b: number): Word {
  const ca = Math.cos(a), sa = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b);
  const tmp = 2 + d * d - 2 * (ca * cb + sa * sb - d * (sb - sa));
  const theta = Math.atan2(ca - cb, d - sa + sb);
  return {
    typeIndex: 1,
    t: mod2pi(a - theta),
    p: Math.sqrt(Math.max(tmp, 0)),
    q: mod2pi(-b + theta),
  };
}

function dubinsRSL(d: number, a: number, b: number): Word | null {
  const ca = Math.cos(a), sa = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b);
  const tmp = d * d - 2 + 2 * (ca * cb + sa * sb - d * (sa + sb));
  if (tmp >= DUBINS_ZERO) {
    const p = Math.sqrt(Math.max(tmp, 0));
    const theta = Math.atan2(ca + cb, d - sa - sb) - Math.atan2(2, p);
    return { typeIndex: 2, t: mod2pi(a - theta), p, q: mod2pi(b - theta) };
  }
  return null;
}

function dubinsLSR(d: number, a: number, b: number): Word | null {
  const ca = Math.cos(a), sa = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b);
  const tmp = -2 + d * d + 2 * (ca * cb + sa * sb + d * (sa + sb));
  if (tmp >= DUBINS_ZERO) {
    const p = Math.sqrt(Math.max(tmp, 0));
    const theta = Math.atan2(-ca - cb, d + sa + sb) - Math.atan2(-2, p);
    return { typeIndex: 3, t: mod2pi(-a + theta), p, q: mod2pi(-b + theta) };
  }
  return null;
}

function dubinsRLR(d: number, a: number, b: number): Word | null {
  const ca = Math.cos(a), sa = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b);
  const tmp = (6 - d * d + 2 * (ca * cb + sa * sb + d * (sa - sb))) / 8;
  if (Math.abs(tmp) < 1) {
    const p = TWO_PI - Math.acos(tmp);
    const theta = Math.atan2(ca - cb, d - sa + sb);
    return {
      typeIndex: 4,
      t: mod2pi(a - theta + 0.5 * p),
      p,
      q: mod2pi(a - b - (a - theta + 0.5 * p) + p),
    };
  }
  return null;
}

function dubinsLRL(d: number, a: number, b: number): Word | null {
  const ca = Math.cos(a), sa = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b);
  const tmp = (6 - d * d + 2 * (ca * cb + sa * sb - d * (sa - sb))) / 8;
  if (Math.abs(tmp) < 1) {
    const p = TWO_PI - Math.acos(tmp);
    const theta = Math.atan2(-ca + cb, d + sa - sb);
    const t = mod2pi(-a + theta + 0.5 * p);
    return { typeIndex: 5, t, p, q: mod2pi(b - a - t + p) };
  }
  return null;
}

function dubinsWord(d: number, a: number, b: number): Word {
  let best: Word | null = null;
  for (const f of [dubinsLSL, dubinsRSR, dubinsRSL, dubinsLSR, dubinsRLR, dubinsLRL]) {
    const w = f(d, a, b);
    if (w && len(w) < len(best)) best = w;
  }
  // At least one word is always valid for a finite query; the fallback is a
  // defensive degenerate straight.
  /* v8 ignore next -- unreachable defensive fallback */
  return best ?? { typeIndex: 0, t: 0, p: d, q: 0 };
}

/** Shortest forward-only Dubins path from `start` to `goal`. */
export function dubinsShortestPath(
  start: Pose,
  goal: Pose,
  radius: number,
): CurvePath {
  const { x, y } = toLocal(start, goal, radius);
  const D = Math.hypot(x, y);
  const d = D;
  const th = Math.atan2(y, x);
  const alpha = mod2pi(-th);
  const beta = mod2pi(goal.theta - start.theta - th);

  if (d < DUBINS_EPS && Math.abs(mod2pi(goal.theta - start.theta)) < DUBINS_EPS) {
    return { kind: 'dubins', word: '', segments: [], length: 0 };
  }

  const w = dubinsWord(d, alpha, beta);
  return buildPath('dubins', DUBINS_TYPES[w.typeIndex]!, [w.t, w.p, w.q], radius);
}
