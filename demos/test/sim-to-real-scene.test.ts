// Unit tests for the pure /sim-to-real helpers. No Rapier / Three needed.

import { describe, it, expect } from 'vitest';
import type { CarKinematicState } from 'kinocat/agent';
import type { ForwardSim } from 'kinocat/primitives';
import {
  rolloutOpenLoop,
  projectFuture,
  poseGap,
  wrapPi,
  GapAccumulator,
  FuturePredictionTracker,
  speedToColor,
} from '../app/lib/sim-to-real-scene';

// A trivial constant-velocity forward sim along +x at `controls[0]` m/s.
const linearSim: ForwardSim<CarKinematicState> = (s, c, dt) => ({
  ...s,
  x: s.x + (c[0] ?? 0) * dt,
  t: s.t + dt,
  speed: c[0] ?? 0,
});

const s0: CarKinematicState = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };

describe('rolloutOpenLoop', () => {
  it('returns initial state + one sample per applied control', () => {
    const trace = [[5], [5], [5], [5]];
    const out = rolloutOpenLoop(s0, trace, 0.1, linearSim);
    expect(out).toHaveLength(5);
    expect(out[0]!.x).toBe(0);
    expect(out[4]!.x).toBeCloseTo(2.0, 6);
    expect(out[4]!.t).toBeCloseTo(0.4, 6);
  });

  it('empty trace returns only initial state', () => {
    expect(rolloutOpenLoop(s0, [], 0.1, linearSim)).toEqual([s0]);
  });
});

describe('projectFuture', () => {
  it('rolls forward for `horizonSec` with constant control', () => {
    const out = projectFuture(s0, [3], linearSim, { horizonSec: 1.0, stepDt: 1 / 60 });
    expect(out.length).toBeGreaterThan(50);
    const last = out[out.length - 1]!;
    expect(last.t).toBeCloseTo(1.0, 2);
    expect(last.x).toBeCloseTo(3.0, 2);
  });
});

describe('wrapPi / poseGap', () => {
  it('wraps large angles into [-pi, pi]', () => {
    expect(wrapPi(0)).toBe(0);
    expect(wrapPi(Math.PI + 0.1)).toBeCloseTo(-Math.PI + 0.1, 6);
    expect(wrapPi(-Math.PI - 0.1)).toBeCloseTo(Math.PI - 0.1, 6);
  });

  it('poseGap reports pos/heading/speed deltas with heading wrapped', () => {
    const real: CarKinematicState = { x: 1, z: 2, heading: 0.1, speed: 5, t: 0 };
    const pred: CarKinematicState = { x: 4, z: 6, heading: Math.PI + 0.2, speed: 7, t: 0 };
    const g = poseGap(real, pred);
    expect(g.posErr).toBeCloseTo(5, 6);
    expect(Math.abs(g.headingErr)).toBeLessThanOrEqual(Math.PI);
    expect(g.speedErr).toBe(2);
  });
});

describe('GapAccumulator', () => {
  it('reports zeros when empty', () => {
    const acc = new GapAccumulator(1.0);
    const r = acc.rms();
    expect(r.count).toBe(0);
    expect(r.posRms).toBe(0);
  });

  it('rolling RMS over a window evicts old samples', () => {
    const acc = new GapAccumulator(1.0);
    // Add 10 samples spaced 0.5 s apart. Only the last 2-3 should remain
    // in the 1 s window after the final push.
    for (let i = 0; i < 10; i++) {
      acc.push({ t: i * 0.5, posErr: 2, headingErr: 0, speedErr: 1 });
    }
    const r = acc.rms();
    expect(r.count).toBeLessThanOrEqual(3);
    expect(r.count).toBeGreaterThanOrEqual(2);
    expect(r.posRms).toBeCloseTo(2, 6);
    expect(r.speedRms).toBeCloseTo(1, 6);
  });
});

describe('FuturePredictionTracker', () => {
  it('drains predictions whose matchAt has been reached', () => {
    const tr = new FuturePredictionTracker();
    const make = (x: number): CarKinematicState => ({ x, z: 0, heading: 0, speed: 0, t: 0 });
    tr.schedule(make(1), 0, 0.5);
    tr.schedule(make(2), 0, 1.0);
    tr.schedule(make(3), 0, 1.5);
    expect(tr.size()).toBe(3);
    const real: CarKinematicState = { x: 5, z: 0, heading: 0, speed: 0, t: 0.6 };
    const matured = tr.drainMatured(real);
    expect(matured).toHaveLength(1);
    expect(matured[0]!.posErr).toBeCloseTo(4, 6); // predicted x=1 vs real x=5
    expect(tr.size()).toBe(2);
  });

  it('reset clears pending predictions', () => {
    const tr = new FuturePredictionTracker();
    tr.schedule({ x: 1, z: 0, heading: 0, speed: 0, t: 0 }, 0, 1);
    tr.reset();
    expect(tr.size()).toBe(0);
  });
});

describe('speedToColor', () => {
  it('returns a green-ish color at v=0 and red-ish at vMax', () => {
    const green = speedToColor(0, 10);
    const red = speedToColor(10, 10);
    // green has dominant G channel, red has dominant R channel
    const gG = (green >> 8) & 0xff;
    const gR = (green >> 16) & 0xff;
    const rR = (red >> 16) & 0xff;
    const rG = (red >> 8) & 0xff;
    expect(gG).toBeGreaterThan(gR);
    expect(rR).toBeGreaterThan(rG);
  });

  it('clamps absurd speeds to vMax', () => {
    expect(speedToColor(1000, 10)).toBe(speedToColor(10, 10));
  });
});
