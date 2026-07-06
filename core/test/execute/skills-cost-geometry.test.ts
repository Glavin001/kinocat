// Skill tests — cost/geometry tier (pure, sub-second, no Rapier).
//
// These isolate the MPPI progress-cost speed profile from the rest of the
// stack. They probe `buildProgressGeometry` (where the allowed-speed caps that
// throttle the car live) and `scoreRolloutProgress` (the cost that must reward
// speed on a straight). See docs/racing-skills-test-plan.md — skills K1, K3, K10.
//
// Design notes:
//  - vAllow is the geometric speed cap the cost enforces; on a straight it must
//    be Infinity (nothing but cruiseSpeed limits a straight — finding #2).
//  - The brake-onset distance is where vAllow first drops below the approach
//    speed ahead of a corner; it MUST equal the braking-envelope distance
//    (v^2 - vc^2)/(2*decel). The scenario currently wires decel = 8 m/s^2, but
//    the real plant brakes at 15.7-52.9 m/s^2 (plant-envelope.json), so the
//    car anticipates every corner ~2-6x too early (the K10 late-braking bug).

import { describe, it, expect } from 'vitest';
import {
  buildProgressGeometry,
  scoreRolloutProgress,
  type ProgressWeights,
} from '../../src/execute/mpc-tracker';
import type { CarKinematicState } from '../../src/agent/types';
import type { PlanPath } from '../../src/execute/types';

/** A straight plan along +x at constant speed, samples ~`ds` apart. */
function straightPlan(length: number, speed: number, ds = 0.4): PlanPath {
  const n = Math.max(2, Math.round(length / ds));
  const out: CarKinematicState[] = [];
  for (let i = 0; i < n; i++) {
    const u = (i / (n - 1)) * length;
    out.push({ x: u, z: 0, heading: 0, speed, t: u / Math.max(speed, 1) });
  }
  return out;
}

/** Straight plan with a smooth low-frequency lateral wander — the chord noise
 *  a replanned+smoothed path realistically carries on a genuine straight (NOT
 *  a max-frequency zig-zag). A correct curvature estimator must average this
 *  out and NOT read it as a corner. `wavelength` in metres. */
function jitteryStraight(
  length: number,
  speed: number,
  amp: number,
  wavelength = 4,
  ds = 0.4,
): PlanPath {
  const p = straightPlan(length, speed, ds);
  return p.map((s) => ({ ...s, z: amp * Math.sin((2 * Math.PI * s.x) / wavelength) }));
}

/** A straight run-up of `runup` metres into a constant-radius `R` arc that
 *  turns `turnRad` radians. Used to locate the brake-onset point ahead of a
 *  genuine corner. */
function straightIntoCorner(
  runup: number,
  R: number,
  turnRad: number,
  speed: number,
  ds = 0.4,
): PlanPath {
  const out: CarKinematicState[] = [];
  const nS = Math.max(2, Math.round(runup / ds));
  for (let i = 0; i < nS; i++) {
    const u = (i / (nS - 1)) * runup;
    out.push({ x: u, z: 0, heading: 0, speed, t: 0 });
  }
  // Arc turning left, centre at (runup, R).
  const arcLen = R * turnRad;
  const nA = Math.max(2, Math.round(arcLen / ds));
  for (let i = 1; i <= nA; i++) {
    const th = (i / nA) * turnRad;
    out.push({
      x: runup + R * Math.sin(th),
      z: R * (1 - Math.cos(th)),
      heading: th,
      speed,
      t: 0,
    });
  }
  // Re-timestamp monotonically.
  let acc = 0;
  for (let i = 1; i < out.length; i++) {
    acc += Math.hypot(out[i]!.x - out[i - 1]!.x, out[i]!.z - out[i - 1]!.z);
    out[i]!.t = acc / Math.max(speed, 1);
  }
  return out;
}

const GEOM_OPTS = {
  envelopeDecel: 8,
  envelopeLateralAccel: 12,
  usePlanSpeeds: false,
  ignoreTerminalSpeed: true, // drive-through horizon
  corridorSlack: 2.5,
};

function progressWeights(over: Partial<ProgressWeights> = {}): ProgressWeights {
  return {
    wProgress: 6,
    wCorridor: 20,
    corridorHalfWidth: 2.5,
    wCenterline: 0.08,
    wOverspeed: 4,
    envelopeDecel: 8,
    wControlRate: 0.15,
    wSteerRate: 10,
    wHeadingAlign: 1.5,
    speedCap: 30,
    ...over,
  };
}

describe('skill K1 — straight drive-through has no speed cap below cruise', () => {
  it('vAllow is Infinity everywhere on a clean straight', () => {
    const geom = buildProgressGeometry(straightPlan(80, 30), GEOM_OPTS);
    // Nothing geometric may cap a straight — only cruiseSpeed (applied later
    // in the cost via speedCap) limits it.
    expect(geom.vAllow.every((v) => !Number.isFinite(v) || v >= 30)).toBe(true);
  });

  it('a full-speed rollout scores strictly better than a slow one on a straight', () => {
    // Two synthetic rollouts along the straight: fast (30 m/s) advances more
    // arc per step than slow (10 m/s); progress is a reward, so fast must win.
    const geom = buildProgressGeometry(straightPlan(120, 30), GEOM_OPTS);
    const H = 20;
    const dt = 0.05;
    const mkRollout = (v: number): CarKinematicState[] => {
      const r: CarKinematicState[] = [];
      for (let i = 1; i <= H; i++) r.push({ x: v * dt * i, z: 0, heading: 0, speed: v, t: dt * i });
      return r;
    };
    const zeroCtl = new Float64Array(H * 3);
    const w = progressWeights();
    const start = { s: 0, idx: 0 };
    const fast = scoreRolloutProgress(mkRollout(30), geom, start, zeroCtl, zeroCtl, w, 1);
    const slow = scoreRolloutProgress(mkRollout(10), geom, start, zeroCtl, zeroCtl, w, 1);
    expect(fast).toBeLessThan(slow);
    // And 30 m/s (== cruise/speedCap) carries NO overspeed penalty (deadband).
    expect(fast).toBeLessThan(0); // net reward, not a penalty
  });
});

describe('skill K3 — real corners cap speed, phantom chord-noise does not', () => {
  it('a genuine tight corner caps vAllow near sqrt(aLat/kappa_eff)', () => {
    const R = 8;
    const geom = buildProgressGeometry(straightIntoCorner(20, R, Math.PI / 2, 30), GEOM_OPTS);
    const minV = Math.min(...geom.vAllow.filter((v) => Number.isFinite(v)));
    // With corridorSlack the effective radius is R+slack, so the cap is
    // sqrt(12/(1/(R+2.5))) = sqrt(12*10.5) ~ 11.2 m/s. It must actually bind
    // (well below cruise) on a real 8 m corner.
    expect(minV).toBeLessThan(16);
    expect(minV).toBeGreaterThan(6);
  });

  it('realistic smooth chord noise on a straight does NOT trigger a phantom cap', () => {
    // 5 cm amplitude, 4 m wavelength — a smoothed replan wandering slightly.
    // Its true radius is (wavelength/2pi)^2 / amp = 0.405/0.05 = ~8 m... which
    // IS a real curvature. Use a gentler, longer wander representative of a
    // smoothed straight: 3 cm over 8 m (radius ~54 m => vAllow ~sqrt(12*54)=25).
    const geom = buildProgressGeometry(jitteryStraight(80, 30, 0.03, 8), GEOM_OPTS);
    const minV = Math.min(...geom.vAllow.filter((v) => Number.isFinite(v)));
    // Should stay well above the tight-gate crawl speeds. Records the actual
    // phantom-throttle floor a smoothed straight suffers.
    expect(minV).toBeGreaterThan(20);
  });
});

describe('skill K10 — brake onset distance matches the braking envelope', () => {
  // Locate how far ahead of the corner vAllow first drops below the approach
  // speed, as a function of the deceleration budget. This is pure geometry —
  // it always holds — and it PINS the relationship the K10 fix depends on:
  // a bigger (more realistic) decel budget => later braking.
  function brakeOnsetDistance(decel: number): number {
    const plan = straightIntoCorner(120, 8, Math.PI / 2, 30);
    const geom = buildProgressGeometry(plan, { ...GEOM_OPTS, envelopeDecel: decel });
    // cumulative arc where the corner cap begins (first finite vAllow from the
    // end going backward that is < 29) minus the straight run-up end.
    let cornerStartArc = Infinity;
    for (let i = 0; i < geom.vAllow.length; i++) {
      if (Number.isFinite(geom.vAllow[i]!) && geom.vAllow[i]! < 29) {
        cornerStartArc = geom.cum[i]!;
        break;
      }
    }
    return cornerStartArc;
  }

  it('a realistic decel budget brakes MUCH later than the timid default (8)', () => {
    const onsetTimid = brakeOnsetDistance(8);
    const onsetReal = brakeOnsetDistance(22); // measured plant brakes 15.7-52.9
    // Later braking = larger onset arc (closer to the corner). The realistic
    // budget must start braking at least 10 m closer to the corner than the
    // timid one — quantifying the K10 "brakes ~3x too early" bug.
    expect(onsetReal).toBeGreaterThan(onsetTimid + 10);
  });
});
