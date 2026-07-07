// Control-set design by dispersion (Pivtoraiko & Kelly, "Generating
// Near-Minimal Spanning Control Sets").
//
// A primitive library is judged by its OUTPUTS — the body-frame endpoints its
// controls reach after one chunk — not by its control inputs. Two controls that
// land in the same place are redundant; a good set spreads its endpoints across
// the reachable set with no large gap (low DISPERSION) while spending the
// fewest slots (branching factor). This module rolls a dense candidate control
// grid through a forward model and selects a near-minimal spanning subset:
// force the extremes (fastest straight, hardest feasible turn each way, hard
// brake, coast, a reverse shunt), enforce left/right symmetry, and fill the
// remaining budget by farthest-point sampling.
//
// Wheeled control convention: [steer, driveForce, brakeForce] (driveForce < 0
// is reverse; the ForwardSim owns the dynamics). The mirror of a control is
// [-steer, driveForce, brakeForce]; its endpoint mirrors as (dx, -dz, -dHeading).

import type { CarKinematicState } from '../agent/types';
import type { ForwardSim } from './types';

export interface ControlSetDesignOptions {
  forwardSim: ForwardSim<CarKinematicState>;
  /** Body-frame start speed for this bucket (m/s). */
  startSpeed: number;
  /** Chunk duration + substep count (must match the library baking). */
  duration: number;
  substeps: number;
  /** Total control slots to emit (including the forced extremes). */
  budget: number;
  maxSteer: number;
  maxDrive: number;
  maxBrake: number;
  /** Candidate grid resolution. */
  steerLevels?: number;
  pedalLevels?: number;
  /** Number of slots reserved for a symmetric reverse sub-set (cusp shunts).
   *  Reverse is only useful in low-speed buckets; high-speed buckets should
   *  pass 0 so the whole budget buys forward coverage. Default 0. */
  reverseSlots?: number;
  /** Endpoint-distance weights: metres per radian of heading, metres per m/s of
   *  speed change. Position is already metres. */
  headingArm?: number;
  speedWeight?: number;
}

export interface Endpoint {
  dx: number;
  dz: number;
  dHeading: number;
  dv: number;
  controls: number[];
}

const wrapPi = (a: number): number => {
  let x = a;
  while (x > Math.PI) x -= 2 * Math.PI;
  while (x < -Math.PI) x += 2 * Math.PI;
  return x;
};

/** Roll one control vector for the bucket and return its body-frame endpoint. */
export function rollEndpoint(
  forwardSim: ForwardSim<CarKinematicState>,
  controls: number[],
  startSpeed: number,
  duration: number,
  substeps: number,
): Endpoint {
  let s: CarKinematicState = { x: 0, z: 0, heading: 0, speed: startSpeed, t: 0 };
  const dt = duration / substeps;
  for (let k = 0; k < substeps; k++) s = forwardSim(s, controls, dt);
  return { dx: s.x, dz: s.z, dHeading: s.heading, dv: s.speed - startSpeed, controls };
}

export function endpointDistance(a: Endpoint, b: Endpoint, headingArm = 2, speedWeight = 0.3): number {
  return Math.hypot(
    a.dx - b.dx,
    a.dz - b.dz,
    headingArm * wrapPi(a.dHeading - b.dHeading),
    speedWeight * (a.dv - b.dv),
  );
}

const mirrorControls = (u: number[]): number[] => [-u[0]!, u[1]!, u[2]!];

/**
 * Design a near-minimal spanning control set for one speed bucket. Returns the
 * control vectors (each `[steer, driveForce, brakeForce]`), symmetric under
 * steer negation, including the reachable extremes, ordered extremes-first.
 */
export function designControlSet(opts: ControlSetDesignOptions): number[][] {
  const {
    forwardSim, startSpeed, duration, substeps, budget,
    maxSteer, maxDrive, maxBrake,
    steerLevels = 15, pedalLevels = 13,
    reverseSlots = 0, headingArm = 2, speedWeight = 0.3,
  } = opts;
  const roll = (u: number[]): Endpoint => rollEndpoint(forwardSim, u, startSpeed, duration, substeps);
  const dist = (a: Endpoint, b: Endpoint): number => endpointDistance(a, b, headingArm, speedWeight);

  // Dense candidate grid over the RIGHT half (steer >= 0); the left half is the
  // exact mirror, added as pairs so the set stays symmetric.
  // Forward candidates: pedal a in [-1, 1] (a>=0 throttle, a<0 brake).
  const halfCandidates: Endpoint[] = [];
  // Reverse candidates: negative drive (a<0), no brake, over the same steer half.
  const reverseCandidates: Endpoint[] = [];
  for (let i = 0; i < steerLevels; i++) {
    const steer = (maxSteer * i) / (steerLevels - 1); // 0 .. maxSteer
    for (let j = 0; j < pedalLevels; j++) {
      const a = -1 + (2 * j) / (pedalLevels - 1); // pedal in [-1, 1]; a<0 = brake
      halfCandidates.push(roll([steer, a >= 0 ? a * maxDrive : 0, a >= 0 ? 0 : -a * maxBrake]));
    }
    // reverse drive magnitude grid (a subset — reverse is a low-speed shunt)
    for (let j = 1; j <= 3; j++) {
      reverseCandidates.push(roll([steer, -(maxDrive * j) / 3, 0]));
    }
  }

  const chosen: Endpoint[] = [];
  const push = (e: Endpoint): void => { chosen.push(e); };
  const pushPair = (e: Endpoint): void => {
    push(e);
    if (Math.abs(e.controls[0]!) > 1e-9) push(roll(mirrorControls(e.controls)));
  };

  // --- Forced self-symmetric extremes (steer = 0) ---
  push(roll([0, maxDrive, 0]));   // fastest straight
  push(roll([0, 0, 0]));          // coast
  push(roll([0, 0, maxBrake]));   // hard brake straight

  // --- Forced hardest feasible turn (max |dHeading| among steered candidates) ---
  let hardest = halfCandidates[0]!;
  for (const c of halfCandidates) if (Math.abs(c.dHeading) > Math.abs(hardest.dHeading)) hardest = c;
  pushPair(hardest);

  const minDistToChosen = (c: Endpoint): number => {
    let m = Infinity;
    for (const e of chosen) m = Math.min(m, dist(c, e));
    return m;
  };
  // Farthest-point fill in symmetric pairs from a candidate pool until the
  // chosen set reaches `limit`. Skips the steer≈0 axis (self-symmetric, added
  // only via forced extremes) so pairs never double the straight.
  const fpsFill = (pool: Endpoint[], limit: number): void => {
    while (chosen.length + 2 <= limit) {
      let best = pool[0]!;
      let bestD = -1;
      for (const c of pool) {
        if (Math.abs(c.controls[0]!) < 1e-9) continue;
        const d = minDistToChosen(c);
        if (d > bestD) { bestD = d; best = c; }
      }
      if (bestD <= 1e-6) break; // fully covered
      pushPair(best);
    }
  };

  // --- Forward coverage: FPS over forward candidates, reserving reverse slots ---
  fpsFill(halfCandidates, budget - reverseSlots);

  // --- Reverse shunt sub-set: SAME dispersion approach, fewer slots. Force the
  // straight-reverse extreme, then FPS-fill the reverse candidate pool. Reverse
  // is a low-speed escape maneuver, so it needs far fewer slots than forward. ---
  if (reverseSlots >= 1) {
    push(roll([0, -maxDrive * 0.6, 0])); // straight reverse (self-symmetric)
    fpsFill(reverseCandidates, budget);
  }

  return chosen.slice(0, budget).map((e) => e.controls);
}

/** Coverage metrics of a control set over its reachable candidate grid — the
 *  numbers a coverage regression test asserts on. */
export interface CoverageReport {
  slots: number;
  /** Worst reachable endpoint far from any set primitive (m). */
  dispersion: number;
  /** Closest pair of set endpoints (m) — small = redundant slot. */
  minPairwise: number;
  /** Set vs reachable max |dHeading| (rad) and max |dx| (m). */
  maxHeadingSet: number;
  maxHeadingReachable: number;
  maxDxSet: number;
  maxDxReachable: number;
  /** Max L/R asymmetry: worst endpoint whose mirror is far from the set (m). */
  asymmetry: number;
}

export function coverageReport(
  controls: number[][],
  opts: Omit<ControlSetDesignOptions, 'budget'> & { budget?: number },
): CoverageReport {
  const { forwardSim, startSpeed, duration, substeps, maxSteer, maxDrive, maxBrake,
    steerLevels = 21, pedalLevels = 13, headingArm = 2, speedWeight = 0.3 } = opts;
  const roll = (u: number[]): Endpoint => rollEndpoint(forwardSim, u, startSpeed, duration, substeps);
  const dist = (a: Endpoint, b: Endpoint): number => endpointDistance(a, b, headingArm, speedWeight);
  const setEnds = controls.map(roll);

  // Dense reachable set (full width for dispersion + asymmetry).
  const reach: Endpoint[] = [];
  for (let i = 0; i < steerLevels; i++) {
    const steer = -maxSteer + (2 * maxSteer * i) / (steerLevels - 1);
    for (let j = 0; j < pedalLevels; j++) {
      const a = -1 + (2 * j) / (pedalLevels - 1);
      reach.push(roll([steer, a >= 0 ? a * maxDrive : 0, a >= 0 ? 0 : -a * maxBrake]));
    }
  }
  let dispersion = 0;
  for (const c of reach) {
    let m = Infinity;
    for (const e of setEnds) m = Math.min(m, dist(c, e));
    dispersion = Math.max(dispersion, m);
  }
  let minPairwise = Infinity;
  for (let i = 0; i < setEnds.length; i++)
    for (let j = i + 1; j < setEnds.length; j++)
      minPairwise = Math.min(minPairwise, dist(setEnds[i]!, setEnds[j]!));
  // Asymmetry: for each set endpoint, distance from its mirror to the nearest
  // set endpoint (a symmetric set has ~0).
  let asymmetry = 0;
  for (const e of setEnds) {
    const mir: Endpoint = { dx: e.dx, dz: -e.dz, dHeading: -e.dHeading, dv: e.dv, controls: e.controls };
    let m = Infinity;
    for (const o of setEnds) m = Math.min(m, dist(mir, o));
    asymmetry = Math.max(asymmetry, m);
  }
  return {
    slots: controls.length,
    dispersion,
    minPairwise,
    maxHeadingSet: Math.max(...setEnds.map((e) => Math.abs(e.dHeading))),
    maxHeadingReachable: Math.max(...reach.map((e) => Math.abs(e.dHeading))),
    maxDxSet: Math.max(...setEnds.map((e) => Math.abs(e.dx))),
    maxDxReachable: Math.max(...reach.map((e) => Math.abs(e.dx))),
    asymmetry,
  };
}
