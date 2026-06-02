// Continuous-follow EXECUTION. Following a moving target is a tracking task, not
// a one-shot plan: the planner expresses + reaches the goal (`reach(behind(lead))`),
// and a controller HOLDS the slot as the target moves. A min-time hybrid-A* plan
// instead rushes to each slot and arrives early (position-vs-time never matches
// the lead) and, with a cruise-only primitive set, overshoots then reverses —
// the visible "mess". This is a small pure-pursuit kinematic tracker that paces
// the target and curves smoothly, producing a clean trailing trajectory.

import type { CarKinematicState } from 'kinocat/agent';
import type { RegionAgent } from 'kinocat/scenario';

export interface FollowOptions {
  /** Slot distance behind the target (m). */
  gap: number;
  /** How long to simulate (s). */
  duration: number;
  /** Integration step (s). */
  dt?: number;
  /** Minimum turning radius (m) — bounds the commanded curvature. */
  minTurnRadius?: number;
  /** Speed cap (m/s). */
  maxSpeed?: number;
  /** Along-track speed gain: closes the gap to the slot without reversing. */
  kpSpeed?: number;
  /** Pure-pursuit lookahead floor (m), avoids curvature blow-up near the slot. */
  lookahead?: number;
}

const wrap = (a: number) => Math.atan2(Math.sin(a), Math.cos(a));
const clamp = (x: number, lo: number, hi: number) => (x < lo ? lo : x > hi ? hi : x);

/** Simulate a kinematic car pure-pursuing the slot `gap` metres behind `lead`. */
export function simulateFollow(
  lead: RegionAgent,
  start: CarKinematicState,
  opts: FollowOptions,
): CarKinematicState[] {
  const dt = opts.dt ?? 0.08;
  const minR = opts.minTurnRadius ?? 3.5;
  const kMax = 1 / minR;
  const vMax = opts.maxSpeed ?? 9;
  const kpSpeed = opts.kpSpeed ?? 0.7;
  const minLd = opts.lookahead ?? 2.5;

  let s: CarKinematicState = { ...start };
  const out: CarKinematicState[] = [{ ...s }];
  const t0 = start.t;
  for (let t = t0 + dt; t <= t0 + opts.duration + 1e-9; t += dt) {
    const la = lead.predict(t);
    if (!la) break;
    const slotX = la.x - Math.cos(la.heading) * opts.gap;
    const slotZ = la.z - Math.sin(la.heading) * opts.gap;
    const dx = slotX - s.x;
    const dz = slotZ - s.z;
    const dist = Math.hypot(dx, dz);
    const hErr = wrap(Math.atan2(dz, dx) - s.heading);

    // Pure-pursuit curvature toward the slot, bounded by the turn radius.
    const kappa = clamp((2 * Math.sin(hErr)) / Math.max(dist, minLd), -kMax, kMax);
    // Pace the lead, plus an along-track term to close the gap (never negative,
    // so the car slows/waits rather than reversing when it overshoots).
    const along = dist * Math.cos(hErr);
    const v = clamp(la.speed + kpSpeed * along, 0, vMax);

    // Kinematic integration with midpoint heading (smooth arcs).
    const midHeading = s.heading + 0.5 * v * kappa * dt;
    s = {
      x: s.x + v * Math.cos(midHeading) * dt,
      z: s.z + v * Math.sin(midHeading) * dt,
      heading: wrap(s.heading + v * kappa * dt),
      speed: v,
      t,
    };
    out.push(s);
  }
  return out;
}
