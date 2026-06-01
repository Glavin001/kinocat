// Cost terms (the `prefer` plane). Each is weighted and summed into the edge
// cost g. Distinct from the hard constraint plane: a cost shapes WHICH feasible
// path is best, never WHAT counts as success. Every term returns a non-negative
// penalty so the summed g remains a valid A* cost.

import type { CostTerm, ScenarioState } from './types';
import { pointSegmentDistance } from '../internal/geom';
import { angleDiff } from '../internal/math';

/** Penalize elapsed time (the racing default). dt is the edge duration. */
export function minTime(weight = 1): CostTerm {
  return {
    name: 'minTime',
    weight,
    edgeCost(_from, _to, dt) {
      return weight * dt;
    },
  };
}

/** Penalize jerk / steering rate via |Δheading| + |Δspeed| (comfort). */
export function smooth(weight = 1): CostTerm {
  return {
    name: 'smooth',
    weight,
    edgeCost(from, to) {
      // Shortest-arc heading delta (the naive modulo formula mishandles
      // negative wraps, e.g. a -6 rad delta reads as 6.0 instead of ~0.28).
      const dHeading = Math.abs(angleDiff(from.heading, to.heading));
      const dSpeed = Math.abs(to.speed - from.speed);
      return weight * (dHeading + dSpeed);
    },
  };
}

/** Soft penalty as clearance to `obstacles` drops below `margin` (distinct from
 *  the hard `avoid`). `obstacles` are point centers; penalty ramps to weight at
 *  contact. */
export function keepClear(
  weight: number,
  margin: number,
  obstacles: ReadonlyArray<{ x: number; z: number }>,
): CostTerm {
  return {
    name: 'keepClear',
    weight,
    edgeCost(_from, to) {
      let nearest = Infinity;
      for (const o of obstacles) {
        nearest = Math.min(nearest, Math.hypot(to.x - o.x, to.z - o.z));
      }
      if (!Number.isFinite(nearest) || nearest >= margin) return 0;
      // Linear ramp: 0 at margin -> weight at contact.
      return weight * (1 - nearest / margin);
    },
  };
}

/** Penalize lateral deviation from a reference (racing) line (polyline). */
export function racingLine(
  weight: number,
  ref: ReadonlyArray<{ x: number; z: number }>,
): CostTerm {
  return {
    name: 'racingLine',
    weight,
    edgeCost(_from, to) {
      let d = Infinity;
      for (let i = 0; i + 1 < ref.length; i++) {
        const a = ref[i]!;
        const b = ref[i + 1]!;
        d = Math.min(d, pointSegmentDistance(to.x, to.z, a.x, a.z, b.x, b.z));
      }
      return Number.isFinite(d) ? weight * d : 0;
    },
  };
}

/** Reward arc-length advanced (pairs with `repeat`). Modeled as a NEGATIVE-going
 *  contribution implemented as a non-negative penalty on LACK of progress: we
 *  penalize the shortfall from the max distance achievable in dt at a reference
 *  speed, so faster progress costs less. Keeps g non-negative + admissible. */
export function maxProgress(weight = 1, refSpeed = 10): CostTerm {
  return {
    name: 'maxProgress',
    weight,
    edgeCost(from, to, dt) {
      const advanced = Math.hypot(to.x - from.x, to.z - from.z);
      const ideal = refSpeed * dt;
      return weight * Math.max(0, ideal - advanced);
    },
  };
}
