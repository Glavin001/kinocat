// One-shot vehicle planning sugar. Wraps `VehicleEnvironment` +
// `TimeAwareEnvironment` + `plan()` so a caller can go from "I have a world,
// an agent, a primitive library, a start and a goal" to a `PlanResult` in
// one call. Designed for per-tick game AI: build the env stack, run a single
// anytime pass within `deadlineMs`, return.
//
// Defaults are tuned for typical car-sized vehicles (`posCell=1.5`,
// `headingBuckets=16`, `goalRadius=4`, analytic Reeds-Shepp shot every 6
// successors at 0.6 m step). All of these can be overridden through
// `envOptions` if a specific game needs tighter resolution or more aggressive
// analytic shots.

import type { VehicleAgent, CarKinematicState } from '../agent/types';
import type { MotionPrimitiveLibrary } from '../primitives/library';
import type { MovingObstacle } from '../predict/types';
import type { AffordanceRegistry } from '../predict/affordance-registry';
import {
  TimeAwareEnvironment,
  type TimeAwareOptions,
} from '../environment/time-aware';
import {
  VehicleEnvironment,
  type VehicleEnvOptions,
} from '../environment/vehicle-environment';
import type { NavWorld } from '../environment/nav-world';
import { plan } from './ighastar';
import type { PlanResult, PlannerOptions } from './types';

export interface PlanVehicleRequest {
  start: CarKinematicState;
  goal: CarKinematicState;
  world: NavWorld;
  agent: VehicleAgent;
  lib: MotionPrimitiveLibrary;
  /** Other agents to treat as moving obstacles (cv predictions, published
   *  plans, etc.). Empty by default. */
  movingObstacles?: MovingObstacle[];
  /** Static off-mesh edges (ramps, boost pads, …). Optional. */
  affordances?: AffordanceRegistry;
  /** Overrides for VehicleEnvironment construction. */
  envOptions?: VehicleEnvOptions;
  /** Overrides for TimeAwareEnvironment construction. `obstacles` and
   *  `affordances` are filled in from the top-level fields. */
  timeOptions?: Omit<TimeAwareOptions, 'obstacles' | 'affordances'>;
  /** Planner options (anytime budget, weight, profile). */
  plannerOptions?: PlannerOptions;
  /** Wall-clock deadline in ms (defaults to 120 ms — single replan slot). */
  deadlineMs?: number;
  /** Hard expansion cap (defaults to 25k — sized for per-tick replans). */
  maxExpansions?: number;
}

const DEFAULT_ENV_OPTIONS: VehicleEnvOptions = {
  posCell: 1.5,
  headingBuckets: 16,
  speedQuant: 4,
  levelDivisors: [4, 2, 1],
  goalRadius: 4,
  goalHeadingTol: Infinity,
  sweepSegmentCheck: false,
  analyticExpansion: { everyN: 6, step: 0.6 },
  // No-op on worlds that don't expose `clearanceAt` / `buildGoalLowerBound`
  // (the gating checks live in `VehicleEnvironment`). `InMemoryNavWorld`
  // now implements both, so the carchase demo and other obstacle-rich
  // scenes pick them up automatically.
  clearanceBroadphase: true,
  gridHeuristic: {},
};

const DEFAULT_TIME_OPTIONS: Omit<TimeAwareOptions, 'obstacles' | 'affordances'> = {
  broadphase: { sampleStep: 0.5, maxSamples: 24 },
};

const DEFAULT_DEADLINE_MS = 120;
const DEFAULT_MAX_EXPANSIONS = 25000;

/** Plan a single vehicle path from `start` to `goal` against `world`. The
 *  agent's circumscribed footprint radius is derived automatically for the
 *  moving-obstacle test; pass `timeOptions.agentRadius` to override.
 *
 *  Same algorithm as building `VehicleEnvironment` + `TimeAwareEnvironment` +
 *  calling `plan()` manually — this is just the boilerplate consolidator
 *  every interactive AI needs. */
export function planVehicleOnce(req: PlanVehicleRequest): PlanResult<CarKinematicState> {
  // An obstacle-routing grid-Dijkstra lower bound is INADMISSIBLE once
  // affordances can bypass obstacles. The goal-distance field routes AROUND a
  // planner-only obstacle (e.g. the ramp demo's "gap") that an affordance jumps
  // straight over, so it overestimates cost-to-go near the launch — and
  // branch-and-bound then prunes the (cheaper) affordance branch the moment a
  // detour incumbent is found, so the jump is never taken. Default the grid
  // heuristic OFF whenever affordances are present (the caller can still force
  // it back on through `envOptions.gridHeuristic`). Moving obstacles do NOT
  // trigger this: they only ADD constraints, so a static lower bound stays
  // admissible.
  const affordanceAware = req.affordances !== undefined;
  const envOpts = {
    ...DEFAULT_ENV_OPTIONS,
    ...(affordanceAware ? { gridHeuristic: false as const } : {}),
    ...(req.envOptions ?? {}),
  };
  // Time participates in the dedup hash only when something is actually
  // time-varying; in static worlds it inflates the search ~3.8x for nothing.
  const hasDynamics = (req.movingObstacles?.length ?? 0) > 0 || affordanceAware;
  const baseEnv = new VehicleEnvironment(req.world, req.agent, req.lib, {
    timeInHash: hasDynamics,
    ...envOpts,
  });

  // Agent circumscribed radius (Euclidean from origin to the farthest
  // footprint vertex) — used to inflate moving-obstacle radii so the
  // time-aware collision test approximates the swept footprint.
  let rCirc = 0;
  for (const [vx, vz] of req.agent.footprint) {
    const r = Math.hypot(vx, vz);
    if (r > rCirc) rCirc = r;
  }

  const timeOpts: TimeAwareOptions = {
    ...DEFAULT_TIME_OPTIONS,
    agentRadius: rCirc,
    ...(req.timeOptions ?? {}),
    obstacles: req.movingObstacles ?? [],
    affordances: req.affordances,
  };
  const env = new TimeAwareEnvironment(baseEnv, timeOpts);

  return plan(
    {
      start: req.start,
      goal: req.goal,
      environment: env,
      options: {
        maxExpansions: req.maxExpansions ?? DEFAULT_MAX_EXPANSIONS,
        ...(req.plannerOptions ?? {}),
      },
    },
    req.deadlineMs ?? DEFAULT_DEADLINE_MS,
  );
}
