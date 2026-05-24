// Multi-gate vehicle planning. Wraps `VehicleEnvironment` +
// `TimeAwareEnvironment` + `MultiGoalEnvironment` so callers can plan a
// SINGLE A* through an ordered sequence of gates, instead of chaining N
// independent `planVehicleOnce` calls.
//
// Why this matters: chained per-gate planning forces each segment to
// commit to its terminal pose, so the planner cannot pick a wider entry
// to gate i that pays off at gate i+1 or i+2 — the racing-line problem.
// One global search frees the planner to make those trade-offs.

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
import {
  MultiGoalEnvironment,
  multiGoalStart,
  multiGoalTerminal,
  type MultiGoalState,
} from '../environment/multi-goal';
import type { NavWorld } from '../environment/nav-world';
import { plan } from './ighastar';
import type { PlanResult, PlannerOptions } from './types';
import { makeCounters } from './perf';

export interface PlanVehicleMultiGoalRequest {
  start: CarKinematicState;
  /** Ordered list of gates the chassis must pass through. Each is treated
   *  as a position to be reached within `gateRadius` — heading and speed
   *  at the gate are not constrained. */
  gates: CarKinematicState[];
  world: NavWorld;
  agent: VehicleAgent;
  lib: MotionPrimitiveLibrary;
  movingObstacles?: MovingObstacle[];
  affordances?: AffordanceRegistry;
  envOptions?: VehicleEnvOptions;
  timeOptions?: Omit<TimeAwareOptions, 'obstacles' | 'affordances'>;
  plannerOptions?: PlannerOptions;
  deadlineMs?: number;
  maxExpansions?: number;
  /** Position radius for the "gate reached" check (Euclidean XZ). Default
   *  4 m (matches plan-vehicle.ts's default goalRadius). */
  gateRadius?: number;
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
};

const DEFAULT_TIME_OPTIONS: Omit<TimeAwareOptions, 'obstacles' | 'affordances'> = {
  broadphase: { sampleStep: 0.5, maxSamples: 24 },
};

const DEFAULT_DEADLINE_MS = 120;
const DEFAULT_MAX_EXPANSIONS = 50_000;
const DEFAULT_GATE_RADIUS = 4;

/** Plan a vehicle path through an ordered sequence of gates in a single
 *  A* search. The chassis is required only to pass within `gateRadius` of
 *  each gate position in order; heading and speed are free. */
export function planVehicleMultiGoal(
  req: PlanVehicleMultiGoalRequest,
): PlanResult<CarKinematicState> {
  if (req.gates.length === 0) {
    return {
      found: false,
      cost: Infinity,
      path: [],
      nodes: [],
      stats: {
        expansions: 0,
        generated: 0,
        deadlineHit: false,
        budgetHit: false,
        passesRun: 0,
        improvements: 0,
        counters: makeCounters(),
        perPass: [],
      },
      solutionHistory: [],
    };
  }
  const envOpts = { ...DEFAULT_ENV_OPTIONS, ...(req.envOptions ?? {}) };
  const baseEnv = new VehicleEnvironment(req.world, req.agent, req.lib, envOpts);
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
  const timeEnv = new TimeAwareEnvironment(baseEnv, timeOpts);

  const gateRadius = req.gateRadius ?? DEFAULT_GATE_RADIUS;
  const gateRadiusSq = gateRadius * gateRadius;
  const multiEnv = new MultiGoalEnvironment<CarKinematicState>(timeEnv, {
    gates: req.gates,
    reachedGate: (s, g) => {
      const dx = s.x - g.x;
      const dz = s.z - g.z;
      return dx * dx + dz * dz <= gateRadiusSq;
    },
    // Leg lower bound: base env's heuristic (Reeds-Shepp distance / time).
    // Already admissible — caller can override with a tighter bound.
    legHeuristic: (from, to) => timeEnv.heuristic(from, to),
  });

  const startState = multiGoalStart(req.start);
  const goalState = multiGoalTerminal(req.gates);

  const result = plan<MultiGoalState<CarKinematicState>>(
    {
      start: startState,
      goal: goalState,
      environment: multiEnv,
      options: {
        maxExpansions: req.maxExpansions ?? DEFAULT_MAX_EXPANSIONS,
        ...(req.plannerOptions ?? {}),
      },
    },
    req.deadlineMs ?? DEFAULT_DEADLINE_MS,
  );

  // Unwrap the multi-goal state → vehicle state for the planner's standard
  // PlanResult<CarKinematicState> contract.
  return {
    found: result.found,
    cost: result.cost,
    path: result.path.map((s) => s.inner),
    // Nodes carry edges + g/h/f; their type-parameter changes but the data
    // is otherwise pass-through. Cast the state inside.
    nodes: result.nodes.map((n) => ({
      state: n.state.inner,
      g: n.g, h: n.h, f: n.f,
      parent: null, // parent chain not exposed in the unwrapped form
      edge: n.edge,
      index: n.index,
      hash: n.hash,
      level: n.level,
      active: n.active,
      seq: n.seq,
    })),
    stats: result.stats,
    solutionHistory: result.solutionHistory.map(
      (path) => path.map((s) => s.inner),
    ),
  };
}
