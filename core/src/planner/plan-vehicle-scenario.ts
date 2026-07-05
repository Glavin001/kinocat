// One-shot scenario planning. Wraps `VehicleEnvironment` +
// `TimeAwareEnvironment` + `ScenarioEnvironment` + `plan()` so a caller can go
// from a canonical Scenario goal (reach/seq/all/any/repeat + invariants + cost)
// to a `PlanResult` in one call — the scenario-layer analogue of
// `planVehicleOnce` / `planVehicleMultiGoal`.
//
// Composition (Scenario outermost): ScenarioEnvironment(TimeAware(Vehicle)).

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
  ScenarioEnvironment,
  scenarioStart,
  scenarioTerminal,
  type ScenarioAugState,
} from '../environment/scenario-environment';
import type { NavWorld } from '../environment/nav-world';
import { compile } from '../scenario/index';
import type { Goal, Invariant, CostTerm } from '../scenario/index';
import { plan } from './ighastar';
import type { PlanResult, PlannerOptions } from './types';
import type { Node } from '../environment/types';

export interface PlanVehicleScenarioRequest {
  start: CarKinematicState;
  /** Objective plane (canonical AST). Compiled to the goal automaton. */
  goal: Goal;
  /** Constraint plane — successor pruning. */
  invariants?: Invariant[];
  /** Cost plane — extra edge g. */
  prefer?: CostTerm[];
  /** Horizon for progress (`repeat`) objectives — required for them. */
  horizon?: { phases?: number; seconds?: number };
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
  heuristicTable: {},
};

const DEFAULT_TIME_OPTIONS: Omit<TimeAwareOptions, 'obstacles' | 'affordances'> = {
  broadphase: { sampleStep: 0.5, maxSamples: 24 },
};

const DEFAULT_DEADLINE_MS = 120;
const DEFAULT_MAX_EXPANSIONS = 50_000;

export interface ScenarioPlanResult {
  /** The raw augmented-state plan result (state = { inner, q, laps? }). */
  raw: PlanResult<ScenarioAugState<CarKinematicState>>;
  /** Convenience: the inner chassis path projected out of the augmented states. */
  path: CarKinematicState[];
}

/** Plan a single vehicle trajectory satisfying a canonical Scenario goal. */
export function planVehicleScenario(
  req: PlanVehicleScenarioRequest,
): ScenarioPlanResult {
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

  const automaton = compile(req.goal);
  const env = new ScenarioEnvironment<CarKinematicState>(timeEnv, {
    automaton,
    invariants: req.invariants ?? [],
    costTerms: req.prefer ?? [],
    ...(req.horizon ? { horizon: req.horizon } : {}),
  });

  const raw = plan<ScenarioAugState<CarKinematicState>>(
    {
      start: scenarioStart(req.start, automaton),
      goal: scenarioTerminal(req.start, automaton),
      environment: env,
      options: {
        maxExpansions: req.maxExpansions ?? DEFAULT_MAX_EXPANSIONS,
        ...(req.plannerOptions ?? {}),
      },
    },
    req.deadlineMs ?? DEFAULT_DEADLINE_MS,
  );

  return { raw, path: raw.path.map((s) => s.inner) };
}

/** Like {@link planVehicleScenario}, but returns a flat
 *  `PlanResult<CarKinematicState>` — the augmented `{inner, q}` nodes are
 *  unwrapped to their inner chassis state while PRESERVING each edge (incl. the
 *  Reeds-Shepp analytic-shot `data`). This is the drop-in replacement for
 *  `planVehicleOnce`/`planVehicleMultiGoal` in runtimes whose post-processing
 *  (analytic-shot lifting, smoothing, pure-pursuit) consumes `result.nodes`. */
export function planVehicleScenarioCar(
  req: PlanVehicleScenarioRequest,
): PlanResult<CarKinematicState> {
  const { raw } = planVehicleScenario(req);
  // Consumers (analytic-shot lifting) read only `state` + `edge`; the parent
  // chain is dropped (re-typed to null) since the augmented parents don't match.
  const nodes: Node<CarKinematicState>[] = raw.nodes.map((n) => ({
    ...n,
    state: n.state.inner,
    parent: null,
  }));
  return {
    found: raw.found,
    ...(raw.partial !== undefined ? { partial: raw.partial } : {}),
    cost: raw.cost,
    path: raw.path.map((s) => s.inner),
    nodes,
    stats: raw.stats,
    solutionHistory: raw.solutionHistory.map((h) => h.map((s) => s.inner)),
  };
}
