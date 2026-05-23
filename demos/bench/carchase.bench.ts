// Performance benchmark for the /carchase planner. Measures the per-replan
// cost a single agent (robber or cop) pays inside the worker, so we can size
// `REPLAN_INTERVAL_MS` honestly and validate optimizations.
//
// Two scenarios, taken straight from the spawn matchup:
//   • robber  — robber state + first robber-loop waypoint, 3 cops as cv
//               moving-obstacle predictions
//   • cop     — cop[0] state + intercept goal (robber's current pose), with the
//               robber + 2 sibling cops as cv moving-obstacle predictions
//
// Each scenario is exercised across four env-option configurations so the
// per-component breakdown (collisions, heuristic calls, expansions, wall ms)
// is comparable apples-to-apples:
//
//   • base       — everything off (`heuristicTable: false` to neutralize the
//                  current default; mirrors the pre-optimization baseline)
//   • +htable    — Reeds-Shepp heuristic LUT enabled
//   • all        — every safe opt-in toggle the carchase NavWorld supports
//
// A `process.stderr` dump runs ONCE per (scenario, config) with
// `profile: 'timings'` so the formatPerf table appears inline above the
// bench() throughput rows.

import { bench, describe } from 'vitest';
import { planVehicleOnce } from 'kinocat/planner';
import { formatPerf } from 'kinocat/planner';
import type { VehicleEnvOptions } from 'kinocat/environment';
import { asObstacle, constantVelocity } from 'kinocat/predict';
import type { MovingObstacle } from 'kinocat/predict';
import { InMemoryNavWorld } from 'kinocat/environment';
import type { VehicleState } from 'kinocat/agent';
import {
  buildCarChaseCourse,
  CARCHASE_AGENT,
  CARCHASE_LIB,
  carChaseAffordances,
  spawnPoses,
} from '../app/lib/carchase-scenarios';

const COURSE = buildCarChaseCourse();
const AFFORDANCES = carChaseAffordances(COURSE);
const { robber: ROBBER, cops: COPS } = spawnPoses();

// Robber's first goal — first waypoint of its loop. Bypass the full
// `robberGoal` helper so the bench scene is deterministic across runs.
const ROBBER_GOAL: VehicleState = {
  x: COURSE.robberLoop[0]!.x,
  z: COURSE.robberLoop[0]!.z,
  heading: COURSE.robberLoop[0]!.heading,
  speed: CARCHASE_AGENT.maxSpeed,
  t: 0,
};

// Cop intercept goal — aim straight at the robber's current pose. Real
// `tacticalGoal` adds lead-time / lateral offset, but for the bench what
// matters is the planner stress, not the tactical layer.
const COP_GOAL: VehicleState = { ...ROBBER, speed: CARCHASE_AGENT.maxSpeed };

function robberObstacles(): MovingObstacle[] {
  return COPS.map((c) => asObstacle(constantVelocity(c, 4), 2.6));
}

function copObstacles(copIndex: number): MovingObstacle[] {
  const out: MovingObstacle[] = [asObstacle(constantVelocity(ROBBER, 4), 2.6)];
  for (let i = 0; i < COPS.length; i++) {
    if (i === copIndex) continue;
    out.push(asObstacle(constantVelocity(COPS[i]!, 4), 2.6));
  }
  return out;
}

const BASE: VehicleEnvOptions = {
  heuristicTable: false,
};

const HTABLE: VehicleEnvOptions = {
  heuristicTable: {},
};

const ALL: VehicleEnvOptions = {
  heuristicTable: {},
};

interface Scenario {
  name: string;
  start: VehicleState;
  goal: VehicleState;
  obstacles: MovingObstacle[];
}

const SCENARIOS: Scenario[] = [
  {
    name: 'robber → next waypoint',
    start: ROBBER,
    goal: ROBBER_GOAL,
    obstacles: robberObstacles(),
  },
  {
    name: 'cop0 → intercept robber',
    start: COPS[0]!,
    goal: COP_GOAL,
    obstacles: copObstacles(0),
  },
];

const CONFIGS: Array<{ name: string; opts: VehicleEnvOptions }> = [
  { name: 'base', opts: BASE },
  { name: '+htable', opts: HTABLE },
  { name: 'all', opts: ALL },
];

// Match production: the worker builds the NavWorld ONCE at init and reuses it
// across all replans (so the coarse-heuristic / clearance grids amortise
// across plans). Allocating a fresh world per bench iteration would charge
// the one-time grid-build cost to every measurement.
const WORLD = new InMemoryNavWorld(COURSE.polygons, COURSE.obstacles);

function runOne(
  scn: Scenario,
  envOptions: VehicleEnvOptions,
  profile?: 'counts' | 'timings',
): ReturnType<typeof planVehicleOnce> {
  return planVehicleOnce({
    start: scn.start,
    goal: scn.goal,
    world: WORLD,
    agent: CARCHASE_AGENT,
    lib: CARCHASE_LIB,
    movingObstacles: scn.obstacles,
    affordances: AFFORDANCES,
    timeOptions: { affordanceRadius: 10 },
    envOptions,
    plannerOptions: profile ? { profile } : undefined,
    deadlineMs: Infinity,
    maxExpansions: 25000,
  });
}

// Module-load breakdown — runs once before any bench() block.
for (const scn of SCENARIOS) {
  for (const cfg of CONFIGS) {
    const t0 = performance.now();
    const result = runOne(scn, cfg.opts, 'timings');
    const wall = performance.now() - t0;
    process.stderr.write(
      `\n=== ${scn.name} [${cfg.name}] (wall ${wall.toFixed(1)} ms) ===\n` +
        `found=${result.found} cost=${
          result.cost === Infinity ? 'inf' : result.cost.toFixed(2)
        } passes=${result.stats.passesRun}\n${formatPerf(result.stats)}\n`,
    );
  }
}

describe('carchase: per-replan planning cost', () => {
  for (const scn of SCENARIOS) {
    for (const cfg of CONFIGS) {
      bench(`${scn.name} [${cfg.name}]`, () => {
        runOne(scn, cfg.opts);
      });
    }
  }
});
