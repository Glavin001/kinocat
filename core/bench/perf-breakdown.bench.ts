// Permanent per-component performance breakdown. Not a CI gate — read the
// console output to see where time/expansions go on representative scenes:
// aircraft 3D (static boxes), aircraft + moving zone (time-aware predicts),
// vehicle on serpentine (Reeds-Shepp heuristic), R2 grid (baseline).
//
// Each scenario runs ONCE up front with `profile: 'timings'` and dumps the
// counter + timing table via `formatPerf`. Then `bench()` measures
// throughput on the same query with `profile: 'counts'` (default) to keep
// the throughput number comparable across runs.

import { bench, describe } from 'vitest';
import { plan } from '../src/planner/ighastar';
import { formatPerf } from '../src/planner/perf';
import {
  AircraftEnvironment,
  type AircraftEnvOptions,
} from '../src/environment/aircraft-environment';
import {
  InMemoryAirspace,
  type AABB,
  type MovingZone,
} from '../src/environment/airspace-world';
import { defaultAircraftAgent } from '../src/agent/aircraft';
import type { AircraftAgent, AircraftState } from '../src/agent/types';
import { R2Environment } from '../src/environment/r2-environment';
import type { PlanResult } from '../src/planner/types';

// ---------- Aircraft: canyon (static boxes, full 3D) ------------------------

const AIR_AGENT: AircraftAgent = defaultAircraftAgent({
  minTurnRadius: 16,
  minSpeed: 8,
  maxSpeed: 18,
  maxClimbAngle: Math.PI / 6,
  maxBank: Math.PI / 2,
  halfLength: 2,
  halfSpan: 1.5,
  halfHeight: 0.3,
});

const AIR_BOUNDS = { floor: 0, ceiling: 80 };

function gate(x: number, y: number, z: number): AircraftState {
  return {
    x,
    y,
    z,
    heading: 0,
    pitch: 0,
    roll: 0,
    speed: AIR_AGENT.maxSpeed,
    t: 0,
  };
}

function makeCanyon(analyticExpansion: AircraftEnvOptions['analyticExpansion'] = false) {
  const f = AIR_BOUNDS.floor;
  const c = AIR_BOUNDS.ceiling;
  const boxes: AABB[] = [
    { min: [44, f, -60], max: [52, c, 4] },
    { min: [92, f, -4], max: [100, c, 60] },
    { min: [130, f, -60], max: [138, 34, 60] },
  ];
  const airspace = new InMemoryAirspace({ floor: f, ceiling: c, boxes });
  const start = gate(8, 22, 0);
  const goal = gate(152, 22, 0);
  const env = new AircraftEnvironment(airspace, AIR_AGENT, {
    posCell: 4,
    altCell: 4,
    headingBuckets: 16,
    pitchBuckets: 4,
    speedQuant: 4,
    levelDivisors: [4, 2, 1],
    goalRadius: 9,
    goalHeadingTol: Infinity,
    primDuration: 1,
    substeps: 4,
    analyticExpansion,
  });
  return { env, start, goal };
}

function makeOpen(analyticExpansion: AircraftEnvOptions['analyticExpansion'] = false) {
  const airspace = new InMemoryAirspace({
    floor: AIR_BOUNDS.floor,
    ceiling: AIR_BOUNDS.ceiling,
  });
  const start = gate(8, 22, 0);
  const goal = gate(152, 22, 0);
  const env = new AircraftEnvironment(airspace, AIR_AGENT, {
    posCell: 4,
    altCell: 4,
    headingBuckets: 16,
    pitchBuckets: 4,
    speedQuant: 4,
    levelDivisors: [4, 2, 1],
    goalRadius: 9,
    goalHeadingTol: Infinity,
    primDuration: 1,
    substeps: 4,
    analyticExpansion,
  });
  return { env, start, goal };
}

function makeRestricted() {
  const radius = 22;
  const z0 = { x: 86, y: 34, z0: -54, vz: 7, horizon: 60 };
  const zoneAt = (t: number) => {
    if (t < 0 || t > z0.horizon) return null;
    return { x: z0.x, y: z0.y, z: z0.z0 + z0.vz * t };
  };
  const zones: MovingZone[] = [{ radius, predict: zoneAt }];
  const airspace = new InMemoryAirspace({
    floor: AIR_BOUNDS.floor,
    ceiling: AIR_BOUNDS.ceiling,
    zones,
  });
  const start = gate(8, 34, 0);
  const goal = gate(152, 34, 0);
  const env = new AircraftEnvironment(airspace, AIR_AGENT, {
    posCell: 4,
    altCell: 4,
    headingBuckets: 16,
    pitchBuckets: 4,
    speedQuant: 4,
    levelDivisors: [4, 2, 1],
    goalRadius: 9,
    goalHeadingTol: Infinity,
    primDuration: 1,
    substeps: 4,
  });
  return { env, start, goal };
}

function makeKnifeEdge() {
  const f = AIR_BOUNDS.floor;
  const c = AIR_BOUNDS.ceiling;
  const boxes: AABB[] = [
    { min: [78, f, -60], max: [92, c, -0.6] },
    { min: [78, f, 0.6], max: [92, c, 60] },
  ];
  const airspace = new InMemoryAirspace({ floor: f, ceiling: c, boxes });
  const start = gate(8, 24, 0);
  const goal = gate(152, 24, 0);
  const env = new AircraftEnvironment(airspace, AIR_AGENT, {
    posCell: 4,
    altCell: 4,
    headingBuckets: 16,
    pitchBuckets: 4,
    rollBuckets: 4,
    speedQuant: 4,
    levelDivisors: [4, 2, 1],
    goalRadius: 10,
    goalHeadingTol: Infinity,
    primDuration: 1,
    substeps: 4,
    rollFractions: [-1, 0, 1],
  });
  return { env, start, goal };
}

// ---------- R2 grid: baseline -----------------------------------------------

function makeR2() {
  const blocked = (cx: number, cy: number): boolean =>
    cx === 20 && cy >= 0 && cy < 30;
  const env = new R2Environment({
    step: 1,
    blocked,
    bounds: { minCx: 0, maxCx: 50, minCy: 0, maxCy: 40 },
  });
  return { env, start: { x: 1, y: 1 }, goal: { x: 48, y: 38 } };
}

function logBreakdown<S>(
  name: string,
  build: () => { env: Parameters<typeof plan<S>>[0]['environment']; start: S; goal: S },
  maxExpansions = 50_000,
): void {
  const { env, start, goal } = build();
  const t0 = performance.now();
  const result = plan<S>(
    { start, goal, environment: env, options: { maxExpansions, profile: 'timings' } },
    Infinity,
  ) as PlanResult<S>;
  const wall = performance.now() - t0;
  // Use stderr — vitest bench output captures stderr inline.
  process.stderr.write(
    `\n=== ${name} (wall ${wall.toFixed(1)} ms) ===\n` +
      `found=${result.found}  cost=${
        result.cost === Infinity ? 'inf' : result.cost.toFixed(2)
      }  passes=${result.stats.passesRun}\n${formatPerf(result.stats)}\n`,
  );
}

// Module-level breakdown — runs once at file load, before any benches.
logBreakdown<AircraftState>('aircraft / open (no shot)', () => makeOpen(false), 80_000);
logBreakdown<AircraftState>('aircraft / open (analytic shot)', () => makeOpen({}), 80_000);
logBreakdown<AircraftState>('aircraft / canyon (no shot)', () => makeCanyon(false), 80_000);
logBreakdown<AircraftState>('aircraft / canyon (analytic shot)', () => makeCanyon({}), 80_000);
logBreakdown<AircraftState>('aircraft / restricted-airspace', makeRestricted, 80_000);
logBreakdown<AircraftState>('aircraft / knife-edge', makeKnifeEdge, 80_000);
logBreakdown('r2 / wall-with-gap', makeR2, 10_000);

describe('IGHA* perf breakdown', () => {
  bench('aircraft canyon (no shot, 80k budget)', () => {
    const { env, start, goal } = makeCanyon(false);
    plan<AircraftState>(
      { start, goal, environment: env, options: { maxExpansions: 80_000 } },
      Infinity,
    );
  });
  bench('aircraft canyon (analytic shot, 80k budget)', () => {
    const { env, start, goal } = makeCanyon({});
    plan<AircraftState>(
      { start, goal, environment: env, options: { maxExpansions: 80_000 } },
      Infinity,
    );
  });
  bench('aircraft restricted-airspace (80k budget)', () => {
    const { env, start, goal } = makeRestricted();
    plan<AircraftState>(
      { start, goal, environment: env, options: { maxExpansions: 80_000 } },
      Infinity,
    );
  });
  bench('aircraft knife-edge (80k budget)', () => {
    const { env, start, goal } = makeKnifeEdge();
    plan<AircraftState>(
      { start, goal, environment: env, options: { maxExpansions: 80_000 } },
      Infinity,
    );
  });
  bench('r2 wall-with-gap (10k budget)', () => {
    const { env, start, goal } = makeR2();
    plan({ start, goal, environment: env, options: { maxExpansions: 10_000 } }, Infinity);
  });
});
