import { bench, describe } from 'vitest';
import { plan } from '../src/planner/ighastar';
import { InMemoryNavWorld } from '../src/environment/nav-world';
import { VehicleEnvironment } from '../src/environment/vehicle-environment';
import { TimeAwareEnvironment } from '../src/environment/time-aware';
import type { TimeAwareOptions } from '../src/environment/time-aware';
import { defaultVehicleAgent, kinematicForwardSim } from '../src/agent/vehicle';
import { characterizeVehicle } from '../src/primitives/characterize';
import { asObstacle } from '../src/predict/factories';
import { PlanRegistry } from '../src/predict/plan-registry';
import type { CarKinematicState, AgentState } from '../src/agent/types';

// Moving-obstacle broadphase for the real-time MULTI-AGENT case: every NPC
// treats the others' published plans as obstacles via
// `asObstacle(registry.predictNPC(id))`. `predictNPC` does an O(plan-length)
// interpolation scan per call and the plan occupies a localized tube over a
// bounded time window — exactly what the time/AABB cheap-reject eliminates.
// Same fixed query; exact per-successor scan vs. pre-sampled broadphase.

const agent = defaultVehicleAgent({
  minTurnRadius: 3,
  maxSpeed: 8,
  maxReverseSpeed: 4,
  footprint: [
    [1.2, 0.6],
    [-1.2, 0.6],
    [-1.2, -0.6],
    [1.2, -0.6],
  ],
});
const k = 1 / agent.minTurnRadius;
const lib = characterizeVehicle({
  forwardSim: kinematicForwardSim(agent),
  controlSets: [
    [0, 6],
    [k, 6],
    [-k, 6],
    [k / 2, 6],
    [-k / 2, 6],
  ],
  duration: 0.5,
  substeps: 6,
  startSpeeds: [0],
});

// 12 "other NPCs", each a published plan of 80 timestamped states tracing a
// crossing arc over t ∈ [0, 8] (predictNPC scans these per query).
const registry = new PlanRegistry();
for (let n = 0; n < 12; n++) {
  const states: AgentState[] = [];
  const z0 = n % 2 === 0 ? 13 : -13;
  const vz = n % 2 === 0 ? -3.2 : 3.2;
  const x0 = 7 + n * 2.4;
  for (let s = 0; s < 80; s++) {
    const t = (s / 79) * 8;
    states.push({ x: x0 + Math.sin(t) * 1.5, z: z0 + vz * t, heading: 0, speed: 0, t });
  }
  registry.publish(`npc${n}`, states);
}
const obstacles = Array.from({ length: 12 }, (_, n) =>
  asObstacle(registry.predictNPC(`npc${n}`), 1.7),
);

const start: CarKinematicState = { x: 2, z: 0, heading: 0, speed: 0, t: 0 };
const goal: CarKinematicState = { x: 40, z: 0, heading: 0, speed: 0, t: 0 };

function run(broadphase: TimeAwareOptions['broadphase']): void {
  const world = new InMemoryNavWorld([
    { id: 1, y: 0, ring: [[0, -16], [44, -16], [44, 16], [0, 16]] },
  ]);
  const base = new VehicleEnvironment(world, agent, lib, {
    goalRadius: 1.5,
    goalHeadingTol: Infinity,
  });
  const env = new TimeAwareEnvironment(base, {
    obstacles,
    agentRadius: 1.4,
    broadphase,
  });
  plan(
    { start, goal, environment: env, options: { maxExpansions: 60000 } },
    Infinity,
  );
}

describe('TimeAwareEnvironment moving-obstacle broadphase (before vs after)', () => {
  bench('before — broadphase DISABLED (exact predictNPC scan per successor)', () => {
    run(false);
  });
  bench('after — broadphase ENABLED (pre-sampled AABB + time cheap-reject)', () => {
    run({});
  });
});
