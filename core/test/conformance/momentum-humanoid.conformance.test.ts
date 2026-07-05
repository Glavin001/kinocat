// The fourth domain through the same battery as the other three — including
// wrapped in the generalized TimeAwareEnvironment with a moving pedestrian.
// This file existing (and passing) is the point of the whole exercise: a new
// inertial agent, added with zero planner-core edits, proven by the same
// packaged contract checks.

import { describe, it, expect } from 'vitest';
import { runConformance, type DomainHarness } from '../../src/testing';
import { MomentumHumanoidEnvironment } from '../../src/environment/momentum-humanoid-environment';
import { TimeAwareEnvironment } from '../../src/environment/time-aware';
import { InMemoryNavWorld } from '../../src/environment/nav-world';
import {
  defaultMomentumHumanoidAgent,
  momentumHumanoidForwardSim,
} from '../../src/agent/momentum-humanoid';
import type { FidelityHooks } from '../../src/testing';
import { linearObstacle } from '../../src/predict/factories';
import type { MomentumHumanoidState } from '../../src/agent/types';
import { rect } from '../fixtures/vehicle-sweep';

const agent = defaultMomentumHumanoidAgent();

// This environment DELIBERATELY applies primitives from nearest-bucket
// canonical starts (speed × velocity-direction buckets) — a fidelity/speed
// trade the fidelity check MEASURES rather than forbids. Velocity teleport
// ceiling per edge: ~1.5 m/s of speed-bucket rounding, plus the sprint
// bucket's direction quantization (relDir 0 only, so a sampled sprint with
// full lateral strafe is ~0.4 rad off at 5 m/s ≈ 2 m/s), amplified by up to
// maxDecel·0.5 s of divergent braking response — call it 5. A frame or
// rotation bug produces deviations at position scale (tens), far past it.
const sim = momentumHumanoidForwardSim(agent);
const PRIM_DURATION = 0.5; // env defaults; the hook must match succ()'s
const SUBSTEPS = 4;
const fidelity: FidelityHooks<MomentumHumanoidState> = {
  tolerance: 5,
  angularFields: ['heading'],
  resimulate: (parent, edge) => {
    if (edge.kind !== 'move') return null;
    const d = edge.data as { controls: number[] };
    const dt = PRIM_DURATION / SUBSTEPS;
    let s = parent;
    for (let i = 0; i < SUBSTEPS; i++) s = sim(s, d.controls, dt);
    return s;
  },
};

function sample(rand: () => number): MomentumHumanoidState {
  const heading = (rand() - 0.5) * 2 * Math.PI;
  // Sample inside the movement envelope: forward up to sprint, lateral up
  // to the strafe cap.
  const vFwd = rand() * agent.maxSpeed;
  const vLat = (rand() - 0.5) * 2 * agent.strafeSpeed;
  return {
    x: 1 + rand() * 16,
    z: 1 + rand() * 10,
    heading,
    vx: vFwd * Math.cos(heading) - vLat * Math.sin(heading),
    vz: vFwd * Math.sin(heading) + vLat * Math.cos(heading),
    t: rand() * 50,
  };
}

// Kinodynamic search over a 5-dim exact state is expansion-hungrier than the
// step-based humanoid, and the battery replans each scenario ~8 times — keep
// the fixture worlds small and the goals well-guided so the whole file stays
// CI-friendly.
const scenarios = [
  {
    name: 'open-sprint',
    start: { x: 2, z: 2, heading: 0, vx: 0, vz: 0, t: 0 },
    goal: { x: 16, z: 9, heading: 0, vx: 0, vz: 0, t: 0 },
    maxExpansions: 120_000,
  },
  {
    name: 'doorway',
    start: { x: 4, z: 9.5, heading: 0, vx: 0, vz: 0, t: 0 },
    goal: { x: 22, z: 9.5, heading: 0, vx: 0, vz: 0, t: 0 },
    maxExpansions: 200_000,
  },
];

describe('MomentumHumanoidEnvironment conformance', () => {
  it('open world passes the full battery', () => {
    const h: DomainHarness<MomentumHumanoidState> = {
      makeEnv: () =>
        new MomentumHumanoidEnvironment(
          new InMemoryNavWorld([rect(1, 0, 0, 18, 12)]),
          agent,
        ),
      sampleState: sample,
      fidelity,
      scenarios: [scenarios[0]!],
    };
    const report = runConformance(h);
    expect(report.failures).toEqual([]);
    expect(report.ok).toBe(true);
  });

  it('doorway world passes the full battery', () => {
    const h: DomainHarness<MomentumHumanoidState> = {
      makeEnv: () =>
        new MomentumHumanoidEnvironment(
          new InMemoryNavWorld([
            rect(1, 0, 4, 12, 15),
            rect(2, 14, 4, 26, 15),
            rect(3, 11, 8, 15, 11), // 3 m-wide doorway
          ]),
          agent,
        ),
      sampleState: (rand) => {
        const s = sample(rand);
        return { ...s, x: 1 + rand() * 10, z: 5 + rand() * 9 };
      },
      fidelity,
      scenarios: [scenarios[1]!],
    };
    const report = runConformance(h);
    expect(report.failures).toEqual([]);
  });

  it('wrapped in TimeAware with a moving pedestrian passes the full battery', () => {
    const h: DomainHarness<MomentumHumanoidState> = {
      makeEnv: () =>
        new TimeAwareEnvironment(
          new MomentumHumanoidEnvironment(
            new InMemoryNavWorld([rect(1, 0, 0, 18, 12)]),
            agent,
          ),
          {
            obstacles: [linearObstacle(9, 0, 0, 0.8, 0.4)],
            agentRadius: agent.radius,
          },
        ),
      sampleState: sample,
      fidelity,
      scenarios: [scenarios[0]!],
    };
    const report = runConformance(h);
    expect(report.failures).toEqual([]);
  });
});
