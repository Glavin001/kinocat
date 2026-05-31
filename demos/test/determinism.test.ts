// Determinism regression test — the prerequisite for trustworthy headless
// testing. Same construction + same inputs must produce the same trajectory,
// every run. If this ever fails, every other scenario assertion is suspect.
//
// Two checks:
//   1. Cross-instance: build the scenario twice and tick both; trajectories
//      must match. (Each instance gets its own Rapier world.)
//   2. reset(): run, reset, run again on the SAME instance; the two runs must
//      match — exercises the runner's reset path independent of cross-world
//      float reproducibility.
//
// We use the kinematic race entry with the default pure-pursuit tracker: no
// RNG is involved (the MPC/MPPI sampler would introduce a seed), so any
// divergence is a genuine determinism bug, not sampling noise.

import { describe, expect, it } from 'vitest';
import { ensureRapier } from 'kinocat/adapters/rapier';
import {
  createRaceScenario,
  type RaceScenarioOptions,
} from '../app/lib/race-scenario';
import { buildKinematicLibrary } from '../app/lib/race-primitives-scenarios';

let RAPIER_OK = false;
try {
  await ensureRapier();
  RAPIER_OK = true;
} catch {
  RAPIER_OK = false;
}

// Start tight (bit-exact). If a CI runner ever shows Rapier float drift across
// worlds, relax THIS constant (and only this) to a tolerance like 1e-6 — the
// escalation path is documented here so it's a one-line change.
const DETERMINISM_EPS = 1e-9;

interface Pose {
  x: number;
  z: number;
  heading: number;
  speed: number;
}

const OPTS: RaceScenarioOptions = {
  entries: [{ name: 'kin', lib: buildKinematicLibrary() }],
  syncHold: false,
  offTrackRecovery: 'spawn',
};

async function runTrajectory(n: number): Promise<Pose[]> {
  const scenario = await createRaceScenario(OPTS);
  const out: Pose[] = [];
  for (let i = 0; i < n; i++) {
    scenario.tick();
    const s = scenario.status()[0]!.state;
    out.push({ x: s.x, z: s.z, heading: s.heading, speed: s.speed });
  }
  scenario.dispose();
  return out;
}

function expectSameTrajectory(a: Pose[], b: Pose[], eps: number): void {
  expect(a.length).toBe(b.length);
  for (let i = 0; i < a.length; i++) {
    const pa = a[i]!;
    const pb = b[i]!;
    const msg = `tick ${i}: ${JSON.stringify(pa)} vs ${JSON.stringify(pb)}`;
    expect(Math.abs(pa.x - pb.x), msg).toBeLessThanOrEqual(eps);
    expect(Math.abs(pa.z - pb.z), msg).toBeLessThanOrEqual(eps);
    expect(Math.abs(pa.heading - pb.heading), msg).toBeLessThanOrEqual(eps);
    expect(Math.abs(pa.speed - pb.speed), msg).toBeLessThanOrEqual(eps);
  }
}

describe.skipIf(!RAPIER_OK)('determinism', () => {
  it('two independent runs of the same scenario produce identical trajectories', { timeout: 90000, retry: 0 }, async () => {
    const a = await runTrajectory(240);
    const b = await runTrajectory(240);
    expectSameTrajectory(a, b, DETERMINISM_EPS);
  });

  it('reset() returns the scenario to a bit-identical starting point', { timeout: 90000, retry: 0 }, async () => {
    const scenario = await createRaceScenario(OPTS);
    const first: Pose[] = [];
    for (let i = 0; i < 120; i++) {
      scenario.tick();
      const s = scenario.status()[0]!.state;
      first.push({ x: s.x, z: s.z, heading: s.heading, speed: s.speed });
    }
    scenario.reset();
    const second: Pose[] = [];
    for (let i = 0; i < 120; i++) {
      scenario.tick();
      const s = scenario.status()[0]!.state;
      second.push({ x: s.x, z: s.z, heading: s.heading, speed: s.speed });
    }
    scenario.dispose();
    expectSameTrajectory(first, second, DETERMINISM_EPS);
  });
});
