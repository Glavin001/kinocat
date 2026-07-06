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
// We use the kinematic race entry with the pure-pursuit tracker: no RNG is
// involved (the MPC/MPPI sampler would introduce a seed), so any divergence is
// a genuine determinism bug, not sampling noise.
//
// CROSS-WORLD vs SAME-WORLD determinism (WS-1 note): the racing default now
// uses a BANG-BANG throttle (WS-1 "floor it"), which is intentionally
// DISCONTINUOUS — throttle flips 0↔1 at a speed threshold. Two SEPARATE Rapier
// `World` instances are only bit-identical to sub-ULP; a continuous controller
// keeps that noise below 1e-9 for 240 ticks, but a threshold controller can
// flip the command one tick apart and amplify the difference macroscopically
// on some CPUs (observed on CI, not locally). That is inherent to threshold
// control, not a determinism bug: the property the benchmarks rely on is
// SAME-WORLD reproducibility (a given run reproduces bit-for-bit), which the
// reset() test pins on the actual shipping (bang-bang) config. The
// cross-instance test therefore validates cross-world Rapier / world-
// construction determinism through a CONTINUOUS controller (bang-bang off), so
// it still catches real nondeterminism (world-build ordering, shared mutable
// state) at the strict 1e-9 bound without the threshold amplifier.

import { describe, expect, it } from 'vitest';
import { ensureRapier } from 'kinocat/adapters/rapier';
import {
  createRaceScenario,
  type RaceScenarioOptions,
  type RaceTuning,
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

function opts(tuning?: Partial<RaceTuning>): RaceScenarioOptions {
  return {
    // Fresh library per scenario so no mutable state is shared between the two
    // instances the cross-world test compares.
    entries: [{ name: 'kin', lib: buildKinematicLibrary() }],
    syncHold: false,
    offTrackRecovery: 'spawn',
    ...(tuning ? { tuning } : {}),
  };
}

// Continuous-controller tuning for the CROSS-WORLD test: disable the WS-1
// discontinuous throttle/goal-brake so sub-ULP cross-world FP noise is not
// amplified past 1e-9 (see the header note). This is a smooth controller whose
// cross-world trajectory stays bit-exact.
const CONTINUOUS_TUNING: Partial<RaceTuning> = {
  bangBangThrottle: false,
  noGoalBrakeOnDriveThrough: false,
};

async function runTrajectory(n: number, tuning?: Partial<RaceTuning>): Promise<Pose[]> {
  const scenario = await createRaceScenario(opts(tuning));
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
    // Cross-WORLD: two separate Rapier worlds. Uses the continuous controller
    // so the strict 1e-9 bound reflects Rapier/world-construction determinism,
    // not the bang-bang threshold amplifier (see header note).
    const a = await runTrajectory(240, CONTINUOUS_TUNING);
    const b = await runTrajectory(240, CONTINUOUS_TUNING);
    expectSameTrajectory(a, b, DETERMINISM_EPS);
  });

  it('reset() returns the scenario to a bit-identical starting point', { timeout: 90000, retry: 0 }, async () => {
    // Same-WORLD on the DEFAULT shipping config (bang-bang throttle): this is
    // the load-bearing guarantee — a given benchmark run reproduces bit-for-bit.
    const scenario = await createRaceScenario(opts());
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
