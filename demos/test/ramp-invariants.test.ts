// Headless ramp invariants — first-ever headless coverage of the /ramp demo,
// driving the SAME createRampScenario the web page uses. Asserts determinism,
// goal-reaching over the ramp, the jump affordance firing, and teleport-free
// honesty (no rescue).

import { describe, expect, it } from 'vitest';
import { ensureRapier } from 'kinocat/adapters/rapier';
import { createRampScenario, PHYSICS_DT } from '../app/lib/ramp-scenario';
import { createSimMonitor, formatReport } from '../app/lib/sim-monitor';
import { RAMP_AGENT } from '../app/lib/ramp-scenarios';

let RAPIER_OK = false;
try {
  await ensureRapier();
  RAPIER_OK = true;
} catch {
  RAPIER_OK = false;
}

interface Pose {
  x: number;
  z: number;
  heading: number;
  speed: number;
}

async function runRamp(maxTicks: number, affordance: boolean) {
  const scenario = await createRampScenario({ affordance });
  const car = scenario.getCar();
  const goal = scenario.status().goal;
  const monitor = createSimMonitor({
    footprint: RAMP_AGENT.footprint,
    obstacles: [],
    dt: PHYSICS_DT,
    goal: { x: goal.x, z: goal.z, heading: goal.heading },
    success: { posTol: 3, headingTol: Math.PI, speedTol: 100 }, // reach the pad area
  });
  const poses: Pose[] = [];
  let everUsedAffordance = false;
  let reached = false;
  let maxAirborneY = -Infinity; // peak chassis height — how high the car flew
  let maxLateral = 0; // peak |z| excursion — detour swings wide, jump hugs z≈0
  for (let i = 0; i < maxTicks; i++) {
    scenario.tick();
    const st = scenario.status();
    monitor.sample(st);
    poses.push({ x: st.state.x, z: st.state.z, heading: st.state.heading, speed: st.state.speed });
    if (st.diagnostics.usedAffordance) everUsedAffordance = true;
    maxAirborneY = Math.max(maxAirborneY, car.chassis.translation().y);
    maxLateral = Math.max(maxLateral, Math.abs(st.state.z));
    if (Math.hypot(st.state.x - goal.x, st.state.z - goal.z) < 3) {
      reached = true;
      break;
    }
  }
  const report = monitor.summary();
  scenario.dispose();
  return { report, poses, everUsedAffordance, reached, maxAirborneY, maxLateral };
}

describe.skipIf(!RAPIER_OK)('ramp invariants', () => {
  it('drives over the ramp to the goal, deterministically and teleport-free', { timeout: 90000, retry: 0 }, async () => {
    const a = await runRamp(900, true);
    const ctx = `\n${formatReport(a.report)}\n reached=${a.reached} usedAffordance=${a.everUsedAffordance}`;
    expect(a.reached, ctx).toBe(true);
    expect(a.report.teleports, ctx).toBe(0);
    expect(a.report.netProgress, ctx).toBeGreaterThan(0);
    expect(a.report.failedReplanRatio, ctx).toBeLessThan(0.5);
  });

  it('takes the jump affordance when it is enabled', { timeout: 90000, retry: 0 }, async () => {
    const a = await runRamp(900, true);
    expect(a.everUsedAffordance, `\n${formatReport(a.report)}`).toBe(true);
  });

  it('PHYSICALLY flies the jump — airborne + centreline, never detours around the gap', { timeout: 90000, retry: 0 }, async () => {
    // The regression the planner-flag test above could not see: even with the
    // planner reporting "affordance used", the car used to physically drive
    // AROUND the gap (|z| ~17 m, wheels never leaving the ground). Assert the
    // real trajectory — a genuine ballistic launch, hugging the z=0 centreline.
    const a = await runRamp(900, true);
    const ctx = `maxAirborneY=${a.maxAirborneY.toFixed(2)} maxLateral=${a.maxLateral.toFixed(2)} reached=${a.reached}`;
    expect(a.reached, ctx).toBe(true);
    // Chassis resting height is ~0.8 m; a real launch clears well above that.
    expect(a.maxAirborneY, ctx).toBeGreaterThan(1.5);
    // The gap is ~30 m wide (half-width 15). Detour swings |z|>15; the jump
    // stays on the ramp centreline. A generous bound cleanly separates them.
    expect(a.maxLateral, ctx).toBeLessThan(6);
  });

  it('WITHOUT the affordance it detours around the gap instead of flying', { timeout: 90000, retry: 0 }, async () => {
    // Contrast: disabling the affordance must produce the opposite trajectory —
    // a wide ground detour, no launch. Proves the jump test above is measuring
    // the affordance choice, not some incidental always-true behaviour.
    const a = await runRamp(1400, false);
    const ctx = `maxAirborneY=${a.maxAirborneY.toFixed(2)} maxLateral=${a.maxLateral.toFixed(2)} reached=${a.reached}`;
    expect(a.maxLateral, ctx).toBeGreaterThan(10);
    expect(a.maxAirborneY, ctx).toBeLessThan(1.5);
  });

  it('is bit-for-bit reproducible across two independent runs', { timeout: 90000, retry: 0 }, async () => {
    const a = await runRamp(240, true);
    const b = await runRamp(240, true);
    expect(a.poses.length).toBe(b.poses.length);
    for (let i = 0; i < a.poses.length; i++) {
      const pa = a.poses[i]!;
      const pb = b.poses[i]!;
      const msg = `tick ${i}: ${JSON.stringify(pa)} vs ${JSON.stringify(pb)}`;
      expect(Math.abs(pa.x - pb.x), msg).toBeLessThanOrEqual(1e-9);
      expect(Math.abs(pa.z - pb.z), msg).toBeLessThanOrEqual(1e-9);
      expect(Math.abs(pa.heading - pb.heading), msg).toBeLessThanOrEqual(1e-9);
      expect(Math.abs(pa.speed - pb.speed), msg).toBeLessThanOrEqual(1e-9);
    }
  });
});
