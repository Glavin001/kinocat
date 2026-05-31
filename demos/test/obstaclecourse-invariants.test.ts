// Headless obstacle-course invariants — first-ever headless coverage of the
// /obstaclecourse behaviour, driving the SAME createObstacleCourseScenario the
// web page uses. Asserts determinism + waypoint progress + no building
// collision + planner health, all teleport-free.

import { describe, expect, it } from 'vitest';
import { ensureRapier } from 'kinocat/adapters/rapier';
import {
  createObstacleCourseScenario,
  PHYSICS_DT,
  type ObstacleCourseScenario,
} from '../app/lib/obstaclecourse-scenario';
import { OBS_AGENT, buildObstacleCourse, OBS_BLOCKS_ALL } from '../app/lib/obstaclecourse-scenarios';
import { createSimMonitor, formatReport } from '../app/lib/sim-monitor';

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

/** Real (un-inflated) building footprints — a physical collision means the car
 *  body actually overlapped a building, not merely the planner's safety margin
 *  (course.obstacles are inflated by BUILDING_INFLATE). */
function realBuildingFootprints(course: ReturnType<typeof buildObstacleCourse>): Array<[number, number][]> {
  return course.buildings.map((b) => [
    [b.x - b.hx, b.z - b.hz],
    [b.x + b.hx, b.z - b.hz],
    [b.x + b.hx, b.z + b.hz],
    [b.x - b.hx, b.z + b.hz],
  ]);
}

async function runObs(maxTicks: number) {
  const scenario = await createObstacleCourseScenario();
  const course = buildObstacleCourse(OBS_BLOCKS_ALL);
  const monitor = createSimMonitor({
    footprint: OBS_AGENT.footprint,
    obstacles: realBuildingFootprints(course),
    dt: PHYSICS_DT,
  });
  const poses: Pose[] = [];
  let maxLoopIndex = 0;
  for (let i = 0; i < maxTicks; i++) {
    scenario.tick();
    const st = scenario.status();
    monitor.sample(st);
    poses.push({ x: st.state.x, z: st.state.z, heading: st.state.heading, speed: st.state.speed });
    maxLoopIndex = Math.max(maxLoopIndex, st.loopIndex);
  }
  const report = monitor.summary();
  const status = scenario.status();
  scenario.dispose();
  return { report, poses, status, maxLoopIndex };
}

describe.skipIf(!RAPIER_OK)('obstacle-course invariants', () => {
  it('drives the course deterministically, clearing waypoints without hitting a building', { timeout: 90000, retry: 0 }, async () => {
    const a = await runObs(480); // 8 s
    const ctx = `\n${formatReport(a.report)}\n maxLoopIndex=${a.maxLoopIndex}`;

    // Actually progressing around the waypoint loop.
    expect(a.maxLoopIndex, ctx).toBeGreaterThan(0);

    // Finite state, no blow-ups.
    expect(Number.isFinite(a.status.state.x), ctx).toBe(true);
    expect(Number.isFinite(a.status.state.speed), ctx).toBe(true);

    // Never overlapped a building footprint.
    expect(a.report.collided, ctx).toBe(false);

    // Teleport-free by construction.
    expect(a.report.teleports, ctx).toBe(0);

    // Planner is healthy (not failing every replan).
    expect(a.report.failedReplanRatio, ctx).toBeLessThan(0.5);
  });

  it('is bit-for-bit reproducible across two independent runs', { timeout: 90000, retry: 0 }, async () => {
    const a = await runObs(240);
    const b = await runObs(240);
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
