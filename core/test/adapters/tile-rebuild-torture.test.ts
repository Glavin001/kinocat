// THE torture test (charter done-when gate): destroy the ground under an
// agent's committed path mid-execution and prove it recovers — the changed
// region is auto-detected (only the crossing agent replans), a replacement
// plan is adopted within the 100 ms replan budget, and the agent neither
// teleports nor freezes while rerouting around the hole to the goal.
//
// Full stack, end to end: real navcat navmesh (regenerated with a hole and
// swapped in via `swapNavMesh`), region-scoped `markTileRebuilt`, the
// ReplanState executor, pure-pursuit tracking, and a 60 fps kinematic sim.

import { describe, it, expect } from 'vitest';
import {
  NavcatWorld,
  navWorldFromTriangleMesh,
  markTileRebuilt,
} from '../../src/adapters/navcat/index';
import { singlePlaneMesh, planeWithHoleMesh } from '../fixtures/mini-navmesh';
import { ReplanState } from '../../src/execute/replan';
import { purePursuit } from '../../src/execute/pure-pursuit';
import {
  planCrossesRegion,
  footprintCircumradius,
  type ChangedRegion,
} from '../../src/execute/invalidation';
import { planVehicleOnce } from '../../src/planner/plan-vehicle';
import { defaultVehicleAgent, kinematicForwardSim } from '../../src/agent/vehicle';
import { characterizeVehicle } from '../../src/primitives/characterize';
import type { CarKinematicState } from '../../src/agent/types';

// ---- fixtures (build-once, skipIf on navcat surprises) --------------------
const AGENT = defaultVehicleAgent({
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
const K = 1 / AGENT.minTurnRadius;
const LIB = characterizeVehicle({
  forwardSim: kinematicForwardSim(AGENT),
  controlSets: [
    [0, 6],
    [K, 6],
    [-K, 6],
    [K / 2, 6],
    [-K / 2, 6],
    [0, -3],
  ],
  duration: 0.5,
  substeps: 4,
  startSpeeds: [0],
});

// The hole spans x∈[13,17], z∈[6,20]: it cuts the committed straight line at
// z=10 but leaves a drivable southern strip (z<6) to reroute through.
const HOLE = { x0: 13, z0: 6, x1: 17, z1: 20 };
const HOLE_REGION: ChangedRegion = HOLE;

let built: ReturnType<typeof navWorldFromTriangleMesh> | null = null;
let rebuilt: ReturnType<typeof navWorldFromTriangleMesh> | null = null;
try {
  const m = singlePlaneMesh();
  built = navWorldFromTriangleMesh(m.positions, m.indices, { cellSize: 0.3 }, {
    clearanceField: true,
  });
  const h = planeWithHoleMesh(HOLE);
  // Throwaway world — only the regenerated navMesh + CHF get swapped in.
  rebuilt = navWorldFromTriangleMesh(h.positions, h.indices, { cellSize: 0.3 }, {
    clearanceField: true,
  });
} catch {
  built = null;
  rebuilt = null;
}
const OK =
  built !== null &&
  rebuilt !== null &&
  built.world.polygonAt(15, 10) !== null &&
  rebuilt.world.polygonAt(15, 3) !== null &&
  rebuilt.world.polygonAt(15, 10) === null;

describe.skipIf(!OK)('torture: destroy the ground under a committed path', () => {
  it('auto-detects, replans <100ms, no teleport, no freeze, avoids the hole', () => {
    const world = new NavcatWorld(built!.navMesh);
    world.attachClearanceField(built!.compactHeightfield);

    const GOAL: CarKinematicState = { x: 27, z: 10, heading: 0, speed: 0, t: 0 };
    // The system's own arrival contract: planVehicleOnce's default goal
    // acceptance (goalRadius 4) — the executor latches its terminal stop
    // inside this region, so "reached" must mean the same thing here.
    const GOAL_RADIUS = 4;
    const doPlan = (start: CarKinematicState, deadlineMs: number) =>
      planVehicleOnce({
        start: { ...start, t: 0 },
        goal: GOAL,
        world,
        agent: AGENT,
        lib: LIB,
        deadlineMs,
        maxExpansions: 25000,
      });

    // Committed initial plan — straight across what will become the hole.
    let state: CarKinematicState = { x: 2, z: 10, heading: 0, speed: 0, t: 0 };
    const rs = new ReplanState({ divergenceThresholdMeters: 2, refreshIntervalMs: 400 });
    const initial = doPlan(state, Infinity);
    expect(initial.found).toBe(true);
    rs.setPlan(initial.path, 0, initial.cost);
    expect(planCrossesRegion(rs.currentPlan!, HOLE_REGION)).toBe(true);

    // A control agent whose committed plan hugs the south-west corner —
    // nowhere near the hole. It must NOT be marked.
    const rsControl = new ReplanState({ divergenceThresholdMeters: 2, refreshIntervalMs: 400 });
    rsControl.setPlan(
      [
        { x: 2, z: 2, heading: 0, speed: 4, t: 0 },
        { x: 8, z: 2, heading: 0, speed: 4, t: 1.5 },
      ],
      0,
    );

    const sim = kinematicForwardSim(AGENT);
    const PP = {
      lookaheadMin: 2,
      lookaheadGain: 0.3,
      lookaheadMax: 6,
      maxLateralAccel: 6,
      maxAccel: 8,
      maxDecel: 8,
      cruiseSpeed: AGENT.maxSpeed,
      goalTolerance: 1.5,
    };
    const inflate = footprintCircumradius(AGENT.footprint);

    const DT = 1 / 60;
    const MAX_TICKS = 60 * 90; // 90 sim-seconds — freeze = never reaching the goal
    const stepMax = AGENT.maxSpeed * DT * 1.05; // teleport bound
    let destroyed = false;
    let recoveryMs = -1;
    let recoveryAdopted = false;
    let marked: ReplanState[] = [];
    let reachedAt = -1;
    const executed: CarKinematicState[] = [state];

    for (let tick = 0; tick < MAX_TICKS; tick++) {
      const nowMs = tick * (1000 * DT);

      // ---- the catastrophe: agent underway, ground ahead vanishes -------
      if (!destroyed && state.x >= 8) {
        destroyed = true;
        world.swapNavMesh(rebuilt!.navMesh, rebuilt!.compactHeightfield);
        marked = markTileRebuilt(world, HOLE_REGION, [
          { replan: rs, inflate },
          { replan: rsControl, inflate },
        ]);
        // Recovery must fit the replan budget: dirty → plan → adopt.
        const t0 = performance.now();
        expect(rs.shouldReplan(state, nowMs)).toBe(true);
        const r = doPlan(state, 80);
        recoveryAdopted = r.path.length > 1 && rs.consider(r.path, r.cost, nowMs);
        recoveryMs = performance.now() - t0;
      }

      // ---- executor tick (adopt-only-usable-plans, like the demos) ------
      const path = rs.currentPlan;
      if (path && path.length > 1) {
        const cmd = purePursuit(state, path, PP);
        if (!cmd.atGoal) {
          const next = sim(state, [cmd.steering, cmd.targetSpeed], DT);
          state = { ...next, t: 0 };
          executed.push(state);
        } else if (Math.hypot(state.x - GOAL.x, state.z - GOAL.z) > GOAL_RADIUS) {
          // Exhausted the committed plan short of the actual goal — the
          // demos' `markDirty('plan-end')` pattern forces adoption of the
          // next replan instead of letting switch-hysteresis reject it.
          rs.markDirty('plan-end');
        }
      }
      if (Math.hypot(state.x - GOAL.x, state.z - GOAL.z) <= GOAL_RADIUS) {
        reachedAt = tick;
        break;
      }
      if (destroyed && rs.shouldReplan(state, nowMs)) {
        const r = doPlan(state, 80);
        if (r.path.length > 1) rs.consider(r.path, r.cost, nowMs);
      }
    }

    // 1. Auto-detect: exactly the crossing agent was marked, not the control.
    expect(marked).toEqual([rs]);
    expect(rsControl.shouldReplan({ x: 2, z: 2, heading: 0, speed: 4, t: 0 }, 1)).toBe(false);

    // 2. The recovery replan (dirty → plan → adopt) fit the 100 ms budget.
    expect(recoveryAdopted).toBe(true);
    expect(recoveryMs).toBeGreaterThanOrEqual(0);
    expect(recoveryMs).toBeLessThan(100);

    // 3. No teleport: every executed step within the kinematic bound.
    for (let i = 1; i < executed.length; i++) {
      const step = Math.hypot(
        executed[i]!.x - executed[i - 1]!.x,
        executed[i]!.z - executed[i - 1]!.z,
      );
      expect(step).toBeLessThanOrEqual(stepMax);
    }

    // 4. No freeze: the agent reached the goal within the sim budget.
    expect(reachedAt).toBeGreaterThan(0);

    // 5. Avoidance: the final committed plan clears the hole, and no
    //    executed pose ever entered it.
    expect(planCrossesRegion(rs.currentPlan!, HOLE_REGION, 0)).toBe(false);
    for (const p of executed) {
      const inHole =
        p.x >= HOLE.x0 && p.x <= HOLE.x1 && p.z >= HOLE.z0 && p.z <= HOLE.z1;
      expect(inHole).toBe(false);
    }

    console.info(
      `torture: recovery=${recoveryMs.toFixed(1)}ms adopted=${recoveryAdopted} ` +
        `reached@${(reachedAt / 60).toFixed(1)}s steps=${executed.length}`,
    );
  }, 120000);
});
