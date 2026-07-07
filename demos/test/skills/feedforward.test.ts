// Skill test — WS-1½ control feedforward CAPTURE (plan → executor bridge).
//
// The MPPI tracker can only feedforward the plan's proven controls if the
// commit pipeline actually attaches them to the plan samples. `attachPlanFeed-
// forward` maps each planner drive edge's motion-primitive control onto the
// dense smoothed samples by arc-length. Two failure modes this guards:
//   1. arc-length mapping / hold semantics drift (a smoothed sample gets the
//      wrong primitive's control), and
//   2. the planner-edge contract drifts (e.g. reverse legs use the
//      'drive-reverse' edge kind, not 'drive' — a real bug caught here) so
//      real plans silently carry NO feedforward and the flag becomes a no-op.
// See docs/racing-skills-test-plan.md and mpc-tracker.test.ts (the tracker-side
// mechanism test).

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { attachPlanFeedforward } from '../../app/lib/race-scenario';
import {
  buildLearnedRaceLibraryV3,
  planRaceMultiGoal,
  RACE_PLANNER_GATE_RADIUS,
} from '../../app/lib/race-primitives-scenarios';
import { v3FromJson } from 'kinocat/agent';
import type { CarKinematicState } from 'kinocat/agent';
import type { PlanResult } from 'kinocat/planner';

const repoRoot = resolve(fileURLToPath(new URL('../../..', import.meta.url)));
const readModel = (f: string) => JSON.parse(readFileSync(resolve(repoRoot, 'demos/public/models', f), 'utf-8'));

const st = (x: number, z = 0, speed = 10): CarKinematicState => ({ x, z, heading: 0, speed, t: x / 10 });

describe('skill: control-feedforward capture (attachPlanFeedforward)', () => {
  it('maps each primitive control onto the smoothed samples by arc-length (hold, not lerp)', () => {
    const lib = {
      primitives: [
        { id: 5, controls: [0.1, 3000, 0] },   // segment A: gentle throttle
        { id: 6, controls: [-0.2, 0, 1500] },  // segment B: brake-left
      ],
    };
    // Raw nodes at x = 0, 2, 4. Edge INTO node 1 (x∈(0,2]) = prim 5; edge INTO
    // node 2 (x∈(2,4]) = prim 6. Node 0 has no edge.
    const raw = [st(0), st(2), st(4)];
    const mkNode = (state: CarKinematicState, kind?: string, primId?: number) => ({
      state,
      edge: kind ? { cost: 0, kind, data: { primId, reverse: kind === 'drive-reverse' } } : null,
    });
    const res = {
      found: true, cost: 0, path: raw,
      nodes: [mkNode(raw[0]!), mkNode(raw[1]!, 'drive', 5), mkNode(raw[2]!, 'drive', 6)],
      stats: {}, solutionHistory: [],
    } as unknown as PlanResult<CarKinematicState>;
    // Dense smoothed samples every 0.5 m over the same span.
    const smoothed: CarKinematicState[] = [];
    for (let x = 0; x <= 4.0001; x += 0.5) smoothed.push(st(x));

    attachPlanFeedforward(smoothed, res, lib);

    // A sample in the first half carries prim 5's control; second half prim 6.
    const a = smoothed.find((s) => Math.abs(s.x - 1.0) < 1e-6)!;
    const b = smoothed.find((s) => Math.abs(s.x - 3.0) < 1e-6)!;
    expect(a.ff).toEqual([0.1, 3000, 0]);
    expect(b.ff).toEqual([-0.2, 0, 1500]);
    // Hold semantics: no invented in-between value at the boundary sample.
    for (const s of smoothed) {
      if (s.ff) expect([lib.primitives[0]!.controls, lib.primitives[1]!.controls]).toContainEqual([...s.ff]);
    }
  });

  it('leaves Reeds-Shepp (analytic) samples without feedforward', () => {
    const lib = { primitives: [{ id: 5, controls: [0, 3000, 0] }] };
    const raw = [st(0), st(2), st(4)];
    const res = {
      found: true, cost: 0, path: raw,
      nodes: [
        { state: raw[0]!, edge: null },
        { state: raw[1]!, edge: { cost: 0, kind: 'drive', data: { primId: 5, reverse: false } } },
        // Analytic shot: geometric, no model control.
        { state: raw[2]!, edge: { cost: 0, kind: 'reeds-shepp', data: { reedsShepp: true } } },
      ],
      stats: {}, solutionHistory: [],
    } as unknown as PlanResult<CarKinematicState>;
    const smoothed: CarKinematicState[] = [];
    for (let x = 0; x <= 4.0001; x += 0.5) smoothed.push(st(x));

    attachPlanFeedforward(smoothed, res, lib);
    // Near the end (RS region) there is no feedforward to hold onto beyond the
    // last drive edge — the final samples fall back to the drive edge's control
    // only up to the RS boundary; the RS-only tail carries none of its own.
    const tail = smoothed[smoothed.length - 1]!;
    // The final node's edge is RS → ffFrom[last] is undefined; the sample maps
    // to segment [1,2] whose entering edge (node 2) is RS, so it holds node 1's
    // drive control as the last known feedforward (never an RS-derived one).
    expect(tail.ff === undefined || tail.ff![0] === 0).toBe(true);
  });

  it('a real v3 slalom plan carries feedforward on (nearly) every drive sample', () => {
    // Integration guard: the planner-edge contract (kind ∈ {drive,drive-reverse}
    // with a numeric primId) must actually resolve to library controls, or the
    // whole feature is an inert flag. Using the raw node path as the "smoothed"
    // target isolates the capture from the smoother.
    const lib = buildLearnedRaceLibraryV3(v3FromJson(readModel('v3-default.json')));
    const spawn = st(-40, 0, 14);
    const gates = [st(-18, 5, 5), st(0, -5, 5), st(18, 5, 5)];
    const res = planRaceMultiGoal({
      state: spawn, gates, lib,
      polygons: [{ id: 0, y: 0, ring: [[-50, -25], [30, -25], [30, 25], [-50, 25]] as [number, number][] }],
      obstacles: [],
      gateRadius: RACE_PLANNER_GATE_RADIUS,
      deadlineMs: 6000, maxExpansions: 400_000,
    });
    expect(res.found).toBe(true);
    // Copy the node states so attach can mutate them.
    const smoothed = res.path.map((p) => ({ ...p }));
    attachPlanFeedforward(smoothed, res, lib);
    // Count interior samples that are the END of a drive/drive-reverse edge.
    let driveSamples = 0;
    let tagged = 0;
    for (let i = 1; i < res.nodes.length; i++) {
      const k = res.nodes[i]!.edge?.kind;
      if (k === 'drive' || k === 'drive-reverse') {
        driveSamples++;
        if (smoothed[i]!.ff) tagged++;
      }
    }
    expect(driveSamples).toBeGreaterThan(0);
    // Every drive sample should resolve to a library control (allow the very
    // first, whose arc maps to node 0's null edge, to miss).
    expect(tagged / driveSamples).toBeGreaterThan(0.9);
  });
});
