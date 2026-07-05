// Shared fixture for the replan-after-rebuild latency gate (bench + test).
// Exercises the full production pipeline as plain function calls — the
// worker message-hop overhead is negligible against the 100 ms budget:
//
//   world-update (setObstacles: AABBs + spatial index + heuristic grid
//   rebuild) → region-scoped invalidation (markAffectedAgents) → dirty
//   replan through the worker's handlePlanMessage → ReplanState.consider.
//
// Two blocker variants alternate so EVERY iteration's setObstacles is a real
// data change, and both sit on the committed path so the region-scoped
// detection always fires.

import {
  initWorkerContext,
  handlePlanMessage,
  handleWorldUpdateMessage,
  type ObstacleDescriptor,
} from 'kinocat/worker';
import { InMemoryNavWorld } from 'kinocat/environment';
import {
  ReplanState,
  planPoseAt,
  markAffectedAgents,
  footprintCircumradius,
  type ChangedRegion,
} from 'kinocat/execute';
import type { CarKinematicState } from 'kinocat/agent';
import {
  buildCarChaseCourse,
  CARCHASE_AGENT,
  CARCHASE_LIB,
  carChaseAffordances,
  spawnPoses,
  type CarChaseCourse,
} from '../../app/lib/carchase-scenarios';

type Pt = [number, number];

/** Kept inside the 100 ms gate: rebuild + detection + adoption need the rest. */
export const REPLAN_DEADLINE_MS = 80;

export interface ReplanRebuildFixture {
  course: CarChaseCourse;
  world: InMemoryNavWorld;
  robber: CarKinematicState;
  goal: CarKinematicState;
  copDescriptors: ObstacleDescriptor[];
  basePlan: CarKinematicState[];
  blockA: Pt[];
  blockB: Pt[];
  region: ChangedRegion;
  replan: ReplanState;
  inflate: number;
}

export function setupReplanRebuildFixture(): ReplanRebuildFixture {
  const course = buildCarChaseCourse();
  const world = new InMemoryNavWorld(course.polygons, course.obstacles);
  const affordances = carChaseAffordances(course);
  initWorkerContext({ world, agent: CARCHASE_AGENT, lib: CARCHASE_LIB, affordances });

  const { robber, cops } = spawnPoses();
  // The robber spawns on its loop — plan to the first waypoint that is
  // actually far enough away to produce a multi-segment committed path.
  const wp =
    course.robberLoop.find(
      (w) => Math.hypot(w.x - robber.x, w.z - robber.z) > 20,
    ) ?? course.robberLoop[1]!;
  const goal: CarKinematicState = {
    x: wp.x,
    z: wp.z,
    heading: wp.heading,
    speed: CARCHASE_AGENT.maxSpeed,
    t: 0,
  };
  const copDescriptors: ObstacleDescriptor[] = cops.map((c) => ({
    kind: 'cv',
    state: c,
    horizon: 4,
    radius: 2.6,
  }));

  // Committed plan against the unmodified course.
  let basePlan: CarKinematicState[] = [];
  handlePlanMessage(
    {
      type: 'plan',
      reqId: 0,
      npcId: 'robber',
      start: robber,
      goal,
      obstacles: copDescriptors,
      deadlineMs: Infinity,
      maxExpansions: 25000,
    },
    (r) => {
      basePlan = r.path;
    },
  );
  if (basePlan.length < 2) throw new Error('fixture: no base plan found');

  // Blockers straddle the committed path's TIME midpoint (interpolated —
  // an analytic edge may carry only its two endpoint samples, and blocking
  // an endpoint would wall off the goal itself).
  const mid = planPoseAt(
    basePlan,
    (basePlan[0]!.t + basePlan[basePlan.length - 1]!.t) / 2,
  )!;
  // Blocker size ≥ the course's smallest obstacle dimension (7 m) — the
  // heuristic grid's adaptive cell size halves the smallest obstacle dim, so
  // a smaller blocker would shrink every cell and quadruple the grid /
  // goal-Dijkstra rebuild cost, which is not what this gate measures.
  // Offset +x so the box covers the committed path's centerline (detection
  // must fire) while leaving a drivable gap on the -x side (the replan swerve
  // is a local maneuver, not a full-city detour — this gate measures pipeline
  // latency, not worst-case search difficulty).
  const half = 4;
  const blockerAt = (cx: number, cz: number): Pt[] => [
    [cx - half, cz - half],
    [cx + half, cz - half],
    [cx + half, cz + half],
    [cx - half, cz + half],
  ];
  const blockA = blockerAt(mid.x + 2.5, mid.z);
  const blockB = blockerAt(mid.x + 3, mid.z);
  const region: ChangedRegion = {
    x0: mid.x + 2.5 - half - 0.5,
    z0: mid.z - half - 0.5,
    x1: mid.x + 3 + half + 0.5,
    z1: mid.z + half + 0.5,
  };

  const replan = new ReplanState({
    divergenceThresholdMeters: 2,
    refreshIntervalMs: 500,
  });
  const inflate = footprintCircumradius(CARCHASE_AGENT.footprint);
  return {
    course,
    world,
    robber,
    goal,
    copDescriptors,
    basePlan,
    blockA,
    blockB,
    region,
    replan,
    inflate,
  };
}

export interface ReplanRebuildResult {
  wallMs: number;
  found: boolean;
  adopted: boolean;
  markedCount: number;
}

/** One full rebuild→detect→replan→adopt round. `i` alternates the blocker. */
export function runReplanAfterRebuildOnce(
  fix: ReplanRebuildFixture,
  i: number,
): ReplanRebuildResult {
  // Recommit the base plan so the toggled blocker always invalidates it
  // (setPlan clears any dirty flag from the previous round).
  fix.replan.setPlan(fix.basePlan, 0);
  const obstacles = [
    ...fix.course.obstacles,
    i % 2 === 0 ? fix.blockA : fix.blockB,
  ];

  const t0 = performance.now();
  handleWorldUpdateMessage({ type: 'world-update', seq: i, obstacles }, () => {});
  const marked = markAffectedAgents(fix.region, [
    { replan: fix.replan, inflate: fix.inflate },
  ]);
  // Consume the dirty flag the way the game loop does.
  fix.replan.shouldReplan(fix.robber, 1);
  let found = false;
  let adopted = false;
  handlePlanMessage(
    {
      type: 'plan',
      reqId: i + 1,
      npcId: 'robber',
      start: fix.robber,
      goal: fix.goal,
      obstacles: fix.copDescriptors,
      deadlineMs: REPLAN_DEADLINE_MS,
      maxExpansions: 25000,
    },
    (r) => {
      found = r.found;
      adopted = fix.replan.consider(r.path, r.cost, 1);
    },
  );
  const wallMs = performance.now() - t0;
  return { wallMs, found, adopted, markedCount: marked.length };
}
