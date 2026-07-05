// The "gauntlet" — a constraint-satisfaction / precision-driving scenario
// (a different axis from the open-space quality tests). Two open lots joined by
// a narrow dog-leg corridor whose width is the vehicle footprint plus a tunable
// MARGIN. The route is obvious; the question is whether the controller can
// physically thread it without clipping the walls.
//
// The crucial element is the PASSABILITY ORACLE: the corridor is built from the
// footprint, the ideal centerline is generated, and we assert that the perfect
// driver (footprint swept along the centerline) clears the walls AND that the
// centerline is dynamically feasible. If the oracle clears it, the task is
// PROVABLY solvable — so any real failure is unambiguously the controller's
// fault (it ate the margin), not an impossible setup.
//
// Sweeping `margin` down to negative surfaces the boundaries the guide (§7)
// cares about: where the task becomes physically impossible (oracle clips) vs.
// where the controller starts clipping a gap the oracle cleared.

import type { CarKinematicState } from '../agent/types';
import type { ForwardSim } from '../primitives/types';
import { placeFootprint, segmentsIntersect, pointSegmentDistance, type Pt } from '../internal/geom';
import { toReferenceTrajectory } from './reference-trajectory';
import { projectOntoPath } from './projection';
import { checkFeasibility, type DynamicLimits, type FeasibilityReport } from './feasibility';
import type { RefController } from './tracking-metrics';

export interface CorridorWorld {
  start: CarKinematicState;
  goal: { x: number; z: number; heading: number };
  /** The ideal route through the corridor (the "obvious" path). */
  centerline: CarKinematicState[];
  /** Corridor wall polylines (world XZ) — used only for clearance measurement;
   *  the controller tracks the centerline and never sees them. */
  leftWall: Pt[];
  rightWall: Pt[];
  /** Corridor width (m) = footprint width + margin. */
  width: number;
  /** Total lateral slack (m) = width − footprint width. Negative ⇒ impossible. */
  margin: number;
  /** Arc-length span [start, end] of the tight dog-leg section. */
  corridor: { sStart: number; sEnd: number };
}

export interface BuildCorridorOptions {
  /** Body-local footprint polygon (heading 0 = +x). */
  footprint: ReadonlyArray<Pt>;
  /** Total lateral slack (m). Corridor width = footprint width + margin. */
  margin: number;
  /** Lateral offset between the two lots (m) — the dog-leg jog. */
  offset?: number;
  /** Longitudinal length of the dog-leg section (m). */
  corridorLength?: number;
  /** Length of the open approach/exit in each lot (m). */
  lotLength?: number;
  /** Constant traversal speed (m/s). */
  speed: number;
  /** Centerline sample spacing (m). */
  ds?: number;
}

/** Smoothstep 3u²−2u³ on [0,1]. */
function smoothstep(u: number): number {
  const c = Math.min(1, Math.max(0, u));
  return c * c * (3 - 2 * c);
}

function footprintHalfWidth(footprint: ReadonlyArray<Pt>): number {
  return Math.max(...footprint.map((p) => Math.abs(p[1])));
}

/** Build a dog-leg corridor between two laterally-offset open lots, sized to the
 *  footprint plus `margin`. */
export function buildDogLegCorridor(opts: BuildCorridorOptions): CorridorWorld {
  const offset = opts.offset ?? 4;
  const corridorLength = opts.corridorLength ?? 16;
  const lotLength = opts.lotLength ?? 8;
  const ds = opts.ds ?? 0.4;
  const hw = footprintHalfWidth(opts.footprint);
  const width = 2 * hw + opts.margin;

  const x0 = -lotLength - corridorLength / 2;
  const xCorrStart = -corridorLength / 2;
  const xCorrEnd = corridorLength / 2;
  const x1 = lotLength + corridorLength / 2;

  // Centerline z(x): 0 in the entry lot, smoothstep jog of `offset` through the
  // corridor, constant `offset` in the exit lot.
  const zAt = (x: number): number => {
    if (x <= xCorrStart) return 0;
    if (x >= xCorrEnd) return offset;
    return offset * smoothstep((x - xCorrStart) / corridorLength);
  };

  const totalX = x1 - x0;
  const n = Math.max(5, Math.round(totalX / ds) + 1);
  const pts: { x: number; z: number }[] = Array.from({ length: n }, (_, i) => {
    const x = x0 + (i / (n - 1)) * totalX;
    return { x, z: zAt(x) };
  });

  // Build states with tangent headings, constant speed, accumulated time.
  const centerline: CarKinematicState[] = new Array(n);
  let t = 0;
  for (let i = 0; i < n; i++) {
    const a = pts[Math.max(0, i - 1)]!;
    const b = pts[Math.min(n - 1, i + 1)]!;
    const heading = Math.atan2(b.z - a.z, b.x - a.x);
    if (i > 0) {
      const prev = pts[i - 1]!;
      t += Math.hypot(pts[i]!.x - prev.x, pts[i]!.z - prev.z) / Math.max(opts.speed, 1e-3);
    }
    centerline[i] = { x: pts[i]!.x, z: pts[i]!.z, heading, speed: opts.speed, t };
  }

  // Walls = centerline offset by ±width/2 along the local normal, but only over
  // the corridor section (+ a 1 m lead-in/out so the mouths are defined). The
  // lots stay open.
  const leftWall: Pt[] = [];
  const rightWall: Pt[] = [];
  for (let i = 0; i < n; i++) {
    const s = centerline[i]!;
    if (s.x < xCorrStart - 1 || s.x > xCorrEnd + 1) continue;
    const nx = -Math.sin(s.heading);
    const nz = Math.cos(s.heading);
    leftWall.push([s.x + (width / 2) * nx, s.z + (width / 2) * nz]);
    rightWall.push([s.x - (width / 2) * nx, s.z - (width / 2) * nz]);
  }

  // Arc-length span of the dog-leg section.
  const ref = toReferenceTrajectory(centerline);
  const sStart = projectOntoPath(ref, xCorrStart, 0).s;
  const sEnd = projectOntoPath(ref, xCorrEnd, offset).s;

  return {
    start: { ...centerline[0]! },
    goal: { x: centerline[n - 1]!.x, z: centerline[n - 1]!.z, heading: centerline[n - 1]!.heading },
    centerline,
    leftWall,
    rightWall,
    width,
    margin: opts.margin,
    corridor: { sStart, sEnd },
  };
}

/** Minimum gap (m) from a placed footprint polygon to a wall polyline, and
 *  whether they intersect. */
function footprintToWall(
  fp: ReadonlyArray<Pt>,
  wall: ReadonlyArray<Pt>,
): { dist: number; hit: boolean } {
  let dist = Infinity;
  for (let i = 0; i < fp.length; i++) {
    const a = fp[i]!;
    const b = fp[(i + 1) % fp.length]!;
    for (let j = 0; j < wall.length - 1; j++) {
      const c = wall[j]!;
      const d = wall[j + 1]!;
      if (segmentsIntersect(a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1])) {
        return { dist: 0, hit: true };
      }
      // Vertex-to-segment both ways (exact for the common closest-vertex case).
      const e1 = pointSegmentDistance(a[0], a[1], c[0], c[1], d[0], d[1]);
      const e2 = pointSegmentDistance(c[0], c[1], a[0], a[1], b[0], b[1]);
      if (e1 < dist) dist = e1;
      if (e2 < dist) dist = e2;
    }
  }
  return { dist, hit: false };
}

/** Sweep the footprint along a trajectory and return the worst (min) clearance
 *  to the corridor walls and whether it ever collided. */
export function sweptClearance(
  trajectory: ReadonlyArray<CarKinematicState>,
  footprint: ReadonlyArray<Pt>,
  world: CorridorWorld,
): { minClearance: number; collided: boolean } {
  let minClearance = Infinity;
  let collided = false;
  for (const st of trajectory) {
    const fp = placeFootprint(footprint, st.x, st.z, st.heading);
    const l = footprintToWall(fp, world.leftWall);
    const r = footprintToWall(fp, world.rightWall);
    if (l.hit || r.hit) {
      collided = true;
      minClearance = 0;
      continue;
    }
    minClearance = Math.min(minClearance, l.dist, r.dist);
  }
  return { minClearance, collided };
}

export interface PassabilityReport {
  /** True iff the perfect driver clears the walls AND the line is feasible. */
  passable: boolean;
  /** Min clearance of the ideal swept footprint to the walls (m). */
  oracleMinClearance: number;
  feasible: boolean;
  feasibility: FeasibilityReport;
}

/** The passability oracle: does the ideal centerline clear the walls and respect
 *  the car's dynamic limits? Guarantees the task is solvable before we blame the
 *  controller for failing it. */
export function assessPassability(
  world: CorridorWorld,
  footprint: ReadonlyArray<Pt>,
  limits: DynamicLimits,
): PassabilityReport {
  const swept = sweptClearance(world.centerline, footprint, world);
  const feasibility = checkFeasibility(toReferenceTrajectory(world.centerline), limits);
  return {
    passable: swept.minClearance > 0 && !swept.collided && feasibility.feasible,
    oracleMinClearance: swept.minClearance,
    feasible: feasibility.feasible,
    feasibility,
  };
}

export interface GauntletReport {
  passability: PassabilityReport;
  reachedGoal: boolean;
  collided: boolean;
  executedMinClearance: number;
  /** Cross-track error WITHIN the tight corridor section (m). */
  corridorCrossTrackRmse: number;
  corridorCrossTrackMax: number;
  /** Fraction of the oracle's available clearance the controller consumed
   *  (1 = shaved to the wall, >1 would mean it clipped). */
  clearanceUtilization: number;
  /** Gated score ∈ [0,1]: 0 on collision or not reaching the goal, otherwise
   *  the remaining clearance fraction (1 − utilization). */
  gatedScore: number;
}

export interface RunGauntletOptions {
  dt: number;
  limits: DynamicLimits;
  maxSteps?: number;
  goalTolerance?: number;
}

/** Run the controller through a (provably-passable) corridor and score how
 *  precisely it threaded it. The controller tracks the centerline — the planner
 *  half (finding the route) is out of scope here; this isolates execution. */
export function runGauntlet(
  world: CorridorWorld,
  footprint: ReadonlyArray<Pt>,
  controller: RefController,
  sim: ForwardSim<CarKinematicState>,
  opts: RunGauntletOptions,
): GauntletReport {
  const passability = assessPassability(world, footprint, opts.limits);
  const ref = toReferenceTrajectory(world.centerline);
  const goalTol = opts.goalTolerance ?? 1.5;
  const maxSteps = opts.maxSteps ?? Math.ceil(world.centerline.length * 4 + 400);

  let state: CarKinematicState = { ...world.start };
  const executed: CarKinematicState[] = [{ ...state }];
  let reachedGoal = false;
  for (let step = 0; step < maxSteps; step++) {
    const cmd = controller(state, world.centerline);
    if (cmd.atGoal || Math.hypot(state.x - world.goal.x, state.z - world.goal.z) < goalTol) {
      reachedGoal = true;
      break;
    }
    state = sim(state, cmd.controls, opts.dt);
    executed.push({ ...state });
  }
  if (Math.hypot(state.x - world.goal.x, state.z - world.goal.z) < goalTol) reachedGoal = true;

  const swept = sweptClearance(executed, footprint, world);

  // Cross-track within the tight corridor section only.
  let ctSumSq = 0;
  let ctMax = 0;
  let count = 0;
  for (const st of executed) {
    const proj = projectOntoPath(ref, st.x, st.z);
    if (proj.s >= world.corridor.sStart && proj.s <= world.corridor.sEnd) {
      const ct = Math.abs(proj.crossTrack);
      ctSumSq += ct * ct;
      if (ct > ctMax) ctMax = ct;
      count++;
    }
  }
  const corridorCrossTrackRmse = count > 0 ? Math.sqrt(ctSumSq / count) : 0;

  const oracleClear = Math.max(passability.oracleMinClearance, 1e-6);
  const clearanceUtilization = swept.collided
    ? Infinity
    : (oracleClear - swept.minClearance) / oracleClear;

  const gatedScore =
    !reachedGoal || swept.collided
      ? 0
      : Math.max(0, Math.min(1, swept.minClearance / oracleClear));

  return {
    passability,
    reachedGoal,
    collided: swept.collided,
    executedMinClearance: swept.minClearance,
    corridorCrossTrackRmse,
    corridorCrossTrackMax: ctMax,
    clearanceUtilization,
    gatedScore,
  };
}
