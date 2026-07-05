// `pnpm run controller-bench` — the chassis calibration & regression
// suite.
//
// Runs the controller through every canonical scenario the kinocat
// stack already maintains (race + parking — extend over time with
// carchase, obstaclecourse, ramp, ...) and reports a pass/fail table.
//
// The architecture the user asked for: scenarios are infrastructure,
// the bench is infrastructure, the controller is a tunable subject of
// the bench, the weights are chosen ONCE per chassis against the
// whole bench. After this CLI passes, the same MPPI config drives
// any plan, any environment, any goal — racing, parking, recovery,
// reverse maneuvers — because each behaviour-mode shows up as a
// scenario in the bench and the weights are picked to clear them all.
//
// CLI:
//   pnpm run controller-bench                 # run the full suite
//   pnpm run controller-bench --tracker=mpc   # MPPI tracker
//   pnpm run controller-bench --filter=parking
//   pnpm run controller-bench --json=out.json # machine-readable

import { readFileSync, existsSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, isAbsolute, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import {
  createRaceScenario,
  type RaceTuning,
  type RaceEntry,
} from '../app/lib/race-scenario';
import {
  kinematicEntry,
  v2Entry,
  parametricOnlyEntry,
} from '../app/lib/headless-race';
import {
  parkingLibrary,
  parkingCourse,
  parkingScenarioOptions,
  buildParkingScenario,
  evaluateParked,
  PARKING_SETTLE,
  PARKING_BUDGETS,
  type ParkingScenarioId,
} from '../app/lib/parking-scenarios';
import { createSettleLatch } from 'kinocat/execute';
import {
  createObstacleCourseScenario,
  PHYSICS_DT as PHYSICS_DT_OBS,
} from '../app/lib/obstaclecourse-scenario';
import { OBS_AGENT, OBS_BLOCKS_ALL, buildObstacleCourse } from '../app/lib/obstaclecourse-scenarios';
import { createRampScenario, PHYSICS_DT as PHYSICS_DT_RAMP } from '../app/lib/ramp-scenario';
import { RAMP_AGENT } from '../app/lib/ramp-scenarios';
import { createSimMonitor } from '../app/lib/sim-monitor';
import { modelFromJson } from '../app/lib/v2-model-file';
import type { PersistedV2Model } from '../app/lib/v2-model-persistence';
import {
  checkFeasibility,
  toReferenceTrajectory,
  referenceLength,
  limitsFromAgent,
  type DynamicLimits,
} from 'kinocat/eval';
import { smoothTrajectory } from 'kinocat/execute';
import type { CarKinematicState } from 'kinocat/agent';
import { TRACKER_MAX_LATERAL_ACCEL } from '../app/lib/race-scenario';
import { RACE_AGENT } from '../app/lib/race-primitives-scenarios';

type EntryKind = 'kinematic' | 'parametric-only' | 'v2-default';

// Plan-feasibility limits for the RACE_AGENT-planned bench scenarios (race +
// parking both plan with RACE_AGENT). Lets the bench measure the PLANNER half of
// the decomposition — whether the committed plan was dynamically executable —
// via the same kinocat/eval check the harness and tests use, instead of only
// measuring terminal outcome + safety.
const BENCH_PLAN_LIMITS: DynamicLimits = limitsFromAgent(RACE_AGENT, {
  frictionLimit: TRACKER_MAX_LATERAL_ACCEL,
  maxAccel: 6,
  maxDecel: 8,
});

/** Accumulate plan feasibility over a run by scoring each newly-committed plan
 *  with kinocat/eval. Guards short / near-degenerate stubs exactly like
 *  eval-harness (≥10 samples, smoothed ≥6, arc-length ≥5 m) so their meaningless
 *  curvature can't skew the fraction. */
function makePlanFeasibilityMeter(limits: DynamicLimits) {
  let lastRef: readonly CarKinematicState[] | null = null;
  let scored = 0;
  let feasible = 0;
  let worst = 0;
  return {
    observe(plan: readonly CarKinematicState[] | null): void {
      if (!plan || plan === lastRef || plan.length < 10) return;
      lastRef = plan;
      const tracked = smoothTrajectory(plan as CarKinematicState[], {
        sampleSpacing: 0.5,
        iterations: 8,
      });
      if (tracked.length < 6) return;
      const ref = toReferenceTrajectory(tracked);
      if (referenceLength(ref) < 5) return;
      const rep = checkFeasibility(ref, limits);
      scored++;
      if (rep.feasible) feasible++;
      if (rep.worstRatio > worst) worst = rep.worstRatio;
    },
    result() {
      return {
        plansScored: scored,
        planFeasibleFrac: scored > 0 ? feasible / scored : 1,
        planWorstRatio: worst,
      };
    },
  };
}

function loadEntry(kind: EntryKind, forScenario: 'race' | 'parking'): RaceEntry {
  if (forScenario === 'parking') {
    // Parking uses the parking-specific primitive library regardless
    // of which planner-side variant we're benching — the parking
    // library is hand-tuned for slow-maneuver clearance and isn't
    // the variable under test here. (A future extension would
    // produce v2-derived parking primitives.)
    return { name: kind, lib: parkingLibrary() };
  }
  // Race scenario varies the library based on the entry kind.
  switch (kind) {
    case 'kinematic':
      return kinematicEntry('kinematic');
    case 'parametric-only':
      return parametricOnlyEntry('parametric-only');
    case 'v2-default': {
      const modelPath = resolve(
        dirname(fileURLToPath(import.meta.url)),
        '..',
        'public',
        'models',
        'v2-default.json',
      );
      if (!existsSync(modelPath)) {
        throw new Error(
          `v2-default.json not found at ${modelPath}. Run \`pnpm run train:quick\` first.`,
        );
      }
      const payload = JSON.parse(readFileSync(modelPath, 'utf-8')) as PersistedV2Model;
      return v2Entry('v2-default', modelFromJson(payload));
    }
  }
}

// `parkingCourse` + `PARKING_RACE_TUNING` are imported from
// `parking-scenarios.ts` — the single source of truth shared with the web page
// and the Vitest invariant tests.

// ---------------------------------------------------------------------------
// Scenario definitions. Each scenario is a closure that returns a
// runner + a success predicate + a metrics formatter. The runner is
// just a thin wrapper around `createRaceScenario` with an appropriate
// course; the same MPPI controller drives every scenario.

interface BenchScenario {
  name: string;
  description: string;
  run: (tuning: Partial<RaceTuning>, entryKind: EntryKind) => Promise<BenchResult>;
}

interface BenchResult {
  scenario: string;
  passed: boolean;
  simTime: number;
  /** Terminal pose error to the scenario's goal pose (m). */
  terminalErrorM: number;
  /** Terminal heading error to the goal heading (rad). */
  terminalHeadingErr: number;
  /** Terminal chassis speed magnitude (m/s) — important for parking. */
  terminalSpeed: number;
  /** Off-track / out-of-bounds events. */
  offTrackEvents: number;
  /** Total planner replans. */
  totalReplans: number;
  /** Safety-invariant summary from the telemetry monitor (when collected). */
  collided?: boolean;
  minClearance?: number;
  maxJerk?: number;
  /** Teleports observed — must be 0 (we removed all teleportation). */
  teleports?: number;
  /** Fraction of committed plans that were dynamically feasible (kinocat/eval).
   *  < 1 attributes trouble to the PLANNER (an inexecutable plan), independent
   *  of how well the controller tracked it. */
  planFeasibleFrac?: number;
  /** How many committed plans were scored for `planFeasibleFrac`. */
  plansScored?: number;
  /** Worst demand/limit ratio across all scored plans (≤ 1 ⇒ all feasible). */
  planWorstRatio?: number;
  /** Scenario-specific notes. */
  note: string;
}

/** Race scenario: lap the existing /raceprimitives course once.
 *  PASS criteria: complete 1 lap in ≤ 90 s sim, zero off-tracks. */
const raceScenario: BenchScenario = {
  name: 'race',
  description: '1 lap of the /raceprimitives slalom + corners course',
  async run(tuning, entryKind) {
    const MAX_SIM = 120;
    const scenario = await createRaceScenario({
      entries: [loadEntry(entryKind, 'race')],
      targetLaps: 1,
      syncHold: false,
      offTrackRecovery: 'spawn',
      tuning,
    });
    const meter = makePlanFeasibilityMeter(BENCH_PLAN_LIMITS);
    while (scenario.simTime() < MAX_SIM) {
      const r = scenario.tick();
      meter.observe(scenario.status()[0]!.plan);
      if (r.allFinished) break;
    }
    const status = scenario.status()[0]!;
    const simTime = scenario.simTime();
    scenario.dispose();
    const feas = meter.result();
    const completed = status.laps.length >= 1;
    const passed = completed && simTime <= 90 && status.offTrackEvents === 0;
    return {
      scenario: 'race',
      passed,
      simTime,
      terminalErrorM: 0,
      terminalHeadingErr: 0,
      terminalSpeed: Math.abs(status.state.speed),
      offTrackEvents: status.offTrackEvents,
      totalReplans: status.diagnostics.totalReplans,
      ...feas,
      note: completed
        ? `${status.laps[0]!.duration.toFixed(1)}s lap · plan ${(feas.planFeasibleFrac * 100).toFixed(0)}% feasible`
        : 'DNF',
    };
  },
};

/** Parking scenario factory — uses the canonical `parking-scenarios.ts`
 *  layouts (forward-pullin, reverse-perp, parallel) ported from the WIP
 *  parking branch. PASS criteria: terminal position within 1.5 m of
 *  goal pose AND chassis stopped (|v|<1 m/s) within the per-scenario
 *  sim budget. */
function makeParkingScenario(
  id: ParkingScenarioId,
  maxSim: number,
  label: string,
): BenchScenario {
  return {
    name: `parking-${id}`,
    description: label,
    async run(tuning, entryKind) {
      const course = parkingCourse(id);
      const goal = course.waypoints[course.waypoints.length - 1]!;
      const parkScenario = buildParkingScenario(id);
      // The complete, canonical parking options (incl. zero teleportation) —
      // the SAME definition the web page and the Vitest tests use. `tuning`
      // here carries only the tracker selection.
      const scenario = await createRaceScenario(
        parkingScenarioOptions(id, [loadEntry(entryKind, 'parking')], tuning),
      );
      // Closed-loop success: the shared `evaluateParked` predicate must hold
      // CONTINUOUSLY at rest (settle latch), then the sim keeps running a
      // post-hold window so creep-out / replan shuffling AFTER "success" fails
      // the bench. The old break-on-first-`parked` accepted a transient
      // snapshot the runner destroyed ~300 ms later — the web page then showed
      // 100+ s of livelock the bench never saw. Budgets (time-to-settled,
      // replans) come from the SAME `PARKING_BUDGETS` table the Vitest
      // invariants assert, so the bar cannot drift per-harness.
      const meter = makePlanFeasibilityMeter(BENCH_PLAN_LIMITS);
      const latch = createSettleLatch({
        holdSeconds: PARKING_SETTLE.holdSeconds,
        speedTol: PARKING_SETTLE.speedTol,
      });
      const budget = PARKING_BUDGETS[id];
      const postTickBudget = Math.round(PARKING_SETTLE.postHoldSeconds * 60);
      let postTicks = 0;
      let replansAtSettle = 0;
      while (scenario.simTime() < maxSim) {
        scenario.tick();
        const status = scenario.status()[0]!;
        meter.observe(status.plan);
        const wasSettled = latch.state.settled;
        latch.update(
          { ok: evaluateParked(status.state, parkScenario).parked, speed: status.state.speed },
          1 / 60,
        );
        if (latch.state.settled && !wasSettled) {
          replansAtSettle = status.diagnostics.totalReplans;
        }
        if (latch.state.settled && ++postTicks >= postTickBudget) break;
      }
      const status = scenario.status()[0]!;
      const simTime = scenario.simTime();
      scenario.dispose();
      const feas = meter.result();
      const dist = Math.hypot(status.state.x - goal.x, status.state.z - goal.z);
      const ev = evaluateParked(status.state, parkScenario);
      const settle = latch.state;
      const withinBudget =
        settle.settled &&
        settle.timeToSettled! < budget.maxTimeToSettledSec &&
        replansAtSettle <= budget.maxReplans;
      const passed = withinBudget && settle.violations === 0 && ev.parked;
      const settleNote = settle.settled
        ? `settled ${settle.timeToSettled!.toFixed(1)}s/${budget.maxTimeToSettledSec}s · replans ${replansAtSettle}/${budget.maxReplans}${settle.violations > 0 ? ` · ${settle.violations} POST-SETTLE VIOLATIONS` : ''}`
        : `NEVER SETTLED (transient parked ≠ success)`;
      return {
        scenario: `parking-${id}`,
        passed,
        simTime,
        terminalErrorM: dist,
        terminalHeadingErr: ev.headingError,
        terminalSpeed: Math.abs(status.state.speed),
        offTrackEvents: status.offTrackEvents,
        totalReplans: status.diagnostics.totalReplans,
        ...feas,
        note: `${passed ? `parked (${(ev.coverage * 100).toFixed(0)}% in stall)` : `${(ev.coverage * 100).toFixed(0)}% in stall, ${((ev.headingError * 180) / Math.PI).toFixed(0)}° off`} · ${settleNote}`,
      };
    },
  };
}

/** Obstacle-course scenario: drive the waypoint loop on the heightfield course
 *  past the buildings. PASS: progresses around the loop, never physically hits
 *  a building, no teleport. Uses the SAME createObstacleCourseScenario the web
 *  page drives. */
const obstacleCourseScenario: BenchScenario = {
  name: 'obstaclecourse',
  description: 'waypoint loop over heightfield terrain + buildings',
  async run() {
    const MAX_TICKS = 480; // 8 s sim
    const scenario = await createObstacleCourseScenario();
    const course = buildObstacleCourse(OBS_BLOCKS_ALL);
    const monitor = createSimMonitor({
      footprint: OBS_AGENT.footprint,
      obstacles: course.buildings.map((b) => [
        [b.x - b.hx, b.z - b.hz],
        [b.x + b.hx, b.z - b.hz],
        [b.x + b.hx, b.z + b.hz],
        [b.x - b.hx, b.z + b.hz],
      ]),
      dt: PHYSICS_DT_OBS,
    });
    let maxLoopIndex = 0;
    for (let i = 0; i < MAX_TICKS; i++) {
      scenario.tick();
      const st = scenario.status();
      monitor.sample(st);
      maxLoopIndex = Math.max(maxLoopIndex, st.loopIndex);
    }
    const r = monitor.summary();
    scenario.dispose();
    const passed = maxLoopIndex > 0 && !r.collided && r.teleports === 0;
    return {
      scenario: 'obstaclecourse',
      passed,
      simTime: r.durationSec,
      terminalErrorM: 0,
      terminalHeadingErr: 0,
      terminalSpeed: r.terminalSpeed,
      offTrackEvents: 0,
      totalReplans: r.totalReplans,
      collided: r.collided,
      minClearance: r.minClearance,
      maxJerk: r.maxJerk,
      teleports: r.teleports,
      note: `wp=${maxLoopIndex} clear=${fmt(r.minClearance)}m`,
    };
  },
};

/** Ramp scenario: drive over the heightfield ramp to the goal, taking the jump
 *  affordance. PASS: reaches the goal, no teleport. Uses the SAME
 *  createRampScenario the web page drives. */
const rampScenario: BenchScenario = {
  name: 'ramp',
  description: 'drive over the heightfield ramp to the goal (jump affordance)',
  async run() {
    const MAX_TICKS = 900; // 15 s sim
    const scenario = await createRampScenario({ affordance: true });
    const goal = scenario.status().goal;
    const monitor = createSimMonitor({
      footprint: RAMP_AGENT.footprint,
      obstacles: [],
      dt: PHYSICS_DT_RAMP,
      goal: { x: goal.x, z: goal.z, heading: goal.heading },
      success: { posTol: 3, headingTol: Math.PI, speedTol: 100 },
    });
    let reached = false;
    let usedAffordance = false;
    for (let i = 0; i < MAX_TICKS; i++) {
      scenario.tick();
      const st = scenario.status();
      monitor.sample(st);
      if (st.diagnostics.usedAffordance) usedAffordance = true;
      if (Math.hypot(st.state.x - goal.x, st.state.z - goal.z) < 3) {
        reached = true;
        break;
      }
    }
    const r = monitor.summary();
    scenario.dispose();
    const passed = reached && r.teleports === 0;
    return {
      scenario: 'ramp',
      passed,
      simTime: r.durationSec,
      terminalErrorM: r.terminalPosError,
      terminalHeadingErr: 0,
      terminalSpeed: r.terminalSpeed,
      offTrackEvents: 0,
      totalReplans: r.totalReplans,
      collided: r.collided,
      minClearance: r.minClearance,
      maxJerk: r.maxJerk,
      teleports: r.teleports,
      note: reached ? `reached${usedAffordance ? ' (jumped)' : ''}` : 'did not reach goal',
    };
  },
};

const ALL_SCENARIOS: BenchScenario[] = [
  raceScenario,
  makeParkingScenario('forward-pullin', 30, 'forward pull-in (easy)'),
  makeParkingScenario('reverse-perp', 60, 'reverse perpendicular (medium)'),
  makeParkingScenario('parallel', 60, 'parallel parking (hard)'),
  obstacleCourseScenario,
  rampScenario,
];

// ---------------------------------------------------------------------------
// CLI plumbing.

function fmt(n: number, digits = 2): string {
  return Number.isFinite(n) ? n.toFixed(digits) : '---';
}

async function main(): Promise<void> {
  const { values } = parseArgs({
    options: {
      tracker: { type: 'string', default: 'pure-pursuit' },
      entry: { type: 'string', default: 'kinematic' },
      filter: { type: 'string' },
      json: { type: 'string' },
      help: { type: 'boolean', short: 'h' },
    },
  });
  if (values.help) {
    process.stdout.write(`Usage: pnpm run controller-bench [--tracker=pure-pursuit|mpc]
                              [--filter=name,name,...]
                              [--json=out.json]
\n`);
    return;
  }
  const tracker = values.tracker as 'pure-pursuit' | 'mpc';
  const entryKind = values.entry as EntryKind;
  if (!['kinematic', 'parametric-only', 'v2-default'].includes(entryKind)) {
    throw new Error(`Invalid --entry=${entryKind}. Use kinematic | parametric-only | v2-default.`);
  }
  const tuning: Partial<RaceTuning> = { tracker };
  const filter = values.filter
    ? new Set(values.filter.split(',').map((s) => s.trim()))
    : null;
  const scenarios = filter
    ? ALL_SCENARIOS.filter((s) => filter.has(s.name))
    : ALL_SCENARIOS;

  process.stdout.write(`controller bench · tracker=${tracker} · entry=${entryKind} · ${scenarios.length} scenarios\n\n`);

  const results: BenchResult[] = [];
  for (const s of scenarios) {
    process.stdout.write(`▶ ${s.name}: ${s.description} ...`);
    const start = performance.now();
    const result = await s.run(tuning, entryKind);
    const wall = ((performance.now() - start) / 1000).toFixed(1);
    process.stdout.write(`  [${result.passed ? 'PASS' : 'FAIL'}] ${wall}s wall, ${result.note}\n`);
    results.push(result);
  }

  process.stdout.write('\n');
  const headers = ['scenario', 'pass', 'sim(s)', 'goalErr(m)', 'hdgErr', '|v|', 'collide', 'teleport', 'note'];
  const widths = [22, 6, 8, 11, 8, 7, 8, 9, 30];
  const sep = '─'.repeat(widths.reduce((a, b) => a + b + 2, 0));
  process.stdout.write(headers.map((h, i) => h.padEnd(widths[i] ?? 0)).join('  ') + '\n' + sep + '\n');
  for (const r of results) {
    const row = [
      r.scenario,
      r.passed ? 'PASS' : 'FAIL',
      fmt(r.simTime),
      fmt(r.terminalErrorM),
      fmt(r.terminalHeadingErr),
      fmt(r.terminalSpeed),
      r.collided === undefined ? '—' : r.collided ? 'HIT' : 'no',
      r.teleports === undefined ? '—' : String(r.teleports),
      r.note,
    ];
    process.stdout.write(row.map((c, i) => String(c).padEnd(widths[i] ?? 0)).join('  ') + '\n');
  }
  process.stdout.write(sep + '\n');

  const passed = results.filter((r) => r.passed).length;
  const total = results.length;
  process.stdout.write(`\n${passed}/${total} scenarios passed\n`);

  if (values.json) {
    const __dirname = dirname(fileURLToPath(import.meta.url));
    const out = isAbsolute(values.json) ? values.json : resolve(__dirname, '..', values.json);
    mkdirSync(dirname(out), { recursive: true });
    writeFileSync(out, JSON.stringify({ tracker, results }, null, 2));
    process.stdout.write(`wrote ${out}\n`);
  }

  if (passed < total) process.exit(1);
}

main().catch((e) => {
  process.stderr.write(`controller-bench failed: ${e instanceof Error ? e.stack ?? e.message : String(e)}\n`);
  process.exit(1);
});

void existsSync; // imports kept for future ledger-append support
