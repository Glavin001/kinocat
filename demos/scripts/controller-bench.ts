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
  buildParkingScenario,
  parkingLibrary,
  checkParkingGoal,
  PARKING_GOAL_TOL,
  type ParkingScenarioId,
} from '../app/lib/parking-scenarios';
import { buildRaceCourse } from '../app/lib/race-primitives-scenarios';
import { modelFromJson } from '../app/lib/v2-model-file';
import type { PersistedV2Model } from '../app/lib/v2-model-persistence';

type EntryKind = 'kinematic' | 'parametric-only' | 'v2-default';

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

/** Build a race-scenario-compatible course from a parking scenario. The
 *  parking scenario's single goal pose becomes the sole "waypoint",
 *  and its obstacles + bounds carry over directly. */
function parkingCourse(id: ParkingScenarioId): ReturnType<typeof buildRaceCourse> {
  const s = buildParkingScenario(id);
  return {
    bounds: { x0: s.bounds.x0, x1: s.bounds.x1, z0: s.bounds.z0, z1: s.bounds.z1 },
    polygons: s.polygons,
    obstacles: s.obstacles,
    waypoints: [{ ...s.goal, speed: 0, t: 0 }],
    spawn: { ...s.spawn, speed: 0, t: 0 },
  };
}

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
    while (scenario.simTime() < MAX_SIM) {
      const r = scenario.tick();
      if (r.allFinished) break;
    }
    const status = scenario.status()[0]!;
    const simTime = scenario.simTime();
    scenario.dispose();
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
      note: completed ? `${status.laps[0]!.duration.toFixed(1)}s lap` : 'DNF',
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
      const scenario = await createRaceScenario({
        entries: [loadEntry(entryKind, 'parking')],
        targetLaps: 1,
        syncHold: false,
        offTrackRecovery: 'none',
        // Parking-specific tuning: tracker-agnostic knobs (low cruise
        // speed, tight goal tolerance, sub-meter arrive radius) that
        // both pure-pursuit AND MPPI obey, PLUS MPC terminal-pose
        // weights for MPPI's parking mode. Race scenarios leave these
        // at defaults / 0 so the same controller code runs both.
        tuning: {
          ...tuning,
          cruiseSpeed: 2,
          goalTolerance: 0.4,
          arriveRadius: PARKING_GOAL_TOL.posM,
          // Sub-meter planner discretisation + terminal-heading
          // constraint — the parking branch's tuning, ported via
          // `RaceTuning` so pure-pursuit + MPPI use the same
          // planner-side knobs. Single-waypoint courses (parking)
          // auto-route through `planRace` (planVehicleOnce with
          // heading constraint).
          plannerPosCell: 0.3,
          plannerHeadingBuckets: 36,
          plannerGoalRadius: 0.35,
          plannerGoalHeadingTol: 0.2,
          plannerBudgetMs: 500,
          plannerMaxExpansions: 80_000,
          mpcWTerminalPosition: 50,
          mpcWTerminalSpeed: 30,
        },
        course,
      });
      // Tick the scenario until either time runs out OR the shared
      // goal-check (same one the web demo uses for its `GOAL — ALL MET`
      // HUD badge) reports `passed`. Early-out so a quick park doesn't
      // sit in the loop for the full sim budget.
      while (scenario.simTime() < maxSim) {
        scenario.tick();
        const status = scenario.status()[0]!;
        if (checkParkingGoal(status.state, goal).passed) break;
      }
      const status = scenario.status()[0]!;
      const simTime = scenario.simTime();
      scenario.dispose();
      // Single source of truth for "did we park?" — exactly what the
      // browser HUD shows. If this fails but the position alone was
      // OK, that's the heading or speed component that needs work,
      // and the report's per-axis fields will say which.
      const check = checkParkingGoal(status.state, goal);
      const passed = check.passed && simTime < maxSim;
      const failPart = !check.posOk
        ? `pos off by ${check.posM.toFixed(2)}m`
        : !check.hdgOk
          ? `hdg off by ${(check.hdgRad * 180 / Math.PI).toFixed(1)}°`
          : !check.spdOk
            ? `not stopped (|v|=${check.spdMS.toFixed(2)})`
            : 'sim timeout';
      return {
        scenario: `parking-${id}`,
        passed,
        simTime,
        terminalErrorM: check.posM,
        terminalHeadingErr: check.hdgRad,
        terminalSpeed: check.spdMS,
        offTrackEvents: status.offTrackEvents,
        totalReplans: status.diagnostics.totalReplans,
        note: passed ? 'parked' : failPart,
      };
    },
  };
}

const ALL_SCENARIOS: BenchScenario[] = [
  raceScenario,
  makeParkingScenario('forward-pullin', 25, 'forward pull-in (easy)'),
  makeParkingScenario('reverse-perp', 40, 'reverse perpendicular (medium)'),
  makeParkingScenario('parallel', 40, 'parallel parking (hard)'),
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
  const headers = ['scenario', 'pass', 'sim(s)', 'goalErr(m)', 'hdgErr', '|v|', 'off-tr', 'note'];
  const widths = [22, 6, 8, 11, 8, 7, 7, 30];
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
      String(r.offTrackEvents),
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
