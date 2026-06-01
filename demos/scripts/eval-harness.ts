// `pnpm run eval` — the component-level evaluation harness.
//
// Implements the evaluation guide's fastest-path-to-insight (§4.1 + §4.2 + §6):
// it decomposes driving into the PLAN (scored as a static artifact), the
// EXECUTION, and the GAP between them (= controller tracking error), so a
// failure localizes to the responsible component.
//
// Part A — controller isolation (§4.1) over a parameter SWEEP (§7): feed the
//   tracker a known-good analytic reference (arc / slalom / lane-change) at a
//   sweep of entry speeds, run ONLY the controller, and report cross-track /
//   heading / velocity error + smoothness. The same references are also scored
//   as plans (§4.2: feasibility + g-g utilization), and the §6 diagnosis 2×2 is
//   read off — so an infeasible-at-high-speed reference shows up as a PLANNER
//   fault, not a controller one.
//
// Per-scenario METRIC VECTORS are aggregated as mean ± std over the sweep and
// written to `docs/eval-results/` as a history JSON (a diff over time).
//
// CLI:
//   pnpm run eval
//   pnpm run eval --json=docs/eval-results/latest.json
//   pnpm run eval --tracker=mpc

import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname, isAbsolute, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

import {
  toReferenceTrajectory,
  runControllerIsolation,
  scorePlan,
  comfortFlags,
  diagnose,
  limitsFromAgent,
  type RefController,
  type DynamicLimits,
} from 'kinocat/eval';
import { arcPath, slalom, laneChange } from 'kinocat/eval';
import { purePursuit, smoothTrajectory } from 'kinocat/execute';
import type { PurePursuitConfig, PlanPath } from 'kinocat/execute';
import { defaultVehicleAgent, kinematicForwardSim } from 'kinocat/agent';
import type { CarKinematicState } from 'kinocat/agent';
import { createRaceScenario, TRACKER_MAX_LATERAL_ACCEL } from '../app/lib/race-scenario';
import { RACE_AGENT } from '../app/lib/race-primitives-scenarios';
import { kinematicEntry } from '../app/lib/headless-race';

const DT = 1 / 60;
const agent = defaultVehicleAgent();

// Friction-circle / accel budget. `frictionLimit` mirrors the pure-pursuit
// `maxLateralAccel`, so plans are scored against the same envelope the
// controller is tuned to.
const FRICTION_LIMIT = 4.0;
const limits: DynamicLimits = limitsFromAgent(agent, {
  frictionLimit: FRICTION_LIMIT,
  maxAccel: 6.5,
  maxDecel: 8,
});

const ppConfig: PurePursuitConfig = {
  lookaheadMin: 2,
  lookaheadGain: 0.3,
  lookaheadMax: 6,
  maxLateralAccel: FRICTION_LIMIT,
  maxAccel: 6.5,
  maxDecel: 8,
  cruiseSpeed: 12,
  goalTolerance: 1.0,
  minTurnRadius: agent.minTurnRadius,
  respectPathSpeed: true,
};

const purePursuitController: RefController = (state, path) => {
  const cmd = purePursuit(state, path as PlanPath, ppConfig);
  return { controls: [cmd.steering, cmd.targetSpeed], steer: cmd.steering, atGoal: cmd.atGoal };
};

// ---------------------------------------------------------------------------
// Reference families for the controller-isolation sweep.

interface RefFamily {
  name: string;
  description: string;
  /** Build the ideal line at a given entry speed. */
  build: (speed: number) => CarKinematicState[];
}

const FAMILIES: RefFamily[] = [
  {
    name: 'sweeping-turn',
    description: 'wide 20 m-radius 90° sweep',
    build: (speed) => arcPath({ radius: 20, sweep: Math.PI / 2, speed, ds: 0.5 }),
  },
  {
    name: 'hairpin',
    description: 'tight 6 m-radius 180° hairpin',
    build: (speed) => arcPath({ radius: 6, sweep: Math.PI, speed, ds: 0.4 }),
  },
  {
    name: 'slalom',
    // 12 m spacing / 2 m amplitude ⇒ peak radius ≈ 7.3 m (≥ the car's 4 m min
    // turn radius), so the weave is geometrically drivable; the entry-speed
    // sweep then crosses the lateral-accel feasibility boundary (guide §7).
    description: 'cone weave, 12 m spacing, 2 m amplitude',
    build: (speed) => slalom({ spacing: 12, amplitude: 2, cones: 4, speed, ds: 0.4 }),
  },
  {
    name: 'lane-change',
    description: 'double-lane-change, 3.5 m over 30 m',
    build: (speed) => laneChange({ width: 3.5, length: 30, speed, ds: 0.5 }),
  },
];

const ENTRY_SPEEDS = [4, 6, 8, 10];

// ---------------------------------------------------------------------------
// Per-(family, speed) metric vector.

interface MetricVector {
  family: string;
  speed: number;
  // controller fidelity (the gap)
  crossTrackRmse: number;
  crossTrackMax: number;
  crossTrackP95: number;
  headingRmse: number;
  velocityRmse: number;
  steerRateRms: number;
  steerReversals: number;
  // plan quality
  feasible: boolean;
  worstFeasRatio: number;
  meanUtil: number;
  peakUtil: number;
  // executed quality
  comfortable: boolean;
  // diagnosis
  verdict: string;
}

function evaluateCase(family: RefFamily, speed: number): MetricVector {
  const reference = family.build(speed);
  const refTraj = toReferenceTrajectory(reference);

  // §4.2 — score the plan as a static artifact.
  const goalState = reference[reference.length - 1]!;
  const plan = scorePlan(reference, limits, {
    goal: { x: goalState.x, z: goalState.z, heading: goalState.heading, speed: goalState.speed },
  });

  // §4.1 — run ONLY the controller against the reference.
  const iso = runControllerIsolation(reference, purePursuitController, kinematicForwardSim(agent), DT, {
    maxSteps: 4000,
  });

  // §5 — executed-trajectory comfort.
  const comfort = comfortFlags(iso.executed, DT);

  // §6 — diagnosis 2×2.
  const dx = diagnose(plan, iso.report);

  return {
    family: family.name,
    speed,
    crossTrackRmse: iso.report.crossTrack.rmse,
    crossTrackMax: iso.report.crossTrack.max,
    crossTrackP95: iso.report.crossTrack.p95,
    headingRmse: iso.report.heading.rmse,
    velocityRmse: iso.report.velocity.rmse,
    steerRateRms: iso.report.steerRateRms,
    steerReversals: iso.report.steerReversals,
    feasible: plan.feasibility.feasible,
    worstFeasRatio: plan.feasibility.worstRatio,
    meanUtil: plan.gg.meanUtil,
    peakUtil: plan.gg.peakUtil,
    comfortable: comfort.comfortable,
    verdict: dx.verdict,
  };
}

// ---------------------------------------------------------------------------
// Aggregation: mean ± std across the speed sweep, per family.

interface Aggregate {
  family: string;
  n: number;
  crossTrackRmse: { mean: number; std: number };
  headingRmse: { mean: number; std: number };
  velocityRmse: { mean: number; std: number };
  meanUtil: { mean: number; std: number };
  feasibleFraction: number;
  comfortableFraction: number;
  verdicts: Record<string, number>;
}

function meanStd(xs: number[]): { mean: number; std: number } {
  if (xs.length === 0) return { mean: 0, std: 0 };
  const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
  const variance = xs.reduce((a, b) => a + (b - mean) * (b - mean), 0) / xs.length;
  return { mean, std: Math.sqrt(variance) };
}

function aggregate(family: string, rows: MetricVector[]): Aggregate {
  const verdicts: Record<string, number> = {};
  for (const r of rows) verdicts[r.verdict] = (verdicts[r.verdict] ?? 0) + 1;
  return {
    family,
    n: rows.length,
    crossTrackRmse: meanStd(rows.map((r) => r.crossTrackRmse)),
    headingRmse: meanStd(rows.map((r) => r.headingRmse)),
    velocityRmse: meanStd(rows.map((r) => r.velocityRmse)),
    meanUtil: meanStd(rows.map((r) => r.meanUtil)),
    feasibleFraction: rows.filter((r) => r.feasible).length / rows.length,
    comfortableFraction: rows.filter((r) => r.comfortable).length / rows.length,
    verdicts,
  };
}

// ---------------------------------------------------------------------------
// Part B — planner isolation on the REAL planner. Run the live race scenario,
// capture every distinct committed plan, and score each as a static artifact
// (feasibility + g-g utilization). This answers "how good is our planner",
// not the analytic-reference proxy in Part A.

interface RacePlannerReport {
  lapSeconds: number;
  plansScored: number;
  /** Fraction of committed plans that are fully feasible. */
  feasibleFraction: number;
  /** Median per-plan mean g-g utilization — robust to recovery-stub outliers. */
  medianUtil: number;
  meanUtil: { mean: number; std: number };
  /** Fraction of plans whose mean utilization exceeds 100% of the envelope. */
  overEnvelopeFraction: number;
}

function median(xs: number[]): number {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid]! : 0.5 * (s[mid - 1]! + s[mid]!);
}

async function evaluateRacePlanner(): Promise<RacePlannerReport | null> {
  const MAX_SIM = 120;
  // Score against the RACE agent's true limits — not the analytic-harness ones.
  // `frictionLimit` is the race tracker's `maxLateralAccel` (the Rapier chassis
  // friction circle), so the plan is judged against the same envelope the
  // controller is tuned to.
  const raceLimits: DynamicLimits = limitsFromAgent(RACE_AGENT, {
    frictionLimit: TRACKER_MAX_LATERAL_ACCEL,
    maxAccel: 6,
    maxDecel: 8,
  });
  const scenario = await createRaceScenario({
    entries: [kinematicEntry('kinematic')],
    targetLaps: 1,
    syncHold: false,
    offTrackRecovery: 'spawn',
    tuning: { tracker: 'pure-pursuit' },
  });

  const utils: number[] = [];
  let feasibleCount = 0;
  let overEnvelope = 0;
  let prevPlanRef: CarKinematicState[] | null = null;

  while (scenario.simTime() < MAX_SIM) {
    const r = scenario.tick();
    const plan = scenario.status()[0]!.plan;
    // Score each newly-committed plan once (dedupe by reference identity).
    // Skip short off-track-recovery stubs (≥ 10 samples, ≥ 5 m long): they are
    // not representative racing plans and their near-degenerate geometry
    // produces meaningless curvature spikes.
    if (plan && plan !== prevPlanRef && plan.length >= 10) {
      prevPlanRef = plan;
      // Score the plan the controller actually tracks: resampled to uniform
      // arc-length via the library's own smoother. This removes Menger
      // curvature spikes from the unevenly-sampled "lifted for visualization"
      // polyline (degenerate near-duplicate triples otherwise blow up v²·κ).
      const tracked = smoothTrajectory(plan, { sampleSpacing: 0.5, iterations: 8 });
      if (tracked.length < 6) continue;
      const q = scorePlan(tracked, raceLimits);
      if (q.pathLength < 5) continue;
      if (q.feasibility.feasible) feasibleCount++;
      if (q.gg.meanUtil > 1) overEnvelope++;
      utils.push(q.gg.meanUtil);
    }
    if (r.allFinished) break;
  }

  const status = scenario.status()[0]!;
  const lapSeconds = status.laps.length >= 1 ? status.laps[0]!.duration : NaN;
  scenario.dispose();

  const plansScored = utils.length;
  if (plansScored === 0) return null;
  return {
    lapSeconds,
    plansScored,
    feasibleFraction: feasibleCount / plansScored,
    medianUtil: median(utils),
    meanUtil: meanStd(utils),
    overEnvelopeFraction: overEnvelope / plansScored,
  };
}

// ---------------------------------------------------------------------------

function fmt(n: number, d = 3): string {
  return Number.isFinite(n) ? n.toFixed(d) : '---';
}

async function main(): Promise<void> {
  const { values } = parseArgs({
    options: {
      json: { type: 'string' },
      filter: { type: 'string' },
      'no-scenarios': { type: 'boolean' },
      help: { type: 'boolean', short: 'h' },
    },
  });
  if (values.help) {
    process.stdout.write(
      'Usage: pnpm run eval [--filter=name,...] [--no-scenarios] [--json=out.json]\n',
    );
    return;
  }
  const filter = values.filter ? new Set(values.filter.split(',').map((s) => s.trim())) : null;
  const families = filter ? FAMILIES.filter((f) => filter.has(f.name)) : FAMILIES;

  process.stdout.write(
    `kinocat eval · controller=pure-pursuit · ${families.length} scenarios × ${ENTRY_SPEEDS.length} entry speeds\n\n`,
  );

  const allRows: MetricVector[] = [];
  const aggregates: Aggregate[] = [];

  for (const family of families) {
    process.stdout.write(`▶ ${family.name}: ${family.description}\n`);
    const rows: MetricVector[] = [];
    for (const speed of ENTRY_SPEEDS) {
      const m = evaluateCase(family, speed);
      rows.push(m);
      allRows.push(m);
      process.stdout.write(
        `   v=${String(speed).padStart(2)} m/s  ` +
          `xtrack RMSE=${fmt(m.crossTrackRmse)}m max=${fmt(m.crossTrackMax)} P95=${fmt(m.crossTrackP95)}  ` +
          `hdg=${fmt(m.headingRmse)}  vel=${fmt(m.velocityRmse)}  ` +
          `feas=${m.feasible ? 'yes' : 'NO '}(${fmt(m.worstFeasRatio, 2)})  ` +
          `util=${fmt(m.meanUtil * 100, 0)}%/${fmt(m.peakUtil * 100, 0)}%  ` +
          `comfy=${m.comfortable ? 'yes' : 'no '}  ⇒ ${m.verdict}\n`,
      );
    }
    aggregates.push(aggregate(family.name, rows));
    process.stdout.write('\n');
  }

  // Summary table — mean ± std per family.
  process.stdout.write('Per-scenario aggregate (mean ± std over the entry-speed sweep):\n');
  const headers = ['scenario', 'xtrackRMSE(m)', 'hdgRMSE(rad)', 'velRMSE(m/s)', 'util', 'feasible', 'comfy'];
  const widths = [14, 16, 15, 14, 14, 10, 7];
  const sep = '─'.repeat(widths.reduce((a, b) => a + b + 2, 0));
  process.stdout.write(headers.map((h, i) => h.padEnd(widths[i] ?? 0)).join('  ') + '\n' + sep + '\n');
  for (const a of aggregates) {
    const row = [
      a.family,
      `${fmt(a.crossTrackRmse.mean)}±${fmt(a.crossTrackRmse.std)}`,
      `${fmt(a.headingRmse.mean)}±${fmt(a.headingRmse.std)}`,
      `${fmt(a.velocityRmse.mean)}±${fmt(a.velocityRmse.std)}`,
      `${fmt(a.meanUtil.mean * 100, 0)}%±${fmt(a.meanUtil.std * 100, 0)}`,
      `${fmt(a.feasibleFraction * 100, 0)}%`,
      `${fmt(a.comfortableFraction * 100, 0)}%`,
    ];
    process.stdout.write(row.map((c, i) => String(c).padEnd(widths[i] ?? 0)).join('  ') + '\n');
  }
  process.stdout.write(sep + '\n\n');

  // Diagnosis roll-up.
  const verdictTotals: Record<string, number> = {};
  for (const r of allRows) verdictTotals[r.verdict] = (verdictTotals[r.verdict] ?? 0) + 1;
  process.stdout.write(
    'Diagnosis 2×2 roll-up: ' +
      Object.entries(verdictTotals)
        .map(([k, v]) => `${k}=${v}`)
        .join('  ') +
      '\n',
  );

  // Part B — score the REAL planner's committed plans on the live race course.
  let racePlanner: RacePlannerReport | null = null;
  if (!values['no-scenarios']) {
    process.stdout.write('\nPlanner isolation on the live race course (real committed plans):\n');
    try {
      racePlanner = await evaluateRacePlanner();
      if (racePlanner) {
        process.stdout.write(
          `   lap=${fmt(racePlanner.lapSeconds, 1)}s  plansScored=${racePlanner.plansScored}  ` +
            `feasible=${fmt(racePlanner.feasibleFraction * 100, 0)}%  ` +
            `util(median)=${fmt(racePlanner.medianUtil * 100, 0)}%  ` +
            `util(mean±std)=${fmt(racePlanner.meanUtil.mean * 100, 0)}±${fmt(racePlanner.meanUtil.std * 100, 0)}%  ` +
            `over-envelope=${fmt(racePlanner.overEnvelopeFraction * 100, 0)}%\n`,
        );
        process.stdout.write(
          '   (the kinematic baseline deliberately plans aggressive speeds the ' +
            'controller clips — see RACE_AGENT notes; the learned model plans honest entry speeds)\n',
        );
      } else {
        process.stdout.write('   (no plans captured)\n');
      }
    } catch (e) {
      process.stdout.write(
        `   skipped (scenario error: ${e instanceof Error ? e.message : String(e)})\n`,
      );
    }
  }

  if (values.json) {
    const __dirname = dirname(fileURLToPath(import.meta.url));
    const out = isAbsolute(values.json) ? values.json : resolve(__dirname, '..', values.json);
    mkdirSync(dirname(out), { recursive: true });
    const payload = {
      generatedAt: new Date().toISOString(),
      controller: 'pure-pursuit',
      dt: DT,
      frictionLimit: FRICTION_LIMIT,
      entrySpeeds: ENTRY_SPEEDS,
      rows: allRows,
      aggregates,
      racePlanner,
    };
    writeFileSync(out, JSON.stringify(payload, null, 2));
    process.stdout.write(`\nwrote ${out}\n`);
  }
}

main().catch((e) => {
  process.stderr.write(`eval-harness failed: ${e instanceof Error ? e.stack ?? e.message : String(e)}\n`);
  process.exit(1);
});
