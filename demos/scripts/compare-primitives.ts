// Experiment: "Stage 2 from ground truth" vs the learned-model pipeline.
//
// Motivation
// ----------
// Our shipped motion-primitive library is derived in TWO stages:
//   1. `train:overnight` distills Rapier into a learned ForwardSim
//      (parametric backbone + residual MLP ensemble) → v2-default.json.
//   2. `characterizeVehicle()` rolls THAT model over a control × start-speed
//      grid to produce the primitive library the planner expands.
//
// This script builds a SIMPLE alternative Stage 2: take the exact same
// control × start-speed grid and roll it directly through the Rapier
// raycast-vehicle (the same ground-truth harness training learns from).
// That gives us the "true" primitive endpoints. We then compare:
//
//   • parametric-only model endpoints   (the safety floor)
//   • full learned model endpoints      (parametric + residual ensemble)
//   • ground-truth Rapier endpoints     (the alternative Stage 2)
//
// and, per primitive, the residual ensemble's disagreement (std) — the
// OOD signal — plus whether the OOD gate fired (full fell back to
// parametric). This directly visualizes the concern: where is the learned
// model biased away from physics, and does the OOD machinery actually
// protect those regimes?
//
// Run:  pnpm --filter @kinocat/demos exec tsx scripts/compare-primitives.ts
//   or: cd demos && npx tsx scripts/compare-primitives.ts

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { resolve, dirname, isAbsolute } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

import {
  coarseWheeledControls,
  fineWheeledControls,
  type ForwardSim,
} from 'kinocat/primitives';
import {
  learnedForwardSimV2,
  buildParametricOnlyModel,
  predictWithUncertainty,
  decodeWheeled,
  type LearnedVehicleModel,
  type CarKinematicState,
} from 'kinocat/agent';
import {
  createHeadlessTrialHarness,
  type HeadlessTrialHarness,
} from 'kinocat/adapters/rapier';
import { modelFromJson } from '../app/lib/v2-model-file';
import { DEFAULT_VEHICLE_OPTS } from '../app/lib/training-driver';

// Mirror of `learned-library-v2.ts` defaults so the learned and ground-truth
// libraries are characterized over an identical grid.
const TIERS = {
  coarse: { startSpeeds: [0, 4, 8, 12], duration: 0.5, substeps: 6 },
  fine: { startSpeeds: [0, 3, 6, 9, 12, 15], duration: 0.15, substeps: 3 },
} as const;

// The model file carries no explicit oodStdThreshold, so the runtime default
// applies. Mirrored here only to report whether the gate WOULD fire.
const OOD_STD_THRESHOLD = [0.5, 0.5, 0.1, 1.0, 0.5, 0.5];
const PHYSICS_DT = 1 / 60;

interface LocalEnd {
  dx: number;
  dz: number;
  dHeading: number;
  speed: number;
}

function wrapAngle(a: number): number {
  let x = a;
  while (x > Math.PI) x -= 2 * Math.PI;
  while (x < -Math.PI) x += 2 * Math.PI;
  return x;
}

/** Express a world-frame end pose in the start sample's local frame
 *  (start at origin, heading 0 — matching `characterizeVehicle`). */
function toLocal(
  start: { x: number; z: number; heading: number },
  end: { x: number; z: number; heading: number; speed: number },
): LocalEnd {
  const dx = end.x - start.x;
  const dz = end.z - start.z;
  const h = start.heading;
  return {
    dx: dx * Math.cos(h) + dz * Math.sin(h),
    dz: -dx * Math.sin(h) + dz * Math.cos(h),
    dHeading: wrapAngle(end.heading - start.heading),
    speed: end.speed,
  };
}

function posErr(a: LocalEnd, b: LocalEnd): number {
  return Math.hypot(a.dx - b.dx, a.dz - b.dz);
}

/** Roll one control vector through Rapier at the physics tick. Returns the
 *  POST-SETTLE start state and the end pose expressed in that start's local
 *  frame. We hand the same post-settle start back to the model rollouts so
 *  both start from byte-identical initial conditions (matched ICs) — this is
 *  what makes the model-vs-physics gap a pure dynamics comparison rather than
 *  one polluted by the harness's settle-phase coast. */
function groundTruth(
  harness: HeadlessTrialHarness,
  startSpeed: number,
  controls: number[],
  duration: number,
): { start: CarKinematicState; localEnd: LocalEnd } | null {
  const ticks = Math.round(duration / PHYSICS_DT);
  const wc = decodeWheeled(controls);
  const outcome = harness.runTrial({
    pose: { x: 0, z: 0, heading: 0 },
    kin: { forwardSpeed: startSpeed },
    controlsTrace: Array.from({ length: ticks }, () => ({ ...wc })),
    sampleEveryNTicks: ticks, // just the post-settle start + the end pose
  });
  if (!outcome.ok) return null;
  const s = outcome.trial.samples;
  const start = s[0]!;
  const end = s[s.length - 1]!;
  return { start, localEnd: toLocal(start, end) };
}

/** Roll a model ForwardSim from a matched start state (re-zeroed to the
 *  local frame: origin, heading 0, but carrying the start's speed / yaw rate
 *  / lateral velocity) for the primitive's duration. */
function rollModel(
  sim: ForwardSim<CarKinematicState>,
  start: CarKinematicState,
  controls: number[],
  duration: number,
  substeps: number,
): LocalEnd {
  const dt = duration / substeps;
  let s: CarKinematicState = {
    x: 0,
    z: 0,
    heading: 0,
    speed: start.speed,
    yawRate: start.yawRate ?? 0,
    lateralVelocity: start.lateralVelocity ?? 0,
    t: 0,
  };
  for (let k = 0; k < substeps; k++) s = sim(s, controls, dt);
  return { dx: s.x, dz: s.z, dHeading: wrapAngle(s.heading), speed: s.speed };
}

interface Row {
  tier: string;
  startSpeed: number;
  controls: number[];
  label: string;
  reverse: boolean;
  gt: LocalEnd;
  para: LocalEnd;
  full: LocalEnd;
  paraErr: number; // parametric vs ground truth (m)
  fullErr: number; // full learned vs ground truth (m)
  headErr: number; // full heading err vs GT (rad)
  speedErr: number; // full speed err vs GT (m/s)
  oodStd: number[]; // ensemble disagreement at the FIRST step
  gateFired: boolean; // would any channel exceed its OOD threshold
  residualActive: boolean; // did the residual move the endpoint off parametric
  settleDecay: number; // bucket speed − Rapier post-settle start speed (m/s)
}

function labelFor(c: number[]): string {
  const w = decodeWheeled(c);
  const dir =
    Math.abs(w.steer) < 1e-6 ? 'straight' : w.steer > 0 ? 'left' : 'right';
  const eff =
    w.brakeForce > 0 ? 'brake' : w.driveForce < 0 ? 'reverse' : w.driveForce > 0 ? 'drive' : 'coast';
  return `${eff}-${dir}`;
}

function rms(xs: number[]): number {
  if (xs.length === 0) return NaN;
  return Math.sqrt(xs.reduce((a, b) => a + b * b, 0) / xs.length);
}

function fmt(n: number, w = 6, d = 3): string {
  return (Number.isFinite(n) ? n.toFixed(d) : 'NaN').padStart(w);
}

async function main(): Promise<void> {
  const { values } = parseArgs({
    options: {
      model: { type: 'string', default: 'demos/public/models/v2-default.json' },
      tier: { type: 'string' }, // coarse | fine | (default: both)
      out: { type: 'string', default: 'docs/primitive-ground-truth-comparison.md' },
      top: { type: 'string', default: '8' },
    },
  });
  const repoRoot = resolve(fileURLToPath(new URL('../..', import.meta.url)));
  const modelPath = isAbsolute(String(values.model))
    ? String(values.model)
    : resolve(repoRoot, String(values.model));

  const model: LearnedVehicleModel = modelFromJson(
    JSON.parse(readFileSync(modelPath, 'utf-8')),
  );
  const paraModel = buildParametricOnlyModel(model.params, model.config);
  const ensembleSize = model.residualEnsemble.length;
  process.stdout.write(
    `Loaded model: ${modelPath}\n` +
      `  ensemble size = ${ensembleSize}, residualReferenceDt = ${model.residualReferenceDt}\n` +
      `  config: maxDriveForce=${model.config.maxDriveForce} maxBrakeForce=${model.config.maxBrakeForce} maxSteerAngle=${model.config.maxSteerAngle}\n\n`,
  );

  const harness = await createHeadlessTrialHarness({
    vehicleOptions: DEFAULT_VEHICLE_OPTS,
    groundBounds: { x0: -500, x1: 500, z0: -500, z1: 500 },
  });

  const tiersToRun = values.tier ? [String(values.tier)] : ['coarse', 'fine'];
  const rows: Row[] = [];

  for (const tierName of tiersToRun) {
    const tier = TIERS[tierName as keyof typeof TIERS];
    if (!tier) throw new Error(`unknown tier ${tierName}`);
    const { duration, substeps } = tier;
    const startSpeeds: number[] = [...tier.startSpeeds];
    const controlSets =
      tierName === 'coarse'
        ? coarseWheeledControls({ config: model.config })
        : fineWheeledControls({ config: model.config });

    // Same ForwardSims the planner library is characterized from: the full
    // learned model and the parametric-only backbone (residual stripped).
    const fullSim = learnedForwardSimV2(model);
    const paraSim = learnedForwardSimV2(paraModel);

    for (const startSpeed of startSpeeds) {
      for (const controls of controlSets) {
        // Ground truth + the post-settle start state we roll the model from.
        const gtResult = groundTruth(harness, startSpeed, controls, duration);
        if (!gtResult) continue; // discarded by Rapier (off-arena / spin)
        const { start, localEnd: gt } = gtResult;

        // Matched ICs: model rollouts start from the identical post-settle
        // state, so the only difference is the dynamics model itself.
        const full = rollModel(fullSim, start, controls, duration, substeps);
        const para = rollModel(paraSim, start, controls, duration, substeps);

        // OOD signal at the first integration step (dt = duration/substeps),
        // the step the residual contributes most strongly to. Evaluated from
        // the same matched start state.
        const dt0 = duration / substeps;
        const pred = predictWithUncertainty(
          model,
          { x: 0, z: 0, heading: 0, speed: start.speed, yawRate: start.yawRate ?? 0, lateralVelocity: start.lateralVelocity ?? 0, t: 0 },
          controls,
          dt0,
        );
        const gateFired = pred.std.some((sd, i) => sd > (OOD_STD_THRESHOLD[i] ?? Infinity));
        const residualActive = posErr(full, para) > 1e-4;

        rows.push({
          tier: tierName,
          startSpeed,
          controls,
          label: labelFor(controls),
          reverse: decodeWheeled(controls).driveForce < 0,
          gt,
          para,
          full,
          paraErr: posErr(para, gt),
          fullErr: posErr(full, gt),
          headErr: Math.abs(wrapAngle(full.dHeading - gt.dHeading)),
          speedErr: Math.abs(full.speed - gt.speed),
          oodStd: pred.std,
          gateFired,
          residualActive,
          settleDecay: startSpeed - start.speed,
        });
      }
    }
  }
  harness.dispose();

  // ---- Report -------------------------------------------------------------
  const lines: string[] = [];
  const out = (s = '') => {
    lines.push(s);
    process.stdout.write(s + '\n');
  };

  out('# Motion primitives: learned model vs ground-truth Rapier');
  out('');
  out(`Model: \`${values.model}\` — ensemble size ${ensembleSize}.`);
  out('');
  out('"Stage 2 from ground truth" = roll the SAME control × start-speed grid');
  out('straight through the Rapier raycast-vehicle (no learned model). Endpoint');
  out('position error is the gap between each model\'s predicted end pose and');
  out('where the real chassis actually ends up, in the start-local frame.');
  out('');
  out('**Matched initial conditions**: the learned/parametric rollouts start from');
  out('Rapier\'s exact post-settle state (speed, yaw rate, lateral velocity), so the');
  out('measured gap is pure dynamics-model error — not a start-state mismatch.');
  out(`Mean settle-phase speed decay across all primitives: ${rms(rows.map((r) => r.settleDecay)).toFixed(3)} m/s (handed to both models, so it cancels).`);
  out('');

  for (const tierName of tiersToRun) {
    const tr = rows.filter((r) => r.tier === tierName);
    if (tr.length === 0) continue;
    const paraRms = rms(tr.map((r) => r.paraErr));
    const fullRms = rms(tr.map((r) => r.fullErr));
    out(`## Tier: ${tierName} (${tr.length} primitives, ${TIERS[tierName as keyof typeof TIERS].duration}s each)`);
    out('');
    out('| metric | value |');
    out('|---|---|');
    out(`| parametric-only endpoint RMS vs Rapier | **${paraRms.toFixed(3)} m** |`);
    out(`| full learned endpoint RMS vs Rapier | **${fullRms.toFixed(3)} m** |`);
    out(`| residual helps (full < para) | ${(fullRms < paraRms ? 'yes' : 'NO — residual hurts')} (${((1 - fullRms / paraRms) * 100).toFixed(1)}%) |`);
    out(`| primitives where OOD gate fires | ${tr.filter((r) => r.gateFired).length} / ${tr.length} |`);
    out(`| primitives where residual is active | ${tr.filter((r) => r.residualActive).length} / ${tr.length} |`);
    out('');

    // Per start-speed bucket — the regime axis where OOD lives.
    out('Per start-speed bucket (full learned endpoint err vs Rapier):');
    out('');
    out('| start speed | full err RMS | para err RMS | mean ensemble σ (pos) | gate fires |');
    out('|---|---|---|---|---|');
    const speeds = [...new Set(tr.map((r) => r.startSpeed))].sort((a, b) => a - b);
    for (const sp of speeds) {
      const g = tr.filter((r) => r.startSpeed === sp);
      const sigPos = g.map((r) => Math.hypot(r.oodStd[0] ?? 0, r.oodStd[1] ?? 0));
      out(
        `| ${sp} m/s | ${rms(g.map((r) => r.fullErr)).toFixed(3)} m | ${rms(g.map((r) => r.paraErr)).toFixed(3)} m | ${rms(sigPos).toFixed(3)} | ${g.filter((r) => r.gateFired).length}/${g.length} |`,
      );
    }
    out('');

    // Worst offenders — where the shipped library diverges most from physics.
    const topN = Number(values.top);
    const worst = [...tr].sort((a, b) => b.fullErr - a.fullErr).slice(0, topN);
    out(`Worst ${topN} primitives (largest full-model endpoint error):`);
    out('');
    out('```');
    out('startSpd  action            fullErr  paraErr  headErr  spdErr  ensσ(pos)  gate  residual');
    for (const r of worst) {
      out(
        `${fmt(r.startSpeed, 6, 1)}  ${r.label.padEnd(16)}  ${fmt(r.fullErr)}  ${fmt(r.paraErr)}  ${fmt(r.headErr)}  ${fmt(r.speedErr)}  ${fmt(Math.hypot(r.oodStd[0] ?? 0, r.oodStd[1] ?? 0))}  ${(r.gateFired ? ' ON' : 'off')}  ${r.residualActive ? 'active' : '—'}`,
      );
    }
    out('```');
    out('');
  }

  out('## How to read this');
  out('');
  out('- **parametric RMS** is the safety floor: the clean analytical model the');
  out('  residual is allowed to correct. **full RMS** is what the planner actually');
  out('  expands. If full < para, the overnight residual is net-helping; if full >');
  out('  para in a bucket, the residual is *confidently wrong* there (shared bias).');
  out('- **ensemble σ** is the OOD signal. High σ → the 3 MLPs disagree → the regime');
  out('  is under-trained. The gate falls back to parametric when any channel σ');
  out('  exceeds its threshold, so high-σ rows *should* show full ≈ para.');
  out('- The Rapier column is the alternative Stage 2: zero model error by');
  out('  construction, at the cost of needing the physics engine in the loop.');
  out('');
  out('Caveats: `spdErr` is the error in *forward-axis-projected* speed (lin·forward,');
  out('measured identically for model and Rapier); under hard steer the chassis slides');
  out('and its velocity rotates off the heading axis, so high-steer rows show large');
  out('`spdErr` even when the path is close. Both models are rolled from Rapier\'s');
  out('exact post-settle start state, so suspension settle and start-speed decay');
  out('cancel — the residual gap is dynamics-model error alone.');

  const outPath = isAbsolute(String(values.out)) ? String(values.out) : resolve(repoRoot, String(values.out));
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, lines.join('\n') + '\n', 'utf-8');
  process.stdout.write(`\nwrote ${outPath}\n`);
}

main().then(
  () => process.exit(0),
  (err) => {
    process.stderr.write(`compare-primitives failed: ${err instanceof Error ? (err.stack ?? err.message) : String(err)}\n`);
    process.exit(1);
  },
);
