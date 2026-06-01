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
  characterizeVehicle,
  coarseWheeledControls,
  fineWheeledControls,
} from 'kinocat/primitives';
import {
  learnedForwardSimV2,
  buildParametricOnlyModel,
  predictWithUncertainty,
  decodeWheeled,
  type LearnedVehicleModel,
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

/** Roll one control vector through Rapier at the physics tick and return the
 *  end pose in the primitive's start-local frame. */
function groundTruthEnd(
  harness: HeadlessTrialHarness,
  startSpeed: number,
  controls: number[],
  duration: number,
  sampleEveryNTicks: number,
): LocalEnd | null {
  const ticks = Math.round(duration / PHYSICS_DT);
  const wc = decodeWheeled(controls);
  const outcome = harness.runTrial({
    pose: { x: 0, z: 0, heading: 0 },
    kin: { forwardSpeed: startSpeed },
    controlsTrace: Array.from({ length: ticks }, () => ({ ...wc })),
    sampleEveryNTicks,
  });
  if (!outcome.ok) return null;
  const s = outcome.trial.samples;
  const start = s[0]!;
  const end = s[s.length - 1]!;
  return toLocal(start, end);
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
    const sampleEveryNTicks = Math.round(duration / PHYSICS_DT / substeps);
    const controlSets =
      tierName === 'coarse'
        ? coarseWheeledControls({ config: model.config })
        : fineWheeledControls({ config: model.config });

    // Learned libraries (full + parametric-only) — exactly the planner's path.
    const fullLib = characterizeVehicle({
      forwardSim: learnedForwardSimV2(model),
      controlSets,
      duration,
      substeps,
      startSpeeds,
    });
    const paraLib = characterizeVehicle({
      forwardSim: learnedForwardSimV2(paraModel),
      controlSets,
      duration,
      substeps,
      startSpeeds,
    });

    // Match by index — characterizeVehicle iterates startSpeed outer, control
    // inner, in array order.
    let idx = 0;
    for (const startSpeed of startSpeeds) {
      for (const controls of controlSets) {
        const full = fullLib.primitives[idx]!.end;
        const para = paraLib.primitives[idx]!.end;
        idx++;
        const gt = groundTruthEnd(harness, startSpeed, controls, duration, sampleEveryNTicks);
        if (!gt) continue; // discarded by Rapier (off-arena / spin) — rare here

        // OOD signal at the first integration step (dt = duration/substeps),
        // the step the residual contributes most strongly to.
        const dt0 = duration / substeps;
        const pred = predictWithUncertainty(
          model,
          { x: 0, z: 0, heading: 0, speed: startSpeed, t: 0 },
          controls,
          dt0,
        );
        const gateFired = pred.std.some((s, i) => s > (OOD_STD_THRESHOLD[i] ?? Infinity));
        const residualActive = posErr(full, para) > 1e-4;

        rows.push({
          tier: tierName,
          startSpeed,
          controls,
          label: labelFor(controls),
          reverse: fullLib.primitives[idx - 1]!.reverse,
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
  out('position error is the gap between each primitive\'s predicted end pose and');
  out('where the real chassis actually ends up, in the start-local frame.');
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
  out('Caveats: (1) `spdErr` is the error in *forward-axis-projected* speed; under');
  out('hard steer the chassis slides and its velocity rotates off the heading axis,');
  out('so high-steer rows show large `spdErr` even when the path is close. (2) The');
  out('ground-truth harness settles the suspension (~0.15s coast) before recording,');
  out('matching training conditions, so the GT primitive starts a hair below its');
  out('bucket speed — a small handicap charged against the learned model. Endpoint');
  out('position error (re-zeroed to the post-settle start) is the robust headline.');

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
