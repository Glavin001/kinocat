// TEMP — motion-primitive COVERAGE analyzer (control-set design by dispersion).
//
// Judges a primitive library by its OUTPUTS (rolled-out endpoints), not its
// inputs: enumerates a dense candidate grid of controls, rolls each through the
// forward model to a body-frame endpoint, and measures
//   - COVERAGE / DISPERSION: the largest reachable endpoint that is FAR from
//     every library primitive (the worst "hole" — e.g. a missing medium-radius
//     arc). This is the number that turns "we forgot an arc" into a metric.
//   - REDUNDANCY: the closest pair of library endpoints (wasted slots).
//   - EXTREMES: does the set reach the fastest straight / tightest turn / hardest
//     brake the chassis can actually do at this speed?
// It then runs farthest-point selection over the dense candidates to a slot
// budget and reports how much lower the dispersion of a coverage-optimal set is.
//
// usage: npx tsx scripts/tmp-primitive-coverage.mts <v2|v3|kin> [startSpeed]
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  buildLearnedRaceLibraryV2,
  buildLearnedRaceLibraryV3,
  buildKinematicLibrary,
  RACE_AGENT,
} from '../app/lib/race-primitives-scenarios';
import { modelFromJson } from '../app/lib/v2-model-file';
import { v3FromJson, forwardSimV3, learnedForwardSimV2, kinematicForwardSim } from 'kinocat/agent';
import type { CarKinematicState } from 'kinocat/agent';
import type { MotionPrimitive } from 'kinocat/primitives';

const root = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const which = process.argv[2] ?? 'v3';
const startSpeed = Number(process.argv[3] ?? 14);
const readModel = (f: string) => JSON.parse(readFileSync(resolve(root, 'demos/public/models', f), 'utf-8'));

const ENGINE = 4000, BRAKE = 2000, MAXSTEER = RACE_AGENT.maxSteerAngle ?? 0.6;
const DUR = 0.55, SUBSTEPS = 6;

// Forward sim + library per model. Learned models use wheeled controls
// [steer, driveForce, brakeForce]; kinematic uses [curvature, targetSpeed].
const wheeled = which !== 'kin';
let sim: (s: CarKinematicState, u: number[], dt: number) => CarKinematicState;
let lib;
if (which === 'v3') { const m = v3FromJson(readModel('v3-default.json')); sim = forwardSimV3(m); lib = buildLearnedRaceLibraryV3(m); }
else if (which === 'v2') { const m = modelFromJson(readModel('v2-default.json')); sim = learnedForwardSimV2(m); lib = buildLearnedRaceLibraryV2(m); }
else { sim = kinematicForwardSim(RACE_AGENT); lib = buildKinematicLibrary(); }

type End = { dx: number; dz: number; dh: number; dv: number; u: number[] };
function rollout(u: number[]): End {
  let s: CarKinematicState = { x: 0, z: 0, heading: 0, speed: startSpeed, t: 0 };
  const dt = DUR / SUBSTEPS;
  for (let k = 0; k < SUBSTEPS; k++) s = sim(s, u, dt);
  return { dx: s.x, dz: s.z, dh: s.heading, dv: s.speed - startSpeed, u };
}

// Endpoint distance: position in metres, heading via a 2 m arm, speed via 0.3 m
// per m/s — so all dims are comparable "metres of state difference".
const HW = 2.0, VW = 0.3;
function endDist(a: End, b: End): number {
  return Math.hypot(a.dx - b.dx, a.dz - b.dz, HW * wrap(a.dh - b.dh), VW * (a.dv - b.dv));
}
function wrap(x: number): number { while (x > Math.PI) x -= 2 * Math.PI; while (x < -Math.PI) x += 2 * Math.PI; return x; }

// ---- Dense candidate grid ----
const candidates: End[] = [];
if (wheeled) {
  const NS = 21, NP = 13;
  for (let i = 0; i < NS; i++) {
    const steer = -MAXSTEER + (2 * MAXSTEER * i) / (NS - 1);
    for (let j = 0; j < NP; j++) {
      const a = -1 + (2 * j) / (NP - 1); // pedal: >0 drive, <0 brake
      const u = [steer, a >= 0 ? a * ENGINE : 0, a >= 0 ? 0 : -a * BRAKE];
      candidates.push(rollout(u));
    }
  }
} else {
  const NC = 21, NV = 13;
  const k = 1 / RACE_AGENT.minTurnRadius;
  for (let i = 0; i < NC; i++) {
    const curv = -k + (2 * k * i) / (NC - 1);
    for (let j = 0; j < NV; j++) {
      const tv = (RACE_AGENT.maxSpeed * j) / (NV - 1);
      candidates.push(rollout([curv, tv]));
    }
  }
}

// ---- Library endpoints for this bucket ----
const prims: MotionPrimitive[] = lib.lookup(startSpeed).filter((p) => !p.reverse);
const libEnds: End[] = prims.map((p) => ({ dx: p.end.dx, dz: p.end.dz, dh: p.end.dHeading, dv: p.end.speed - startSpeed, u: p.controls }));

// ---- Coverage / dispersion: worst-covered reachable candidate ----
let hole: { d: number; c: End } = { d: 0, c: candidates[0]! };
let covSum = 0;
for (const c of candidates) {
  let nearest = Infinity;
  for (const e of libEnds) nearest = Math.min(nearest, endDist(c, e));
  covSum += nearest;
  if (nearest > hole.d) hole = { d: nearest, c };
}
// ---- Redundancy: closest library pair ----
let redun = { d: Infinity, i: 0, j: 0 };
for (let i = 0; i < libEnds.length; i++)
  for (let j = i + 1; j < libEnds.length; j++) {
    const d = endDist(libEnds[i]!, libEnds[j]!);
    if (d < redun.d) redun = { d, i, j };
  }

// ---- Farthest-point selection to the same budget over the dense grid ----
function fps(pts: End[], budget: number): End[] {
  const chosen: End[] = [pts[0]!];
  const mind = pts.map((p) => endDist(p, pts[0]!));
  while (chosen.length < budget) {
    let bi = 0;
    for (let i = 1; i < pts.length; i++) if (mind[i]! > mind[bi]!) bi = i;
    chosen.push(pts[bi]!);
    for (let i = 0; i < pts.length; i++) mind[i] = Math.min(mind[i]!, endDist(pts[i]!, pts[bi]!));
  }
  // dispersion of `chosen` over the candidate set = max over candidates of
  // nearest-chosen distance.
  let disp = 0;
  for (const c of pts) {
    let n = Infinity;
    for (const s of chosen) n = Math.min(n, endDist(c, s));
    disp = Math.max(disp, n);
  }
  return chosen.map((c) => ({ ...c, dispersion: disp } as End & { dispersion: number }));
}
const sel = fps(candidates, libEnds.length) as (End & { dispersion: number })[];

console.log(`\n=== primitive coverage: ${which} @ ${startSpeed} m/s (forward primitives) ===`);
console.log(`library slots (fwd): ${libEnds.length} | dense candidates: ${candidates.length}`);
console.log(`COVERAGE  mean nearest-lib dist = ${(covSum / candidates.length).toFixed(2)} m`);
console.log(`DISPERSION (worst hole)         = ${hole.d.toFixed(2)} m  at endpoint (dx=${hole.c.dx.toFixed(1)}, dz=${hole.c.dz.toFixed(1)}, dh=${hole.c.dh.toFixed(2)}, dv=${hole.c.dv.toFixed(1)})`);
console.log(`   hole controls = [${hole.c.u.map((x) => x.toFixed(0)).join(', ')}]  ${wheeled ? '(steer, drive, brake)' : '(curv, targetV)'}`);
console.log(`REDUNDANCY closest lib pair     = ${redun.d.toFixed(2)} m  (prims #${redun.i} & #${redun.j})`);
console.log(`EXTREMES  max|dx|=${Math.max(...libEnds.map((e) => Math.abs(e.dx))).toFixed(1)}  max|dh|=${Math.max(...libEnds.map((e) => Math.abs(e.dh))).toFixed(2)}  max brake dv=${Math.min(...libEnds.map((e) => e.dv)).toFixed(1)}`);
console.log(`   (dense reachable: max|dx|=${Math.max(...candidates.map((e) => Math.abs(e.dx))).toFixed(1)}  max|dh|=${Math.max(...candidates.map((e) => Math.abs(e.dh))).toFixed(2)}  max brake dv=${Math.min(...candidates.map((e) => e.dv)).toFixed(1)})`);
console.log(`FPS-SELECTED set of ${libEnds.length}: dispersion = ${sel[0]!.dispersion.toFixed(2)} m  (vs library ${hole.d.toFixed(2)} m — lower is better coverage)`);
