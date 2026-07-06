// TEMP — sample-level MPPI dissection at wedge moments. Not committed.
// usage: npx tsx scripts/tmp-solve-probe.mts <kin|v2|v3> [maxSec]
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRaceScenario } from '../app/lib/race-scenario';
import { kinematicEntry, v2Entry, v3Entry } from '../app/lib/headless-race';
import { buildRaceCourse } from '../app/lib/race-primitives-scenarios';
import { modelFromJson } from '../app/lib/v2-model-file';
import { v3FromJson } from 'kinocat/agent';
import type { MPCDebugInfo } from 'kinocat/execute';

const repoRoot = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const which = process.argv[2] ?? 'v3';
const maxSec = Number(process.argv[3] ?? 30);
const entry =
  which === 'v3'
    ? v3Entry('v3', v3FromJson(JSON.parse(readFileSync(resolve(repoRoot, 'demos/public/models/v3-default.json'), 'utf-8'))))
    : which === 'v2'
      ? v2Entry('v2', modelFromJson(JSON.parse(readFileSync(resolve(repoRoot, 'demos/public/models/v2-default.json'), 'utf-8'))))
      : kinematicEntry('kin');

let lastInfo: MPCDebugInfo | null = null;
const scenario = await createRaceScenario({
  entries: [entry],
  targetLaps: 2,
  syncHold: false,
  course: buildRaceCourse('open'),
  tuning: {
    plannerBudgetMs: 10_000,
    tracker: 'mpc',
    mpcOverrides: { onDebug: (info) => { lastInfo = info; } },
  },
});

function dissect(info: MPCDebugInfo): void {
  const K = info.costs.length;
  const H = info.horizon;
  const order = Array.from({ length: K }, (_, i) => i).sort((a, b) => info.costs[a]! - info.costs[b]!);
  const q = (p: number): number => info.costs[order[Math.floor(p * (K - 1))]!]!;
  console.log(
    `  costs: min=${info.minCost.toFixed(1)} q25=${q(0.25).toFixed(1)} med=${q(0.5).toFixed(1)} q75=${q(0.75).toFixed(1)} max=${q(1).toFixed(1)} ` +
    `bestShare=${info.bestWeightShare.toFixed(3)} gear=${info.gear} anchor=${info.anchor ? `${info.anchor.s.toFixed(1)}m@${info.anchor.idx}` : '-'}`,
  );
  for (const rank of [0, 1, 2, K - 1]) {
    const k = order[rank]!;
    const s0 = info.samples[k * H * 3]!;
    const d0 = info.samples[k * H * 3 + 1]!;
    const b0 = info.samples[k * H * 3 + 2]!;
    // Mean drive/brake over the horizon for this sample.
    let dSum = 0; let bSum = 0;
    for (let i = 0; i < H; i++) {
      dSum += info.samples[k * H * 3 + i * 3 + 1]!;
      bSum += info.samples[k * H * 3 + i * 3 + 2]!;
    }
    console.log(
      `  ${rank === K - 1 ? 'worst' : `best${rank}`}: cost=${info.costs[k]!.toFixed(1)} u0=[${s0.toFixed(2)}, ${d0.toFixed(0)}, ${b0.toFixed(0)}] ` +
      `meanDrive=${(dSum / H).toFixed(0)} meanBrake=${(bSum / H).toFixed(0)}`,
    );
  }
  console.log(
    `  emitted: steer=${info.emitted.steer.toFixed(2)} drive=${info.emitted.driveForce.toFixed(0)} brake=${info.emitted.brakeForce.toFixed(0)}`,
  );
}

let stopSince = -1;
let dumped = 0;
while (scenario.simTime() < maxSec && dumped < 4) {
  const r = scenario.tick();
  const c = r.cars[0]!;
  if (Math.abs(c.state.speed) < 0.2 && r.simTime > 3) {
    if (stopSince < 0) stopSince = r.simTime;
    if (r.simTime - stopSince > 1.2 && lastInfo) {
      dumped++;
      console.log(`\nWEDGE t=${r.simTime.toFixed(1)} pos=(${c.state.x.toFixed(1)},${c.state.z.toFixed(1)}) h=${c.state.heading.toFixed(2)} v=${c.state.speed.toFixed(2)} planAge=${c.diagnostics.planAgeMs.toFixed(0)}`);
      if (c.plan) {
        for (let i = 0; i < Math.min(30, c.plan.length); i += 4) {
          const p = c.plan[i]!;
          console.log(`  plan[${String(i).padStart(2)}] (${p.x.toFixed(1)},${p.z.toFixed(1)}) h=${p.heading.toFixed(2)} v=${p.speed.toFixed(1)} | dToCar=${Math.hypot(p.x - c.state.x, p.z - c.state.z).toFixed(1)}`);
        }
      }
      dissect(lastInfo);
      // Score hand-built maneuvers under the exact solve cost.
      const H = lastInfo.horizon;
      const mk = (steer: number, a: number): Float64Array => {
        const u = new Float64Array(H * 3);
        for (let i = 0; i < H; i++) {
          u[i * 3] = steer;
          u[i * 3 + 1] = a >= 0 ? a * 4000 : 0;
          u[i * 3 + 2] = a >= 0 ? 0 : -a * 2000;
        }
        return u;
      };
      for (const [name, u] of [
        ['hold-brake', mk(0, -0.3)],
        ['full-throttle-straight', mk(0, 1)],
        ['full-throttle-left', mk(0.6, 1)],
        ['half-throttle-left', mk(0.6, 0.5)],
        ['quarter-throttle-left', mk(0.6, 0.25)],
        ['full-throttle-right', mk(-0.6, 1)],
        ['half-throttle-right', mk(-0.6, 0.5)],
      ] as const) {
        console.log(`  manual ${name.padEnd(24)} cost=${lastInfo.scoreSequence(u).toFixed(1)}`);
      }
      // Watch replans for the next second.
      (globalThis as Record<string, unknown>).__replanLog = true;
      stopSince = r.simTime + 3;
    }
  } else {
    if ((globalThis as Record<string, unknown>).__replanLog && stopSince < 0) {
      (globalThis as Record<string, unknown>).__replanLog = false;
    }
    stopSince = -1;
  }
}
scenario.dispose();
