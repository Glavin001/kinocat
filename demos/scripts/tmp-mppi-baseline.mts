// TEMP — baseline: current MPPI (tracker:'mpc') on the race courses.
// Not committed. Usage: pnpm --filter @kinocat/demos exec tsx scripts/tmp-mppi-baseline.mts [open|technical] [pp|mpc]
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { runHeadlessRace, kinematicEntry, v2Entry, v3Entry } from '../app/lib/headless-race';
import { buildRaceCourse } from '../app/lib/race-primitives-scenarios';
import { modelFromJson } from '../app/lib/v2-model-file';
import { isV3Payload, v3FromJson } from 'kinocat/agent';

const repoRoot = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const variant = (process.argv[2] === 'technical' ? 'technical' : 'open') as 'open' | 'technical';
const tracker = process.argv[3] === 'pp' ? 'pure-pursuit' : 'mpc';

const v2Payload = JSON.parse(readFileSync(resolve(repoRoot, 'demos/public/models/v2-default.json'), 'utf-8'));
const v3Path = resolve(repoRoot, 'demos/public/models/v3-default.json');
let entries = [kinematicEntry('kinematic'), v2Entry('v2-trained', modelFromJson(v2Payload))];
try {
  const v3Payload = JSON.parse(readFileSync(v3Path, 'utf-8'));
  if (isV3Payload(v3Payload)) entries.push(v3Entry('v3', v3FromJson(v3Payload)));
} catch { /* no v3 artifact */ }

const results = await runHeadlessRace({
  entries,
  targetLaps: 2,
  maxSimTime: 180,
  course: buildRaceCourse(variant),
  tuning: { plannerBudgetMs: 10_000, tracker: tracker as 'mpc' | 'pure-pursuit' },
  onProgress: (m) => process.stdout.write(`  . ${m}\n`),
  progressEverySec: 15,
});
for (const r of results) {
  const q = r.quality;
  console.log(
    `${r.name}: finished=${r.finished} laps=${r.laps.length} avg=${r.avg.toFixed(1)}s best=${r.best.toFixed(1)}s ` +
    `wall=${r.wallStrikes} offTrack=${r.offTrackEvents} failedReplans=${r.totalReplans - r.successfulReplans} ` +
    `meanSpd=${q.meanSpeed.toFixed(1)} ggMean=${q.ggMeanUtil.toFixed(2)} stopped=${q.timeStopped.toFixed(1)}s`,
  );
}
