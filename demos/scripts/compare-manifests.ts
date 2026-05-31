// Compare two trained-model manifests side by side. Useful for
// "did the new training plan improve heading RMS at t=1 s?" without
// staring at two JSON files in different tabs.
//
// Usage: pnpm exec tsx demos/scripts/compare-manifests.ts <a.manifest.json> <b.manifest.json>
//        (path to either the .manifest.json file or the dir containing it)

import { readFileSync, existsSync, statSync } from 'node:fs';
import { resolve, basename } from 'node:path';

interface OpenLoopRow { tSec: number; posRms: number; headingRms: number; speedRms: number }
interface PerStateRms { name: string; rms: number }
interface Manifest {
  version?: number;
  profile?: string;
  totalTrials?: number;
  diagnostics?: {
    openLoopDivergence?: OpenLoopRow[];
    perStateRms?: PerStateRms[];
    baselines?: Record<string, OpenLoopRow[]>;
  };
}

function loadManifest(arg: string): { label: string; m: Manifest } {
  const abs = resolve(arg);
  let path = abs;
  if (statSync(abs).isDirectory()) {
    // Pick the first *.manifest.json in the directory.
    const { readdirSync } = require('node:fs') as typeof import('node:fs');
    const files = readdirSync(abs).filter((f) => f.endsWith('.manifest.json'));
    if (files.length === 0) {
      throw new Error(`no .manifest.json in ${abs}`);
    }
    path = `${abs}/${files[0]!}`;
  }
  if (!existsSync(path)) throw new Error(`not found: ${path}`);
  const label = basename(path).replace(/\.manifest\.json$/, '');
  return { label, m: JSON.parse(readFileSync(path, 'utf-8')) as Manifest };
}

function fmtDelta(before: number, after: number): string {
  if (!Number.isFinite(before) || !Number.isFinite(after)) return '   --- ';
  if (before === 0) return after === 0 ? '  0.0% ' : '   ∞% ';
  const pct = ((after - before) / before) * 100;
  const sign = pct >= 0 ? '+' : '';
  const colour = pct < 0 ? '✓' : pct > 5 ? '✗' : ' ';
  return `${sign}${pct.toFixed(1)}% ${colour}`;
}

function main(): void {
  const [aArg, bArg] = process.argv.slice(2);
  if (!aArg || !bArg) {
    process.stderr.write('Usage: tsx compare-manifests.ts <a.manifest.json> <b.manifest.json>\n');
    process.exit(2);
  }
  const a = loadManifest(aArg);
  const b = loadManifest(bArg);
  process.stdout.write(`A = ${a.label}  (profile=${a.m.profile ?? '?'}, trials=${a.m.totalTrials ?? '?'})\n`);
  process.stdout.write(`B = ${b.label}  (profile=${b.m.profile ?? '?'}, trials=${b.m.totalTrials ?? '?'})\n`);
  process.stdout.write(`\n`);
  process.stdout.write(`Open-loop divergence (RMS — lower is better):\n`);
  process.stdout.write(`  t=         A.pos     A.hdg     A.spd     B.pos     B.hdg     B.spd      Δpos      Δhdg      Δspd\n`);
  const aRows = a.m.diagnostics?.openLoopDivergence ?? [];
  const bRows = b.m.diagnostics?.openLoopDivergence ?? [];
  const allT = Array.from(new Set([...aRows.map((r) => r.tSec), ...bRows.map((r) => r.tSec)])).sort((x, y) => x - y);
  for (const t of allT) {
    const ar = aRows.find((r) => r.tSec === t) ?? { tSec: t, posRms: NaN, headingRms: NaN, speedRms: NaN };
    const br = bRows.find((r) => r.tSec === t) ?? { tSec: t, posRms: NaN, headingRms: NaN, speedRms: NaN };
    process.stdout.write(
      `  ${`${t.toFixed(1)}s`.padStart(5)}  `
      + `${ar.posRms.toFixed(2).padStart(8)}m `
      + `${ar.headingRms.toFixed(3).padStart(8)}r `
      + `${ar.speedRms.toFixed(2).padStart(8)}m `
      + `${br.posRms.toFixed(2).padStart(8)}m `
      + `${br.headingRms.toFixed(3).padStart(8)}r `
      + `${br.speedRms.toFixed(2).padStart(8)}m `
      + `${fmtDelta(ar.posRms, br.posRms)}  ${fmtDelta(ar.headingRms, br.headingRms)}  ${fmtDelta(ar.speedRms, br.speedRms)}\n`,
    );
  }
  process.stdout.write(`\n`);
  process.stdout.write(`Per-state RMS:\n`);
  const aPer = a.m.diagnostics?.perStateRms ?? [];
  const bPer = b.m.diagnostics?.perStateRms ?? [];
  const allN = Array.from(new Set([...aPer.map((r) => r.name), ...bPer.map((r) => r.name)]));
  for (const name of allN) {
    const ar = aPer.find((r) => r.name === name)?.rms ?? NaN;
    const br = bPer.find((r) => r.name === name)?.rms ?? NaN;
    process.stdout.write(`  ${name.padEnd(20)}  A=${ar.toFixed(3).padStart(8)}  B=${br.toFixed(3).padStart(8)}  ${fmtDelta(ar, br)}\n`);
  }
}

main();
