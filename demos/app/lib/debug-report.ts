// Markdown debug-report generator for the /raceprimitives page. Bundles
// everything you'd otherwise screenshot or read off six different panels
// into a single sharable document: model state, current race state, per-
// car planner stats, full primitive-library diagnostics, and system info.
//
// Used by the "Export debug" button in the TopBar. Output is both copied
// to the clipboard and downloaded as a .md file so it can be pasted into
// a chat or saved for later comparison.

import type {
  LearnedVehicleModel,
  LearnedVehicleParamsV2,
  LearnableVehicleConfig,
} from 'kinocat/agent';
import type { MotionPrimitiveLibrary } from 'kinocat/primitives';
import { diagnoseLibrary } from './primitive-diagnostics';
import type { PersistedV2Model } from './v2-model-persistence';
import type { RaceMetrics } from './race-primitives-scenarios';

export interface DebugReportArgs {
  // Page / scene state
  phase: string;
  useV2: boolean;
  v2Active: boolean;
  winner: 'kinematic' | 'learned' | 'tie' | null;
  // v2 model + meta (null if not loaded)
  v2Model: LearnedVehicleModel | null;
  v2Meta: PersistedV2Model['meta'] | null;
  // Per-car metrics + lap data
  kinematicMetrics: RaceMetrics;
  learnedMetrics: RaceMetrics;
  kinematicLapTimes: number[];
  learnedLapTimes: number[];
  kinematicSectors: number[][];
  learnedSectors: number[][];
  // Course
  waypointCount: number;
  // Libraries — for fan-resolution diagnostics
  kinematicLibrary: MotionPrimitiveLibrary;
  learnedLibrary: MotionPrimitiveLibrary | null; // null when v2 not loaded
  startSpeeds: number[];
  // Planner config snapshot
  plannerConfig: {
    lookaheadCount: number;
    replanIntervalMs: number;
    perCarBudgetMs: number;
    plannerGateRadius: number;
    advanceRadius: number;
    trackerMaxLateralAccel: number;
  };
  // User note (free-text) — caller can prepend "what was happening when
  // you exported this". Optional.
  note?: string;
}

export function buildDebugReport(args: DebugReportArgs): string {
  const now = new Date();
  const lines: string[] = [];
  const push = (s: string) => lines.push(s);
  const tableHeader = (cols: string[]) => {
    push('| ' + cols.join(' | ') + ' |');
    push('|' + cols.map(() => '---').join('|') + '|');
  };

  push('# kinocat /raceprimitives debug report');
  push('');
  push(`**Generated:** ${now.toISOString()}`);
  if (typeof window !== 'undefined') {
    push(`**URL:** ${window.location.href}`);
    push(`**User-agent:** ${navigator.userAgent}`);
    push(`**Viewport:** ${window.innerWidth} × ${window.innerHeight} · dpr ${window.devicePixelRatio || 1}`);
  }
  if (args.note) {
    push('');
    push('## Note');
    push(args.note);
  }

  // Phase
  push('');
  push('## Phase');
  push(`- **Phase:** \`${args.phase}\``);
  push(`- **v2 toggle:** ${args.useV2 ? 'ON' : 'off'}${args.v2Active ? ' · v2 active' : ''}`);
  push(`- **Winner:** ${args.winner ?? '—'}`);

  // v2 model
  push('');
  push('## v2 model');
  if (args.v2Model && args.v2Meta) {
    const m = args.v2Meta;
    push('- **Loaded:** yes');
    push(`- **Trials used:** ${m.trialsUsed}`);
    push(`- **Open-loop RMS @ 1 s:** ${m.openLoopRmsAt1s.toFixed(3)} m`);
    if (m.legacyRmsAt1s !== undefined) {
      const delta = m.legacyRmsAt1s > 0 ? ((1 - m.openLoopRmsAt1s / m.legacyRmsAt1s) * 100).toFixed(1) : '—';
      push(`- **vs legacy 5-param:** ${m.legacyRmsAt1s.toFixed(3)} m (${delta}% better)`);
    }
    if (m.kinematicRmsAt1s !== undefined) {
      push(`- **vs kinematic:** ${m.kinematicRmsAt1s.toFixed(3)} m`);
    }
    push(`- **Trained:** ${new Date(m.createdAt).toISOString()}`);
    push('');
    push('### v2 parameters');
    tableHeader(['coefficient', 'value']);
    const p = args.v2Model.params as unknown as Record<string, number>;
    for (const k of Object.keys(p)) {
      const v = p[k]!;
      const s = Math.abs(v) >= 1 || v === 0 ? v.toFixed(4) : v.toExponential(3);
      push(`| ${k} | ${s} |`);
    }
    push('');
    push('### v2 vehicle config');
    tableHeader(['field', 'value']);
    const c = args.v2Model.config as unknown as Record<string, number | string>;
    for (const k of Object.keys(c)) {
      const v = c[k];
      const s = typeof v === 'number'
        ? (Math.abs(v) >= 1 || v === 0 ? v.toFixed(3) : v.toExponential(3))
        : String(v);
      push(`| ${k} | ${s} |`);
    }
  } else {
    push('- **Loaded:** no (train one in Model Lab to populate this section)');
  }

  // Race state
  push('');
  push('## Race');
  push(`- **Course waypoints:** ${args.waypointCount}`);
  push(`- **Race time (kinematic / learned):** ${args.kinematicMetrics.raceTime.toFixed(2)} s / ${args.learnedMetrics.raceTime.toFixed(2)} s`);

  // Per-car blocks
  for (const car of [
    { name: 'KINEMATIC (pink)', m: args.kinematicMetrics, laps: args.kinematicLapTimes, sectors: args.kinematicSectors },
    { name: 'LEARNED (cyan, v2)', m: args.learnedMetrics, laps: args.learnedLapTimes, sectors: args.learnedSectors },
  ]) {
    push('');
    push(`### ${car.name}`);
    push(`- **Laps:** ${car.m.laps}`);
    push(`- **Waypoints cleared:** ${car.m.waypointsCleared}`);
    push(`- **Best lap:** ${Number.isFinite(car.m.bestLapTime) ? `${car.m.bestLapTime.toFixed(2)} s` : '—'}`);
    push(`- **Last lap:** ${Number.isFinite(car.m.lastLapTime) ? `${car.m.lastLapTime.toFixed(2)} s` : '—'}`);
    const last5 = car.laps.slice(-5);
    if (last5.length > 0) {
      const mean = last5.reduce((a, b) => a + b, 0) / last5.length;
      push(`- **Mean (last 5):** ${mean.toFixed(2)} s`);
    }
    push(`- **Lap history:** ${car.laps.length === 0 ? '[]' : '[' + car.laps.map((l) => l.toFixed(2)).join(', ') + ']'}`);
    push(`- **0.55 s pred err (RMS):** ${car.m.trackingErrorRms.toFixed(3)} m`);
    push(`- **Peak speed:** ${car.m.peakSpeed.toFixed(1)} m/s`);
    push(`- **Live controls:** steer=${car.m.liveControls.steer.toFixed(3)} rad · throttle=${(car.m.liveControls.throttle * 100).toFixed(0)}% · brake=${(car.m.liveControls.brake * 100).toFixed(0)}% · target=${car.m.liveControls.targetSpeed.toFixed(1)} m/s`);
    const d = car.m.planDiagnostics;
    push(`- **Planner:** last replan ${d.lastReplanMs.toFixed(0)} ms (${d.lastReplanFound ? 'ok' : 'FAIL'}) · plan age ${d.planAgeMs.toFixed(0)} ms · success ${d.totalReplans > 0 ? `${((d.successfulReplans / d.totalReplans) * 100).toFixed(0)}% (${d.successfulReplans}/${d.totalReplans})` : '—'}${d.consecutiveFailedReplans > 0 ? ` · **failed streak ${d.consecutiveFailedReplans}**` : ''}`);
    if (car.sectors.length > 0) {
      push(`- **Sector times** (lap × gate-cumulative-sec):`);
      for (let li = 0; li < car.sectors.length; li++) {
        push(`  - lap ${li + 1}: [${car.sectors[li]!.map((s) => s.toFixed(2)).join(', ')}]`);
      }
    }
  }

  // Primitive library diagnostics
  push('');
  push('## Primitive libraries (action-space resolution)');
  const cols = ['speed', 'prims', 'fwd', 'rev', 'hull (m²)', 'max gap (°)', 'fwd x-span (m)', 'fwd z-span (m)'];
  push('');
  push('### Kinematic library');
  tableHeader(cols);
  for (const s of args.startSpeeds) {
    const prims = args.kinematicLibrary.lookup(s);
    const d = diagnoseLibrary(prims);
    const xSpan = (d.forwardEndpointBBox.xMax - d.forwardEndpointBBox.xMin).toFixed(2);
    const zSpan = (d.forwardEndpointBBox.zMax - d.forwardEndpointBBox.zMin).toFixed(2);
    push(`| ${s} | ${d.count} | ${d.forwardCount} | ${d.reverseCount} | ${d.hullAreaM2.toFixed(2)} | ${d.maxAngularGapDeg.toFixed(1)} | ${xSpan} | ${zSpan} |`);
  }
  if (args.learnedLibrary) {
    push('');
    push('### v2-learned library');
    tableHeader(cols);
    for (const s of args.startSpeeds) {
      const prims = args.learnedLibrary.lookup(s);
      const d = diagnoseLibrary(prims);
      const xSpan = (d.forwardEndpointBBox.xMax - d.forwardEndpointBBox.xMin).toFixed(2);
      const zSpan = (d.forwardEndpointBBox.zMax - d.forwardEndpointBBox.zMin).toFixed(2);
      push(`| ${s} | ${d.count} | ${d.forwardCount} | ${d.reverseCount} | ${d.hullAreaM2.toFixed(2)} | ${d.maxAngularGapDeg.toFixed(1)} | ${xSpan} | ${zSpan} |`);
    }
    push('');
    push('### v2 vs kinematic hull ratio (at each speed bucket)');
    tableHeader(['speed', 'kin hull (m²)', 'v2 hull (m²)', 'ratio v2/kin']);
    for (const s of args.startSpeeds) {
      const k = diagnoseLibrary(args.kinematicLibrary.lookup(s));
      const v = diagnoseLibrary(args.learnedLibrary.lookup(s));
      const ratio = k.hullAreaM2 > 0 ? (v.hullAreaM2 / k.hullAreaM2).toFixed(3) : '—';
      push(`| ${s} | ${k.hullAreaM2.toFixed(2)} | ${v.hullAreaM2.toFixed(2)} | ${ratio} |`);
    }
  }

  // Planner config
  push('');
  push('## Planner config');
  const pc = args.plannerConfig;
  push(`- **Lookahead:** ${pc.lookaheadCount} gates (multi-goal A*)`);
  push(`- **Replan interval:** ${pc.replanIntervalMs} ms`);
  push(`- **Per-car deadline budget:** ${pc.perCarBudgetMs} ms`);
  push(`- **Planner gate radius:** ${pc.plannerGateRadius} m`);
  push(`- **Demo advance radius:** ${pc.advanceRadius} m (planner < advance ⇒ valid plans always reach the advance circle)`);
  push(`- **Pure-pursuit max lateral accel:** ${pc.trackerMaxLateralAccel} m/s²`);

  // System
  push('');
  push('## System');
  if (typeof navigator !== 'undefined') {
    const nav = navigator as Navigator & { deviceMemory?: number };
    push(`- **Hardware concurrency:** ${navigator.hardwareConcurrency ?? '—'}`);
    if (nav.deviceMemory) push(`- **Device memory:** ${nav.deviceMemory} GB`);
    push(`- **Language:** ${navigator.language}`);
    push(`- **Online:** ${navigator.onLine}`);
  }

  return lines.join('\n') + '\n';
}

/** Try to copy text to the clipboard; resolves true on success. */
export async function copyToClipboard(text: string): Promise<boolean> {
  if (typeof navigator === 'undefined' || !navigator.clipboard) return false;
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}

/** Trigger a .md download for the supplied text. */
export function downloadMarkdown(text: string, filename: string): void {
  if (typeof window === 'undefined') return;
  const blob = new Blob([text], { type: 'text/markdown' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
