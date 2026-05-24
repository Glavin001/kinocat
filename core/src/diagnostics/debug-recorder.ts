// Generic rolling debug recorder.
//
// Each tick the scene controller emits a `RawFrame<S, C>` (real state +
// applied controls + per-ghost predicted state + an arbitrary extras blob
// for domain-specific telemetry — e.g. wheel forces for a car, lift/drag
// for an airplane). The recorder rings the last N frames and on demand
// serializes them to JSON or Markdown.
//
// Domain-agnostic: the recorder has no idea what "x" or "speed" or "wheel"
// means. The caller supplies a `formatState` projection (S -> Record) and a
// `formatControls` projection (C -> Record) so the JSON/Markdown output is
// nicely-named without leaking the recorder's generics into the car/airplane
// specifics.
//
// Aggregation (mean / RMS gap) is also generic: the caller supplies a
// `gapMetrics` function that, for any (real, predicted) pair, returns a
// flat number record. We compute mean + RMS over the ring buffer per metric.

import type { GhostStepResult, RecorderHook } from '../scene';

/** One captured tick. */
export interface RawFrame<S, C> {
  simTime: number;
  real: S;
  controls: C;
  ghosts: ReadonlyArray<GhostStepResult<S>>;
  /** Domain-specific extras (e.g. wheel telemetry, aero coefficients). */
  extras?: Record<string, unknown>;
}

/** Format projections + gap metric. */
export interface RecorderFormatters<S, C> {
  /** S -> {key: number}. Used in JSON / Markdown. */
  formatState: (s: S) => Record<string, number>;
  /** C -> {key: number}. Used in JSON / Markdown + controls-range stats. */
  formatControls: (c: C) => Record<string, number>;
  /** (real, predicted) -> {metricName: signedDelta}. Used for aggregate
   *  mean + RMS. The recorder pushes the result through abs() for RMS. */
  gapMetrics?: (real: S, predicted: S) => Record<string, number>;
}

export interface RecorderMeta {
  /** Free-form label for the run ("playback", "free-drive", ...). */
  mode: string;
  /** Physics tick the body was stepped at (s). */
  physicsDt: number;
  /** Anything else the demo wants to surface in the export. */
  [k: string]: unknown;
}

export interface DebugStats {
  ticks: number;
  durationSec: number;
  /** Aggregate gap per ghost: mean signed + RMS per metric. */
  perGhost: Record<
    string,
    Record<string, { mean: number; rms: number; final: number }>
  >;
  /** min/max of each control key over the buffer. */
  controlsRange: Record<string, { min: number; max: number }>;
}

export class DebugRecorder<S, C> implements RecorderHook<S, C> {
  private buf: RawFrame<S, C>[] = [];
  private capacity: number;
  private formatters: RecorderFormatters<S, C>;
  /** Sidecar data attached to the next captured frame. Cleared after capture. */
  private pendingExtras: Record<string, unknown> | undefined;

  constructor(opts: {
    capacity?: number;
    formatters: RecorderFormatters<S, C>;
  }) {
    this.capacity = opts.capacity ?? 600;
    this.formatters = opts.formatters;
  }

  /** Schedule extras to be attached to the next `capture()` call. The
   *  scene controller calls `capture` for us, so the demo uses this to
   *  inject per-frame telemetry that lives outside the generic types. */
  attachExtras(extras: Record<string, unknown>): void {
    this.pendingExtras = extras;
  }

  capture(frame: {
    simTime: number;
    real: S;
    controls: C;
    ghosts: ReadonlyArray<GhostStepResult<S>>;
  }): void {
    const f: RawFrame<S, C> = {
      simTime: frame.simTime,
      real: frame.real,
      controls: frame.controls,
      ghosts: frame.ghosts,
      extras: this.pendingExtras,
    };
    this.pendingExtras = undefined;
    if (this.buf.length >= this.capacity) this.buf.shift();
    this.buf.push(f);
  }

  clear(): void {
    this.buf = [];
    this.pendingExtras = undefined;
  }

  size(): number {
    return this.buf.length;
  }

  frames(): ReadonlyArray<RawFrame<S, C>> {
    return this.buf;
  }

  tail(n: number): RawFrame<S, C>[] {
    return this.buf.slice(Math.max(0, this.buf.length - n));
  }

  /** Aggregate mean + RMS gap per ghost, plus controls range. */
  stats(): DebugStats {
    const perGhost: DebugStats['perGhost'] = {};
    const controlsRange: DebugStats['controlsRange'] = {};
    if (this.buf.length === 0) {
      return { ticks: 0, durationSec: 0, perGhost, controlsRange };
    }
    const ghostNames = new Set<string>();
    for (const f of this.buf) for (const g of f.ghosts) ghostNames.add(g.name);

    const gapFn = this.formatters.gapMetrics;
    for (const name of ghostNames) {
      const sums: Record<string, number> = {};
      const sqs: Record<string, number> = {};
      let n = 0;
      let lastDeltas: Record<string, number> = {};
      for (const f of this.buf) {
        const g = f.ghosts.find((x) => x.name === name);
        if (!g) continue;
        const delta = gapFn ? gapFn(f.real, g.state) : {};
        for (const [k, v] of Object.entries(delta)) {
          sums[k] = (sums[k] ?? 0) + v;
          sqs[k] = (sqs[k] ?? 0) + v * v;
        }
        lastDeltas = delta;
        n++;
      }
      const m: Record<string, { mean: number; rms: number; final: number }> = {};
      const denom = Math.max(1, n);
      for (const k of Object.keys(sums)) {
        m[k] = {
          mean: sums[k]! / denom,
          rms: Math.sqrt(sqs[k]! / denom),
          final: lastDeltas[k] ?? 0,
        };
      }
      perGhost[name] = m;
    }
    for (const f of this.buf) {
      const ctrl = this.formatters.formatControls(f.controls);
      for (const [k, v] of Object.entries(ctrl)) {
        const r = controlsRange[k] ?? { min: Infinity, max: -Infinity };
        if (v < r.min) r.min = v;
        if (v > r.max) r.max = v;
        controlsRange[k] = r;
      }
    }
    return {
      ticks: this.buf.length,
      durationSec: this.buf[this.buf.length - 1]!.simTime - this.buf[0]!.simTime,
      perGhost,
      controlsRange,
    };
  }

  /** Serialize the buffer + stats + meta to a JSON string. Numbers are
   *  rounded to 4 decimals for size. */
  toJSON(meta: RecorderMeta, tailFrames?: number): string {
    const slice = tailFrames !== undefined ? this.tail(tailFrames) : this.buf;
    const stats = this.stats();
    const round = (x: number, n = 4): number =>
      Number.isFinite(x) ? Number(x.toFixed(n)) : x;
    const roundRec = (r: Record<string, number>): Record<string, number> => {
      const out: Record<string, number> = {};
      for (const [k, v] of Object.entries(r)) out[k] = round(v);
      return out;
    };
    const frames = slice.map((f) => ({
      t: round(f.simTime),
      real: roundRec(this.formatters.formatState(f.real)),
      controls: roundRec(this.formatters.formatControls(f.controls)),
      ghosts: Object.fromEntries(
        f.ghosts.map((g) => [g.name, roundRec(this.formatters.formatState(g.state))]),
      ),
      ...(f.extras ? { extras: f.extras } : {}),
    }));
    return JSON.stringify({ meta, stats, frames }, null, 2);
  }

  /** Markdown summary: header, per-ghost gap table, controls range, then
   *  a fenced JSON tail with the last `tailFrames` frames. */
  toMarkdown(meta: RecorderMeta, tailFrames = 30): string {
    const stats = this.stats();
    const fmt = (x: number, n = 3): string =>
      Number.isFinite(x) ? x.toFixed(n) : String(x);
    const lines: string[] = [];
    lines.push(`# Debug snapshot — ${meta.mode}`);
    lines.push('');
    lines.push(`- frames: ${stats.ticks} (${fmt(stats.durationSec, 2)} s)`);
    lines.push(`- physics dt: ${meta.physicsDt}`);
    for (const [k, v] of Object.entries(meta)) {
      if (k === 'mode' || k === 'physicsDt') continue;
      lines.push(`- ${k}: ${typeof v === 'object' ? JSON.stringify(v) : String(v)}`);
    }
    lines.push('');
    lines.push('## Per-ghost gap (over capture window)');
    lines.push('');
    const metricNames = new Set<string>();
    for (const g of Object.values(stats.perGhost)) for (const k of Object.keys(g)) metricNames.add(k);
    const metricList = Array.from(metricNames);
    if (metricList.length > 0) {
      const header = `| ghost | ${metricList.map((m) => `mean ${m} | rms ${m} | final ${m}`).join(' | ')} |`;
      const sep = `|---|${metricList.map(() => '---:|---:|---:').join('|')}|`;
      lines.push(header);
      lines.push(sep);
      for (const [name, mets] of Object.entries(stats.perGhost)) {
        const cells = metricList.map((m) => {
          const r = mets[m];
          return r ? `${fmt(r.mean)} | ${fmt(r.rms)} | ${fmt(r.final)}` : ` |  | `;
        });
        lines.push(`| \`${name}\` | ${cells.join(' | ')} |`);
      }
    } else {
      lines.push('_(no gap metrics supplied)_');
    }
    lines.push('');
    lines.push('## Controls range');
    lines.push('');
    for (const [k, r] of Object.entries(stats.controlsRange)) {
      lines.push(`- ${k}: [${fmt(r.min)}, ${fmt(r.max)}]`);
    }
    lines.push('');
    lines.push(`## Last ${Math.min(tailFrames, this.buf.length)} frames (JSON)`);
    lines.push('');
    lines.push('```json');
    lines.push(this.toJSON(meta, tailFrames));
    lines.push('```');
    return lines.join('\n');
  }
}
