'use client';

// EvalHUD — renders the EvalProbe snapshots (one column per car). Plain
// component fed by throttled React state from the scene loop (the page emits at
// ~4 Hz, so React stays out of the 60 Hz loop). It displays the decomposed
// metrics: controller fidelity (cross-track vs the committed plan), plan
// feasibility, capability utilization (with a live friction-circle dot), the
// comfort flag, and the diagnosis 2×2 verdict.

import type { EvalSnapshot } from '../lib/eval-probe';
import type { Verdict } from 'kinocat/eval';

export interface EvalHUDEntry {
  label: string;
  color: number;
  snap: EvalSnapshot | null;
}

const VERDICT_COLOR: Record<Verdict, string> = {
  ok: '#5fd38a',
  controller: '#ffcf6b',
  'planner-timid': '#7fb8ff',
  'planner-infeasible': '#ff6b6b',
  both: '#ff6b6b',
};

const VERDICT_LABEL: Record<Verdict, string> = {
  ok: 'OK',
  controller: 'CONTROLLER',
  'planner-timid': 'PLANNER (timid)',
  'planner-infeasible': 'PLANNER (infeasible)',
  both: 'BOTH',
};

function hex(c: number): string {
  return '#' + c.toString(16).padStart(6, '0');
}

function fmt(n: number, d = 2): string {
  return Number.isFinite(n) ? n.toFixed(d) : '—';
}

/** Tiny friction-circle (g-g) inset: a ring at μ·g, a live dot at (a_lat, a_long). */
function GgDot({ snap }: { snap: EvalSnapshot }) {
  const R = 26; // px radius for the friction limit
  const L = snap.frictionLimit || 1;
  const x = 32 + (snap.ggNow.aLat / L) * R;
  const y = 32 - (snap.ggNow.aLong / L) * R;
  const utilPct = Math.round(snap.ggNow.util * 100);
  const dotColor = snap.ggNow.util > 1 ? '#ff6b6b' : snap.ggNow.util > 0.8 ? '#ffcf6b' : '#7fb8ff';
  return (
    <svg width={64} height={64} style={{ display: 'block' }}>
      <circle cx={32} cy={32} r={R} fill="none" stroke="#2a3447" strokeWidth={1} />
      <line x1={32} y1={32 - R} x2={32} y2={32 + R} stroke="#1f2735" strokeWidth={1} />
      <line x1={32 - R} y1={32} x2={32 + R} y2={32} stroke="#1f2735" strokeWidth={1} />
      <circle cx={Math.max(2, Math.min(62, x))} cy={Math.max(2, Math.min(62, y))} r={3} fill={dotColor} />
      <text x={32} y={61} textAnchor="middle" fontSize={8} fill="#7e8aa0">{utilPct}%</text>
    </svg>
  );
}

function Light({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
      <span
        style={{
          width: 8,
          height: 8,
          borderRadius: '50%',
          background: ok ? '#5fd38a' : '#ff6b6b',
          display: 'inline-block',
        }}
      />
      {label}
    </span>
  );
}

function Row({ k, v, warn }: { k: string; v: string; warn?: boolean }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10 }}>
      <span style={{ opacity: 0.6 }}>{k}</span>
      <span style={{ color: warn ? '#ffcf6b' : '#cdd3de', fontVariantNumeric: 'tabular-nums' }}>{v}</span>
    </div>
  );
}

function Column({ entry }: { entry: EvalHUDEntry }) {
  const s = entry.snap;
  return (
    <div style={{ minWidth: 168, padding: '4px 8px' }}>
      <div style={{ color: hex(entry.color), fontWeight: 700, marginBottom: 4 }}>{entry.label}</div>
      {!s ? (
        <div style={{ opacity: 0.5 }}>—</div>
      ) : (
        <>
          <div
            style={{
              display: 'inline-block',
              padding: '1px 6px',
              borderRadius: 4,
              background: VERDICT_COLOR[s.verdict] + '22',
              color: VERDICT_COLOR[s.verdict],
              fontWeight: 700,
              fontSize: 11,
              marginBottom: 6,
            }}
          >
            {VERDICT_LABEL[s.verdict]}
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 4 }}>
            <GgDot snap={s} />
            <div style={{ flex: 1, fontSize: 11 }}>
              <Row k="xtrack" v={`${fmt(s.crossTrackNow)} m`} warn={s.crossTrackNow > 0.5} />
              <Row k="  rmse" v={`${fmt(s.crossTrackRmse)} / ${fmt(s.crossTrackMax)}`} />
              <Row k="hdg err" v={`${fmt(s.headingErrNow)} rad`} />
            </div>
          </div>
          <Row k="plan util" v={`${fmt(s.planMeanUtil * 100, 0)}% / ${fmt(s.planPeakUtil * 100, 0)}%`} />
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 3 }}>
            <Light ok={s.planFeasible} label="feasible" />
            <Light ok={s.comfortable} label="comfy" />
          </div>
          <div style={{ marginTop: 5, opacity: 0.85, fontSize: 11 }}>
            <Row k="peak spd" v={`${fmt(s.report.peakSpeed)} m/s`} />
            <Row k="steer rev" v={`${s.report.steerReversals}`} warn={s.report.steerReversals > 40} />
            <Row k="max jerk" v={`${fmt(s.report.maxJerk, 1)} m/s³`} />
            <Row k="replans" v={`${s.report.totalReplans} (${fmt(s.report.failedReplanRatio * 100, 0)}% fail)`} />
          </div>
        </>
      )}
    </div>
  );
}

export function EvalHUD({ entries, title = 'eval' }: { entries: EvalHUDEntry[]; title?: string }) {
  return (
    <div
      style={{
        position: 'absolute',
        left: 12,
        bottom: 12,
        background: 'rgba(13, 17, 25, 0.86)',
        border: '1px solid #1f2735',
        borderRadius: 8,
        color: '#cdd3de',
        font: '12px ui-monospace, monospace',
        padding: 8,
        zIndex: 40,
        pointerEvents: 'none',
      }}
    >
      <div style={{ fontWeight: 700, color: '#9cc4ff', marginBottom: 4, paddingLeft: 8 }}>
        {title} · plan-vs-execution
      </div>
      <div style={{ display: 'flex', gap: 4 }}>
        {entries.map((e) => (
          <Column key={e.label} entry={e} />
        ))}
      </div>
    </div>
  );
}
