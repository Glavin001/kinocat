'use client';

// Top-of-dashboard scorecard. Answers "how trained is the model RIGHT
// NOW, and how much of that is parametric vs the residual MLP?" at a
// glance. The user requested the residual MLP contribution be made
// visible — it's the difference between `parametricOnly` and the full
// `v2` open-loop divergence rows.

import type { ModelDiagnostics } from 'kinocat/learning';

export interface HeadlineStripProps {
  /** Final-round diagnostics (after both parametric fit and residual MLP). */
  diag: ModelDiagnostics | null;
  /** Cached meta from the persisted model — used when the user reloaded
   *  the page and the in-memory diag is empty but a model still exists. */
  fallback?: {
    openLoopRmsAt1s: number;
    legacyRmsAt1s?: number;
    kinematicRmsAt1s?: number;
    trialsUsed: number;
  } | null;
}

const TARGET_HORIZON = 1.0;

function at(rows: { tSec: number; posRms: number }[] | undefined, t = TARGET_HORIZON): number | null {
  if (!rows || rows.length === 0) return null;
  return rows.find((r) => r.tSec >= t)?.posRms ?? null;
}

export function HeadlineStrip({ diag, fallback }: HeadlineStripProps) {
  if (!diag && !fallback) {
    return (
      <div style={emptyStyle}>
        Train a model to see headline numbers, or load one from the existing cache (top right).
      </div>
    );
  }

  const v2 = diag ? at(diag.openLoopDivergence) ?? 0 : fallback!.openLoopRmsAt1s;
  const parametricOnly = diag ? at(diag.baselines['parametricOnly']) : null;
  const legacy = diag ? at(diag.baselines['legacyV1']) : fallback?.legacyRmsAt1s ?? null;
  const kin = diag ? at(diag.baselines['kinematic']) : fallback?.kinematicRmsAt1s ?? null;
  const residualGain = parametricOnly !== null && parametricOnly > 0
    ? (1 - v2 / parametricOnly) * 100
    : null;
  const vsLegacy = legacy && legacy > 0 ? (1 - v2 / legacy) * 100 : null;

  return (
    <div style={wrapStyle}>
      <Card color="#55dcff" label="v2 full @ 1s" value={`${v2.toFixed(3)} m`} hint="parametric + residual MLP" />
      {parametricOnly !== null && (
        <Card
          color="#a6e9ff"
          label="parametric-only"
          value={`${parametricOnly.toFixed(3)} m`}
          hint={residualGain !== null ? `MLP saves ${residualGain.toFixed(1)}%` : 'no residual yet'}
        />
      )}
      {legacy !== null && (
        <Card
          color="#ff8aa0"
          label="legacy 5-param"
          value={`${legacy.toFixed(3)} m`}
          hint={vsLegacy !== null ? `${vsLegacy.toFixed(1)}% better` : undefined}
        />
      )}
      {kin !== null && (
        <Card color="#ffd070" label="kinematic" value={`${kin.toFixed(3)} m`} hint="planner-blind baseline" />
      )}
      {/* Sweet spot: residualGain green when meaningful, vsLegacy emphatic. */}
      {vsLegacy !== null && (
        <Card
          color={vsLegacy > 0 ? '#55ff88' : '#ff5566'}
          label="vs legacy"
          value={`${vsLegacy >= 0 ? '+' : ''}${vsLegacy.toFixed(1)}%`}
          big
        />
      )}
    </div>
  );
}

function Card({
  color, label, value, hint, big,
}: { color: string; label: string; value: string; hint?: string; big?: boolean }) {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', gap: 2,
      padding: '10px 14px', minWidth: big ? 130 : 110,
      borderRadius: 6,
      background: 'rgba(13, 17, 25, 0.85)',
      border: `1px solid ${color}55`,
      boxShadow: big ? `0 0 16px ${color}33` : 'none',
    }}>
      <span style={{ fontSize: 10, opacity: 0.7, letterSpacing: 0.4, textTransform: 'uppercase' }}>{label}</span>
      <span style={{ color, fontWeight: 700, fontSize: big ? 20 : 16 }}>{value}</span>
      {hint && <span style={{ opacity: 0.5, fontSize: 10 }}>{hint}</span>}
    </div>
  );
}

const wrapStyle: React.CSSProperties = {
  display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'stretch',
};

const emptyStyle: React.CSSProperties = {
  padding: '14px 18px',
  borderRadius: 6,
  background: 'rgba(13, 17, 25, 0.85)',
  border: '1px solid #1f2735',
  opacity: 0.7,
  fontSize: 12,
};
