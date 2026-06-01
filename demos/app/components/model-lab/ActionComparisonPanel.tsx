'use client';

// Action-comparison panel for the Model Lab.
//
// Replaces the old "three fan plots side-by-side + a number strip" view, which
// forced the eye to register-align across canvases and gave every error arrow
// the same color regardless of whether it was harmless or dangerous.
//
// The redesign answers one question at a glance, per driving action:
//   "How wrong is the learned model vs the real car — and is that wrongness
//    HONEST (the model knows it's unsure / the OOD gate catches it) or
//    DANGEROUS (confidently wrong, so the bias flows into the plan)?"
//
// Three coordinated pieces:
//   1. A plain-English verdict banner (counts + one sentence).
//   2. A single overlay map: real endpoint (◎) vs model endpoint (●) joined by
//      an error line, colored by verdict. The selected action also shows its
//      swept path + the parametric→full residual correction.
//   3. A scorecard sorted worst-first: per action, a verdict badge, a dual
//      error bar (parametric vs full, so you see if the residual helped), and
//      the raw numbers. Hovering a row highlights it on the map and vice-versa.
//
// A glossary is always visible so there's no guessing what a glyph means.

import { useEffect, useMemo, useRef } from 'react';
import {
  ACCURATE_ABS_M,
  ACCURATE_PCT,
  type ActionComparison,
  type ActionComparisonSummary,
  type Verdict,
} from '../../lib/fan-plot-ground-truth';

const VERDICT_COLOR: Record<Verdict, string> = {
  accurate: '#7CFFB2',
  flagged: '#ffd479',
  'confident-bias': '#ff6b8a',
};
const VERDICT_LABEL: Record<Verdict, string> = {
  accurate: 'accurate',
  flagged: 'flagged · safe',
  'confident-bias': 'confident bias',
};
const TRUTH_COLOR = '#ffffff';

export function ActionComparisonPanel({
  summary,
  speed,
  selectedIndex,
  onSelectIndex,
}: {
  summary: ActionComparisonSummary;
  speed: number;
  selectedIndex: number | null;
  onSelectIndex: (i: number | null) => void;
}) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <VerdictBanner summary={summary} speed={speed} />
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.1fr) minmax(0, 1fr)', gap: 14, alignItems: 'start' }}>
        <ActionMap
          summary={summary}
          selectedIndex={selectedIndex}
          onSelectIndex={onSelectIndex}
        />
        <Scorecard
          summary={summary}
          selectedIndex={selectedIndex}
          onSelectIndex={onSelectIndex}
        />
      </div>
      <Glossary />
    </div>
  );
}

// ---------------------------------------------------------------------------

function VerdictBanner({ summary, speed }: { summary: ActionComparisonSummary; speed: number }) {
  const { accurate, flagged, confidentBias, count, residualHelpPct } = summary;
  const residualHurts = residualHelpPct < 0;
  const sentence =
    confidentBias > 0
      ? `${confidentBias} of ${count} actions are confidently wrong — the model is biased here and the OOD safety gate does NOT catch it. This is the failure mode that flows silently into the plan.`
      : flagged > 0
        ? `No confident-bias actions: every large miss is flagged by the OOD gate, so the planner falls back to the safe parametric model.`
        : `Every action lands close to the real car — the model is faithful at this speed.`;
  return (
    <div style={{
      border: `1px solid ${confidentBias > 0 ? '#5a2230' : '#1f2735'}`,
      background: confidentBias > 0 ? '#1a0e13' : '#0b0f17',
      borderRadius: 8, padding: '10px 14px', display: 'flex', flexDirection: 'column', gap: 8,
    }}>
      <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
        <span style={{ fontSize: 12, color: '#7fd6ff' }}>{speed} m/s · {count} actions</span>
        <Pill color={VERDICT_COLOR.accurate} label={`${accurate} accurate`} />
        <Pill color={VERDICT_COLOR.flagged} label={`${flagged} flagged · safe`} />
        <Pill color={VERDICT_COLOR['confident-bias']} label={`${confidentBias} confident bias`} strong={confidentBias > 0} />
        <span style={{ flex: 1 }} />
        <span style={{ fontSize: 11, color: residualHurts ? VERDICT_COLOR['confident-bias'] : VERDICT_COLOR.accurate }}>
          residual MLP net: {residualHurts ? '+' : '−'}{Math.abs(residualHelpPct).toFixed(0)}% error {residualHurts ? '(hurting)' : '(helping)'}
        </span>
      </div>
      <p style={{ margin: 0, fontSize: 11.5, lineHeight: 1.5, color: '#cdd3de' }}>{sentence}</p>
    </div>
  );
}

function Pill({ color, label, strong }: { color: string; label: string; strong?: boolean }) {
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 5,
      fontSize: 11, color: strong ? '#0a0d14' : color,
      background: strong ? color : 'transparent',
      border: `1px solid ${color}`, borderRadius: 12, padding: '2px 9px', fontWeight: strong ? 700 : 500,
    }}>
      <span style={{ width: 7, height: 7, borderRadius: '50%', background: strong ? '#0a0d14' : color }} />
      {label}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Overlay map — real vs model endpoints, colored by verdict.

const FOOTPRINT: [number, number][] = [[1.2, 0.6], [-1.2, 0.6], [-1.2, -0.6], [1.2, -0.6]];

function ActionMap({
  summary, selectedIndex, onSelectIndex,
}: {
  summary: ActionComparisonSummary;
  selectedIndex: number | null;
  onSelectIndex: (i: number | null) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dimsRef = useRef({ cw: 480, ch: 360 });
  const aspect = 4 / 3;

  const extent = useMemo(() => {
    let xMin = -2, xMax = 2, zMin = -2, zMax = 2;
    const grow = (x: number, z: number) => {
      if (x < xMin) xMin = x; if (x > xMax) xMax = x;
      if (z < zMin) zMin = z; if (z > zMax) zMax = z;
    };
    for (const a of summary.actions) {
      grow(a.truth.dx, a.truth.dz);
      grow(a.full.dx, a.full.dz);
      for (const s of a.sweep) grow(s.x, s.z);
    }
    const pad = 1.5;
    return { xMin: xMin - pad, xMax: xMax + pad, zMin: zMin - pad, zMax: zMax + pad };
  }, [summary]);

  const projectRef = useRef<(x: number, z: number) => [number, number]>(() => [0, 0]);

  function render() {
    const cv = canvasRef.current;
    if (!cv) return;
    const ctx = cv.getContext('2d');
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const { cw, ch } = dimsRef.current;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = '#0d1119';
    ctx.fillRect(0, 0, cw, ch);

    const xRange = extent.xMax - extent.xMin;
    const zRange = extent.zMax - extent.zMin;
    const scale = Math.min(cw / xRange, ch / zRange);
    const ox = cw / 2 - ((extent.xMin + extent.xMax) / 2) * scale;
    const oy = ch / 2 - ((extent.zMin + extent.zMax) / 2) * scale;
    const px = (x: number, z: number): [number, number] => [ox + x * scale, oy + z * scale];
    projectRef.current = px;

    // grid
    ctx.strokeStyle = '#1a2030'; ctx.lineWidth = 1; ctx.beginPath();
    for (let x = Math.ceil(extent.xMin / 5) * 5; x <= extent.xMax; x += 5) { const [sx] = px(x, 0); ctx.moveTo(sx, 0); ctx.lineTo(sx, ch); }
    for (let z = Math.ceil(extent.zMin / 5) * 5; z <= extent.zMax; z += 5) { const [, sy] = px(0, z); ctx.moveTo(0, sy); ctx.lineTo(cw, sy); }
    ctx.stroke();

    // axes + origin footprint
    const [oxA, oyA] = px(0, 0);
    ctx.strokeStyle = '#3a4458'; ctx.lineWidth = 1.5; ctx.beginPath();
    ctx.moveTo(0, oyA); ctx.lineTo(cw, oyA); ctx.moveTo(oxA, 0); ctx.lineTo(oxA, ch); ctx.stroke();
    ctx.strokeStyle = '#5566aa'; ctx.lineWidth = 1.5; ctx.beginPath();
    FOOTPRINT.forEach(([x, z], i) => { const [a, b] = px(x, z); i === 0 ? ctx.moveTo(a, b) : ctx.lineTo(a, b); });
    ctx.closePath(); ctx.stroke();

    const selected = selectedIndex != null ? summary.actions.find((a) => a.index === selectedIndex) : null;
    const dim = selected != null;

    // Selected action's full swept path + residual correction (drawn under dots).
    if (selected) {
      const col = VERDICT_COLOR[selected.verdict];
      ctx.strokeStyle = withAlpha(col, 0.9); ctx.lineWidth = 2.5; ctx.beginPath();
      selected.sweep.forEach((s, i) => { const [a, b] = px(s.x, s.z); i === 0 ? ctx.moveTo(a, b) : ctx.lineTo(a, b); });
      ctx.stroke();
      // parametric endpoint (hollow) + dashed line full→para = residual nudge
      const [fpx, fpy] = px(selected.full.dx, selected.full.dz);
      const [ppx, ppy] = px(selected.para.dx, selected.para.dz);
      ctx.strokeStyle = '#c8b6ff'; ctx.lineWidth = 1.25; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(ppx, ppy); ctx.lineTo(fpx, fpy); ctx.stroke(); ctx.setLineDash([]);
      ctx.beginPath(); ctx.arc(ppx, ppy, 4, 0, Math.PI * 2); ctx.stroke();
    }

    // Each action: error line truth↔full, truth ◎, full ●.
    for (const a of summary.actions) {
      const isSel = selected != null && a.index === selected.index;
      const alpha = dim && !isSel ? 0.28 : 1;
      const col = VERDICT_COLOR[a.verdict];
      const [fx, fy] = px(a.full.dx, a.full.dz);
      const [tx, ty] = px(a.truth.dx, a.truth.dz);
      // error line
      ctx.strokeStyle = withAlpha(col, alpha * 0.85);
      ctx.lineWidth = isSel ? 2.5 : 1.5;
      ctx.beginPath(); ctx.moveTo(fx, fy); ctx.lineTo(tx, ty); ctx.stroke();
      // truth marker (ring)
      ctx.strokeStyle = withAlpha(TRUTH_COLOR, alpha); ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(tx, ty, isSel ? 5 : 3.5, 0, Math.PI * 2); ctx.stroke();
      // model endpoint (filled)
      ctx.fillStyle = withAlpha(col, alpha);
      ctx.beginPath(); ctx.arc(fx, fy, isSel ? 5.5 : 4, 0, Math.PI * 2); ctx.fill();
    }

    ctx.strokeStyle = '#3a4458'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(oxA, oyA, 4, 0, Math.PI * 2); ctx.stroke();
  }

  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const observe = () => {
      const rect = cv.getBoundingClientRect();
      if (rect.width <= 0) return;
      const cw = Math.round(rect.width);
      const ch = Math.round(rect.width / aspect);
      dimsRef.current = { cw, ch };
      cv.width = cw * (window.devicePixelRatio || 1);
      cv.height = ch * (window.devicePixelRatio || 1);
      cv.style.height = `${ch}px`;
      render();
    };
    const ro = new ResizeObserver(observe);
    ro.observe(cv);
    observe();
    return () => ro.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(render); // eslint-disable-line react-hooks/exhaustive-deps

  function handleMove(e: React.MouseEvent<HTMLCanvasElement>) {
    const cv = canvasRef.current;
    if (!cv) return;
    const rect = cv.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const px = projectRef.current;
    let best = -1; let bestD = 14;
    for (const a of summary.actions) {
      const [fx, fy] = px(a.full.dx, a.full.dz);
      const d = Math.hypot(mx - fx, my - fy);
      if (d < bestD) { bestD = d; best = a.index; }
    }
    onSelectIndex(best >= 0 ? best : null);
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <canvas
        ref={canvasRef}
        onMouseMove={handleMove}
        onMouseLeave={() => onSelectIndex(null)}
        style={{ width: '100%', aspectRatio: `${aspect}`, borderRadius: 6, display: 'block', cursor: 'crosshair' }}
      />
      <div style={{ display: 'flex', gap: 14, fontSize: 10.5, opacity: 0.75, flexWrap: 'wrap' }}>
        <LegendGlyph><Ring /> real car (Rapier)</LegendGlyph>
        <LegendGlyph><Dot color="#9aa6b2" /> model prediction</LegendGlyph>
        <LegendGlyph><span style={{ width: 16, height: 0, borderTop: '1.5px solid #9aa6b2', display: 'inline-block' }} /> error</LegendGlyph>
        <LegendGlyph><Dot color="#c8b6ff" hollow /> parametric (selected)</LegendGlyph>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Scorecard — one row per action, sorted worst-first.

function Scorecard({
  summary, selectedIndex, onSelectIndex,
}: {
  summary: ActionComparisonSummary;
  selectedIndex: number | null;
  onSelectIndex: (i: number | null) => void;
}) {
  const scale = Math.max(summary.maxErrM, 0.01);
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 3, maxHeight: 360, overflowY: 'auto' }}>
      {summary.actions.map((a) => (
        <ScoreRow
          key={a.index}
          a={a}
          scale={scale}
          selected={a.index === selectedIndex}
          onSelect={() => onSelectIndex(a.index)}
          onClear={() => onSelectIndex(null)}
        />
      ))}
    </div>
  );
}

function ScoreRow({
  a, scale, selected, onSelect, onClear,
}: {
  a: ActionComparison;
  scale: number;
  selected: boolean;
  onSelect: () => void;
  onClear: () => void;
}) {
  const col = VERDICT_COLOR[a.verdict];
  const fullPct = Math.min(100, (a.fullErrM / scale) * 100);
  const paraPct = Math.min(100, (a.paraErrM / scale) * 100);
  const residualHurts = a.residualDeltaPct < 0;
  return (
    <div
      onMouseEnter={onSelect}
      onMouseLeave={onClear}
      onClick={onSelect}
      style={{
        display: 'flex', flexDirection: 'column', gap: 4, padding: '6px 8px', borderRadius: 5,
        background: selected ? '#141a26' : 'transparent',
        border: `1px solid ${selected ? col : 'transparent'}`,
        cursor: 'pointer',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11.5 }}>
        <span style={{ width: 9, height: 9, borderRadius: '50%', background: col, flexShrink: 0 }}
          title={VERDICT_LABEL[a.verdict]} />
        <span style={{ color: '#e3e7ee', minWidth: 96 }}>{a.label}</span>
        {a.verdict === 'confident-bias' && (
          <span style={{ color: col, fontSize: 10 }} title="wrong, and the OOD gate did not fire">⚠ bias</span>
        )}
        <span style={{ flex: 1 }} />
        <span style={{ color: col, fontVariantNumeric: 'tabular-nums', fontWeight: 600 }}>{a.fullErrM.toFixed(2)} m</span>
        <span style={{ opacity: 0.5, fontSize: 10 }} title="OOD ensemble gate">{a.gate ? 'gate ON' : 'gate off'}</span>
      </div>
      {/* dual error bar: parametric tick over full bar, shared scale */}
      <div style={{ position: 'relative', height: 8, background: '#11161f', borderRadius: 4, overflow: 'hidden' }}>
        <div style={{ position: 'absolute', left: 0, top: 0, height: '100%', width: `${fullPct}%`, background: withAlpha(col, 0.55) }} />
        <div title={`parametric error ${a.paraErrM.toFixed(2)} m`} style={{
          position: 'absolute', left: `calc(${paraPct}% - 1px)`, top: -1, height: 10, width: 2, background: '#c8b6ff',
        }} />
      </div>
      <div style={{ display: 'flex', gap: 10, fontSize: 9.5, opacity: 0.6 }}>
        <span style={{ color: residualHurts ? VERDICT_COLOR['confident-bias'] : VERDICT_COLOR.accurate }}>
          residual {residualHurts ? '+' : '−'}{Math.abs(a.residualDeltaPct).toFixed(0)}% {residualHurts ? '(worse)' : '(better)'}
        </span>
        <span>vs parametric {a.paraErrM.toFixed(2)} m</span>
        <span>σ {a.ensSigmaPos.toFixed(2)}</span>
        <span>travel {a.travelM.toFixed(1)} m</span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------

function Glossary() {
  return (
    <div style={{
      border: '1px solid #1f2735', borderRadius: 6, padding: '10px 14px',
      display: 'flex', flexDirection: 'column', gap: 6, fontSize: 11, lineHeight: 1.5,
    }}>
      <span style={{ color: '#7f8a99', fontSize: 10.5, letterSpacing: 0.4 }}>HOW TO READ THIS</span>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 10 }}>
        <VerdictDef
          color={VERDICT_COLOR.accurate}
          title="accurate"
          body={`Model endpoint lands within ${ACCURATE_ABS_M} m of the real car (or ${(ACCURATE_PCT * 100).toFixed(0)}% of how far it travelled). The primitive is trustworthy.`}
        />
        <VerdictDef
          color={VERDICT_COLOR.flagged}
          title="flagged · safe"
          body="A bigger miss, BUT the ensemble disagrees (OOD gate ON), so at runtime the planner falls back to the safe parametric model. The system knows it's unsure — honest."
        />
        <VerdictDef
          color={VERDICT_COLOR['confident-bias']}
          title="confident bias"
          body="A bigger miss AND the ensemble agreed (gate OFF). The wrong prediction is delivered confidently and flows straight into the plan. This is the dangerous case."
        />
      </div>
      <p style={{ margin: '2px 0 0', opacity: 0.6 }}>
        The <strong style={{ color: '#c8b6ff' }}>violet tick</strong> on each bar is the
        parametric-only (residual-stripped) error — the safety floor. If the colored
        bar is shorter than the tick, the residual MLP <em>helped</em>; if it extends past
        the tick, the residual <em>made that action worse</em>. Endpoints are in the
        chassis's start-local frame (it sits at the origin facing right), measured after
        the suspension settles so the settle-coast doesn't inflate the error.
      </p>
    </div>
  );
}

function VerdictDef({ color, title, body }: { color: string; title: string; body: string }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, color, fontWeight: 600 }}>
        <span style={{ width: 9, height: 9, borderRadius: '50%', background: color }} /> {title}
      </span>
      <span style={{ opacity: 0.7 }}>{body}</span>
    </div>
  );
}

function LegendGlyph({ children }: { children: React.ReactNode }) {
  return <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>{children}</span>;
}
function Ring() {
  return <span style={{ width: 9, height: 9, borderRadius: '50%', border: '1.5px solid #ffffff', display: 'inline-block' }} />;
}
function Dot({ color, hollow }: { color: string; hollow?: boolean }) {
  return <span style={{ width: 9, height: 9, borderRadius: '50%', background: hollow ? 'transparent' : color, border: hollow ? `1.5px solid ${color}` : 'none', display: 'inline-block' }} />;
}

function withAlpha(hex: string, a: number): string {
  if (hex.startsWith('#') && (hex.length === 7 || hex.length === 4)) {
    const r = parseInt(hex.length === 7 ? hex.slice(1, 3) : hex[1]! + hex[1], 16);
    const g = parseInt(hex.length === 7 ? hex.slice(3, 5) : hex[2]! + hex[2], 16);
    const b = parseInt(hex.length === 7 ? hex.slice(5, 7) : hex[3]! + hex[3], 16);
    return `rgba(${r}, ${g}, ${b}, ${a})`;
  }
  return hex;
}
