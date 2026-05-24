'use client';

// Compact training controls panel: sliders for rounds/trials/ticks/seed,
// Train/Cancel buttons, and a live loss + per-round-progress strip. The
// dashboard places this in the page header so training observability is
// the FIRST thing the user sees rather than buried in a panel.

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { useEffect, useMemo, useState } from 'react';
import { useModelLab } from '../../lib/model-lab-store';

export function TrainingControls() {
  const { state, setSettings, train, cancel, clearTrained } = useModelLab();
  const {
    settings, status, currentRound, rounds,
    trialsCollected, trialsDiscarded,
    trialsAttemptedThisRound, trialsTargetThisRound,
    latestLoss, currentPhase, currentIter, currentIterTotal,
    trainingStartedAt,
    liveFitCurve, error, model,
  } = state;

  const running = status === 'running';

  // Build the chart data using the per-sample normalized loss so values
  // are comparable across rounds (raw parametric loss is a SUM that
  // doubles every round as the trial store grows) AND across phases
  // (residual MLP loss is already per-sample). Split into one series
  // per (round, phase) so each segment renders as its own line — that
  // way the chart shows a sawtooth of per-round descents instead of a
  // single misleading line that jumps between scales.
  const { lossSeries, roundBoundaries } = useMemo(() => {
    const series: {
      key: string;
      round: number;
      phase: 'parametric' | 'residual';
      points: { x: number; loss: number }[];
    }[] = [];
    const boundaries: number[] = [];
    let lastRound = -1;
    let lastPhase: 'parametric' | 'residual' | null = null;
    let cur: typeof series[number] | null = null;
    liveFitCurve.forEach((p, i) => {
      if (p.round !== lastRound) {
        if (i > 0) boundaries.push(i);
        lastRound = p.round;
      }
      if (cur === null || cur.round !== p.round || cur.phase !== p.phase) {
        cur = {
          key: `r${p.round}-${p.phase}`,
          round: p.round,
          phase: p.phase,
          points: [],
        };
        series.push(cur);
        lastPhase = p.phase;
      }
      void lastPhase;
      cur.points.push({ x: i, loss: Math.max(1e-9, p.lossNormalized) });
    });
    return { lossSeries: series, roundBoundaries: boundaries };
  }, [liveFitCurve]);

  return (
    <div style={wrapStyle}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 10 }}>
        <Slider label={`Rounds: ${settings.rounds}`} value={settings.rounds} min={1} max={6}
          onChange={(v) => setSettings({ rounds: v })} disabled={running} />
        <Slider label={`Trials/round: ${settings.trialsPerRound}`} value={settings.trialsPerRound} min={16} max={96} step={8}
          onChange={(v) => setSettings({ trialsPerRound: v })} disabled={running} />
        <Slider label={`Trial ticks: ${settings.trialTicks} (~${(settings.trialTicks / 60).toFixed(1)}s)`}
          value={settings.trialTicks} min={60} max={240} step={30}
          onChange={(v) => setSettings({ trialTicks: v })} disabled={running} />
        <Slider label={`Seed: ${settings.seed}`} value={settings.seed} min={1} max={999}
          onChange={(v) => setSettings({ seed: v })} disabled={running} />
      </div>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
        {!running ? (
          <button onClick={train} style={primaryBtn}>Train v2 model</button>
        ) : (
          <button onClick={cancel} style={dangerBtn}>Cancel</button>
        )}
        {model && (
          <button onClick={clearTrained} style={ghostBtn}>Clear cached</button>
        )}
        {error && <span style={{ color: '#ff5566', fontSize: 11 }}>err: {error}</span>}
      </div>
      {running && (
        <ProgressCard
          round={currentRound}
          rounds={rounds}
          phase={currentPhase}
          trialsCollected={trialsCollected}
          trialsDiscarded={trialsDiscarded}
          trialsAttemptedThisRound={trialsAttemptedThisRound}
          trialsTargetThisRound={trialsTargetThisRound}
          iter={currentIter}
          iterTotal={currentIterTotal}
          latestLoss={latestLoss}
          startedAt={trainingStartedAt}
        />
      )}
      {lossSeries.length > 0 && lossSeries[0]!.points.length > 3 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 8 }}>
            <span style={{ fontSize: 11, opacity: 0.7 }}>
              Per-sample loss (normalized so rounds + phases are comparable; log y)
            </span>
            <span style={{ fontSize: 10, opacity: 0.6, display: 'inline-flex', gap: 10 }}>
              <span><span style={{ color: '#55dcff' }}>━</span> parametric</span>
              <span><span style={{ color: '#a888ff' }}>━</span> residual</span>
              <span><span style={{ color: '#1f2735' }}>┊</span> round boundary</span>
            </span>
          </div>
          <div style={{ height: 140 }}>
            <ResponsiveContainer>
              <LineChart margin={{ top: 4, right: 12, bottom: 8, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2735" />
                <XAxis
                  type="number" dataKey="x" stroke="#cdd3de" fontSize={10}
                  domain={[0, 'dataMax']}
                  allowDuplicatedCategory={false}
                  label={{ value: 'iter (concat across rounds)', position: 'insideBottom', offset: -2, fill: '#cdd3de', fontSize: 10 }}
                />
                <YAxis stroke="#cdd3de" fontSize={10} scale="log" domain={['auto', 'auto']} />
                <Tooltip
                  contentStyle={{ background: '#141a26', border: '1px solid #1f2735', fontSize: 11 }}
                  formatter={(value) => typeof value === 'number' ? value.toExponential(2) : String(value)}
                />
                {roundBoundaries.map((b) => (
                  <ReferenceLine key={`b-${b}`} x={b} stroke="#33425a" strokeDasharray="2 3" />
                ))}
                {lossSeries.map((s) => (
                  <Line
                    key={s.key}
                    type="monotone"
                    data={s.points}
                    dataKey="loss"
                    stroke={s.phase === 'parametric' ? '#55dcff' : '#a888ff'}
                    dot={false}
                    strokeWidth={1.6}
                    isAnimationActive={false}
                    name={`r${s.round + 1} ${s.phase}`}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}

// Prominent live-progress card shown while a training run is active.
// Designed to make it obvious-at-a-glance that training is alive and how
// far along it is, since a 6-round run can take 30-60+ seconds and the
// previous one-line strip was easy to overlook.
//
// Layout:
//   ┌─────────────────────────────────────────────────────────────────┐
//   │ TRAINING  ●○○○○○○  round 1/6 · 12% overall · elapsed 0:08 · ETA ~1:00 │
//   │ PARAMETRIC  iter 42 / 200  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
//   │ overall ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
//   │ trials 96 collected (0 discarded) · latest loss 0.0214             │
//   └─────────────────────────────────────────────────────────────────┘

type Phase = 'initializing' | 'collecting' | 'parametric' | 'residual' | 'evaluating' | null;

const PHASE_COLOR: Record<Exclude<Phase, null> | 'starting', string> = {
  starting: '#cdd3de',
  initializing: '#7fbfff',
  collecting: '#ffd070',
  parametric: '#55dcff',
  residual: '#a888ff',
  evaluating: '#55ff88',
};

// Weights describing roughly how much of one round's wall-clock time
// each phase consumes. Used for the overall-progress estimate.
const PHASE_WEIGHTS = {
  collecting: 0.20,
  parametric: 0.60,
  residual: 0.15,
  evaluating: 0.05,
};

function ProgressCard({
  round, rounds, phase,
  trialsCollected, trialsDiscarded,
  trialsAttemptedThisRound, trialsTargetThisRound,
  iter, iterTotal, latestLoss, startedAt,
}: {
  round: number;
  rounds: number;
  phase: Phase;
  trialsCollected: number;
  trialsDiscarded: number;
  trialsAttemptedThisRound: number;
  trialsTargetThisRound: number;
  iter: number;
  iterTotal: number;
  latestLoss: number | null;
  startedAt: number | null;
}) {
  const phaseKey = (phase ?? 'starting') as keyof typeof PHASE_COLOR;
  const color = PHASE_COLOR[phaseKey];

  // Per-phase progress (0..1).
  let phaseDetail = '';
  let phaseFraction = 0;
  if (phase === 'initializing') {
    phaseDetail = 'loading physics engine (Rapier WASM)…';
    phaseFraction = 0.15;
  } else if (phase === 'collecting' && trialsTargetThisRound > 0) {
    phaseDetail = `${trialsAttemptedThisRound} / ${trialsTargetThisRound} trials`;
    phaseFraction = trialsAttemptedThisRound / trialsTargetThisRound;
  } else if ((phase === 'parametric' || phase === 'residual') && iterTotal > 0) {
    phaseDetail = `iter ${iter + 1} / ${iterTotal}`;
    phaseFraction = (iter + 1) / iterTotal;
  } else if (phase === 'evaluating') {
    phaseDetail = 'computing diagnostics…';
    phaseFraction = 0.5;
  }
  phaseFraction = Math.max(0, Math.min(1, phaseFraction));

  // Overall progress estimate: completed rounds + fraction of current.
  // Residual only runs on the last round, so non-last rounds reach 100%
  // after parametric + evaluating finish (no residual to wait on).
  const intraRound = (() => {
    const isLastRound = round === rounds - 1;
    const w = PHASE_WEIGHTS;
    const totalForThisRound = isLastRound
      ? w.collecting + w.parametric + w.residual + w.evaluating
      : w.collecting + w.parametric + w.evaluating;
    let done = 0;
    if (phase === 'collecting') {
      done = w.collecting * phaseFraction;
    } else if (phase === 'parametric') {
      done = w.collecting + w.parametric * phaseFraction;
    } else if (phase === 'residual') {
      done = w.collecting + w.parametric + w.residual * phaseFraction;
    } else if (phase === 'evaluating') {
      done = w.collecting + w.parametric
        + (isLastRound ? w.residual : 0)
        + w.evaluating * phaseFraction;
    }
    return done / totalForThisRound;
  })();
  // During initializing there's no round yet; show a small overall
  // fraction (~1%) so the bar isn't a dead zero while the user waits.
  const overall = phase === 'initializing'
    ? 0.01
    : Math.max(0, Math.min(1, (round + intraRound) / rounds));

  // Tick elapsed time once per second so the user sees the clock moving.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(id);
  }, []);
  const elapsedMs = startedAt ? Math.max(0, now - startedAt) : 0;
  const etaMs = overall > 0.02 ? Math.max(0, elapsedMs / overall - elapsedMs) : null;

  return (
    <div style={cardStyle}>
      <div style={cardTopRowStyle}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <ActivityDot color={color} />
          <span style={{
            color, fontWeight: 700, fontSize: 13,
            textTransform: 'uppercase', letterSpacing: 0.6,
          }}>
            {phase ?? 'starting'}
          </span>
          {phase !== 'initializing' && phase !== null && (
            <>
              <span style={{ opacity: 0.5, fontSize: 12 }}>·</span>
              <span style={{ fontSize: 12 }}>
                round <span style={{ color: '#cdeaff', fontWeight: 700 }}>{round + 1}</span>
                <span style={{ opacity: 0.5 }}> / {rounds}</span>
              </span>
              <RoundDots round={round} rounds={rounds} />
            </>
          )}
        </div>
        <div style={{ display: 'flex', gap: 14, fontSize: 11, opacity: 0.85 }}>
          <span>elapsed <span style={{ color: '#cdeaff' }}>{formatDuration(elapsedMs)}</span></span>
          {etaMs !== null && (
            <span>ETA <span style={{ color: '#cdeaff' }}>~{formatDuration(etaMs)}</span></span>
          )}
          <span>overall <span style={{ color: '#cdeaff', fontWeight: 600 }}>{Math.round(overall * 100)}%</span></span>
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, opacity: 0.9 }}>
          <span style={{ color }}>{phaseDetail || '\u00a0'}</span>
          {latestLoss !== null && (
            <span style={{ opacity: 0.85 }}>
              loss <span style={{ color: '#cdeaff', fontWeight: 600 }}>{latestLoss.toFixed(4)}</span>
            </span>
          )}
        </div>
        <ProgressTrack fraction={phaseFraction} color={color} animate />
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, opacity: 0.7 }}>
          <span>overall progress</span>
          <span>
            {trialsCollected} trials collected
            {trialsDiscarded > 0 && <> · <span style={{ color: '#ff8aa0' }}>{trialsDiscarded} discarded</span></>}
          </span>
        </div>
        <ProgressTrack fraction={overall} color="#55dcff" thin />
      </div>
    </div>
  );
}

function ProgressTrack({ fraction, color, animate, thin }: {
  fraction: number; color: string; animate?: boolean; thin?: boolean;
}) {
  return (
    <div style={{
      position: 'relative',
      height: thin ? 4 : 8, borderRadius: 4, overflow: 'hidden',
      background: 'rgba(255, 255, 255, 0.06)',
    }}>
      <div style={{
        width: `${fraction * 100}%`, height: '100%',
        background: color,
        transition: 'width 200ms linear',
        boxShadow: animate ? `0 0 8px ${color}40` : undefined,
      }} />
      {animate && (
        <div style={{
          position: 'absolute', inset: 0, pointerEvents: 'none',
          background: `linear-gradient(90deg, transparent 0%, ${color}30 50%, transparent 100%)`,
          backgroundSize: '200% 100%',
          animation: 'modelLabShimmer 1.4s linear infinite',
        }} />
      )}
    </div>
  );
}

function RoundDots({ round, rounds }: { round: number; rounds: number }) {
  return (
    <span style={{ display: 'inline-flex', gap: 4, marginLeft: 4 }}>
      {Array.from({ length: rounds }, (_, i) => (
        <span
          key={i}
          style={{
            width: 8, height: 8, borderRadius: '50%',
            background: i < round ? '#55dcff' : i === round ? '#cdeaff' : 'rgba(205, 211, 222, 0.18)',
            boxShadow: i === round ? '0 0 6px #55dcff80' : 'none',
          }}
        />
      ))}
    </span>
  );
}

function ActivityDot({ color }: { color: string }) {
  return (
    <span style={{
      width: 9, height: 9, borderRadius: '50%',
      background: color,
      boxShadow: `0 0 8px ${color}`,
      animation: 'modelLabPulse 1.1s ease-in-out infinite',
    }} />
  );
}

function formatDuration(ms: number): string {
  const totalSec = Math.round(ms / 1000);
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  if (min === 0) return `${sec}s`;
  return `${min}:${String(sec).padStart(2, '0')}`;
}

function Slider({ label, value, min, max, step = 1, onChange, disabled }:
  { label: string; value: number; min: number; max: number; step?: number; onChange: (v: number) => void; disabled?: boolean }) {
  return (
    <label style={{ display: 'flex', flexDirection: 'column', gap: 2, fontSize: 11 }}>
      <span>{label}</span>
      <input type="range" min={min} max={max} step={step} value={value} disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))} style={{ accentColor: '#55dcff' }} />
    </label>
  );
}

const wrapStyle: React.CSSProperties = {
  display: 'flex', flexDirection: 'column', gap: 12,
  padding: 14, borderRadius: 8,
  background: 'rgba(13, 17, 25, 0.85)', border: '1px solid #1f2735',
};

const cardStyle: React.CSSProperties = {
  display: 'flex', flexDirection: 'column', gap: 10,
  padding: '12px 14px', borderRadius: 8,
  background: 'rgba(85, 220, 255, 0.05)',
  border: '1px solid rgba(85, 220, 255, 0.35)',
  boxShadow: '0 0 18px rgba(85, 220, 255, 0.06) inset',
};

const cardTopRowStyle: React.CSSProperties = {
  display: 'flex', alignItems: 'center', justifyContent: 'space-between',
  gap: 12, flexWrap: 'wrap',
};

// Inject keyframes once (Recharts/Next don't ship a CSS-in-JS solution
// for arbitrary @keyframes, so we drop them in a global <style> tag).
if (typeof document !== 'undefined' && !document.getElementById('model-lab-progress-keyframes')) {
  const style = document.createElement('style');
  style.id = 'model-lab-progress-keyframes';
  style.textContent = `
    @keyframes modelLabPulse {
      0%, 100% { transform: scale(1); opacity: 1; }
      50% { transform: scale(1.35); opacity: 0.55; }
    }
    @keyframes modelLabShimmer {
      0% { background-position: 200% 0; }
      100% { background-position: -200% 0; }
    }
  `;
  document.head.appendChild(style);
}

const primaryBtn: React.CSSProperties = {
  background: '#55dcff', color: '#0a0d14', border: 'none',
  padding: '8px 14px', borderRadius: 4, fontWeight: 700, cursor: 'pointer',
  font: 'inherit',
};

const dangerBtn: React.CSSProperties = {
  background: '#ff5566', color: '#0a0d14', border: 'none',
  padding: '8px 14px', borderRadius: 4, fontWeight: 700, cursor: 'pointer',
  font: 'inherit',
};

const ghostBtn: React.CSSProperties = {
  background: 'transparent', color: '#cdd3de', border: '1px solid #1f2735',
  padding: '8px 14px', borderRadius: 4, cursor: 'pointer',
  font: 'inherit',
};
