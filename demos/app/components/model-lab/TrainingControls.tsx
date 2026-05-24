'use client';

// Compact training controls panel: sliders for rounds/trials/ticks/seed,
// Train/Cancel buttons, and a live loss + per-round-progress strip. The
// dashboard places this in the page header so training observability is
// the FIRST thing the user sees rather than buried in a panel.

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { useMemo } from 'react';
import { useModelLab } from '../../lib/model-lab-store';

export function TrainingControls() {
  const { state, setSettings, train, cancel, clearTrained } = useModelLab();
  const { settings, status, currentRound, rounds, trialsCollected, trialsDiscarded, latestLoss, currentPhase, liveFitCurve, error, model } = state;

  const running = status === 'running';

  const lossData = useMemo(() => {
    // De-duplicate consecutive same-loss entries to keep the chart snappy.
    return liveFitCurve.map((p, i) => ({ x: i, loss: Math.max(1e-6, p.loss), phase: p.phase }));
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
        {running && (
          <span style={{ opacity: 0.8, fontSize: 11 }}>
            round {currentRound + 1}/{rounds} · trials {trialsCollected} (
            <span style={{ color: '#ff8aa0' }}>{trialsDiscarded} discarded</span>
            ) {latestLoss !== null && (
              <>
                {' '}· loss <span style={{ color: '#cdeaff' }}>{latestLoss.toFixed(4)}</span>
                {currentPhase && <span style={{ opacity: 0.6 }}> ({currentPhase})</span>}
              </>
            )}
          </span>
        )}
        {error && <span style={{ color: '#ff5566', fontSize: 11 }}>err: {error}</span>}
      </div>
      {lossData.length > 5 && (
        <div style={{ height: 130 }}>
          <ResponsiveContainer>
            <LineChart data={lossData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2735" />
              <XAxis dataKey="x" stroke="#cdd3de" fontSize={10} label={{ value: 'iter (concat across rounds)', position: 'insideBottom', offset: -2, fill: '#cdd3de', fontSize: 10 }} />
              <YAxis stroke="#cdd3de" fontSize={10} scale="log" domain={['auto', 'auto']} />
              <Tooltip contentStyle={{ background: '#141a26', border: '1px solid #1f2735', fontSize: 11 }} />
              <Line type="monotone" dataKey="loss" stroke="#55dcff" dot={false} strokeWidth={1.6} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
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
