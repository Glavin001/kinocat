'use client';

// Model Lab UI: offline training + live diagnostics for the v2 learned
// vehicle model. Rendered as a collapsible overlay on /raceprimitives.
//
// Three sections:
//   1. Training controls (Train / Cancel / settings sliders)
//   2. Live progress (round / trials / loss; visible during training)
//   3. Results (open-loop divergence comparison + Recharts loss + divergence
//      charts; visible once at least one round has completed)
//
// Persistence + A/B toggle live in the parent (RacePrimitives.tsx).

import { useMemo, useRef, useState } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, BarChart, Bar, ResponsiveContainer,
} from 'recharts';
import { runOfflineTraining, type TrainingEvent } from '../lib/training-driver';
import { useIsMobile } from '../lib/use-is-mobile';
import type { LearnedVehicleModel } from 'kinocat/agent';
import type { ModelDiagnostics, FitProgressEvent } from 'kinocat/learning';

export interface ModelLabProps {
  onTrained: (model: LearnedVehicleModel, diag: ModelDiagnostics, trialsUsed: number) => void;
  /** If a model was loaded from localStorage, show its meta + allow clearing. */
  loadedMeta?: { trialsUsed: number; openLoopRmsAt1s: number; legacyRmsAt1s?: number; kinematicRmsAt1s?: number; createdAt: number } | null;
  onClearLoaded?: () => void;
  /** Optional download trigger when the user has a trained or loaded model. */
  onExport?: () => void;
  /** Whether the v2 library is currently driving the learned car. */
  useV2: boolean;
  onToggleUseV2: (v: boolean) => void;
  /** True if a trained v2 model is currently available (loaded or just trained). */
  hasV2Model: boolean;
}

type TrainStatus = 'idle' | 'running' | 'done' | 'error';

interface RoundSnapshot {
  round: number;
  trialsAfter: number;
  totalLossCurve: FitProgressEvent[];
  diag: ModelDiagnostics;
}

export function ModelLab(props: ModelLabProps) {
  const isMobile = useIsMobile();
  const [open, setOpen] = useState(false);
  const [rounds, setRounds] = useState(3);
  const [trialsPerRound, setTrialsPerRound] = useState(48);
  const [trialTicks, setTrialTicks] = useState(120); // ~2s
  const [seed, setSeed] = useState(42);
  const [status, setStatus] = useState<TrainStatus>('idle');
  const [currentRound, setCurrentRound] = useState(0);
  const [trialsCollected, setTrialsCollected] = useState(0);
  const [trialsDiscarded, setTrialsDiscarded] = useState(0);
  const [latestLoss, setLatestLoss] = useState<number | null>(null);
  const [roundHistory, setRoundHistory] = useState<RoundSnapshot[]>([]);
  const [error, setError] = useState<string | null>(null);
  const cancelRef = useRef<{ cancelled: boolean }>({ cancelled: false });
  const fitProgressRef = useRef<FitProgressEvent[]>([]);
  const trialsAccumRef = useRef(0);
  const pendingDiagRef = useRef<ModelDiagnostics | null>(null);

  async function startTraining() {
    cancelRef.current = { cancelled: false };
    fitProgressRef.current = [];
    trialsAccumRef.current = 0;
    pendingDiagRef.current = null;
    setStatus('running');
    setError(null);
    setCurrentRound(0);
    setTrialsCollected(0);
    setTrialsDiscarded(0);
    setLatestLoss(null);
    setRoundHistory([]);
    try {
      const onEvent = (e: TrainingEvent) => {
        if (cancelRef.current.cancelled) return;
        switch (e.type) {
          case 'round-start':
            setCurrentRound(e.round);
            fitProgressRef.current = [];
            pendingDiagRef.current = null;
            break;
          case 'trial-batch':
            trialsAccumRef.current += e.collected;
            setTrialsCollected(trialsAccumRef.current);
            setTrialsDiscarded((d) => d + e.discarded);
            break;
          case 'fit-progress':
            fitProgressRef.current.push(e.event);
            setLatestLoss(e.event.loss);
            break;
          case 'evaluation':
            pendingDiagRef.current = e.diagnostics;
            break;
          case 'round-end': {
            const diag = pendingDiagRef.current ?? { openLoopDivergence: [], perStateRms: [], coverage: [], baselines: {} };
            const trialsAfter = trialsAccumRef.current;
            const curve = [...fitProgressRef.current];
            setRoundHistory((h) => [...h, {
              round: e.round, trialsAfter, totalLossCurve: curve, diag,
            }]);
            break;
          }
          case 'done':
            break;
        }
      };
      const result = await runOfflineTraining({
        rounds, trialsPerActiveRound: trialsPerRound, trialTicks,
        sampleEveryNTicks: 6, seed, onEvent,
      });
      if (cancelRef.current.cancelled) {
        setStatus('idle');
        return;
      }
      setStatus('done');
      props.onTrained(result.model, result.finalDiagnostics, result.trials.size());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus('error');
    }
  }

  function cancelTraining() {
    cancelRef.current.cancelled = true;
    setStatus('idle');
  }

  const showResults = roundHistory.length > 0 || props.loadedMeta !== null;
  const panelBody = (
    <>
      <Section title="Training">
        <Row>
          <SliderField label={`Rounds: ${rounds}`} value={rounds} min={1} max={6} onChange={setRounds} disabled={status === 'running'} />
          <SliderField label={`Trials/round: ${trialsPerRound}`} value={trialsPerRound} min={16} max={96} step={8} onChange={setTrialsPerRound} disabled={status === 'running'} />
        </Row>
        <Row>
          <SliderField label={`Trial ticks: ${trialTicks} (~${(trialTicks / 60).toFixed(1)}s)`} value={trialTicks} min={60} max={240} step={30} onChange={setTrialTicks} disabled={status === 'running'} />
          <SliderField label={`RNG seed: ${seed}`} value={seed} min={1} max={999} onChange={setSeed} disabled={status === 'running'} />
        </Row>
        <Row>
          {status !== 'running' ? (
            <button onClick={startTraining} style={primaryBtnStyle}>Train v2 model</button>
          ) : (
            <button onClick={cancelTraining} style={dangerBtnStyle}>Cancel</button>
          )}
          {props.hasV2Model && (
            <label style={toggleStyle}>
              <input type="checkbox" checked={props.useV2} onChange={(e) => props.onToggleUseV2(e.target.checked)} />
              Use v2 library for learned car
            </label>
          )}
          {props.hasV2Model && props.onExport && (
            <button onClick={props.onExport} style={ghostBtnStyle}>Export</button>
          )}
          {props.hasV2Model && props.onClearLoaded && (
            <button onClick={props.onClearLoaded} style={ghostBtnStyle}>Clear cached</button>
          )}
        </Row>
      </Section>

      {status === 'running' && (
        <Section title={`Progress — Round ${currentRound + 1}/${rounds}`}>
          <KV k="Trials collected" v={String(trialsCollected)} />
          <KV k="Trials discarded" v={String(trialsDiscarded)} />
          {latestLoss !== null && <KV k="Latest loss" v={latestLoss.toFixed(4)} />}
        </Section>
      )}

      {error && (
        <Section title="Error">
          <pre style={{ color: '#ff5566', whiteSpace: 'pre-wrap', fontSize: 11 }}>{error}</pre>
        </Section>
      )}

      {showResults && (
        <ResultsView roundHistory={roundHistory} loadedMeta={props.loadedMeta ?? undefined} />
      )}

      <div style={{
        marginTop: 8, padding: '8px 10px',
        background: 'rgba(85, 220, 255, 0.08)',
        border: '1px solid rgba(85, 220, 255, 0.25)',
        borderRadius: 4, fontSize: 11, lineHeight: 1.5,
      }}>
        Looking for a deeper view? Open the dedicated{' '}
        <a href="/model-lab" style={{ color: '#55dcff', fontWeight: 700 }}>Model Lab dashboard</a>{' '}
        for rollout playback, ground-truth fan plots, coverage heatmap, and
        the scenario playground.
      </div>
    </>
  );

  if (isMobile) {
    // Mobile: small floating launcher button at the top-right (so it
    // doesn't fight for space with the per-car stat panels). Tapping it
    // opens a full-width bottom-sheet over a dim backdrop.
    return (
      <>
        <button
          onClick={() => setOpen(true)}
          style={{
            position: 'absolute', top: 8, right: 8, zIndex: 40,
            background: 'rgba(13, 17, 25, 0.92)', color: '#cdd3de',
            border: '1px solid #223044', borderRadius: 6,
            padding: '6px 10px', font: '11px ui-monospace, monospace',
            cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 6,
          }}
        >
          Model Lab
          {props.hasV2Model && <span style={{ color: '#55dcff' }}>●</span>}
        </button>
        {open && (
          <>
            <div
              onClick={() => setOpen(false)}
              style={{
                position: 'fixed', inset: 0, background: 'rgba(0, 0, 0, 0.55)',
                zIndex: 60,
              }}
            />
            <div
              style={{
                position: 'fixed', left: 0, right: 0, bottom: 0,
                maxHeight: '85vh', overflowY: 'auto',
                background: '#0d1119', color: '#cdd3de',
                font: '12px ui-monospace, monospace',
                borderTop: '1px solid #223044',
                borderTopLeftRadius: 12, borderTopRightRadius: 12,
                padding: '12px 14px 24px', zIndex: 61,
                boxShadow: '0 -8px 24px rgba(0, 0, 0, 0.6)',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
                <div style={{ fontWeight: 700 }}>
                  Model Lab {props.hasV2Model && <span style={{ color: '#55dcff' }}>● v2 ready</span>}
                </div>
                <button onClick={() => setOpen(false)} style={ghostBtnStyle}>Close</button>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>{panelBody}</div>
            </div>
          </>
        )}
      </>
    );
  }

  return (
    <div style={panelOuterStyle(open)}>
      <button onClick={() => setOpen(!open)} style={panelToggleStyle}>
        {open ? '▼ Model Lab' : '▲ Model Lab'}
        {props.hasV2Model && <span style={{ marginLeft: 8, color: '#55dcff' }}>● v2 ready</span>}
      </button>
      {open && <div style={panelContentStyle}>{panelBody}</div>}
    </div>
  );
}

interface ResultsViewProps {
  roundHistory: RoundSnapshot[];
  loadedMeta?: ModelLabProps['loadedMeta'];
}

function ResultsView({ roundHistory, loadedMeta }: ResultsViewProps) {
  const lossChartData = useMemo(() => {
    const rows: { iter: number; loss: number }[] = [];
    let cumIter = 0;
    for (const r of roundHistory) {
      for (const ev of r.totalLossCurve) {
        rows.push({ iter: cumIter + ev.iter, loss: ev.loss });
      }
      cumIter += r.totalLossCurve.length > 0 ? r.totalLossCurve[r.totalLossCurve.length - 1]!.iter + 1 : 0;
    }
    return rows;
  }, [roundHistory]);

  const divergenceData = useMemo(() => {
    if (roundHistory.length === 0) return [];
    const final = roundHistory[roundHistory.length - 1]!.diag;
    const horizons = final.openLoopDivergence.map((r) => r.tSec);
    return horizons.map((t) => {
      const v2 = final.openLoopDivergence.find((r) => r.tSec === t)?.posRms ?? 0;
      const legacy = final.baselines['legacyV1']?.find((r) => r.tSec === t)?.posRms ?? 0;
      const kin = final.baselines['kinematic']?.find((r) => r.tSec === t)?.posRms ?? 0;
      return { horizon: `${t.toFixed(1)}s`, v2, legacy, kinematic: kin };
    });
  }, [roundHistory]);

  const headline = useMemo(() => {
    if (roundHistory.length === 0 && loadedMeta) {
      return {
        v2: loadedMeta.openLoopRmsAt1s,
        legacy: loadedMeta.legacyRmsAt1s ?? null,
        kin: loadedMeta.kinematicRmsAt1s ?? null,
      };
    }
    if (roundHistory.length === 0) return null;
    const final = roundHistory[roundHistory.length - 1]!.diag;
    const mid = (rows: { tSec: number; posRms: number }[] | undefined) =>
      rows?.find((r) => r.tSec >= 1.0)?.posRms ?? null;
    return {
      v2: mid(final.openLoopDivergence) ?? 0,
      legacy: mid(final.baselines['legacyV1']),
      kin: mid(final.baselines['kinematic']),
    };
  }, [roundHistory, loadedMeta]);

  return (
    <>
      <Section title="Results — open-loop divergence at T=1s">
        {headline && (
          <Row>
            <Stat color="#55dcff" label="v2 model" value={headline.v2.toFixed(3) + ' m'} />
            {headline.legacy !== null && <Stat color="#ff8aa0" label="legacy 5-param" value={headline.legacy.toFixed(3) + ' m'} />}
            {headline.kin !== null && <Stat color="#ffd070" label="kinematic" value={headline.kin.toFixed(3) + ' m'} />}
            {headline.legacy !== null && headline.legacy > 0 && (
              <Stat color="#55ff88" label="vs legacy" value={`${((1 - headline.v2 / headline.legacy) * 100).toFixed(1)}% better`} />
            )}
          </Row>
        )}
      </Section>

      {divergenceData.length > 0 && (
        <Section title="Open-loop divergence vs horizon">
          <div style={{ width: '100%', height: 180 }}>
            <ResponsiveContainer>
              <BarChart data={divergenceData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#223044" />
                <XAxis dataKey="horizon" stroke="#cdd3de" fontSize={11} />
                <YAxis stroke="#cdd3de" fontSize={11} label={{ value: 'pos RMS (m)', angle: -90, position: 'insideLeft', fill: '#cdd3de', fontSize: 11 }} />
                <Tooltip contentStyle={{ background: '#141a26', border: '1px solid #223044', fontSize: 11 }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="v2" fill="#55dcff" />
                <Bar dataKey="legacy" fill="#ff8aa0" />
                <Bar dataKey="kinematic" fill="#ffd070" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Section>
      )}

      {lossChartData.length > 1 && (
        <Section title="Training loss (per Nelder-Mead iteration)">
          <div style={{ width: '100%', height: 160 }}>
            <ResponsiveContainer>
              <LineChart data={lossChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#223044" />
                <XAxis dataKey="iter" stroke="#cdd3de" fontSize={11} />
                <YAxis stroke="#cdd3de" fontSize={11} scale="log" domain={['auto', 'auto']} />
                <Tooltip contentStyle={{ background: '#141a26', border: '1px solid #223044', fontSize: 11 }} />
                <Line type="monotone" dataKey="loss" stroke="#55dcff" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Section>
      )}
    </>
  );
}

// ---------------------------------------------------------------------------
// Small style helpers — keep dependencies minimal (no styled-components etc.)

function panelOuterStyle(open: boolean): React.CSSProperties {
  return {
    position: 'absolute',
    top: 8,
    right: 8,
    width: open ? 'min(420px, calc(100vw - 32px))' : 'auto',
    maxWidth: 'calc(100vw - 32px)',
    background: 'rgba(10, 13, 20, 0.95)',
    border: '1px solid #223044',
    borderRadius: 6,
    fontFamily: 'ui-monospace, monospace',
    fontSize: 12,
    color: '#cdd3de',
    zIndex: 40,
    maxHeight: 'calc(100vh - 100px)',
    overflow: 'auto',
    boxShadow: open ? '0 6px 24px rgba(0, 0, 0, 0.4)' : 'none',
  };
}

const panelToggleStyle: React.CSSProperties = {
  width: '100%',
  padding: '8px 12px',
  background: '#141a26',
  border: 'none',
  borderBottom: '1px solid #223044',
  color: '#cdd3de',
  font: 'inherit',
  textAlign: 'left',
  cursor: 'pointer',
};

const panelContentStyle: React.CSSProperties = {
  padding: 12,
  display: 'flex',
  flexDirection: 'column',
  gap: 12,
};

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <div style={{ fontSize: 11, opacity: 0.7, marginBottom: 6, textTransform: 'uppercase', letterSpacing: 0.5 }}>{title}</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>{children}</div>
    </div>
  );
}

function Row({ children }: { children: React.ReactNode }) {
  return <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center' }}>{children}</div>;
}

function SliderField({ label, value, min, max, step = 1, onChange, disabled }: { label: string; value: number; min: number; max: number; step?: number; onChange: (v: number) => void; disabled?: boolean }) {
  return (
    <label style={{ display: 'flex', flexDirection: 'column', gap: 2, flex: '1 1 200px' }}>
      <span style={{ fontSize: 11 }}>{label}</span>
      <input
        type="range"
        min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        disabled={disabled}
        style={{ accentColor: '#55dcff' }}
      />
    </label>
  );
}

function Stat({ color, label, value }: { color: string; label: string; value: string }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 2, padding: '4px 10px', border: '1px solid #223044', borderRadius: 4, minWidth: 100 }}>
      <span style={{ fontSize: 10, opacity: 0.7 }}>{label}</span>
      <span style={{ color, fontWeight: 600 }}>{value}</span>
    </div>
  );
}

function KV({ k, v }: { k: string; v: string }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
      <span style={{ opacity: 0.7 }}>{k}</span>
      <span>{v}</span>
    </div>
  );
}

const primaryBtnStyle: React.CSSProperties = {
  padding: '6px 12px',
  background: '#55dcff',
  border: 'none',
  borderRadius: 3,
  color: '#0a0d14',
  cursor: 'pointer',
  font: 'inherit',
  fontWeight: 600,
};

const dangerBtnStyle: React.CSSProperties = {
  padding: '6px 12px',
  background: '#ff5566',
  border: 'none',
  borderRadius: 3,
  color: '#0a0d14',
  cursor: 'pointer',
  font: 'inherit',
  fontWeight: 600,
};

const ghostBtnStyle: React.CSSProperties = {
  padding: '6px 12px',
  background: 'transparent',
  border: '1px solid #223044',
  borderRadius: 3,
  color: '#cdd3de',
  cursor: 'pointer',
  font: 'inherit',
};

const toggleStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 6,
  cursor: 'pointer',
};
