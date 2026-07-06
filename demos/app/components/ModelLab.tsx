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

import { useEffect, useMemo, useRef, useState } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, BarChart, Bar, ResponsiveContainer,
} from 'recharts';
import { runManeuverTraining, type TrainingEvent, CAR_COVERAGE_AXES } from '../lib/training-driver';
import { useIsMobile } from '../lib/use-is-mobile';
import type { LearnedVehicleModel } from 'kinocat/agent';
import type { ModelDiagnostics, FitProgressEvent } from 'kinocat/learning';
import type { CoverageCellSummary } from 'kinocat/training';

export interface ModelLabProps {
  /** Controlled open state (drawer). When provided, ModelLab renders no
   *  inline launcher — the host (e.g. the raceprimitives toolbar) owns the
   *  trigger. Omit for the legacy self-launching floating panel. */
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  onTrained: (model: LearnedVehicleModel, diag: ModelDiagnostics, trialsUsed: number) => void;
  /** If a model was loaded from localStorage, show its meta + allow clearing. */
  loadedMeta?: { trialsUsed: number; openLoopRmsAt1s: number; legacyRmsAt1s?: number; kinematicRmsAt1s?: number; createdAt: number } | null;
  onClearLoaded?: () => void;
  /** Optional download trigger when the user has a trained or loaded model. */
  onExport?: () => void;
  /** Reset back to the preloaded `/models/v2-default.json` artifact (the
   *  model the `pnpm run train` CLI shipped with the project). Optional —
   *  when absent, the button is hidden. */
  onResetToDefault?: () => void;
  /** True iff a preloaded default artifact is known to be available
   *  (the page tried to fetch it on mount and got a 200). Drives the
   *  reset button's enabled state. */
  hasPreloadedDefault?: boolean;
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
  coverage?: CoverageCellSummary[];
}

type Phase = 'initializing' | 'collecting' | 'parametric' | 'residual' | 'evaluating' | null;

export function ModelLab(props: ModelLabProps) {
  const isMobile = useIsMobile();
  // Controlled (host owns the trigger) or uncontrolled (self-launching).
  const controlled = props.open !== undefined;
  const [openInternal, setOpenInternal] = useState(false);
  const open = controlled ? !!props.open : openInternal;
  const setOpen = (v: boolean) => {
    if (controlled) props.onOpenChange?.(v);
    else setOpenInternal(v);
  };
  const [rounds, setRounds] = useState(3);
  const [trialsPerRound, setTrialsPerRound] = useState(48);
  const [trialTicks, setTrialTicks] = useState(120); // ~2s
  const [seed, setSeed] = useState(42);
  const [status, setStatus] = useState<TrainStatus>('idle');
  const [currentRound, setCurrentRound] = useState(0);
  const [trialsCollected, setTrialsCollected] = useState(0);
  const [trialsDiscarded, setTrialsDiscarded] = useState(0);
  const [trialsAttemptedThisRound, setTrialsAttemptedThisRound] = useState(0);
  const [trialsTargetThisRound, setTrialsTargetThisRound] = useState(0);
  const [latestLoss, setLatestLoss] = useState<number | null>(null);
  const [currentPhase, setCurrentPhase] = useState<Phase>(null);
  const [currentIter, setCurrentIter] = useState(0);
  const [currentIterTotal, setCurrentIterTotal] = useState(0);
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [roundHistory, setRoundHistory] = useState<RoundSnapshot[]>([]);
  const [error, setError] = useState<string | null>(null);
  const cancelRef = useRef<{ cancelled: boolean }>({ cancelled: false });
  const fitProgressRef = useRef<FitProgressEvent[]>([]);
  const trialsAccumRef = useRef(0);
  const pendingDiagRef = useRef<ModelDiagnostics | null>(null);
  const pendingCoverageRef = useRef<CoverageCellSummary[] | null>(null);

  async function startTraining() {
    cancelRef.current = { cancelled: false };
    fitProgressRef.current = [];
    trialsAccumRef.current = 0;
    pendingDiagRef.current = null;
    pendingCoverageRef.current = null;
    setStatus('running');
    setError(null);
    setCurrentRound(0);
    setTrialsCollected(0);
    setTrialsDiscarded(0);
    setTrialsAttemptedThisRound(0);
    setTrialsTargetThisRound(0);
    setLatestLoss(null);
    setCurrentPhase('initializing');
    setCurrentIter(0);
    setCurrentIterTotal(0);
    setStartedAt(Date.now());
    setRoundHistory([]);
    try {
      const onEvent = (e: TrainingEvent) => {
        if (cancelRef.current.cancelled) return;
        switch (e.type) {
          case 'round-start':
            setCurrentRound(e.round);
            fitProgressRef.current = [];
            pendingDiagRef.current = null;
            pendingCoverageRef.current = null;
            setCurrentPhase('collecting');
            setTrialsAttemptedThisRound(0);
            setTrialsTargetThisRound(0);
            setCurrentIter(0);
            setCurrentIterTotal(0);
            break;
          case 'phase':
            setCurrentPhase(e.phase);
            setCurrentIter(0);
            setCurrentIterTotal(0);
            break;
          case 'trial-batch':
            trialsAccumRef.current += e.collected;
            setTrialsCollected(trialsAccumRef.current);
            setTrialsDiscarded((d) => d + e.discarded);
            if (typeof e.runSoFar === 'number') setTrialsAttemptedThisRound(e.runSoFar);
            if (typeof e.runTarget === 'number') setTrialsTargetThisRound(e.runTarget);
            break;
          case 'fit-progress':
            fitProgressRef.current.push(e.event);
            setLatestLoss(e.event.loss);
            setCurrentPhase(e.phase);
            if (typeof e.iterIndex === 'number') setCurrentIter(e.iterIndex);
            if (typeof e.iterTotal === 'number') setCurrentIterTotal(e.iterTotal);
            break;
          case 'evaluation':
            pendingDiagRef.current = e.diagnostics;
            break;
          case 'coverage':
            pendingCoverageRef.current = e.cells;
            break;
          case 'round-end': {
            const diag = pendingDiagRef.current ?? { openLoopDivergence: [], perStateRms: [], coverage: [], baselines: {} };
            const trialsAfter = trialsAccumRef.current;
            const curve = [...fitProgressRef.current];
            const coverage = pendingCoverageRef.current ?? undefined;
            pendingCoverageRef.current = null;
            setRoundHistory((h) => [...h, {
              round: e.round, trialsAfter, totalLossCurve: curve, diag, coverage,
            }]);
            break;
          }
          case 'done':
            break;
        }
      };
      // Use the same maneuver-based trial sourcing the `pnpm run train`
      // CLI uses, so a model trained in the browser is structurally the
      // same as one trained at the command line — same fit + same trial
      // distribution.
      const result = await runManeuverTraining({
        rounds, trialsPerRound, trialTicks,
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
          {props.onResetToDefault && props.hasPreloadedDefault && (
            <button
              onClick={props.onResetToDefault}
              style={ghostBtnStyle}
              title="Reload the v2 model the `pnpm run train` CLI shipped with this project"
            >
              Reset to default
            </button>
          )}
        </Row>
      </Section>

      {status === 'running' && (
        <Section title={`Progress — Round ${currentRound + 1}/${rounds}`}>
          <ProgressBar
            phase={currentPhase}
            trialsAttemptedThisRound={trialsAttemptedThisRound}
            trialsTargetThisRound={trialsTargetThisRound}
            iter={currentIter}
            iterTotal={currentIterTotal}
            startedAt={startedAt}
            round={currentRound}
            rounds={rounds}
          />
          <KV k="Trials collected" v={String(trialsCollected)} />
          {trialsDiscarded > 0 && <KV k="Trials discarded" v={String(trialsDiscarded)} />}
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

  // Drawer geometry: a right-docked panel on desktop/tablet, a bottom sheet
  // on mobile. Both open over a dim backdrop and are dismissible via the
  // backdrop, the Close button, or Escape. This replaces the old floating
  // top-right panel that overlapped the LEARNED metrics readout — the host
  // toolbar now owns the launcher (controlled mode).
  const drawerStyle: React.CSSProperties = isMobile
    ? {
        position: 'fixed', left: 0, right: 0, bottom: 0,
        maxHeight: '85vh', overflowY: 'auto',
        borderTop: '1px solid #223044',
        borderTopLeftRadius: 12, borderTopRightRadius: 12,
        padding: '12px 14px 24px',
        boxShadow: '0 -8px 24px rgba(0, 0, 0, 0.6)',
      }
    : {
        position: 'fixed', top: 0, right: 0, bottom: 0,
        width: 'min(460px, 100vw)', overflowY: 'auto',
        borderLeft: '1px solid #223044',
        padding: '14px 16px 24px',
        boxShadow: '-8px 0 24px rgba(0, 0, 0, 0.5)',
      };

  const launcher = !controlled && !open ? (
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
  ) : null;

  return (
    <>
      {launcher}
      {open && (
        <>
          <div
            onClick={() => setOpen(false)}
            style={{ position: 'fixed', inset: 0, background: 'rgba(0, 0, 0, 0.55)', zIndex: 60 }}
          />
          <div
            role="dialog"
            aria-label="Model Lab"
            style={{
              ...drawerStyle,
              background: '#0d1119', color: '#cdd3de',
              font: '12px ui-monospace, monospace',
              zIndex: 61,
            }}
            onKeyDown={(e) => { if (e.key === 'Escape') setOpen(false); }}
          >
            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              marginBottom: 12, position: 'sticky', top: 0, background: '#0d1119',
              paddingBottom: 8, borderBottom: '1px solid #161d2b',
            }}>
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

interface ResultsViewProps {
  roundHistory: RoundSnapshot[];
  loadedMeta?: ModelLabProps['loadedMeta'];
}

function ResultsView({ roundHistory, loadedMeta }: ResultsViewProps) {
  const lossChartData = useMemo(() => {
    const rows: { iter: number; loss: number; valLoss?: number }[] = [];
    let cumIter = 0;
    for (const r of roundHistory) {
      for (const ev of r.totalLossCurve) {
        rows.push({
          iter: cumIter + ev.iter,
          loss: ev.loss,
          valLoss: ev.valLoss,
        });
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
        <Section title="Training loss (train vs val)">
          <div style={{ width: '100%', height: 160 }}>
            <ResponsiveContainer>
              <LineChart data={lossChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#223044" />
                <XAxis dataKey="iter" stroke="#cdd3de" fontSize={11} />
                <YAxis stroke="#cdd3de" fontSize={11} scale="log" domain={['auto', 'auto']} />
                <Tooltip contentStyle={{ background: '#141a26', border: '1px solid #223044', fontSize: 11 }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line type="monotone" dataKey="loss" name="train" stroke="#55dcff" dot={false} strokeWidth={2} />
                <Line type="monotone" dataKey="valLoss" name="val" stroke="#ffd070" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Section>
      )}

      <SplitRmsTable roundHistory={roundHistory} loadedMeta={loadedMeta} />

      <CoverageHeatmap roundHistory={roundHistory} />
    </>
  );
}

// ---------------------------------------------------------------------------
// Phase 0 — per-horizon train/val/test RMS table. Honest cross-phase
// progress surface: the columns are populated from `diag.perSplit` when
// the trial store carried `split` tags; otherwise the table renders the
// legacy heldOut row only.

function SplitRmsTable({
  roundHistory,
  loadedMeta,
}: {
  roundHistory: RoundSnapshot[];
  loadedMeta?: ModelLabProps['loadedMeta'];
}) {
  const finalDiag = roundHistory.length > 0 ? roundHistory[roundHistory.length - 1]!.diag : null;
  if (!finalDiag) return null;
  const headlineHorizons = finalDiag.openLoopDivergence.map((r) => r.tSec);
  if (headlineHorizons.length === 0) return null;
  const perSplit = finalDiag.perSplit;
  // Build rows: one per horizon. Columns: held-out (always), train, val, test (when perSplit available).
  const rows = headlineHorizons.map((tSec) => {
    const heldOut = finalDiag.openLoopDivergence.find((r) => r.tSec === tSec)?.posRms;
    const train = perSplit?.train?.find((r) => r.tSec === tSec)?.posRms;
    const val = perSplit?.val?.find((r) => r.tSec === tSec)?.posRms;
    const test = perSplit?.test?.find((r) => r.tSec === tSec)?.posRms;
    return { tSec, heldOut, train, val, test };
  });
  const fmt = (v?: number) => (v !== undefined && Number.isFinite(v) ? v.toFixed(3) + ' m' : '—');
  void loadedMeta; // not used yet
  return (
    <Section title="Per-horizon position RMS — train / val / test">
      <div style={{ fontSize: 11, lineHeight: 1.4 }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid #223044' }}>
              <th style={thStyle}>horizon</th>
              <th style={thStyle}>held-out</th>
              <th style={thStyle}>train</th>
              <th style={thStyle}>val</th>
              <th style={thStyle}>test</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.tSec} style={{ borderBottom: '1px solid #141a26' }}>
                <td style={tdStyle}>{r.tSec.toFixed(1)}s</td>
                <td style={tdStyle}>{fmt(r.heldOut)}</td>
                <td style={tdStyle}>{fmt(r.train)}</td>
                <td style={tdStyle}>{fmt(r.val)}</td>
                <td style={{ ...tdStyle, color: '#55dcff' }}>{fmt(r.test)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {!perSplit && (
          <div style={{ marginTop: 6, color: '#7a8290' }}>
            Per-split numbers populate when training uses split-tagged trials
            (Phase 0 of the training-dataset plan). Re-train via the maneuver
            pipeline to see them.
          </div>
        )}
      </div>
    </Section>
  );
}

const thStyle: React.CSSProperties = {
  textAlign: 'left',
  padding: '4px 6px',
  color: '#7a8290',
  fontWeight: 600,
};
const tdStyle: React.CSSProperties = {
  padding: '4px 6px',
};

// ---------------------------------------------------------------------------
// Phase 0 — coverage heatmap. Two-axis slice picker; cell color = log(count).
// Empty cells show in dark; hottest cells show in cyan. Pure visualization
// of the per-bin counts the meter produces.

function CoverageHeatmap({ roundHistory }: { roundHistory: RoundSnapshot[] }) {
  const lastWithCoverage = [...roundHistory].reverse().find((r) => r.coverage && r.coverage.length > 0);
  const [xAxisIdx, setXAxisIdx] = useState(0);
  const [yAxisIdx, setYAxisIdx] = useState(3); // yawRate default — Phase 1 gap is most visible here
  if (!lastWithCoverage || !lastWithCoverage.coverage) return null;
  const axes = CAR_COVERAGE_AXES;
  const cells = lastWithCoverage.coverage;
  // Project cells onto the chosen (x, y) sub-grid by summing counts across other axes.
  const xBins = axes[xAxisIdx]!.bins;
  const yBins = axes[yAxisIdx]!.bins;
  const grid: number[][] = Array.from({ length: yBins }, () => Array(xBins).fill(0));
  let maxCount = 0;
  for (const cell of cells) {
    const xi = cell.binIndex[xAxisIdx]!;
    const yi = cell.binIndex[yAxisIdx]!;
    if (xi < 0 || xi >= xBins || yi < 0 || yi >= yBins) continue;
    const row = grid[yi]!;
    row[xi] = (row[xi] ?? 0) + cell.count;
    if (row[xi]! > maxCount) maxCount = row[xi]!;
  }
  const cellW = 18;
  const cellH = 14;
  const color = (count: number): string => {
    if (count === 0) return '#0d1119';
    const t = Math.log(count + 1) / Math.log(maxCount + 1);
    // Dark navy to cyan ramp.
    const r = Math.round(20 + (85 - 20) * t);
    const g = Math.round(30 + (220 - 30) * t);
    const b = Math.round(50 + (255 - 50) * t);
    return `rgb(${r},${g},${b})`;
  };
  return (
    <Section title={`Coverage heatmap — round ${lastWithCoverage.round + 1}`}>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 8, fontSize: 11 }}>
        <label>
          X axis:{' '}
          <select value={xAxisIdx} onChange={(e) => setXAxisIdx(Number(e.target.value))} style={selectStyle}>
            {axes.map((a, i) => (
              <option key={a.name} value={i}>{a.name}</option>
            ))}
          </select>
        </label>
        <label>
          Y axis:{' '}
          <select value={yAxisIdx} onChange={(e) => setYAxisIdx(Number(e.target.value))} style={selectStyle}>
            {axes.map((a, i) => (
              <option key={a.name} value={i}>{a.name}</option>
            ))}
          </select>
        </label>
        <span style={{ color: '#7a8290' }}>max cell count = {maxCount}</span>
      </div>
      {xAxisIdx === yAxisIdx ? (
        <div style={{ fontSize: 11, color: '#7a8290' }}>Pick two different axes.</div>
      ) : (
        <svg width={xBins * cellW + 60} height={yBins * cellH + 24} style={{ fontSize: 10 }}>
          <g transform="translate(40, 4)">
            {grid.map((row, yi) =>
              row.map((count, xi) => (
                <g key={`${xi}-${yi}`}>
                  <rect
                    x={xi * cellW}
                    y={(yBins - 1 - yi) * cellH}
                    width={cellW - 1}
                    height={cellH - 1}
                    fill={color(count)}
                  >
                    <title>{`${axes[xAxisIdx]!.name}=${axes[xAxisIdx]!.lo + (xi + 0.5) * (axes[xAxisIdx]!.hi - axes[xAxisIdx]!.lo) / xBins}, ${axes[yAxisIdx]!.name}=${axes[yAxisIdx]!.lo + (yi + 0.5) * (axes[yAxisIdx]!.hi - axes[yAxisIdx]!.lo) / yBins}, n=${count}`}</title>
                  </rect>
                </g>
              )),
            )}
            <text x={(xBins * cellW) / 2} y={yBins * cellH + 16} textAnchor="middle" fill="#cdd3de">
              {axes[xAxisIdx]!.name} →
            </text>
            <text x={-6} y={(yBins * cellH) / 2} textAnchor="end" fill="#cdd3de">
              ↑ {axes[yAxisIdx]!.name}
            </text>
          </g>
        </svg>
      )}
    </Section>
  );
}

const selectStyle: React.CSSProperties = {
  background: '#0d1119',
  color: '#cdd3de',
  border: '1px solid #223044',
  borderRadius: 4,
  padding: '2px 4px',
  fontSize: 11,
};

// ---------------------------------------------------------------------------
// Small style helpers — keep dependencies minimal (no styled-components etc.)

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

// Live progress block shown during training in the overlay. Shows the
// current phase with a pulsing activity dot, a per-phase progress bar,
// elapsed time + ETA, and an overall progress bar so the user always
// has something visibly moving even during long Nelder-Mead fits.

const OVERLAY_PHASE_COLORS = {
  starting: '#cdd3de',
  initializing: '#7fbfff',
  collecting: '#ffd070',
  parametric: '#55dcff',
  residual: '#a888ff',
  evaluating: '#55ff88',
} as const;

const OVERLAY_PHASE_WEIGHTS = {
  collecting: 0.20,
  parametric: 0.60,
  residual: 0.15,
  evaluating: 0.05,
};

function ProgressBar({
  phase, trialsAttemptedThisRound, trialsTargetThisRound, iter, iterTotal,
  startedAt, round, rounds,
}: {
  phase: Phase;
  trialsAttemptedThisRound: number;
  trialsTargetThisRound: number;
  iter: number;
  iterTotal: number;
  startedAt: number | null;
  round: number;
  rounds: number;
}) {
  let fraction = 0;
  let label = '';
  const color = OVERLAY_PHASE_COLORS[(phase ?? 'starting') as keyof typeof OVERLAY_PHASE_COLORS];
  if (phase === 'initializing') {
    fraction = 0.15;
    label = 'loading Rapier WASM…';
  } else if (phase === 'collecting' && trialsTargetThisRound > 0) {
    fraction = trialsAttemptedThisRound / trialsTargetThisRound;
    label = `${trialsAttemptedThisRound} / ${trialsTargetThisRound} trials`;
  } else if ((phase === 'parametric' || phase === 'residual') && iterTotal > 0) {
    fraction = (iter + 1) / iterTotal;
    label = `iter ${iter + 1} / ${iterTotal}`;
  } else if (phase === 'evaluating') {
    fraction = 0.5;
    label = 'computing diagnostics…';
  } else {
    label = 'starting…';
  }
  fraction = Math.max(0, Math.min(1, fraction));

  // Overall progress estimate (weighted by phase) for the secondary bar.
  const intra = (() => {
    const isLast = round === rounds - 1;
    const w = OVERLAY_PHASE_WEIGHTS;
    const total = isLast
      ? w.collecting + w.parametric + w.residual + w.evaluating
      : w.collecting + w.parametric + w.evaluating;
    let done = 0;
    if (phase === 'collecting') done = w.collecting * fraction;
    else if (phase === 'parametric') done = w.collecting + w.parametric * fraction;
    else if (phase === 'residual') done = w.collecting + w.parametric + w.residual * fraction;
    else if (phase === 'evaluating') {
      done = w.collecting + w.parametric + (isLast ? w.residual : 0) + w.evaluating * fraction;
    }
    return done / total;
  })();
  const overall = Math.max(0, Math.min(1, (round + intra) / rounds));

  // Heartbeat tick so elapsed + ETA update even between training events.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(id);
  }, []);
  const elapsedMs = startedAt ? Math.max(0, now - startedAt) : 0;
  const etaMs = overall > 0.02 ? Math.max(0, elapsedMs / overall - elapsedMs) : null;
  const fmt = (ms: number) => {
    const s = Math.round(ms / 1000);
    const m = Math.floor(s / 60);
    const r = s % 60;
    return m === 0 ? `${r}s` : `${m}:${String(r).padStart(2, '0')}`;
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11 }}>
        <span style={{
          width: 8, height: 8, borderRadius: '50%', background: color,
          boxShadow: `0 0 6px ${color}`,
          animation: 'modelLabPulse 1.1s ease-in-out infinite',
        }} />
        <span style={{ color, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 0.4 }}>
          {phase ?? 'starting'}
        </span>
        <span style={{ opacity: 0.85 }}>{label}</span>
      </div>
      <ProgressTrackOverlay fraction={fraction} color={color} />
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, opacity: 0.75 }}>
        <span>overall {Math.round(overall * 100)}%</span>
        <span>
          elapsed <span style={{ color: '#cdeaff' }}>{fmt(elapsedMs)}</span>
          {etaMs !== null && <> · ETA <span style={{ color: '#cdeaff' }}>~{fmt(etaMs)}</span></>}
        </span>
      </div>
      <ProgressTrackOverlay fraction={overall} color="#55dcff" thin />
    </div>
  );
}

// Inject pulse + shimmer keyframes once. Idempotent across both ModelLab
// (raceprimitives overlay) and TrainingControls (model-lab dashboard).
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

function ProgressTrackOverlay({ fraction, color, thin }: { fraction: number; color: string; thin?: boolean }) {
  return (
    <div style={{
      position: 'relative',
      height: thin ? 3 : 6, borderRadius: 3, overflow: 'hidden',
      background: 'rgba(255, 255, 255, 0.06)',
    }}>
      <div style={{
        width: `${fraction * 100}%`, height: '100%',
        background: color,
        transition: 'width 200ms linear',
      }} />
      {!thin && (
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
