'use client';

// The first-class /model-lab dashboard. Provides:
//   - Training controls + live loss chart (TrainingControls)
//   - Headline strip (HeadlineStrip)
//   - Per-round evolution (RoundEvolutionTable)
//   - Action-space fan plot with ground-truth dots + uncertainty halos
//   - Coverage heatmap
//   - Trial Rollout Player (cached trials)
//   - Per-component RMS bar chart
//   - Scenario Playground (on-demand Rapier)
//
// Provider wraps the dashboard so all components share the same state.

import { useEffect, useMemo, useRef, useState } from 'react';
import type {
  LearnedVehicleModel,
  LearnableVehicleConfig,
  CarKinematicState,
  WheeledCarControls,
} from 'kinocat/agent';
import type { MotionPrimitive } from 'kinocat/primitives';
import type { Trial } from 'kinocat/learning';
import { ModelLabProvider, useModelLab } from '../lib/model-lab-store';
import { TrainingControls } from '../components/model-lab/TrainingControls';
import { HeadlineStrip } from '../components/model-lab/HeadlineStrip';
import { RoundEvolutionTable } from '../components/model-lab/RoundEvolutionTable';
import { CoverageHeatmap } from '../components/model-lab/CoverageHeatmap';
import { RolloutPlayer } from '../components/model-lab/RolloutPlayer';
import { PerComponentRmsChart } from '../components/model-lab/PerComponentRmsChart';
import { ScenarioPlayground } from '../components/model-lab/ScenarioPlayground';
import { PrimitiveFanPlot, type FanPlotGroundTruth, type FanPlotUncertainty } from '../components/PrimitiveFanPlot';
import {
  buildLearnedRaceLibraryV2,
  RACE_START_SPEEDS,
} from '../lib/race-primitives-scenarios';
import {
  computeGroundTruthDots,
  computeUncertaintyHalos,
  computeActionComparison,
  type ActionComparisonSummary,
} from '../lib/fan-plot-ground-truth';
import { ActionComparisonPanel } from '../components/model-lab/ActionComparisonPanel';
import { buildParametricOnlyModel } from 'kinocat/agent';

export default function ModelLabDashboard() {
  return (
    <ModelLabProvider>
      <DashboardInner />
    </ModelLabProvider>
  );
}

function DashboardInner() {
  const { state, ensureHarness } = useModelLab();
  const { model, meta, trialStore, harness, config, roundHistory } = state;

  const finalDiag = roundHistory.length > 0
    ? roundHistory[roundHistory.length - 1]!.diagnostics
    : null;

  const [selectedSpeed, setSelectedSpeed] = useState<number>(RACE_START_SPEEDS[2]!);
  const [selectedTrial, setSelectedTrial] = useState<Trial<CarKinematicState, WheeledCarControls, LearnableVehicleConfig> | null>(null);
  const [extraTrack, setExtraTrack] = useState<{ name: string; color: string; states: CarKinematicState[]; times: number[] } | null>(null);

  // Sync selectedTrial to the store contents when trials load.
  useEffect(() => {
    if (!selectedTrial && trialStore && trialStore.size() > 0) {
      // Default to the highest-error trial — that's the most informative.
      const all = trialStore.all();
      setSelectedTrial(all[Math.floor(all.length * 0.85)] ?? all[0] ?? null);
    }
  }, [trialStore, selectedTrial]);

  return (
    <main style={pageStyle}>
      <header style={headerStyle}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 16, flexWrap: 'wrap' }}>
          <a href="/" style={linkStyle}>← demos</a>
          <a href="/raceprimitives" style={linkStyle}>/raceprimitives</a>
          <a href="/sim-to-real" style={linkStyle}>/sim-to-real</a>
          <h1 style={{ fontSize: 20, margin: 0 }}>model lab · v2 training + diagnostics</h1>
          <span style={{ opacity: 0.6, fontSize: 12, marginLeft: 'auto' }}>
            {model ? 'model loaded' : 'no model trained'}
            {meta && (
              <> · trained {new Date(meta.createdAt).toLocaleString()}</>
            )}
          </span>
        </div>
        <p style={{ opacity: 0.7, fontSize: 12, margin: 0, maxWidth: 780, lineHeight: 1.5 }}>
          Train the v2 dynamics model and inspect it from every angle: loss curves,
          round-by-round evolution, action-space fan plots with Rapier ground-truth
          dots, ensemble uncertainty halos, coverage heatmap, full open-loop rollout
          playback, and a scenario playground for "what would the model do here?"
        </p>
      </header>

      <Section>
        <TrainingControls />
      </Section>

      <Section title="At a glance">
        <HeadlineStrip diag={finalDiag} fallback={meta} />
      </Section>

      <Section title="Per-round evolution" subtitle="Each round adds active-explored trials, refits the parametric backbone, and (on the final round) trains the residual MLP ensemble. Trend across rounds tells you whether the model is still learning.">
        <RoundEvolutionTable rounds={roundHistory} />
      </Section>

      <Section title="How right is the model? — actions graded against the real car"
        subtitle="Pick a speed, then “compare against real car”: every action the planner can take at that speed is rolled through Rapier, and the model's predicted endpoint is graded against where the chassis actually ends up. Each action gets a plain verdict — accurate, flagged-but-safe, or confidently wrong — so you can see at a glance which primitives the planner can trust and which carry hidden bias.">
        <FanPlotRow
          model={model}
          config={config}
          harness={harness}
          ensureHarness={ensureHarness}
          selectedSpeed={selectedSpeed}
          onSelectSpeed={setSelectedSpeed}
        />
      </Section>

      <Section title="Coverage heatmap"
        subtitle="Held-out RMS by (speed × steer) bin. Brighter red = worse error; cell counts show where data is thin. Click to jump the fan plot to that speed bucket.">
        <CoverageHeatmap
          diag={finalDiag}
          onSelect={(speedBin) => {
            // Map speedBin → start-speed bucket. Bins are [<=2, <=6, <=10, >10].
            const speedMap = [0, 4, 8, 20];
            setSelectedSpeed(speedMap[speedBin] ?? selectedSpeed);
          }}
        />
      </Section>

      <Section title="Trial rollout playback"
        subtitle="Pick a held-out trial; watch Rapier vs v2 (full) vs parametric-only vs kinematic diverge over time. Worst trials are sorted to the top by default — they show you what the model is missing.">
        <RolloutPlayer
          trial={selectedTrial}
          model={model}
          trials={trialStore?.all() ?? []}
          onSelectTrial={(t) => { setSelectedTrial(t); setExtraTrack(null); }}
          extraTrack={extraTrack}
        />
      </Section>

      <Section title="Per-component RMS"
        subtitle="Position is one metric. Heading, speed, yawRate, and lateral velocity show whether the model is wrong in distinct ways.">
        <PerComponentRmsChart diag={finalDiag} />
      </Section>

      <Section title="Scenario playground"
        subtitle="Synthesize an initial state + constant controls; the v2 model predicts the trajectory and (optionally) you run the same setup through Rapier to overlay ground truth. Useful for what-if questions the cached trials don't cover.">
        <ScenarioPlayground
          model={model}
          config={config}
          ensureHarness={ensureHarness}
          onScenarioReady={(trial, extra) => {
            setSelectedTrial(trial);
            setExtraTrack(extra ?? null);
          }}
        />
      </Section>
    </main>
  );
}

function FanPlotRow({
  model, config, harness, ensureHarness, selectedSpeed, onSelectSpeed,
}: {
  model: LearnedVehicleModel | null;
  config: LearnableVehicleConfig | null;
  harness: import('kinocat/adapters/rapier').HeadlessTrialHarness | null;
  ensureHarness: () => Promise<{ harness: import('kinocat/adapters/rapier').HeadlessTrialHarness; config: LearnableVehicleConfig }>;
  selectedSpeed: number;
  onSelectSpeed: (v: number) => void;
}) {
  const [groundTruth, setGroundTruth] = useState<FanPlotGroundTruth[]>([]);
  const [uncertainty, setUncertainty] = useState<FanPlotUncertainty[]>([]);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [computing, setComputing] = useState(false);
  const cacheRef = useRef<Map<string, { dx: number; dz: number }>>(new Map());

  const v2Lib = useMemo(() => (model ? buildLearnedRaceLibraryV2(model) : null), [model]);
  // Parametric-only (residual MLP stripped) — the safety floor the residual is
  // allowed to correct. Same control sets/order as the full v2 library.
  const paraLib = useMemo(
    () => (model ? buildLearnedRaceLibraryV2(buildParametricOnlyModel(model.params, model.config)) : null),
    [model],
  );
  const v2AtSpeed = useMemo(() => (v2Lib ? v2Lib.lookup(selectedSpeed) : null), [v2Lib, selectedSpeed]);
  const paraAtSpeed = useMemo(() => (paraLib ? paraLib.lookup(selectedSpeed) : null), [paraLib, selectedSpeed]);

  // Per-action comparison vs the Rapier ground truth (only once GT computed).
  const summary: ActionComparisonSummary | null = useMemo(() => {
    if (!model || !v2AtSpeed || !paraAtSpeed || groundTruth.length === 0) return null;
    return computeActionComparison({
      full: v2AtSpeed, parametric: paraAtSpeed, groundTruth, model,
      startSpeed: selectedSpeed,
    });
  }, [model, v2AtSpeed, paraAtSpeed, groundTruth, selectedSpeed]);

  // Recompute uncertainty when the speed bucket or model changes. Cheap.
  useEffect(() => {
    if (!model || !config || !v2AtSpeed) {
      setUncertainty([]);
      return;
    }
    setUncertainty(computeUncertaintyHalos({
      primitives: v2AtSpeed, model, config, startSpeed: selectedSpeed,
    }));
  }, [model, config, v2AtSpeed, selectedSpeed]);

  // Compute ground truth on demand.
  const hasGT = groundTruth.length > 0;
  const computeGT = async () => {
    if (!v2AtSpeed) return;
    setComputing(true);
    try {
      const { harness: h } = await ensureHarness();
      const gt = computeGroundTruthDots({
        primitives: v2AtSpeed, startSpeed: selectedSpeed,
        duration: v2AtSpeed[0]?.duration ?? 0.55,
        harness: h, cache: cacheRef.current,
      });
      setGroundTruth(gt);
    } finally {
      setComputing(false);
    }
  };

  // Reset GT + selection when speed changes (it's bucket-specific).
  useEffect(() => {
    setGroundTruth([]);
    setSelectedIndex(null);
  }, [selectedSpeed]);

  // Extent for the pre-GT "action-space shape" preview (v2 sweeps only).
  const extent = useMemo(() => {
    let xMin = -2, xMax = 2, zMin = -2, zMax = 2;
    if (v2AtSpeed) for (const p of v2AtSpeed) for (const s of p.sweep) {
      if (s.x < xMin) xMin = s.x;
      if (s.x > xMax) xMax = s.x;
      if (s.z < zMin) zMin = s.z;
      if (s.z > zMax) zMax = s.z;
    }
    const pad = 1.5;
    return { xMin: xMin - pad, xMax: xMax + pad, zMin: zMin - pad, zMax: zMax + pad };
  }, [v2AtSpeed]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap' }}>
        <span style={{ opacity: 0.6, fontSize: 11, marginRight: 4 }}>START SPEED</span>
        {RACE_START_SPEEDS.map((s) => (
          <button
            key={s}
            onClick={() => onSelectSpeed(s)}
            style={{
              padding: '4px 10px',
              background: selectedSpeed === s ? '#1a2030' : 'transparent',
              border: `1px solid ${selectedSpeed === s ? '#55dcff' : '#1f2735'}`,
              borderRadius: 4,
              color: selectedSpeed === s ? '#55dcff' : '#cdd3de',
              cursor: 'pointer',
              font: '11px ui-monospace, monospace',
            }}
          >
            {s} m/s
          </button>
        ))}
        <span style={{ flex: 1 }} />
        <button
          onClick={computeGT}
          disabled={!v2AtSpeed || computing}
          style={{
            padding: '4px 12px', borderRadius: 4, cursor: v2AtSpeed ? 'pointer' : 'default',
            background: hasGT ? '#0d1119' : '#13233a',
            border: `1px solid ${hasGT ? '#1f2735' : '#2a4a6e'}`,
            color: hasGT ? '#9aa6b2' : '#7fd6ff',
            font: '11px ui-monospace, monospace',
          }}
        >
          {computing ? 'running Rapier…' : hasGT ? 'recompute vs Rapier' : '▶ compare against real car (Rapier)'}
        </button>
      </div>

      {summary ? (
        <ActionComparisonPanel
          summary={summary}
          speed={selectedSpeed}
          selectedIndex={selectedIndex}
          onSelectIndex={setSelectedIndex}
        />
      ) : v2AtSpeed ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <PrimitiveFanPlot
            primitives={v2AtSpeed}
            forwardColor="#55dcff"
            title={`Action space · ${selectedSpeed} m/s`}
            subtitle={`${v2AtSpeed.length} primitives the planner can pick from${uncertainty.length > 0 ? ' · σ halos' : ''}`}
            fixedExtent={extent}
            uncertainty={uncertainty}
            highlightIndex={selectedIndex ?? undefined}
            onHover={setSelectedIndex}
          />
          <p style={{ margin: 0, fontSize: 11, opacity: 0.6, lineHeight: 1.5 }}>
            Each curve is one control held for the full primitive; the dot is where the
            model predicts the chassis ends up. Click{' '}
            <strong style={{ color: '#7fd6ff' }}>“compare against real car”</strong>{' '}
            to roll every action through Rapier and grade how right or wrong the model is.
          </p>
        </div>
      ) : (
        <NoModelPanel />
      )}
    </div>
  );
}

function NoModelPanel() {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      border: '1px dashed #1f2735', borderRadius: 6, padding: 24,
      opacity: 0.6, fontSize: 12, textAlign: 'center', minHeight: 200,
    }}>
      No v2 model — train one above to see its action space.
    </div>
  );
}


function Section({ title, subtitle, children }: { title?: string; subtitle?: string; children: React.ReactNode }) {
  return (
    <section style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {title && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <h2 style={{ fontSize: 14, margin: 0, color: '#7fd6ff', letterSpacing: 0.4 }}>{title}</h2>
          {subtitle && <p style={{ opacity: 0.6, fontSize: 11, margin: 0, lineHeight: 1.5, maxWidth: 760 }}>{subtitle}</p>}
        </div>
      )}
      {children}
    </section>
  );
}

const pageStyle: React.CSSProperties = {
  background: '#0a0d14', color: '#cdd3de',
  fontFamily: 'ui-monospace, monospace', fontSize: 12,
  minHeight: '100vh',
  padding: '20px 28px 40px',
  display: 'flex', flexDirection: 'column', gap: 20,
};

const headerStyle: React.CSSProperties = {
  display: 'flex', flexDirection: 'column', gap: 8,
  paddingBottom: 8, borderBottom: '1px solid #1f2735',
};

const linkStyle: React.CSSProperties = {
  color: '#7fd6ff', textDecoration: 'none', fontSize: 12,
};
