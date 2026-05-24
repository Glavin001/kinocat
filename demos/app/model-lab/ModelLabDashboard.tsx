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
  VehicleState,
  WheeledControls,
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
  buildKinematicLibrary,
  buildLearnedRaceLibraryV2,
  RACE_START_SPEEDS,
} from '../lib/race-primitives-scenarios';
import { computeGroundTruthDots, computeUncertaintyHalos } from '../lib/fan-plot-ground-truth';

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
  const [selectedTrial, setSelectedTrial] = useState<Trial<VehicleState, WheeledControls, LearnableVehicleConfig> | null>(null);
  const [extraTrack, setExtraTrack] = useState<{ name: string; color: string; states: VehicleState[]; times: number[] } | null>(null);

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

      <Section title="Action space at a speed bucket"
        subtitle="The fan plot shows where the model thinks each control will end up after 0.55–1.5s. White dots are the Rapier ground-truth endpoints for the same controls; arrows = per-control error. Cyan halos = ensemble uncertainty (1σ).">
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
  const [showGT, setShowGT] = useState(false);
  const [computing, setComputing] = useState(false);
  const cacheRef = useRef<Map<string, { dx: number; dz: number }>>(new Map());

  const kinematicLib = useMemo(() => buildKinematicLibrary(), []);
  const v2Lib = useMemo(() => (model ? buildLearnedRaceLibraryV2(model) : null), [model]);
  const kinAtSpeed = useMemo(() => kinematicLib.lookup(selectedSpeed), [kinematicLib, selectedSpeed]);
  const v2AtSpeed = useMemo(() => (v2Lib ? v2Lib.lookup(selectedSpeed) : null), [v2Lib, selectedSpeed]);

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
      setShowGT(true);
    } finally {
      setComputing(false);
    }
  };

  // Reset GT when speed changes (it's bucket-specific).
  useEffect(() => {
    setGroundTruth([]);
    setShowGT(false);
  }, [selectedSpeed]);

  const extent = useMemo(() => {
    let xMin = -1, xMax = 1, zMin = -1, zMax = 1;
    const consume = (prims: ReadonlyArray<MotionPrimitive>) => {
      for (const p of prims) for (const s of p.sweep) {
        if (s.x < xMin) xMin = s.x;
        if (s.x > xMax) xMax = s.x;
        if (s.z < zMin) zMin = s.z;
        if (s.z > zMax) zMax = s.z;
      }
    };
    consume(kinAtSpeed);
    if (v2AtSpeed) consume(v2AtSpeed);
    // Also widen for GT dots so arrows stay visible.
    if (showGT) {
      for (const g of groundTruth) {
        if (g.dx < xMin) xMin = g.dx;
        if (g.dx > xMax) xMax = g.dx;
        if (g.dz < zMin) zMin = g.dz;
        if (g.dz > zMax) zMax = g.dz;
      }
    }
    const pad = 1.5;
    return { xMin: xMin - pad, xMax: xMax + pad, zMin: zMin - pad, zMax: zMax + pad };
  }, [kinAtSpeed, v2AtSpeed, groundTruth, showGT]);

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
            padding: '4px 10px', borderRadius: 4, cursor: 'pointer',
            background: showGT ? '#1a2030' : '#0d1119',
            border: '1px solid #1f2735', color: '#cdd3de',
            font: '11px ui-monospace, monospace',
          }}
        >
          {computing ? 'running Rapier…' : showGT ? 'refresh ground truth' : 'overlay Rapier ground truth'}
        </button>
        <label style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, opacity: 0.7 }}>
          <input
            type="checkbox"
            checked={showGT}
            onChange={(e) => setShowGT(e.target.checked)}
            disabled={groundTruth.length === 0}
          />
          show GT
        </label>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1fr) minmax(0,1fr)', gap: 12 }}>
        <PrimitiveFanPlot
          primitives={kinAtSpeed}
          forwardColor="#ff8aa0"
          title={`KINEMATIC · ${selectedSpeed} m/s`}
          subtitle={`${kinAtSpeed.length} primitives`}
          fixedExtent={extent}
        />
        {v2AtSpeed ? (
          <PrimitiveFanPlot
            primitives={v2AtSpeed}
            forwardColor="#55dcff"
            title={`V2 LEARNED · ${selectedSpeed} m/s`}
            subtitle={`${v2AtSpeed.length} primitives${uncertainty.length > 0 ? ' · uncertainty halos' : ''}`}
            fixedExtent={extent}
            groundTruth={showGT ? groundTruth : undefined}
            uncertainty={uncertainty}
          />
        ) : (
          <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            border: '1px dashed #1f2735', borderRadius: 6, padding: 24,
            opacity: 0.6, fontSize: 12, textAlign: 'center', minHeight: 200,
          }}>
            No v2 model — train one above to see its action space.
          </div>
        )}
      </div>
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
