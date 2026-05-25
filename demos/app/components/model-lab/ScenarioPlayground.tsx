'use client';

// "What would the model do here?" — drop a custom or canned scenario
// (initial speed + constant controls + duration), render the model's
// open-loop prediction immediately, and optionally run the same setup
// through Rapier on demand to overlay ground truth.
//
// The output is wired back to the parent's RolloutPlayer as a `Trial`-
// shaped synthetic + extra track so the existing canvas does all the
// rendering. Keeps the dashboard layout consistent.

import { useCallback, useMemo, useState } from 'react';
import type {
  LearnedVehicleModel,
  LearnableVehicleConfig,
  CarKinematicState,
  WheeledCarControls,
} from 'kinocat/agent';
import { learnedForwardSimV2 } from 'kinocat/agent';
import type { Trial } from 'kinocat/learning';
import type { HeadlessTrialHarness } from 'kinocat/adapters/rapier';

export interface CannedScenario {
  id: string;
  label: string;
  startSpeed: number;
  steer: number;
  driveForce: number;
  brakeForce: number;
  durationSec: number;
  description?: string;
}

const PHYSICS_DT = 1 / 60;

export const CANNED_SCENARIOS: CannedScenario[] = [
  { id: 'brake-high', label: 'hard brake from 28 m/s', startSpeed: 28, steer: 0, driveForce: 0, brakeForce: 2000, durationSec: 2.5, description: 'Tests longitudinal decel modeling at high speed.' },
  { id: 'snap-left-low', label: 'snap left @ 8 m/s', startSpeed: 8, steer: -0.6, driveForce: 1200, brakeForce: 0, durationSec: 2.0, description: 'Low-speed full-lock — agile regime.' },
  { id: 'trail-brake-r', label: 'trail brake right from 22', startSpeed: 22, steer: 0.18, driveForce: 0, brakeForce: 600, durationSec: 2.0, description: 'Combined braking + cornering load transfer.' },
  { id: 'top-gentle-r', label: 'gentle right @ 28', startSpeed: 28, steer: 0.09, driveForce: 0, brakeForce: 0, durationSec: 2.0, description: 'Friction circle should clamp; v2 should plan a wider arc than kinematic.' },
  { id: 'coast', label: 'coast from 20', startSpeed: 20, steer: 0, driveForce: 0, brakeForce: 0, durationSec: 3.0, description: 'Pure rolling resistance / drag.' },
  { id: 'reverse', label: 'gentle reverse turn', startSpeed: 0, steer: 0.3, driveForce: -1500, brakeForce: 0, durationSec: 2.0, description: 'Reverse + steering.' },
];

export interface ScenarioPlaygroundProps {
  model: LearnedVehicleModel | null;
  config: LearnableVehicleConfig | null;
  ensureHarness: () => Promise<{ harness: HeadlessTrialHarness; config: LearnableVehicleConfig }>;
  /** Push a synthesized trial up so the parent can hand it to
   *  `RolloutPlayer.trial`. */
  onScenarioReady: (
    trial: Trial<CarKinematicState, WheeledCarControls, LearnableVehicleConfig> | null,
    extra?: { name: string; color: string; states: CarKinematicState[]; times: number[] } | null,
  ) => void;
}

interface ScenarioInputs {
  startSpeed: number;
  steer: number;
  driveForce: number;
  brakeForce: number;
  durationSec: number;
}

const DEFAULT_INPUTS: ScenarioInputs = {
  startSpeed: 16,
  steer: 0.2,
  driveForce: 1500,
  brakeForce: 0,
  durationSec: 2.0,
};

function buildSyntheticTrial(
  inputs: ScenarioInputs,
  model: LearnedVehicleModel,
  config: LearnableVehicleConfig,
): Trial<CarKinematicState, WheeledCarControls, LearnableVehicleConfig> {
  // Build a "fake" trial whose `samples` are the model's own prediction
  // so RolloutPlayer treats the model as ground truth visually. The
  // Rapier overlay (extra track) is what the user then compares
  // against — the more they diverge, the worse the model's prediction
  // for that scenario.
  const ticks = Math.max(1, Math.round(inputs.durationSec / PHYSICS_DT));
  const controls: WheeledCarControls[] = Array.from({ length: ticks }, () => ({
    steer: inputs.steer, driveForce: inputs.driveForce, brakeForce: inputs.brakeForce,
  }));
  const sim = learnedForwardSimV2(model);
  const sampleStride = 6;
  let s: CarKinematicState = {
    x: 0, z: 0, heading: 0, speed: inputs.startSpeed, t: 0,
    yawRate: 0, lateralVelocity: 0,
  };
  const samples: { t: number; state: CarKinematicState }[] = [{ t: 0, state: s }];
  for (let i = 0; i < ticks; i++) {
    const c = controls[i]!;
    s = sim(s, [c.steer, c.driveForce, c.brakeForce], PHYSICS_DT);
    if ((i + 1) % sampleStride === 0) {
      samples.push({ t: (i + 1) * PHYSICS_DT, state: { ...s } });
    }
  }
  return {
    id: `scenario-${Date.now()}`,
    initialState: samples[0]!.state,
    controlsTrace: controls,
    dt: PHYSICS_DT,
    samples,
    config,
    configKey: 'scenario',
  };
}

function runRapierScenario(
  harness: HeadlessTrialHarness,
  inputs: ScenarioInputs,
): { states: CarKinematicState[]; times: number[] } | null {
  const ticks = Math.max(1, Math.round(inputs.durationSec / PHYSICS_DT));
  const trace = Array.from({ length: ticks }, () => ({
    steer: inputs.steer, driveForce: inputs.driveForce, brakeForce: inputs.brakeForce,
  }));
  const result = harness.runTrial({
    pose: { x: 0, z: 0, heading: 0 },
    kin: { forwardSpeed: inputs.startSpeed },
    controlsTrace: trace,
    sampleEveryNTicks: 6,
    id: `rapier-${Date.now()}`,
  });
  if (!result.ok) return null;
  const states = result.trial.samples;
  const times = states.map((_, i) => i * 6 * PHYSICS_DT);
  return { states, times };
}

export function ScenarioPlayground({ model, config, ensureHarness, onScenarioReady }: ScenarioPlaygroundProps) {
  const [inputs, setInputs] = useState<ScenarioInputs>(DEFAULT_INPUTS);
  const [running, setRunning] = useState<'idle' | 'predicting' | 'rapier'>('idle');
  const [lastResult, setLastResult] = useState<{ predicted: boolean; rapierError?: string }>({ predicted: false });

  const onApplyCanned = useCallback((s: CannedScenario) => {
    setInputs({
      startSpeed: s.startSpeed, steer: s.steer, driveForce: s.driveForce, brakeForce: s.brakeForce,
      durationSec: s.durationSec,
    });
  }, []);

  const predict = useCallback(() => {
    if (!model || !config) return;
    setRunning('predicting');
    const trial = buildSyntheticTrial(inputs, model, config);
    onScenarioReady(trial, null);
    setLastResult({ predicted: true });
    setRunning('idle');
  }, [model, config, inputs, onScenarioReady]);

  const runRapier = useCallback(async () => {
    if (!model || !config) return;
    setRunning('rapier');
    try {
      const { harness } = await ensureHarness();
      const rapier = runRapierScenario(harness, inputs);
      const trial = buildSyntheticTrial(inputs, model, config);
      if (!rapier) {
        setLastResult({ predicted: true, rapierError: 'discarded (off-arena / spin)' });
        onScenarioReady(trial, null);
      } else {
        onScenarioReady(trial, {
          name: 'Rapier ground truth',
          color: '#ffffff',
          states: rapier.states,
          times: rapier.times,
        });
        setLastResult({ predicted: true });
      }
    } catch (e) {
      setLastResult({ predicted: true, rapierError: e instanceof Error ? e.message : String(e) });
    } finally {
      setRunning('idle');
    }
  }, [model, config, inputs, ensureHarness, onScenarioReady]);

  const disabled = !model || !config;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <strong style={{ color: '#7fd6ff', fontSize: 13 }}>Scenario playground</strong>
      <p style={{ opacity: 0.65, fontSize: 11, margin: 0 }}>
        Apply a canned scenario or pick custom controls + start speed. <em>Predict</em> rolls
        the v2 model forward; <em>Run in Rapier</em> additionally drops the ground-truth
        trajectory next to it. Their divergence = the model's blind spot in this regime.
      </p>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
        {CANNED_SCENARIOS.map((s) => (
          <button
            key={s.id}
            onClick={() => onApplyCanned(s)}
            title={s.description}
            style={chipStyle}
          >
            {s.label}
          </button>
        ))}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 8 }}>
        <NumField label="startSpeed (m/s)" value={inputs.startSpeed} min={-6} max={30} step={1}
          onChange={(v) => setInputs((p) => ({ ...p, startSpeed: v }))} />
        <NumField label="steer (rad)" value={inputs.steer} min={-0.6} max={0.6} step={0.01}
          onChange={(v) => setInputs((p) => ({ ...p, steer: v }))} />
        <NumField label="driveForce (N)" value={inputs.driveForce} min={-4000} max={4000} step={100}
          onChange={(v) => setInputs((p) => ({ ...p, driveForce: v }))} />
        <NumField label="brakeForce (N)" value={inputs.brakeForce} min={0} max={2000} step={50}
          onChange={(v) => setInputs((p) => ({ ...p, brakeForce: v }))} />
        <NumField label="duration (s)" value={inputs.durationSec} min={0.5} max={5} step={0.1}
          onChange={(v) => setInputs((p) => ({ ...p, durationSec: v }))} />
      </div>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
        <button onClick={predict} disabled={disabled || running !== 'idle'} style={primaryBtnStyle}>
          {running === 'predicting' ? 'predicting…' : 'predict with v2'}
        </button>
        <button onClick={runRapier} disabled={disabled || running !== 'idle'} style={btnStyle}>
          {running === 'rapier' ? 'running rapier…' : 'run in Rapier'}
        </button>
        {disabled && <span style={{ opacity: 0.55, fontSize: 11 }}>Train or load a model first.</span>}
        {lastResult.rapierError && (
          <span style={{ color: '#ff8aa0', fontSize: 11 }}>Rapier: {lastResult.rapierError}</span>
        )}
      </div>
    </div>
  );
}

function NumField({ label, value, min, max, step, onChange }:
  { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void }) {
  return (
    <label style={{ display: 'flex', flexDirection: 'column', gap: 3, fontSize: 11, opacity: 0.85 }}>
      <span>{label}</span>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{
          background: '#0d1119', color: '#cdeaff',
          border: '1px solid #1f2735', borderRadius: 4,
          padding: '4px 8px', font: '11px ui-monospace, monospace',
        }}
      />
    </label>
  );
}

const chipStyle: React.CSSProperties = {
  background: '#0d1119', color: '#cdd3de',
  border: '1px solid #1f2735', borderRadius: 14,
  padding: '4px 10px', fontSize: 11,
  cursor: 'pointer',
};

const btnStyle: React.CSSProperties = {
  background: '#1a2030', color: '#cdd3de',
  border: '1px solid #1f2735', borderRadius: 4,
  padding: '6px 12px', fontSize: 11,
  font: 'inherit', cursor: 'pointer',
};

const primaryBtnStyle: React.CSSProperties = {
  ...btnStyle,
  background: '#55dcff', color: '#0a0d14', borderColor: '#55dcff',
  fontWeight: 700,
};
