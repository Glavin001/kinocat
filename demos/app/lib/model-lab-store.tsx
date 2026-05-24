'use client';

// In-memory React store for the /model-lab dashboard. Holds the most-
// recent training run's outputs:
//   - per-round snapshots (params + diagnostics + accumulated loss
//     curves), so we can show the model EVOLVING across rounds rather
//     than just the final scalar.
//   - the trial store (held-out trials drive RolloutPlayer + fan-plot
//     ground-truth dots).
//   - a long-lived Rapier headless harness so the Scenario Playground
//     can run on-demand trials without re-initializing the world each
//     click.
//
// Nothing here persists across reloads on purpose: trial samples can be
// several MB and would blow localStorage. The persisted v2 model (in
// `v2-model-persistence.ts`) is enough to re-derive the headline
// numbers; rollouts and per-control GT dots require re-collecting
// trials (one click).

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import type {
  LearnedVehicleModel,
  LearnedVehicleParamsV2,
  VehicleState,
  WheeledControls,
  LearnableVehicleConfig,
} from 'kinocat/agent';
import type {
  ModelDiagnostics,
  FitProgressEvent,
  TrialStore,
} from 'kinocat/learning';
import type { HeadlessTrialHarness } from 'kinocat/adapters/rapier';
import {
  runOfflineTraining,
  createScenarioHarness,
  type TrainingEvent,
  type RunOfflineTrainingResult,
} from './training-driver';
import {
  loadV2Model,
  saveV2Model,
  clearV2Model,
  type PersistedV2Model,
} from './v2-model-persistence';

export interface FitCurvePoint {
  iter: number;
  loss: number;
  phase: 'parametric' | 'residual';
}

export interface RoundSnapshot {
  round: number;
  trialsAfter: number;
  params: LearnedVehicleParamsV2;
  diagnostics: ModelDiagnostics;
  /** Concatenated per-round (parametric + residual) loss curve. */
  fitCurve: FitCurvePoint[];
}

export interface TrainingSettings {
  rounds: number;
  trialsPerRound: number;
  trialTicks: number;
  seed: number;
}

export const DEFAULT_TRAINING_SETTINGS: TrainingSettings = {
  rounds: 3,
  trialsPerRound: 48,
  trialTicks: 120,
  seed: 42,
};

export interface ModelLabState {
  model: LearnedVehicleModel | null;
  meta: PersistedV2Model['meta'] | null;
  trialStore: TrialStore<VehicleState, WheeledControls, LearnableVehicleConfig> | null;
  harness: HeadlessTrialHarness | null;
  config: LearnableVehicleConfig | null;
  sampleDt: number;
  status: 'idle' | 'running' | 'done' | 'error';
  currentRound: number;
  rounds: number;
  trialsCollected: number;
  trialsDiscarded: number;
  latestLoss: number | null;
  currentPhase: 'parametric' | 'residual' | null;
  roundHistory: RoundSnapshot[];
  liveFitCurve: FitCurvePoint[];
  error: string | null;
  settings: TrainingSettings;
}

export interface ModelLabApi {
  state: ModelLabState;
  setSettings: (s: Partial<TrainingSettings>) => void;
  train: () => Promise<void>;
  cancel: () => void;
  clearTrained: () => void;
  /** Lazily ensure a headless harness exists for Scenario Playground. */
  ensureHarness: () => Promise<{ harness: HeadlessTrialHarness; config: LearnableVehicleConfig }>;
}

const ModelLabContext = createContext<ModelLabApi | null>(null);

export function ModelLabProvider({ children }: { children: ReactNode }) {
  const [model, setModel] = useState<LearnedVehicleModel | null>(null);
  const [meta, setMeta] = useState<PersistedV2Model['meta'] | null>(null);
  const [trialStore, setTrialStore] = useState<ModelLabState['trialStore']>(null);
  const [harness, setHarness] = useState<HeadlessTrialHarness | null>(null);
  const [config, setConfig] = useState<LearnableVehicleConfig | null>(null);
  const [sampleDt, setSampleDt] = useState(0.1);
  const [status, setStatus] = useState<ModelLabState['status']>('idle');
  const [currentRound, setCurrentRound] = useState(0);
  const [trialsCollected, setTrialsCollected] = useState(0);
  const [trialsDiscarded, setTrialsDiscarded] = useState(0);
  const [latestLoss, setLatestLoss] = useState<number | null>(null);
  const [currentPhase, setCurrentPhase] = useState<'parametric' | 'residual' | null>(null);
  const [roundHistory, setRoundHistory] = useState<RoundSnapshot[]>([]);
  const [liveFitCurve, setLiveFitCurve] = useState<FitCurvePoint[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [settings, setSettingsState] = useState<TrainingSettings>(DEFAULT_TRAINING_SETTINGS);

  const cancelRef = useRef<{ cancelled: boolean }>({ cancelled: false });
  const liveCurveRef = useRef<FitCurvePoint[]>([]);
  const flushTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Load any previously-saved model on mount.
  useEffect(() => {
    const cached = loadV2Model();
    if (cached) {
      setModel(cached.model);
      setMeta(cached.meta);
      setConfig(cached.model.config);
    }
    return () => {
      // Best-effort harness cleanup on unmount.
      try { harness?.dispose(); } catch { /* noop */ }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const setSettings = useCallback((s: Partial<TrainingSettings>) => {
    setSettingsState((prev) => ({ ...prev, ...s }));
  }, []);

  const cancel = useCallback(() => {
    cancelRef.current.cancelled = true;
    setStatus('idle');
  }, []);

  const clearTrained = useCallback(() => {
    clearV2Model();
    setModel(null);
    setMeta(null);
    setTrialStore(null);
    setRoundHistory([]);
    setLiveFitCurve([]);
    liveCurveRef.current = [];
  }, []);

  const ensureHarness = useCallback(async () => {
    if (harness && config) return { harness, config };
    const built = await createScenarioHarness();
    setHarness(built.harness);
    setConfig((c) => c ?? built.config);
    return built;
  }, [harness, config]);

  const flushLiveCurve = useCallback(() => {
    setLiveFitCurve([...liveCurveRef.current]);
    flushTimerRef.current = null;
  }, []);

  const train = useCallback(async () => {
    cancelRef.current = { cancelled: false };
    liveCurveRef.current = [];
    setLiveFitCurve([]);
    setError(null);
    setStatus('running');
    setCurrentRound(0);
    setTrialsCollected(0);
    setTrialsDiscarded(0);
    setLatestLoss(null);
    setCurrentPhase(null);
    setRoundHistory([]);

    // Per-round accumulators so each snapshot owns its own curve copy.
    let perRoundCurve: FitCurvePoint[] = [];
    let trialsAccum = 0;
    let pendingDiag: ModelDiagnostics | null = null;

    const onEvent = (e: TrainingEvent) => {
      if (cancelRef.current.cancelled) return;
      switch (e.type) {
        case 'round-start':
          setCurrentRound(e.round);
          perRoundCurve = [];
          pendingDiag = null;
          setCurrentPhase('parametric');
          break;
        case 'trial-batch':
          trialsAccum += e.collected;
          setTrialsCollected(trialsAccum);
          setTrialsDiscarded((d) => d + e.discarded);
          break;
        case 'fit-progress': {
          const point: FitCurvePoint = {
            iter: e.event.iter,
            loss: e.event.loss,
            phase: e.phase,
          };
          perRoundCurve.push(point);
          liveCurveRef.current.push(point);
          setLatestLoss(e.event.loss);
          setCurrentPhase(e.phase);
          // Throttle live curve flushes — Recharts is heavy.
          if (flushTimerRef.current === null) {
            flushTimerRef.current = setTimeout(flushLiveCurve, 100);
          }
          break;
        }
        case 'evaluation':
          pendingDiag = e.diagnostics;
          break;
        case 'round-end': {
          const diag = pendingDiag ?? e.diagnostics;
          setRoundHistory((h) => [
            ...h,
            {
              round: e.round,
              trialsAfter: e.trialsAfter,
              params: e.params,
              diagnostics: diag,
              fitCurve: [...perRoundCurve],
            },
          ]);
          break;
        }
        case 'done':
          break;
      }
    };

    try {
      // If a harness already exists, dispose it before training builds a
      // fresh one with the canonical vehicle config.
      try { harness?.dispose(); } catch { /* noop */ }
      setHarness(null);

      const result: RunOfflineTrainingResult = await runOfflineTraining({
        rounds: settings.rounds,
        trialsPerActiveRound: settings.trialsPerRound,
        trialTicks: settings.trialTicks,
        sampleEveryNTicks: 6,
        seed: settings.seed,
        keepHarness: true,
        onEvent,
      });
      if (cancelRef.current.cancelled) {
        try { result.harness?.dispose(); } catch { /* noop */ }
        setStatus('idle');
        return;
      }
      // Final flush.
      if (flushTimerRef.current !== null) {
        clearTimeout(flushTimerRef.current);
        flushTimerRef.current = null;
      }
      setLiveFitCurve([...liveCurveRef.current]);

      // Persist the model + meta so reloads keep it.
      const final = result.finalDiagnostics;
      const at1s = (rows: { tSec: number; posRms: number }[] | undefined) =>
        rows?.find((r) => r.tSec >= 1.0)?.posRms ?? 0;
      const newMeta: PersistedV2Model['meta'] = {
        trialsUsed: result.trials.size(),
        openLoopRmsAt1s: at1s(final.openLoopDivergence),
        legacyRmsAt1s: at1s(final.baselines['legacyV1']),
        kinematicRmsAt1s: at1s(final.baselines['kinematic']),
        createdAt: Date.now(),
      };
      saveV2Model(result.model, newMeta);
      setModel(result.model);
      setMeta(newMeta);
      setTrialStore(result.trials);
      setConfig(result.config);
      setSampleDt(result.sampleDt);
      if (result.harness) setHarness(result.harness);
      setStatus('done');
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus('error');
    }
  }, [settings, harness, flushLiveCurve]);

  const state: ModelLabState = useMemo(() => ({
    model, meta, trialStore, harness, config, sampleDt,
    status, currentRound, rounds: settings.rounds,
    trialsCollected, trialsDiscarded, latestLoss, currentPhase,
    roundHistory, liveFitCurve, error, settings,
  }), [
    model, meta, trialStore, harness, config, sampleDt,
    status, currentRound, settings, trialsCollected, trialsDiscarded,
    latestLoss, currentPhase, roundHistory, liveFitCurve, error,
  ]);

  const api: ModelLabApi = useMemo(() => ({
    state, setSettings, train, cancel, clearTrained, ensureHarness,
  }), [state, setSettings, train, cancel, clearTrained, ensureHarness]);

  return <ModelLabContext.Provider value={api}>{children}</ModelLabContext.Provider>;
}

export function useModelLab(): ModelLabApi {
  const ctx = useContext(ModelLabContext);
  if (!ctx) throw new Error('useModelLab must be used within ModelLabProvider');
  return ctx;
}
