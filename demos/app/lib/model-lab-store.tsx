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
  CarKinematicState,
  WheeledCarControls,
  LearnableVehicleConfig,
} from 'kinocat/agent';
import type {
  ModelDiagnostics,
  FitProgressEvent,
  TrialStore,
} from 'kinocat/learning';
import type { HeadlessTrialHarness } from 'kinocat/adapters/rapier';
import { ensureRapier } from 'kinocat/adapters/rapier';
import {
  runOfflineTraining,
  createScenarioHarness,
  type TrainingEvent,
  type RunOfflineTrainingResult,
} from './training-driver';
import {
  loadV2Model,
  loadV2ModelWithFallback,
  saveV2Model,
  clearV2Model,
  type PersistedV2Model,
} from './v2-model-persistence';

export interface FitCurvePoint {
  iter: number;
  /** Raw optimizer loss (parametric = SUM across all sample residuals,
   *  residual = per-sample MSE). Different magnitudes across phases. */
  loss: number;
  /** Per-sample mean loss — comparable across rounds AND across phases.
   *  Prefer this for any cross-round / cross-phase chart. */
  lossNormalized: number;
  phase: 'parametric' | 'residual';
  /** Which training round this point came from (0-based). */
  round: number;
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
  trialStore: TrialStore<CarKinematicState, WheeledCarControls, LearnableVehicleConfig> | null;
  harness: HeadlessTrialHarness | null;
  config: LearnableVehicleConfig | null;
  sampleDt: number;
  status: 'idle' | 'running' | 'done' | 'error';
  currentRound: number;
  rounds: number;
  trialsCollected: number;
  trialsDiscarded: number;
  /** Cumulative trial slots attempted in the current round. */
  trialsAttemptedThisRound: number;
  /** Target trial count for the current round (cells.length). */
  trialsTargetThisRound: number;
  latestLoss: number | null;
  currentPhase: 'initializing' | 'collecting' | 'parametric' | 'residual' | 'evaluating' | null;
  /** Most-recent fit iteration (0-based). Resets per phase. */
  currentIter: number;
  /** Configured maximum iterations for the current fit phase. */
  currentIterTotal: number;
  /** Epoch (Date.now()) when the current training run started, or null
   *  when idle. Used by the UI to render elapsed time + ETA. */
  trainingStartedAt: number | null;
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
  const [trialsAttemptedThisRound, setTrialsAttemptedThisRound] = useState(0);
  const [trialsTargetThisRound, setTrialsTargetThisRound] = useState(0);
  const [latestLoss, setLatestLoss] = useState<number | null>(null);
  const [currentPhase, setCurrentPhase] = useState<ModelLabState['currentPhase']>(null);
  const [currentIter, setCurrentIter] = useState(0);
  const [currentIterTotal, setCurrentIterTotal] = useState(0);
  const [trainingStartedAt, setTrainingStartedAt] = useState<number | null>(null);
  const [roundHistory, setRoundHistory] = useState<RoundSnapshot[]>([]);
  const [liveFitCurve, setLiveFitCurve] = useState<FitCurvePoint[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [settings, setSettingsState] = useState<TrainingSettings>(DEFAULT_TRAINING_SETTINGS);

  const cancelRef = useRef<{ cancelled: boolean }>({ cancelled: false });
  const liveCurveRef = useRef<FitCurvePoint[]>([]);
  const flushTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Load any previously-saved model on mount AND pre-warm the Rapier
  // WASM module in the background. Cold-loading Rapier takes 1-3 s, and
  // doing it eagerly here means the first "Train" click doesn't sit at
  // a frozen "initializing" state — the harness builder finishes
  // almost instantly once `ensureRapier()` is cached.
  useEffect(() => {
    let cancelled = false;
    // Synchronous localStorage hit paints immediately; otherwise fall back to
    // the shipped overnight-trained artifact (`/models/v2-default.json`) so a
    // fresh visitor still sees the parametric-vs-learned-vs-Rapier comparison
    // without having to train a model first.
    const cached = loadV2Model();
    if (cached) {
      setModel(cached.model);
      setMeta(cached.meta);
      setConfig(cached.model.config);
    } else {
      void loadV2ModelWithFallback().then((loaded) => {
        if (cancelled || !loaded) return;
        setModel(loaded.model);
        setMeta(loaded.meta);
        setConfig(loaded.model.config);
      }).catch(() => { /* no shipped model — train path still works */ });
    }
    void ensureRapier().catch(() => { /* surface lazily on first train */ });
    return () => {
      cancelled = true;
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
    setTrialsAttemptedThisRound(0);
    setTrialsTargetThisRound(0);
    setLatestLoss(null);
    // Start in `initializing` so the user sees a colored, animated
    // progress indicator immediately when they hit Train (the actual
    // training pipeline still emits its own `phase` events from
    // `runOfflineTraining`, which overwrite this).
    setCurrentPhase('initializing');
    setCurrentIter(0);
    setCurrentIterTotal(0);
    setTrainingStartedAt(Date.now());
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
          trialsAccum += e.collected;
          setTrialsCollected(trialsAccum);
          setTrialsDiscarded((d) => d + e.discarded);
          if (typeof e.runSoFar === 'number') setTrialsAttemptedThisRound(e.runSoFar);
          if (typeof e.runTarget === 'number') setTrialsTargetThisRound(e.runTarget);
          break;
        case 'fit-progress': {
          const normalized = e.event.lossNormalized ?? e.event.loss;
          const point: FitCurvePoint = {
            iter: e.event.iter,
            loss: e.event.loss,
            lossNormalized: normalized,
            phase: e.phase,
            round: e.round,
          };
          perRoundCurve.push(point);
          liveCurveRef.current.push(point);
          // Headline "latest loss" should be the per-sample mean so the
          // number is meaningful (not a scary 10000+ that's actually
          // just "10k samples × small per-sample error").
          setLatestLoss(normalized);
          setCurrentPhase(e.phase);
          if (typeof e.iterIndex === 'number') setCurrentIter(e.iterIndex);
          if (typeof e.iterTotal === 'number') setCurrentIterTotal(e.iterTotal);
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
      // Yield once so React can flush the "initializing" state to the
      // DOM BEFORE we enter the long-running async training call. Without
      // this, the first paint can happen after Rapier WASM has already
      // started loading, making the card look frozen at "starting".
      await new Promise<void>((resolve) => setTimeout(resolve, 0));

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
    trialsCollected, trialsDiscarded,
    trialsAttemptedThisRound, trialsTargetThisRound,
    latestLoss, currentPhase, currentIter, currentIterTotal,
    trainingStartedAt,
    roundHistory, liveFitCurve, error, settings,
  }), [
    model, meta, trialStore, harness, config, sampleDt,
    status, currentRound, settings, trialsCollected, trialsDiscarded,
    trialsAttemptedThisRound, trialsTargetThisRound,
    latestLoss, currentPhase, currentIter, currentIterTotal,
    trainingStartedAt,
    roundHistory, liveFitCurve, error,
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
