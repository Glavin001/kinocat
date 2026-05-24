// kinocat/training — generic offline-training contract.
//
// The car-specific training pipeline (currently in
// `demos/app/lib/training-driver.ts`) already operates on generic types
// `<CarKinematicState, WheeledCarControls, LearnableVehicleConfig>`. This module
// just formalizes the contract so future vehicle pipelines (airplane,
// hovercraft, ...) plug into the same orchestrator without each one
// re-implementing the round / trial-collect / fit-parametric / fit-residual
// / evaluate loop from scratch.
//
// Concrete pipelines (the v2 car pipeline first, an airplane pipeline next)
// implement `TrainingPipeline<S, C, P, Cfg>` and feed it to
// `runOfflineTraining`.

import type { ForwardSim } from '../primitives/types';
import type {
  Trial,
  TrialStore,
  ModelDiagnostics,
  FitProgressEvent,
} from '../learning';

export type {
  CoverageAxis,
  CoverageProjection,
  CoverageCellSummary,
  CoverageMeter,
  CoverageMeterOptions,
} from './coverage-meter';
export { createCoverageMeter } from './coverage-meter';

export type { ManeuverRunOptions } from './maneuver-runner';
export { runManeuver } from './maneuver-runner';

export type { BuildControlsTraceOptions } from './maneuver-trace';
export { buildControlsTrace } from './maneuver-trace';

export type {
  HardExampleMiner,
  HardExampleMinerOptions,
  MinerFrame,
} from './hard-example-miner';
export { createHardExampleMiner } from './hard-example-miner';

/** A learned model's fit, ready to plug into a planner. Concrete vehicle
 *  pipelines instantiate `model` with their own parametric + residual
 *  blocks; here we only assert it has a `ForwardSim<S>` for evaluation. */
export interface TrainedModel<S, P> {
  /** Hand-tuned (or fit) parametric coefficients. */
  params: P;
  /** Forward simulator for this model — what `evaluate` runs. */
  forwardSim: ForwardSim<S>;
}

/** Optional rolling progress event emitted from the orchestrator. Concrete
 *  pipelines may emit additional pipeline-specific events; this is the
 *  base type. */
export type TrainingEvent<S, C, P, Cfg> =
  | { type: 'round-start'; round: number; trialsBefore: number }
  | { type: 'trial-batch'; round: number; collected: number; discarded: number }
  | { type: 'fit-progress'; round: number; phase: 'parametric' | 'residual'; event: FitProgressEvent }
  | { type: 'evaluation'; round: number; diagnostics: ModelDiagnostics }
  | { type: 'round-end'; round: number; trainedModel: TrainedModel<S, P>; diagnostics: ModelDiagnostics; trialsAfter: number }
  | { type: 'done'; totalTrials: number; finalModel: TrainedModel<S, P>; finalDiagnostics: ModelDiagnostics }
  // Pipelines escape-hatch for domain-specific events without losing the
  // generic envelope.
  | { type: 'custom'; round: number; payload: Record<string, unknown> };

export interface TrainingContext<S, C, Cfg> {
  /** Index of the current round (0-based). */
  round: number;
  /** Trials accumulated so far. */
  store: TrialStore<S, C, Cfg>;
}

/** Concrete vehicle pipelines implement this interface to plug into
 *  `runOfflineTraining`. Each method is pure-ish (may be async). */
export interface TrainingPipeline<S, C, P, Cfg> {
  /** Human-readable name (for diagnostics). */
  name: string;

  /** Run one round of trial collection. The pipeline owns its own headless
   *  trial harness (Rapier for cars, future flight-physics for airplanes)
   *  and decides what initial states + controls trace to sample. Returns
   *  the trials to append to the store + a count of trials it tried and
   *  discarded (e.g. off-arena, NaN). */
  collectTrials(ctx: TrainingContext<S, C, Cfg>): Promise<{
    collected: Trial<S, C, Cfg>[];
    discarded: number;
  }>;

  /** Fit the parametric backbone against the current store contents.
   *  Streams `FitProgressEvent`s through `onProgress` (or no-op). */
  fitParametric(
    ctx: TrainingContext<S, C, Cfg>,
    onProgress?: (e: FitProgressEvent) => void,
  ): Promise<P>;

  /** Fit the residual MLP (if any) against parametric residuals. May be
   *  a no-op for pipelines that don't use a residual layer. */
  fitResidual(
    ctx: TrainingContext<S, C, Cfg>,
    params: P,
    onProgress?: (e: FitProgressEvent) => void,
  ): Promise<TrainedModel<S, P>>;

  /** Evaluate the trained model on held-out trials. */
  evaluate(
    ctx: TrainingContext<S, C, Cfg>,
    model: TrainedModel<S, P>,
  ): ModelDiagnostics;

  /** Total number of rounds to run. */
  totalRounds(): number;
}

/** Generic orchestrator. Each round: collect trials -> fit parametric ->
 *  fit residual -> evaluate. Streams events through `onEvent`. */
export async function runOfflineTraining<S, C, P, Cfg>(opts: {
  pipeline: TrainingPipeline<S, C, P, Cfg>;
  store: TrialStore<S, C, Cfg>;
  onEvent?: (e: TrainingEvent<S, C, P, Cfg>) => void;
}): Promise<{ model: TrainedModel<S, P>; diagnostics: ModelDiagnostics }> {
  const { pipeline, store, onEvent } = opts;
  const rounds = pipeline.totalRounds();
  let model: TrainedModel<S, P> | null = null;
  let diagnostics: ModelDiagnostics = { openLoopDivergence: [], perStateRms: [], coverage: [], baselines: {} };
  for (let round = 0; round < rounds; round++) {
    onEvent?.({ type: 'round-start', round, trialsBefore: store.size() });
    const ctx: TrainingContext<S, C, Cfg> = { round, store };
    const { collected, discarded } = await pipeline.collectTrials(ctx);
    for (const t of collected) store.add(t);
    onEvent?.({ type: 'trial-batch', round, collected: collected.length, discarded });
    const params = await pipeline.fitParametric(ctx, (event) =>
      onEvent?.({ type: 'fit-progress', round, phase: 'parametric', event }),
    );
    model = await pipeline.fitResidual(ctx, params, (event) =>
      onEvent?.({ type: 'fit-progress', round, phase: 'residual', event }),
    );
    diagnostics = pipeline.evaluate(ctx, model);
    onEvent?.({ type: 'evaluation', round, diagnostics });
    onEvent?.({ type: 'round-end', round, trainedModel: model, diagnostics, trialsAfter: store.size() });
  }
  if (!model) {
    throw new Error('runOfflineTraining: pipeline.totalRounds() returned 0');
  }
  onEvent?.({ type: 'done', totalTrials: store.size(), finalModel: model, finalDiagnostics: diagnostics });
  return { model, diagnostics };
}
