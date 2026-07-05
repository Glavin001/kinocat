// kinocat/learning — domain-agnostic ML helpers for fitting and exploring
// any `ForwardSim<State>` against recorded `(state, controls, dt, next_state)`
// data. Used by the v2 vehicle model + the offline training pipeline, but
// the surface is generic enough to apply to other agent kinds (humanoid,
// aircraft, ...) when their learned models arrive.

export type {
  Trial,
  TrialSample,
  TrialStore,
  SerializedTrials,
  TrialSplit,
  SplitPolicy,
} from './trial-store';
export {
  createTrialStore,
  serializeTrials,
  deserializeTrials,
  assignSplit,
  trialSplitKey,
  hashString,
  DEFAULT_SPLIT_POLICY,
} from './trial-store';

export type {
  ParametricFitOptions,
  ParametricFitAsyncOptions,
  ParametricFitResult,
  FitProgressEvent,
  LossDecomposition,
} from './parametric-fit';
export { runParametricFit, runParametricFitAsync } from './parametric-fit';

export type {
  ResidualMLPFitOptions,
  ResidualMLPFitAsyncOptions,
  ResidualMLPFitResult,
  ResidualFitProgressEvent,
} from './residual-mlp-fit';
export { runResidualMLPFit, runResidualMLPFitAsync } from './residual-mlp-fit';

export type {
  ModelDiagnostics,
  OpenLoopRow,
  CoverageCell,
  EvaluateOptions,
  ForwardSimUnderTest,
} from './evaluate';
export { evaluateModel } from './evaluate';

export type {
  ActiveExplorerOptions,
  ExplorationCell,
  ProposedTrial,
} from './active-explorer';
export { proposeNextBatch } from './active-explorer';
