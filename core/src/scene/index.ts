// kinocat/scene — generic vehicle scene runtime.
//
// `Body<S, C>` + `Driver<S, C>` + `SceneController<S, C>` + `runTrial<S, C>` +
// `OpenLoopGhostTracker<S, C>` are domain-agnostic and parametric in the
// state/controls types. The current car-on-Rapier system is one consumer
// (see `kinocat/vehicle/car` + `kinocat/adapters/rapier`); a future airplane
// or any other vehicle plugs in by providing its own `Body` impl, drivers,
// and parametric `ForwardSim<S>` without modifying this layer.

export type { Body, BodyFactory } from './body';
export type { Driver } from './driver';
export { IdleDriver, ScriptedDriver, SwitchableDriver, RecordingDriver } from './driver';
export type { GhostTrackerOptions } from './open-loop-ghost';
export { OpenLoopGhostTracker } from './open-loop-ghost';
export type {
  GhostStepResult,
  StepResult,
  RecorderHook,
  SceneControllerOptions,
} from './scene-controller';
export { SceneController } from './scene-controller';
export type { RunTrialOptions, TrialResult } from './run-trial';
export { runTrial } from './run-trial';
