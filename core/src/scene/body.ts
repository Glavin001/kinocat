// Generic vehicle / agent "body" abstraction.
//
// A `Body<S, C>` is the ground-truth simulator for one agent: it owns the
// authoritative state (vehicle pose, velocity, attitude, etc.), accepts
// per-frame controls, and steps forward by `dt`. Implementations bind to
// whatever physics backend the game uses (Rapier raycast vehicle, custom
// flight model, kinematic stub, etc.).
//
// The generic system in `core/src/scene/` knows nothing about wheels, throttle,
// lift, or any specific physics engine — it just calls `applyControls`, `step`,
// and `readState` in lockstep. This is the contract that lets cars, airplanes,
// boats, and future vehicles share the same scene controller, training
// pipeline, and dataset-generation infrastructure.

/** Generic vehicle body — the ground truth simulator for one agent. */
export interface Body<S, C> {
  /** Snapshot the current authoritative state. Should be cheap. */
  readState(): S;
  /** Queue controls for the next physics step. Idempotent within a tick. */
  applyControls(controls: C): void;
  /** Advance the simulation by `dt` seconds using the last applied controls. */
  step(dt: number): void;
  /** Force-set the state (teleport, reset, anchor). */
  teleport(state: S): void;
  /** Release any resources held by the body (collider handles, etc.). */
  dispose?(): void;
}

/** Factory that materializes a fresh `Body` rooted at a given initial state.
 *  Used by trial/dataset generation: each run spawns a clean body, runs a
 *  driver, then disposes. */
export interface BodyFactory<S, C> {
  create(initialState: S): Body<S, C>;
}
