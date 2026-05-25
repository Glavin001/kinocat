// Generic per-tick scene controller. Combines:
//   - one ground-truth `Body<S, C>` ("real" — Rapier raycast car, custom
//     flight model, etc.),
//   - any number of `OpenLoopGhostTracker<S, C>` (open-loop predictions
//     run alongside, for visualization + gap measurement),
//   - one `Driver<S, C>` (interactive, plan-tracking, scripted, ...),
//   - an optional `DebugRecorder<S, C>` (rolling ring buffer for
//     diagnostics export).
//
// Each call to `step(simTime)` advances real by `dt`, advances every ghost
// open-loop by the same `dt` under the same controls, and emits a
// `StepResult` for the renderer / HUD. The controller itself is framework-
// agnostic: no Three.js, no React, no Rapier. The React layer in demos is
// reduced to "mount Three, create controller, per-frame step + render".

import type { Body } from './body';
import type { Driver } from './driver';
import type { OpenLoopGhostTracker } from './open-loop-ghost';

export interface GhostStepResult<S> {
  name: string;
  state: S;
}

export interface StepResult<S, C> {
  /** Ground-truth state after the physics step. */
  real: S;
  /** Controls applied to produce that state. */
  controls: C;
  /** Per-ghost predicted state at the same `simTime`. */
  ghosts: GhostStepResult<S>[];
  /** Sim-time of this step (end of tick). */
  simTime: number;
  /** Tick interval used. */
  dt: number;
}

/** Optional debug-recorder hook. The generic controller knows only that the
 *  recorder accepts a frame containing the real state + controls + ghosts.
 *  The concrete `DebugRecorder<S, C>` lives in `core/src/diagnostics`. */
export interface RecorderHook<S, C> {
  capture(frame: {
    simTime: number;
    real: S;
    controls: C;
    ghosts: ReadonlyArray<GhostStepResult<S>>;
  }): void;
}

export interface SceneControllerOptions<S, C> {
  body: Body<S, C>;
  driver: Driver<S, C>;
  ghosts?: OpenLoopGhostTracker<S, C>[];
  recorder?: RecorderHook<S, C>;
  /** Default tick interval (s). Overridable per `step` call. */
  dt?: number;
}

export class SceneController<S, C> {
  private body: Body<S, C>;
  private driver: Driver<S, C>;
  private ghosts: OpenLoopGhostTracker<S, C>[];
  private recorder: RecorderHook<S, C> | undefined;
  private readonly defaultDt: number;

  constructor(opts: SceneControllerOptions<S, C>) {
    this.body = opts.body;
    this.driver = opts.driver;
    this.ghosts = opts.ghosts ?? [];
    this.recorder = opts.recorder;
    this.defaultDt = opts.dt ?? 1 / 60;
  }

  /** Advance real + every ghost by `dt` (default = controller's `dt`).
   *  Returns a `StepResult` for the renderer/HUD to consume. */
  step(simTime: number, dtOverride?: number): StepResult<S, C> {
    const dt = dtOverride ?? this.defaultDt;
    const preState = this.body.readState();
    const controls = this.driver.sample(preState, simTime, dt);
    this.body.applyControls(controls);
    this.body.step(dt);
    const real = this.body.readState();
    const ghosts: GhostStepResult<S>[] = [];
    for (const g of this.ghosts) {
      const s = g.step(controls, dt, real, simTime + dt);
      ghosts.push({ name: g.name, state: s });
    }
    const result: StepResult<S, C> = {
      real,
      controls,
      ghosts,
      simTime: simTime + dt,
      dt,
    };
    this.recorder?.capture({
      simTime: result.simTime,
      real,
      controls,
      ghosts,
    });
    return result;
  }

  /** Swap the driver atomically. Calls `reset?()` on the new driver. */
  setDriver(driver: Driver<S, C>): void {
    this.driver = driver;
    this.driver.reset?.();
  }

  /** Add a new ghost at runtime (anchors on next step). */
  addGhost(g: OpenLoopGhostTracker<S, C>): void {
    this.ghosts.push(g);
  }

  /** Remove all ghosts. */
  clearGhosts(): void {
    this.ghosts = [];
  }

  getGhosts(): ReadonlyArray<OpenLoopGhostTracker<S, C>> {
    return this.ghosts;
  }

  /** Teleport real to `state`, reset every ghost, reset driver. */
  resetTo(state: S): void {
    this.body.teleport(state);
    for (const g of this.ghosts) g.reset();
    this.driver.reset?.();
  }

  getBody(): Body<S, C> {
    return this.body;
  }

  getDriver(): Driver<S, C> {
    return this.driver;
  }
}
