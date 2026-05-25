// Headless kernel: run one trial of `Body<S, C>` driven by a `Driver<S, C>`
// for a fixed number of ticks. Returns the per-tick controls and resulting
// states. This is the shared code path between:
//
//   - The headless trial harness (offline training data collection)
//   - Dataset generation (sampling new policies for free-drive coverage)
//   - Unit tests (synthetic linear body + scripted controls)
//   - Future evaluation suites (sweep a driver over scenarios)
//
// The live demos use `SceneController` instead, which wraps the same per-tick
// step pattern but is geared for render loops and open-loop ghost tracking.

import type { Body } from './body';
import type { Driver } from './driver';

export interface RunTrialOptions<S, C> {
  /** Tick interval in seconds (typically 1/60). */
  dt: number;
  /** Number of ticks to simulate. */
  steps: number;
  /** Optional initial state. If provided, the body is teleported here
   *  before the first step. If omitted, the body's current state is used. */
  initialState?: S;
  /** Initial simulation time (defaults to 0). */
  startTime?: number;
}

export interface TrialResult<S, C> {
  /** State at index `i` is the state observed BEFORE the controls at
   *  index `i` were applied. `states[steps]` is the final state after
   *  the last applied control. Length = `steps + 1`. */
  states: S[];
  /** Controls at index `i` were sampled from `states[i]` and applied to
   *  produce `states[i + 1]`. Length = `steps`. */
  controls: C[];
}

export function runTrial<S, C>(
  body: Body<S, C>,
  driver: Driver<S, C>,
  opts: RunTrialOptions<S, C>,
): TrialResult<S, C> {
  const dt = opts.dt;
  const steps = opts.steps;
  const startTime = opts.startTime ?? 0;
  if (opts.initialState !== undefined) body.teleport(opts.initialState);
  driver.reset?.();
  const states: S[] = [];
  const controls: C[] = [];
  let s = body.readState();
  states.push(s);
  for (let i = 0; i < steps; i++) {
    const t = startTime + i * dt;
    const c = driver.sample(s, t, dt);
    controls.push(c);
    body.applyControls(c);
    body.step(dt);
    s = body.readState();
    states.push(s);
  }
  return { states, controls };
}
