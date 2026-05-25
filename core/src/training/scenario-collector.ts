// Generic closed-loop scenario collector — Phase 3 of the training-dataset
// plan. The richest training data comes from the model driving itself
// around the actual race track (or any closed-loop scenario): the planner's
// queries become the training inputs by construction, so the training
// distribution tracks the deployment distribution. This is the DAgger
// (Dataset Aggregation, Ross et al. 2011) pattern.
//
// Usage:
//   const collector = createScenarioCollector({ scenarioId, dt, ... });
//   while (running) {
//     const state = body.readState();
//     const controls = driver.sample(state, simTime, dt);
//     body.applyControls(controls); body.step(dt);
//     const next = body.readState();
//     const trial = collector.record(simTime, state, controls, next);
//     if (trial) store.add(trial);
//   }
//
// Each emitted trial spans `windowSec` seconds of the closed-loop run.
// Adjacent trials don't overlap. The collector tags every emitted trial
// with `scenarioId` so the split policy + downstream filtering can keep
// scenario-sourced trials distinct from synthetic ones.
//
// Domain-agnostic — `S`, `C`, `Cfg` are opaque.

import type { Trial, TrialSplit } from '../learning/trial-store';
import { assignSplit } from '../learning/trial-store';

export interface ScenarioCollectorOptions<S, C, Cfg> {
  /** Stable scenario id for the trials. */
  scenarioId: string;
  /** Tick interval (s). */
  dt: number;
  /** Sample every Nth state into the emitted `samples`. */
  sampleEveryNTicks: number;
  /** Window size (s) per emitted trial. Default 1.0 s. */
  windowSec?: number;
  /** Vehicle config to attach to emitted trials. */
  config: Cfg;
  /** Stable config key. */
  configKey: string;
  /** Optional explicit split override (default: hash-based). */
  split?: TrialSplit;
  /** Optional id factory; default counter. */
  idFactory?: (n: number) => string;
}

export interface ScenarioCollector<S, C, Cfg> {
  /** Feed one tick of `(state_before, controls, state_after)`. Returns a
   *  freshly-emitted trial when the current window has filled, otherwise
   *  null. */
  record(
    simTime: number,
    state: S,
    controls: C,
    nextState: S,
  ): Trial<S, C, Cfg> | null;
  /** Force-emit the current partial window as a trial (if at least 2
   *  samples have accumulated). Useful when the scenario ends mid-window. */
  flush(): Trial<S, C, Cfg> | null;
  /** Total trials emitted. */
  emittedCount(): number;
  /** Reset the buffer + counters. */
  reset(): void;
}

interface Sample<S> {
  simTime: number;
  state: S;
}

export function createScenarioCollector<S, C, Cfg>(
  opts: ScenarioCollectorOptions<S, C, Cfg>,
): ScenarioCollector<S, C, Cfg> {
  const windowSec = opts.windowSec ?? 1.0;
  const sampleEvery = Math.max(1, opts.sampleEveryNTicks);
  const idFactory = opts.idFactory ?? ((n: number) => `${opts.scenarioId}-${n}`);
  // Windowed buffers.
  let samples: Sample<S>[] = [];
  let controls: C[] = [];
  let nextStates: S[] = [];
  let windowStart: number | null = null;
  let emitted = 0;
  // Stride counter for sampleEveryNTicks decimation of `samples`.
  let tickIdx = 0;

  function emit(): Trial<S, C, Cfg> | null {
    if (samples.length < 2 || controls.length === 0) return null;
    const initialState = samples[0]!.state;
    // Append the post-window state so the trial has a final sample at
    // window end (so evaluateModel can score a 1.0s horizon for trials
    // collected over a 1s window).
    const finalState = nextStates[nextStates.length - 1]!;
    const t0 = samples[0]!.simTime;
    const finalT = controls.length * opts.dt;
    const localSamples = samples.map((s) => ({ t: s.simTime - t0, state: s.state }));
    // Avoid duplicating the final sample.
    if (Math.abs((localSamples[localSamples.length - 1]?.t ?? -1) - finalT) > 1e-6) {
      localSamples.push({ t: finalT, state: finalState });
    }
    const trial: Trial<S, C, Cfg> = {
      id: idFactory(emitted++),
      initialState,
      controlsTrace: controls.slice(),
      dt: opts.dt,
      samples: localSamples,
      config: opts.config,
      configKey: opts.configKey,
      scenarioId: opts.scenarioId,
      maneuverId: 'scenario',
      maneuverParams: { window: windowSec },
    };
    trial.split = opts.split ?? assignSplit(trial);
    samples = [];
    controls = [];
    nextStates = [];
    windowStart = null;
    tickIdx = 0;
    return trial;
  }

  return {
    record(simTime, state, c, nextState) {
      if (windowStart === null) {
        windowStart = simTime;
        // Seed the window with the initial state.
        samples.push({ simTime, state });
        tickIdx = 0;
      }
      controls.push(c);
      nextStates.push(nextState);
      tickIdx++;
      if (tickIdx % sampleEvery === 0) {
        samples.push({ simTime: simTime + opts.dt, state: nextState });
      }
      // Emit when the window has filled.
      if (simTime + opts.dt - windowStart >= windowSec - 1e-9) {
        return emit();
      }
      return null;
    },
    flush() {
      return emit();
    },
    emittedCount() {
      return emitted;
    },
    reset() {
      samples = [];
      controls = [];
      nextStates = [];
      windowStart = null;
      emitted = 0;
      tickIdx = 0;
    },
  };
}
