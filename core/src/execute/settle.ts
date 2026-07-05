// Settle latch — the single, shared definition of "the maneuver is DONE".
//
// A goal predicate (in the stall / at the pose / …) that is true for one tick
// is not success: a car can drive THROUGH the success region, or pause inside
// it mid-shunt and leave again. Every observer that judged kinocat on a
// transient snapshot (bench break-on-first-true, test `done` gates, HUD
// banners) has historically disagreed with what a human watching the sim saw.
//
// `createSettleLatch` turns an instantaneous predicate into closed-loop
// success semantics: the predicate must hold CONTINUOUSLY for `holdSeconds`
// with the vehicle at rest (|speed| ≤ speedTol) before `settled` latches.
// After the latch, every tick where the condition no longer holds is counted
// as a `violation` — a settled vehicle that creeps back out of the region is
// a controller bug that must fail benches, not vanish from them.
//
// Pure and framework-free: feed it one sample per fixed physics tick.

export interface SettleConfig {
  /** Continuous hold required before `settled` latches (seconds). */
  holdSeconds: number;
  /** |speed| at or below this counts as "at rest" (m/s). */
  speedTol: number;
}

export interface SettleSample {
  /** The instantaneous goal predicate (e.g. `evaluateParked(...).parked`). */
  ok: boolean;
  /** Signed vehicle speed (m/s); compared as |speed| ≤ speedTol. */
  speed: number;
}

export interface SettleState {
  /** Latched: the condition held continuously for `holdSeconds`. Never unlatches. */
  settled: boolean;
  /** The condition (predicate + at-rest) holds on the current tick. */
  holding: boolean;
  /** Current continuous hold duration (seconds; resets to 0 pre-latch on a break). */
  heldFor: number;
  /**
   * Sim-time (seconds since the first `update`) at which the successful hold
   * BEGAN — i.e. the moment the vehicle came to rest in the goal region and
   * stayed. This is the honest "time to parked", excluding the hold window
   * itself. `null` until `settled` latches.
   */
  timeToSettled: number | null;
  /** Post-latch ticks where the condition did NOT hold (creep-out, shunt, …). */
  violations: number;
  /** Total sim-time fed through the latch (seconds). */
  elapsed: number;
}

export interface SettleLatch {
  /** Advance one fixed physics tick. Returns the (mutated) current state. */
  update(sample: SettleSample, dt: number): SettleState;
  readonly state: SettleState;
}

export function createSettleLatch(config: SettleConfig): SettleLatch {
  const state: SettleState = {
    settled: false,
    holding: false,
    heldFor: 0,
    timeToSettled: null,
    violations: 0,
    elapsed: 0,
  };
  let holdStart = 0;

  function update(sample: SettleSample, dt: number): SettleState {
    state.elapsed += dt;
    state.holding = sample.ok && Math.abs(sample.speed) <= config.speedTol;

    if (state.settled) {
      if (!state.holding) state.violations++;
      return state;
    }

    if (state.holding) {
      if (state.heldFor === 0) holdStart = state.elapsed - dt;
      state.heldFor += dt;
      if (state.heldFor >= config.holdSeconds) {
        state.settled = true;
        state.timeToSettled = holdStart;
      }
    } else {
      state.heldFor = 0;
    }
    return state;
  }

  return { update, state };
}
