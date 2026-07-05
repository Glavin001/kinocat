// Generic driver abstraction.
//
// A `Driver<S, C>` decides what controls to apply for the next tick given the
// current state and time. It is the per-vehicle policy and is the SINGLE
// integration point for:
//
//   - Interactive control (WASD, joystick, gamepad)
//   - Plan tracking (pure-pursuit, MPC, autopilot)
//   - Scripted replay (recorded controls trace)
//   - Programmatic patterns (slalom, throttle pulse, brake cycle)
//   - Dataset-generation policies (random-walk, step input, diverse-IS)
//
// Drivers compose: `SwitchableDriver` lets you swap a sub-driver atomically
// (mode switches in a live demo), and `RecordingDriver` wraps any other
// driver and captures its emitted controls (useful for "drive then save
// the trial").

/** Decide controls for one tick. Pure: `sample` should be deterministic
 *  given `(state, simTime, dt)` for any driver intended to be replayed. */
export interface Driver<S, C> {
  sample(state: S, simTime: number, dt: number): C;
  /** Optional reset hook so stateful drivers (replan timers, key edge
   *  detectors) can clear themselves on teleport / new run. */
  reset?(): void;
}

/** Returns a constant control vector every tick (typically "zero"). */
export class IdleDriver<S, C> implements Driver<S, C> {
  constructor(private readonly zero: C) {}
  sample(_state: S, _simTime: number, _dt: number): C {
    return this.zero;
  }
}

/** Replays a pre-recorded controls trace, indexed by `floor(simTime / dt)`.
 *  Once the trace runs out, falls back to `zero`. */
export class ScriptedDriver<S, C> implements Driver<S, C> {
  private startTime = 0;
  private started = false;
  constructor(
    private readonly trace: ReadonlyArray<C>,
    private readonly zero: C,
  ) {}
  sample(_state: S, simTime: number, dt: number): C {
    if (!this.started) {
      this.startTime = simTime;
      this.started = true;
    }
    const idx = Math.floor((simTime - this.startTime) / dt + 1e-9);
    if (idx < 0 || idx >= this.trace.length) return this.zero;
    return this.trace[idx]!;
  }
  reset(): void {
    this.started = false;
    this.startTime = 0;
  }
}

/** Wraps a swappable inner driver. Use for live mode switches; the outer
 *  reference handed to `SceneController` stays stable while the policy
 *  underneath changes. */
export class SwitchableDriver<S, C> implements Driver<S, C> {
  constructor(private inner: Driver<S, C>) {}
  setInner(d: Driver<S, C>): void {
    this.inner = d;
  }
  getInner(): Driver<S, C> {
    return this.inner;
  }
  sample(state: S, simTime: number, dt: number): C {
    return this.inner.sample(state, simTime, dt);
  }
  reset(): void {
    this.inner.reset?.();
  }
}

/** Wraps any driver and appends every emitted control to a buffer. */
export class RecordingDriver<S, C> implements Driver<S, C> {
  private readonly buf: C[] = [];
  constructor(private readonly inner: Driver<S, C>) {}
  sample(state: S, simTime: number, dt: number): C {
    const c = this.inner.sample(state, simTime, dt);
    this.buf.push(c);
    return c;
  }
  trace(): ReadonlyArray<C> {
    return this.buf;
  }
  clear(): void {
    this.buf.length = 0;
  }
  reset(): void {
    this.inner.reset?.();
    this.buf.length = 0;
  }
}
