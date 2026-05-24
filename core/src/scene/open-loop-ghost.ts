// Generic open-loop "ghost" tracker.
//
// Every tick the real `Body<S, C>` advances one step of ground-truth physics.
// In parallel we want to roll a learned `ForwardSim<S>` open-loop, so we can
// VISUALIZE the prediction divergence and ACCUMULATE rolling RMS error.
// That's what this class owns:
//
//   1. An anchor pose — the state the ghost is currently rolling FROM.
//   2. An open-loop trace — the predicted states since the last anchor.
//   3. A re-anchoring policy — periodically (or on excessive drift) reset
//      the anchor back to the real state, so the ghost shows SHORT-TERM
//      prediction error rather than indefinite accumulation.
//
// The class is generic in `<S, C>`. The mapping from a `C` (e.g. a car's
// `WheeledCarControls`) to the `number[]` shape that `ForwardSim<S>` consumes
// is supplied by the caller via `encodeControls`. That is where domain-
// specific quirks live (e.g. the car's steer sign flip between Rapier's
// native frame and the kinocat planning frame).

import type { ForwardSim } from '../primitives/types';

export interface GhostTrackerOptions<S, C> {
  name: string;
  forwardSim: ForwardSim<S>;
  /** Translate domain controls into the opaque vector ForwardSim expects.
   *  Encodes any per-sim quirks (sign conventions, unit conversions). */
  encodeControls: (c: C) => number[];
  /** Re-anchor the ghost to the real state if this many seconds have
   *  elapsed since the last anchor. `Infinity` disables time-based
   *  re-anchoring. Default: `Infinity` (no time-based re-anchor). */
  reAnchorSec?: number;
  /** Re-anchor if the predicted state drifts more than this from real,
   *  by whatever metric `driftFn` measures. `Infinity` disables.
   *  Default: `Infinity`. */
  maxDrift?: number;
  /** Function that returns a scalar drift between real and predicted
   *  (used by `maxDrift`). Defaults to Euclidean distance assuming
   *  `{ x, z }` fields are present; pass a custom function for
   *  non-XZ-planar bodies. */
  driftFn?: (real: S, predicted: S) => number;
  /** Cap on the open-loop trace ring buffer. Default 2048 states. */
  traceCap?: number;
}

const DEFAULT_DRIFT_FN = <S>(real: S, predicted: S): number => {
  const r = real as unknown as { x?: number; y?: number; z?: number };
  const p = predicted as unknown as { x?: number; y?: number; z?: number };
  const dx = (p.x ?? 0) - (r.x ?? 0);
  const dy = (p.y ?? 0) - (r.y ?? 0);
  const dz = (p.z ?? 0) - (r.z ?? 0);
  return Math.hypot(dx, dy, dz);
};

export class OpenLoopGhostTracker<S, C> {
  readonly name: string;
  private readonly forwardSim: ForwardSim<S>;
  private readonly encodeControls: (c: C) => number[];
  private readonly reAnchorSec: number;
  private readonly maxDrift: number;
  private readonly driftFn: (real: S, predicted: S) => number;
  private readonly traceCap: number;

  private current: S | null = null;
  private anchorTime = 0;
  private readonly buf: S[] = [];

  constructor(opts: GhostTrackerOptions<S, C>) {
    this.name = opts.name;
    this.forwardSim = opts.forwardSim;
    this.encodeControls = opts.encodeControls;
    this.reAnchorSec = opts.reAnchorSec ?? Infinity;
    this.maxDrift = opts.maxDrift ?? Infinity;
    this.driftFn = opts.driftFn ?? (DEFAULT_DRIFT_FN as (r: S, p: S) => number);
    this.traceCap = opts.traceCap ?? 2048;
  }

  /** Snap the ghost back to the real state. Called on init, mode switch,
   *  excessive drift, or once every `reAnchorSec`. */
  anchor(state: S, simTime: number): void {
    this.current = state;
    this.anchorTime = simTime;
    this.buf.length = 0;
    this.buf.push(state);
  }

  /** Step the ghost open-loop by `dt`, then return the new predicted state.
   *  If `current` is null (no anchor yet), this anchors first. Also
   *  re-anchors if the time / drift policy says so. */
  step(controls: C, dt: number, real: S, simTime: number): S {
    if (this.current === null) {
      this.anchor(real, simTime);
      return this.current!;
    }
    if (simTime - this.anchorTime >= this.reAnchorSec) {
      this.anchor(real, simTime);
      return this.current!;
    }
    const encoded = this.encodeControls(controls);
    const next = this.forwardSim(this.current, encoded, dt);
    this.current = next;
    if (this.driftFn(real, next) > this.maxDrift) {
      this.anchor(real, simTime);
      return this.current!;
    }
    this.buf.push(next);
    while (this.buf.length > this.traceCap) this.buf.shift();
    return next;
  }

  /** Last predicted state (or null if not yet anchored). */
  state(): S | null {
    return this.current;
  }

  /** Open-loop predicted trail since the last anchor. */
  trace(): ReadonlyArray<S> {
    return this.buf;
  }

  /** Wipe both anchor and trace. */
  reset(): void {
    this.current = null;
    this.anchorTime = 0;
    this.buf.length = 0;
  }
}
