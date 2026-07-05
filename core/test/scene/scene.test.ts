// Unit tests for the generic scene runtime. Exercises Body / Driver /
// SceneController / OpenLoopGhostTracker / runTrial with a synthetic
// linear body to prove the runtime is domain-agnostic.

import { describe, expect, it } from 'vitest';

import type { Body, Driver } from '../../src/scene';
import {
  IdleDriver,
  OpenLoopGhostTracker,
  RecordingDriver,
  ScriptedDriver,
  SceneController,
  SwitchableDriver,
  runTrial,
} from '../../src/scene';

// ---------------------------------------------------------------------------
// Synthetic linear body.
//
// State: 1-D position + velocity. Controls: scalar acceleration.
//
//   x_{t+1} = x_t + v_t * dt
//   v_{t+1} = v_t + a * dt
//
// Used both as the real Body and as the ForwardSim for ghost tracking.

interface S1D {
  x: number;
  v: number;
  t: number;
}
interface C1D {
  a: number;
}

class LinearBody implements Body<S1D, C1D> {
  private state: S1D;
  private pending: C1D = { a: 0 };
  constructor(initial: S1D = { x: 0, v: 0, t: 0 }) {
    this.state = { ...initial };
  }
  readState(): S1D {
    return { ...this.state };
  }
  applyControls(c: C1D): void {
    this.pending = c;
  }
  step(dt: number): void {
    const { x, v, t } = this.state;
    this.state = { x: x + v * dt, v: v + this.pending.a * dt, t: t + dt };
  }
  teleport(s: S1D): void {
    this.state = { ...s };
  }
}

// "Wrong" ghost forward sim: also linear, but with a wrong accel gain
// (1.5x) so ghosts intentionally drift from real.
const wrongSim = (s: S1D, controls: number[], dt: number): S1D => {
  const a = controls[0] ?? 0;
  return { x: s.x + s.v * dt, v: s.v + 1.5 * a * dt, t: s.t + dt };
};

// Perfect ghost: matches LinearBody exactly. Used to verify the ghost
// path lines up with real when the model is perfect.
const perfectSim = (s: S1D, controls: number[], dt: number): S1D => {
  const a = controls[0] ?? 0;
  return { x: s.x + s.v * dt, v: s.v + a * dt, t: s.t + dt };
};

// ---------------------------------------------------------------------------

describe('IdleDriver', () => {
  it('returns the same zero every tick', () => {
    const d = new IdleDriver<S1D, C1D>({ a: 0 });
    expect(d.sample({ x: 0, v: 0, t: 0 }, 0, 1 / 60)).toEqual({ a: 0 });
    expect(d.sample({ x: 5, v: 1, t: 1 }, 1, 1 / 60)).toEqual({ a: 0 });
  });
});

describe('ScriptedDriver', () => {
  it('replays a trace by simTime index', () => {
    const trace: C1D[] = [{ a: 1 }, { a: 2 }, { a: 3 }];
    const d = new ScriptedDriver<S1D, C1D>(trace, { a: 0 });
    const dt = 1 / 60;
    expect(d.sample({ x: 0, v: 0, t: 0 }, 0, dt)).toEqual({ a: 1 });
    expect(d.sample({ x: 0, v: 0, t: 0 }, dt, dt)).toEqual({ a: 2 });
    expect(d.sample({ x: 0, v: 0, t: 0 }, 2 * dt, dt)).toEqual({ a: 3 });
    expect(d.sample({ x: 0, v: 0, t: 0 }, 3 * dt, dt)).toEqual({ a: 0 });
  });
});

describe('SwitchableDriver', () => {
  it('delegates to the inner driver and can swap atomically', () => {
    const a = new IdleDriver<S1D, C1D>({ a: 1 });
    const b = new IdleDriver<S1D, C1D>({ a: 2 });
    const sw = new SwitchableDriver<S1D, C1D>(a);
    const s: S1D = { x: 0, v: 0, t: 0 };
    expect(sw.sample(s, 0, 1 / 60)).toEqual({ a: 1 });
    sw.setInner(b);
    expect(sw.sample(s, 0, 1 / 60)).toEqual({ a: 2 });
  });
});

describe('RecordingDriver', () => {
  it('appends every sampled control to a trace', () => {
    const inner: Driver<S1D, C1D> = {
      sample: (_s, t) => ({ a: t * 10 }),
    };
    const rec = new RecordingDriver(inner);
    rec.sample({ x: 0, v: 0, t: 0 }, 0, 1 / 60);
    rec.sample({ x: 0, v: 0, t: 0 }, 1, 1 / 60);
    expect(rec.trace()).toEqual([{ a: 0 }, { a: 10 }]);
  });
});

describe('runTrial', () => {
  it('produces N+1 states and N controls for N steps', () => {
    const body = new LinearBody();
    const driver = new IdleDriver<S1D, C1D>({ a: 1 });
    const out = runTrial(body, driver, { dt: 1, steps: 3 });
    expect(out.states.length).toBe(4);
    expect(out.controls.length).toBe(3);
    // x_0 = 0, v_0 = 0
    // tick 0: apply a=1 -> v becomes 1; x stays 0 (x updates with v_pre).
    // After: state[1] = { x: 0, v: 1, t: 1 }
    // tick 1: x = 0 + 1*1 = 1; v = 1 + 1*1 = 2
    // tick 2: x = 1 + 2*1 = 3; v = 2 + 1*1 = 3
    expect(out.states[1]).toEqual({ x: 0, v: 1, t: 1 });
    expect(out.states[2]).toEqual({ x: 1, v: 2, t: 2 });
    expect(out.states[3]).toEqual({ x: 3, v: 3, t: 3 });
  });

  it('respects an explicit initialState (teleport before stepping)', () => {
    const body = new LinearBody({ x: 99, v: 0, t: 0 });
    const driver = new IdleDriver<S1D, C1D>({ a: 0 });
    const out = runTrial(body, driver, {
      dt: 1,
      steps: 1,
      initialState: { x: 5, v: 0, t: 0 },
    });
    expect(out.states[0]).toEqual({ x: 5, v: 0, t: 0 });
  });
});

describe('OpenLoopGhostTracker', () => {
  it('anchors on first step then rolls open-loop until re-anchor', () => {
    const ghost = new OpenLoopGhostTracker<S1D, C1D>({
      name: 'perfect',
      forwardSim: perfectSim,
      encodeControls: (c) => [c.a],
      driftFn: (a, b) => Math.abs(b.x - a.x),
    });
    const real0: S1D = { x: 0, v: 0, t: 0 };
    const real1: S1D = { x: 0, v: 1, t: 1 };
    const real2: S1D = { x: 1, v: 2, t: 2 };
    const out0 = ghost.step({ a: 1 }, 1, real0, 0); // anchors on real0
    expect(out0).toEqual(real0);
    const out1 = ghost.step({ a: 1 }, 1, real1, 1);
    // perfectSim from (0,0): x=0, v=1
    expect(out1).toEqual({ x: 0, v: 1, t: 1 });
    const out2 = ghost.step({ a: 1 }, 1, real2, 2);
    // from (x=0, v=1): x=1, v=2
    expect(out2).toEqual({ x: 1, v: 2, t: 2 });
  });

  it('re-anchors when drift exceeds maxDrift', () => {
    const ghost = new OpenLoopGhostTracker<S1D, C1D>({
      name: 'wrong',
      forwardSim: wrongSim, // 1.5x accel — drifts
      encodeControls: (c) => [c.a],
      maxDrift: 0.6, // tight bound so drift triggers immediately
      driftFn: (real, pred) => Math.abs(pred.v - real.v),
    });
    const body = new LinearBody();
    ghost.step({ a: 0 }, 1, body.readState(), 0); // anchor
    body.applyControls({ a: 1 });
    body.step(1); // real: x=0, v=1
    const pred = ghost.step({ a: 1 }, 1, body.readState(), 1);
    // wrongSim gives v=1.5; drift = |1.5 - 1| = 0.5 < 0.6 -> NO re-anchor
    expect(pred.v).toBeCloseTo(1.5, 6);
    body.applyControls({ a: 1 });
    body.step(1); // real: x=1, v=2
    const pred2 = ghost.step({ a: 1 }, 1, body.readState(), 2);
    // From wrong (x=0, v=1.5) with a=1: x=1.5, v=3. drift=|3-2|=1.0 > 0.6 -> RE-ANCHOR to real
    expect(pred2).toEqual(body.readState());
  });

  it('re-anchors when reAnchorSec elapses', () => {
    const ghost = new OpenLoopGhostTracker<S1D, C1D>({
      name: 'time-anchored',
      forwardSim: wrongSim,
      encodeControls: (c) => [c.a],
      reAnchorSec: 2,
    });
    const real: S1D = { x: 0, v: 0, t: 0 };
    ghost.step({ a: 0 }, 1, real, 0); // anchor at t=0
    const realB: S1D = { x: 1, v: 1, t: 1 };
    ghost.step({ a: 1 }, 1, realB, 1); // t-since-anchor=1, no re-anchor
    const realC: S1D = { x: 2, v: 2, t: 2 };
    const out = ghost.step({ a: 1 }, 1, realC, 2); // t-since-anchor=2 -> re-anchor
    expect(out).toEqual(realC);
  });
});

describe('SceneController', () => {
  it('drives real + ghosts in lockstep and emits a StepResult', () => {
    const body = new LinearBody();
    const driver = new IdleDriver<S1D, C1D>({ a: 1 });
    const ghost = new OpenLoopGhostTracker<S1D, C1D>({
      name: 'perfect',
      forwardSim: perfectSim,
      encodeControls: (c) => [c.a],
    });
    const ctl = new SceneController({ body, driver, ghosts: [ghost], dt: 1 });
    const r0 = ctl.step(0);
    expect(r0.controls).toEqual({ a: 1 });
    expect(r0.real).toEqual({ x: 0, v: 1, t: 1 });
    expect(r0.ghosts.length).toBe(1);
    expect(r0.ghosts[0]!.name).toBe('perfect');
    const r1 = ctl.step(1);
    expect(r1.real).toEqual({ x: 1, v: 2, t: 2 });
  });

  it('routes captures to the recorder hook', () => {
    const body = new LinearBody();
    const driver = new IdleDriver<S1D, C1D>({ a: 0 });
    const frames: Array<{ simTime: number }> = [];
    const ctl = new SceneController<S1D, C1D>({
      body,
      driver,
      dt: 1,
      recorder: {
        capture: (f) => frames.push({ simTime: f.simTime }),
      },
    });
    ctl.step(0);
    ctl.step(1);
    expect(frames).toEqual([{ simTime: 1 }, { simTime: 2 }]);
  });

  it('resetTo teleports + resets ghosts + resets driver', () => {
    const body = new LinearBody();
    const driver = new ScriptedDriver<S1D, C1D>([{ a: 1 }, { a: 2 }], { a: 0 });
    const ghost = new OpenLoopGhostTracker<S1D, C1D>({
      name: 'g',
      forwardSim: perfectSim,
      encodeControls: (c) => [c.a],
    });
    const ctl = new SceneController({ body, driver, ghosts: [ghost], dt: 1 });
    ctl.step(0); // consumes trace[0]
    ctl.resetTo({ x: 100, v: 0, t: 5 });
    expect(body.readState()).toEqual({ x: 100, v: 0, t: 5 });
    // After reset, driver should restart from index 0, so the very next sample is { a: 1 }.
    const r = ctl.step(5);
    // applied {a:1}; from x=100,v=0: new state x=100,v=1,t=6
    expect(r.real).toEqual({ x: 100, v: 1, t: 6 });
    expect(r.controls).toEqual({ a: 1 });
  });
});
