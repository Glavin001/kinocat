// Generic maneuver-runner test using a synthetic 1-D body.

import { describe, expect, it } from 'vitest';
import type { Body, Driver } from 'kinocat/scene';
import { runManeuver } from 'kinocat/training';

interface S { x: number; v: number; t: number }
interface C { a: number }
interface Cfg { id: string }

class LinearBody implements Body<S, C> {
  private state: S = { x: 0, v: 0, t: 0 };
  private pending: C = { a: 0 };
  readState(): S { return { ...this.state }; }
  applyControls(c: C): void { this.pending = c; }
  step(dt: number): void {
    this.state = {
      x: this.state.x + this.state.v * dt,
      v: this.state.v + this.pending.a * dt,
      t: this.state.t + dt,
    };
  }
  teleport(s: S): void { this.state = { ...s }; }
}

class ConstDriver implements Driver<S, C> {
  constructor(private c: C) {}
  sample(): C { return this.c; }
}

describe('runManeuver', () => {
  it('produces a trial with correctly-sized samples + controlsTrace', () => {
    const body = new LinearBody();
    const trial = runManeuver(body, new ConstDriver({ a: 1 }), {
      initialState: { x: 0, v: 0, t: 0 },
      dt: 0.1,
      steps: 10,
      sampleEveryNTicks: 2,
      id: 'lin-1',
      config: { id: 'A' },
      configKey: 'A',
      maneuverId: 'unit',
      maneuverParams: { a: 1 },
    });
    expect(trial.controlsTrace.length).toBe(10);
    expect(trial.samples[0]!.t).toBe(0);
    // sampleEveryNTicks=2 → indices 0, 2, 4, 6, 8, 10 → 6 samples
    expect(trial.samples.length).toBe(6);
    expect(trial.samples[trial.samples.length - 1]!.state.v).toBeCloseTo(1.0, 6);
  });

  it('assigns a split deterministically (no explicit split)', () => {
    const body = new LinearBody();
    const a = runManeuver(body, new ConstDriver({ a: 0 }), {
      initialState: { x: 0, v: 0, t: 0 },
      dt: 0.1, steps: 5, sampleEveryNTicks: 1, id: 'A', config: { id: 'A' }, configKey: 'A',
      maneuverId: 'unit', maneuverParams: { idx: 1 },
    });
    const b = runManeuver(body, new ConstDriver({ a: 0 }), {
      initialState: { x: 0, v: 0, t: 0 },
      dt: 0.1, steps: 5, sampleEveryNTicks: 1, id: 'B', config: { id: 'A' }, configKey: 'A',
      maneuverId: 'unit', maneuverParams: { idx: 1 },
    });
    expect(a.split).toBeDefined();
    expect(b.split).toBe(a.split);
  });
});
