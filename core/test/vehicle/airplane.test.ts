// Architecture sanity check: the generic scene runtime works for a
// non-car (state, controls) pair without modification.

import { describe, expect, it } from 'vitest';

import {
  AIRPLANE_CONTROLS_ZERO,
  StubAirplaneBody,
  type AirplaneControls,
  type PlaneState,
} from '../../src/vehicle/airplane';
import {
  IdleDriver,
  ScriptedDriver,
  SceneController,
  runTrial,
} from '../../src/scene';

const INIT: PlaneState = {
  x: 0, y: 100, z: 0, heading: 0, pitch: 0, roll: 0, speed: 50, t: 0,
};

describe('airplane on generic scene runtime', () => {
  it('runTrial accepts a non-car body', () => {
    const body = new StubAirplaneBody(INIT);
    const driver = new IdleDriver<PlaneState, AirplaneControls>(AIRPLANE_CONTROLS_ZERO);
    const out = runTrial(body, driver, { dt: 1 / 60, steps: 60 });
    expect(out.states.length).toBe(61);
    expect(out.controls.length).toBe(60);
    // Coasting at speed=50 forward; should have moved forward in x.
    expect(out.states[60]!.x).toBeGreaterThan(out.states[0]!.x);
  });

  it('SceneController accepts a non-car body + scripted driver', () => {
    const body = new StubAirplaneBody(INIT);
    const trace: AirplaneControls[] = [];
    for (let i = 0; i < 30; i++) {
      trace.push({ throttle: 0.8, elevator: 0.1, aileron: 0, rudder: 0 });
    }
    const driver = new ScriptedDriver<PlaneState, AirplaneControls>(trace, AIRPLANE_CONTROLS_ZERO);
    const ctl = new SceneController({ body, driver, dt: 1 / 60 });
    let last: PlaneState | null = null;
    for (let i = 0; i < 30; i++) {
      const r = ctl.step(i / 60);
      last = r.real;
    }
    // Throttled up + slight elevator — should be climbing.
    expect(last!.y).toBeGreaterThan(INIT.y);
  });
});
