// Minimal placeholder `Body<PlaneState, AirplaneControls>` for the
// architecture sanity check. The physics are intentionally trivial — the
// purpose is to verify that the generic scene runtime in `kinocat/scene`
// works for a non-car state/controls pair without modification.
//
// A real airplane body will live in `kinocat/adapters/<flight-physics>/`
// when we have a flight model worth integrating.

import type { Body } from '../../scene/body';
import type { AirplaneControls, PlaneState } from './types';
import { AIRPLANE_CONTROLS_ZERO } from './types';

/** Toy 6DOF placeholder: integrates straight-line flight forward + simple
 *  pitch/roll/yaw control coupling. Not physically realistic. */
export class StubAirplaneBody implements Body<PlaneState, AirplaneControls> {
  private state: PlaneState;
  private pending: AirplaneControls = AIRPLANE_CONTROLS_ZERO;

  constructor(initial: PlaneState) {
    this.state = { ...initial };
  }

  readState(): PlaneState {
    return { ...this.state };
  }

  applyControls(c: AirplaneControls): void {
    this.pending = c;
  }

  step(dt: number): void {
    const c = this.pending;
    const speed = Math.max(0, this.state.speed + (c.throttle - 0.3) * 5 * dt);
    const pitch = this.state.pitch + c.elevator * 0.5 * dt;
    const roll = this.state.roll + c.aileron * 1.0 * dt;
    const heading = this.state.heading + c.rudder * 0.5 * dt + roll * 0.3 * dt;
    this.state = {
      x: this.state.x + speed * Math.cos(heading) * Math.cos(pitch) * dt,
      y: this.state.y + speed * Math.sin(pitch) * dt,
      z: this.state.z + speed * Math.sin(heading) * Math.cos(pitch) * dt,
      heading,
      pitch,
      roll,
      speed,
      t: this.state.t + dt,
    };
  }

  teleport(s: PlaneState): void {
    this.state = { ...s };
    this.pending = AIRPLANE_CONTROLS_ZERO;
  }
}
