import { describe, it, expect } from 'vitest';
import {
  constantVelocity,
  constantAcceleration,
  fromPhysicsRollout,
  fromObservations,
  linearObstacle,
  asObstacle,
} from '../../src/predict/factories';
import { kinematicForwardSim, defaultVehicleAgent } from '../../src/agent/vehicle';
import type { CarKinematicState } from '../../src/agent/types';

const base: CarKinematicState = { x: 0, z: 0, heading: 0, speed: 4, t: 0 };

describe('predict factories', () => {
  it('constantVelocity extrapolates along heading and clamps the horizon', () => {
    const p = constantVelocity(base, 5);
    expect(p(0)).toEqual({ x: 0, z: 0, heading: 0, speed: 4, t: 0 });
    expect(p(2)!.x).toBeCloseTo(8, 9);
    expect(p(2)!.z).toBeCloseTo(0, 9);
    expect(p(-1)).toBeNull();
    expect(p(6)).toBeNull(); // beyond horizon
  });

  it('constantAcceleration matches the kinematic equations', () => {
    const s: CarKinematicState = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
    const p = constantAcceleration(s, { ax: 2, az: 0 });
    const at2 = p(2)!;
    expect(at2.x).toBeCloseTo(4, 9); // 0.5*2*2^2
    expect(at2.speed).toBeCloseTo(4, 9); // 2*2
  });

  it('fromPhysicsRollout tracks the simulated trajectory', () => {
    const agent = defaultVehicleAgent();
    const sim = kinematicForwardSim(agent);
    const p = fromPhysicsRollout(base, [0, 4], sim, 0.1, 5);
    const at1 = p(1)!;
    expect(at1.x).toBeCloseTo(4, 1); // ~4 m/s for 1 s
    expect(p(99)).toBeNull();
  });

  it('fromObservations re-reads current state each query', () => {
    let cur: CarKinematicState = { x: 0, z: 0, heading: 0, speed: 2, t: 0 };
    const p = fromObservations(() => cur, { horizon: 10 });
    expect(p(1)!.x).toBeCloseTo(2, 9);
    cur = { x: 10, z: 0, heading: Math.PI / 2, speed: 3, t: 1 };
    const r = p(3)!;
    expect(r.x).toBeCloseTo(10, 9);
    expect(r.z).toBeCloseTo(6, 6); // 3 m/s * 2 s along +z
  });

  it('linearObstacle / asObstacle expose a circular predictor', () => {
    const o = linearObstacle(0, 0, 1, 0, 1.5, 0, 10);
    expect(o.radius).toBe(1.5);
    expect(o.predict(3)).toEqual({ x: 3, z: 0 });
    expect(o.predict(20)).toBeNull();

    const wrapped = asObstacle(constantVelocity(base, 10), 2);
    expect(wrapped.radius).toBe(2);
    expect(wrapped.predict(1)).toEqual({ x: 4, z: 0 });
  });
});
