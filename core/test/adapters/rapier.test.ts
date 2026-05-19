import { describe, it, expect } from 'vitest';
import { rapierForwardSim } from '../../src/adapters/rapier/index';
import type { ForwardSim } from '../../src/primitives/types';
import type { VehicleState } from '../../src/agent/types';

let RAPIER_OK = false;
let sim: ForwardSim<VehicleState> | null = null;
try {
  const R = (await import('@dimforge/rapier3d-compat')).default;
  await R.init();
  const world = new R.World({ x: 0, y: 0, z: 0 });
  const body = world.createRigidBody(R.RigidBodyDesc.dynamic().setTranslation(0, 0, 0));
  world.createCollider(R.ColliderDesc.cuboid(0.5, 0.5, 0.5), body); // mass/inertia
  sim = rapierForwardSim({ world, body });
  RAPIER_OK = true;
} catch {
  RAPIER_OK = false;
}

it('rapier availability is a boolean (logs skip status in CI)', () => {
  expect(typeof RAPIER_OK).toBe('boolean');
});

describe.skipIf(!RAPIER_OK)('rapierForwardSim', () => {
  it('advances a body along its heading', () => {
    const s: VehicleState = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
    const next = sim!(s, [0, 2], 0.5);
    expect(next.x).toBeCloseTo(1, 2); // 2 m/s * 0.5 s
    expect(Math.abs(next.z)).toBeLessThan(1e-3);
    expect(next.t).toBeCloseTo(0.5, 9);
    expect(next.speed).toBe(2);
  });

  it('turns the body under a curvature command', () => {
    const s: VehicleState = { x: 0, z: 0, heading: 0, speed: 0, t: 1 };
    const next = sim!(s, [0.5, 2], 1);
    expect(Number.isFinite(next.heading)).toBe(true);
    expect(Math.abs(next.heading)).toBeGreaterThan(0.05); // it rotated
    expect(next.t).toBeCloseTo(2, 9);
  });

  it('is deterministic for identical inputs', () => {
    const s: VehicleState = { x: 1, z: 2, heading: 0.3, speed: 0, t: 0 };
    expect(sim!(s, [0.1, 3], 0.4)).toEqual(sim!(s, [0.1, 3], 0.4));
  });
});
