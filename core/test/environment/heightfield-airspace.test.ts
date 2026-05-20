import { describe, it, expect } from 'vitest';
import {
  HeightfieldAirspace,
  type HeightfieldSampler,
} from '../../src/environment/heightfield-airspace';
import {
  InMemoryAirspace,
  type AABB,
} from '../../src/environment/airspace-world';
import { defaultAircraftAgent } from '../../src/agent/aircraft';
import type { AircraftState } from '../../src/agent/types';

const agent = defaultAircraftAgent({
  halfLength: 2,
  halfSpan: 1.5,
  halfHeight: 0.3,
});

const HALF: [number, number, number] = [
  agent.halfLength,
  agent.halfSpan,
  agent.halfHeight,
];

function pose(over: Partial<AircraftState> = {}) {
  const s: AircraftState = {
    x: 0,
    y: 20,
    z: 0,
    heading: 0,
    pitch: 0,
    roll: 0,
    speed: 18,
    t: 0,
    ...over,
  };
  return { x: s.x, y: s.y, z: s.z, yaw: s.heading, pitch: s.pitch, roll: s.roll };
}

describe('HeightfieldAirspace', () => {
  it('with a constant-Y sampler matches InMemoryAirspace at every pose', () => {
    const flat: HeightfieldSampler = () => 0;
    const box: AABB = { min: [40, 0, -10], max: [50, 30, 10] };
    const ref = new InMemoryAirspace({ floor: 0, ceiling: 60, boxes: [box] });
    const hf = new HeightfieldAirspace({
      floor: 0,
      ceiling: 60,
      boxes: [box],
      sampler: flat,
    });
    const samples = [
      pose({ x: 0, y: 20 }),
      pose({ x: 45, y: 15 }), // inside box
      pose({ x: 60, y: 20 }),
      pose({ x: 5, y: -1 }), // below floor
      pose({ x: 5, y: 70 }), // above ceiling
    ];
    for (const p of samples) {
      expect(hf.clear(p, HALF, 0)).toBe(ref.clear(p, HALF, 0));
    }
  });

  it('rejects an aircraft whose underside dips into a ridge', () => {
    // A sinusoidal ridge peaking at y = 30 around x = 50.
    const sampler: HeightfieldSampler = (x) =>
      Math.max(0, 30 * Math.exp(-((x - 50) * (x - 50)) / 80));
    const hf = new HeightfieldAirspace({ floor: -1, ceiling: 100, sampler });
    // Wings-level at y = 20 above the ridge ⇒ underside ≈ 19.7, terrain ≈ 30 ⇒ collide.
    expect(hf.clear(pose({ x: 50, y: 20 }), HALF, 0)).toBe(false);
    // Climb to y = 60 ⇒ underside ≈ 59.7, terrain ≈ 30 ⇒ clear.
    expect(hf.clear(pose({ x: 50, y: 60 }), HALF, 0)).toBe(true);
    // Off-ridge at y = 20 ⇒ terrain near 0 ⇒ clear.
    expect(hf.clear(pose({ x: 0, y: 20 }), HALF, 0)).toBe(true);
  });

  it('applies the safety margin', () => {
    const sampler: HeightfieldSampler = () => 18; // flat 18 m terrain
    const hf = new HeightfieldAirspace({
      floor: -1,
      ceiling: 100,
      sampler,
      sampleMargin: 4,
    });
    // OBB bottom at y = 20 - 0.3 = 19.7; terrain at 18 ⇒ clearance 1.7 < margin 4 ⇒ reject.
    expect(hf.clear(pose({ x: 0, y: 20 }), HALF, 0)).toBe(false);
    // Higher up: bottom at y = 23 - 0.3 = 22.7; clearance 4.7 > 4 ⇒ accept.
    expect(hf.clear(pose({ x: 0, y: 23 }), HALF, 0)).toBe(true);
  });

  it('samples across the OBB extent, not just the centre (catches edge peaks)', () => {
    // Tall, narrow Gaussian centred at (1.8, 0). The OBB centre at (0, 0) is
    // ~1.8 away — the centre sample reads near 0. The +x edge-midpoint
    // sample at (2, 0) is only 0.2 from the peak; if the implementation
    // sampled only the centre it would falsely report clear.
    const sampler: HeightfieldSampler = (x, z) => {
      const dx = x - 1.8;
      return 30 * Math.exp(-(dx * dx + z * z) / 1.5);
    };
    const hf = new HeightfieldAirspace({ floor: -1, ceiling: 100, sampler });
    // y = 22 ⇒ underside ≈ 21.7. Centre sample ≈ 30·exp(-1.8²/1.5) ≈ 3.6; far
    // below 21.7. Edge-midpoint sample at (2, 0) ≈ 30·exp(-0.027) ≈ 29.2 ⇒
    // hits 21.7 ⇒ rejected.
    expect(hf.clear(pose({ x: 0, y: 22, heading: 0 }), HALF, 0)).toBe(false);
    // Far away: clear.
    expect(hf.clear(pose({ x: 30, y: 22, heading: 0 }), HALF, 0)).toBe(true);
  });

  it('clearAABB rejects rectangles that intersect a hill', () => {
    const sampler: HeightfieldSampler = (x) =>
      Math.max(0, 30 * Math.exp(-((x - 50) * (x - 50)) / 80));
    const hf = new HeightfieldAirspace({ floor: -1, ceiling: 100, sampler });
    // Query box low over the hill.
    expect(hf.clearAABB(45, 5, -5, 55, 25, 5)).toBe(false);
    // Query box well above the hill.
    expect(hf.clearAABB(45, 50, -5, 55, 80, 5)).toBe(true);
    // Query box off the hill at low altitude.
    expect(hf.clearAABB(0, 5, -5, 10, 25, 5)).toBe(true);
  });

  it('clearAABB returns false when moving zones exist (delegates to inner)', () => {
    const hf = new HeightfieldAirspace({
      floor: -1,
      ceiling: 100,
      sampler: () => 0,
      zones: [{ radius: 5, predict: () => ({ x: 0, y: 0, z: 0 }) }],
    });
    // Even high above terrain, the inner static-only broadphase must report
    // false because it can't certify clearance over an unknown time span.
    expect(hf.clearAABB(0, 80, 0, 10, 90, 10)).toBe(false);
  });

  it('honours zone collisions through the inner airspace', () => {
    const hf = new HeightfieldAirspace({
      floor: -1,
      ceiling: 100,
      sampler: () => 0,
      zones: [{ radius: 6, predict: () => ({ x: 30, y: 20, z: 0 }) }],
    });
    expect(hf.clear(pose({ x: 30, y: 20 }), HALF, 0)).toBe(false);
    expect(hf.clear(pose({ x: 50, y: 20 }), HALF, 0)).toBe(true);
  });
});
