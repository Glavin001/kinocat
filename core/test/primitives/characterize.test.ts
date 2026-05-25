import { describe, it, expect } from 'vitest';
import { characterizeVehicle } from '../../src/primitives/characterize';
import { MotionPrimitiveLibrary } from '../../src/primitives/library';
import { defaultVehicleAgent, kinematicForwardSim } from '../../src/agent/vehicle';
import type { CarKinematicState } from '../../src/agent/types';

const agent = defaultVehicleAgent({ minTurnRadius: 3, maxSpeed: 8, maxReverseSpeed: 4 });
const sim = kinematicForwardSim(agent);

describe('characterizeVehicle', () => {
  it('records the correct end offset for a straight primitive', () => {
    const lib = characterizeVehicle({
      forwardSim: sim,
      controlSets: [[0, 6]],
      duration: 1,
      substeps: 10,
      startSpeeds: [0],
    });
    const p = lib.primitives[0]!;
    expect(p.end.dx).toBeCloseTo(6, 6); // 6 m/s * 1 s
    expect(p.end.dz).toBeCloseTo(0, 9);
    expect(p.end.dHeading).toBeCloseTo(0, 9);
    expect(p.end.speed).toBeCloseTo(6, 9);
    expect(p.sweep.length).toBe(11); // substeps + 1
    expect(p.reverse).toBe(false);
  });

  it('end offset matches an independent integration of the same model', () => {
    const controls = [1 / 3, 6];
    const duration = 0.8;
    const substeps = 8;
    const lib = characterizeVehicle({
      forwardSim: sim,
      controlSets: [controls],
      duration,
      substeps,
      startSpeeds: [0],
    });
    let s: CarKinematicState = { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
    for (let k = 0; k < substeps; k++) s = sim(s, controls, duration / substeps);
    const p = lib.primitives[0]!;
    expect(p.end.dx).toBeCloseTo(s.x, 9);
    expect(p.end.dz).toBeCloseTo(s.z, 9);
    expect(p.end.dHeading).toBeCloseTo(s.heading, 9);
  });

  it('flags reverse primitives', () => {
    const lib = characterizeVehicle({
      forwardSim: sim,
      controlSets: [[0, -4]],
      duration: 0.5,
      substeps: 4,
      startSpeeds: [0],
    });
    expect(lib.primitives[0]!.reverse).toBe(true);
  });

  it('JSON round-trips exactly', () => {
    const lib = characterizeVehicle({
      forwardSim: sim,
      controlSets: [[0, 6], [1 / 3, 6], [0, -4]],
      duration: 0.5,
      substeps: 5,
      startSpeeds: [0, 6],
    });
    const restored = MotionPrimitiveLibrary.fromJSON(lib.toJSON());
    expect(restored.primitives).toEqual(lib.primitives);
    expect(restored.startSpeeds).toEqual(lib.startSpeeds);
    expect(restored.lookup(6).length).toBe(lib.lookup(6).length);
  });
});
