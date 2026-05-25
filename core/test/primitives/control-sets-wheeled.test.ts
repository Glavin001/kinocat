// Verify the wheeled-action primitive library compiles via characterizeVehicle
// against the v2 learned model and produces sensible end-state offsets.

import { describe, it, expect } from 'vitest';
import {
  coarseWheeledControls,
  fineWheeledControls,
  DEFAULT_WHEELED_START_SPEEDS,
  FINE_WHEELED_START_SPEEDS,
  characterizeVehicle,
} from 'kinocat/primitives';
import {
  DEFAULT_LEARNABLE_CONFIG,
  DEFAULT_LEARNED_PARAMS_V2,
  parametricForwardV2,
  buildParametricOnlyModel,
  learnedForwardSimV2,
} from 'kinocat/agent';

const cfg = DEFAULT_LEARNABLE_CONFIG;

describe('control-sets-wheeled — coarse + fine tiers', () => {
  it('coarse has 5 actions, all in [steer, driveForce, brakeForce] shape', () => {
    const set = coarseWheeledControls({ config: cfg });
    expect(set.length).toBe(5);
    for (const v of set) {
      expect(v.length).toBe(3);
      expect(typeof v[0]).toBe('number');
      expect(typeof v[1]).toBe('number');
      expect(typeof v[2]).toBe('number');
      // Brake force is non-negative.
      expect(v[2]).toBeGreaterThanOrEqual(0);
      // Steer is within limits.
      expect(Math.abs(v[0]!)).toBeLessThanOrEqual(cfg.maxSteerAngle);
    }
  });

  it('fine has at least 15 actions with denser steer coverage', () => {
    const set = fineWheeledControls({ config: cfg });
    expect(set.length).toBeGreaterThanOrEqual(15);
    const steers = new Set(set.map((v) => v[0]!));
    expect(steers.size).toBeGreaterThan(5); // more variety than coarse
  });
});

describe('characterizeVehicle + parametricForwardV2 + wheeled controls', () => {
  it('builds a coarse library: each primitive has a sweep of expected length', () => {
    const sim = parametricForwardV2(DEFAULT_LEARNED_PARAMS_V2, cfg);
    const lib = characterizeVehicle({
      forwardSim: sim,
      controlSets: coarseWheeledControls({ config: cfg }),
      duration: 0.5,
      substeps: 6,
      startSpeeds: DEFAULT_WHEELED_START_SPEEDS,
    });
    const all = lib.primitives;
    expect(all.length).toBe(DEFAULT_WHEELED_START_SPEEDS.length * 5);
    for (const p of all) {
      expect(p.sweep.length).toBe(7); // 1 start + 6 substeps
      expect(p.controls.length).toBe(3);
    }
  });

  it('builds a fine library at finer start-speed bucketing', () => {
    const sim = learnedForwardSimV2(buildParametricOnlyModel());
    const lib = characterizeVehicle({
      forwardSim: sim,
      controlSets: fineWheeledControls({ config: cfg }),
      duration: 0.15,
      substeps: 3,
      startSpeeds: FINE_WHEELED_START_SPEEDS,
    });
    expect(lib.primitives.length).toBeGreaterThan(80);
  });

  it('cruise-straight primitive at high speed moves forward', () => {
    const sim = parametricForwardV2(DEFAULT_LEARNED_PARAMS_V2, cfg);
    const set = coarseWheeledControls({ config: cfg });
    const lib = characterizeVehicle({
      forwardSim: sim,
      controlSets: set,
      duration: 0.5,
      substeps: 6,
      startSpeeds: [10],
    });
    // The first action in coarse is cruise: steer=0, drive=full, brake=0.
    const cruise = lib.primitives[0]!;
    expect(cruise.end.dx).toBeGreaterThan(3); // moved forward at ~10 m/s
    expect(Math.abs(cruise.end.dz)).toBeLessThan(0.5); // ~ straight line
    expect(cruise.end.speed).toBeGreaterThanOrEqual(10);
  });
});
