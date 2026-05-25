import { describe, expect, it } from 'vitest';
import {
  CAR_COVERAGE_AXES,
  carCoverageProjection,
  wheeledControlsToVec,
} from 'kinocat/vehicle/car';
import { createCoverageMeter } from 'kinocat/training';
import type { CarKinematicState, WheeledCarControls } from 'kinocat/vehicle/car';

describe('car coverage projection', () => {
  it('returns a vector with one entry per axis', () => {
    const state: CarKinematicState = {
      x: 0, z: 0, heading: 0, speed: 12, t: 0, yawRate: 0.5, lateralVelocity: 2,
    };
    const ctrls = wheeledControlsToVec({ steer: 0.1, driveForce: 3000, brakeForce: 0 });
    const v = carCoverageProjection(state, ctrls, undefined);
    expect(v.length).toBe(CAR_COVERAGE_AXES.length);
  });

  it('discriminates throttle / brake / coast / combined inputs', () => {
    const state: CarKinematicState = { x: 0, z: 0, heading: 0, speed: 5, t: 0 };
    const inputKindIdx = CAR_COVERAGE_AXES.findIndex((a) => a.name === 'inputKind');
    const coast = carCoverageProjection(state, [0, 0, 0], undefined)[inputKindIdx]!;
    const drive = carCoverageProjection(state, [0, 2000, 0], undefined)[inputKindIdx]!;
    const brake = carCoverageProjection(state, [0, 0, 1000], undefined)[inputKindIdx]!;
    const both = carCoverageProjection(state, [0, 2000, 1000], undefined)[inputKindIdx]!;
    const allFour = new Set([
      Math.floor(coast), Math.floor(drive), Math.floor(brake), Math.floor(both),
    ]);
    expect(allFour.size).toBe(4);
  });

  it('integrates with createCoverageMeter end-to-end', () => {
    const meter = createCoverageMeter<CarKinematicState, WheeledCarControls, unknown>({
      axes: CAR_COVERAGE_AXES,
      project: carCoverageProjection,
      controlsToVec: wheeledControlsToVec,
    });
    meter.record({
      id: 't', initialState: { x: 0, z: 0, heading: 0, speed: 10, t: 0 },
      controlsTrace: [{ steer: 0.1, driveForce: 2000, brakeForce: 0 }],
      dt: 0.05, samples: [
        { t: 0, state: { x: 0, z: 0, heading: 0, speed: 10, t: 0 } },
        { t: 0.05, state: { x: 0.5, z: 0, heading: 0.01, speed: 10.1, t: 0.05 } },
      ],
      config: undefined, configKey: 'A', split: 'train',
    });
    expect(meter.summary().length).toBeGreaterThan(0);
  });
});
