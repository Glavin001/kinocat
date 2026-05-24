import { describe, expect, it } from 'vitest';
import {
  WHEELED_CONTROL_DIM,
  encodeWheeled,
  decodeWheeled,
  clampWheeled,
} from 'kinocat/agent';

describe('WheeledCarControls — encode/decode round-trip', () => {
  it('round-trips identity', () => {
    const c = { steer: 0.3, driveForce: 2000, brakeForce: 500 };
    const vec = encodeWheeled(c);
    expect(vec).toHaveLength(WHEELED_CONTROL_DIM);
    expect(decodeWheeled(vec)).toEqual(c);
  });

  it('decodes missing entries to zero', () => {
    expect(decodeWheeled([])).toEqual({ steer: 0, driveForce: 0, brakeForce: 0 });
    expect(decodeWheeled([0.5])).toEqual({ steer: 0.5, driveForce: 0, brakeForce: 0 });
  });

  it('clamps against limits including negative brake', () => {
    const out = clampWheeled(
      { steer: 99, driveForce: -1e9, brakeForce: -5 },
      { maxSteerAngle: 0.6, maxDriveForce: 4000, maxBrakeForce: 2000 },
    );
    expect(out.steer).toBe(0.6);
    expect(out.driveForce).toBe(-4000);
    expect(out.brakeForce).toBe(0);
  });
});
