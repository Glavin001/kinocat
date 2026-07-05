import { describe, it, expect } from 'vitest';
import { toReferenceTrajectory } from '../../src/eval/reference-trajectory';
import { ggUtilization } from '../../src/eval/gg-utilization';
import { arcPath, straightLine } from '../../src/eval/reference-shapes';

describe('ggUtilization', () => {
  it('matches v²κ/limit for a constant-radius constant-speed arc', () => {
    const R = 10;
    const v = 6;
    const frictionLimit = 4;
    const ref = toReferenceTrajectory(arcPath({ radius: R, sweep: Math.PI / 2, speed: v, ds: 0.25 }));
    const gg = ggUtilization(ref, frictionLimit);
    // Expected lateral-only utilization (a_long ≈ 0 at constant speed).
    const expected = (v * v / R) / frictionLimit; // 36/10/4 = 0.9
    // Peak should approach the expected steady-state value.
    expect(gg.peakUtil).toBeGreaterThan(expected * 0.8);
    expect(gg.peakUtil).toBeLessThan(expected * 1.2);
  });

  it('reports near-zero utilization for a slow straight line (timid)', () => {
    const ref = toReferenceTrajectory(straightLine({ length: 20, speed: 2, ds: 0.5 }));
    const gg = ggUtilization(ref, 4);
    expect(gg.meanUtil).toBeLessThan(0.1);
  });
});
