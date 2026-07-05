import { describe, it, expect } from 'vitest';
import { smoothTrajectory } from '../../src/execute/trajectory-smoother';
import type { CarKinematicState } from '../../src/agent/types';

function sparseDoglegPath(): CarKinematicState[] {
  // Two straight segments meeting at a 60° corner — exactly the sort of
  // sharp seam a primitive-based A* polyline contains.
  return [
    { x: 0, z: 0, heading: 0, speed: 6, t: 0 },
    { x: 5, z: 0, heading: 0, speed: 6, t: 0.8 },
    { x: 10, z: 0, heading: 0, speed: 6, t: 1.6 },
    { x: 13, z: 4, heading: Math.PI / 4, speed: 6, t: 2.4 },
    { x: 16, z: 8, heading: Math.PI / 4, speed: 6, t: 3.2 },
  ];
}

describe('smoothTrajectory', () => {
  it('returns a densely-sampled polyline at the requested spacing', () => {
    const out = smoothTrajectory(sparseDoglegPath(), { sampleSpacing: 0.4 });
    // Roughly 20 m / 0.4 m = ~50 samples.
    expect(out.length).toBeGreaterThan(30);
    expect(out.length).toBeLessThan(80);
    // Inter-sample spacing should be ≈ 0.4 m (allow 50% slack since smoothing
    // moves points slightly).
    for (let i = 1; i < out.length; i++) {
      const ds = Math.hypot(out[i]!.x - out[i - 1]!.x, out[i]!.z - out[i - 1]!.z);
      expect(ds).toBeGreaterThan(0.05);
      expect(ds).toBeLessThan(0.8);
    }
  });

  it('preserves the start and end points (anchored)', () => {
    const path = sparseDoglegPath();
    const out = smoothTrajectory(path);
    expect(out[0]!.x).toBeCloseTo(path[0]!.x, 5);
    expect(out[0]!.z).toBeCloseTo(path[0]!.z, 5);
    expect(out[out.length - 1]!.x).toBeCloseTo(path[path.length - 1]!.x, 5);
    expect(out[out.length - 1]!.z).toBeCloseTo(path[path.length - 1]!.z, 5);
  });

  it('spreads heading discontinuity at primitive seams across many samples', () => {
    const path = sparseDoglegPath();
    const out = smoothTrajectory(path, { sampleSpacing: 0.4, iterations: 30 });
    function headingDeltas(p: ReadonlyArray<CarKinematicState>): number[] {
      const ds: number[] = [];
      for (let i = 1; i < p.length; i++) {
        const raw = p[i]!.heading - p[i - 1]!.heading;
        ds.push(Math.abs(((raw + Math.PI) % (2 * Math.PI)) - Math.PI));
      }
      return ds;
    }
    const inputDs = headingDeltas(path);
    const outDs = headingDeltas(out);
    const inputMax = Math.max(...inputDs);
    const outMax = Math.max(...outDs);
    // Max jump is at least halved — the seam is no longer a single step.
    expect(inputMax).toBeGreaterThan(0.5);
    expect(outMax).toBeLessThan(inputMax * 0.6);
    // Mean inter-sample heading delta is small (most of the polyline is
    // straight and the turn is now spread across many tiny steps).
    const meanOut = outDs.reduce((a, b) => a + b, 0) / outDs.length;
    expect(meanOut).toBeLessThan(0.05);
  });

  it('keeps a straight line straight (idempotent on smooth input)', () => {
    const path: CarKinematicState[] = [];
    for (let i = 0; i <= 10; i++) {
      path.push({ x: i, z: 0, heading: 0, speed: 6, t: i / 6 });
    }
    const out = smoothTrajectory(path, { sampleSpacing: 0.5, iterations: 30 });
    // Every smoothed sample stays within 1 cm of the z=0 line.
    for (const p of out) expect(Math.abs(p.z)).toBeLessThan(0.01);
  });

  it('handles short paths without crashing', () => {
    const short: CarKinematicState[] = [
      { x: 0, z: 0, heading: 0, speed: 6, t: 0 },
      { x: 0.1, z: 0, heading: 0, speed: 6, t: 0.02 },
    ];
    const out = smoothTrajectory(short);
    expect(out.length).toBeGreaterThanOrEqual(2);
  });

  it('does not mutate the input', () => {
    const path = sparseDoglegPath();
    const snapshot = path.map((p) => ({ ...p }));
    smoothTrajectory(path);
    for (let i = 0; i < path.length; i++) {
      expect(path[i]!.x).toBe(snapshot[i]!.x);
      expect(path[i]!.heading).toBe(snapshot[i]!.heading);
    }
  });
});
