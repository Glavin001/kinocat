import { describe, it, expect } from 'vitest';
import { smoothSpeedProfile } from '../../src/execute/speed-profile';
import type { CarKinematicState } from '../../src/agent/types';

function straightLine(n: number, ds: number, vEntry: number): CarKinematicState[] {
  const out: CarKinematicState[] = [];
  for (let i = 0; i < n; i++) {
    out.push({ x: i * ds, z: 0, heading: 0, speed: vEntry, t: i * (ds / vEntry) });
  }
  return out;
}

function quarterCircle(n: number, R: number, vEntry: number): CarKinematicState[] {
  const out: CarKinematicState[] = [];
  for (let i = 0; i < n; i++) {
    const u = i / (n - 1);
    const theta = u * (Math.PI / 2);
    out.push({
      x: R * Math.sin(theta),
      z: R - R * Math.cos(theta),
      heading: theta,
      speed: vEntry,
      t: i,
    });
  }
  return out;
}

describe('smoothSpeedProfile', () => {
  it('leaves a straight line at cruise speed alone (no curvature)', () => {
    const path = straightLine(10, 1, 8);
    const out = smoothSpeedProfile(path, {
      aLatMax: 4,
      aLonMaxAccel: 6,
      aLonMaxDecel: 8,
      maxSpeed: 8,
    });
    expect(out.length).toBe(path.length);
    // Speed magnitude stays at 8 within tolerance.
    for (const p of out) expect(Math.abs(p.speed)).toBeCloseTo(8, 1);
    // Geometry preserved.
    for (let i = 0; i < path.length; i++) {
      expect(out[i]!.x).toBeCloseTo(path[i]!.x);
      expect(out[i]!.z).toBeCloseTo(path[i]!.z);
    }
  });

  it('caps speed through a tight corner at sqrt(aLat/k)', () => {
    const R = 5; // curvature k = 1/R = 0.2
    const aLat = 4;
    const vCap = Math.sqrt(aLat / (1 / R)); // = sqrt(20) ≈ 4.47
    const path = quarterCircle(20, R, 8); // entry 8, well above cap
    const out = smoothSpeedProfile(path, {
      aLatMax: aLat,
      aLonMaxAccel: 6,
      aLonMaxDecel: 8,
      maxSpeed: 10,
      honorEntrySpeed: false,
    });
    // Middle of the arc must be at or below vCap (allow 5% slack).
    const mid = out[Math.floor(out.length / 2)]!;
    expect(Math.abs(mid.speed)).toBeLessThan(vCap * 1.05);
  });

  it('brakes BEFORE a tight corner (backward pass)', () => {
    // Straight approach then corner: speeds before the corner must drop.
    const R = 4;
    const aLat = 4;
    const straight = straightLine(10, 1, 8); // entry at 8 m/s, ds=1m
    const corner = quarterCircle(15, R, 8).map((p, i) => ({
      ...p,
      x: p.x + 9, // shift so it starts after the straight
      t: 10 + i,
    }));
    const path = [...straight, ...corner];
    const out = smoothSpeedProfile(path, {
      aLatMax: aLat,
      aLonMaxAccel: 6,
      aLonMaxDecel: 8,
      maxSpeed: 8,
      honorEntrySpeed: false,
    });
    // Speed at the entry-to-corner sample should be < cruise (8) because
    // braking distance was applied during the straight.
    const cornerEntry = out[10]!;
    expect(Math.abs(cornerEntry.speed)).toBeLessThan(8);
  });

  it('preserves reverse-segment sign', () => {
    const path = straightLine(5, 1, 8).map((p) => ({ ...p, speed: -p.speed }));
    const out = smoothSpeedProfile(path, {
      aLatMax: 4,
      aLonMaxAccel: 6,
      aLonMaxDecel: 8,
      maxSpeed: 8,
    });
    for (const p of out) expect(p.speed).toBeLessThanOrEqual(0);
  });

  it('returns a copy (does not mutate input)', () => {
    const path = straightLine(5, 1, 4);
    const before = path.map((p) => ({ ...p }));
    smoothSpeedProfile(path, { aLatMax: 4, aLonMaxAccel: 4, aLonMaxDecel: 4 });
    for (let i = 0; i < path.length; i++) {
      expect(path[i]!.speed).toBe(before[i]!.speed);
      expect(path[i]!.x).toBe(before[i]!.x);
    }
  });
});
