import { describe, it, expect } from 'vitest';
import {
  wrapAngle,
  angleDiff,
  clamp,
  lerp,
  lerpAngle,
  dist,
  dist2,
  quantize,
  approxEqual,
  TWO_PI,
} from '../../src/internal/math';

describe('math', () => {
  it('wrapAngle maps to (-pi, pi]', () => {
    expect(approxEqual(wrapAngle(0), 0)).toBe(true);
    expect(approxEqual(wrapAngle(Math.PI), Math.PI)).toBe(true);
    expect(approxEqual(wrapAngle(-Math.PI), Math.PI)).toBe(true);
    expect(approxEqual(wrapAngle(TWO_PI), 0)).toBe(true);
    expect(approxEqual(wrapAngle(3 * Math.PI), Math.PI)).toBe(true);
    expect(approxEqual(wrapAngle(-3 * Math.PI), Math.PI)).toBe(true);
  });

  it('angleDiff returns shortest signed delta', () => {
    expect(approxEqual(angleDiff(0, Math.PI / 2), Math.PI / 2)).toBe(true);
    expect(approxEqual(angleDiff(0.1, -0.1), -0.2)).toBe(true);
    expect(approxEqual(angleDiff(-3, 3), 6 - TWO_PI)).toBe(true);
  });

  it('clamp/lerp', () => {
    expect(clamp(5, 0, 1)).toBe(1);
    expect(clamp(-5, 0, 1)).toBe(0);
    expect(clamp(0.5, 0, 1)).toBe(0.5);
    expect(lerp(0, 10, 0.25)).toBe(2.5);
  });

  it('lerpAngle takes the short arc across the wrap', () => {
    const r = lerpAngle(3, -3, 0.5);
    // halfway between 3 and -3 the short way passes through ±pi
    expect(Math.abs(Math.abs(r) - Math.PI)).toBeLessThan(0.2);
  });

  it('dist/dist2', () => {
    expect(dist2(0, 0, 3, 4)).toBe(25);
    expect(dist(0, 0, 3, 4)).toBe(5);
  });

  it('quantize buckets by cell size', () => {
    expect(quantize(0.0, 0.5)).toBe(0);
    expect(quantize(0.6, 0.5)).toBe(1);
    expect(quantize(-0.1, 0.5)).toBe(-1);
  });
});
