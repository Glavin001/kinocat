import { describe, it, expect } from 'vitest';
import { decideLevel } from '../../src/planner/hysteresis';

const opts = { threshold: 100, band: 20 };

describe('decideLevel', () => {
  it('stays while the signal is below threshold+band', () => {
    for (const s of [0, 50, 99, 100, 119]) {
      expect(decideLevel(0, 2, s, opts)).toBe(0);
    }
  });

  it('advances by exactly one when the signal crosses threshold+band', () => {
    expect(decideLevel(0, 2, 120, opts)).toBe(1);
    expect(decideLevel(0, 2, 1e6, opts)).toBe(1);
    expect(decideLevel(1, 2, 120, opts)).toBe(2);
  });

  it('clamps at maxLevel', () => {
    expect(decideLevel(2, 2, 1e9, opts)).toBe(2);
    expect(decideLevel(5, 2, 1e9, opts)).toBe(2);
  });

  it('does not thrash: oscillating signals within the band never advance', () => {
    let level = 0;
    for (let i = 0; i < 1000; i++) {
      const signal = 80 + (i % 40); // stays in [80, 119], inside the band
      level = decideLevel(level, 3, signal, opts);
    }
    expect(level).toBe(0);
  });

  it('is idempotent for fixed inputs', () => {
    expect(decideLevel(0, 2, 130, opts)).toBe(decideLevel(0, 2, 130, opts));
  });
});
