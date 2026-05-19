import { describe, it, expect } from 'vitest';
import { DominanceTable, pack2, pack3 } from '../../src/planner/resolution';

describe('pack2 / pack3', () => {
  it('pack2 is injective over a range incl. negatives', () => {
    const seen = new Set<number>();
    for (let a = -50; a <= 50; a++) {
      for (let b = -50; b <= 50; b++) {
        const k = pack2(a, b);
        expect(Number.isSafeInteger(k)).toBe(true);
        expect(seen.has(k)).toBe(false);
        seen.add(k);
      }
    }
  });

  it('pack3 is injective over a small range', () => {
    const seen = new Set<number>();
    for (let a = -8; a <= 8; a++)
      for (let b = -8; b <= 8; b++)
        for (let c = -8; c <= 8; c++) {
          const k = pack3(a, b, c);
          expect(Number.isSafeInteger(k)).toBe(true);
          expect(seen.has(k)).toBe(false);
          seen.add(k);
        }
  });
});

describe('DominanceTable', () => {
  it('unknown cells are Infinity; relax records strictly-better g only', () => {
    const d = new DominanceTable(2);
    expect(d.best(0, 42)).toBe(Infinity);
    expect(d.relax(0, 42, 10)).toBe(true);
    expect(d.best(0, 42)).toBe(10);
    expect(d.relax(0, 42, 10)).toBe(false); // not strictly better
    expect(d.relax(0, 42, 9.9999999999)).toBe(false); // within eps
    expect(d.relax(0, 42, 5)).toBe(true);
    expect(d.best(0, 42)).toBe(5);
  });

  it('levels are independent and clear() resets', () => {
    const d = new DominanceTable(2);
    d.relax(0, 1, 3);
    d.relax(1, 1, 7);
    expect(d.best(0, 1)).toBe(3);
    expect(d.best(1, 1)).toBe(7);
    d.clear();
    expect(d.best(0, 1)).toBe(Infinity);
    expect(d.best(1, 1)).toBe(Infinity);
  });

  it('out-of-range level relax is a no-op', () => {
    const d = new DominanceTable(1);
    expect(d.relax(5, 0, 1)).toBe(false);
  });
});
