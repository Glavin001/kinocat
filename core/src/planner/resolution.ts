// Multi-resolution dominance bookkeeping + integer cell packing helpers used
// by Environment implementations to build Node.index.

const OFFSET = 1 << 20; // shift coords non-negative; supports ±~1e6 cells
const STRIDE = 1 << 22;

/** Pack 2 small signed integers into one safe JS integer (< 2^53). */
export function pack2(a: number, b: number): number {
  return (a + OFFSET) * STRIDE + (b + OFFSET);
}

/** Pack 3 small signed integers. Range per field ~±2^16 to stay < 2^53. */
export function pack3(a: number, b: number, c: number): number {
  const S = 1 << 17;
  const O = 1 << 16;
  return ((a + O) * S + (b + O)) * S + (c + O);
}

/** Best g-cost seen per (level, cell) — the dominance front per resolution. */
export class DominanceTable {
  private readonly maps: Map<number, number>[];

  constructor(levels: number) {
    this.maps = Array.from({ length: Math.max(1, levels) }, () => new Map<number, number>());
  }

  best(level: number, key: number): number {
    return this.maps[level]?.get(key) ?? Infinity;
  }

  /** Record `g` for (level, key) if it strictly improves; report improvement. */
  relax(level: number, key: number, g: number, eps = 1e-9): boolean {
    const m = this.maps[level];
    if (!m) return false;
    const cur = m.get(key);
    if (cur === undefined || g < cur - eps) {
      m.set(key, g);
      return true;
    }
    return false;
  }

  clear(): void {
    for (const m of this.maps) m.clear();
  }
}
