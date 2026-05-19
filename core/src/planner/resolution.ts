// Multi-resolution dominance bookkeeping + cell-key packing helpers used by
// Environment implementations to build Node.index. Keys are strings so
// composite multi-dimensional keys (incl. the time dimension, M4) compose
// without integer-overflow concerns.

export type CellKey = string;

/** Pack 2 integer cell coordinates into a dominance key. */
export function pack2(a: number, b: number): CellKey {
  return `${a}:${b}`;
}

/** Pack 3 integer cell coordinates into a dominance key. */
export function pack3(a: number, b: number, c: number): CellKey {
  return `${a}:${b}:${c}`;
}

/** Best g-cost seen per (level, cell) — the dominance front per resolution. */
export class DominanceTable {
  private readonly maps: Map<CellKey, number>[];

  constructor(levels: number) {
    this.maps = Array.from({ length: Math.max(1, levels) }, () => new Map<CellKey, number>());
  }

  best(level: number, key: CellKey): number {
    return this.maps[level]?.get(key) ?? Infinity;
  }

  /** Record `g` for (level, key) if it strictly improves; report improvement. */
  relax(level: number, key: CellKey, g: number, eps = 1e-9): boolean {
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
