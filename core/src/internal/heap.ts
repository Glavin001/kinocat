// Tiny array-backed binary min-heap. Zero deps. Lazy-deletion friendly:
// callers push duplicates and discard stale items on pop.

export class BinaryHeap<T> {
  private readonly data: T[] = [];

  /** `compare(a, b) < 0` means `a` has higher priority (pops first). */
  constructor(private readonly compare: (a: T, b: T) => number) {}

  get size(): number {
    return this.data.length;
  }

  isEmpty(): boolean {
    return this.data.length === 0;
  }

  peek(): T | undefined {
    return this.data[0];
  }

  push(item: T): void {
    const d = this.data;
    d.push(item);
    let i = d.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (this.compare(d[i]!, d[parent]!) < 0) {
        const tmp = d[i]!;
        d[i] = d[parent]!;
        d[parent] = tmp;
        i = parent;
      } else break;
    }
  }

  pop(): T | undefined {
    const d = this.data;
    const n = d.length;
    if (n === 0) return undefined;
    const top = d[0]!;
    const last = d.pop()!;
    if (n > 1) {
      d[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let smallest = i;
        if (l < d.length && this.compare(d[l]!, d[smallest]!) < 0) smallest = l;
        if (r < d.length && this.compare(d[r]!, d[smallest]!) < 0) smallest = r;
        if (smallest === i) break;
        const tmp = d[i]!;
        d[i] = d[smallest]!;
        d[smallest] = tmp;
        i = smallest;
      }
    }
    return top;
  }

  clear(): void {
    this.data.length = 0;
  }
}
