import type { MotionPrimitive, SerializedLibrary } from './types';
import { toJSON, fromJSON } from '../internal/serialize';

/** A JSON-serializable set of motion primitives indexed by start-speed bucket. */
export class MotionPrimitiveLibrary {
  readonly primitives: MotionPrimitive[];
  readonly startSpeeds: number[];
  private readonly byBucket = new Map<number, MotionPrimitive[]>();

  constructor(primitives: MotionPrimitive[], startSpeeds: number[]) {
    this.primitives = primitives;
    this.startSpeeds = [...startSpeeds].sort((a, b) => a - b);
    for (const p of primitives) {
      const list = this.byBucket.get(p.startSpeed) ?? [];
      list.push(p);
      this.byBucket.set(p.startSpeed, list);
    }
  }

  /** Nearest start-speed bucket to `speed`. */
  private nearestBucket(speed: number): number {
    let best = this.startSpeeds[0] ?? 0;
    let bestD = Infinity;
    for (const s of this.startSpeeds) {
      const d = Math.abs(s - speed);
      if (d < bestD) {
        bestD = d;
        best = s;
      }
    }
    return best;
  }

  /** Primitives applicable from the given current speed. */
  lookup(speed: number): MotionPrimitive[] {
    return this.byBucket.get(this.nearestBucket(speed)) ?? [];
  }

  toJSON(): string {
    const data: SerializedLibrary = {
      primitives: this.primitives,
      startSpeeds: this.startSpeeds,
    };
    return toJSON(data);
  }

  static fromJSON(json: string): MotionPrimitiveLibrary {
    const data = fromJSON<SerializedLibrary>(json);
    return new MotionPrimitiveLibrary(data.primitives, data.startSpeeds);
  }
}
