// IGHA*-H̄ resolution-shift decision. Pure: given the current level and a
// stagnation signal (expansions since the incumbent last improved), decide
// whether to step to the next finer resolution. A hysteresis band prevents
// thrash when the signal hovers near the threshold.

export interface HysteresisOptions {
  /** Expansions-since-improvement at which a shift is considered. */
  threshold: number;
  /** Extra margin above `threshold` actually required to shift (stickiness). */
  band: number;
}

export const DEFAULT_HYSTERESIS: HysteresisOptions = { threshold: 256, band: 64 };

/** Returns the level to search at next: `current` or `current + 1`, clamped
 *  to `maxLevel`. Pure and idempotent for fixed inputs. */
export function decideLevel(
  current: number,
  maxLevel: number,
  signal: number,
  opts: HysteresisOptions,
): number {
  if (current >= maxLevel) return maxLevel;
  if (signal >= opts.threshold + opts.band) return current + 1;
  return current;
}
