// Per-round trial cache for the CLI training pipeline (`pnpm run train`).
//
// Saves Rapier-collected trials to disk so re-runs with identical parameters
// skip the expensive physics simulation. Each round is cached independently
// under a content-addressed key derived from every parameter that affects
// trial generation.
//
// **Bump `CACHE_BUSTER` when you change maneuver bundle generation,
// controls-trace building, headless-trial harness physics, or the
// start-speed schedule logic.**

import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { createRequire } from 'node:module';
import type { Trial } from 'kinocat/learning';
import type { CarKinematicState, WheeledCarControls, LearnableVehicleConfig } from 'kinocat/agent';

/** Bump this when trial-generation logic changes in a way the cache key
 *  inputs can't detect (e.g. new maneuver types, harness physics tweaks).
 *
 *  v2: defaultManeuverBundle rebalanced for racing — 40% OU (was 60%),
 *      25% racing primitives (raceSlalom / raceBrakeIntoCorner /
 *      raceSustainedTurn / raceThrottleOnApex; new), 5% named ident
 *      (was 10%), other shares unchanged. Trial distributions and ids
 *      are now different for the same seed, so cached trials from
 *      pre-v2 runs are stale. */
export const CACHE_BUSTER = 2;

export interface TrialCacheKeyInputs {
  seed: number;
  round: number;
  trialsPerRound: number;
  trialTicks: number;
  sampleEveryNTicks: number;
  bundle: 'default' | 'universal';
  startSpeedSchedule: number[];
  vehicleOptions: Record<string, unknown>;
  rapierVersion: string;
  cacheBuster: number;
}

type CachedTrial = Trial<CarKinematicState, WheeledCarControls, LearnableVehicleConfig>;

interface CacheFile {
  version: 1;
  cacheKey: string;
  params: TrialCacheKeyInputs;
  createdAt: number;
  trials: CachedTrial[];
}

export function computeCacheKey(inputs: TrialCacheKeyInputs): string {
  const canonical = JSON.stringify(inputs, Object.keys(inputs).sort());
  return createHash('sha256').update(canonical).digest('hex').slice(0, 16);
}

function roundCachePath(cacheDir: string, cacheKey: string): string {
  return join(cacheDir, 'rounds', `${cacheKey}.json`);
}

/** Try to read cached trials for a round. Returns null on miss. */
export function tryReadRoundCache(
  cacheDir: string,
  cacheKey: string,
): CachedTrial[] | null {
  try {
    const raw = readFileSync(roundCachePath(cacheDir, cacheKey), 'utf-8');
    const parsed: CacheFile = JSON.parse(raw);
    if (parsed.version !== 1 || parsed.cacheKey !== cacheKey) return null;
    return parsed.trials;
  } catch {
    return null;
  }
}

/** Write trials for a round to the cache. Creates directories as needed. */
export function writeRoundCache(
  cacheDir: string,
  cacheKey: string,
  trials: CachedTrial[],
  params: TrialCacheKeyInputs,
): void {
  const dir = join(cacheDir, 'rounds');
  mkdirSync(dir, { recursive: true });
  const payload: CacheFile = {
    version: 1,
    cacheKey,
    params,
    createdAt: Date.now(),
    trials,
  };
  writeFileSync(roundCachePath(cacheDir, cacheKey), JSON.stringify(payload));
}

/** Resolve the Rapier major.minor version string for cache key inclusion. */
export function getRapierVersionTag(): string {
  try {
    const req = createRequire(import.meta.url);
    const pkg = req('@dimforge/rapier3d-compat/package.json') as { version: string };
    return pkg.version.split('.').slice(0, 2).join('.');
  } catch {
    return 'unknown';
  }
}
