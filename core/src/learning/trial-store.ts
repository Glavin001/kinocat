// Generic trial database for `(state, controls, dt, next_state)` data
// collected from a ground-truth physics rollout. Domain-agnostic — `S` is the
// state type, `Cfg` is the config type. Persistence (localStorage, file) is
// the consumer's responsibility; this module only provides in-memory storage
// + JSON round-trip.

export interface TrialSample<S> {
  /** Time since trial start, seconds. */
  t: number;
  state: S;
}

/** Train / validation / test partition assignment. Hash-stable per trial
 *  (default policy: hash of `(maneuverId, maneuverParams, configKey,
 *  scenarioId)`) so the same logical maneuver always lands in the same
 *  split — eliminates leakage when active exploration asks for "more
 *  like this one". The `test` partition is FROZEN for the life of the
 *  project: never modified by any fit, never touched by active
 *  exploration. Cross-phase progress reports always quote the test-set
 *  numbers. */
export type TrialSplit = 'train' | 'val' | 'test';

export interface Trial<S, C, Cfg> {
  /** Stable id for bookkeeping. */
  id: string;
  /** Initial state at t=0 (also the first entry in `samples`). */
  initialState: S;
  /** Native controls trace, one per sub-tick of length `dt`. */
  controlsTrace: C[];
  /** Physics dt the trial was run at. */
  dt: number;
  /** Sub-sampled state recordings. Always includes t=0 first. */
  samples: TrialSample<S>[];
  /** Vehicle config the trial was run with. */
  config: Cfg;
  /** Stable key identifying which config this trial belongs to (used for
   *  grouping when training across multiple configs). */
  configKey: string;
  /** Optional split partition. When absent, consumers may default to
   *  `train` or compute a hash-based assignment via `assignSplit`. */
  split?: TrialSplit;
  /** Optional maneuver identifier (Phase 1+). Lets the coverage meter
   *  group by maneuver class and the hash-based split policy keep the
   *  same maneuver in the same partition. */
  maneuverId?: string;
  /** Optional maneuver parameters as a flat record (numbers only). */
  maneuverParams?: Record<string, number>;
  /** Optional closed-loop scenario identifier (Phase 3+). */
  scenarioId?: string;
  /** Optional terrain identifier (Phase 5+). */
  terrainKind?: string;
}

export interface TrialStore<S, C, Cfg> {
  add(trial: Trial<S, C, Cfg>): void;
  all(split?: TrialSplit): ReadonlyArray<Trial<S, C, Cfg>>;
  byConfig(key: string): ReadonlyArray<Trial<S, C, Cfg>>;
  size(): number;
  clear(): void;
}

export function createTrialStore<S, C, Cfg>(): TrialStore<S, C, Cfg> {
  const trials: Trial<S, C, Cfg>[] = [];
  return {
    add(trial) {
      trials.push(trial);
    },
    all(split?: TrialSplit) {
      if (split === undefined) return trials;
      return trials.filter((t) => (t.split ?? 'train') === split);
    },
    byConfig(key) {
      return trials.filter((t) => t.configKey === key);
    },
    size() {
      return trials.length;
    },
    clear() {
      trials.length = 0;
    },
  };
}

export interface SerializedTrials<S, C, Cfg> {
  version: 1;
  trials: Trial<S, C, Cfg>[];
}

export function serializeTrials<S, C, Cfg>(
  store: TrialStore<S, C, Cfg>,
): string {
  const payload: SerializedTrials<S, C, Cfg> = {
    version: 1,
    trials: [...store.all()],
  };
  return JSON.stringify(payload);
}

export function deserializeTrials<S, C, Cfg>(json: string): TrialStore<S, C, Cfg> {
  const obj = JSON.parse(json) as SerializedTrials<S, C, Cfg>;
  if (obj.version !== 1) {
    throw new Error(`Unsupported trial-store version ${obj.version}`);
  }
  const store = createTrialStore<S, C, Cfg>();
  for (const t of obj.trials) store.add(t);
  return store;
}

// ---------------------------------------------------------------------------
// Train / val / test split assignment — Phase 0 of the training-dataset plan.
//
// Hash-based: the same logical maneuver always lands in the same partition,
// so active exploration asking for "more like this one" never leaks the
// held-out set. Default ratios 70/15/15 are the project-wide convention.

export interface SplitPolicy {
  /** Train fraction (0..1). */
  train: number;
  /** Val fraction (0..1). Test gets the remainder. */
  val: number;
}

export const DEFAULT_SPLIT_POLICY: Readonly<SplitPolicy> = Object.freeze({
  train: 0.70,
  val: 0.15,
});

/** FNV-1a 32-bit hash. Cheap, deterministic, stable across runs. */
export function hashString(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/** Stable hash key for a trial under the default split policy. */
export function trialSplitKey<S, C, Cfg>(t: Trial<S, C, Cfg>): string {
  const params = t.maneuverParams
    ? Object.keys(t.maneuverParams).sort().map((k) => `${k}=${t.maneuverParams![k]}`).join('|')
    : '';
  return [
    t.maneuverId ?? 'untagged',
    params,
    t.configKey,
    t.scenarioId ?? '',
    t.terrainKind ?? '',
  ].join('::');
}

/** Decide the split for a trial under a (hash-based) policy. Pure +
 *  deterministic given the key. */
export function assignSplit<S, C, Cfg>(
  t: Trial<S, C, Cfg>,
  policy: SplitPolicy = DEFAULT_SPLIT_POLICY,
): TrialSplit {
  const u = (hashString(trialSplitKey(t)) % 10000) / 10000;
  if (u < policy.train) return 'train';
  if (u < policy.train + policy.val) return 'val';
  return 'test';
}
