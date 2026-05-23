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
}

export interface TrialStore<S, C, Cfg> {
  add(trial: Trial<S, C, Cfg>): void;
  all(): ReadonlyArray<Trial<S, C, Cfg>>;
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
    all() {
      return trials;
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
