// Generic "every divergence the user sees is a training trial we are
// missing" miner. Subscribe it to a per-tick stream of
// `(state, controls, real, predicted)` frames; supply a `gapPredicate`
// that returns `true` for the frames that should be promoted into the
// hard-cases pool; the miner buffers the surrounding window and emits
// a `Trial` covering it.
//
// Domain-agnostic — `S`, `C`, `Cfg` are opaque. The car-side wiring
// supplies a predicate like "Δheading > 10° between ghost and real" or
// "Δposition > 1.5m at horizon 1.0s" and a trial-id factory.

import type { Trial, TrialSplit } from '../learning/trial-store';
import { assignSplit } from '../learning/trial-store';

/** One tick of observed deployment data. */
export interface MinerFrame<S, C> {
  simTime: number;
  state: S;
  controls: C;
  /** Predicted state at this tick (e.g. the ghost). Optional — the
   *  predicate may use only `state` if it's looking for absolute
   *  conditions (collisions, off-track, …). */
  predicted?: S;
}

export interface HardExampleMinerOptions<S, C, Cfg> {
  /** True when this frame is divergent enough to be worth promoting. */
  gapPredicate: (frame: MinerFrame<S, C>) => boolean;
  /** Window radius (in ticks) captured around each trigger. The emitted
   *  trial spans `2*windowTicks + 1` frames centered on the trigger. */
  windowTicks: number;
  /** Tick interval (s). */
  dt: number;
  /** Sample every Nth state into the emitted `samples`. */
  sampleEveryNTicks: number;
  /** Vehicle config to attach to emitted trials. */
  config: Cfg;
  /** Stable config key. */
  configKey: string;
  /** Where the miner gets its trial ids. Defaults to a counter. */
  idFactory?: (n: number) => string;
  /** Scenario id to attach to mined trials (helps the split policy keep
   *  miner-captured trials distinct from synthetic ones). */
  scenarioId?: string;
  /** Optional split override; default = hash-based. */
  split?: TrialSplit;
  /** Cooldown ticks between successive trigger captures to avoid emitting
   *  N nearly-identical trials around one long divergence. Default = the
   *  window size, so captures never overlap. */
  cooldownTicks?: number;
}

export interface HardExampleMiner<S, C, Cfg> {
  /** Feed the next observed frame. May emit zero or one trials. */
  observe(frame: MinerFrame<S, C>): Trial<S, C, Cfg> | null;
  /** Number of trials emitted so far. */
  emittedCount(): number;
  /** Reset the buffer + counters. */
  reset(): void;
}

/** Build a miner. Pure: the only side-effects are inside the returned
 *  object's closure state. */
export function createHardExampleMiner<S, C, Cfg>(
  opts: HardExampleMinerOptions<S, C, Cfg>,
): HardExampleMiner<S, C, Cfg> {
  const ring: MinerFrame<S, C>[] = [];
  const windowTicks = Math.max(1, opts.windowTicks);
  const cooldown = Math.max(0, opts.cooldownTicks ?? windowTicks);
  const sampleEvery = Math.max(1, opts.sampleEveryNTicks);
  const idFactory = opts.idFactory ?? ((n: number) => `mined-${n}`);
  let emitted = 0;
  let cooldownLeft = 0;

  function tryEmit(): Trial<S, C, Cfg> | null {
    // Need at least `2*windowTicks+1` frames + a trigger at the center.
    const needed = 2 * windowTicks + 1;
    if (ring.length < needed) return null;
    const centerIdx = windowTicks;
    const center = ring[centerIdx]!;
    if (!opts.gapPredicate(center)) return null;
    if (cooldownLeft > 0) return null;
    const frames = ring.slice(0, needed);
    const dt = opts.dt;
    const controls: C[] = frames.slice(0, -1).map((f) => f.controls);
    const samples: { t: number; state: S }[] = [];
    for (let i = 0; i < frames.length; i += sampleEvery) {
      samples.push({ t: i * dt, state: frames[i]!.state });
    }
    const id = idFactory(emitted++);
    const trial: Trial<S, C, Cfg> = {
      id,
      initialState: frames[0]!.state,
      controlsTrace: controls,
      dt,
      samples,
      config: opts.config,
      configKey: opts.configKey,
      maneuverId: 'mined',
      maneuverParams: { trigger: center.simTime },
      scenarioId: opts.scenarioId,
    };
    trial.split = opts.split ?? assignSplit(trial);
    cooldownLeft = cooldown;
    return trial;
  }

  return {
    observe(frame) {
      ring.push(frame);
      if (cooldownLeft > 0) cooldownLeft--;
      // Trigger evaluation requires `2*windowTicks + 1` frames in the
      // buffer; the trigger is at the CENTER so we evaluate it once the
      // post-window has filled in. Then drop the oldest.
      const needed = 2 * windowTicks + 1;
      if (ring.length > needed) ring.shift();
      const emittedTrial = tryEmit();
      if (emittedTrial) {
        // Drop the buffered window so we don't re-emit on the same frames.
        ring.length = 0;
      }
      return emittedTrial;
    },
    emittedCount() {
      return emitted;
    },
    reset() {
      ring.length = 0;
      emitted = 0;
      cooldownLeft = 0;
    },
  };
}
