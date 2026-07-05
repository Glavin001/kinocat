// The domain conformance kit's public contract. A domain author wraps their
// Environment<State> in a DomainHarness and runs `runConformance` (or the
// individual checks) to prove the implementation satisfies everything the
// IGHA* core assumes: admissible/consistent heuristics, monotone time,
// stable hashing, deterministic replanning, anytime monotonicity, and
// budgeted solvability. Framework-agnostic by design — checks return a
// structured report instead of throwing, so the kit runs under any test
// runner (or in a game's own CI) with zero dependencies.

import type { EdgeRef, Environment } from '../environment/types';

/** A representative planning problem the domain must solve within budget. */
export interface DomainScenario<State> {
  name: string;
  start: State;
  goal: State;
  /** Deterministic budget — an expansion count, NOT wall-clock, so results
   *  are bit-repeatable in CI (maps to `PlannerOptions.maxExpansions`). */
  maxExpansions: number;
}

export interface DomainHarness<State> {
  /** A fresh, fully-configured environment. MUST be deterministic across
   *  calls — two makeEnv() instances given the same plan request must
   *  behave identically (this is itself verified by `checkDeterminism`). */
  makeEnv(): Environment<State>;
  /** Deterministic valid-state sampler driven by the kit's seeded rng.
   *  Should cover the domain's interesting envelope: positions near and far
   *  from obstacles, the full heading range, speed/attitude extremes, and a
   *  spread of times. States that fail the env's own start-validity check
   *  are skipped (sampling near obstacles is encouraged, not penalized). */
  sampleState(rand: () => number): State;
  /** Representative problems, ≥ 3 recommended, including at least one
   *  obstacle-constrained one. Used by the plan-level checks and as goal
   *  poses for the sampled-state checks. */
  scenarios: DomainScenario<State>[];
  /** Numeric tolerance for cost / state comparisons (default 1e-9). */
  eps?: number;
  /** Optional hooks for `checkSuccessorFidelity`: prove that succ()'s
   *  cached/transformed successors match what the domain's forward sim
   *  actually produces from the parent state. Supplying this makes the
   *  fidelity check part of `runConformance`. */
  fidelity?: FidelityHooks<State>;
}

/** Hooks for the successor-fidelity check. `resimulate` re-runs the edge's
 *  dynamics from the ACTUAL parent state (reconstruct the controls from
 *  `edge.data` and roll the domain's ForwardSim); return null for edge kinds
 *  that cannot be re-simulated (affordances, analytic shots) — they are
 *  skipped. `tolerance` is the max |cached − resimulated| per numeric state
 *  field: near machine-eps for environments that integrate live or from
 *  exact buckets; the bucket-quantization magnitude for environments that
 *  deliberately apply primitives from nearest-bucket canonical starts (the
 *  check then MEASURES the teleport error and pins it from growing). */
export interface FidelityHooks<State> {
  resimulate: (parent: State, edge: EdgeRef) => State | null;
  tolerance: number;
  /** State fields to compare on the circle (deviation taken mod 2π), e.g.
   *  ['heading'] — a cached −3.08 rad and a re-simulated +3.06 rad are the
   *  same physical angle. */
  angularFields?: string[];
}

export interface ConformanceFailure {
  /** Which check failed, e.g. 'heuristic-consistency'. */
  check: string;
  /** Human-readable description including the offending values. */
  message: string;
  /** The state / edge / scenario that failed (JSON-serializable). */
  sample?: unknown;
}

export interface ConformanceReport {
  ok: boolean;
  /** Names of every check that ran. */
  checks: string[];
  failures: ConformanceFailure[];
}

export interface CheckOptions {
  /** Seed for the deterministic sampler (default 0xc0ffee). */
  seed?: number;
  /** Sampled states per sampling-based check (default 200). */
  samples?: number;
}

/** Effective tolerance: harness eps floored at 1e-9 so float noise in an
 *  otherwise-exact domain never trips a check. */
export function tol(h: { eps?: number }): number {
  return Math.max(h.eps ?? 1e-9, 1e-9);
}

/** Structural numeric comparison of two states: every own enumerable key
 *  present on either must match (numbers within eps, others ===). */
export function statesClose(a: unknown, b: unknown, eps: number): boolean {
  if (a === b) return true;
  if (typeof a !== 'object' || typeof b !== 'object' || !a || !b) return false;
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  const ra = a as Record<string, unknown>;
  const rb = b as Record<string, unknown>;
  for (const k of keys) {
    const va = ra[k];
    const vb = rb[k];
    if (typeof va === 'number' && typeof vb === 'number') {
      if (Math.abs(va - vb) > eps) return false;
    } else if (va !== vb) {
      return false;
    }
  }
  return true;
}
