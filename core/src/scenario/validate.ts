// Load-time validation & linting. Scenarios are authored by hand AND by
// generators, so a validation pass catches errors at definition time instead of
// as a silent "no path found". `validate` returns STRUCTURED diagnostics (which
// node, which check) — never a boolean — so tooling can surface precise errors.
// None of these guarantees feasibility; they remove the most common silent
// failures.

import type { Scenario, Goal, Region, Invariant } from './types';
import { normalize } from './normalize';
import { compile, nextGuardPose } from './automaton';

export type Severity = 'error' | 'warning';

export interface Diagnostic {
  check: string;
  severity: Severity;
  /** AST path, e.g. "goal.seq[1]" or "invariants[0]". */
  path: string;
  message: string;
}

export interface ValidateOptions {
  /** Planner discretization (m). Acceptance margins below this can never land a
   *  node inside the region. */
  posCell?: number;
  /** Reference speed (m/s) for the coarse time-feasibility check. */
  refSpeed?: number;
}

/** Walk every `reach` region in the goal, yielding [region, path]. */
function* reachRegions(goal: Goal, path = 'goal'): Generator<[Region, string, Goal]> {
  switch (goal.kind) {
    case 'reach':
      yield [goal.region, path, goal];
      return;
    case 'repeat':
      yield* reachRegions(goal.goal, `${path}.repeat`);
      return;
    case 'seq':
    case 'all':
    case 'any':
      for (let i = 0; i < goal.goals.length; i++) {
        yield* reachRegions(goal.goals[i]!, `${path}.${goal.kind}[${i}]`);
      }
      return;
  }
}

/** Parse the agent id out of a dynamic region key like `within:cop1,3`. */
function agentIdOf(region: Region): string | null {
  if (!region.dynamic) return null;
  const m = /^[a-zA-Z-]+:([^,]+),/.exec(region.key);
  return m ? m[1]! : null;
}

/** Parse `at` position margins from the key (`...,dx,dz,...` box or `...,r<radius>,...`
 *  disk) as an effective (dx, dz) for the margins-vs-resolution check. */
function atMargins(region: Region): { dx: number; dz: number } | null {
  if (region.kind !== 'at') return null;
  const parts = region.key.slice('at:'.length).split(',');
  const p3 = parts[3] ?? '';
  if (p3.startsWith('r')) {
    const r = Number(p3.slice(1));
    return Number.isFinite(r) ? { dx: r, dz: r } : null;
  }
  const dx = Number(p3);
  const dz = Number(parts[4]);
  if (Number.isFinite(dx) && Number.isFinite(dz)) return { dx, dz };
  return null;
}

/** Parse a cond-speed bound from the key `cond-speed:min,max`. */
function condSpeedBound(region: Region): { min?: number; max?: number } | null {
  if (region.kind !== 'cond-speed') return null;
  const parts = region.key.slice('cond-speed:'.length).split(',');
  const min = parts[0] === '' ? undefined : Number(parts[0]);
  const max = parts[1] === '' ? undefined : Number(parts[1]);
  return { min, max };
}

export function validate(scenario: Scenario, opts: ValidateOptions = {}): Diagnostic[] {
  const diags: Diagnostic[] = [];
  const posCell = opts.posCell ?? 0;
  const refSpeed = opts.refSpeed ?? 10;
  const agentIds = new Set((scenario.agents ?? []).map((a) => a.id));

  // --- Unsatisfiable / empty structure -------------------------------------
  const normalized = normalize(scenario.goal);
  if (normalized.kind === 'any' && normalized.goals.length === 0) {
    diags.push({
      check: 'unsatisfiable-structure',
      severity: 'error',
      path: 'goal',
      message: 'goal normalizes to ⊥ (any() with no alternatives) — unsatisfiable',
    });
  }

  // --- Per reach-region checks ---------------------------------------------
  for (const [region, path, g] of reachRegions(scenario.goal)) {
    // Margins vs resolution.
    if (posCell > 0) {
      const m = atMargins(region);
      if (m && (m.dx < posCell || m.dz < posCell)) {
        diags.push({
          check: 'margins-vs-resolution',
          severity: 'warning',
          path,
          message: `acceptance margins (dx=${m.dx}, dz=${m.dz}) are below the planner discretization (${posCell}); the search may never land a node inside the region`,
        });
      }
    }
    // Dynamic-region agent presence.
    const aid = agentIdOf(region);
    if (aid !== null && !agentIds.has(aid.replace(/#.*$/, ''))) {
      diags.push({
        check: 'dynamic-region-agent',
        severity: 'error',
        path,
        message: `dynamic region '${region.kind}' references agent '${aid}' which is not in scenario.agents`,
      });
    }
    // Coarse time feasibility for deadlines.
    if (g.kind === 'reach' && g.accept?.by !== undefined) {
      const dist = region.costToGo(scenario.start);
      const earliest = scenario.start.t + dist / Math.max(1e-6, refSpeed);
      if (earliest > g.accept.by) {
        diags.push({
          check: 'time-feasibility',
          severity: 'warning',
          path,
          message: `deadline by=${g.accept.by}s is unreachable even at refSpeed=${refSpeed} m/s (earliest ≈ ${earliest.toFixed(1)}s)`,
        });
      }
    }
  }

  // --- Contradictory maintain(speed) bounds --------------------------------
  let combinedMin = -Infinity;
  let combinedMax = Infinity;
  (scenario.invariants ?? []).forEach((inv: Invariant, i) => {
    if (inv.kind === 'maintain') {
      const b = condSpeedBound(inv.region);
      if (b) {
        if (b.min !== undefined) combinedMin = Math.max(combinedMin, b.min);
        if (b.max !== undefined) combinedMax = Math.min(combinedMax, b.max);
      }
      // Dynamic maintain region agent presence.
      const aid = agentIdOf(inv.region);
      if (aid !== null && !agentIds.has(aid.replace(/#.*$/, ''))) {
        diags.push({
          check: 'dynamic-region-agent',
          severity: 'error',
          path: `invariants[${i}]`,
          message: `maintain references agent '${aid}' not in scenario.agents`,
        });
      }
    }
    if (inv.kind === 'avoid') {
      const aid = agentIdOf(inv.region);
      if (aid !== null && !agentIds.has(aid.replace(/#.*$/, ''))) {
        diags.push({
          check: 'dynamic-region-agent',
          severity: 'error',
          path: `invariants[${i}]`,
          message: `avoid references agent '${aid}' not in scenario.agents`,
        });
      }
    }
  });
  if (combinedMin > combinedMax) {
    diags.push({
      check: 'contradictory-invariants',
      severity: 'error',
      path: 'invariants',
      message: `maintain(speed) bounds have empty intersection (min ${combinedMin} > max ${combinedMax})`,
    });
  }

  // --- Compile sanity (dead start) -----------------------------------------
  try {
    const automaton = compile(scenario.goal);
    if (!automaton.progress && automaton.accepting.length === 0) {
      diags.push({
        check: 'unsatisfiable-structure',
        severity: 'error',
        path: 'goal',
        message: 'compiled automaton has no accepting state and is not a progress (repeat) objective',
      });
    }
    if (nextGuardPose(automaton, automaton.start) === null && !automaton.accepting.includes(automaton.start)) {
      diags.push({
        check: 'dead-start',
        severity: 'error',
        path: 'goal',
        message: 'compiled automaton start has no outgoing guard and is not accepting',
      });
    }
  } catch (e) {
    diags.push({
      check: 'compile-error',
      severity: 'error',
      path: 'goal',
      message: (e as Error).message,
    });
  }

  return diags;
}
