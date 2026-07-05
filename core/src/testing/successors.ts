// Successor and node-construction contract checks — the invariants the
// planner's main loop assumes of every Environment implementation:
// positive edge costs (termination), monotone time (the kinodynamic
// contract; checked when the state carries a `t` field), g/h/f bookkeeping,
// dominance-key arity (`index.length === levels`), and hash stability
// (identical states must hash identically or optimal dedup is unsound).

import { rng } from './rng';
import {
  tol,
  type CheckOptions,
  type ConformanceFailure,
  type DomainHarness,
} from './types';

export function checkSuccessorInvariants<State>(
  h: DomainHarness<State>,
  opts: CheckOptions = {},
): ConformanceFailure[] {
  const failures: ConformanceFailure[] = [];
  const env = h.makeEnv();
  const eps = tol(h);
  const rand = rng(opts.seed ?? 0xc0ffee);
  const samples = opts.samples ?? 200;
  const goals = h.scenarios.map((s) => s.goal);
  if (goals.length === 0) {
    return [
      {
        check: 'successor-invariants',
        message: 'harness has no scenarios — need at least one goal state',
      },
    ];
  }

  let tested = 0;
  for (let i = 0; i < samples; i++) {
    const s = h.sampleState(rand);
    if (!env.checkValidity(s, s)[0]) continue;
    const goalNode = env.createNode(goals[i % goals.length]!, null, null);
    const node = env.createNode(s, null, null);

    if (node.index.length !== env.levels) {
      failures.push({
        check: 'successor-invariants',
        message: `createNode index arity ${node.index.length} !== env.levels ${env.levels}`,
        sample: s,
      });
      break; // structural — every node will fail identically
    }

    for (const c of env.succ(node, goalNode, env.levels - 1)) {
      if (!c.edge) {
        failures.push({
          check: 'successor-invariants',
          message: 'successor has no edge (EdgeRef required on generated nodes)',
          sample: c.state,
        });
        continue;
      }
      if (!(c.edge.cost > 0)) {
        failures.push({
          check: 'successor-invariants',
          message: `edge '${c.edge.kind}' has non-positive cost ${c.edge.cost} — breaks search termination`,
          sample: { from: s, to: c.state, edge: c.edge },
        });
      }
      if (Math.abs(c.g - (node.g + c.edge.cost)) > eps) {
        failures.push({
          check: 'successor-invariants',
          message: `successor g=${c.g} !== parent.g + edge.cost = ${node.g + c.edge.cost}`,
          sample: { from: s, to: c.state, edge: c.edge },
        });
      }
      if (Math.abs(c.f - (c.g + c.h)) > eps) {
        failures.push({
          check: 'successor-invariants',
          message: `successor f=${c.f} !== g + h = ${c.g + c.h}`,
          sample: c.state,
        });
      }
      if (c.index.length !== env.levels) {
        failures.push({
          check: 'successor-invariants',
          message: `successor index arity ${c.index.length} !== env.levels ${env.levels}`,
          sample: c.state,
        });
      }
      // Monotone time — the kinodynamic contract. Only checked when the
      // domain's state carries a numeric `t` (every kinocat state does).
      const pt = (s as { t?: unknown }).t;
      const ct = (c.state as { t?: unknown }).t;
      if (typeof pt === 'number' && typeof ct === 'number' && !(ct > pt)) {
        failures.push({
          check: 'successor-invariants',
          message: `time did not advance along '${c.edge.kind}': parent t=${pt}, successor t=${ct}`,
          sample: { from: s, to: c.state },
        });
      }
    }
    tested++;
  }

  if (tested === 0) {
    failures.push({
      check: 'successor-invariants',
      message: `sampler produced 0 valid states out of ${samples} — fix sampleState`,
    });
  }
  return failures;
}

export function checkNodeStability<State>(
  h: DomainHarness<State>,
  opts: CheckOptions & {
    /** Max tolerated fraction of sampled states sharing a hash (default
     *  0.05). Random samples over a continuous space should rarely land in
     *  the same exact-state class; a high rate means the hash quantizes so
     *  coarsely that distinct states dedup against each other. */
    maxHashCollisionRate?: number;
  } = {},
): ConformanceFailure[] {
  const failures: ConformanceFailure[] = [];
  const env = h.makeEnv();
  const rand = rng(opts.seed ?? 0xc0ffee);
  const samples = opts.samples ?? 200;
  const maxRate = opts.maxHashCollisionRate ?? 0.05;

  const hashes = new Set<string>();
  let tested = 0;
  for (let i = 0; i < samples; i++) {
    const s = h.sampleState(rand);
    if (!env.checkValidity(s, s)[0]) continue;
    const a = env.createNode(s, null, null);
    // States are JSON-serializable by contract (PlanResult.path is), so a
    // JSON round-trip yields a structurally-equal clone.
    const b = env.createNode(JSON.parse(JSON.stringify(s)) as State, null, null);
    if (a.hash !== b.hash) {
      failures.push({
        check: 'node-stability',
        message: `hash not deterministic: same state hashed '${a.hash}' then '${b.hash}'`,
        sample: s,
      });
    }
    if (a.index.length !== b.index.length || a.index.some((v, L) => v !== b.index[L])) {
      failures.push({
        check: 'node-stability',
        message: 'index not deterministic for structurally-equal states',
        sample: s,
      });
    }
    hashes.add(a.hash);
    tested++;
  }

  if (tested === 0) {
    failures.push({
      check: 'node-stability',
      message: `sampler produced 0 valid states out of ${samples} — fix sampleState`,
    });
  } else {
    const collisions = tested - hashes.size;
    if (collisions / tested > maxRate) {
      failures.push({
        check: 'node-stability',
        message:
          `${collisions}/${tested} sampled states share an exact hash ` +
          `(> ${maxRate * 100}%) — hash quantization may be coarse enough to ` +
          `dedup genuinely distinct states (tune maxHashCollisionRate if the ` +
          `sampler is intentionally clustered)`,
      });
    }
  }
  return failures;
}
