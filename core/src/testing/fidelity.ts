// Successor fidelity: succ()'s successors must be what the domain's forward
// sim actually produces from the parent's ACTUAL state. Environments that
// apply cached primitives from canonical bucket starts "teleport" state dims
// to the bucket (speed for the car and momentum humanoid); environments that
// roll the sim live (aircraft) are exact. This check re-simulates every
// re-simulatable edge and reports the worst per-field deviation — catching
// transform bugs outright and pinning deliberate bucket error at the
// tolerance the harness declares.

import { rng } from './rng';
import {
  type CheckOptions,
  type ConformanceFailure,
  type DomainHarness,
} from './types';

export function checkSuccessorFidelity<State>(
  h: DomainHarness<State>,
  opts: CheckOptions = {},
): ConformanceFailure[] {
  const failures: ConformanceFailure[] = [];
  const hooks = h.fidelity;
  if (!hooks) {
    return [
      {
        check: 'successor-fidelity',
        message: 'harness has no fidelity hooks — supply DomainHarness.fidelity',
      },
    ];
  }
  const env = h.makeEnv();
  const rand = rng(opts.seed ?? 0xc0ffee);
  const samples = opts.samples ?? 200;
  const goals = h.scenarios.map((s) => s.goal);
  if (goals.length === 0) {
    return [
      {
        check: 'successor-fidelity',
        message: 'harness has no scenarios — need at least one goal state',
      },
    ];
  }

  let tested = 0;
  let resimulated = 0;
  for (let i = 0; i < samples; i++) {
    const s = h.sampleState(rand);
    if (!env.checkValidity(s, s)[0]) continue;
    const goalNode = env.createNode(goals[i % goals.length]!, null, null);
    const node = env.createNode(s, null, null);
    for (const c of env.succ(node, goalNode, env.levels - 1)) {
      if (!c.edge) continue;
      const resim = hooks.resimulate(s, c.edge);
      if (resim === null) continue;
      resimulated++;
      const ca = c.state as Record<string, unknown>;
      const rb = resim as Record<string, unknown>;
      for (const key of new Set([...Object.keys(ca), ...Object.keys(rb)])) {
        const va = ca[key];
        const vb = rb[key];
        if (typeof va !== 'number' || typeof vb !== 'number') continue;
        let dev = Math.abs(va - vb);
        if (hooks.angularFields?.includes(key)) {
          // Compare on the circle: ±π represent the same angle.
          let d = (va - vb) % (2 * Math.PI);
          if (d > Math.PI) d -= 2 * Math.PI;
          if (d < -Math.PI) d += 2 * Math.PI;
          dev = Math.abs(d);
        }
        if (dev > hooks.tolerance) {
          failures.push({
            check: 'successor-fidelity',
            message:
              `successor '${c.edge.kind}' deviates from the re-simulated ` +
              `edge on '${key}' by ${dev} (tolerance ${hooks.tolerance}) — ` +
              `succ() is not applying the forward sim faithfully`,
            sample: { parent: s, cached: c.state, resimulated: resim, edge: c.edge },
          });
        }
      }
    }
    tested++;
  }

  if (tested === 0) {
    failures.push({
      check: 'successor-fidelity',
      message: `sampler produced 0 valid states out of ${samples} — fix sampleState`,
    });
  } else if (resimulated === 0) {
    failures.push({
      check: 'successor-fidelity',
      message:
        'resimulate() returned null for every edge — the hook must handle ' +
        'the primitive edge kind, not just skip everything',
    });
  }
  return failures;
}
