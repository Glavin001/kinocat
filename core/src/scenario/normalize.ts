// Canonicalization. The AST is only *canonical* if equivalent specs reduce to
// identical trees. `normalize(goal)` applies a fixed set of rewrites to a fixed
// point; `hashGoal` / `structuralEqual` are then defined on the normalized tree
// so scenarios can be hashed, deduplicated, diffed, and their compiled automata
// cached by AST hash.
//
// Two rewrites are DELIBERATELY NOT applied: never reorder `seq` children
// (order is semantic) and never dedup `seq` children (a course may legitimately
// pass the same gate twice).

import type { Goal, Acceptance } from './types';

/** Trivially-satisfied goal — the normal form of `all()`. Encoded as an empty
 *  `seq` (an empty conjunction is true). */
export const TOP: Goal = { kind: 'seq', goals: [] };
/** Unsatisfiable goal — the normal form of `any()`. Encoded as an empty `any`.
 *  The linter (`validate`) flags this. */
export const BOTTOM: Goal = { kind: 'any', goals: [] };

function acceptKey(a?: Acceptance): string {
  if (!a) return '';
  const sp = a.speed ? `s${a.speed.min ?? ''}:${a.speed.max ?? ''}` : '';
  const hd = a.heading ? `h${a.heading.min ?? ''}:${a.heading.max ?? ''}` : '';
  const wn = a.window ? `w${a.window[0]}:${a.window[1]}` : '';
  const by = a.by !== undefined ? `b${a.by}` : '';
  return `${sp}|${hd}|${wn}|${by}`;
}

/** A stable structural signature of a (normalized) goal. */
export function hashGoal(goal: Goal): string {
  switch (goal.kind) {
    case 'reach':
      return `R(${goal.region.key};${acceptKey(goal.accept)})`;
    case 'repeat':
      return `*(${hashGoal(goal.goal)})`;
    case 'seq':
      return `>(${goal.goals.map(hashGoal).join(',')})`;
    case 'all':
      return `&(${goal.goals.map(hashGoal).join(',')})`;
    case 'any':
      return `|(${goal.goals.map(hashGoal).join(',')})`;
  }
}

export function structuralEqual(a: Goal, b: Goal): boolean {
  return hashGoal(a) === hashGoal(b);
}

function dedupByHash(goals: Goal[]): Goal[] {
  const seen = new Set<string>();
  const out: Goal[] = [];
  for (const g of goals) {
    const h = hashGoal(g);
    if (!seen.has(h)) {
      seen.add(h);
      out.push(g);
    }
  }
  return out;
}

/** One bottom-up normalization pass. `normalize` iterates this to a fixed
 *  point (the rewrites are confluent + terminating, so two passes always
 *  suffice, but we loop on the hash to be safe). */
function pass(goal: Goal): Goal {
  switch (goal.kind) {
    case 'reach':
      return goal;

    case 'repeat': {
      let inner = pass(goal.goal);
      // Idempotent repeat: repeat(repeat(g)) -> repeat(g).
      while (inner.kind === 'repeat') inner = inner.goal;
      return { kind: 'repeat', goal: inner };
    }

    case 'seq': {
      const flat: Goal[] = [];
      for (const child of goal.goals.map(pass)) {
        // Flatten nested seq; drop TOP (empty seq) identities.
        if (child.kind === 'seq') flat.push(...child.goals);
        else flat.push(child);
      }
      // NB: no reorder, no dedup for seq.
      if (flat.length === 1) return flat[0]!;
      return { kind: 'seq', goals: flat };
    }

    case 'all': {
      const flat: Goal[] = [];
      for (const child of goal.goals.map(pass)) {
        if (child.kind === 'all') flat.push(...child.goals);
        else flat.push(child);
      }
      // Drop TOP children (trivially satisfied conjuncts).
      const nonTop = flat.filter((g) => !(g.kind === 'seq' && g.goals.length === 0));
      const deduped = dedupByHash(nonTop);
      if (deduped.length === 0) return TOP; // all() -> ⊤
      if (deduped.length === 1) return deduped[0]!;
      return { kind: 'all', goals: deduped };
    }

    case 'any': {
      const flat: Goal[] = [];
      for (const child of goal.goals.map(pass)) {
        if (child.kind === 'any') flat.push(...child.goals);
        else flat.push(child);
      }
      const deduped = dedupByHash(flat);
      if (deduped.length === 0) return BOTTOM; // any() -> ⊥
      if (deduped.length === 1) return deduped[0]!;
      return { kind: 'any', goals: deduped };
    }
  }
}

export function normalize(goal: Goal): Goal {
  let current = goal;
  let prev = '';
  for (let i = 0; i < 8; i++) {
    current = pass(current);
    const h = hashGoal(current);
    if (h === prev) break;
    prev = h;
  }
  return current;
}
