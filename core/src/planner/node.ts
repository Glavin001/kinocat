import type { EdgeRef, Node } from '../environment/types';

/** Build a bare Node with planner fields defaulted. Environments call this
 *  from `createNode`, supplying the per-level `index` and exact `hash`. */
export function makeNode<State>(
  state: State,
  parent: Node<State> | null,
  edge: EdgeRef | null,
  index: number[],
  hash: string,
): Node<State> {
  return {
    state,
    g: 0,
    h: 0,
    f: 0,
    parent,
    edge,
    index,
    hash,
    level: 0,
    active: false,
  };
}

/** Walk parent links to produce the state sequence start → node. */
export function reconstructStates<State>(node: Node<State>): State[] {
  const out: State[] = [];
  let n: Node<State> | null = node;
  while (n) {
    out.push(n.state);
    n = n.parent;
  }
  out.reverse();
  return out;
}

/** Walk parent links to produce the node sequence start → node. */
export function reconstructNodes<State>(node: Node<State>): Node<State>[] {
  const out: Node<State>[] = [];
  let n: Node<State> | null = node;
  while (n) {
    out.push(n);
    n = n.parent;
  }
  out.reverse();
  return out;
}
