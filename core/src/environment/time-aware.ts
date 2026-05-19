// Composable time-extension. Wraps any static-world Environment to (a) treat
// time as an extra dimension in the per-level dominance key and the exact
// hash, and (b) prune any successor that would collide with a predicted
// moving obstacle at its arrival time. This — time participating in the
// multi-resolution dominance — is kinocat's novel contribution over the
// IGHA* paper. The static env stays independently unit-testable; the
// time-aware behaviour composes on top.

import type { Environment, EdgeRef, Node } from './types';
import type { MovingObstacle } from '../predict/types';

export interface TimeAwareOptions {
  obstacles?: MovingObstacle[];
  /** Circumscribed agent radius added to each obstacle radius. */
  agentRadius?: number;
  /** Fine time bucket (seconds) for the exact hash. */
  timeQuantum?: number;
  /** Per-level time-bucket divisors (coarse → fine); length = base.levels.
   *  Defaults to coupled halving (2^(levels-1-L)). */
  levelTimeDivisors?: number[];
}

type HasXZT = { x: number; z: number; t: number };

export class TimeAwareEnvironment<State extends HasXZT>
  implements Environment<State>
{
  readonly levels: number;
  private readonly obstacles: MovingObstacle[];
  private readonly agentRadius: number;
  private readonly timeQuantum: number;
  private readonly divisors: number[];

  constructor(
    private readonly base: Environment<State>,
    opts: TimeAwareOptions = {},
  ) {
    this.levels = base.levels;
    this.obstacles = opts.obstacles ?? [];
    this.agentRadius = opts.agentRadius ?? 0;
    this.timeQuantum = opts.timeQuantum ?? 0.2;
    this.divisors =
      opts.levelTimeDivisors ??
      Array.from({ length: this.levels }, (_, L) => 2 ** (this.levels - 1 - L));
  }

  private augment(node: Node<State>): Node<State> {
    const tb = Math.round(node.state.t / this.timeQuantum);
    node.index = node.index.map((k, L) => {
      const d = this.divisors[L] ?? 1;
      return `${k}@${Math.floor(tb / d)}`;
    });
    node.hash = `${node.hash}@t${tb}`;
    return node;
  }

  /** True if `state` overlaps any predicted obstacle at its own time. */
  private collides(state: State): boolean {
    for (const obs of this.obstacles) {
      const p = obs.predict(state.t);
      if (!p) continue;
      const rr = obs.radius + this.agentRadius;
      const dx = state.x - p.x;
      const dz = state.z - p.z;
      if (dx * dx + dz * dz <= rr * rr) return true;
    }
    return false;
  }

  createNode(
    state: State,
    parent: Node<State> | null,
    edge: EdgeRef | null,
  ): Node<State> {
    return this.augment(this.base.createNode(state, parent, edge));
  }

  succ(node: Node<State>, goal: Node<State>): Node<State>[] {
    const out: Node<State>[] = [];
    for (const c of this.base.succ(node, goal)) {
      if (this.collides(c.state)) continue;
      out.push(this.augment(c));
    }
    return out;
  }

  heuristic(from: State, to: State): number {
    return this.base.heuristic(from, to);
  }

  checkValidity(start: State, goal: State): [boolean, boolean] {
    const [s, g] = this.base.checkValidity(start, goal);
    return [s && !this.collides(start), g];
  }

  reachedGoalRegion(node: Node<State>, goal: Node<State>): boolean {
    return this.base.reachedGoalRegion(node, goal);
  }
}
