// R² grid environment — the IGHA* port-correctness oracle (mirrors the
// reference's `simple_example`). An 8-connected lattice with an occupancy
// predicate and a consistent Euclidean heuristic; the finest resolution level
// keys on exact cells, so the planner must return the optimal lattice cost.
// Not part of the shipped algorithm; excluded from the coverage gate.

import type { Environment, EdgeRef, Node } from './types';
import { makeNode } from '../planner/node';
import { pack2 } from '../planner/resolution';

export interface R2State {
  x: number;
  y: number;
}

export interface R2Bounds {
  minCx: number;
  maxCx: number;
  minCy: number;
  maxCy: number;
}

export interface R2Options {
  step: number;
  /** Cell-coordinate occupancy predicate; true ⇒ blocked. */
  blocked: (cx: number, cy: number) => boolean;
  bounds: R2Bounds;
  /** Per-level cell divisors, coarse → fine; the last MUST be 1. */
  levelDivisors?: number[];
  /** Goal reached if within this many cells (Chebyshev); 0 = exact cell. */
  goalCellRadius?: number;
}

const SQRT2 = Math.SQRT2;
const DIRS: ReadonlyArray<readonly [number, number]> = [
  [1, 0],
  [-1, 0],
  [0, 1],
  [0, -1],
  [1, 1],
  [1, -1],
  [-1, 1],
  [-1, -1],
];

export class R2Environment implements Environment<R2State> {
  readonly levels: number;
  private readonly divisors: number[];
  private readonly step: number;
  private readonly blockedFn: (cx: number, cy: number) => boolean;
  private readonly bounds: R2Bounds;
  private readonly goalCellRadius: number;

  constructor(opts: R2Options) {
    this.step = opts.step;
    this.blockedFn = opts.blocked;
    this.bounds = opts.bounds;
    this.divisors = opts.levelDivisors ?? [4, 2, 1];
    this.goalCellRadius = opts.goalCellRadius ?? 0;
    this.levels = this.divisors.length;
  }

  cellOf(s: R2State): [number, number] {
    return [Math.round(s.x / this.step), Math.round(s.y / this.step)];
  }

  private inBounds(cx: number, cy: number): boolean {
    const b = this.bounds;
    return cx >= b.minCx && cx <= b.maxCx && cy >= b.minCy && cy <= b.maxCy;
  }

  private free(cx: number, cy: number): boolean {
    return this.inBounds(cx, cy) && !this.blockedFn(cx, cy);
  }

  createNode(
    state: R2State,
    parent: Node<R2State> | null,
    edge: EdgeRef | null,
  ): Node<R2State> {
    const [cx, cy] = this.cellOf(state);
    const index: string[] = [];
    for (const d of this.divisors) {
      index.push(pack2(Math.floor(cx / d), Math.floor(cy / d)));
    }
    return makeNode(state, parent, edge, index, `${cx},${cy}`);
  }

  succ(node: Node<R2State>, goal: Node<R2State>): Node<R2State>[] {
    const [cx, cy] = this.cellOf(node.state);
    const [gx, gy] = this.cellOf(goal.state);
    const out: Node<R2State>[] = [];
    for (const [dx, dy] of DIRS) {
      const ncx = cx + dx;
      const ncy = cy + dy;
      if (!this.free(ncx, ncy)) continue;
      const cost = (dx !== 0 && dy !== 0 ? SQRT2 : 1) * this.step;
      const ns: R2State = { x: ncx * this.step, y: ncy * this.step };
      const edge: EdgeRef = { cost, kind: 'move' };
      const n = this.createNode(ns, node, edge);
      n.g = node.g + cost;
      n.h = Math.hypot((ncx - gx) * this.step, (ncy - gy) * this.step);
      n.f = n.g + n.h;
      out.push(n);
    }
    return out;
  }

  heuristic(from: R2State, to: R2State): number {
    return Math.hypot(from.x - to.x, from.y - to.y);
  }

  checkValidity(start: R2State, goal: R2State): [boolean, boolean] {
    const [sx, sy] = this.cellOf(start);
    const [gx, gy] = this.cellOf(goal);
    return [this.free(sx, sy), this.free(gx, gy)];
  }

  reachedGoalRegion(node: Node<R2State>, goal: Node<R2State>): boolean {
    const [cx, cy] = this.cellOf(node.state);
    const [gx, gy] = this.cellOf(goal.state);
    return (
      Math.abs(cx - gx) <= this.goalCellRadius &&
      Math.abs(cy - gy) <= this.goalCellRadius
    );
  }
}
