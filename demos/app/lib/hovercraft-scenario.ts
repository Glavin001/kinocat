// Hovercraft lagoon scenario — headless-testable builder for the /hovercraft
// 3D demo. An ice lagoon with island obstacles and two patrolling ice floes
// (deterministic predictors); the hovercraft plans in space-time through
// TimeAwareEnvironment, so a single plan is valid against the floes' whole
// future — replans happen only when the user retargets, and they start from
// the CURRENT velocity, so momentum visibly carries into the new plan.

import { plan } from 'kinocat/planner';
import type { PlanResult } from 'kinocat/planner';
import { InMemoryNavWorld, TimeAwareEnvironment } from 'kinocat/environment';
import type { MovingObstacle } from 'kinocat/predict';
import {
  HOVER_AGENT,
  HovercraftEnvironment,
  type HovercraftState,
} from './hovercraft-domain';

export const HOVER_BOUNDS = { x0: 0, z0: -22, x1: 64, z1: 22 };

export interface HoverIsland {
  x: number;
  z: number;
  hx: number;
  hz: number;
}

export const HOVER_DEFAULT_ISLANDS: HoverIsland[] = [
  { x: 18, z: -6, hx: 4, hz: 5 },
  { x: 32, z: 8, hx: 5, hz: 4 },
  { x: 46, z: -4, hx: 4, hz: 6 },
];

function box(x: number, z: number, hx: number, hz: number): [number, number][] {
  return [
    [x - hx, z - hz],
    [x + hx, z - hz],
    [x + hx, z + hz],
    [x - hx, z + hz],
  ];
}

export function hovercraftWorldFrom(islands: HoverIsland[]): InMemoryNavWorld {
  const b = HOVER_BOUNDS;
  return new InMemoryNavWorld(
    [
      {
        id: 1,
        y: 0,
        ring: [
          [b.x0, b.z0],
          [b.x1, b.z0],
          [b.x1, b.z1],
          [b.x0, b.z1],
        ],
      },
    ],
    islands.map((o) => box(o.x, o.z, o.hx, o.hz)),
  );
}

/** Two ice floes on deterministic patrol loops (any Predict works — these
 *  are sinusoids, so they are valid for the plan's whole horizon). */
export function hovercraftFloes(): MovingObstacle[] {
  return [
    {
      radius: 2.2,
      predict: (t) => ({ x: 25, z: 10 * Math.sin(t * 0.45) }),
    },
    {
      radius: 2.2,
      predict: (t) => ({ x: 39, z: -10 * Math.sin(t * 0.35 + 1.1) }),
    },
  ];
}

export const HOVER_MAX_EXPANSIONS = 250_000;

export function planHovercraftLeg(
  world: InMemoryNavWorld,
  floes: MovingObstacle[],
  start: HovercraftState,
  goal: HovercraftState,
): PlanResult<HovercraftState> {
  const env = new TimeAwareEnvironment(new HovercraftEnvironment(world, HOVER_AGENT), {
    obstacles: floes,
    agentRadius: HOVER_AGENT.radius,
    broadphase: {},
  });
  return plan(
    { start, goal, environment: env, options: { maxExpansions: HOVER_MAX_EXPANSIONS } },
    Infinity,
  );
}

export interface HovercraftScene {
  bounds: typeof HOVER_BOUNDS;
  islands: HoverIsland[];
  floes: MovingObstacle[];
  start: HovercraftState;
  goal: HovercraftState;
  result: PlanResult<HovercraftState>;
}

export function buildHovercraft(): HovercraftScene {
  const islands = HOVER_DEFAULT_ISLANDS;
  const world = hovercraftWorldFrom(islands);
  const floes = hovercraftFloes();
  const start: HovercraftState = { x: 4, z: 0, heading: 0, vx: 0, vz: 0, t: 0 };
  const goal: HovercraftState = { x: 60, z: 0, heading: 0, vx: 0, vz: 0, t: 0 };
  const result = planHovercraftLeg(world, floes, start, goal);
  return { bounds: HOVER_BOUNDS, islands, floes, start, goal, result };
}
