// Crowd-run scenario: the momentum humanoid (inertial person — sprint,
// strafe cap, launch/brake limits, speed-degraded turning) sprints across a
// plaza while pedestrians cut across its line at exactly the times it would
// meet them. Planned through TimeAwareEnvironment, so avoidance happens in
// space-time: the runner adjusts SPEED and line, not just geometry. Built
// entirely from public kinocat seams — this scenario is the demo-side proof
// of the fourth agent domain.

import { plan } from 'kinocat/planner';
import type { PlanResult } from 'kinocat/planner';
import {
  InMemoryNavWorld,
  MomentumHumanoidEnvironment,
  TimeAwareEnvironment,
} from 'kinocat/environment';
import { defaultMomentumHumanoidAgent } from 'kinocat/agent';
import type { MomentumHumanoidAgent, MomentumHumanoidState } from 'kinocat/agent';
import { linearObstacle } from 'kinocat/predict';
import type { MovingObstacle } from 'kinocat/predict';

export interface CrowdScene {
  bounds: { x0: number; z0: number; x1: number; z1: number };
  floor: [number, number][];
  pedestrians: MovingObstacle[];
  agent: MomentumHumanoidAgent;
  start: MomentumHumanoidState;
  goal: MomentumHumanoidState;
  result: PlanResult<MomentumHumanoidState>;
  goalRadius: number;
}

export const CROWD_MAX_EXPANSIONS = 400_000;

export function buildCrowd(): CrowdScene {
  const bounds = { x0: 0, z0: 0, x1: 28, z1: 14 };
  const floor: [number, number][] = [
    [0, 0],
    [28, 0],
    [28, 14],
    [0, 14],
  ];
  const world = new InMemoryNavWorld([{ id: 1, y: 0, ring: floor }]);
  const agent = defaultMomentumHumanoidAgent();

  // Pedestrians timed to cross z = 7 roughly when a straight sprint would
  // meet them — pure geometry can't solve this; space-time planning can.
  const pedestrians: MovingObstacle[] = [
    linearObstacle(8, 9.5, 0, -1.1, 0.45),
    linearObstacle(14, 3.5, 0, 1.0, 0.45),
    linearObstacle(21, 11.5, 0, -0.9, 0.45),
  ];

  const goalRadius = 0.6;
  const env = new TimeAwareEnvironment(
    new MomentumHumanoidEnvironment(world, agent, { goalRadius }),
    {
      obstacles: pedestrians,
      agentRadius: agent.radius,
      broadphase: {},
    },
  );

  const start: MomentumHumanoidState = { x: 2, z: 7, heading: 0, vx: 0, vz: 0, t: 0 };
  const goal: MomentumHumanoidState = { x: 26, z: 7, heading: 0, vx: 0, vz: 0, t: 0 };
  const result = plan(
    {
      start,
      goal,
      environment: env,
      options: { maxExpansions: CROWD_MAX_EXPANSIONS },
    },
    Infinity,
  );

  return { bounds, floor, pedestrians, agent, start, goal, result, goalRadius };
}
