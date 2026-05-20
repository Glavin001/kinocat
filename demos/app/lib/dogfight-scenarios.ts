// Pure, headless-testable dogfight scenario. The interactive Dogfight demo
// imports the terrain sampler, obstacle list, AI tactical layer, and per-tick
// planning helper from here; the headless test (demos/test/scenarios.test.ts)
// imports the same to assert the AIs always produce a plan within budget for
// the scripted matchup. No React / three imports.

import { plan } from 'kinocat/planner';
import type { PlanResult } from 'kinocat/planner';
import {
  AircraftEnvironment,
  HeightfieldAirspace,
  TimeAwareEnvironment,
} from 'kinocat/environment';
import type {
  AABB,
  MovingZone,
  HeightfieldSampler,
  AircraftEnvOptions,
} from 'kinocat/environment';
import { PlanRegistry, asObstacle } from 'kinocat/predict';
import type { Predict } from 'kinocat/predict';
import { defaultAircraftAgent } from 'kinocat/agent';
import type { AircraftAgent, AircraftState } from 'kinocat/agent';

export const DOGFIGHT_PALETTE = {
  bg: '#0a0e16',
  fog: '#101a26',
  terrainLow: '#1f3a26',
  terrainMid: '#5a4a2a',
  terrainHigh: '#d8d4cf',
  wall: '#3a2330',
  blimp: '#cdb583',
  ring: '#55ff88',
  player: '#7fd6ff',
  enemy: [0xff5566, 0xffaa33, 0x9b6cff, 0x4dd0e1] as readonly number[],
  pathPlayer: '#7fd6ff',
  pathEnemy: '#ff7799',
};

/** Anytime budget per AI replan (one tick of the round-robin loop). */
export const DOGFIGHT_REPLAN_BUDGET_MS = 110;
/** Hard expansion cap so a stuck plan can never hang the frame. */
export const DOGFIGHT_MAX_EXPANSIONS = 60000;
/** Headless scenario test budget (one-shot, generous). */
export const DOGFIGHT_TEST_MAX_EXPANSIONS = 220000;

/** Same envelope as the /plane demo so existing primitive characterizations
 *  remain in scope; tweaked slightly so the AI feels nimble in the dogfight. */
export const DOGFIGHT_AGENT: AircraftAgent = defaultAircraftAgent({
  minTurnRadius: 14,
  minSpeed: 8,
  maxSpeed: 22,
  maxClimbAngle: Math.PI / 5,
  maxBank: Math.PI / 3,
  halfLength: 2,
  halfSpan: 1.8,
  halfHeight: 0.35,
});

export const DOGFIGHT_HALF: [number, number, number] = [
  DOGFIGHT_AGENT.halfLength,
  DOGFIGHT_AGENT.halfSpan,
  DOGFIGHT_AGENT.halfHeight,
];

export const DOGFIGHT_BOUNDS = {
  x0: -10,
  x1: 240,
  z0: -100,
  z1: 100,
  floor: 0,
  ceiling: 90,
};

// ---------------------------------------------------------------------------
// Terrain — analytic, deterministic, smooth. The same function feeds both
// the planner's HeightfieldAirspace and the renderer's displaced PlaneGeometry,
// so what the player sees IS what the planner respects.

/** Returns ground elevation Y at world (x, z). Bounded ~[0, 28]. */
export const dogfightTerrain: HeightfieldSampler = (x, z) => {
  // Long rolling base ridge along x = 60.
  const ridgeA = 12 * Math.exp(-((x - 60) * (x - 60)) / 700) *
    (0.7 + 0.3 * Math.cos(z * 0.04));
  // Twin peaks straddling the centre corridor at x = 130.
  const peakN = 22 * Math.exp(-(((x - 130) * (x - 130)) / 250 + ((z - 35) * (z - 35)) / 220));
  const peakS = 22 * Math.exp(-(((x - 130) * (x - 130)) / 250 + ((z + 35) * (z + 35)) / 220));
  // Far wall: a high mesa near the east edge to keep play bounded.
  const mesa = 18 * smoothStep(x, 200, 230) * (0.6 + 0.4 * Math.cos(z * 0.03));
  // Subtle dunes for texture (low amplitude — kept small so the planner's
  // 9-sample OBB clearance check stays well above terrain at normal cruise).
  const dunes = 1.2 * (Math.sin(x * 0.08) + Math.cos(z * 0.11));
  return Math.max(0, ridgeA + peakN + peakS + mesa + dunes);
};

function smoothStep(t: number, a: number, b: number): number {
  const u = Math.max(0, Math.min(1, (t - a) / (b - a)));
  return u * u * (3 - 2 * u);
}

// ---------------------------------------------------------------------------
// Static obstacles + moving zones in addition to the terrain.

/** Tall thin pylons + a horizontal "wall of pillars" the AI must thread. */
export function dogfightStaticObstacles(): AABB[] {
  const f = DOGFIGHT_BOUNDS.floor;
  const c = DOGFIGHT_BOUNDS.ceiling;
  const out: AABB[] = [];
  // Four pylons in a diamond around (95, 0).
  for (const [px, pz] of [
    [85, -14],
    [105, -14],
    [85, 14],
    [105, 14],
  ] as [number, number][]) {
    out.push({ min: [px - 1.2, f, pz - 1.2], max: [px + 1.2, c, pz + 1.2] });
  }
  // Vertical "wall of pillars" — five tall thin boxes the AI must weave
  // around, leaving alternating gaps at different x positions.
  for (let i = 0; i < 5; i++) {
    const px = 160 + i * 2;
    const pz = -40 + i * 20; // staggered
    out.push({ min: [px - 1.5, f, pz - 6], max: [px + 1.5, c, pz + 6] });
  }
  return out;
}

/** A blimp drifting in a sine along the corridor, and an oscillating barrier
 *  that sweeps the canyon entrance — both as spherical moving zones. */
export function dogfightMovingZones(): MovingZone[] {
  return [
    {
      // Blimp: cruises along x, gently bobbing in z and y.
      radius: 8,
      predict: (t) => ({
        x: 40 + 7 * Math.sin(t * 0.25),
        y: 50 + 3 * Math.sin(t * 0.6),
        z: 18 * Math.sin(t * 0.35),
      }),
    },
    {
      // Sweeping barrier between the twin peaks.
      radius: 9,
      predict: (t) => ({
        x: 130,
        y: 36,
        z: 30 * Math.sin(t * 0.55),
      }),
    },
  ];
}

/** Build the HeightfieldAirspace that all AIs share. Terrain + statics +
 *  zones; no per-AI plan-registry obstacles here (those wrap on top via
 *  TimeAwareEnvironment). */
export function dogfightAirspace(): HeightfieldAirspace {
  return new HeightfieldAirspace({
    floor: DOGFIGHT_BOUNDS.floor,
    ceiling: DOGFIGHT_BOUNDS.ceiling,
    sampler: dogfightTerrain,
    sampleMargin: 1.5, // keep at least ~1.5 m ground clearance
    boxes: dogfightStaticObstacles(),
    zones: dogfightMovingZones(),
  });
}

// ---------------------------------------------------------------------------
// Boost rings — visual / simulator features (no planner affordance edge):
// when the player or an AI flies through a ring, a +25% airspeed bonus
// applies for 2s. Used purely by the demo's runtime; the planner doesn't
// route through them deliberately, but the AI naturally passes them en-route.

export interface BoostRing {
  id: string;
  x: number;
  y: number;
  z: number;
  /** Ring normal — passage is detected along this axis. */
  axis: { x: number; y: number; z: number };
  radius: number;
}

export function dogfightBoostRings(): BoostRing[] {
  return [
    { id: 'R0', x: 20, y: 22, z: -10, axis: { x: 1, y: 0, z: 0 }, radius: 6 },
    { id: 'R1', x: 95, y: 38, z: 0, axis: { x: 1, y: 0, z: 0 }, radius: 6 },
    { id: 'R2', x: 130, y: 30, z: 0, axis: { x: 1, y: 0, z: 0 }, radius: 6 },
    { id: 'R3', x: 180, y: 28, z: 20, axis: { x: 1, y: 0, z: 0.4 }, radius: 6 },
  ];
}

// ---------------------------------------------------------------------------
// Tactical layer (GOAP-style). Outside kinocat's scope by design — kinocat
// finds the trajectory once the goal pose is supplied. This layer picks the
// goal mode from relative geometry so the same NPC plays "tail the player",
// "flank from the right", or "break off and re-engage" without scripting.

export type TacticalMode =
  | 'PURSUE'
  | 'INTERCEPT'
  | 'FLANK_LEFT'
  | 'FLANK_RIGHT'
  | 'EVADE'
  | 'REGROUP';

/** Pick a tactical mode for `npc` relative to the player. Pure function of
 *  observable geometry — no per-NPC state. */
export function selectTacticalMode(
  player: AircraftState,
  npc: AircraftState,
  npcIndex: number,
): TacticalMode {
  const dx = player.x - npc.x;
  const dz = player.z - npc.z;
  const dist = Math.hypot(dx, dz);
  // Bearing from NPC to player (signed angle off NPC's nose).
  const bearing = wrapPi(Math.atan2(dz, dx) - npc.heading);
  // Bearing from player to NPC (is the NPC in the player's rear? front?).
  const bearingFromPlayer = wrapPi(Math.atan2(-dz, -dx) - player.heading);
  // If the player is BEHIND the NPC at close range, the NPC is in trouble.
  if (dist < 35 && Math.abs(bearingFromPlayer) < Math.PI / 4) {
    return 'EVADE';
  }
  // Far away → reposition / regroup at altitude.
  if (dist > 110) return 'REGROUP';
  // If the player is ahead in the NPC's cone, just chase.
  if (Math.abs(bearing) < Math.PI / 6) return 'PURSUE';
  // Otherwise, take an intercept geometry; rotate flanks per-index for spread.
  if (dist > 60) return 'INTERCEPT';
  return npcIndex % 2 === 0 ? 'FLANK_LEFT' : 'FLANK_RIGHT';
}

/** Translate a tactical mode into an AircraftState goal pose. The planner
 *  uses goalRadius ~ 10, so these need only be in the right neighbourhood. */
export function tacticalGoal(
  player: AircraftState,
  playerPredict: Predict<AircraftState>,
  npc: AircraftState,
  mode: TacticalMode,
): AircraftState {
  const ahead = (t: number) => playerPredict(t) ?? player;
  const dx = player.x - npc.x;
  const dz = player.z - npc.z;
  const dist = Math.hypot(dx, dz);
  const eta = Math.min(8, Math.max(1, dist / DOGFIGHT_AGENT.maxSpeed));
  const future = ahead(npc.t + eta);

  const speed = DOGFIGHT_AGENT.maxSpeed;
  const base = {
    heading: 0,
    pitch: 0,
    roll: 0,
    speed,
    t: 0,
  };

  switch (mode) {
    case 'PURSUE': {
      // 18 m behind the predicted player position, same heading.
      const c = Math.cos(future.heading);
      const s = Math.sin(future.heading);
      return {
        ...base,
        x: future.x - 18 * c,
        y: clampAlt(future.y),
        z: future.z - 18 * s,
        heading: future.heading,
      };
    }
    case 'INTERCEPT': {
      // 12 m ahead of the predicted player position, opposing heading so the
      // NPC arrives head-on. Higher cruise altitude for a top attack.
      const c = Math.cos(future.heading);
      const s = Math.sin(future.heading);
      return {
        ...base,
        x: future.x + 12 * c,
        y: clampAlt(future.y + 8),
        z: future.z + 12 * s,
        heading: future.heading + Math.PI,
      };
    }
    case 'FLANK_LEFT':
    case 'FLANK_RIGHT': {
      // Sit 22 m off the predicted player's wing.
      const sign = mode === 'FLANK_LEFT' ? -1 : 1;
      const c = Math.cos(future.heading + (sign * Math.PI) / 2);
      const s = Math.sin(future.heading + (sign * Math.PI) / 2);
      return {
        ...base,
        x: future.x + 22 * c,
        y: clampAlt(future.y),
        z: future.z + 22 * s,
        heading: future.heading,
      };
    }
    case 'EVADE': {
      // Bug out: head 90° off the player's heading, climb, max speed.
      const c = Math.cos(player.heading + Math.PI / 2);
      const s = Math.sin(player.heading + Math.PI / 2);
      return {
        ...base,
        x: npc.x + 80 * c,
        y: clampAlt(npc.y + 18),
        z: npc.z + 80 * s,
        heading: player.heading + Math.PI / 2,
      };
    }
    case 'REGROUP': {
      // Climb to a vantage point on the same side as the player but higher,
      // ~70 m back so re-engagement happens at altitude.
      const c = Math.cos(player.heading);
      const s = Math.sin(player.heading);
      return {
        ...base,
        x: player.x - 70 * c,
        y: clampAlt(player.y + 22),
        z: player.z - 70 * s,
        heading: player.heading,
      };
    }
  }
}

function clampAlt(y: number): number {
  const lo = DOGFIGHT_BOUNDS.floor + 14;
  const hi = DOGFIGHT_BOUNDS.ceiling - 8;
  return Math.max(lo, Math.min(hi, y));
}

function wrapPi(a: number): number {
  let r = ((a + Math.PI) % (2 * Math.PI)) - Math.PI;
  if (r < -Math.PI) r += 2 * Math.PI;
  return r;
}

// ---------------------------------------------------------------------------
// Player-trajectory predictor — constant-curvature, constant-pitch
// extrapolation from the player's current state. Lifted slightly from the
// 2D `fromObservations` factory but in full 3D so the planner can route
// around the player's altitude as well as its XZ position.

export function playerForecast(
  current: AircraftState,
  horizon = 4,
): Predict<AircraftState> {
  // Snapshot at call time; the caller is expected to rebuild this each plan.
  const s = current;
  const omega = s.speed * (1 / DOGFIGHT_AGENT.minTurnRadius) *
    Math.tanh(s.roll / DOGFIGHT_AGENT.maxBank);
  const hSpeed = s.speed * Math.cos(s.pitch);
  const vSpeed = s.speed * Math.sin(s.pitch);
  return (t) => {
    const dt = t - s.t;
    if (dt < 0 || dt > horizon) return null;
    const heading = s.heading + omega * dt;
    // Integrate the constant-curvature arc; for small omega·dt this collapses
    // to straight-line motion (avoids a 0/0).
    let x: number;
    let z: number;
    if (Math.abs(omega) > 1e-4) {
      const r = hSpeed / omega;
      x = s.x + r * (Math.sin(heading) - Math.sin(s.heading));
      z = s.z - r * (Math.cos(heading) - Math.cos(s.heading));
    } else {
      x = s.x + hSpeed * Math.cos(s.heading) * dt;
      z = s.z + hSpeed * Math.sin(s.heading) * dt;
    }
    return {
      x,
      y: s.y + vSpeed * dt,
      z,
      heading,
      pitch: s.pitch,
      roll: s.roll,
      speed: s.speed,
      t,
    };
  };
}

// ---------------------------------------------------------------------------
// One planning call per AI. Wraps a HeightfieldAirspace+AircraftEnvironment in
// a TimeAwareEnvironment, with the player + the OTHER AIs as moving obstacles
// (each AI's published plan from the registry is converted via asObstacle).

export interface AIPlanRequest {
  /** Stable AI id (also the PlanRegistry key for this NPC's published plan). */
  npcId: string;
  /** Current AI pose. */
  state: AircraftState;
  /** Goal pose from the tactical layer. */
  goal: AircraftState;
  /** Current player pose (for the playerForecast predictor). */
  player: AircraftState;
  /** Shared registry — this NPC reads sibling NPC plans from it. */
  registry: PlanRegistry;
  /** Ids of OTHER NPCs whose plans should be treated as moving obstacles. */
  otherNpcs: string[];
  /** Anytime planning deadline (ms). */
  deadlineMs?: number;
  /** Hard expansion cap. */
  maxExpansions?: number;
  /** Optional override for the AircraftEnvironment options. */
  envOpts?: AircraftEnvOptions;
}

export function planAI(
  airspace: HeightfieldAirspace,
  req: AIPlanRequest,
): PlanResult<AircraftState> {
  const baseEnv = new AircraftEnvironment(airspace, DOGFIGHT_AGENT, {
    posCell: 4,
    altCell: 4,
    headingBuckets: 16,
    pitchBuckets: 4,
    speedQuant: 4,
    levelDivisors: [4, 2, 1],
    goalRadius: 10,
    goalHeadingTol: Infinity,
    primDuration: 1,
    substeps: 4,
    analyticExpansion: { everyN: 6, step: 3 },
    ...req.envOpts,
  });
  const playerPredict = playerForecast(req.player, 6);
  // Circumscribed agent radius for the time-aware moving-obstacle test.
  const agentR = Math.hypot(
    DOGFIGHT_AGENT.halfLength,
    DOGFIGHT_AGENT.halfSpan,
  );
  const obstacles = [
    asObstacle(playerPredict, 4),
    ...req.otherNpcs.map((id) =>
      asObstacle(req.registry.predictNPC(id) as Predict<{ x: number; z: number }>, 4),
    ),
  ];
  const env = new TimeAwareEnvironment(baseEnv, {
    obstacles,
    agentRadius: agentR,
    broadphase: { sampleStep: 0.5, maxSamples: 32 },
  });
  return plan(
    {
      start: req.state,
      goal: req.goal,
      environment: env,
      options: {
        maxExpansions: req.maxExpansions ?? DOGFIGHT_MAX_EXPANSIONS,
      },
    },
    req.deadlineMs ?? DOGFIGHT_REPLAN_BUDGET_MS,
  );
}

// ---------------------------------------------------------------------------
// Headless deterministic snapshot — what the scenario test asserts. Spawns
// the player + 2 AIs at canonical positions, has each AI pick a tactical
// goal against the player, and runs one plan call per AI. The test then
// asserts every AI found a plan within the (one-shot, generous) budget.

export interface DogfightSnapshot {
  player: AircraftState;
  ais: Array<{
    id: string;
    state: AircraftState;
    mode: TacticalMode;
    goal: AircraftState;
    result: PlanResult<AircraftState>;
  }>;
}

export function buildDogfightSnapshot(): DogfightSnapshot {
  const airspace = dogfightAirspace();
  const player: AircraftState = {
    x: 30,
    y: 35,
    z: 0,
    heading: 0,
    pitch: 0,
    roll: 0,
    speed: DOGFIGHT_AGENT.maxSpeed,
    t: 0,
  };
  const ais: DogfightSnapshot['ais'] = [];
  const registry = new PlanRegistry();
  const starts: AircraftState[] = [
    {
      x: 180,
      y: 40,
      z: -25,
      heading: Math.PI,
      pitch: 0,
      roll: 0,
      speed: DOGFIGHT_AGENT.maxSpeed,
      t: 0,
    },
    {
      x: 200,
      y: 50,
      z: 30,
      heading: Math.PI,
      pitch: 0,
      roll: 0,
      speed: DOGFIGHT_AGENT.maxSpeed,
      t: 0,
    },
  ];
  for (let i = 0; i < starts.length; i++) {
    const id = `AI${i}`;
    const state = starts[i]!;
    const mode = selectTacticalMode(player, state, i);
    const playerPredict = playerForecast(player, 6);
    const goal = tacticalGoal(player, playerPredict, state, mode);
    const result = planAI(airspace, {
      npcId: id,
      state,
      goal,
      player,
      registry,
      otherNpcs: starts.map((_, j) => `AI${j}`).filter((_, j) => j !== i),
      deadlineMs: Infinity,
      maxExpansions: DOGFIGHT_TEST_MAX_EXPANSIONS,
    });
    if (result.found) registry.publish(id, result.path);
    ais.push({ id, state, mode, goal, result });
  }
  return { player, ais };
}
