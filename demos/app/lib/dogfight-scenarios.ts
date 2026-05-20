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
  ResourceAwareEnvironment,
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

// --- Speed modes (shared action space for player + NPCs) -------------------
// Three discrete primitives. SLOW conserves fuel + tightens turn radius;
// NORMAL is the baseline cruise; BOOST is twice NORMAL but consumes fuel
// and (via `turnRadiusAt`) sweeps a much wider turn circle, so the planner
// naturally avoids it near terrain.
export const DOGFIGHT_SPEED_SLOW = 24;
export const DOGFIGHT_SPEED_NORMAL = 32;
export const DOGFIGHT_SPEED_BOOST = 64;
export type SpeedMode = 'SLOW' | 'NORMAL' | 'BOOST';
export const DOGFIGHT_SPEED: Record<SpeedMode, number> = {
  SLOW: DOGFIGHT_SPEED_SLOW,
  NORMAL: DOGFIGHT_SPEED_NORMAL,
  BOOST: DOGFIGHT_SPEED_BOOST,
};
/** A primitive is "boost-class" when its target speed exceeds this. Used by
 *  the fuel-aware planner gate and the runtime fuel accountant. */
export const DOGFIGHT_BOOST_SPEED_THRESHOLD = DOGFIGHT_SPEED_NORMAL * 1.3;

// --- Boost-fuel mechanics ---------------------------------------------------
export const DOGFIGHT_FUEL_MAX = 100;
/** Fuel regen (units/sec) while not boosting. */
export const DOGFIGHT_FUEL_REGEN = 8;
/** Fuel consume (units/sec) while boosting. */
export const DOGFIGHT_FUEL_CONSUME = 30;
/** Fuel restored when a pilot flies through a ring. */
export const DOGFIGHT_FUEL_RING_GIFT = 45;
/** Below this, boost is unavailable (engine must "spool up"). */
export const DOGFIGHT_FUEL_BOOST_MIN = 5;
/** Quantization for the planner's fuel dimension (5 buckets over [0,100]). */
export const DOGFIGHT_FUEL_QUANTUM = 25;

/** Same envelope as the /plane demo so existing primitive characterizations
 *  remain in scope; tuned for the dogfight at the demo's elevated cruise.
 *  `turnRadiusAt` scales the min turn radius with (v/NORMAL)² so the planner
 *  natively weighs boost's reduced agility against its forward speed. */
export const DOGFIGHT_AGENT: AircraftAgent = defaultAircraftAgent({
  minTurnRadius: 22,
  minSpeed: 18,
  // BOOST sits at the cap so the simulator's clamp never silently
  // throttles a boosting NPC's planned speed.
  maxSpeed: DOGFIGHT_SPEED_BOOST,
  maxClimbAngle: Math.PI / 5,
  maxBank: Math.PI / 3,
  halfLength: 2,
  halfSpan: 1.8,
  halfHeight: 0.35,
  turnRadiusAt: (speed) =>
    22 * Math.pow(speed / DOGFIGHT_SPEED_NORMAL, 2),
});

export const DOGFIGHT_HALF: [number, number, number] = [
  DOGFIGHT_AGENT.halfLength,
  DOGFIGHT_AGENT.halfSpan,
  DOGFIGHT_AGENT.halfHeight,
];

export const DOGFIGHT_BOUNDS = {
  x0: -20,
  x1: 280,
  z0: -130,
  z1: 130,
  floor: 0,
  ceiling: 130,
};

const MAP_CX = (DOGFIGHT_BOUNDS.x0 + DOGFIGHT_BOUNDS.x1) / 2;
const MAP_CZ = (DOGFIGHT_BOUNDS.z0 + DOGFIGHT_BOUNDS.z1) / 2;
const MAP_HX = (DOGFIGHT_BOUNDS.x1 - DOGFIGHT_BOUNDS.x0) / 2;
const MAP_HZ = (DOGFIGHT_BOUNDS.z1 - DOGFIGHT_BOUNDS.z0) / 2;

// ---------------------------------------------------------------------------
// Terrain — perimeter mountains hugging a smoothly-curving low-altitude ring,
// with two canyon passes cutting between the mountains, and a flat-ish city
// floor at the centre. The same analytic function feeds both the planner's
// HeightfieldAirspace and the renderer's displaced PlaneGeometry, so what the
// pilot sees IS what the planner respects.

/** Closed parametric spline radius (normalized 0..1) at polar angle `a`.
 *  Sum-of-sines so it's smooth, closed (period 2π), and visibly non-circular. */
export function dogfightSplineRadius(a: number): number {
  return (
    0.78 +
    0.08 * Math.sin(3 * a + 0.7) +
    0.05 * Math.sin(5 * a - 0.4) +
    0.03 * Math.sin(7 * a + 1.2)
  );
}

/** Mountain depression at canyon passes — 1 = full mountain, 0 = open pass. */
function canyonGate(a: number): number {
  // Two canyon openings: one east (a≈0), one south-west (a≈π+0.6).
  const g1 = Math.exp(-((a - 0) * (a - 0)) / 0.04);
  const g2 = Math.exp(-((a - Math.PI - 0.6) * (a - Math.PI - 0.6)) / 0.06);
  const g3 = Math.exp(-((a + Math.PI + 0.6) * (a + Math.PI + 0.6)) / 0.06);
  return Math.max(0, 1 - 0.85 * Math.max(g1, g2, g3));
}

export const dogfightTerrain: HeightfieldSampler = (x, z) => {
  const u = (x - MAP_CX) / MAP_HX;
  const v = (z - MAP_CZ) / MAP_HZ;
  const r = Math.hypot(u, v);
  const a = Math.atan2(v, u);
  const R = dogfightSplineRadius(a);

  // Distance outward from the spline path (negative = inside city, positive
  // = entering perimeter mountains).
  const dOut = r - R;

  // Mountains rise on the outside of the spline; canyon gates suppress them.
  let mountain = 0;
  if (dOut > -0.02) {
    const t = Math.min(1, Math.max(0, (dOut + 0.02) / 0.38));
    const peakHeight = 78; // metres
    const rough = 0.92 + 0.08 * Math.cos(a * 11);
    mountain = peakHeight * smoothStep01(t) * rough * canyonGate(a);
    // Beyond r ≈ R + 0.4 we're on the high plateau — clamp.
    if (dOut > 0.4) mountain *= 1 - smoothStep01((dOut - 0.4) / 0.2) * 0.15;
  }

  // City floor: low rolling ground inside the spline, with gentle hills.
  const cityBase =
    1.2 * Math.sin(x * 0.07) + 0.9 * Math.cos(z * 0.09) +
    0.6 * Math.sin((x + z) * 0.13);

  // Soft transition band so the city blends into the mountains.
  let band = 0;
  if (r > 0.45 && r < R + 0.02) {
    const t = (r - 0.45) / (R + 0.02 - 0.45);
    band = 6 * smoothStep01(t);
  }

  return Math.max(0, cityBase + band + mountain);
};

function smoothStep01(t: number): number {
  const u = Math.max(0, Math.min(1, t));
  return u * u * (3 - 2 * u);
}

// ---------------------------------------------------------------------------
// City buildings + moving zones. Buildings are procedurally placed inside the
// spline ring (the "city zone"), with two clear roadway avenues left empty so
// the pilot can fly straight along the cardinal axes between buildings.

interface BuildingSpec {
  x: number;
  z: number;
  hx: number; // half-width along X
  hz: number; // half-depth along Z
  height: number;
}

const CITY_SEED = 0x9e3779b1;

/** Tiny seeded PRNG (mulberry32) so building placement is deterministic. */
function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function inCityZone(x: number, z: number): boolean {
  const u = (x - MAP_CX) / MAP_HX;
  const v = (z - MAP_CZ) / MAP_HZ;
  const r = Math.hypot(u, v);
  return r < 0.55;
}

function inAvenue(x: number, z: number): boolean {
  // Two roadway avenues — one east-west (z ≈ MAP_CZ), one north-south (x ≈ MAP_CX).
  const ROAD_HALF = 9;
  return (
    Math.abs(z - MAP_CZ) < ROAD_HALF ||
    Math.abs(x - MAP_CX) < ROAD_HALF
  );
}

function buildCitySpecs(): BuildingSpec[] {
  const r = rng(CITY_SEED);
  const out: BuildingSpec[] = [];
  const TRIES = 600;
  const TARGET = 38;
  for (let i = 0; i < TRIES && out.length < TARGET; i++) {
    // Random footprint somewhere inside the city zone.
    const x = MAP_CX + (r() - 0.5) * MAP_HX * 1.15;
    const z = MAP_CZ + (r() - 0.5) * MAP_HZ * 1.15;
    if (!inCityZone(x, z)) continue;
    if (inAvenue(x, z)) continue;
    // Mix of short / mid / tall — biased toward shorter so the skyline reads.
    const tier = r();
    let height: number;
    let hxz: number;
    if (tier < 0.45) {
      // Short buildings: 6–14 m.
      height = 6 + r() * 8;
      hxz = 3 + r() * 2.5;
    } else if (tier < 0.85) {
      // Mid: 16–30 m.
      height = 16 + r() * 14;
      hxz = 3.5 + r() * 3;
    } else {
      // Tall: 36–60 m (skyscrapers, hard to fly over).
      height = 36 + r() * 24;
      hxz = 4 + r() * 4;
    }
    const hx = hxz * (0.7 + r() * 0.6);
    const hz = hxz * (0.7 + r() * 0.6);
    // Reject overlap with already-placed buildings (with a 3 m flight gap).
    const gap = 3;
    let ok = true;
    for (const b of out) {
      if (
        Math.abs(x - b.x) < hx + b.hx + gap &&
        Math.abs(z - b.z) < hz + b.hz + gap
      ) {
        ok = false;
        break;
      }
    }
    if (!ok) continue;
    // Reject if straddling an avenue after expansion.
    if (Math.abs(z - MAP_CZ) - hz < 9) continue;
    if (Math.abs(x - MAP_CX) - hx < 9) continue;
    out.push({ x, z, hx, hz, height });
  }
  return out;
}

const _CITY = buildCitySpecs();

/** Mixed-height city buildings inside the spline ring (no grid; avenues left
 *  empty for east-west and north-south flight). */
export function dogfightStaticObstacles(): AABB[] {
  const f = DOGFIGHT_BOUNDS.floor;
  return _CITY.map((b) => ({
    min: [b.x - b.hx, f, b.z - b.hz] as [number, number, number],
    max: [b.x + b.hx, b.height, b.z + b.hz] as [number, number, number],
  }));
}

/** Building specs for visual reference (extends the AABB list with metadata
 *  the renderer uses to colour the skyline). */
export function dogfightBuildings(): BuildingSpec[] {
  return _CITY;
}

/** A drifting blimp + a sweeping barrier along the central road. Kept after
 *  the redesign so the time-aware planning still has live moving obstacles. */
export function dogfightMovingZones(): MovingZone[] {
  return [
    {
      // Blimp drifting east-west over the central avenue.
      radius: 9,
      predict: (t) => ({
        x: MAP_CX + 70 * Math.sin(t * 0.18),
        y: 65 + 4 * Math.sin(t * 0.6),
        z: MAP_CZ + 8 * Math.sin(t * 0.35),
      }),
    },
    {
      // Sweeping barrier across the east canyon entrance.
      radius: 10,
      predict: (t) => ({
        x: DOGFIGHT_BOUNDS.x1 - 35,
        y: 35,
        z: 40 * Math.sin(t * 0.55),
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
// Boost rings — placed along the perimeter flightway (one per canyon mouth
// plus a string along the curved low ring) so the pilot has a natural reason
// to fly the spline at speed.

export interface BoostRing {
  id: string;
  x: number;
  y: number;
  z: number;
  /** Ring normal — passage is detected by sphere overlap, but the normal is
   *  used to orient the visual torus along the expected flight direction. */
  axis: { x: number; y: number; z: number };
  radius: number;
}

/** Sample the spline ring at angle `a` and return the world-space position. */
export function dogfightSplinePoint(a: number, yAt = 26): {
  x: number;
  y: number;
  z: number;
} {
  const R = dogfightSplineRadius(a);
  return {
    x: MAP_CX + Math.cos(a) * R * MAP_HX,
    y: yAt,
    z: MAP_CZ + Math.sin(a) * R * MAP_HZ,
  };
}

/** Tangent direction along the spline at angle `a` (XZ plane). */
export function dogfightSplineTangent(a: number): {
  x: number;
  y: number;
  z: number;
} {
  // Numerical derivative — analytic is messy with R(a), and this is cheap.
  const eps = 1e-3;
  const p0 = dogfightSplinePoint(a - eps);
  const p1 = dogfightSplinePoint(a + eps);
  const dx = p1.x - p0.x;
  const dz = p1.z - p0.z;
  const m = Math.hypot(dx, dz) || 1;
  return { x: dx / m, y: 0, z: dz / m };
}

export function dogfightBoostRings(): BoostRing[] {
  const out: BoostRing[] = [];
  const COUNT = 8;
  for (let i = 0; i < COUNT; i++) {
    const a = (i / COUNT) * Math.PI * 2;
    const p = dogfightSplinePoint(a, 28 + 4 * Math.sin(a * 2));
    const tan = dogfightSplineTangent(a);
    out.push({
      id: `R${i}`,
      x: p.x,
      y: p.y,
      z: p.z,
      axis: tan,
      radius: 7,
    });
  }
  // Plus one boost ring in each city avenue so the player has a reason to
  // dive through the buildings.
  out.push({
    id: 'CITY_EW',
    x: MAP_CX,
    y: 18,
    z: MAP_CZ,
    axis: { x: 1, y: 0, z: 0 },
    radius: 6,
  });
  out.push({
    id: 'CITY_NS',
    x: MAP_CX + 35,
    y: 22,
    z: MAP_CZ + 20,
    axis: { x: 0, y: 0, z: 1 },
    radius: 6,
  });
  return out;
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

/** Pick the *goal* speed for the NPC's plan based on tactic + fuel. The
 *  planner chooses which speeds to use *along* the trajectory from its
 *  fuel-gated allowed set; this picks where the NPC wants to *end up*. */
export function chooseNpcSpeedMode(
  mode: TacticalMode,
  fuel: number,
): SpeedMode {
  const canBoost = fuel > DOGFIGHT_FUEL_BOOST_MIN;
  switch (mode) {
    case 'EVADE':
      return canBoost ? 'BOOST' : 'NORMAL';
    case 'PURSUE':
      return canBoost && fuel > 30 ? 'BOOST' : 'NORMAL';
    case 'INTERCEPT':
    case 'FLANK_LEFT':
    case 'FLANK_RIGHT':
      return 'NORMAL';
    case 'REGROUP':
      return 'SLOW';
  }
}

/** Translate a tactical mode + chosen speed into an AircraftState goal
 *  pose. The planner uses goalRadius ~ 14, so these need only be in the
 *  right neighbourhood. */
export function tacticalGoal(
  player: AircraftState,
  playerPredict: Predict<AircraftState>,
  npc: AircraftState,
  mode: TacticalMode,
  speedMode: SpeedMode = 'NORMAL',
): AircraftState {
  const ahead = (t: number) => playerPredict(t) ?? player;
  const dx = player.x - npc.x;
  const dz = player.z - npc.z;
  const dist = Math.hypot(dx, dz);
  const eta = Math.min(8, Math.max(1, dist / DOGFIGHT_AGENT.maxSpeed));
  const future = ahead(npc.t + eta);

  const speed = DOGFIGHT_SPEED[speedMode];
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
  /** NPC's current boost-fuel reserve (0..DOGFIGHT_FUEL_MAX). The planner
   *  treats this as the start-node resource and gates BOOST primitives
   *  accordingly; without it, the wrapper defaults to a full tank. */
  startFuel?: number;
  /** World features the planner should see as fuel-refill affordances.
   *  Defaults to `dogfightBoostRings()`. */
  rings?: BoostRing[];
  /** Optional override for the AircraftEnvironment options. */
  envOpts?: AircraftEnvOptions;
}

/** Sphere-vs-segment overlap. The runtime ring trigger uses a sphere test
 *  too (`Dogfight.tsx` line ~750), so the planner sees the same affordance
 *  geometry the runtime enforces. Conservative: any point on the segment
 *  within `radius` of the sphere centre counts as a hit. */
function segmentHitsSphere(
  a: { x: number; y: number; z: number },
  b: { x: number; y: number; z: number },
  c: { x: number; y: number; z: number; radius: number },
): boolean {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const dz = b.z - a.z;
  const len2 = dx * dx + dy * dy + dz * dz;
  let t = 0;
  if (len2 > 1e-9) {
    t = ((c.x - a.x) * dx + (c.y - a.y) * dy + (c.z - a.z) * dz) / len2;
    if (t < 0) t = 0;
    else if (t > 1) t = 1;
  }
  const px = a.x + dx * t;
  const py = a.y + dy * t;
  const pz = a.z + dz * t;
  const ex = px - c.x;
  const ey = py - c.y;
  const ez = pz - c.z;
  return ex * ex + ey * ey + ez * ez <= c.radius * c.radius;
}

type FuelR = { fuel: number };

function clampFuel(f: number): number {
  return f < 0 ? 0 : f > DOGFIGHT_FUEL_MAX ? DOGFIGHT_FUEL_MAX : f;
}

export function planAI(
  airspace: HeightfieldAirspace,
  req: AIPlanRequest,
): PlanResult<AircraftState> {
  const baseEnv = new AircraftEnvironment(airspace, DOGFIGHT_AGENT, {
    posCell: 6,
    altCell: 5,
    headingBuckets: 16,
    pitchBuckets: 4,
    speedQuant: 6,
    levelDivisors: [4, 2, 1],
    goalRadius: 14,
    goalHeadingTol: Infinity,
    // Faster aircraft + ~5 m city footprints: shorter primitives + finer
    // substeps so OBB-vs-building swept tests don't tunnel.
    primDuration: 0.7,
    substeps: 8,
    analyticExpansion: { everyN: 6, step: 4 },
    // Three discrete speed primitives. The fuel-aware wrapper gates BOOST
    // by the per-state fuel resource so the planner never expands a boost
    // edge from an empty tank.
    speeds: [DOGFIGHT_SPEED_SLOW, DOGFIGHT_SPEED_NORMAL, DOGFIGHT_SPEED_BOOST],
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
  const timeEnv = new TimeAwareEnvironment(baseEnv, {
    obstacles,
    agentRadius: agentR,
    broadphase: { sampleStep: 0.5, maxSamples: 32 },
  });
  // Fuel-aware layer. Hooks are pure dogfight policy: BOOST edges gated by
  // current fuel; fuel drains at boost speed and regenerates at slow/normal;
  // segment crossing a ring refills. The planner sees rings as affordances,
  // so a multi-step "detour through ring → boost to target" is found by A*
  // when the path-cost arithmetic works out — no scripting.
  const rings = req.rings ?? dogfightBoostRings();
  // Inflate ring radius by 1m to match the runtime overlap test
  // (`Dogfight.tsx` uses `radius + 1` for the trigger sphere).
  const planRings = rings.map((r) => ({
    x: r.x,
    y: r.y,
    z: r.z,
    radius: r.radius + 1,
  }));
  const env = new ResourceAwareEnvironment<AircraftState, FuelR>(timeEnv, {
    initial: { fuel: req.startFuel ?? DOGFIGHT_FUEL_MAX },
    bucket: ({ fuel }) =>
      String(Math.round(fuel / DOGFIGHT_FUEL_QUANTUM)),
    allow: ({ fuel }, _from, _edge, to) =>
      to.speed < DOGFIGHT_BOOST_SPEED_THRESHOLD ||
      fuel >= DOGFIGHT_FUEL_BOOST_MIN,
    step: ({ fuel }, _from, _edge, to, dt) => {
      const isBoost = to.speed >= DOGFIGHT_BOOST_SPEED_THRESHOLD;
      const delta = isBoost
        ? -DOGFIGHT_FUEL_CONSUME * dt
        : DOGFIGHT_FUEL_REGEN * dt;
      return { fuel: clampFuel(fuel + delta) };
    },
    affordance: ({ fuel }, from, to) => {
      for (const r of planRings) {
        if (segmentHitsSphere(from, to, r)) {
          return { fuel: clampFuel(fuel + DOGFIGHT_FUEL_RING_GIFT) };
        }
      }
      return null;
    },
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
  // Spawn on the east-west avenue (z=0) — the city building generator leaves
  // a clear runway along it, so these poses always clear the airspace
  // regardless of the procedural skyline.
  const player: AircraftState = {
    x: 40,
    y: 50,
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
      x: 240,
      y: 45,
      z: 0,
      heading: Math.PI,
      pitch: 0,
      roll: 0,
      speed: DOGFIGHT_AGENT.maxSpeed,
      t: 0,
    },
    {
      x: 220,
      y: 60,
      z: 0,
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
