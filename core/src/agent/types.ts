import type { Pt } from '../internal/geom';

/** Car (ground-vehicle) kinematic search state. Planning plane is world XZ;
 *  Y is derived from polygon containment and is NOT part of the search
 *  state. `speed` is signed (negative = reverse). `t` is absolute time.
 *
 *  `yawRate` (rad/s about world +Y, planning-frame sign) and `lateralVelocity`
 *  (m/s along chassis +right) are OPTIONAL: legacy producers (kinematic sim,
 *  scenarios, older recorded data) omit them and consumers should default to
 *  0. The v2 learned dynamics model and the Rapier adapter populate them so
 *  the planner can carry yaw / slip continuity across primitives — the missing
 *  Markov state in the original 4-D `(x, z, heading, speed)` formulation. */
export interface CarKinematicState {
  x: number;
  z: number;
  heading: number;
  speed: number;
  t: number;
  /** rad/s about world +Y. Defaults to 0 when absent. */
  yawRate?: number;
  /** m/s along chassis +right (slip indicator). Defaults to 0 when absent. */
  lateralVelocity?: number;
}

/** Humanoid search state — no inertial `speed` dimension (M7). */
export interface HumanoidState {
  x: number;
  z: number;
  heading: number;
  t: number;
}

/** Momentum humanoid search state. Velocity is a world-frame vector kept
 *  SEPARATE from facing (`heading`): people strafe and backpedal at low
 *  speed but must face their motion to run, and a sprinter cannot turn a
 *  corner the way a walker can. This is the "realistic person through space
 *  and time" state — inertial, kinodynamic, planned by the same IGHA* core. */
export interface MomentumHumanoidState {
  x: number;
  z: number;
  /** Facing (rad). Distinct from the motion direction. */
  heading: number;
  /** World-frame velocity (m/s). */
  vx: number;
  vz: number;
  t: number;
}

/** Aircraft search state. Unlike CarKinematicState, altitude `y` is part of the
 *  searched state — a genuinely 3D plan, not an XZ plan with derived height.
 *  `heading` is the XZ-plane bearing (yaw), `pitch` the flight-path angle
 *  (climb positive), `roll` the bank angle around the forward axis (lets the
 *  planner knife-edge through tight slots when the OBB footprint demands it),
 *  `speed` forward-only (the airframe cannot fly backward). */
export interface AircraftState {
  x: number;
  y: number;
  z: number;
  heading: number;
  pitch: number;
  roll: number;
  speed: number;
  t: number;
}

export type AgentState =
  | CarKinematicState
  | HumanoidState
  | MomentumHumanoidState
  | AircraftState;

export interface VehicleAgent {
  kind: 'vehicle';
  /** Minimum turning radius (world units). */
  minTurnRadius: number;
  maxSpeed: number;
  /** Max reverse speed (positive magnitude). */
  maxReverseSpeed: number;
  /** Footprint polygon in body-local frame (heading 0 = +x). */
  footprint: Pt[];
  /** g-cost multiplier applied to reverse-gear edges (default 2). */
  reverseCostMultiplier: number;
  /** Extra g-cost when an edge flips gear vs. its parent edge (sec-equiv). */
  directionChangePenalty: number;
}

export interface HumanoidAgent {
  kind: 'humanoid';
  /** Body radius (round footprint). */
  radius: number;
  maxSpeed: number;
}

export interface MomentumHumanoidAgent {
  kind: 'momentum-humanoid';
  /** Body radius (round footprint). */
  radius: number;
  /** Sprint speed (m/s) — reachable only when facing the motion. */
  maxSpeed: number;
  /** Max speed of NON-facing motion (strafe / backpedal), m/s. */
  strafeSpeed: number;
  /** Max launch acceleration (m/s²). */
  maxAccel: number;
  /** Max braking deceleration (m/s²) — humans brake harder than they launch. */
  maxDecel: number;
  /** Turn rate at rest (rad/s); the effective rate shrinks with speed. */
  maxTurnRate: number;
}

export interface AircraftAgent {
  kind: 'aircraft';
  /** Minimum turning radius in the horizontal plane (world units). */
  minTurnRadius: number;
  /** Stall speed — the airframe cannot fly slower than this. */
  minSpeed: number;
  maxSpeed: number;
  /** Max |flight-path angle| (radians) for climb or descent. */
  maxClimbAngle: number;
  /** Max |bank angle| (radians). At ±π/2 the wings go vertical so the OBB
   *  footprint can slip through a tall narrow slot. */
  maxBank: number;
  /** Body-frame half-extents of the oriented collision box, in world units:
   *  along body forward (X), lateral wingspan (Y), and vertical thickness (Z). */
  halfLength: number;
  halfSpan: number;
  halfHeight: number;
}

export type AgentModel =
  | VehicleAgent
  | HumanoidAgent
  | MomentumHumanoidAgent
  | AircraftAgent;
