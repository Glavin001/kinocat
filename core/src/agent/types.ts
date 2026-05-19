import type { Pt } from '../internal/geom';

/** Vehicle search state. Planning plane is world XZ; Y is derived from
 *  polygon containment and is NOT part of the search state. `speed` is signed
 *  (negative = reverse). `t` is absolute time (used from M4 on). */
export interface VehicleState {
  x: number;
  z: number;
  heading: number;
  speed: number;
  t: number;
}

/** Humanoid search state — no inertial `speed` dimension (M7). */
export interface HumanoidState {
  x: number;
  z: number;
  heading: number;
  t: number;
}

/** Aircraft search state. Unlike VehicleState, altitude `y` is part of the
 *  searched state — a genuinely 3D plan, not an XZ plan with derived height.
 *  `heading` is the XZ-plane bearing (yaw), `pitch` the flight-path angle
 *  (climb positive), `speed` forward-only (the airframe cannot fly backward). */
export interface AircraftState {
  x: number;
  y: number;
  z: number;
  heading: number;
  pitch: number;
  speed: number;
  t: number;
}

export type AgentState = VehicleState | HumanoidState | AircraftState;

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

export interface AircraftAgent {
  kind: 'aircraft';
  /** Minimum turning radius in the horizontal plane (world units). */
  minTurnRadius: number;
  /** Stall speed — the airframe cannot fly slower than this. */
  minSpeed: number;
  maxSpeed: number;
  /** Max |flight-path angle| (radians) for climb or descent. */
  maxClimbAngle: number;
  /** Collision-sphere radius circumscribing the airframe. */
  radius: number;
}

export type AgentModel = VehicleAgent | HumanoidAgent | AircraftAgent;
