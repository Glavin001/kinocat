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

export type AgentState = VehicleState | HumanoidState;

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

export type AgentModel = VehicleAgent | HumanoidAgent;
