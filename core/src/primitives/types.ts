/** Game-supplied forward model: advance `state` under `controls` for `dt`. */
export type ForwardSim<S> = (state: S, controls: number[], dt: number) => S;

/** A pose in the primitive's start-local frame (start at origin, heading 0). */
export interface LocalPose {
  x: number;
  z: number;
  heading: number;
}

/** A pre-characterized short feasible trajectory the agent can execute. */
export interface MotionPrimitive {
  id: number;
  /** Representative start-speed bucket this primitive was characterized at. */
  startSpeed: number;
  /** Opaque control vector — meaningful only to the ForwardSim. */
  controls: number[];
  duration: number;
  /** End-state offset in the start-local frame. */
  end: { dx: number; dz: number; dHeading: number; speed: number };
  /** Sampled swept poses (local frame) for collision sweeping. */
  sweep: LocalPose[];
  /** Gear: true if this primitive travels in reverse. */
  reverse: boolean;
}

export interface SerializedLibrary {
  primitives: MotionPrimitive[];
  startSpeeds: number[];
}
