// Prediction is the abstraction boundary for everything dynamic. The planner
// never knows what produced a predictor — linear extrapolation, plan-registry
// lookup, physics rollout, observation smoothing — it just queries predict(t).

/** Returns the predicted value at absolute time `t`, or null if unknown /
 *  outside the predictor's validity horizon. */
export type Predict<T> = (t: number) => T | null;

/** A circular footprint proxy used for fast dynamic-collision tests. */
export interface MovingObstacle {
  /** World position over time (XZ + Y ignored for planning-plane tests). */
  predict: Predict<{ x: number; z: number }>;
  /** Collision radius (added to the agent's footprint circumscribed radius). */
  radius: number;
}

export interface AffordanceState {
  position: { x: number; y: number; z: number };
  heading?: number;
}
