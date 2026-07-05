// Prediction is the abstraction boundary for everything dynamic. The planner
// never knows what produced a predictor — linear extrapolation, plan-registry
// lookup, physics rollout, observation smoothing — it just queries predict(t).

/** Returns the predicted value at absolute time `t`, or null if unknown /
 *  outside the predictor's validity horizon. */
export type Predict<T> = (t: number) => T | null;

/** A circular/spherical footprint proxy used for fast dynamic-collision
 *  tests. `y` on the prediction is optional: when BOTH the prediction and
 *  the agent's search state carry a numeric `y`, the proxy is a 3D sphere;
 *  otherwise the planning-plane XZ circle (the original behavior). A y-less
 *  obstacle against a 3D agent therefore acts as an infinite vertical
 *  cylinder — conservative by construction. */
export interface MovingObstacle {
  /** World position over time. */
  predict: Predict<{ x: number; z: number; y?: number }>;
  /** Collision radius (added to the agent's footprint circumscribed radius). */
  radius: number;
}

export interface AffordanceState {
  position: { x: number; y: number; z: number };
  heading?: number;
}
