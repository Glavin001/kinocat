// Keyboard -> Ackermann command mapping for car demos.
//
// WASD / arrow-key snapshot in, `(steer, throttle, brake)` out. Steering uses
// the planning sign convention: positive steer = +X-toward-+Z rotation. Demos
// pass the result to a `Body.applyControls` that converts to the Rapier-side
// sign as needed.

export interface KeyState {
  /** Steer left key (A / ArrowLeft). */
  left: boolean;
  /** Steer right key (D / ArrowRight). */
  right: boolean;
  /** Throttle forward (W / ArrowUp). */
  forward: boolean;
  /** Reverse / decelerate (S / ArrowDown). */
  backward: boolean;
  /** Brake (Space). */
  brake: boolean;
}

export interface AckermannKeyboardCommand {
  /** Steer in planning sign, normalized to [-1, 1]. */
  steer: number;
  /** Throttle in [-1, 1] (negative = reverse). */
  throttle: number;
  /** Brake in [0, 1]. */
  brake: number;
}

export interface KeyboardOpts {
  /** Steering gain — full deflection sends `steer = ±steerGain`. Default 0.55. */
  steerGain?: number;
}

/** Pure mapping; no state. */
export function keyboardAckermann(
  keys: KeyState,
  opts: KeyboardOpts = {},
): AckermannKeyboardCommand {
  const steerGain = opts.steerGain ?? 0.55;
  const steer = (keys.right ? 1 : 0) - (keys.left ? 1 : 0);
  const throttle = (keys.forward ? 1 : 0) - (keys.backward ? 1 : 0);
  const brake = keys.brake ? 1 : 0;
  return { steer: steer * steerGain, throttle, brake };
}

/** Common keymap shape: map a Set<string> of currently-pressed keys (lower-
 *  case) onto a `KeyState`. Demos that already keep a `Set<string>` for the
 *  rAF loop can call this for free. */
export function keysFromSet(active: ReadonlySet<string>): KeyState {
  return {
    left: active.has('a') || active.has('arrowleft'),
    right: active.has('d') || active.has('arrowright'),
    forward: active.has('w') || active.has('arrowup'),
    backward: active.has('s') || active.has('arrowdown'),
    brake: active.has(' ') || active.has('space'),
  };
}
