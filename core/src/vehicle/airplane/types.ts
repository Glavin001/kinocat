// Airplane-domain types. Stub — fleshed out alongside a future airplane
// demo. Purpose right now is to prove the generic scene runtime in
// `kinocat/scene` compiles and unit-tests pass with non-car state/controls.
//
// Re-exports `AircraftState` (already defined in `kinocat/agent/types`) and
// introduces a fresh `AirplaneControls` for 4-channel flight (throttle +
// elevator + aileron + rudder).

export type { AircraftState as PlaneState } from '../../agent/types';

/** Pilot-frame controls for a fixed-wing airplane. Each axis in [-1, 1]
 *  except throttle (typically [0, 1]). */
export interface AirplaneControls {
  /** Engine power, [0, 1] (0 = idle, 1 = full thrust). */
  throttle: number;
  /** Elevator deflection, [-1, 1] (+ = nose-up). */
  elevator: number;
  /** Aileron deflection, [-1, 1] (+ = roll right). */
  aileron: number;
  /** Rudder deflection, [-1, 1] (+ = yaw right). */
  rudder: number;
}

/** Convenience zero. */
export const AIRPLANE_CONTROLS_ZERO: AirplaneControls = {
  throttle: 0,
  elevator: 0,
  aileron: 0,
  rudder: 0,
};
