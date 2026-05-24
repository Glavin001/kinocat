// Car-domain types. Canonical home for the kinematic state + controls that
// every car-based demo / training pipeline uses. The names here are the
// preferred forward-looking names; older `VehicleState` / `WheeledControls`
// aliases remain in `core/src/agent/` and structurally identical.
//
// Where a generic `<S, C>` parameter wants concrete types for the car layer,
// it picks `<CarKinematicState, WheeledCarControls>`.

export type { CarKinematicState } from '../../agent/types';
export type { WheeledCarControls } from '../../agent/controls';
