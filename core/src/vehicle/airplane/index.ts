// kinocat/vehicle/airplane — STUB.
//
// Architecture-sanity-check namespace that picks concrete types
// `<PlaneState, AirplaneControls>` for the generic scene runtime. A future
// real airplane demo replaces `StubAirplaneBody` with a properly-integrated
// flight model; the rest of the system (drivers, scene controller, training
// pipeline) keeps the same shape.

export type { PlaneState, AirplaneControls } from './types';
export { AIRPLANE_CONTROLS_ZERO } from './types';
export { StubAirplaneBody } from './body';
