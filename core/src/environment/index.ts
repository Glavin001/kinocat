// kinocat/environment — Environment interface, NavWorld seam + implementations.
export type { Environment, Node, EdgeRef } from './types';
export type { NavWorld, PolygonRef, OffMeshLink, NavPolygon } from './nav-world';
export { InMemoryNavWorld } from './nav-world';
export { R2Environment } from './r2-environment';
export type { R2State, R2Bounds, R2Options } from './r2-environment';
export { VehicleEnvironment } from './vehicle-environment';
export type { VehicleEnvOptions, AnalyticEdgeData } from './vehicle-environment';
export { HumanoidEnvironment } from './humanoid-environment';
export type { HumanoidEnvOptions } from './humanoid-environment';
export { AircraftEnvironment } from './aircraft-environment';
export type { AircraftEnvOptions } from './aircraft-environment';
export { InMemoryAirspace } from './airspace-world';
export type {
  AirspaceWorld,
  AirspaceOptions,
  AABB,
  MovingZone,
} from './airspace-world';
export { HeightfieldAirspace } from './heightfield-airspace';
export type {
  HeightfieldAirspaceOptions,
  HeightfieldSampler,
} from './heightfield-airspace';
export { TimeAwareEnvironment } from './time-aware';
export type { TimeAwareOptions } from './time-aware';
