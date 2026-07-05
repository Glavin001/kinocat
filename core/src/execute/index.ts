// kinocat/execute — curvature-aware pure-pursuit tracker + replan logic.
export type {
  PurePursuitConfig,
  TrackingCommand,
  ReplanTrigger,
  ReplanReason,
  PlanPath,
} from './types';
export { purePursuit } from './pure-pursuit';
export { ReplanState, planPoseAt } from './replan';
export {
  createSettleLatch,
  type SettleConfig,
  type SettleSample,
  type SettleState,
  type SettleLatch,
} from './settle';
export {
  smoothSpeedProfile,
  curvaturePerSample,
  resampleScalarByArcLength,
  type SpeedProfileOptions,
} from './speed-profile';
export { smoothTrajectory, type TrajectorySmoothOptions } from './trajectory-smoother';
export {
  mpcTrack,
  createMPCTrackerState,
  type MPCTrackerConfig,
  type MPCTrackerState,
  type MPCCommand,
} from './mpc-tracker';
