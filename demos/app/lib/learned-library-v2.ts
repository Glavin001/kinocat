// Demo-side helper that builds a planner-ready `MotionPrimitiveLibrary`
// from the v2 learned vehicle model. Drop-in replacement for the legacy
// `buildLearnedLibrary()` so the race-primitives scenario can compare the
// two side by side.
//
// Architecture note: the planner is control-vector-agnostic — it just
// chains state transitions. The library here uses the native wheeled
// control vector `[steer, driveForce, brakeForce]`. Execution-side
// pure-pursuit follows the resulting state path the same way it would for
// any other library.

import { characterizeVehicle, MotionPrimitiveLibrary, coarseWheeledControls, fineWheeledControls } from 'kinocat/primitives';
import {
  learnedForwardSimV2,
  buildParametricOnlyModel,
  DEFAULT_LEARNABLE_CONFIG,
  DEFAULT_LEARNED_PARAMS_V2,
  type LearnedVehicleModel,
  type LearnedVehicleParamsV2,
  type LearnableVehicleConfig,
} from 'kinocat/agent';

export interface BuildV2LibraryOptions {
  params?: LearnedVehicleParamsV2;
  config?: LearnableVehicleConfig;
  model?: LearnedVehicleModel;
  /** Library tier: 'coarse' (5 actions, 0.5s primitives) or 'fine' (≥15
   *  actions, 0.15s primitives). Coarse is the IGHA* default expansion;
   *  fine is for bottleneck refinement. */
  tier?: 'coarse' | 'fine';
  startSpeeds?: number[];
  duration?: number;
  substeps?: number;
}

const DEFAULT_COARSE_START_SPEEDS = [0, 4, 8, 12];
const DEFAULT_FINE_START_SPEEDS = [0, 3, 6, 9, 12, 15];

export function buildLearnedLibraryV2(opts: BuildV2LibraryOptions = {}): MotionPrimitiveLibrary {
  const tier = opts.tier ?? 'coarse';
  const config = opts.config ?? DEFAULT_LEARNABLE_CONFIG;
  const model = opts.model ?? buildParametricOnlyModel(
    opts.params ?? DEFAULT_LEARNED_PARAMS_V2,
    config,
  );
  const controlSets = tier === 'coarse'
    ? coarseWheeledControls({ config })
    : fineWheeledControls({ config });
  const startSpeeds = opts.startSpeeds ?? (tier === 'coarse' ? DEFAULT_COARSE_START_SPEEDS : DEFAULT_FINE_START_SPEEDS);
  const duration = opts.duration ?? (tier === 'coarse' ? 0.5 : 0.15);
  const substeps = opts.substeps ?? (tier === 'coarse' ? 6 : 3);
  return characterizeVehicle({
    forwardSim: learnedForwardSimV2(model),
    controlSets,
    duration,
    substeps,
    startSpeeds,
  });
}
