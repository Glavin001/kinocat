import type { ForwardSim, LocalPose, MotionPrimitive } from './types';
import { MotionPrimitiveLibrary } from './library';
import type { VehicleState } from '../agent/types';
import { wrapAngle } from '../internal/math';

export interface CharacterizeVehicleOptions {
  forwardSim: ForwardSim<VehicleState>;
  /** Opaque control vectors to sweep (one primitive per control × speed). */
  controlSets: number[][];
  /** Wall-clock duration of each primitive (seconds). */
  duration: number;
  /** Integration / sweep-sampling substeps per primitive. */
  substeps: number;
  /** Start-speed buckets to characterize from. */
  startSpeeds: number[];
}

/**
 * Roll the supplied ForwardSim across the control × start-speed grid and
 * record each resulting short trajectory as a motion primitive (in the
 * start-local frame: start at origin, heading 0). Deterministic and
 * physics-engine agnostic.
 */
export function characterizeVehicle(
  opts: CharacterizeVehicleOptions,
): MotionPrimitiveLibrary {
  const { forwardSim, controlSets, duration, substeps, startSpeeds } = opts;
  const dt = duration / substeps;
  const primitives: MotionPrimitive[] = [];
  let id = 0;

  for (const startSpeed of startSpeeds) {
    for (const controls of controlSets) {
      let s: VehicleState = { x: 0, z: 0, heading: 0, speed: startSpeed, t: 0 };
      const sweep: LocalPose[] = [{ x: 0, z: 0, heading: 0 }];
      for (let k = 0; k < substeps; k++) {
        s = forwardSim(s, controls, dt);
        sweep.push({ x: s.x, z: s.z, heading: wrapAngle(s.heading) });
      }
      primitives.push({
        id: id++,
        startSpeed,
        controls: [...controls],
        duration,
        end: { dx: s.x, dz: s.z, dHeading: wrapAngle(s.heading), speed: s.speed },
        sweep,
        reverse: s.speed < 0 || (controls[1] ?? 0) < 0,
      });
    }
  }
  return new MotionPrimitiveLibrary(primitives, startSpeeds);
}
