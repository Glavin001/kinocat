// Car-domain `RecorderFormatters` for the generic `DebugRecorder<S, C>`.
//
// Lifts `CarKinematicState` and `WheeledCarControls` into the flat
// number-record shapes the recorder needs for JSON/Markdown export and
// per-ghost gap aggregation. Domain-agnostic recorder + domain-specific
// formatter is the seam that lets airplane / hovercraft / etc reuse the
// same diagnostics machinery without per-demo duplication.

import type { RecorderFormatters } from '../../diagnostics';
import type { CarKinematicState } from './types';
import type { WheeledCarControls } from './types';

function wrapPi(a: number): number {
  let x = a;
  while (x > Math.PI) x -= 2 * Math.PI;
  while (x < -Math.PI) x += 2 * Math.PI;
  return x;
}

/** Default formatters for any car demo wiring up the generic recorder.
 *  Emits flat number records suitable for JSON / Markdown export and
 *  computes signed per-axis gaps + a derived `posDist` so the recorder's
 *  RMS table reports both component-wise and total position error. */
export const carRecorderFormatters: RecorderFormatters<CarKinematicState, WheeledCarControls> = {
  formatState: (s) => ({
    x: s.x,
    z: s.z,
    heading: s.heading,
    speed: s.speed,
    yawRate: s.yawRate ?? 0,
    lateralVelocity: s.lateralVelocity ?? 0,
    t: s.t,
  }),
  formatControls: (c) => ({
    steer: c.steer,
    driveForce: c.driveForce,
    brakeForce: c.brakeForce,
  }),
  gapMetrics: (real, predicted) => {
    const dx = predicted.x - real.x;
    const dz = predicted.z - real.z;
    return {
      posDist: Math.hypot(dx, dz),
      dx,
      dz,
      headingErr: wrapPi(predicted.heading - real.heading),
      speedErr: predicted.speed - real.speed,
    };
  },
};
