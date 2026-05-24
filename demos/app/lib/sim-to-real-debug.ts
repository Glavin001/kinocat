// Rolling debug recorder for the /sim-to-real scope. Captures the last
// N frames (~10s @ 60Hz = 600 frames) of:
//   - applied controls (steer/throttle/brake AND the encoded
//     [steer-norm, driveForce, brakeForce] vector that the open-loop
//     models receive),
//   - real Rapier state after the step,
//   - each ghost model's predicted state after the same step,
//   - per-frame gap (pos / heading / speed) per model,
//   - per-wheel telemetry (contact, impulses, suspension, frictionSlip).
//
// On demand, dumps to JSON (full fidelity) or Markdown (human-readable
// summary + a fenced JSON tail) for sharing with the agent so it can
// reason about WHY the open-loop models diverge from the Rapier chassis.
//
// Designed to be cheap inside the 60Hz loop: pre-allocated ring buffer,
// plain objects, no clone of incoming states (the caller is expected to
// pass already-frozen snapshots — see SimToRealScope).

import type { VehicleState } from 'kinocat/agent';
import type { WheelTelemetry } from 'kinocat/adapters/rapier';

export interface DebugFrame {
  /** Sim-time of this frame (seconds). */
  t: number;
  /** User-side controls (post-WASD / pure-pursuit, pre-actuator). */
  applied: { steer: number; throttle: number; brake: number };
  /** Native-action controls vector fed to the open-loop forward sims
   *  AND used for `predictWithUncertainty`. Shape:
   *  [steer-norm-or-radians, driveForce N, brakeForce N]. */
  ctrlVec: [number, number, number];
  /** Rapier-real chassis state after the physics step. */
  real: VehicleState;
  /** Per-model predicted state at the SAME sim-time. Keys are the
   *  ghost ids ('v2-full' / 'parametric' / 'kinematic'). */
  ghosts: Record<string, VehicleState>;
  /** Per-model gap snapshot (signed deltas, raw — no wrap). */
  gaps: Record<string, { posErr: number; headingErr: number; speedErr: number }>;
  /** Per-wheel telemetry from the last Rapier sub-step. */
  wheels: WheelTelemetry[];
}

export interface DebugMeta {
  /** Which scope mode this run was in. */
  mode: string;
  /** Whether the model rollout matched Rapier's sub-stepping this run. */
  matchSubsteps: boolean;
  /** Physics tick used by Rapier (s). */
  physicsDt: number;
  /** Sub-steps per Rapier physics tick. */
  physicsSubsteps: number;
  /** dt the models were integrated at (= physicsDt if not matching,
   *  else physicsDt/substeps). */
  modelDt: number;
  /** Engine / brake force used to convert throttle/brake -> N. */
  engineForceN: number;
  brakeForceN: number;
  /** Was a persisted v2 model loaded, or are we on defaults? */
  hasPersistedV2: boolean;
}

export class DebugRecorder {
  private buf: DebugFrame[] = [];
  constructor(private readonly capacity: number = 600) {}

  push(frame: DebugFrame): void {
    if (this.buf.length >= this.capacity) this.buf.shift();
    this.buf.push(frame);
  }

  clear(): void { this.buf = []; }
  size(): number { return this.buf.length; }
  frames(): ReadonlyArray<DebugFrame> { return this.buf; }

  /** Pull the most recent N frames (or all if N >= size). */
  tail(n: number): DebugFrame[] {
    return this.buf.slice(Math.max(0, this.buf.length - n));
  }
}

// ---------------------------------------------------------------------------
// Aggregate statistics.

export interface DebugStats {
  perModel: Record<string, {
    /** Mean signed delta (pred - real) across the buffer. */
    meanDx: number;
    meanDz: number;
    meanDheading: number;
    meanDspeed: number;
    /** RMS over the buffer. */
    rmsPos: number;
    rmsHeading: number;
    rmsSpeed: number;
    /** Final-frame snapshot. */
    finalPosErr: number;
    finalHeadingErr: number;
    finalSpeedErr: number;
  }>;
  /** Total ticks captured. */
  ticks: number;
  /** Duration covered (s). */
  durationSec: number;
  /** Controls summary: range observed. */
  controls: {
    steer: { min: number; max: number };
    throttle: { min: number; max: number };
    brake: { min: number; max: number };
    driveForce: { min: number; max: number };
  };
}

function wrapPi(a: number): number {
  let x = a;
  while (x > Math.PI) x -= 2 * Math.PI;
  while (x < -Math.PI) x += 2 * Math.PI;
  return x;
}

export function aggregateStats(frames: ReadonlyArray<DebugFrame>): DebugStats {
  const perModel: DebugStats['perModel'] = {};
  if (frames.length === 0) {
    return {
      perModel,
      ticks: 0,
      durationSec: 0,
      controls: {
        steer: { min: 0, max: 0 },
        throttle: { min: 0, max: 0 },
        brake: { min: 0, max: 0 },
        driveForce: { min: 0, max: 0 },
      },
    };
  }
  const modelIds = Object.keys(frames[0]!.ghosts);
  for (const id of modelIds) {
    let sumDx = 0, sumDz = 0, sumDh = 0, sumDv = 0;
    let sqPos = 0, sqH = 0, sqV = 0;
    let n = 0;
    for (const f of frames) {
      const g = f.ghosts[id];
      if (!g) continue;
      const dx = g.x - f.real.x;
      const dz = g.z - f.real.z;
      const dh = wrapPi(g.heading - f.real.heading);
      const dv = g.speed - f.real.speed;
      sumDx += dx; sumDz += dz; sumDh += dh; sumDv += dv;
      sqPos += dx * dx + dz * dz;
      sqH += dh * dh;
      sqV += dv * dv;
      n++;
    }
    const last = frames[frames.length - 1]!;
    const lastG = last.ghosts[id]!;
    const fdx = lastG.x - last.real.x;
    const fdz = lastG.z - last.real.z;
    perModel[id] = {
      meanDx: sumDx / Math.max(1, n),
      meanDz: sumDz / Math.max(1, n),
      meanDheading: sumDh / Math.max(1, n),
      meanDspeed: sumDv / Math.max(1, n),
      rmsPos: Math.sqrt(sqPos / Math.max(1, n)),
      rmsHeading: Math.sqrt(sqH / Math.max(1, n)),
      rmsSpeed: Math.sqrt(sqV / Math.max(1, n)),
      finalPosErr: Math.hypot(fdx, fdz),
      finalHeadingErr: wrapPi(lastG.heading - last.real.heading),
      finalSpeedErr: lastG.speed - last.real.speed,
    };
  }
  const ctrl = {
    steer: { min: Infinity, max: -Infinity },
    throttle: { min: Infinity, max: -Infinity },
    brake: { min: Infinity, max: -Infinity },
    driveForce: { min: Infinity, max: -Infinity },
  };
  for (const f of frames) {
    const s = f.applied.steer, t = f.applied.throttle, b = f.applied.brake;
    const drv = f.ctrlVec[1];
    if (s < ctrl.steer.min) ctrl.steer.min = s;
    if (s > ctrl.steer.max) ctrl.steer.max = s;
    if (t < ctrl.throttle.min) ctrl.throttle.min = t;
    if (t > ctrl.throttle.max) ctrl.throttle.max = t;
    if (b < ctrl.brake.min) ctrl.brake.min = b;
    if (b > ctrl.brake.max) ctrl.brake.max = b;
    if (drv < ctrl.driveForce.min) ctrl.driveForce.min = drv;
    if (drv > ctrl.driveForce.max) ctrl.driveForce.max = drv;
  }
  return {
    perModel,
    ticks: frames.length,
    durationSec: frames[frames.length - 1]!.t - frames[0]!.t,
    controls: ctrl,
  };
}

// ---------------------------------------------------------------------------
// Formatters.

/** JSON dump with full per-frame fidelity. Suitable for piping into a
 *  notebook / agent for offline diff. Frames are rounded to 4 decimals
 *  to keep the file small while remaining unambiguous. */
export function toJSON(
  meta: DebugMeta,
  frames: ReadonlyArray<DebugFrame>,
  stats: DebugStats,
): string {
  const round = (x: number, n = 4): number =>
    Number.isFinite(x) ? Number(x.toFixed(n)) : x;
  const rs = (s: VehicleState) => ({
    x: round(s.x), z: round(s.z),
    heading: round(s.heading), speed: round(s.speed),
    yawRate: round(s.yawRate ?? 0), lateralVelocity: round(s.lateralVelocity ?? 0),
    t: round(s.t),
  });
  const compact = frames.map((f) => ({
    t: round(f.t),
    applied: { steer: round(f.applied.steer), throttle: round(f.applied.throttle), brake: round(f.applied.brake) },
    ctrlVec: f.ctrlVec.map((v) => round(v)) as [number, number, number],
    real: rs(f.real),
    ghosts: Object.fromEntries(Object.entries(f.ghosts).map(([k, v]) => [k, rs(v)])),
    gaps: Object.fromEntries(Object.entries(f.gaps).map(([k, v]) => [k, {
      posErr: round(Math.hypot(v.posErr, 0)), // posErr is already a distance
      headingErr: round(v.headingErr),
      speedErr: round(v.speedErr),
    }])),
    wheels: f.wheels.map((w) => ({
      inContact: w.inContact,
      cp: w.contactPoint ? { x: round(w.contactPoint.x), y: round(w.contactPoint.y), z: round(w.contactPoint.z) } : null,
      fwdImp: round(w.forwardImpulse),
      sideImp: round(w.sideImpulse),
      susp: round(w.suspensionForce),
      slip: round(w.frictionSlip),
    })),
  }));
  return JSON.stringify({ meta, stats, frames: compact }, null, 2);
}

/** Markdown summary: meta header, per-model aggregate table, controls
 *  range, then a fenced JSON tail with the most recent N frames so a
 *  reader (or agent) can drill in.  */
export function toMarkdown(
  meta: DebugMeta,
  frames: ReadonlyArray<DebugFrame>,
  stats: DebugStats,
  tailFrames = 30,
): string {
  const f = (x: number, n = 3): string => Number.isFinite(x) ? x.toFixed(n) : String(x);
  const lines: string[] = [];
  lines.push('# Sim-to-Real debug snapshot');
  lines.push('');
  lines.push(`- mode: \`${meta.mode}\``);
  lines.push(`- frames: ${stats.ticks} (${f(stats.durationSec, 2)} s @ physicsDt=${meta.physicsDt})`);
  lines.push(`- physics: dt=${meta.physicsDt} × ${meta.physicsSubsteps} sub-steps`);
  lines.push(`- model rollout dt: ${meta.modelDt} (matchSubsteps=${meta.matchSubsteps})`);
  lines.push(`- chassis forces: engine=${meta.engineForceN} N · brake=${meta.brakeForceN} N`);
  lines.push(`- persisted v2 model loaded: ${meta.hasPersistedV2}`);
  lines.push('');
  lines.push('## Per-model gap (over capture window)');
  lines.push('');
  lines.push('| model | mean Δx | mean Δz | rms pos | rms heading (deg) | rms speed | final pos | final hdg (deg) | final spd |');
  lines.push('|---|---:|---:|---:|---:|---:|---:|---:|---:|');
  for (const [id, s] of Object.entries(stats.perModel)) {
    lines.push(
      `| \`${id}\` | ${f(s.meanDx)} | ${f(s.meanDz)} | ${f(s.rmsPos)} | ${f(s.rmsHeading * 180 / Math.PI, 2)} | ${f(s.rmsSpeed)} | ${f(s.finalPosErr)} | ${f(s.finalHeadingErr * 180 / Math.PI, 2)} | ${f(s.finalSpeedErr)} |`,
    );
  }
  lines.push('');
  lines.push('## Controls range observed');
  lines.push('');
  lines.push(`- steer (applied): [${f(stats.controls.steer.min, 3)}, ${f(stats.controls.steer.max, 3)}]`);
  lines.push(`- throttle: [${f(stats.controls.throttle.min, 3)}, ${f(stats.controls.throttle.max, 3)}]`);
  lines.push(`- brake: [${f(stats.controls.brake.min, 3)}, ${f(stats.controls.brake.max, 3)}]`);
  lines.push(`- driveForce (ctrlVec[1], N): [${f(stats.controls.driveForce.min, 1)}, ${f(stats.controls.driveForce.max, 1)}]`);
  lines.push('');
  lines.push(`## Last ${Math.min(tailFrames, frames.length)} frames (JSON)`);
  lines.push('');
  lines.push('```json');
  const tail = frames.slice(-tailFrames);
  lines.push(toJSON(meta, tail, stats));
  lines.push('```');
  return lines.join('\n');
}
