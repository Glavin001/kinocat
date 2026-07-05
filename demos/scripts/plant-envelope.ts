// WS-0 — Plant-envelope characterization.
//
// Measures the TRUE limits of the Rapier raycast-vehicle plant (the race
// tuning) on the headless trial harness, instead of trusting hand-set
// constants or the analytic `deriveVehicleCapabilities` (which underestimates
// the measured brake ~2×). Emits `demos/public/models/plant-envelope.json`
// plus a console table. Deterministic: no RNG, fixed spawn, fixed control
// traces.
//
// Usage: pnpm tsx demos/scripts/plant-envelope.ts [--out=path]

import { writeFileSync } from 'node:fs';
import { resolve, isAbsolute } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { createHeadlessTrialHarness } from 'kinocat/adapters/rapier';
import type { WheeledCarControls } from 'kinocat/agent';
import { DEFAULT_VEHICLE_OPTS } from '../app/lib/training-driver';

const PHYSICS_DT = 1 / 60;

/** Measured plant envelope. All quantities are m, s, m/s, m/s², rad/s. */
export interface PlantEnvelope {
  /** Provenance: the vehicle options this envelope was measured from. */
  vehicle: typeof DEFAULT_VEHICLE_OPTS;
  /** Terminal speed reached under full drive force in a 12 s launch (m/s).
   *  NOTE: Rapier models no aerodynamic drag, so accel stays positive
   *  (~11 m/s² even at 28 m/s — see `launchCurve`); this is NOT an intrinsic
   *  ceiling. Real-world "top speed" here is bounded only by track length and
   *  the planner's policy `RACE_AGENT.maxSpeed`, not by the plant. */
  vMax: number;
  /** Longitudinal accel a(v) at full drive force, sampled on the launch. */
  launchCurve: Array<{ v: number; a: number }>;
  /** Best sustained braking deceleration (m/s²) from each entry speed,
   *  over a sweep of brake-force fractions, excluding lockup-yaw runs. */
  brakeDecel: Array<{ entrySpeed: number; decel: number; brakeFrac: number }>;
  /** Steady-state cornering: sustained lateral accel per (steer, entry v). */
  corneringBoundary: Array<{
    steer: number;
    entrySpeed: number;
    sustainedSpeed: number;
    yawRate: number;
    radius: number;
    latAccel: number;
  }>;
  /** Max sustained lateral accel observed across the whole corner grid. */
  maxLateralAccel: number;
  /** Tightest sustained radius observed (executed min turn radius). */
  minTurnRadiusExecuted: number;
}

function trace(n: number, c: WheeledCarControls): WheeledCarControls[] {
  return Array.from({ length: n }, () => ({ ...c }));
}

/** Measure the plant envelope by driving fixed control traces on the headless
 *  Rapier harness. Deterministic (no RNG). Exported so the regression test can
 *  re-run it and detect plant-tuning drift. */
export async function measurePlantEnvelope(): Promise<PlantEnvelope> {
  const harness = await createHeadlessTrialHarness({
    vehicleOptions: DEFAULT_VEHICLE_OPTS,
    groundFriction: 1.5,
    groundBounds: { x0: -2000, x1: 2000, z0: -2000, z1: 2000 },
    offArenaThreshold: 5000,
  });
  const drv = DEFAULT_VEHICLE_OPTS.engineForce;
  const brk = DEFAULT_VEHICLE_OPTS.brakeForce;
  const maxSteer = DEFAULT_VEHICLE_OPTS.maxSteerAngle;

  // --- vMax + launch curve ------------------------------------------------
  // Full drive force from rest for 12 s; vMax = terminal speed, launch curve
  // = a(v) sampled from the speed trace.
  const launchTicks = Math.round(12 / PHYSICS_DT);
  const launch = harness.runTrial({
    pose: { x: 0, z: 0, heading: 0 },
    kin: { forwardSpeed: 0 },
    controlsTrace: trace(launchTicks, { steer: 0, driveForce: drv, brakeForce: 0 }),
    sampleEveryNTicks: 1,
    id: 'launch',
  });
  if (!launch.ok) throw new Error(`launch trial failed: ${launch.reason}`);
  const s = launch.trial.samples;
  let vMax = 0;
  for (const st of s) vMax = Math.max(vMax, st.speed);
  // a(v) at target speeds via finite difference over a small window.
  const launchCurve: PlantEnvelope['launchCurve'] = [];
  for (const target of [0, 4, 8, 12, 16, 20, 24, 28]) {
    // find first sample at/after target speed
    let idx = s.findIndex((st) => st.speed >= target);
    if (idx < 1) idx = target === 0 ? 1 : -1;
    if (idx < 1 || idx >= s.length - 1) continue;
    const a = (s[idx + 1]!.speed - s[idx - 1]!.speed) / (2 * PHYSICS_DT);
    launchCurve.push({ v: target, a: round(a) });
  }

  // --- brake decel --------------------------------------------------------
  // From each entry speed: accelerate to it, then apply a brake-force
  // fraction and measure the deceleration over the first 0.5 s, discarding
  // runs whose heading deviates > 5° (lockup-induced yaw).
  const brakeDecel: PlantEnvelope['brakeDecel'] = [];
  for (const entrySpeed of [8, 16, 24, 28]) {
    let best = { decel: 0, brakeFrac: 0 };
    for (const frac of [0.25, 0.5, 0.75, 1.0]) {
      const measTicks = Math.round(0.5 / PHYSICS_DT);
      const r = harness.runTrial({
        pose: { x: 0, z: 0, heading: 0 },
        kin: { forwardSpeed: entrySpeed },
        controlsTrace: trace(measTicks, { steer: 0, driveForce: 0, brakeForce: brk * frac }),
        sampleEveryNTicks: 1,
        id: `brake-${entrySpeed}-${frac}`,
      });
      if (!r.ok) continue;
      const bs = r.trial.samples;
      const v0 = bs[0]!.speed;
      const v1 = bs[bs.length - 1]!.speed;
      const headingDev = Math.abs(bs[bs.length - 1]!.heading - bs[0]!.heading);
      if (headingDev > (5 * Math.PI) / 180) continue;
      const decel = (v0 - v1) / ((bs.length - 1) * PHYSICS_DT);
      if (decel > best.decel) best = { decel: round(decel), brakeFrac: frac };
    }
    brakeDecel.push({ entrySpeed, ...best });
  }

  // --- cornering boundary -------------------------------------------------
  // Hold a fixed steer at a given entry speed for 3 s; the steady state (last
  // 1 s) gives sustained yaw rate, radius, and lateral accel.
  const corneringBoundary: PlantEnvelope['corneringBoundary'] = [];
  let maxLat = 0;
  let minRadius = Infinity;
  for (const steer of [0.15, 0.3, 0.45, 0.6].map((f) => f === 0.6 ? maxSteer : maxSteer * (f / 0.6))) {
    for (const entrySpeed of [8, 12, 16, 20, 24, 28]) {
      const holdTicks = Math.round(3 / PHYSICS_DT);
      const r = harness.runTrial({
        pose: { x: 0, z: 0, heading: 0 },
        kin: { forwardSpeed: entrySpeed },
        // Hold entry speed with a mild drive term so the corner is sustained
        // rather than decaying to a stop; brake off.
        controlsTrace: trace(holdTicks, { steer, driveForce: drv * 0.35, brakeForce: 0 }),
        sampleEveryNTicks: 1,
        id: `corner-${round(steer)}-${entrySpeed}`,
      });
      if (!r.ok) continue;
      const cs = r.trial.samples;
      const tail = cs.slice(Math.floor(cs.length * 0.66));
      const yawRate = mean(tail.map((st) => Math.abs(st.yawRate ?? 0)));
      const sustainedSpeed = mean(tail.map((st) => Math.abs(st.speed)));
      if (yawRate < 1e-3) continue;
      const radius = sustainedSpeed / yawRate;
      const latAccel = sustainedSpeed * yawRate;
      corneringBoundary.push({
        steer: round(steer), entrySpeed,
        sustainedSpeed: round(sustainedSpeed), yawRate: round(yawRate),
        radius: round(radius), latAccel: round(latAccel),
      });
      if (latAccel > maxLat) maxLat = latAccel;
      if (radius < minRadius && radius > 0.5) minRadius = radius;
    }
  }

  harness.dispose();

  return {
    vehicle: DEFAULT_VEHICLE_OPTS,
    vMax: round(vMax),
    launchCurve,
    brakeDecel,
    corneringBoundary,
    maxLateralAccel: round(maxLat),
    minTurnRadiusExecuted: round(minRadius),
  };
}

async function main(): Promise<void> {
  const { values } = parseArgs({
    options: { out: { type: 'string' } },
  });
  const repoRoot = resolve(fileURLToPath(new URL('../..', import.meta.url)));
  const outPath = values.out
    ? isAbsolute(values.out) ? values.out : resolve(repoRoot, values.out)
    : resolve(repoRoot, 'demos/public/models/plant-envelope.json');

  const envelope = await measurePlantEnvelope();
  const { launchCurve, brakeDecel, corneringBoundary } = envelope;

  writeFileSync(outPath, JSON.stringify(envelope, null, 2) + '\n');

  // Console table.
  process.stdout.write(`\nPlant envelope (race tuning) — written to ${outPath}\n\n`);
  process.stdout.write(`vMax = ${envelope.vMax} m/s\n`);
  process.stdout.write(`maxLateralAccel = ${envelope.maxLateralAccel} m/s²\n`);
  process.stdout.write(`minTurnRadiusExecuted = ${envelope.minTurnRadiusExecuted} m\n\n`);
  process.stdout.write('launch a(v):  ' + launchCurve.map((p) => `v${p.v}:${p.a}`).join('  ') + '\n');
  process.stdout.write('brake decel:  ' + brakeDecel.map((p) => `${p.entrySpeed}m/s:${p.decel}(@${p.brakeFrac})`).join('  ') + '\n');
  process.stdout.write('\ncornering (steer × entry → sustained latAccel):\n');
  for (const c of corneringBoundary) {
    process.stdout.write(`  steer=${c.steer} v=${c.entrySpeed} → v_ss=${c.sustainedSpeed} R=${c.radius} aLat=${c.latAccel}\n`);
  }
}

function mean(xs: number[]): number {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
}
function round(x: number): number {
  return Math.round(x * 1000) / 1000;
}

// Only run as a script when invoked directly (not when imported by the test).
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((e) => {
    process.stderr.write(String(e?.stack ?? e) + '\n');
    process.exit(1);
  });
}
