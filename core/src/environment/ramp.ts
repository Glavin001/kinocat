// Drivable ramp primitives shared by every 3D car demo.
//
// The "ramp" is a continuous height function, not a cuboid wall: the car
// physically climbs the up-slope, launches off the lip, and falls under
// gravity. Demos add a `BallisticJump` Affordance at the crest so the
// planner can take the arc as a shortcut, but execution is always real
// raycast-vehicle physics — the car always drives.
//
// What's reusable here:
//   - `RampSpec`: pose + dimensions of a single ramp on the XZ plane.
//   - `rampHeightSampler(ramps)`: a `HeightSampler` you can pass straight
//     to `createHeightfieldCollider` (physics) AND to a displaced
//     `PlaneGeometry` (visual mesh) so physics + visual agree.
//   - `combineHeightSamplers(...)`: max() blend of multiple samplers
//     (terrain + ramps).
//   - `jumpSpecFromRamp(ramp)`: derives the launch + ballistic landing
//     pose for a `BallisticJump` Affordance from the ramp geometry.
//
// What lives in the demo, not here: the planner-only "gap" obstacle and
// affordance cost tuning. Those are per-scenario.

export type HeightSampler = (x: number, z: number) => number;

export interface RampSpec {
  id: string;
  /** Ramp base centre on the XZ plane (centre of the up-slope footprint). */
  base: { x: number; z: number };
  /** Up-slope length along `heading`. */
  length: number;
  /** Lateral width perpendicular to `heading`. */
  width: number;
  /** Crest height (world Y). */
  height: number;
  /** Forward direction of the slope (radians, 0 = +X). The slope rises
   *  from `base - (length/2) * heading_dir` (foot) to
   *  `base + (length/2) * heading_dir` (crest), then continues to a steep
   *  back-slope past the crest. */
  heading: number;
  /** Steep back-slope length past the crest (default 2.5 m). Keeps the
   *  back face continuous instead of a vertical cliff — vertical
   *  triangles in the heightfield mesh intermittently WASM-trap Rapier's
   *  wheel raycaster. Still steep enough that the car launches. */
  backSkirt?: number;
  /** Lateral skirt length on each side (default 1.5 m). Wedges the
   *  height down at the sides for the same reason as `backSkirt`. */
  lateralSkirt?: number;
}

/** Build a `HeightSampler` that returns the maximum ramp height at
 *  `(x, z)`, or 0 if `(x, z)` is outside every ramp footprint. */
export function rampHeightSampler(
  ramps: ReadonlyArray<RampSpec>,
): HeightSampler {
  return (x, z) => {
    let y = 0;
    for (const r of ramps) {
      const c = Math.cos(r.heading);
      const s = Math.sin(r.heading);
      const dx = x - r.base.x;
      const dz = z - r.base.z;
      const along = dx * c + dz * s;
      const lateral = -dx * s + dz * c;
      const halfL = r.length / 2;
      const halfW = r.width / 2;
      const lateralSkirt = r.lateralSkirt ?? 1.5;
      const backSkirt = r.backSkirt ?? 2.5;
      const lateralInset = halfW - Math.abs(lateral);
      if (lateralInset <= 0) continue;
      const lateralScale = Math.min(1, lateralInset / lateralSkirt);
      if (along < -halfL) continue;
      if (along > halfL + backSkirt) continue;
      let alongH: number;
      if (along <= halfL) {
        const u = (along + halfL) / r.length;
        alongH = r.height * u;
      } else {
        const u = (along - halfL) / backSkirt;
        alongH = r.height * (1 - u);
      }
      const h = alongH * lateralScale;
      if (h > y) y = h;
    }
    return y;
  };
}

/** Max-blend several samplers (terrain + ramps). The result returns the
 *  highest sampled height at each `(x, z)`. */
export function combineHeightSamplers(
  ...samplers: ReadonlyArray<HeightSampler>
): HeightSampler {
  if (samplers.length === 1) return samplers[0]!;
  return (x, z) => {
    let y = -Infinity;
    for (const s of samplers) {
      const v = s(x, z);
      if (v > y) y = v;
    }
    return y;
  };
}

export interface RampJumpSpec {
  id: string;
  /** Launch point — top of the ramp crest. */
  launch: { x: number; z: number };
  /** Landing pose on the far side of the gap. */
  land: { x: number; z: number; heading: number };
  /** Crest height in world Y (handy for arc helpers). */
  height: number;
  /** Approach heading (radians, +X = 0). */
  heading: number;
}

export interface JumpSpecFromRampOptions {
  /** Override the launch→land distance directly. If unset, the helper
   *  estimates ballistic range from `cruiseSpeed`, the ramp's lip slope,
   *  and `gravity`. */
  launchDist?: number;
  /** Cruise speed used to estimate ballistic range. Default 12 m/s. */
  cruiseSpeed?: number;
  /** Gravity magnitude. Default 9.81. */
  gravity?: number;
  /** Override the launch→land distance with this minimum value. */
  minLaunchDist?: number;
}

/** Derive a launch/land pair for a `BallisticJump` Affordance from a
 *  ramp's geometry. Defaults estimate the ballistic range for a car
 *  cruising up the ramp at `cruiseSpeed` and lifting off the lip at the
 *  ramp's slope angle. */
export function jumpSpecFromRamp(
  ramp: RampSpec,
  opts: JumpSpecFromRampOptions = {},
): RampJumpSpec {
  const c = Math.cos(ramp.heading);
  const s = Math.sin(ramp.heading);
  const crestX = ramp.base.x + (ramp.length / 2) * c;
  const crestZ = ramp.base.z + (ramp.length / 2) * s;
  const cruiseSpeed = opts.cruiseSpeed ?? 12;
  const g = opts.gravity ?? 9.81;
  const slope = Math.atan2(ramp.height, ramp.length);
  const vx = cruiseSpeed * Math.cos(slope);
  const vy = cruiseSpeed * Math.sin(slope);
  const ballistic = (vx * (vy + Math.sqrt(vy * vy + 2 * g * ramp.height))) / g;
  const launchDist = Math.max(
    opts.minLaunchDist ?? 0,
    opts.launchDist ?? ballistic,
  );
  return {
    id: `${ramp.id}-jump`,
    launch: { x: crestX, z: crestZ },
    land: {
      x: crestX + launchDist * c,
      z: crestZ + launchDist * s,
      heading: ramp.heading,
    },
    height: ramp.height,
    heading: ramp.heading,
  };
}
