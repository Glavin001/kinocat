// Oriented-bounding-box collision primitives. Aircraft footprint is a box
// rotated by yaw (about world +y) then pitch (about new body +y') then roll
// (about new body +x''). The result is the agent's world-space basis; SAT
// against an AABB and a closest-point check against a sphere live here.
//
// Per-call allocation matters: AirspaceWorld.clear() runs >900k times in the
// canyon bench. The `*Into` and `*Cached` variants below let hot callers
// hold a single scratch OBB / scratch separating-axes table and avoid the
// dozens of Vec3 array allocations the convenient forms make per call.

export type Vec3 = [number, number, number];

export interface Pose {
  x: number;
  y: number;
  z: number;
  yaw: number;
  pitch: number;
  roll: number;
}

export interface OBB {
  center: Vec3;
  /** World-space body axes: [forward, right-wing, up]. Unit vectors. */
  axes: [Vec3, Vec3, Vec3];
  /** Half-extents along each body axis (length, span, height). */
  half: [number, number, number];
}

/** Precomputed separating-axes table for SAT vs world-AABB. The 6 face axes
 *  (3 OBB face normals + 3 world-axis normals) are implicit; this stores the
 *  9 cross-product axes (one per (OBB axis, world axis) pair). For axes too
 *  close to parallel (cross magnitude ~0) the entry is null and skipped. */
export interface OBBSepAxes {
  /** 9 unit cross-product axes (or null for degenerate pairs). */
  cross: (Vec3 | null)[];
}

/** Allocate an OBB scratch with proper-length axes/half so it can be
 *  mutated in place by `poseToOBBInto`. */
export function makeOBB(): OBB {
  return {
    center: [0, 0, 0],
    axes: [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ],
    half: [0, 0, 0],
  };
}

/** Allocate a scratch separating-axes table with 9 cross-axis slots. */
export function makeOBBSepAxes(): OBBSepAxes {
  return {
    cross: [
      [0, 0, 0], [0, 0, 0], [0, 0, 0],
      [0, 0, 0], [0, 0, 0], [0, 0, 0],
      [0, 0, 0], [0, 0, 0], [0, 0, 0],
    ],
  };
}

/** Build the world-space OBB for a pose into a caller-supplied scratch OBB. */
export function poseToOBBInto(
  obb: OBB,
  pose: Pose,
  half: [number, number, number],
): void {
  const ch = Math.cos(pose.yaw);
  const sh = Math.sin(pose.yaw);
  const cp = Math.cos(pose.pitch);
  const sp = Math.sin(pose.pitch);
  const cr = Math.cos(pose.roll);
  const sr = Math.sin(pose.roll);
  // Forward = yaw then pitch applied to world +x.
  const fwdX = cp * ch;
  const fwdY = sp;
  const fwdZ = cp * sh;
  // Pre-roll right wing (horizontal plane, perpendicular to ground track).
  const rightHorizX = sh;
  const rightHorizY = 0;
  const rightHorizZ = -ch;
  // Pre-roll up = forward × rightHoriz (right-handed body frame).
  const upPreX = fwdY * rightHorizZ - fwdZ * rightHorizY;
  const upPreY = fwdZ * rightHorizX - fwdX * rightHorizZ;
  const upPreZ = fwdX * rightHorizY - fwdY * rightHorizX;
  // Roll about the forward axis.
  const rightX = cr * rightHorizX + sr * upPreX;
  const rightY = cr * rightHorizY + sr * upPreY;
  const rightZ = cr * rightHorizZ + sr * upPreZ;
  const upX = -sr * rightHorizX + cr * upPreX;
  const upY = -sr * rightHorizY + cr * upPreY;
  const upZ = -sr * rightHorizZ + cr * upPreZ;

  obb.center[0] = pose.x;
  obb.center[1] = pose.y;
  obb.center[2] = pose.z;
  obb.axes[0][0] = fwdX;
  obb.axes[0][1] = fwdY;
  obb.axes[0][2] = fwdZ;
  obb.axes[1][0] = rightX;
  obb.axes[1][1] = rightY;
  obb.axes[1][2] = rightZ;
  obb.axes[2][0] = upX;
  obb.axes[2][1] = upY;
  obb.axes[2][2] = upZ;
  obb.half[0] = half[0];
  obb.half[1] = half[1];
  obb.half[2] = half[2];
}

/** Build the world-space OBB for a pose with body-frame half-extents.
 *  Allocates — prefer `poseToOBBInto` in hot paths. */
export function poseToOBB(pose: Pose, half: [number, number, number]): OBB {
  const obb = makeOBB();
  poseToOBBInto(obb, pose, half);
  return obb;
}

const AABB_AXES: [Vec3, Vec3, Vec3] = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
];

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

/** Project an OBB onto a unit axis: returns the half-width of the projection. */
function projectOBB(o: OBB, axis: Vec3): number {
  return (
    o.half[0] * Math.abs(dot(o.axes[0], axis)) +
    o.half[1] * Math.abs(dot(o.axes[1], axis)) +
    o.half[2] * Math.abs(dot(o.axes[2], axis))
  );
}

/** Project an OBB onto a world axis (cardinal +x/+y/+z). Faster than
 *  projectOBB when the axis is known to be a coordinate axis (i is 0/1/2). */
function projectOBBOnWorldAxis(o: OBB, i: 0 | 1 | 2): number {
  return (
    o.half[0] * Math.abs(o.axes[0][i]) +
    o.half[1] * Math.abs(o.axes[1][i]) +
    o.half[2] * Math.abs(o.axes[2][i])
  );
}

/** Precompute the 9 cross-product axes for SAT vs world-AABB. Cross-pairs
 *  whose magnitude is ~0 (axes parallel) are set to null and skipped at
 *  test time. */
export function computeOBBSepAxes(obb: OBB, out: OBBSepAxes): void {
  for (let i = 0; i < 3; i++) {
    const ai = obb.axes[i]!;
    for (let j = 0; j < 3; j++) {
      const slot = out.cross[i * 3 + j]!;
      // axis = obb.axes[i] × world.axes[j]
      // For j=0 (world +x): (ai[1]*0 - ai[2]*0, ai[2]*1 - ai[0]*0, ai[0]*0 - ai[1]*1) = (0, ai[2], -ai[1])
      // For j=1 (world +y): (-ai[2], 0, ai[0])
      // For j=2 (world +z): (ai[1], -ai[0], 0)
      let cx: number;
      let cy: number;
      let cz: number;
      if (j === 0) {
        cx = 0;
        cy = ai[2];
        cz = -ai[1];
      } else if (j === 1) {
        cx = -ai[2];
        cy = 0;
        cz = ai[0];
      } else {
        cx = ai[1];
        cy = -ai[0];
        cz = 0;
      }
      const m = Math.sqrt(cx * cx + cy * cy + cz * cz);
      if (m < 1e-9) {
        out.cross[i * 3 + j] = null;
      } else {
        const inv = 1 / m;
        if (slot) {
          slot[0] = cx * inv;
          slot[1] = cy * inv;
          slot[2] = cz * inv;
        } else {
          out.cross[i * 3 + j] = [cx * inv, cy * inv, cz * inv];
        }
      }
    }
  }
}

/** Fast OBB-vs-AABB SAT using a precomputed cross-axes table. */
export function obbHitsAABBCached(
  o: OBB,
  sep: OBBSepAxes,
  min: Vec3,
  max: Vec3,
): boolean {
  const cx = (min[0] + max[0]) * 0.5;
  const cy = (min[1] + max[1]) * 0.5;
  const cz = (min[2] + max[2]) * 0.5;
  const hx = (max[0] - min[0]) * 0.5;
  const hy = (max[1] - min[1]) * 0.5;
  const hz = (max[2] - min[2]) * 0.5;
  const dx = o.center[0] - cx;
  const dy = o.center[1] - cy;
  const dz = o.center[2] - cz;

  // 3 OBB face axes.
  for (let i = 0; i < 3; i++) {
    const a = o.axes[i]!;
    const rA = o.half[i]!;
    const rB = hx * Math.abs(a[0]) + hy * Math.abs(a[1]) + hz * Math.abs(a[2]);
    if (Math.abs(dx * a[0] + dy * a[1] + dz * a[2]) > rA + rB) return false;
  }
  // 3 world face axes (cardinal).
  const rB0 = hx;
  if (Math.abs(dx) > projectOBBOnWorldAxis(o, 0) + rB0) return false;
  const rB1 = hy;
  if (Math.abs(dy) > projectOBBOnWorldAxis(o, 1) + rB1) return false;
  const rB2 = hz;
  if (Math.abs(dz) > projectOBBOnWorldAxis(o, 2) + rB2) return false;
  // 9 cross-product axes (from precomputed table).
  for (let i = 0; i < 9; i++) {
    const axis = sep.cross[i];
    if (!axis) continue;
    const rA = projectOBB(o, axis);
    const rB =
      hx * Math.abs(axis[0]) + hy * Math.abs(axis[1]) + hz * Math.abs(axis[2]);
    if (Math.abs(dx * axis[0] + dy * axis[1] + dz * axis[2]) > rA + rB) return false;
  }
  return true;
}

/** Separating Axis Theorem: OBB vs world-axis-aligned box specified by min/max
 *  corners. Returns true if the volumes overlap.
 *  Allocates internally — prefer `obbHitsAABBCached` in hot paths (callers
 *  that test the same OBB against many boxes). */
export function obbHitsAABB(o: OBB, min: Vec3, max: Vec3): boolean {
  const aabbCenter: Vec3 = [
    (min[0] + max[0]) / 2,
    (min[1] + max[1]) / 2,
    (min[2] + max[2]) / 2,
  ];
  const aabbHalf: Vec3 = [
    (max[0] - min[0]) / 2,
    (max[1] - min[1]) / 2,
    (max[2] - min[2]) / 2,
  ];
  const d: Vec3 = [
    o.center[0] - aabbCenter[0],
    o.center[1] - aabbCenter[1],
    o.center[2] - aabbCenter[2],
  ];
  const axes: Vec3[] = [
    o.axes[0], o.axes[1], o.axes[2],
    AABB_AXES[0], AABB_AXES[1], AABB_AXES[2],
  ];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      const c = cross(o.axes[i]!, AABB_AXES[j]!);
      const m = Math.hypot(c[0], c[1], c[2]);
      if (m > 1e-9) axes.push([c[0] / m, c[1] / m, c[2] / m]);
    }
  }
  for (const a of axes) {
    const rA = projectOBB(o, a);
    const rB =
      aabbHalf[0] * Math.abs(a[0]) +
      aabbHalf[1] * Math.abs(a[1]) +
      aabbHalf[2] * Math.abs(a[2]);
    if (Math.abs(dot(d, a)) > rA + rB) return false;
  }
  return true;
}

/** OBB vs sphere: distance from sphere centre to closest point on the OBB. */
export function obbHitsSphere(o: OBB, c: Vec3, radius: number): boolean {
  const dx = c[0] - o.center[0];
  const dy = c[1] - o.center[1];
  const dz = c[2] - o.center[2];
  let sqDist = 0;
  for (let i = 0; i < 3; i++) {
    const ax = o.axes[i]!;
    const e = dx * ax[0] + dy * ax[1] + dz * ax[2];
    const h = o.half[i]!;
    const clamped = e < -h ? -h : e > h ? h : e;
    const diff = e - clamped;
    sqDist += diff * diff;
  }
  return sqDist < radius * radius;
}

/** OBB vs sphere using a separate centre (no Vec3 alloc at call site). */
export function obbHitsSphereXYZ(
  o: OBB,
  cx: number,
  cy: number,
  cz: number,
  radius: number,
): boolean {
  const dx = cx - o.center[0];
  const dy = cy - o.center[1];
  const dz = cz - o.center[2];
  let sqDist = 0;
  for (let i = 0; i < 3; i++) {
    const ax = o.axes[i]!;
    const e = dx * ax[0] + dy * ax[1] + dz * ax[2];
    const h = o.half[i]!;
    const clamped = e < -h ? -h : e > h ? h : e;
    const diff = e - clamped;
    sqDist += diff * diff;
  }
  return sqDist < radius * radius;
}

/** Conservative world-space AABB enclosing the OBB (for fast altitude tests).
 *  Allocates — prefer `obbWorldExtentInto` in hot paths. */
export function obbWorldExtent(o: OBB): { min: Vec3; max: Vec3 } {
  const ex =
    o.half[0] * Math.abs(o.axes[0][0]) +
    o.half[1] * Math.abs(o.axes[1][0]) +
    o.half[2] * Math.abs(o.axes[2][0]);
  const ey =
    o.half[0] * Math.abs(o.axes[0][1]) +
    o.half[1] * Math.abs(o.axes[1][1]) +
    o.half[2] * Math.abs(o.axes[2][1]);
  const ez =
    o.half[0] * Math.abs(o.axes[0][2]) +
    o.half[1] * Math.abs(o.axes[1][2]) +
    o.half[2] * Math.abs(o.axes[2][2]);
  return {
    min: [o.center[0] - ex, o.center[1] - ey, o.center[2] - ez],
    max: [o.center[0] + ex, o.center[1] + ey, o.center[2] + ez],
  };
}

/** Write the OBB's world-space AABB into caller-supplied min/max scratch. */
export function obbWorldExtentInto(o: OBB, outMin: Vec3, outMax: Vec3): void {
  const ex =
    o.half[0] * Math.abs(o.axes[0][0]) +
    o.half[1] * Math.abs(o.axes[1][0]) +
    o.half[2] * Math.abs(o.axes[2][0]);
  const ey =
    o.half[0] * Math.abs(o.axes[0][1]) +
    o.half[1] * Math.abs(o.axes[1][1]) +
    o.half[2] * Math.abs(o.axes[2][1]);
  const ez =
    o.half[0] * Math.abs(o.axes[0][2]) +
    o.half[1] * Math.abs(o.axes[1][2]) +
    o.half[2] * Math.abs(o.axes[2][2]);
  outMin[0] = o.center[0] - ex;
  outMin[1] = o.center[1] - ey;
  outMin[2] = o.center[2] - ez;
  outMax[0] = o.center[0] + ex;
  outMax[1] = o.center[1] + ey;
  outMax[2] = o.center[2] + ez;
}
