// Oriented-bounding-box collision primitives. Aircraft footprint is a box
// rotated by yaw (about world +y) then pitch (about new body +y') then roll
// (about new body +x''). The result is the agent's world-space basis; SAT
// against an AABB and a closest-point check against a sphere live here.

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

/** Build the world-space OBB for a pose with body-frame half-extents
 *  (halfLength along forward, halfSpan along right-wing, halfHeight along
 *  body-up). Heading 0 = +x world; pitch positive = nose up; roll positive
 *  = right wing down (standard right-handed roll about the forward axis). */
export function poseToOBB(pose: Pose, half: [number, number, number]): OBB {
  const ch = Math.cos(pose.yaw);
  const sh = Math.sin(pose.yaw);
  const cp = Math.cos(pose.pitch);
  const sp = Math.sin(pose.pitch);
  const cr = Math.cos(pose.roll);
  const sr = Math.sin(pose.roll);
  // Forward = yaw then pitch applied to world +x.
  const fwd: Vec3 = [cp * ch, sp, cp * sh];
  // Pre-roll right wing (in horizontal plane, perpendicular to ground track).
  // For heading 0 (+x) the right wing points to world +z (heading + 90°).
  const rightHoriz: Vec3 = [sh, 0, -ch];
  // Pre-roll up = forward × rightHoriz (right-handed body frame).
  const upPre: Vec3 = [
    fwd[1] * rightHoriz[2] - fwd[2] * rightHoriz[1],
    fwd[2] * rightHoriz[0] - fwd[0] * rightHoriz[2],
    fwd[0] * rightHoriz[1] - fwd[1] * rightHoriz[0],
  ];
  // Roll about the forward axis.
  const right: Vec3 = [
    cr * rightHoriz[0] + sr * upPre[0],
    cr * rightHoriz[1] + sr * upPre[1],
    cr * rightHoriz[2] + sr * upPre[2],
  ];
  const up: Vec3 = [
    -sr * rightHoriz[0] + cr * upPre[0],
    -sr * rightHoriz[1] + cr * upPre[1],
    -sr * rightHoriz[2] + cr * upPre[2],
  ];
  return {
    center: [pose.x, pose.y, pose.z],
    axes: [fwd, right, up],
    half,
  };
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

function projectAABBHalf(half: Vec3, axis: Vec3): number {
  return (
    half[0] * Math.abs(axis[0]) +
    half[1] * Math.abs(axis[1]) +
    half[2] * Math.abs(axis[2])
  );
}

/** Separating Axis Theorem: OBB vs world-axis-aligned box specified by min/max
 *  corners. Returns true if the volumes overlap. */
export function obbHitsAABB(
  o: OBB,
  min: Vec3,
  max: Vec3,
): boolean {
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
    const rB = projectAABBHalf(aabbHalf, a);
    if (Math.abs(dot(d, a)) > rA + rB) return false;
  }
  return true;
}

/** OBB vs sphere: distance from sphere centre to closest point on the OBB. */
export function obbHitsSphere(
  o: OBB,
  c: Vec3,
  radius: number,
): boolean {
  const d: Vec3 = [c[0] - o.center[0], c[1] - o.center[1], c[2] - o.center[2]];
  let sqDist = 0;
  for (let i = 0; i < 3; i++) {
    const e = dot(d, o.axes[i]!);
    const clamped = e < -o.half[i]! ? -o.half[i]! : e > o.half[i]! ? o.half[i]! : e;
    const diff = e - clamped;
    sqDist += diff * diff;
  }
  return sqDist < radius * radius;
}

/** Conservative world-space AABB enclosing the OBB (for fast altitude tests). */
export function obbWorldExtent(o: OBB): { min: Vec3; max: Vec3 } {
  const extent = (i: 0 | 1 | 2) =>
    o.half[0] * Math.abs(o.axes[0][i]) +
    o.half[1] * Math.abs(o.axes[1][i]) +
    o.half[2] * Math.abs(o.axes[2][i]);
  const ex = extent(0);
  const ey = extent(1);
  const ez = extent(2);
  return {
    min: [o.center[0] - ex, o.center[1] - ey, o.center[2] - ez],
    max: [o.center[0] + ex, o.center[1] + ey, o.center[2] + ez],
  };
}
