// Zero-dependency planar math helpers. Planning plane is XZ (navcat/OpenGL,
// right-handed, +Y up); Y is derived from polygon containment, never here.

export const TWO_PI = Math.PI * 2;
export const HALF_PI = Math.PI / 2;

/** Wrap an angle (radians) to (-pi, pi]. */
export function wrapAngle(a: number): number {
  let x = (a + Math.PI) % TWO_PI;
  if (x <= 0) x += TWO_PI;
  return x - Math.PI;
}

/** Smallest signed delta from `a` to `b`, in (-pi, pi]. */
export function angleDiff(a: number, b: number): number {
  return wrapAngle(b - a);
}

export function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/** Linear interpolation between two angles along the shortest arc. */
export function lerpAngle(a: number, b: number, t: number): number {
  return wrapAngle(a + angleDiff(a, b) * t);
}

/** Squared planar distance (XZ). Avoids a sqrt when only comparing. */
export function dist2(ax: number, az: number, bx: number, bz: number): number {
  const dx = ax - bx;
  const dz = az - bz;
  return dx * dx + dz * dz;
}

export function dist(ax: number, az: number, bx: number, bz: number): number {
  return Math.sqrt(dist2(ax, az, bx, bz));
}

/** Quantize a value to an integer bucket index given a cell size. */
export function quantize(value: number, cell: number): number {
  return Math.floor(value / cell);
}

/** True if scalar `x` is finite (not NaN/Infinity). */
export function isFiniteNumber(x: number): boolean {
  return Number.isFinite(x);
}

export function approxEqual(a: number, b: number, eps = 1e-9): boolean {
  return Math.abs(a - b) <= eps;
}
