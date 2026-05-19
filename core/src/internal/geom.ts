// Zero-dependency 2D geometry on the planning plane (XZ). Polygons are arrays
// of [x, z] vertices; winding-agnostic.

export type Pt = readonly [number, number];

/** Ray-casting point-in-polygon (boundary counts as inside-ish). */
export function pointInPolygon(px: number, pz: number, poly: ReadonlyArray<Pt>): boolean {
  let inside = false;
  const n = poly.length;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = poly[i]![0];
    const zi = poly[i]![1];
    const xj = poly[j]![0];
    const zj = poly[j]![1];
    const intersect =
      zi > pz !== zj > pz &&
      px < ((xj - xi) * (pz - zi)) / (zj - zi) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

function orient(ax: number, az: number, bx: number, bz: number, cx: number, cz: number): number {
  return (bx - ax) * (cz - az) - (bz - az) * (cx - ax);
}

function onSeg(ax: number, az: number, bx: number, bz: number, px: number, pz: number): boolean {
  return (
    Math.min(ax, bx) - 1e-12 <= px &&
    px <= Math.max(ax, bx) + 1e-12 &&
    Math.min(az, bz) - 1e-12 <= pz &&
    pz <= Math.max(az, bz) + 1e-12
  );
}

/** Proper-or-improper segment intersection test. */
export function segmentsIntersect(
  ax: number, az: number, bx: number, bz: number,
  cx: number, cz: number, dx: number, dz: number,
): boolean {
  const d1 = orient(cx, cz, dx, dz, ax, az);
  const d2 = orient(cx, cz, dx, dz, bx, bz);
  const d3 = orient(ax, az, bx, bz, cx, cz);
  const d4 = orient(ax, az, bx, bz, dx, dz);
  if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
      ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) return true;
  if (Math.abs(d1) < 1e-12 && onSeg(cx, cz, dx, dz, ax, az)) return true;
  if (Math.abs(d2) < 1e-12 && onSeg(cx, cz, dx, dz, bx, bz)) return true;
  if (Math.abs(d3) < 1e-12 && onSeg(ax, az, bx, bz, cx, cz)) return true;
  if (Math.abs(d4) < 1e-12 && onSeg(ax, az, bx, bz, dx, dz)) return true;
  return false;
}

/** Convex/concave polygon overlap: edge crossing OR containment either way. */
export function polygonsIntersect(a: ReadonlyArray<Pt>, b: ReadonlyArray<Pt>): boolean {
  for (let i = 0; i < a.length; i++) {
    const a0 = a[i]!;
    const a1 = a[(i + 1) % a.length]!;
    for (let j = 0; j < b.length; j++) {
      const b0 = b[j]!;
      const b1 = b[(j + 1) % b.length]!;
      if (segmentsIntersect(a0[0], a0[1], a1[0], a1[1], b0[0], b0[1], b1[0], b1[1])) {
        return true;
      }
    }
  }
  if (pointInPolygon(a[0]![0], a[0]![1], b)) return true;
  if (pointInPolygon(b[0]![0], b[0]![1], a)) return true;
  return false;
}

/** Rotate `local` ([x,z] in a frame with heading 0 = +x) by `heading`, then
 *  translate to (ox, oz). Returns world-space polygon. */
export function placeFootprint(
  local: ReadonlyArray<Pt>,
  ox: number,
  oz: number,
  heading: number,
): Pt[] {
  const c = Math.cos(heading);
  const s = Math.sin(heading);
  return local.map(([lx, lz]) => [ox + lx * c - lz * s, oz + lx * s + lz * c] as Pt);
}
