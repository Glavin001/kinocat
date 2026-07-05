// Pure path animation helpers for the GoalLab visualizer. A planner path is a
// sparse sequence of nodes (one per motion primitive, plus a possibly long
// analytic Reeds-Shepp shot-to-goal edge). To animate a car along it smoothly
// AND keep the car oriented along the path it's visibly travelling, we
// interpolate with a heading-aware cubic Hermite spline:
//
//   - position = Hermite(node_i, node_{i+1}, u), tangents pointing along TRAVEL
//     (signed by gear), so a long shot edge bends as a curve, not a straight
//     chord, and reverse (parking) curves the correct way;
//   - heading  = the Hermite TANGENT direction (flipped 180° for reverse), so
//     the car nose is always aligned with the drawn path — independent of how
//     far a plan node's stored heading drifts from travel on the shot edge.

export interface PathSample {
  x: number;
  z: number;
  heading: number;
  speed: number;
  t: number;
}

function tangents(a: PathSample, c: PathSample) {
  const chord = Math.hypot(c.x - a.x, c.z - a.z) || 1;
  const sa = Math.sign(a.speed) || 1;
  const sc = Math.sign(c.speed) || 1;
  return {
    m0x: Math.cos(a.heading) * sa * chord,
    m0z: Math.sin(a.heading) * sa * chord,
    m1x: Math.cos(c.heading) * sc * chord,
    m1z: Math.sin(c.heading) * sc * chord,
  };
}

export function hermitePoint(a: PathSample, c: PathSample, u: number): { x: number; z: number } {
  const { m0x, m0z, m1x, m1z } = tangents(a, c);
  const u2 = u * u;
  const u3 = u2 * u;
  const h00 = 2 * u3 - 3 * u2 + 1;
  const h10 = u3 - 2 * u2 + u;
  const h01 = -2 * u3 + 3 * u2;
  const h11 = u3 - u2;
  return {
    x: h00 * a.x + h10 * m0x + h01 * c.x + h11 * m1x,
    z: h00 * a.z + h10 * m0z + h01 * c.z + h11 * m1z,
  };
}

/** Heading from the Hermite curve's tangent at `u` (the actual travel direction
 *  along the drawn path), flipped 180° for reverse. */
export function hermiteHeading(a: PathSample, c: PathSample, u: number): number {
  const { m0x, m0z, m1x, m1z } = tangents(a, c);
  const u2 = u * u;
  const d00 = 6 * u2 - 6 * u;
  const d10 = 3 * u2 - 4 * u + 1;
  const d01 = -6 * u2 + 6 * u;
  const d11 = 3 * u2 - 2 * u;
  const tx = d00 * a.x + d10 * m0x + d01 * c.x + d11 * m1x;
  const tz = d00 * a.z + d10 * m0z + d01 * c.z + d11 * m1z;
  if (Math.hypot(tx, tz) < 1e-6) return a.heading; // degenerate cusp
  const travel = Math.atan2(tz, tx);
  return a.speed < 0 ? travel + Math.PI : travel;
}

/** Sample the animated car pose at simulation time `tSim`. */
export function hermitePose(path: ReadonlyArray<PathSample>, tSim: number): PathSample {
  if (path.length === 0) return { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
  if (tSim <= path[0]!.t) return path[0]!;
  for (let i = 0; i + 1 < path.length; i++) {
    const a = path[i]!;
    const c = path[i + 1]!;
    if (tSim >= a.t && tSim <= c.t) {
      const u = c.t > a.t ? (tSim - a.t) / (c.t - a.t) : 0;
      const p = hermitePoint(a, c, u);
      return { x: p.x, z: p.z, heading: hermiteHeading(a, c, u), speed: a.speed, t: tSim };
    }
  }
  return path[path.length - 1]!;
}

/** A dense, smooth polyline of the path (for the green plan overlay). */
export function densifyPath(
  path: ReadonlyArray<PathSample>,
  stepsPerSegment = 10,
): PathSample[] {
  if (path.length < 2) return path.slice();
  const out: PathSample[] = [];
  for (let i = 0; i + 1 < path.length; i++) {
    const a = path[i]!;
    const c = path[i + 1]!;
    for (let s = 0; s < stepsPerSegment; s++) {
      const u = s / stepsPerSegment;
      const p = hermitePoint(a, c, u);
      out.push({ x: p.x, z: p.z, heading: a.heading, speed: a.speed, t: a.t });
    }
  }
  out.push(path[path.length - 1]!);
  return out;
}
