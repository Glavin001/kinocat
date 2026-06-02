// Region visualization helpers for the Scenario & Goal Spec Layer. Given a
// `Region`, build a THREE object that deterministically depicts its shape, so a
// goal/invariant overlay can be drawn directly from the canonical scenario.
// Geometry is recovered from the region's serializable `key` (+ `representative`
// for dynamic regions evaluated at a given time), so this stays a pure factory
// with no coupling to the scenario internals.

import * as THREE from 'three';
import type { Region, ScenarioState } from '../../scenario/types';

export interface RegionHelperOptions {
  /** Draw height (world Y). */
  y?: number;
  /** Line / fill color. */
  color?: number;
  /** Time to evaluate dynamic regions at (defaults to 0). */
  t?: number;
}

/** Suggested color by the plane a region belongs to. Objective = cyan,
 *  avoid = red, soft/condition = amber. */
export const REGION_COLORS = {
  objective: 0x44ddff,
  avoid: 0xff4466,
  soft: 0xffaa33,
} as const;

function nums(key: string, prefix: string): number[] {
  return key
    .slice(prefix.length)
    .split(/[,;]/)
    .map(Number)
    .filter((n) => Number.isFinite(n));
}

function lineLoop(points: THREE.Vector3[], color: number): THREE.LineLoop {
  const geo = new THREE.BufferGeometry().setFromPoints(points);
  return new THREE.LineLoop(geo, new THREE.LineBasicMaterial({ color }));
}
function line(points: THREE.Vector3[], color: number): THREE.Line {
  const geo = new THREE.BufferGeometry().setFromPoints(points);
  return new THREE.Line(geo, new THREE.LineBasicMaterial({ color }));
}

function ringPoints(cx: number, cz: number, r: number, y: number, seg = 48): THREE.Vector3[] {
  const pts: THREE.Vector3[] = [];
  for (let i = 0; i < seg; i++) {
    const a = (i / seg) * Math.PI * 2;
    pts.push(new THREE.Vector3(cx + Math.cos(a) * r, y, cz + Math.sin(a) * r));
  }
  return pts;
}

/** Build a THREE object depicting `region`. Returns a Group so callers can add
 *  it to the scene and dispose uniformly. Unknown region kinds fall back to a
 *  small ring at the region's representative pose. */
export function createRegionHelper(region: Region, opts: RegionHelperOptions = {}): THREE.Group {
  const y = opts.y ?? 0.06;
  const color = opts.color ?? REGION_COLORS.objective;
  const group = new THREE.Group();
  const k = region.key;

  if (region.kind === 'near' || region.kind === 'within' || region.kind === 'ahead' ||
      region.kind === 'behind' || region.kind === 'beside') {
    // Ball: center from representative (works for static + dynamic), radius from key.
    const rep = region.dynamic ? region.representative() : null;
    let cx: number, cz: number, r: number;
    if (region.kind === 'near') {
      const [x, z, rad] = nums(k, 'near:');
      cx = x!; cz = z!; r = rad!;
    } else {
      const p = rep ?? region.representative();
      cx = p.x; cz = p.z;
      const parts = k.split(',');
      r = Number(parts[1]) || 1;
    }
    group.add(lineLoop(ringPoints(cx, cz, r, y), color));
    return group;
  }

  if (region.kind === 'at') {
    const [x, z, heading, dx, dz] = nums(k, 'at:');
    const box: THREE.Vector3[] = [];
    const corners: [number, number][] = [
      [dx!, dz!], [-dx!, dz!], [-dx!, -dz!], [dx!, -dz!],
    ];
    const c = Math.cos(heading!);
    const s = Math.sin(heading!);
    for (const [lx, lz] of corners) {
      box.push(new THREE.Vector3(x! + lx * c - lz * s, y, z! + lx * s + lz * c));
    }
    group.add(lineLoop(box, color));
    // heading tick
    group.add(line(
      [new THREE.Vector3(x!, y, z!), new THREE.Vector3(x! + c * (dx! + 0.5), y, z! + s * (dx! + 0.5))],
      color,
    ));
    return group;
  }

  if (region.kind === 'inside') {
    const flat = nums(k, 'inside:');
    const pts: THREE.Vector3[] = [];
    for (let i = 0; i + 1 < flat.length; i += 2) {
      pts.push(new THREE.Vector3(flat[i]!, y, flat[i + 1]!));
    }
    group.add(lineLoop(pts, color));
    return group;
  }

  if (region.kind === 'gate') {
    const [ax, az, bx, bz] = nums(k, 'gate:');
    group.add(line([new THREE.Vector3(ax!, y, az!), new THREE.Vector3(bx!, y, bz!)], color));
    // arrow along the forward normal at the midpoint
    const rep = region.representative();
    const mx = (ax! + bx!) / 2;
    const mz = (az! + bz!) / 2;
    group.add(line(
      [new THREE.Vector3(mx, y, mz), new THREE.Vector3(mx + Math.cos(rep.heading) * 2, y, mz + Math.sin(rep.heading) * 2)],
      color,
    ));
    return group;
  }

  if (region.kind === 'corridor') {
    // key: corridor:width:x0,z0;x1,z1;...
    const widthPart = k.slice('corridor:'.length).split(':')[0]!;
    const width = Number(widthPart);
    const rest = k.slice(`corridor:${widthPart}:`.length);
    const flat = rest.split(/[,;]/).map(Number).filter((n) => Number.isFinite(n));
    const center: THREE.Vector3[] = [];
    for (let i = 0; i + 1 < flat.length; i += 2) {
      center.push(new THREE.Vector3(flat[i]!, y, flat[i + 1]!));
    }
    group.add(line(center, color));
    // offset edges (approximate, segment-normal based)
    for (const sign of [1, -1]) {
      const edge: THREE.Vector3[] = [];
      for (let i = 0; i < center.length; i++) {
        const a = center[Math.max(0, i - 1)]!;
        const b = center[Math.min(center.length - 1, i + 1)]!;
        const dx = b.x - a.x;
        const dz = b.z - a.z;
        const len = Math.hypot(dx, dz) || 1;
        edge.push(new THREE.Vector3(
          center[i]!.x + (-dz / len) * (width / 2) * sign,
          y,
          center[i]!.z + (dx / len) * (width / 2) * sign,
        ));
      }
      group.add(line(edge, color));
    }
    return group;
  }

  if (region.kind === 'cone') {
    const apex = region.representative(); // mid-range pose; back it up to the agent
    const parts = k.split(',');
    const fov = Number(parts[1]) || Math.PI / 6;
    const range = Number(parts[2]) || 10;
    // representative is at range*0.5 ahead; recover apex by stepping back.
    const ax = apex.x - Math.cos(apex.heading) * range * 0.5;
    const az = apex.z - Math.sin(apex.heading) * range * 0.5;
    const left = apex.heading + fov;
    const right = apex.heading - fov;
    group.add(line([
      new THREE.Vector3(ax + Math.cos(left) * range, y, az + Math.sin(left) * range),
      new THREE.Vector3(ax, y, az),
      new THREE.Vector3(ax + Math.cos(right) * range, y, az + Math.sin(right) * range),
    ], color));
    return group;
  }

  // Fallback: a ring at the representative pose.
  const rep = region.representative();
  group.add(lineLoop(ringPoints(rep.x, rep.z, 1, y), color));
  return group;
}
