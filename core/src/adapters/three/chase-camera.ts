// Chase-cam follow helper for car demos. Identical math copied verbatim
// across /ramp, /obstaclecourse, /carchase, and /raceprimitives — extract
// once, use everywhere.
//
// Three.js is an optional peer; this module is loaded only when the demo
// imports `kinocat/adapters/three`.

import type * as THREE from 'three';

export interface ChaseCameraTarget {
  x: number;
  z: number;
  heading: number;
  /** Optional Y coordinate of the chassis for the look-at target. Default 1.5. */
  y?: number;
}

export interface ChaseCameraOpts {
  /** Distance behind the chassis (world units). Default 14. */
  distance?: number;
  /** Camera height above the ground (world units). Default 7. */
  height?: number;
  /** Lerp coefficient toward the desired position (0..1). Default 0.12. */
  smoothing?: number;
  /** OrbitControls instance (or anything with `target` + `update()`).
   *  When supplied, its target is set to the chassis pose so user-driven
   *  rotation pivots around the car. */
  orbit?: { target: THREE.Vector3; update: () => void };
}

/** One tick of chase-cam follow. Lerps `camera.position` toward a point
 *  `distance` behind the chassis at `height`, and (when supplied) snaps
 *  `orbit.target` onto the chassis. */
export function updateChaseCamera(
  camera: THREE.PerspectiveCamera | THREE.Camera,
  target: ChaseCameraTarget,
  opts: ChaseCameraOpts = {},
): void {
  const distance = opts.distance ?? 14;
  const height = opts.height ?? 7;
  const smoothing = opts.smoothing ?? 0.12;
  const cc = Math.cos(target.heading);
  const ss = Math.sin(target.heading);
  const tx = target.x - distance * cc;
  const tz = target.z - distance * ss;
  // Lerp position toward the chase point.
  const p = (camera as THREE.PerspectiveCamera).position;
  p.x += (tx - p.x) * smoothing;
  p.y += (height - p.y) * smoothing;
  p.z += (tz - p.z) * smoothing;
  if (opts.orbit) {
    opts.orbit.target.set(target.x, target.y ?? 1.5, target.z);
    opts.orbit.update();
  }
}
