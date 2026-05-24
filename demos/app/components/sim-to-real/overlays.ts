// Three.js overlay helpers for the /sim-to-real scope. All are pure
// factory functions returning a THREE.Object3D (+ a small API) so they
// can be added/removed from the scene imperatively, like the rest of
// the kinocat 3D adapters.
//
// Each overlay owns its own buffers and `update(...)` reuses them when
// possible — we tick at 60 Hz with three model ghosts + their trails,
// so allocating fresh BufferGeometries every frame is a non-starter.

import * as THREE from 'three';
import type { CarKinematicState } from 'kinocat/agent';
import { createCarMeshHelper, syncCarMesh } from 'kinocat/adapters/three';
import type { WheelTelemetry } from 'kinocat/adapters/rapier';
import { speedToColor } from '../../lib/sim-to-real-scene';

// ---------------------------------------------------------------------------
// GhostCar — semi-transparent car mesh posed each frame to a predicted
// CarKinematicState. One per model.

export interface GhostCar {
  group: THREE.Group;
  setPose(s: CarKinematicState): void;
  setVisible(v: boolean): void;
  dispose(scene: THREE.Scene): void;
}

export function createGhostCar(color: number, opacity = 0.45): GhostCar {
  const car = createCarMeshHelper({ color });
  car.group.traverse((o) => {
    const m = (o as THREE.Mesh).material as THREE.Material | undefined;
    if (m && 'transparent' in m) {
      (m as THREE.MeshStandardMaterial).transparent = true;
      (m as THREE.MeshStandardMaterial).opacity = opacity;
      (m as THREE.MeshStandardMaterial).depthWrite = false;
    }
  });
  return {
    group: car.group,
    setPose(s) { syncCarMesh(car.group, s); },
    setVisible(v) { car.group.visible = v; },
    dispose(scene) {
      scene.remove(car.group);
      car.group.traverse((o) => {
        const mesh = o as THREE.Mesh;
        if (mesh.geometry) mesh.geometry.dispose();
        const m = mesh.material as THREE.Material | undefined;
        if (m) m.dispose();
      });
    },
  };
}

// ---------------------------------------------------------------------------
// TrailRibbon — append-only THREE.Line with per-vertex colors mapped from
// speed. One per (real chassis + each ghost). Caller calls `push(state)`
// each tick; we cap length so memory doesn't grow unbounded over long
// sessions.

export interface TrailRibbon {
  line: THREE.Line;
  push(s: CarKinematicState): void;
  reset(): void;
  setVisible(v: boolean): void;
  dispose(scene: THREE.Scene): void;
}

export function createTrailRibbon(
  vMax: number,
  capacity = 2048,
  y = 0.05,
): TrailRibbon {
  const positions = new Float32Array(capacity * 3);
  const colors = new Float32Array(capacity * 3);
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geom.setDrawRange(0, 0);
  const mat = new THREE.LineBasicMaterial({ vertexColors: true });
  const line = new THREE.Line(geom, mat);
  let count = 0;
  function push(s: CarKinematicState) {
    let idx = count;
    if (idx >= capacity) {
      // Shift left by 1: drop oldest sample. Cheap enough at 2k.
      positions.copyWithin(0, 3);
      colors.copyWithin(0, 3);
      idx = capacity - 1;
    } else {
      count++;
    }
    positions[idx * 3 + 0] = s.x;
    positions[idx * 3 + 1] = y;
    positions[idx * 3 + 2] = s.z;
    const hex = speedToColor(s.speed, vMax);
    colors[idx * 3 + 0] = ((hex >> 16) & 0xff) / 255;
    colors[idx * 3 + 1] = ((hex >> 8) & 0xff) / 255;
    colors[idx * 3 + 2] = (hex & 0xff) / 255;
    (geom.attributes.position as THREE.BufferAttribute).needsUpdate = true;
    (geom.attributes.color as THREE.BufferAttribute).needsUpdate = true;
    geom.setDrawRange(0, count);
    geom.computeBoundingSphere();
  }
  return {
    line,
    push,
    reset() {
      count = 0;
      geom.setDrawRange(0, 0);
    },
    setVisible(v) { line.visible = v; },
    dispose(scene) {
      scene.remove(line);
      geom.dispose();
      mat.dispose();
    },
  };
}

// ---------------------------------------------------------------------------
// FuturePolyline — short straight-ahead prediction polyline (T seconds).
// Re-built each time `set` is called with a fresh state array. Small N
// (~60 samples), so cheap to recreate.

export interface FuturePolyline {
  line: THREE.Line;
  setPath(states: ReadonlyArray<CarKinematicState>): void;
  setVisible(v: boolean): void;
  dispose(scene: THREE.Scene): void;
}

export function createFuturePolyline(color: number, y = 0.2): FuturePolyline {
  const geom = new THREE.BufferGeometry();
  const mat = new THREE.LineDashedMaterial({
    color,
    dashSize: 0.4,
    gapSize: 0.3,
    transparent: true,
    opacity: 0.9,
  });
  const line = new THREE.Line(geom, mat);
  function setPath(states: ReadonlyArray<CarKinematicState>) {
    const pts = states.map((s) => new THREE.Vector3(s.x, y, s.z));
    geom.setFromPoints(pts);
    (line as THREE.Line).computeLineDistances();
  }
  return {
    line,
    setPath,
    setVisible(v) { line.visible = v; },
    dispose(scene) { scene.remove(line); geom.dispose(); mat.dispose(); },
  };
}

// ---------------------------------------------------------------------------
// UncertaintyCloud — translucent ellipsoid at a ghost's predicted pose,
// axes sized from `predictWithUncertainty(...).std`.

export interface UncertaintyCloud {
  mesh: THREE.Mesh;
  setAt(pos: { x: number; z: number }, stdX: number, stdZ: number): void;
  setVisible(v: boolean): void;
  dispose(scene: THREE.Scene): void;
}

export function createUncertaintyCloud(color: number, y = 1.0): UncertaintyCloud {
  const geom = new THREE.SphereGeometry(1, 16, 12);
  const mat = new THREE.MeshBasicMaterial({
    color,
    transparent: true,
    opacity: 0.18,
    depthWrite: false,
  });
  const mesh = new THREE.Mesh(geom, mat);
  function setAt(pos: { x: number; z: number }, stdX: number, stdZ: number) {
    const rx = Math.max(0.1, Math.min(stdX, 10));
    const rz = Math.max(0.1, Math.min(stdZ, 10));
    mesh.position.set(pos.x, y, pos.z);
    mesh.scale.set(rx, Math.max(rx, rz) * 0.5, rz);
  }
  return {
    mesh,
    setAt,
    setVisible(v) { mesh.visible = v; },
    dispose(scene) { scene.remove(mesh); geom.dispose(); mat.dispose(); },
  };
}

// ---------------------------------------------------------------------------
// ErrorArrows — one ArrowHelper per ghost, pointing from real chassis
// position to ghost's current predicted position. Length = live gap.

export interface ErrorArrow {
  arrow: THREE.ArrowHelper;
  setFromTo(
    from: { x: number; z: number },
    to: { x: number; z: number },
  ): void;
  setVisible(v: boolean): void;
  dispose(scene: THREE.Scene): void;
}

export function createErrorArrow(color: number, y = 0.6): ErrorArrow {
  const dir = new THREE.Vector3(1, 0, 0);
  const origin = new THREE.Vector3();
  const arrow = new THREE.ArrowHelper(dir, origin, 0.001, color, 0.4, 0.25);
  function setFromTo(from: { x: number; z: number }, to: { x: number; z: number }) {
    const v = new THREE.Vector3(to.x - from.x, 0, to.z - from.z);
    const len = v.length();
    if (len < 1e-3) {
      arrow.visible = false;
      return;
    }
    arrow.visible = true;
    arrow.position.set(from.x, y, from.z);
    arrow.setDirection(v.clone().normalize());
    arrow.setLength(len, Math.min(0.5, len * 0.25), Math.min(0.3, len * 0.18));
  }
  return {
    arrow,
    setFromTo,
    setVisible(v) { arrow.visible = v; },
    dispose(scene) { scene.remove(arrow); },
  };
}

// ---------------------------------------------------------------------------
// FrictionCirclePanel — four ground-plane disks, one per wheel. Each
// disk shows the friction circle (radius = suspensionForce * frictionSlip)
// with a vector inside showing the actual (forward, side) impulse,
// normalized to that radius. Color flips red when the impulse vector tip
// reaches the circle's edge (tire is saturated / sliding).

export interface FrictionCircles {
  group: THREE.Group;
  update(wheels: ReadonlyArray<WheelTelemetry>, dt: number): void;
  setVisible(v: boolean): void;
  dispose(scene: THREE.Scene): void;
  /** Last-frame max grip utilization (% of circle radius). */
  lastMaxUtil: () => number;
}

export function createFrictionCircles(diskRadius = 0.45): FrictionCircles {
  const group = new THREE.Group();
  const discs: THREE.Mesh[] = [];
  const vectors: THREE.ArrowHelper[] = [];
  let maxUtil = 0;
  for (let i = 0; i < 4; i++) {
    const disc = new THREE.Mesh(
      new THREE.RingGeometry(diskRadius * 0.95, diskRadius, 24),
      new THREE.MeshBasicMaterial({ color: 0x44ffaa, transparent: true, opacity: 0.65, side: THREE.DoubleSide }),
    );
    disc.rotation.x = -Math.PI / 2;
    disc.visible = false;
    group.add(disc);
    discs.push(disc);
    const arrow = new THREE.ArrowHelper(
      new THREE.Vector3(1, 0, 0),
      new THREE.Vector3(),
      0.001,
      0xffffff,
      0.12,
      0.08,
    );
    arrow.visible = false;
    group.add(arrow);
    vectors.push(arrow);
  }
  function update(wheels: ReadonlyArray<WheelTelemetry>, dt: number) {
    maxUtil = 0;
    for (let i = 0; i < discs.length; i++) {
      const w = wheels[i];
      const disc = discs[i]!;
      const vec = vectors[i]!;
      if (!w || !w.inContact || !w.contactPoint) {
        disc.visible = false;
        vec.visible = false;
        continue;
      }
      // Normalize impulses (N·s over the tick) into forces (N) for a
      // fair comparison with suspensionForce * frictionSlip (also N).
      const denom = Math.max(1e-3, w.suspensionForce * w.frictionSlip);
      const fForward = w.forwardImpulse / Math.max(1e-6, dt);
      const fSide = w.sideImpulse / Math.max(1e-6, dt);
      const mag = Math.hypot(fForward, fSide);
      const util = mag / denom;
      maxUtil = Math.max(maxUtil, util);
      disc.visible = true;
      disc.position.set(w.contactPoint.x, w.contactPoint.y + 0.02, w.contactPoint.z);
      const mat = disc.material as THREE.MeshBasicMaterial;
      mat.color.setHex(util > 1 ? 0xff3030 : util > 0.8 ? 0xffcc00 : 0x44ffaa);
      // Direction in WORLD x/z plane — the impulse vector is in the
      // wheel-local frame, but the panel is meant to be qualitative
      // (saturated vs not). Plot it as (forward, side) in a small disc
      // frame aligned with world axes — good enough for the saturation
      // overlay, even though strictly the disc axes rotate with the
      // wheel. (Documented caveat per plan.)
      const v = new THREE.Vector3(fForward, 0, fSide);
      const len = Math.min(diskRadius, mag / denom * diskRadius);
      if (len < 1e-3) { vec.visible = false; continue; }
      vec.visible = true;
      vec.position.set(w.contactPoint.x, w.contactPoint.y + 0.05, w.contactPoint.z);
      vec.setDirection(v.normalize());
      vec.setLength(len, 0.1, 0.06);
      vec.setColor(util > 1 ? 0xff5555 : 0xffffff);
    }
  }
  return {
    group,
    update,
    setVisible(v) { group.visible = v; },
    dispose(scene) {
      scene.remove(group);
      for (const d of discs) { d.geometry.dispose(); (d.material as THREE.Material).dispose(); }
    },
    lastMaxUtil: () => maxUtil,
  };
}
