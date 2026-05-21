// kinocat/adapters/three — debug visualization + reusable scene helpers for
// kinocat demos. `three` is an OPTIONAL peer; only this subpath imports it.
//
// Two families of helpers:
//
//   1. PLANNER DEBUG (existing) — draw what the planner sees: planned paths,
//      world-space footprints, motion-primitive sweeps.
//
//   2. SCENE BUILDING BLOCKS — the meshes every "small vehicle world" demo
//      needs: ground/grid, buildings, jump ramps with affordance arcs, boost
//      pads, drift gates, a 4-wheel car, waypoint loops, goal markers, and
//      inflated-obstacle overlays. Pure factories returning THREE objects;
//      kinocat itself doesn't manage scene graphs.

import * as THREE from 'three';
import type { VehicleState } from '../../agent/types';

export interface PlanarPoint {
  x: number;
  z: number;
}

export interface MotionPrimitiveLike {
  primitives: ReadonlyArray<{ sweep: ReadonlyArray<{ x: number; z: number }> }>;
}

function toVecs(pts: ReadonlyArray<PlanarPoint>, y: number): THREE.Vector3[] {
  return pts.map((p) => new THREE.Vector3(p.x, y, p.z));
}

// ---------------------------------------------------------------------------
// Planner debug helpers.

/** A polyline for a planned path (states with x,z), drawn at height `y`. */
export function createPlanPathHelper(
  path: ReadonlyArray<PlanarPoint>,
  opts: { y?: number; color?: number } = {},
): THREE.Line {
  const geo = new THREE.BufferGeometry().setFromPoints(toVecs(path, opts.y ?? 0.05));
  const mat = new THREE.LineBasicMaterial({ color: opts.color ?? 0x44ddff });
  return new THREE.Line(geo, mat);
}

/** A closed loop for a world-space footprint polygon. */
export function createFootprintHelper(
  footprint: ReadonlyArray<readonly [number, number]>,
  opts: { y?: number; color?: number } = {},
): THREE.LineLoop {
  const pts = footprint.map((p) => ({ x: p[0], z: p[1] }));
  const geo = new THREE.BufferGeometry().setFromPoints(toVecs(pts, opts.y ?? 0.05));
  const mat = new THREE.LineBasicMaterial({ color: opts.color ?? 0xffaa33 });
  return new THREE.LineLoop(geo, mat);
}

/** A group of polylines, one per motion primitive's local sweep. */
export function createMotionPrimitiveHelper(
  lib: MotionPrimitiveLike,
  opts: { y?: number; color?: number } = {},
): THREE.Group {
  const group = new THREE.Group();
  const mat = new THREE.LineBasicMaterial({ color: opts.color ?? 0x88ff88 });
  for (const p of lib.primitives) {
    const geo = new THREE.BufferGeometry().setFromPoints(toVecs(p.sweep, opts.y ?? 0.05));
    group.add(new THREE.Line(geo, mat));
  }
  return group;
}

// ---------------------------------------------------------------------------
// Scene building blocks.

export interface BoundsXZ {
  x0: number;
  x1: number;
  z0: number;
  z1: number;
}

export interface GroundPlaneOptions {
  bounds: BoundsXZ;
  color?: number;
  /** Grid line divisions; pass 0 to skip the grid. */
  gridDivisions?: number;
  gridColor?: number;
  gridSubColor?: number;
}

/** Flat ground rectangle at y=0 plus an optional grid overlay, returned as a
 *  single group for easy add/remove. */
export function createGroundPlaneHelper(opts: GroundPlaneOptions): THREE.Group {
  const w = opts.bounds.x1 - opts.bounds.x0;
  const d = opts.bounds.z1 - opts.bounds.z0;
  const cx = (opts.bounds.x0 + opts.bounds.x1) / 2;
  const cz = (opts.bounds.z0 + opts.bounds.z1) / 2;
  const group = new THREE.Group();
  const g = new THREE.PlaneGeometry(w, d, 1, 1);
  g.rotateX(-Math.PI / 2);
  g.translate(cx, 0, cz);
  group.add(
    new THREE.Mesh(g, new THREE.MeshStandardMaterial({ color: opts.color ?? 0x1a2233 })),
  );
  const div = opts.gridDivisions ?? 24;
  if (div > 0) {
    const grid = new THREE.GridHelper(
      Math.max(w, d),
      div,
      opts.gridColor ?? 0x2a3040,
      opts.gridSubColor ?? 0x1a1f2c,
    );
    grid.position.set(cx, 0.02, cz);
    group.add(grid);
  }
  return group;
}

export interface BuildingSpec {
  /** Footprint centre on the XZ plane. */
  x: number;
  z: number;
  /** Footprint half-extents. */
  hx: number;
  hz: number;
  /** Visual height (world Y units). */
  height: number;
}

export interface BuildingOptions {
  color?: number;
  edgeColor?: number;
}

/** Cuboid building plus an edge wireframe, as a single group positioned in
 *  world space (centre at `(x, height/2, z)`). */
export function createBuildingHelper(
  spec: BuildingSpec,
  opts: BuildingOptions = {},
): THREE.Group {
  const group = new THREE.Group();
  const g = new THREE.BoxGeometry(spec.hx * 2, spec.height, spec.hz * 2);
  const mesh = new THREE.Mesh(
    g,
    new THREE.MeshStandardMaterial({ color: opts.color ?? 0x3a4458 }),
  );
  mesh.position.set(spec.x, spec.height / 2, spec.z);
  group.add(mesh);
  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(g),
    new THREE.LineBasicMaterial({ color: opts.edgeColor ?? 0x6c7a94 }),
  );
  edges.position.copy(mesh.position);
  group.add(edges);
  return group;
}

export interface JumpRampSpec {
  launch: { x: number; z: number };
  land: { x: number; z: number };
  hx: number;
  hz: number;
  height: number;
}

export interface JumpRampOptions {
  rampColor?: number;
  arcColor?: number;
}

/** Launch ramp cuboid + dashed parabolic arc from launch to land + landing
 *  ring. The arc apex is `height + 2` world units above the midpoint, which
 *  matches the typical `createJumpAffordance({ apexY })` choice. */
export function createJumpRampHelper(
  spec: JumpRampSpec,
  opts: JumpRampOptions = {},
): THREE.Group {
  const group = new THREE.Group();
  const rampColor = opts.rampColor ?? 0x915b3a;
  const arcColor = opts.arcColor ?? 0xffd0a0;
  const g = new THREE.BoxGeometry(spec.hx * 2, spec.height, spec.hz * 2);
  const ramp = new THREE.Mesh(
    g,
    new THREE.MeshStandardMaterial({ color: rampColor }),
  );
  ramp.position.set(spec.launch.x, spec.height / 2, spec.launch.z);
  group.add(ramp);

  const pts: THREE.Vector3[] = [];
  const N = 24;
  const apexY = spec.height + 2;
  for (let i = 0; i <= N; i++) {
    const u = i / N;
    const x = spec.launch.x + (spec.land.x - spec.launch.x) * u;
    const z = spec.launch.z + (spec.land.z - spec.launch.z) * u;
    const y = 4 * apexY * u * (1 - u);
    pts.push(new THREE.Vector3(x, y + 0.3, z));
  }
  const arc = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineDashedMaterial({ color: arcColor, dashSize: 2, gapSize: 1.5 }),
  );
  arc.computeLineDistances();
  group.add(arc);

  const land = new THREE.Mesh(
    new THREE.RingGeometry(2, 2.6, 24),
    new THREE.MeshBasicMaterial({
      color: arcColor,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.7,
    }),
  );
  land.rotation.x = -Math.PI / 2;
  land.position.set(spec.land.x, 0.05, spec.land.z);
  group.add(land);

  return group;
}

export interface BoostPadOptions {
  x: number;
  z: number;
  padColor?: number;
  ringColor?: number;
  radius?: number;
}

/** Glowing boost-pad disc + torus ring. */
export function createBoostPadHelper(opts: BoostPadOptions): THREE.Group {
  const group = new THREE.Group();
  const r = opts.radius ?? 2.4;
  const ringColor = opts.ringColor ?? 0xffa030;
  const padColor = opts.padColor ?? 0xffe066;
  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(r, 0.35, 10, 28),
    new THREE.MeshStandardMaterial({
      color: ringColor,
      emissive: ringColor,
      emissiveIntensity: 0.6,
    }),
  );
  ring.rotation.x = Math.PI / 2;
  ring.position.set(opts.x, 0.6, opts.z);
  group.add(ring);
  const pad = new THREE.Mesh(
    new THREE.CircleGeometry(r, 28),
    new THREE.MeshBasicMaterial({
      color: padColor,
      transparent: true,
      opacity: 0.6,
      side: THREE.DoubleSide,
    }),
  );
  pad.rotation.x = -Math.PI / 2;
  pad.position.set(opts.x, 0.06, opts.z);
  group.add(pad);
  return group;
}

export interface DriftGateOptions {
  x: number;
  z: number;
  /** Direction of travel through the gate (radians, 0 = +X). */
  heading: number;
  /** Half-spacing between the two cones. */
  spacing?: number;
  color?: number;
}

/** A pair of cones flanking a heading axis — a visual marker for drift gates
 *  or slalom pillars. */
export function createDriftGateHelper(opts: DriftGateOptions): THREE.Group {
  const group = new THREE.Group();
  const color = opts.color ?? 0xffd070;
  const phi = opts.heading + Math.PI / 2;
  const off = opts.spacing ?? 3;
  for (const sign of [-1, 1] as const) {
    const cone = new THREE.Mesh(
      new THREE.ConeGeometry(0.45, 1.4, 12),
      new THREE.MeshStandardMaterial({
        color,
        emissive: color,
        emissiveIntensity: 0.4,
      }),
    );
    cone.position.set(
      opts.x + sign * off * Math.cos(phi),
      0.7,
      opts.z + sign * off * Math.sin(phi),
    );
    group.add(cone);
  }
  return group;
}

export interface CarMeshOptions {
  color: number;
  /** Add a flashing police lightbar to the cabin. */
  withLightbar?: boolean;
  lightbarColor?: number;
}

export interface CarMesh {
  group: THREE.Group;
  body: THREE.Mesh;
  lightbar?: THREE.Mesh;
}

/** A simple chassis + cabin + four wheels mesh, ready to be synced from a
 *  planner VehicleState via {@link syncCarMesh}. Visual only; physics wheels
 *  live in Rapier. */
export function createCarMeshHelper(opts: CarMeshOptions): CarMesh {
  const group = new THREE.Group();
  const body = new THREE.Mesh(
    new THREE.BoxGeometry(4.8, 1.0, 2.0),
    new THREE.MeshStandardMaterial({
      color: opts.color,
      metalness: 0.4,
      roughness: 0.5,
    }),
  );
  body.position.y = 0.5;
  group.add(body);
  const cabin = new THREE.Mesh(
    new THREE.BoxGeometry(2.4, 0.7, 1.7),
    new THREE.MeshStandardMaterial({
      color: 0x101218,
      metalness: 0.6,
      roughness: 0.3,
    }),
  );
  cabin.position.set(-0.2, 1.35, 0);
  group.add(cabin);
  const wheelMat = new THREE.MeshStandardMaterial({ color: 0x111111 });
  for (const [fx, fz] of [
    [1.6, -1.0],
    [1.6, 1.0],
    [-1.6, -1.0],
    [-1.6, 1.0],
  ] as const) {
    const w = new THREE.Mesh(
      new THREE.CylinderGeometry(0.4, 0.4, 0.3, 14),
      wheelMat,
    );
    w.rotation.x = Math.PI / 2;
    w.position.set(fx, 0.4, fz);
    group.add(w);
  }
  const out: CarMesh = { group, body };
  if (opts.withLightbar) {
    const c = opts.lightbarColor ?? 0x3322ff;
    const bar = new THREE.Mesh(
      new THREE.BoxGeometry(1.4, 0.18, 1.6),
      new THREE.MeshStandardMaterial({
        color: c,
        emissive: c,
        emissiveIntensity: 0.7,
      }),
    );
    bar.position.set(-0.2, 1.78, 0);
    group.add(bar);
    out.lightbar = bar;
  }
  return out;
}

/** Position + rotate a car group from a planner `VehicleState`. kinocat
 *  heading 0 = +X; THREE mesh forward (BoxGeometry +X) aligns with
 *  `rotation.y = -heading` (THREE Y-up right-handed yaw sign-flips kinocat). */
export function syncCarMesh(group: THREE.Group, s: VehicleState): void {
  group.position.set(s.x, group.position.y, s.z);
  group.rotation.y = -s.heading;
}

export interface WaypointLoopOptions {
  color?: number;
  opacity?: number;
  y?: number;
  /** Close the loop (default true). */
  closed?: boolean;
}

/** A faint polyline through a waypoint sequence. */
export function createWaypointLoopHelper(
  waypoints: ReadonlyArray<PlanarPoint>,
  opts: WaypointLoopOptions = {},
): THREE.Line {
  const y = opts.y ?? 0.1;
  const pts = waypoints.map((w) => new THREE.Vector3(w.x, y, w.z));
  if ((opts.closed ?? true) && pts.length > 0) pts.push(pts[0]!.clone());
  return new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineBasicMaterial({
      color: opts.color ?? 0x4a5570,
      transparent: true,
      opacity: opts.opacity ?? 0.45,
    }),
  );
}

export interface GoalMarkerOptions {
  color: number;
  size?: number;
  opacity?: number;
}

/** Translucent octahedron, hidden by default. Move it with `.position.set` and
 *  toggle visibility from the caller. */
export function createGoalMarkerHelper(opts: GoalMarkerOptions): THREE.Mesh {
  const m = new THREE.Mesh(
    new THREE.OctahedronGeometry(opts.size ?? 1.1),
    new THREE.MeshStandardMaterial({
      color: opts.color,
      emissive: opts.color,
      emissiveIntensity: 0.6,
      transparent: true,
      opacity: opts.opacity ?? 0.55,
    }),
  );
  m.visible = false;
  return m;
}

export interface InflatedObstacleOptions {
  color?: number;
  y?: number;
}

/** The wireframe of an obstacle's planner-inflated footprint (visual half-
 *  extents + `inflate` on each side). Use this to debug planner-vs-visual
 *  collision discrepancies. */
export function createInflatedObstacleHelper(
  spec: BuildingSpec,
  inflate: number,
  opts: InflatedObstacleOptions = {},
): THREE.Line {
  const hx = spec.hx + inflate;
  const hz = spec.hz + inflate;
  const y = opts.y ?? 0.15;
  const ring: THREE.Vector3[] = [
    new THREE.Vector3(spec.x - hx, y, spec.z - hz),
    new THREE.Vector3(spec.x + hx, y, spec.z - hz),
    new THREE.Vector3(spec.x + hx, y, spec.z + hz),
    new THREE.Vector3(spec.x - hx, y, spec.z + hz),
    new THREE.Vector3(spec.x - hx, y, spec.z - hz),
  ];
  return new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(ring),
    new THREE.LineBasicMaterial({ color: opts.color ?? 0xff66aa }),
  );
}

/** Outline the planning rectangle. */
export function createNavBoundsHelper(
  bounds: BoundsXZ,
  opts: { color?: number; y?: number } = {},
): THREE.Line {
  const y = opts.y ?? 0.1;
  const pts: THREE.Vector3[] = [
    new THREE.Vector3(bounds.x0, y, bounds.z0),
    new THREE.Vector3(bounds.x1, y, bounds.z0),
    new THREE.Vector3(bounds.x1, y, bounds.z1),
    new THREE.Vector3(bounds.x0, y, bounds.z1),
    new THREE.Vector3(bounds.x0, y, bounds.z0),
  ];
  return new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineBasicMaterial({ color: opts.color ?? 0x66ffaa }),
  );
}

/** An agent footprint rectangle, in body-local coords. Move/rotate with
 *  `.position.set(x, 0, z); .rotation.y = -heading;` to track an agent. */
export function createAgentFootprintHelper(
  footprint: ReadonlyArray<readonly [number, number]>,
  opts: { color?: number; y?: number } = {},
): THREE.Line {
  // Tightest axis-aligned wrapper around the body-frame footprint.
  let hx = 0;
  let hz = 0;
  for (const [fx, fz] of footprint) {
    if (Math.abs(fx) > hx) hx = Math.abs(fx);
    if (Math.abs(fz) > hz) hz = Math.abs(fz);
  }
  const y = opts.y ?? 0.2;
  const ring: THREE.Vector3[] = [
    new THREE.Vector3(hx, y, hz),
    new THREE.Vector3(hx, y, -hz),
    new THREE.Vector3(-hx, y, -hz),
    new THREE.Vector3(-hx, y, hz),
    new THREE.Vector3(hx, y, hz),
  ];
  return new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(ring),
    new THREE.LineBasicMaterial({ color: opts.color ?? 0xffffff }),
  );
}
