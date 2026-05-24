'use client';

// A single-car obstacle-course demo: heightfield terrain, buildings, ramp +
// jump affordance, boost pad, drift gates, waypoint loop. Each block is
// toggleable from the HUD so a regression in any one of them reproduces in
// isolation — flip "buildings" off and watch the planner; flip the ramp on
// and watch the affordance fire. The point is to drive the core's new
// building-block APIs (`kinocat/adapters/rapier`, `kinocat/adapters/three`,
// `kinocat/environment`, `kinocat/planner`) and surface latent bugs before
// they hide inside the multi-AI car-chase demo.

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import RAPIER from '@dimforge/rapier3d-compat';
import {
  InMemoryNavWorld,
  rampHeightSampler,
  combineHeightSamplers,
} from 'kinocat/environment';
import type { HeightSampler } from 'kinocat/environment';
import type { CarKinematicState } from 'kinocat/agent';
import {
  ensureRapier,
  createRaycastVehicle,
  createGroundCollider,
  createBoxCollider,
  createHeightfieldCollider,
  planToAckermannControls,
  stepRaycastVehicle,
} from 'kinocat/adapters/rapier';
import {
  createGroundPlaneHelper,
  createBuildingHelper,
  createBoostPadHelper,
  createDriftGateHelper,
  createCarMeshHelper,
  syncCarMesh,
  createWaypointLoopHelper,
  createGoalMarkerHelper,
  createInflatedObstacleHelper,
  createNavBoundsHelper,
  createAgentFootprintHelper,
  createHeightfieldMeshHelper,
  createRampChevronsHelper,
  createJumpArcHelper,
  createRapierDebugRenderer,
  updateChaseCamera,
} from 'kinocat/adapters/three';
import { trimPlan } from 'kinocat/vehicle/car';
import {
  OBS_AGENT,
  OBS_BOUNDS,
  OBS_PALETTE as C,
  OBS_BLOCKS_ALL,
  buildObstacleCourse,
  type ObstacleCourse as Course,
  obsPickWaypoint,
  obsSpawn,
  planObstacleCourse,
  type ObsBlocks,
} from '../lib/obstaclecourse-scenarios';

const PHYSICS_DT = 1 / 60;
const VEHICLE_SUBSTEPS = 4;
const REPLAN_INTERVAL_MS = 120;
const WHEEL_BASE = 1.6; // matches default in createRaycastVehicle

// Gentle terrain so a flat-ground vehicle still copes. Bumps + bowls let us
// see suspension behaviour without making the planner-vs-physics gap obvious.
function terrainSampler(x: number, z: number): number {
  return 0.6 * Math.sin(x / 18) + 0.6 * Math.cos(z / 14);
}

export default function ObstacleCourse() {
  const mountRef = useRef<HTMLDivElement>(null);

  const [paused, setPaused] = useState(false);
  const [chase, setChase] = useState(true);
  const [showPath, setShowPath] = useState(true);
  const [showDebug, setShowDebug] = useState(false);
  const [showRapierDebug, setShowRapierDebug] = useState(false);
  const [playerDriving, setPlayerDriving] = useState(false);
  const [blocks, setBlocks] = useState<ObsBlocks>(OBS_BLOCKS_ALL);
  const [ready, setReady] = useState(false);
  const [hud, setHud] = useState('');
  const [status, setStatus] = useState('');

  const pausedRef = useRef(paused);
  const chaseRef = useRef(chase);
  const showPathRef = useRef(showPath);
  const showDebugRef = useRef(showDebug);
  const showRapierDebugRef = useRef(showRapierDebug);
  const playerDrivingRef = useRef(playerDriving);
  const blocksRef = useRef(blocks);
  const rebuildRef = useRef<(() => void) | null>(null);
  pausedRef.current = paused;
  chaseRef.current = chase;
  showPathRef.current = showPath;
  showDebugRef.current = showDebug;
  showRapierDebugRef.current = showRapierDebug;
  playerDrivingRef.current = playerDriving;
  blocksRef.current = blocks;

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    let disposed = false;
    let cleanup: (() => void) | null = null;
    (async () => {
      await ensureRapier();
      if (disposed) return;
      cleanup = setupScene(mount);
      setReady(true);
    })();
    return () => {
      disposed = true;
      cleanup?.();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Whenever the block toggles change, ask the scene to rebuild the course.
  useEffect(() => {
    rebuildRef.current?.();
  }, [blocks]);

  function setupScene(mount: HTMLDivElement): () => void {
    const W0 = window.innerWidth;
    const H0 = window.innerHeight;

    // ---- three.js setup ----
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(C.bg);
    scene.fog = new THREE.Fog(C.fog, 120, 320);
    const camera = new THREE.PerspectiveCamera(60, W0 / H0, 0.1, 800);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(W0, H0);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);

    const mapCx = (OBS_BOUNDS.x0 + OBS_BOUNDS.x1) / 2;
    const mapCz = (OBS_BOUNDS.z0 + OBS_BOUNDS.z1) / 2;
    const orbit = new OrbitControls(camera, renderer.domElement);
    orbit.target.set(mapCx, 0, mapCz);
    camera.position.set(mapCx + 40, 70, mapCz + 80);
    orbit.update();

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const sun = new THREE.DirectionalLight(0xffffff, 1.1);
    sun.position.set(60, 140, 40);
    scene.add(sun);

    // ---- physics + course state, rebuilt on toggle changes ----
    const world = new RAPIER.World({ x: 0, y: -9.81, z: 0 });

    let course: Course = buildObstacleCourse(blocksRef.current);
    let navWorld = new InMemoryNavWorld(course.polygons, course.obstacles);

    // Visual groups we tear down on rebuild. Path / goal / footprint persist
    // across rebuilds (they belong to the car, not the course).
    let coursePhysics: RAPIER.Collider[] = [];
    let courseVisuals = new THREE.Group();
    scene.add(courseVisuals);

    function buildCourseVisualsAndPhysics() {
      // Wipe previous visuals.
      scene.remove(courseVisuals);
      courseVisuals.traverse((o) => {
        const mesh = o as THREE.Mesh;
        if (mesh.geometry) mesh.geometry.dispose();
        const mat = mesh.material as THREE.Material | THREE.Material[] | undefined;
        if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
        else if (mat) mat.dispose();
      });
      courseVisuals = new THREE.Group();
      scene.add(courseVisuals);

      // Wipe previous physics colliders.
      for (const col of coursePhysics) {
        const body = col.parent();
        world.removeCollider(col, true);
        if (body) world.removeRigidBody(body);
      }
      coursePhysics = [];

      // Heightfield surface = (optional) bumpy terrain max-blended with the
      // drivable ramps. When neither block is on, fall back to a flat ground
      // slab. Using a single heightfield for both means the car physically
      // climbs the ramp via the same drivable surface as the /ramp demo —
      // no cuboid colliders for ramps.
      const samplers: HeightSampler[] = [];
      if (blocksRef.current.heightfield) samplers.push(terrainSampler);
      if (course.ramps.length > 0) samplers.push(rampHeightSampler(course.ramps));
      if (samplers.length > 0) {
        const sampler = combineHeightSamplers(...samplers);
        courseVisuals.add(
          createHeightfieldMeshHelper({
            bounds: OBS_BOUNDS,
            sampler,
            segmentsX: 60,
            segmentsZ: 40,
            groundColor: C.ground,
            // Vertex-colour the ramp body brown above 0.2 m so the drivable
            // ramp visually pops out of the bumpy terrain.
            vertexColorAbove: course.ramps.length > 0 ? 0.2 : undefined,
            aboveColor: C.ramp,
          }),
        );
        coursePhysics.push(
          createHeightfieldCollider(world, {
            sampler,
            bounds: OBS_BOUNDS,
            cellSize: 2,
          }),
        );
      } else {
        courseVisuals.add(
          createGroundPlaneHelper({ bounds: OBS_BOUNDS, color: 0x1a2233 }),
        );
        coursePhysics.push(
          createGroundCollider(world, { bounds: OBS_BOUNDS, pad: 20 }),
        );
      }

      for (const b of course.buildings) {
        courseVisuals.add(
          createBuildingHelper(b, { color: 0x3a4458, edgeColor: 0x6c7a94 }),
        );
        coursePhysics.push(
          createBoxCollider(world, {
            x: b.x,
            y: b.height / 2,
            z: b.z,
            hx: b.hx,
            hy: b.height / 2,
            hz: b.hz,
          }),
        );
      }

      // Drivable ramps: directional chevrons on the surface + the jump
      // affordance arc overlay. No cuboid collider — physics is the
      // heightfield above.
      for (const r of course.ramps) {
        courseVisuals.add(createRampChevronsHelper(r));
      }
      for (const j of course.jumps) {
        courseVisuals.add(
          createJumpArcHelper(
            { launch: j.launch, land: j.land, hx: 0, hz: 0, height: j.height },
            { launchY: j.height, apexClearance: 2 },
          ),
        );
      }

      for (const p of course.boosts) {
        courseVisuals.add(createBoostPadHelper({ x: p.x, z: p.z }));
      }

      for (const g of course.driftGates) {
        courseVisuals.add(createDriftGateHelper({ x: g.x, z: g.z, heading: g.heading }));
      }

      courseVisuals.add(createWaypointLoopHelper(course.waypoints));
    }

    function rebuildCourse() {
      course = buildObstacleCourse(blocksRef.current);
      navWorld = new InMemoryNavWorld(course.polygons, course.obstacles);
      buildCourseVisualsAndPhysics();
      // Refresh inflated overlay too.
      debugGroup.clear();
      for (const b of course.buildings) {
        debugGroup.add(createInflatedObstacleHelper(b, 0.5));
      }
      debugGroup.add(createNavBoundsHelper(OBS_BOUNDS));
      debugGroup.add(carFootprint);
      // Wake the AI.
      ai.plan = null;
    }
    rebuildRef.current = rebuildCourse;

    // Debug overlay: inflated obstacles + nav bounds + agent footprint.
    const debugGroup = new THREE.Group();
    debugGroup.visible = false;
    scene.add(debugGroup);
    const carFootprint = createAgentFootprintHelper(OBS_AGENT.footprint, {
      color: C.car,
    });

    // ---- rapier debug wireframe ----
    const rapierDebug = createRapierDebugRenderer();
    rapierDebug.mesh.visible = false;
    scene.add(rapierDebug.mesh);

    // ---- car ----
    const car = createRaycastVehicle(world, {
      id: 'obs-car',
      position: { x: obsSpawn().x, z: obsSpawn().z },
      heading: obsSpawn().heading,
    });
    const carMesh = createCarMeshHelper({ color: C.car });
    scene.add(carMesh.group);

    const ai = {
      plan: null as CarKinematicState[] | null,
      planStartWall: performance.now(),
      loopIndex: 0,
      goal: null as CarKinematicState | null,
      lastExpansions: 0,
      lastBudgetMs: 0,
    };

    const goalMarker = createGoalMarkerHelper({ color: C.goal });
    scene.add(goalMarker);

    let pathLine: THREE.Line | null = null;
    function replacePathLine(path: CarKinematicState[]): void {
      if (pathLine) {
        scene.remove(pathLine);
        pathLine.geometry.dispose();
        (pathLine.material as THREE.Material).dispose();
      }
      const pts = path.map((p) => new THREE.Vector3(p.x, 0.3, p.z));
      pathLine = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineBasicMaterial({
          color: C.carPath,
          transparent: true,
          opacity: 0.85,
        }),
      );
      scene.add(pathLine);
    }

    // First build.
    buildCourseVisualsAndPhysics();
    for (const b of course.buildings) {
      debugGroup.add(createInflatedObstacleHelper(b, 0.5));
    }
    debugGroup.add(createNavBoundsHelper(OBS_BOUNDS));
    debugGroup.add(carFootprint);

    // ---- input ----
    const keys = new Set<string>();
    const onKeyDown = (e: KeyboardEvent) => {
      const k = e.key.toLowerCase();
      keys.add(k);
      if (k === 'p') setPaused((v) => !v);
      if (k === 'c') setChase((v) => !v);
      if (k === '1') setShowPath((v) => !v);
      if (k === 'd') setShowDebug((v) => !v);
      if (k === '2') setShowRapierDebug((v) => !v);
      if (k === 't') setPlayerDriving((v) => !v);
      if (k === 'r') {
        car.teleport({ x: obsSpawn().x, z: obsSpawn().z, heading: obsSpawn().heading });
        ai.plan = null;
        ai.loopIndex = 0;
      }
    };
    const onKeyUp = (e: KeyboardEvent) => keys.delete(e.key.toLowerCase());
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);

    // ---- replan ----
    function replan() {
      const now = performance.now();
      const state = car.readState(now);
      const pick = obsPickWaypoint(state, course, ai.loopIndex, navWorld);
      ai.loopIndex = pick.nextIndex;
      ai.goal = pick.goal;
      const t0 = performance.now();
      const res = planObstacleCourse({
        state: { ...state, t: 0 },
        goal: { ...pick.goal, t: 0 },
        course,
        world: navWorld,
      });
      ai.lastBudgetMs = performance.now() - t0;
      ai.lastExpansions = res.stats.expansions;
      if (res.found && res.path.length > 1) {
        ai.plan = res.path;
        ai.planStartWall = now;
        if (showPathRef.current) replacePathLine(res.path);
      }
    }
    const replanTimer = window.setInterval(() => {
      if (pausedRef.current || playerDrivingRef.current) return;
      replan();
    }, REPLAN_INTERVAL_MS);

    // ---- resize ----
    const onResize = () => {
      const w = window.innerWidth;
      const h = window.innerHeight;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    window.addEventListener('resize', onResize);

    // ---- main loop ----
    let stopped = false;
    let lastHudWall = 0;
    function tick() {
      if (stopped) return;
      requestAnimationFrame(tick);
      if (pausedRef.current) {
        renderer.render(scene, camera);
        return;
      }
      const now = performance.now();
      const state = car.readState(now);

      // Drive — player or AI.
      if (playerDrivingRef.current) {
        const accel =
          (keys.has('w') || keys.has('arrowup') ? 1 : 0) -
          (keys.has('s') || keys.has('arrowdown') ? 1 : 0);
        const steerIn =
          (keys.has('a') || keys.has('arrowleft') ? 1 : 0) -
          (keys.has('d') || keys.has('arrowright') ? 1 : 0);
        const brake = keys.has(' ') ? 1 : 0;
        car.applyControls({ steer: steerIn * 0.55, throttle: accel, brake });
      } else if (ai.plan && ai.plan.length > 1) {
        const elapsed = (now - ai.planStartWall) / 1000;
        const live = trimPlan(ai.plan, elapsed);
        if (live.length >= 2) {
          car.applyControls(
            planToAckermannControls(state, live, {
              wheelBase: 2 * WHEEL_BASE,
              lookaheadMin: 3,
              lookaheadGain: 0.45,
              lookaheadMax: 14,
              maxLateralAccel: 8,
              maxAccel: 6,
              maxDecel: 8,
              cruiseSpeed: OBS_AGENT.maxSpeed,
              goalTolerance: 2,
              minTurnRadius: OBS_AGENT.minTurnRadius,
            }),
          );
        } else {
          car.applyControls({ steer: 0, throttle: 0.2, brake: 0 });
        }
      } else {
        // No plan yet — coast in neutral. Idling forward into the first plan
        // sounds harmless until a heightfield slope flips the car into a
        // wall; see bug-audit notes.
        car.applyControls({ steer: 0, throttle: 0, brake: 0 });
      }

      // Physics: vehicle update before world step, sub-stepped (Rapier
      // raycast-vehicle is twitchy at 60 Hz on its own).
      stepRaycastVehicle(world, [car], { dt: PHYSICS_DT, substeps: VEHICLE_SUBSTEPS });

      const after = car.readState(now);
      syncCarMesh(carMesh.group, after);
      if (pathLine) pathLine.visible = showPathRef.current;

      debugGroup.visible = showDebugRef.current;
      rapierDebug.mesh.visible = showRapierDebugRef.current;
      if (showRapierDebugRef.current) rapierDebug.update(world);
      if (showDebugRef.current) {
        carFootprint.position.set(after.x, 0, after.z);
        carFootprint.rotation.y = -after.heading;
      }

      goalMarker.visible = !!ai.goal;
      if (ai.goal) goalMarker.position.set(ai.goal.x, 2, ai.goal.z);

      if (chaseRef.current) {
        updateChaseCamera(camera, { x: after.x, z: after.z, heading: after.heading }, { orbit });
      }

      if (now - lastHudWall > 100) {
        lastHudWall = now;
        const v = Math.abs(after.speed).toFixed(1);
        const hdg = ((after.heading * 180) / Math.PI).toFixed(0);
        setHud(
          `${playerDrivingRef.current ? 'YOU' : 'AI'} · v=${v} m/s · hdg=${hdg}°`,
        );
        setStatus(
          `plan=${ai.plan?.length ?? 0} exp=${ai.lastExpansions} budget=${ai.lastBudgetMs.toFixed(0)}ms wp=${ai.loopIndex}/${course.waypoints.length}`,
        );
      }

      renderer.render(scene, camera);
    }
    tick();

    return () => {
      stopped = true;
      window.clearInterval(replanTimer);
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('resize', onResize);
      car.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === mount)
        mount.removeChild(renderer.domElement);
      world.free();
    };
  }

  return (
    <div ref={mountRef} style={{ position: 'fixed', inset: 0, background: '#0a0d14' }}>
      {!ready && (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#9bd0ff',
            font: '14px ui-monospace, monospace',
          }}
        >
          loading Rapier physics…
        </div>
      )}
      <div
        style={{
          position: 'absolute',
          top: 12,
          left: 12,
          color: '#cdd3de',
          font: '12px ui-monospace, monospace',
          background: 'rgba(10,14,22,0.7)',
          padding: '10px 14px',
          borderRadius: 8,
          border: '1px solid #1f2735',
          maxWidth: 380,
          lineHeight: 1.45,
        }}
      >
        <div style={{ color: '#7fd6ff', fontWeight: 700, marginBottom: 4 }}>
          obstacle course — building blocks
        </div>
        <div>{hud}</div>
        <div style={{ opacity: 0.8 }}>{status}</div>
        <div style={{ opacity: 0.6, marginTop: 6, fontSize: 11 }}>
          Toggle blocks to bisect any "doesn't work" bug. Then graduate to /carchase.
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 8 }}>
          {(Object.keys(OBS_BLOCKS_ALL) as Array<keyof ObsBlocks>).map((k) => (
            <ToggleButton
              key={k}
              label={k}
              on={blocks[k]}
              onClick={() => setBlocks((b) => ({ ...b, [k]: !b[k] }))}
            />
          ))}
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 6 }}>
          <ToggleButton
            label="drive"
            shortcut="t"
            on={playerDriving}
            onClick={() => setPlayerDriving((v) => !v)}
          />
          <ToggleButton
            label="pause"
            shortcut="p"
            on={paused}
            onClick={() => setPaused((v) => !v)}
          />
          <ToggleButton
            label="chase cam"
            shortcut="c"
            on={chase}
            onClick={() => setChase((v) => !v)}
          />
          <ToggleButton
            label="path"
            shortcut="1"
            on={showPath}
            onClick={() => setShowPath((v) => !v)}
          />
          <ToggleButton
            label="debug"
            shortcut="d"
            on={showDebug}
            onClick={() => setShowDebug((v) => !v)}
          />
          <ToggleButton
            label="physics"
            shortcut="2"
            on={showRapierDebug}
            onClick={() => setShowRapierDebug((v) => !v)}
          />
        </div>
        <div style={{ opacity: 0.55, marginTop: 8, fontSize: 11 }}>
          [wasd] drive · [space] brake · [r] reset
        </div>
      </div>
    </div>
  );
}

function ToggleButton({
  label,
  shortcut,
  on,
  onClick,
}: {
  label: string;
  shortcut?: string;
  on: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        font: '11px ui-monospace, monospace',
        padding: '4px 8px',
        borderRadius: 6,
        border: `1px solid ${on ? '#7fd6ff' : '#1f2735'}`,
        background: on ? 'rgba(127, 214, 255, 0.18)' : 'rgba(20, 26, 38, 0.85)',
        color: on ? '#cdeaff' : '#8c95a4',
        cursor: 'pointer',
        letterSpacing: 0.3,
      }}
    >
      {shortcut && <span style={{ opacity: 0.65, marginRight: 4 }}>[{shortcut}]</span>}
      {label}
      <span style={{ marginLeft: 6, opacity: 0.85, fontWeight: 600 }}>
        {on ? 'on' : 'off'}
      </span>
    </button>
  );
}
