'use client';

// Ramp + Affordance demo. A single drivable heightfield ramp the car
// physically climbs and launches off, plus a BallisticJump Affordance the
// planner can pick as a shortcut over a planner-only "gap" obstacle. Toggle
// the affordance from the HUD and watch the planned path change. Execution
// is always real Rapier physics — the car always drives, never poses.

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import RAPIER from '@dimforge/rapier3d-compat';
import { InMemoryNavWorld } from 'kinocat/environment';
import type { VehicleState } from 'kinocat/agent';
import {
  ensureRapier,
  createRaycastVehicle,
  createHeightfieldCollider,
  planToAckermannControls,
  type RaycastVehicleOptions,
} from 'kinocat/adapters/rapier';
import {
  createCarMeshHelper,
  syncCarMesh,
  createGoalMarkerHelper,
  createInflatedObstacleHelper,
  createNavBoundsHelper,
  createAgentFootprintHelper,
  createJumpArcHelper,
  createRapierDebugRenderer,
} from 'kinocat/adapters/three';
import {
  RAMP_AGENT,
  RAMP_BOUNDS,
  RAMP_PALETTE as C,
  buildRampCourse,
  planRampDemo,
  rampHeightSampler,
  type RampCourse,
} from '../lib/ramp-scenarios';

const PHYSICS_DT = 1 / 60;
const VEHICLE_SUBSTEPS = 4;
const REPLAN_INTERVAL_MS = 120;
const WHEEL_BASE = 1.6;

// Slightly more suspension travel than the obstacle-course tuning — the
// ramp lip + landing benefit from a softer chassis.
const RAMP_VEHICLE_TUNING: Omit<RaycastVehicleOptions, 'id' | 'position' | 'heading'> = {
  chassisHalf: { x: 2.4, y: 0.5, z: 1.0 },
  chassisDensity: 60,
  wheelBase: WHEEL_BASE,
  wheelTrack: 0.85,
  wheelRadius: 0.35,
  suspensionRestLength: 0.4,
  suspensionMaxTravel: 0.3,
  engineForce: 4500,
  brakeForce: 2000,
  maxSteerAngle: 0.6,
  driveTrain: 'rwd',
};

export default function RampDemo() {
  const mountRef = useRef<HTMLDivElement>(null);

  const [paused, setPaused] = useState(false);
  const [chase, setChase] = useState(true);
  const [showPath, setShowPath] = useState(true);
  const [showDebug, setShowDebug] = useState(false);
  const [showRapierDebug, setShowRapierDebug] = useState(false);
  const [playerDriving, setPlayerDriving] = useState(false);
  const [affordanceOn, setAffordanceOn] = useState(true);
  const [ready, setReady] = useState(false);
  const [hud, setHud] = useState('');
  const [status, setStatus] = useState('');

  const pausedRef = useRef(paused);
  const chaseRef = useRef(chase);
  const showPathRef = useRef(showPath);
  const showDebugRef = useRef(showDebug);
  const showRapierDebugRef = useRef(showRapierDebug);
  const playerDrivingRef = useRef(playerDriving);
  const affordanceOnRef = useRef(affordanceOn);
  const replanNowRef = useRef<(() => void) | null>(null);
  pausedRef.current = paused;
  chaseRef.current = chase;
  showPathRef.current = showPath;
  showDebugRef.current = showDebug;
  showRapierDebugRef.current = showRapierDebug;
  playerDrivingRef.current = playerDriving;
  affordanceOnRef.current = affordanceOn;

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

  // Replan whenever the affordance toggle changes so the user sees the
  // path swap immediately.
  useEffect(() => {
    replanNowRef.current?.();
  }, [affordanceOn]);

  function setupScene(mount: HTMLDivElement): () => void {
    const W0 = window.innerWidth;
    const H0 = window.innerHeight;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(C.bg);
    scene.fog = new THREE.Fog(C.fog, 120, 320);
    const camera = new THREE.PerspectiveCamera(60, W0 / H0, 0.1, 800);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(W0, H0);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);

    const mapCx = (RAMP_BOUNDS.x0 + RAMP_BOUNDS.x1) / 2;
    const mapCz = (RAMP_BOUNDS.z0 + RAMP_BOUNDS.z1) / 2;
    const orbit = new OrbitControls(camera, renderer.domElement);
    orbit.target.set(mapCx, 0, mapCz);
    camera.position.set(mapCx + 30, 45, mapCz + 60);
    orbit.update();

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const sun = new THREE.DirectionalLight(0xffffff, 1.1);
    sun.position.set(60, 140, 40);
    scene.add(sun);

    const course: RampCourse = buildRampCourse();
    const navWorld = new InMemoryNavWorld(course.polygons, course.obstacles);
    const sampler = rampHeightSampler(course.ramps);

    // ---- physics: heightfield (flat ground + ramp lip) ----
    const world = new RAPIER.World({ x: 0, y: -9.81, z: 0 });
    // Finer cellSize than ObstacleCourse — we need a sharp lip at the crest
    // for a clean launch instead of a smooth roll-off.
    createHeightfieldCollider(world, {
      sampler,
      bounds: RAMP_BOUNDS,
      // Matches the obstacle-course demo. Finer resolution (cellSize=1)
      // produced denser triangulation at the ramp's sharp crest "lip"
      // that occasionally WASM-trapped the vehicle controller's wheel
      // raycasts under load.
      cellSize: 2,
      friction: 1.5,
    });

    // ---- visuals: heightfield mesh + arc + gap + waypoints ----
    const courseVisuals = new THREE.Group();
    scene.add(courseVisuals);

    const w = RAMP_BOUNDS.x1 - RAMP_BOUNDS.x0;
    const d = RAMP_BOUNDS.z1 - RAMP_BOUNDS.z0;
    const segX = 120;
    const segZ = 60;
    const g = new THREE.PlaneGeometry(w, d, segX, segZ);
    g.rotateX(-Math.PI / 2);
    g.translate(mapCx, 0, mapCz);
    const pos = g.attributes.position as THREE.BufferAttribute;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i);
      const z = pos.getZ(i);
      pos.setY(i, sampler(x, z));
    }
    pos.needsUpdate = true;
    g.computeVertexNormals();
    // Vertex-colour the ramp body brown vs ground blue-grey.
    const colors = new Float32Array(pos.count * 3);
    const groundCol = new THREE.Color(C.ground);
    const rampCol = new THREE.Color(C.ramp);
    for (let i = 0; i < pos.count; i++) {
      const y = pos.getY(i);
      const c = y > 0.05 ? rampCol : groundCol;
      colors[i * 3 + 0] = c.r;
      colors[i * 3 + 1] = c.g;
      colors[i * 3 + 2] = c.b;
    }
    g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    courseVisuals.add(
      new THREE.Mesh(
        g,
        new THREE.MeshStandardMaterial({ vertexColors: true, flatShading: true }),
      ),
    );
    const grid = new THREE.GridHelper(Math.max(w, d), 24, 0x2a3040, 0x1a1f2c);
    grid.position.set(mapCx, 0.02, mapCz);
    courseVisuals.add(grid);

    // Directional chevrons on each ramp surface so the launch direction is
    // visually unambiguous (otherwise a heightfield ramp viewed from the
    // chase-cam angle looks like a generic mound).
    for (const r of course.ramps) {
      const c = Math.cos(r.heading);
      const s = Math.sin(r.heading);
      // Three chevrons spaced along the ramp, sized half the ramp width.
      const chevHalf = r.width * 0.25;
      for (let k = 1; k <= 3; k++) {
        const u = k / 4; // 0.25, 0.5, 0.75 along the ramp
        const along = -r.length / 2 + u * r.length;
        const cx = r.base.x + along * c;
        const cz = r.base.z + along * s;
        // Height at chevron midpoint, plus a tiny lift to avoid z-fighting.
        const cy = r.height * u + 0.05;
        // Chevron = "v" pointing along +heading, lying flat on the ramp top.
        const tipX = cx + chevHalf * 0.7 * c;
        const tipZ = cz + chevHalf * 0.7 * s;
        const backLX = cx - chevHalf * 0.7 * c - chevHalf * s;
        const backLZ = cz - chevHalf * 0.7 * s + chevHalf * c;
        const backRX = cx - chevHalf * 0.7 * c + chevHalf * s;
        const backRZ = cz - chevHalf * 0.7 * s - chevHalf * c;
        const pts = [
          new THREE.Vector3(backLX, cy, backLZ),
          new THREE.Vector3(tipX, cy, tipZ),
          new THREE.Vector3(backRX, cy, backRZ),
        ];
        const chev = new THREE.Line(
          new THREE.BufferGeometry().setFromPoints(pts),
          new THREE.LineBasicMaterial({ color: 0xffe066, linewidth: 2 }),
        );
        courseVisuals.add(chev);
      }
    }

    // Affordance arc overlay (only when affordances enabled).
    const arcGroups: THREE.Group[] = [];
    for (const j of course.jumps) {
      const arc = createJumpArcHelper(
        {
          launch: j.launch,
          land: j.land,
          hx: 0,
          hz: 0,
          height: j.height,
        },
        { launchY: j.height, arcColor: C.arc, apexClearance: 2 },
      );
      arcGroups.push(arc);
      courseVisuals.add(arc);
    }

    // Gap overlay — pink rectangle for the planner-only obstacle.
    for (const gap of course.gaps) {
      const overlay = createInflatedObstacleHelper(
        { x: gap.x, z: gap.z, hx: gap.hx, hz: gap.hz, height: 0 },
        0.5,
        { color: C.gap, y: 0.2 },
      );
      courseVisuals.add(overlay);
    }

    // Start / goal markers.
    const startMarker = new THREE.Mesh(
      new THREE.RingGeometry(1.4, 1.8, 24),
      new THREE.MeshBasicMaterial({
        color: 0x66ffaa,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.7,
      }),
    );
    startMarker.rotation.x = -Math.PI / 2;
    startMarker.position.set(course.spawn.x, 0.05, course.spawn.z);
    courseVisuals.add(startMarker);

    const goalMarker = createGoalMarkerHelper({ color: C.goal });
    goalMarker.position.set(course.goal.x, 2, course.goal.z);
    goalMarker.visible = true;
    scene.add(goalMarker);

    // Debug overlay.
    const debugGroup = new THREE.Group();
    debugGroup.visible = false;
    scene.add(debugGroup);
    debugGroup.add(createNavBoundsHelper(RAMP_BOUNDS));
    const carFootprint = createAgentFootprintHelper(RAMP_AGENT.footprint, {
      color: C.car,
    });
    debugGroup.add(carFootprint);

    // ---- rapier debug wireframe ----
    const rapierDebug = createRapierDebugRenderer();
    rapierDebug.mesh.visible = false;
    scene.add(rapierDebug.mesh);

    // ---- car ----
    const car = createRaycastVehicle(world, {
      id: 'ramp-car',
      position: { x: course.spawn.x, z: course.spawn.z },
      heading: course.spawn.heading,
      ...RAMP_VEHICLE_TUNING,
    });
    const carMesh = createCarMeshHelper({ color: C.car });
    scene.add(carMesh.group);

    const ai = {
      plan: null as VehicleState[] | null,
      planStartWall: performance.now(),
      goal: course.goal,
      lastExpansions: 0,
      lastBudgetMs: 0,
      lastUsedAffordance: false,
    };

    let pathLine: THREE.Line | null = null;
    function replacePathLine(path: VehicleState[]): void {
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
      if (k === 'j') setAffordanceOn((v) => !v);
      if (k === 'r') {
        car.teleport({
          x: course.spawn.x,
          z: course.spawn.z,
          heading: course.spawn.heading,
        });
        ai.plan = null;
      }
    };
    const onKeyUp = (e: KeyboardEvent) => keys.delete(e.key.toLowerCase());
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);

    // ---- replan ----
    function replan() {
      const now = performance.now();
      const state = car.readState(now);
      const t0 = performance.now();
      const res = planRampDemo({
        state: { ...state, t: 0 },
        goal: { ...course.goal, t: 0 },
        course,
        world: navWorld,
        withoutAffordances: !affordanceOnRef.current,
      });
      ai.lastBudgetMs = performance.now() - t0;
      ai.lastExpansions = res.stats.expansions;
      if (res.found && res.path.length > 1) {
        ai.plan = res.path;
        ai.planStartWall = now;
        ai.lastUsedAffordance = res.path.some(
          (_, i) =>
            i > 0 &&
            // Heuristic: an affordance step jumps the planner state by more
            // than any primitive could in a single tick.
            Math.hypot(
              res.path[i]!.x - res.path[i - 1]!.x,
              res.path[i]!.z - res.path[i - 1]!.z,
            ) > 10,
        );
        if (showPathRef.current) replacePathLine(res.path);
      }
      for (const arc of arcGroups) arc.visible = affordanceOnRef.current;
    }
    replanNowRef.current = replan;

    const replanTimer = window.setInterval(() => {
      if (pausedRef.current || playerDrivingRef.current) return;
      replan();
    }, REPLAN_INTERVAL_MS);

    // ---- resize ----
    const onResize = () => {
      const wW = window.innerWidth;
      const hH = window.innerHeight;
      renderer.setSize(wW, hH);
      camera.aspect = wW / hH;
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
              cruiseSpeed: RAMP_AGENT.maxSpeed,
              goalTolerance: 2,
              minTurnRadius: RAMP_AGENT.minTurnRadius,
            }),
          );
        } else {
          car.applyControls({ steer: 0, throttle: 0.2, brake: 0 });
        }
      } else {
        car.applyControls({ steer: 0, throttle: 0, brake: 0 });
      }

      const subDt = PHYSICS_DT / VEHICLE_SUBSTEPS;
      world.timestep = subDt;
      const wheelFilter = RAPIER.QueryFilterFlags.EXCLUDE_DYNAMIC;
      for (let s = 0; s < VEHICLE_SUBSTEPS; s++) {
        car.vehicle.updateVehicle(subDt, wheelFilter);
        world.step();
      }

      const after = car.readState(now);
      // syncCarMesh only updates X, Z, yaw. For the ramp demo we MUST also
      // sync world-Y and full chassis attitude (pitch/roll) — the car
      // physically climbs the ramp and flies through the air, and without
      // this the mesh stays glued to y=0 and looks "stuck behind the ramp"
      // even when physics is doing the right thing.
      syncCarMesh(carMesh.group, after);
      const tr = car.chassis.translation();
      const qr = car.chassis.rotation();
      carMesh.group.position.y = tr.y;
      carMesh.group.quaternion.set(qr.x, qr.y, qr.z, qr.w);
      if (pathLine) pathLine.visible = showPathRef.current;

      rapierDebug.mesh.visible = showRapierDebugRef.current;
      if (showRapierDebugRef.current) rapierDebug.update(world);

      debugGroup.visible = showDebugRef.current;
      if (showDebugRef.current) {
        carFootprint.position.set(after.x, 0, after.z);
        carFootprint.rotation.y = -after.heading;
      }

      if (chaseRef.current) {
        const cc = Math.cos(after.heading);
        const ss = Math.sin(after.heading);
        const cam = new THREE.Vector3(
          after.x - 14 * cc,
          7,
          after.z - 14 * ss,
        );
        camera.position.lerp(cam, 0.12);
        orbit.target.set(after.x, 1.5, after.z);
        orbit.update();
      }

      if (now - lastHudWall > 100) {
        lastHudWall = now;
        const v = Math.abs(after.speed).toFixed(1);
        const hdg = ((after.heading * 180) / Math.PI).toFixed(0);
        const y = car.chassis.translation().y;
        setHud(
          `${playerDrivingRef.current ? 'YOU' : 'AI'} · v=${v} m/s · hdg=${hdg}° · y=${y.toFixed(2)}m`,
        );
        setStatus(
          `plan=${ai.plan?.length ?? 0} exp=${ai.lastExpansions} budget=${ai.lastBudgetMs.toFixed(0)}ms aff=${ai.lastUsedAffordance ? 'used' : '—'}`,
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
          ramp + affordance
        </div>
        <div>{hud}</div>
        <div style={{ opacity: 0.8 }}>{status}</div>
        <div style={{ opacity: 0.6, marginTop: 6, fontSize: 11 }}>
          A real drivable heightfield ramp + a BallisticJump Affordance over a
          planner-only gap (pink). Toggle the affordance and watch the path
          flip from detour to jump. The car always drives physically.
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 8 }}>
          <ToggleButton
            label="affordance"
            shortcut="j"
            on={affordanceOn}
            onClick={() => setAffordanceOn((v) => !v)}
          />
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

function trimPlan(plan: VehicleState[], elapsed: number): VehicleState[] {
  let i = 0;
  while (i < plan.length - 1 && plan[i + 1]!.t <= elapsed) i++;
  return plan.slice(i);
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
