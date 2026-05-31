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
import { ensureRapier } from 'kinocat/adapters/rapier';
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
import { wheeledFromNormalized } from 'kinocat/vehicle/car';
import type { CarKinematicState } from 'kinocat/agent';
import {
  OBS_AGENT,
  OBS_BOUNDS,
  OBS_PALETTE as C,
  OBS_BLOCKS_ALL,
  type ObstacleCourse as Course,
  type ObsBlocks,
} from '../lib/obstaclecourse-scenarios';
import {
  createObstacleCourseScenario,
  OBSTACLE_FORCE_TUNING,
  PHYSICS_DT,
  type ObstacleCourseScenario,
} from '../lib/obstaclecourse-scenario';

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
      const scenario = await createObstacleCourseScenario({ blocks: blocksRef.current });
      if (disposed) {
        scenario.dispose();
        return;
      }
      cleanup = setupScene(mount, scenario);
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

  function setupScene(mount: HTMLDivElement, scenario: ObstacleCourseScenario): () => void {
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

    // ---- course state lives in the headless scenario ----
    // The scenario owns the Rapier world, car, colliders, planner and control
    // loop (identical to the headless tests). This component only renders the
    // visuals for `scenario.course` and uses `scenario.heightSampler` so the
    // terrain mesh matches the physics surface exactly.
    const world = scenario.getWorld();
    let course: Course = scenario.course;

    let courseVisuals = new THREE.Group();
    scene.add(courseVisuals);

    function buildCourseVisuals() {
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

      // Heightfield surface — uses the SAME sampler the scenario's physics
      // collider was built from (terrain ⊕ ramps), or a flat ground slab.
      const sampler = scenario.heightSampler;
      if (sampler) {
        courseVisuals.add(
          createHeightfieldMeshHelper({
            bounds: OBS_BOUNDS,
            sampler,
            segmentsX: 60,
            segmentsZ: 40,
            groundColor: C.ground,
            vertexColorAbove: course.ramps.length > 0 ? 0.2 : undefined,
            aboveColor: C.ramp,
          }),
        );
      } else {
        courseVisuals.add(
          createGroundPlaneHelper({ bounds: OBS_BOUNDS, color: 0x1a2233 }),
        );
      }

      for (const b of course.buildings) {
        courseVisuals.add(
          createBuildingHelper(b, { color: 0x3a4458, edgeColor: 0x6c7a94 }),
        );
      }

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
      scenario.rebuild(blocksRef.current);
      course = scenario.course;
      buildCourseVisuals();
      // Refresh inflated overlay too.
      debugGroup.clear();
      for (const b of course.buildings) {
        debugGroup.add(createInflatedObstacleHelper(b, 0.5));
      }
      debugGroup.add(createNavBoundsHelper(OBS_BOUNDS));
      debugGroup.add(carFootprint);
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

    // ---- car (owned by the scenario) ----
    const car = scenario.getCar();
    const carMesh = createCarMeshHelper({ color: C.car });
    scene.add(carMesh.group);

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
    buildCourseVisuals();
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
      if (k === 'r') scenario.reset();
    };
    const onKeyUp = (e: KeyboardEvent) => keys.delete(e.key.toLowerCase());
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    // NOTE: replanning is driven inside scenario.tick() on a SIM-time cadence —
    // no wall-clock setInterval (that was a source of headed↔headless drift).

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
    let lastPlanRef: CarKinematicState[] | null = null;
    let prevWall = performance.now();
    function tick() {
      if (stopped) return;
      requestAnimationFrame(tick);
      if (pausedRef.current) {
        renderer.render(scene, camera);
        return;
      }
      const now = performance.now();

      // Accumulate wall-time into whole fixed sim ticks (capped), then advance
      // the headless scenario — the SAME engine the tests drive. The component
      // never touches physics or the planner directly.
      const dt = Math.min((now - prevWall) / 1000, 1 / 30);
      prevWall = now;
      const steps = Math.max(1, Math.min(8, Math.round(dt / PHYSICS_DT)));
      for (let i = 0; i < steps; i++) {
        const playerControls = playerDrivingRef.current
          ? wheeledFromNormalized(
              {
                steer:
                  ((keys.has('a') || keys.has('arrowleft') ? 1 : 0) -
                    (keys.has('d') || keys.has('arrowright') ? 1 : 0)) *
                  0.55,
                throttle:
                  (keys.has('w') || keys.has('arrowup') ? 1 : 0) -
                  (keys.has('s') || keys.has('arrowdown') ? 1 : 0),
                brake: keys.has(' ') ? 1 : 0,
              },
              OBSTACLE_FORCE_TUNING,
            )
          : null;
        scenario.tick(playerControls);
      }

      const st = scenario.status();
      const after = st.state;
      syncCarMesh(carMesh.group, after);

      // Refresh the path line only when the plan actually changed.
      if (st.plan !== lastPlanRef) {
        lastPlanRef = st.plan;
        if (st.plan && st.plan.length > 1) replacePathLine(st.plan);
      }
      if (pathLine) pathLine.visible = showPathRef.current;

      debugGroup.visible = showDebugRef.current;
      rapierDebug.mesh.visible = showRapierDebugRef.current;
      if (showRapierDebugRef.current) rapierDebug.update(world);
      if (showDebugRef.current) {
        carFootprint.position.set(after.x, 0, after.z);
        carFootprint.rotation.y = -after.heading;
      }

      goalMarker.visible = !!st.goal;
      if (st.goal) goalMarker.position.set(st.goal.x, 2, st.goal.z);

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
          `plan=${st.plan?.length ?? 0} exp=${st.diagnostics.lastExpansions} replans=${st.diagnostics.successfulReplans}/${st.diagnostics.totalReplans} wp=${st.loopIndex}/${course.waypoints.length}`,
        );
      }

      renderer.render(scene, camera);
    }
    tick();

    return () => {
      stopped = true;
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('resize', onResize);
      scenario.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === mount)
        mount.removeChild(renderer.domElement);
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
