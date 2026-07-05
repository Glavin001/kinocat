'use client';

// Ramp + Affordance demo. A single drivable heightfield ramp the car
// physically climbs and launches off, plus a BallisticJump Affordance the
// planner can pick as a shortcut over a planner-only "gap" obstacle. Toggle
// the affordance from the HUD and watch the planned path change. Execution
// is always real Rapier physics — the car always drives, never poses.

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { ensureRapier } from 'kinocat/adapters/rapier';
import type { CarKinematicState } from 'kinocat/agent';
import {
  createCarMeshHelper,
  syncCarMesh,
  createGoalMarkerHelper,
  createInflatedObstacleHelper,
  createNavBoundsHelper,
  createAgentFootprintHelper,
  createJumpArcHelper,
  createHeightfieldMeshHelper,
  createRampChevronsHelper,
  createRapierDebugRenderer,
  updateChaseCamera,
} from 'kinocat/adapters/three';
import { wheeledFromNormalized } from 'kinocat/vehicle/car';
import {
  RAMP_AGENT,
  RAMP_BOUNDS,
  RAMP_PALETTE as C,
  rampHeightSampler,
} from '../lib/ramp-scenarios';
import {
  createRampScenario,
  RAMP_FORCE_TUNING,
  PHYSICS_DT,
  type RampScenario,
} from '../lib/ramp-scenario';

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
  const affordanceToggleRef = useRef<((on: boolean) => void) | null>(null);
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
      const scenario = await createRampScenario({ affordance: affordanceOnRef.current });
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

  // Swap the affordance on the scenario when the toggle changes so the user
  // sees the path flip immediately.
  useEffect(() => {
    affordanceToggleRef.current?.(affordanceOn);
  }, [affordanceOn]);

  function setupScene(mount: HTMLDivElement, scenario: RampScenario): () => void {
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

    // Course + physics + planner live in the headless scenario (the same one
    // the tests drive). This component only renders.
    const course = scenario.course;
    const world = scenario.getWorld();
    const sampler = rampHeightSampler(course.ramps);

    // ---- visuals: heightfield mesh + arc + gap + waypoints ----
    const courseVisuals = new THREE.Group();
    scene.add(courseVisuals);

    courseVisuals.add(
      createHeightfieldMeshHelper({
        bounds: RAMP_BOUNDS,
        sampler,
        segmentsX: 120,
        segmentsZ: 60,
        groundColor: C.ground,
        vertexColorAbove: 0.05,
        aboveColor: C.ramp,
      }),
    );
    for (const r of course.ramps) {
      courseVisuals.add(createRampChevronsHelper(r));
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

    // ---- car (owned by the scenario) ----
    const car = scenario.getCar();
    const carMesh = createCarMeshHelper({ color: C.car });
    scene.add(carMesh.group);

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
      if (k === 'r') scenario.reset();
    };
    const onKeyUp = (e: KeyboardEvent) => keys.delete(e.key.toLowerCase());
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);

    // Affordance toggle: hand it to the scenario (forces a replan so the path
    // flips) and update the arc overlays. Replanning itself runs inside
    // scenario.tick() on a SIM-time cadence — no wall-clock setInterval.
    affordanceToggleRef.current = (on: boolean) => {
      scenario.setAffordance(on);
      for (const arc of arcGroups) arc.visible = on;
    };
    for (const arc of arcGroups) arc.visible = affordanceOnRef.current;

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

      // Accumulate wall-time into whole fixed sim ticks, then advance the
      // headless scenario — the SAME engine the tests drive.
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
              RAMP_FORCE_TUNING,
            )
          : null;
        scenario.tick(playerControls);
      }

      const st = scenario.status();
      const after = st.state;
      // syncCarMesh only updates X, Z, yaw. For the ramp demo we MUST also
      // sync world-Y and full chassis attitude (pitch/roll) — the car
      // physically climbs the ramp and flies through the air.
      syncCarMesh(carMesh.group, after);
      const tr = car.chassis.translation();
      const qr = car.chassis.rotation();
      carMesh.group.position.y = tr.y;
      carMesh.group.quaternion.set(qr.x, qr.y, qr.z, qr.w);

      if (st.plan !== lastPlanRef) {
        lastPlanRef = st.plan;
        if (st.plan && st.plan.length > 1) replacePathLine(st.plan);
      }
      if (pathLine) pathLine.visible = showPathRef.current;

      rapierDebug.mesh.visible = showRapierDebugRef.current;
      if (showRapierDebugRef.current) rapierDebug.update(world);

      debugGroup.visible = showDebugRef.current;
      if (showDebugRef.current) {
        carFootprint.position.set(after.x, 0, after.z);
        carFootprint.rotation.y = -after.heading;
      }

      if (chaseRef.current) {
        updateChaseCamera(camera, { x: after.x, z: after.z, heading: after.heading }, { orbit });
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
          `plan=${st.plan?.length ?? 0} exp=${st.diagnostics.lastExpansions} replans=${st.diagnostics.successfulReplans}/${st.diagnostics.totalReplans} aff=${st.diagnostics.usedAffordance ? 'used' : '—'}`,
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
