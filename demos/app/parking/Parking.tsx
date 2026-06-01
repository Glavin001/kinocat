'use client';

// Tight-parking demo: three progressively-harder parking scenarios driven
// by the SAME `createRaceScenario` runner the headless `controller-bench`
// CLI uses. Web demo ↔ CLI parity: a scenario that passes the bench
// works in the browser exactly the same way.
//
// [f] toggles a footprint overlay — the green box is the ego's collision
// footprint (matches the car mesh + the chassis the planner reserves clearance
// for) and the red boxes are the obstacle polygons the planner avoids. Use it
// to confirm the car isn't clipping (the boxes never overlap). Playback runs at
// SIM_SPEED× so the slow 2 m/s precision maneuver is watchable.
//
// Three scenarios cycle via the [1] / [2] / [3] keys (or HUD buttons):
//   1. forward pull-in (easy)
//   2. reverse perpendicular (medium)
//   3. parallel parking (hard)
//
// Behaviour comes from the shared `createRaceScenario` runner — the
// planner (sub-meter discretisation with terminal heading constraint via
// `planRace`), the multi-cusp segment executor, and the pure-pursuit
// tracker — all configured through the same `RaceTuning` overrides the
// CLI bench uses. Press [r] to reset, [p] to pause, [l] to toggle path
// rendering, [d] to toggle the parked-car clearance overlay.

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import {
  createGroundPlaneHelper,
  createCarMeshHelper,
  syncCarMesh,
  createGoalMarkerHelper,
} from 'kinocat/adapters/three';
import {
  PARKING_BOUNDS,
  PARKING_PALETTE as C,
  PARKING_SCENARIOS,
  PARKING_LABELS,
  PARKING_AGENT,
  buildParkingScenario,
  parkingLibrary,
  parkingCourse,
  parkingScenarioOptions,
  evaluateParked,
  type ParkingScenarioId,
  type ParkingScenario,
} from '../lib/parking-scenarios';
import {
  createRaceScenario,
  type RaceScenario,
  type RaceEntry,
} from '../lib/race-scenario';

function parkingEntry(name: string): RaceEntry {
  return { name, lib: parkingLibrary() };
}

// Playback rate. Parking cruises at the planner's 2 m/s precision speed, so
// realtime (1×) reads as a glacial creep — a 25 s reverse-S takes 25 s of
// wall-clock to watch. Step the fixed-rate physics this many sim ticks per
// rendered frame so the maneuver plays at a brisk pace. Physics is still the
// same deterministic 1/60 s tick; we just run several per frame.
const SIM_SPEED = 4;

/** World-XZ corners of the agent footprint placed at a pose (heading 0 = +x). */
function placeFootprintXZ(
  footprint: ReadonlyArray<readonly [number, number]>,
  x: number,
  z: number,
  heading: number,
): THREE.Vector3[] {
  const c = Math.cos(heading);
  const s = Math.sin(heading);
  return footprint.map(
    ([fx, fz]) => new THREE.Vector3(x + fx * c - fz * s, 0.4, z + fx * s + fz * c),
  );
}

export default function Parking() {
  const mountRef = useRef<HTMLDivElement>(null);
  const [scenarioId, setScenarioId] = useState<ParkingScenarioId>('forward-pullin');
  const [paused, setPaused] = useState(false);
  const [showPath, setShowPath] = useState(true);
  const [showFootprints, setShowFootprints] = useState(false);
  const [hud, setHud] = useState('');
  const [status, setStatus] = useState('initialising...');

  const scenarioIdRef = useRef(scenarioId);
  const pausedRef = useRef(paused);
  const showPathRef = useRef(showPath);
  const showFootprintsRef = useRef(showFootprints);
  scenarioIdRef.current = scenarioId;
  pausedRef.current = paused;
  showPathRef.current = showPath;
  showFootprintsRef.current = showFootprints;

  useEffect(() => {
    if (!mountRef.current) return;
    const host = mountRef.current;

    // Three.js scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(C.bg);
    scene.fog = new THREE.Fog(C.fog, 30, 90);
    const camera = new THREE.PerspectiveCamera(50, host.clientWidth / host.clientHeight, 0.1, 200);
    camera.position.set(20, 20, 25);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(host.clientWidth, host.clientHeight);
    host.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0);

    // Lights
    scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const sun = new THREE.DirectionalLight(0xffffff, 0.7);
    sun.position.set(20, 30, 15);
    scene.add(sun);

    // Ground grid
    scene.add(createGroundPlaneHelper({ bounds: PARKING_BOUNDS, color: parseInt(C.ground.slice(1), 16) }));

    // Per-scenario meshes
    const scenarioMeshes: THREE.Object3D[] = [];
    let pathLine: THREE.Line | null = null;
    let carMesh: ReturnType<typeof createCarMeshHelper> | null = null;
    let goalMesh: THREE.Mesh | null = null;
    let raceScenario: RaceScenario | null = null;
    // Current scenario geometry — held so the frame loop can read `targetStall`
    // for the shared `evaluateParked` "in-the-stall" check that drives the HUD.
    let currentScenario: ParkingScenario | null = null;
    let cancelled = false;
    // Monotonic token guarding the async scenario build. `buildScenario` awaits
    // `createRaceScenario`, so two builds can be in flight at once (mount fires
    // the initial build and a fast scenario switch fires another). Each build
    // captures the token it started with; when it resumes after the await it
    // bails if a newer build has superseded it — otherwise the stale build's
    // car mesh / runner leak into the scene as a static "ghost" ego car.
    let buildToken = 0;

    // Footprint-overlay group: the polygons the PLANNER actually reasons about
    // — the ego's collision footprint (so you can see it matches the car mesh)
    // and every obstacle polygon (the parked cars + walls as the planner sees
    // them). Toggled with [f]; the obstacle outlines are rebuilt per scenario,
    // the ego outline is re-placed every frame from the chassis pose.
    const fpGroup = new THREE.Group();
    fpGroup.visible = showFootprintsRef.current;
    scene.add(fpGroup);
    const fpObstacleLines: THREE.Line[] = [];
    let egoFpLine: THREE.LineLoop | null = null;
    const fpEgoMat = new THREE.LineBasicMaterial({ color: 0x55ff88 });
    const fpObsMat = new THREE.LineBasicMaterial({ color: 0xff6688, transparent: true, opacity: 0.8 });

    function clearFootprintOverlay() {
      for (const l of fpObstacleLines) {
        fpGroup.remove(l);
        l.geometry.dispose();
      }
      fpObstacleLines.length = 0;
      if (egoFpLine) {
        fpGroup.remove(egoFpLine);
        egoFpLine.geometry.dispose();
        egoFpLine = null;
      }
    }

    function buildFootprintOverlay(id: ParkingScenarioId) {
      clearFootprintOverlay();
      // Obstacle polygons exactly as passed to the planner.
      const course = parkingCourse(id);
      for (const poly of course.obstacles) {
        const pts = poly.map(([x, z]) => new THREE.Vector3(x, 0.42, z));
        const geo = new THREE.BufferGeometry().setFromPoints(pts);
        const line = new THREE.LineLoop(geo, fpObsMat);
        fpGroup.add(line);
        fpObstacleLines.push(line);
      }
      // Ego footprint outline (placed each frame).
      const egoGeo = new THREE.BufferGeometry().setFromPoints(
        placeFootprintXZ(PARKING_AGENT.footprint, 0, 0, 0),
      );
      egoFpLine = new THREE.LineLoop(egoGeo, fpEgoMat);
      fpGroup.add(egoFpLine);
    }

    function clearScenarioMeshes() {
      for (const m of scenarioMeshes) scene.remove(m);
      scenarioMeshes.length = 0;
      if (pathLine) {
        scene.remove(pathLine);
        pathLine.geometry.dispose();
        (pathLine.material as THREE.Material).dispose();
        pathLine = null;
      }
      if (carMesh) {
        scene.remove(carMesh.group);
        carMesh = null;
      }
      if (goalMesh) {
        scene.remove(goalMesh);
        goalMesh.geometry.dispose();
        (goalMesh.material as THREE.Material).dispose();
        goalMesh = null;
      }
    }

    function renderObstacles(s: ParkingScenario) {
      // Parked cars
      for (const pk of s.parkedCars) {
        const geo = new THREE.BoxGeometry(pk.hx * 2, 1.2, pk.hz * 2);
        const mat = new THREE.MeshStandardMaterial({
          color: parseInt(C.parkedCar.slice(1), 16),
          metalness: 0.1,
          roughness: 0.7,
        });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(pk.x, 0.6, pk.z);
        mesh.rotation.y = -pk.heading;
        scene.add(mesh);
        scenarioMeshes.push(mesh);
      }
      // Walls
      for (const w of s.walls) {
        const geo = new THREE.BoxGeometry(w.hx * 2, 0.6, w.hz * 2);
        const mat = new THREE.MeshStandardMaterial({
          color: parseInt(C.curb.slice(1), 16),
          metalness: 0.05,
          roughness: 0.9,
        });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(w.x, 0.3, w.z);
        scene.add(mesh);
        scenarioMeshes.push(mesh);
      }
      // Target stall outline
      const outlineGeo = new THREE.PlaneGeometry(s.targetStall.hx * 2, s.targetStall.hz * 2);
      const outlineMat = new THREE.MeshBasicMaterial({
        color: parseInt(C.stallLine.slice(1), 16),
        transparent: true,
        opacity: 0.18,
        side: THREE.DoubleSide,
      });
      const outline = new THREE.Mesh(outlineGeo, outlineMat);
      outline.rotation.x = -Math.PI / 2;
      outline.rotation.z = -s.targetStall.heading;
      outline.position.set(s.targetStall.x, 0.02, s.targetStall.z);
      scene.add(outline);
      scenarioMeshes.push(outline);
    }

    async function buildScenario(id: ParkingScenarioId) {
      const myToken = ++buildToken;
      clearScenarioMeshes();
      if (raceScenario) {
        raceScenario.dispose();
        raceScenario = null;
      }
      const s = buildParkingScenario(id);
      currentScenario = s;
      renderObstacles(s);
      buildFootprintOverlay(id);
      // Goal marker.
      goalMesh = createGoalMarkerHelper({ color: C.goal, size: 1.5 });
      goalMesh.position.set(s.goal.x, 0.3, s.goal.z);
      scene.add(goalMesh);
      // Build the shared scenario runner — same code path as the
      // controller-bench CLI.
      // Canonical parking options shared with the CLI bench + Vitest tests
      // (incl. zero teleportation — no stall/off-track rescue masking a stuck
      // maneuver). The page is a thin view over the exact same config.
      const instance = await createRaceScenario(parkingScenarioOptions(id, [parkingEntry('ego')]));
      // A newer build (or unmount) superseded us while we awaited — drop this
      // runner instead of leaking it (and its car mesh) into the scene.
      if (cancelled || myToken !== buildToken) {
        instance.dispose();
        return;
      }
      raceScenario = instance;
      // Car mesh attached to the chassis from the scenario.
      const handle = raceScenario.getCarHandle('ego');
      if (handle) {
        carMesh = createCarMeshHelper({ color: C.ego });
        scene.add(carMesh.group);
      }
      setStatus(`scenario: ${PARKING_LABELS[id]}`);
    }

    function refreshPathLine(plan: ReadonlyArray<{ x: number; z: number }> | null) {
      if (pathLine) {
        scene.remove(pathLine);
        pathLine.geometry.dispose();
        (pathLine.material as THREE.Material).dispose();
        pathLine = null;
      }
      if (!plan || plan.length < 2 || !showPathRef.current) return;
      const pts = plan.map((p) => new THREE.Vector3(p.x, 0.35, p.z));
      const geo = new THREE.BufferGeometry().setFromPoints(pts);
      const mat = new THREE.LineBasicMaterial({
        color: C.egoPath,
        transparent: true,
        opacity: 0.85,
      });
      pathLine = new THREE.Line(geo, mat);
      scene.add(pathLine);
    }

    // Animation loop
    let prev = performance.now();
    function frame() {
      if (cancelled) return;
      const now = performance.now();
      const dt = Math.min((now - prev) / 1000, 1 / 30);
      prev = now;
      // Footprint overlay visibility follows the toggle every frame (so it
      // responds even while paused).
      fpGroup.visible = showFootprintsRef.current;
      if (!pausedRef.current && raceScenario) {
        // The shared runner uses a fixed-step physics tick (1/60s); accumulate
        // wall-time deltas into discrete sim ticks, scaled by SIM_SPEED so the
        // slow 2 m/s parking maneuver plays back at a watchable pace.
        const baseSteps = Math.max(1, Math.round(dt / (1 / 60)));
        const stepsThisFrame = Math.min(8 * SIM_SPEED, baseSteps * SIM_SPEED);
        for (let i = 0; i < stepsThisFrame; i++) {
          raceScenario.tick();
        }
        const s = raceScenario.status()[0];
        if (s && carMesh) {
          syncCarMesh(carMesh.group, s.state);
          if (egoFpLine && showFootprintsRef.current) {
            egoFpLine.geometry.setFromPoints(
              placeFootprintXZ(PARKING_AGENT.footprint, s.state.x, s.state.z, s.state.heading),
            );
          }
          refreshPathLine(s.plan);
          const goalDist = (() => {
            const c = parkingCourse(scenarioIdRef.current);
            const wp = c?.waypoints[0];
            if (!wp) return NaN;
            return Math.hypot(s.state.x - wp.x, s.state.z - wp.z);
          })();
          setHud(
            `v=${s.state.speed.toFixed(2)} m/s · goal=${goalDist.toFixed(2)} m · replans=${s.diagnostics.totalReplans} · plan-ms=${s.diagnostics.lastReplanMs.toFixed(0)}`,
          );
          // Honest PARKED status from the shared `evaluateParked` predicate:
          // the car must actually sit inside the stall silhouette, squared up,
          // and stopped — NOT merely be within arrive-radius of the goal point
          // (which is all the old `laps.length >= 1` check tested). A car that
          // arrives at the slot but comes to rest offset or angled reads NOT
          // PARKED, exactly as it would earn a real-world ticket. The verdict is
          // gated on lap completion (the car has reached the slot) so it doesn't
          // flash at the at-rest spawn.
          if (currentScenario) {
            const ev = evaluateParked(s.state, currentScenario);
            const cov = `${(ev.coverage * 100).toFixed(0)}% in stall`;
            if (ev.parked) {
              const lapT = s.laps[0] ? ` · ${s.laps[0].duration.toFixed(2)}s` : '';
              setStatus(`PARKED ✓ · ${cov}${lapT}`);
            } else if (s.laps.length >= 1) {
              const hdgDeg = `${((ev.headingError * 180) / Math.PI).toFixed(0)}° off`;
              setStatus(`NOT PARKED · ${cov} · ${hdgDeg}`);
            } else {
              setStatus(`parking… · goal=${goalDist.toFixed(2)} m · ${cov}`);
            }
          }
        }
      }
      controls.update();
      renderer.render(scene, camera);
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);

    rebuildRef.current = (id) => void buildScenario(id);
    setReady(true);

    function onKey(e: KeyboardEvent) {
      if (e.key === '1') setScenarioId('forward-pullin');
      else if (e.key === '2') setScenarioId('reverse-perp');
      else if (e.key === '3') setScenarioId('parallel');
      else if (e.key === 'r' || e.key === 'R') rebuildRef.current?.(scenarioIdRef.current);
      else if (e.key === 'p' || e.key === 'P') setPaused((p) => !p);
      else if (e.key === 'l' || e.key === 'L') setShowPath((s) => !s);
      else if (e.key === 'f' || e.key === 'F') setShowFootprints((s) => !s);
    }
    window.addEventListener('keydown', onKey);

    function onResize() {
      if (!host) return;
      camera.aspect = host.clientWidth / host.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(host.clientWidth, host.clientHeight);
    }
    window.addEventListener('resize', onResize);

    return () => {
      cancelled = true;
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('resize', onResize);
      clearScenarioMeshes();
      clearFootprintOverlay();
      scene.remove(fpGroup);
      if (raceScenario) raceScenario.dispose();
      renderer.dispose();
      host.removeChild(renderer.domElement);
    };
  // Mount-once effect — scenarios are switched via `rebuildRef`.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const rebuildRef = useRef<((id: ParkingScenarioId) => void) | null>(null);
  const [, setReady] = useState(false);

  useEffect(() => {
    rebuildRef.current?.(scenarioId);
  }, [scenarioId]);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100vh' }}>
      <div ref={mountRef} style={{ width: '100%', height: '100%' }} />
      <div
        style={{
          position: 'absolute',
          top: 12,
          left: 12,
          color: '#cde',
          fontFamily: 'ui-monospace, monospace',
          fontSize: 12,
          background: 'rgba(10,13,20,0.85)',
          padding: '10px 12px',
          borderRadius: 6,
          maxWidth: 340,
          lineHeight: 1.5,
        }}
      >
        <div style={{ color: '#55dcff', fontWeight: 600, marginBottom: 4 }}>
          tight parking — bench-parity runner
        </div>
        <div>{status}</div>
        <div style={{ opacity: 0.75, marginTop: 4 }}>{hud}</div>
        <div style={{ marginTop: 8, opacity: 0.65, fontSize: 11 }}>
          Same `createRaceScenario` runner as the controller-bench CLI:
          identical planner discretisation, multi-cusp segment executor,
          and pure-pursuit tracker. What passes the bench works here.
        </div>
        <div style={{ marginTop: 10 }}>
          {PARKING_SCENARIOS.map((id) => (
            <button
              key={id}
              onClick={() => setScenarioId(id)}
              style={{
                display: 'block',
                width: '100%',
                marginBottom: 4,
                padding: '6px 8px',
                background: scenarioId === id ? '#1c2840' : '#0e1622',
                color: '#cde',
                border: '1px solid ' + (scenarioId === id ? '#55dcff' : '#1c2840'),
                borderRadius: 4,
                cursor: 'pointer',
                fontFamily: 'inherit',
                fontSize: 11,
                textAlign: 'left',
              }}
            >
              {PARKING_LABELS[id]}
            </button>
          ))}
        </div>
        <div style={{ marginTop: 8, display: 'flex', gap: 6, flexWrap: 'wrap', fontSize: 11 }}>
          <button onClick={() => setPaused((p) => !p)}>{`[p] ${paused ? 'paused' : 'running'}`}</button>
          <button onClick={() => setShowPath((s) => !s)}>{`[l] path ${showPath ? 'on' : 'off'}`}</button>
          <button onClick={() => setShowFootprints((s) => !s)}>{`[f] footprints ${showFootprints ? 'on' : 'off'}`}</button>
          <button onClick={() => rebuildRef.current?.(scenarioId)}>[r] reset</button>
        </div>
        <div style={{ marginTop: 6, opacity: 0.5, fontSize: 10 }}>
          [1] [2] [3] scenarios · [r] reset · [p] pause · [l] path · [f] footprints
        </div>
        <div style={{ marginTop: 4, opacity: 0.5, fontSize: 10 }}>
          footprints: <span style={{ color: '#55ff88' }}>green</span> = car collision box ·{' '}
          <span style={{ color: '#ff6688' }}>red</span> = obstacle boxes the planner avoids
        </div>
      </div>
    </div>
  );
}
