'use client';

// Tight-parking demo: three progressively-harder parking scenarios driven
// by the SAME `createRaceScenario` runner the headless `controller-bench`
// CLI uses. Web demo ↔ CLI parity: a scenario that passes the bench
// works in the browser exactly the same way.
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
  buildParkingScenario,
  parkingLibrary,
  type ParkingScenarioId,
  type ParkingScenario,
} from '../lib/parking-scenarios';
import {
  createRaceScenario,
  type RaceScenario,
  type RaceEntry,
} from '../lib/race-scenario';

const PARKING_BENCH_TUNING = {
  // Same knobs the CLI bench uses for parking entries — identical
  // behaviour browser↔CLI by construction.
  cruiseSpeed: 2,
  goalTolerance: 0.4,
  arriveRadius: 0.6,
  plannerPosCell: 0.3,
  plannerHeadingBuckets: 36,
  plannerGoalRadius: 0.35,
  plannerGoalHeadingTol: 0.2,
  plannerBudgetMs: 500,
  plannerMaxExpansions: 80_000,
  mpcWTerminalPosition: 50,
  mpcWTerminalSpeed: 30,
};

function parkingEntry(name: string): RaceEntry {
  return { name, lib: parkingLibrary() };
}

/** Convert a parking scenario to the `createRaceScenario` course shape
 *  — single goal pose at speed=0 signals "terminal pose intent" to the
 *  planner. */
function parkingCourse(s: ParkingScenario): import('../lib/race-scenario').RaceScenarioOptions['course'] {
  return {
    bounds: { x0: s.bounds.x0, x1: s.bounds.x1, z0: s.bounds.z0, z1: s.bounds.z1 },
    polygons: s.polygons,
    obstacles: s.obstacles,
    waypoints: [{ ...s.goal, speed: 0, t: 0 }],
    spawn: { ...s.spawn, speed: 0, t: 0 },
  };
}

export default function Parking() {
  const mountRef = useRef<HTMLDivElement>(null);
  const [scenarioId, setScenarioId] = useState<ParkingScenarioId>('forward-pullin');
  const [paused, setPaused] = useState(false);
  const [showPath, setShowPath] = useState(true);
  const [hud, setHud] = useState('');
  const [status, setStatus] = useState('initialising...');

  const scenarioIdRef = useRef(scenarioId);
  const pausedRef = useRef(paused);
  const showPathRef = useRef(showPath);
  scenarioIdRef.current = scenarioId;
  pausedRef.current = paused;
  showPathRef.current = showPath;

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
    let cancelled = false;

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
      clearScenarioMeshes();
      if (raceScenario) {
        raceScenario.dispose();
        raceScenario = null;
      }
      const s = buildParkingScenario(id);
      renderObstacles(s);
      // Goal marker.
      goalMesh = createGoalMarkerHelper({ color: C.goal, size: 1.5 });
      goalMesh.position.set(s.goal.x, 0.3, s.goal.z);
      scene.add(goalMesh);
      // Build the shared scenario runner — same code path as the
      // controller-bench CLI.
      raceScenario = await createRaceScenario({
        entries: [parkingEntry('ego')],
        targetLaps: 1,
        syncHold: false,
        offTrackRecovery: 'none',
        tuning: PARKING_BENCH_TUNING,
        course: parkingCourse(s),
      });
      // Car mesh attached to the chassis from the scenario.
      const handle = raceScenario.getCarHandle('ego');
      if (handle) {
        carMesh = createCarMeshHelper({ color: C.ego });
        scene.add(carMesh.group);
      }
      setStatus(`scenario: ${PARKING_LABELS[id]}`);
    }

    void buildScenario(scenarioId);

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
      if (!pausedRef.current && raceScenario) {
        // The shared runner uses a fixed-step physics tick (1/60s);
        // accumulate wall-time deltas into discrete sim ticks.
        const stepsThisFrame = Math.max(1, Math.min(8, Math.round(dt / (1 / 60))));
        for (let i = 0; i < stepsThisFrame; i++) {
          raceScenario.tick();
        }
        const s = raceScenario.status()[0];
        if (s && carMesh) {
          syncCarMesh(carMesh.group, s.state);
          refreshPathLine(s.plan);
          const goalDist = (() => {
            const c = parkingCourse(buildParkingScenario(scenarioIdRef.current));
            const wp = c?.waypoints[0];
            if (!wp) return NaN;
            return Math.hypot(s.state.x - wp.x, s.state.z - wp.z);
          })();
          setHud(
            `v=${s.state.speed.toFixed(2)} m/s · goal=${goalDist.toFixed(2)} m · replans=${s.diagnostics.totalReplans} · plan-ms=${s.diagnostics.lastReplanMs.toFixed(0)}`,
          );
          if (s.laps.length >= 1 && s.laps[0]) setStatus(`PARKED · ${s.laps[0].duration.toFixed(2)}s`);
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
        <div style={{ marginTop: 8, display: 'flex', gap: 6, fontSize: 11 }}>
          <button onClick={() => setPaused((p) => !p)}>{`[p] ${paused ? 'paused' : 'running'}`}</button>
          <button onClick={() => setShowPath((s) => !s)}>{`[l] path ${showPath ? 'on' : 'off'}`}</button>
          <button onClick={() => rebuildRef.current?.(scenarioId)}>[r] reset</button>
        </div>
        <div style={{ marginTop: 6, opacity: 0.5, fontSize: 10 }}>
          [1] [2] [3] scenarios · [r] reset · [p] pause · [l] path
        </div>
      </div>
    </div>
  );
}
