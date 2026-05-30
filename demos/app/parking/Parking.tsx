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
  createAgentFootprintHelper,
} from 'kinocat/adapters/three';
import { samplePlanAt } from 'kinocat/vehicle/car';
import {
  PARKING_BOUNDS,
  PARKING_PALETTE as C,
  PARKING_SCENARIOS,
  PARKING_LABELS,
  PARKING_AGENT,
  PARKING_FOOTPRINT_INFLATE,
  PARKING_GOAL_TOL,
  buildParkingScenario,
  checkParkingGoal,
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
  arriveRadius: PARKING_GOAL_TOL.posM,
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

// Goal satisfaction display thresholds — same `PARKING_GOAL_TOL`
// constants the bench's pass criterion uses (via `checkParkingGoal`).
// The HUD just renders the per-axis result in degrees for human-
// readable headings. Diverging here would mean the CLI claims success
// while the demo shows failure (or vice versa) — exactly the bug we
// solved by centralising the criterion.
const GOAL_POS_TOL = PARKING_GOAL_TOL.posM;
const GOAL_HEADING_TOL_DEG = PARKING_GOAL_TOL.hdgRad * (180 / Math.PI);
const GOAL_SPEED_TOL = PARKING_GOAL_TOL.speedMS;

interface ParkingHud {
  speed: number;
  goalDist: number;
  headingError: number;
  totalReplans: number;
  lastReplanMs: number;
  steer: number;
  throttle: number;
  brake: number;
  targetSpeed: number;
  trackingErrorRms: number;
  planAgeMs: number;
  consecutiveFailedReplans: number;
  predErrorRms: number;
  planLength: number;
  activeSegmentIndex: number;
  totalSegments: number;
  activeSegmentGear: 'fwd' | 'rev' | 'unknown';
  finished: boolean;
  /** Per-axis pass flags from `checkParkingGoal` — the SAME criterion
   *  the headless bench's pass condition uses. The HUD shows ✓/✗
   *  badges driven directly by these so browser↔CLI agree. */
  posOk: boolean;
  hdgOk: boolean;
  spdOk: boolean;
  allOk: boolean;
  /** Distance (m) between chassis and the plan's predicted-pose-at-now
   *  (the "ghost car"). NaN when ghost is hidden or no plan. */
  ghostDivM: number;
}

export default function Parking() {
  const mountRef = useRef<HTMLDivElement>(null);
  const [scenarioId, setScenarioId] = useState<ParkingScenarioId>('forward-pullin');
  const [paused, setPaused] = useState(false);
  const [showPath, setShowPath] = useState(true);
  const [showGhost, setShowGhost] = useState(true);
  const [debug, setDebug] = useState(false);
  const [hud, setHud] = useState<ParkingHud | null>(null);
  const [status, setStatus] = useState('initialising...');

  const scenarioIdRef = useRef(scenarioId);
  const pausedRef = useRef(paused);
  const showPathRef = useRef(showPath);
  const showGhostRef = useRef(showGhost);
  const debugRef = useRef(debug);
  scenarioIdRef.current = scenarioId;
  pausedRef.current = paused;
  showPathRef.current = showPath;
  showGhostRef.current = showGhost;
  debugRef.current = debug;

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
    // "Ghost car" — translucent chassis rendered at the pose the plan
    // PREDICTS the chassis will be in right now (`samplePlanAt(plan,
    // simTime - planStartSimTime)`). When the real chassis sits on
    // top of the ghost, the tracker is faithfully executing the plan.
    // When the ghost moves toward the goal but the real chassis moves
    // away (or stalls), we know the plan was fine but execution
    // diverged — separates planner bugs from tracker bugs visually.
    let ghostMesh: ReturnType<typeof createCarMeshHelper> | null = null;
    let goalMesh: THREE.Mesh | null = null;
    let raceScenario: RaceScenario | null = null;
    let cachedGoalWp: { x: number; z: number; heading: number } | null = null;
    let lastPlanRef: ReadonlyArray<{ x: number; z: number }> | null = null;
    let cancelled = false;
    let buildGeneration = 0;
    let hudFrameCounter = 0;

    // Debug overlays: ego footprint wireframe + obstacle clearance rings.
    const footprintLine = createAgentFootprintHelper(PARKING_AGENT.footprint, {
      color: C.ego,
    });
    footprintLine.visible = false;
    scene.add(footprintLine);

    let clearanceGroup: THREE.Group = new THREE.Group();
    clearanceGroup.visible = false;
    scene.add(clearanceGroup);

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
      if (ghostMesh) {
        scene.remove(ghostMesh.group);
        // Materials are clones we own — dispose to avoid leaks across rebuilds.
        ghostMesh.group.traverse((obj) => {
          if (obj instanceof THREE.Mesh) {
            const mat = obj.material;
            if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
            else mat.dispose();
          }
        });
        ghostMesh = null;
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

      // Rebuild obstacle clearance rings (pink outlines showing the
      // planner's footprintInflate band around each obstacle polygon).
      scene.remove(clearanceGroup);
      clearanceGroup = new THREE.Group();
      clearanceGroup.visible = debugRef.current;
      for (const poly of s.obstacles) {
        let x0 = Infinity, x1 = -Infinity, z0 = Infinity, z1 = -Infinity;
        for (const [px, pz] of poly) {
          if (px < x0) x0 = px;
          if (px > x1) x1 = px;
          if (pz < z0) z0 = pz;
          if (pz > z1) z1 = pz;
        }
        const inf = PARKING_FOOTPRINT_INFLATE;
        const y = 0.12;
        const ring = [
          new THREE.Vector3(x0 - inf, y, z0 - inf),
          new THREE.Vector3(x1 + inf, y, z0 - inf),
          new THREE.Vector3(x1 + inf, y, z1 + inf),
          new THREE.Vector3(x0 - inf, y, z1 + inf),
          new THREE.Vector3(x0 - inf, y, z0 - inf),
        ];
        clearanceGroup.add(new THREE.Line(
          new THREE.BufferGeometry().setFromPoints(ring),
          new THREE.LineBasicMaterial({ color: 0xff66aa, transparent: true, opacity: 0.6 }),
        ));
      }
      scene.add(clearanceGroup);
    }

    async function buildScenario(id: ParkingScenarioId) {
      const gen = ++buildGeneration;
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
      const rs = await createRaceScenario({
        entries: [parkingEntry('ego')],
        targetLaps: 1,
        syncHold: false,
        offTrackRecovery: 'none',
        tuning: PARKING_BENCH_TUNING,
        course: parkingCourse(s),
      });
      // Guard: a newer build may have started while we awaited.
      if (gen !== buildGeneration) { rs.dispose(); return; }
      raceScenario = rs;
      cachedGoalWp = { x: s.goal.x, z: s.goal.z, heading: s.goal.heading };
      // Car mesh attached to the chassis from the scenario.
      const handle = raceScenario.getCarHandle('ego');
      if (handle) {
        carMesh = createCarMeshHelper({ color: C.ego });
        scene.add(carMesh.group);
        // Ghost mesh: same geometry, translucent. Clone every material
        // up-front so dimming the ghost doesn't dim the real car.
        ghostMesh = createCarMeshHelper({ color: 0xffaa33 });
        ghostMesh.group.traverse((obj) => {
          if (obj instanceof THREE.Mesh) {
            const mat = obj.material as THREE.MeshStandardMaterial;
            const ghost = mat.clone();
            ghost.transparent = true;
            ghost.opacity = 0.35;
            ghost.depthWrite = false;
            obj.material = ghost;
          }
        });
        scene.add(ghostMesh.group);
      }
      setStatus(`scenario: ${PARKING_LABELS[id]}`);
    }

    // Initial build is triggered by the [scenarioId] effect via rebuildRef.
    // Do NOT also call buildScenario here — the two concurrent async builds
    // race and produce duplicate car meshes.

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
          // Ghost car: pose the plan PREDICTS for the chassis right now.
          // If the plan has been executing for `elapsed` sim seconds,
          // `samplePlanAt(plan, elapsed)` is exactly what the plan said
          // the chassis would be doing. Compare visually to the real
          // mesh to separate planner-wrong from tracker-wrong bugs.
          if (ghostMesh) {
            let ghostState: { x: number; z: number; heading: number; speed: number } | null = null;
            if (showGhostRef.current && s.plan && s.plan.length >= 2) {
              const elapsed = Math.max(0, raceScenario.simTime() - s.planStartSimTime);
              ghostState = samplePlanAt(s.plan, elapsed);
            }
            if (ghostState) {
              ghostMesh.group.visible = true;
              syncCarMesh(ghostMesh.group, {
                x: ghostState.x,
                z: ghostState.z,
                heading: ghostState.heading,
                speed: ghostState.speed,
                t: 0,
              });
            } else {
              ghostMesh.group.visible = false;
            }
          }
          // Debug overlays: track ego footprint + toggle clearance rings.
          footprintLine.visible = debugRef.current;
          clearanceGroup.visible = debugRef.current;
          if (debugRef.current) {
            footprintLine.position.set(s.state.x, 0, s.state.z);
            footprintLine.rotation.y = -s.state.heading;
          }
          // Only rebuild the path line when the plan reference changes.
          if (s.plan !== lastPlanRef || !showPathRef.current !== !pathLine) {
            lastPlanRef = s.plan;
            refreshPathLine(s.plan);
          }
          // Throttle React HUD updates to ~10 fps instead of every frame.
          hudFrameCounter++;
          if (hudFrameCounter >= 6) {
            hudFrameCounter = 0;
            const wp = cachedGoalWp;
            const check = wp
              ? checkParkingGoal(s.state, { x: wp.x, z: wp.z, heading: wp.heading })
              : null;
            // Signed heading delta (degrees) for the HUD's "+/-12.3°"
            // readout. The pass flag itself comes from `check.hdgOk`
            // (above) which uses the unsigned tolerance.
            const headingErrorSigned = wp
              ? ((s.state.heading - wp.heading + Math.PI * 3) % (Math.PI * 2) - Math.PI) * (180 / Math.PI)
              : NaN;
            // Plan-vs-execution divergence: where the plan said the
            // chassis would be NOW vs where it actually is. Big values
            // mean the tracker is fighting the plan; small values
            // mean tracker is faithful and any remaining goal error
            // is the planner's terminal pose itself.
            let ghostDivM = NaN;
            if (showGhostRef.current && s.plan && s.plan.length >= 2) {
              const elapsed = Math.max(0, raceScenario.simTime() - s.planStartSimTime);
              const gs = samplePlanAt(s.plan, elapsed);
              if (gs) ghostDivM = Math.hypot(s.state.x - gs.x, s.state.z - gs.z);
            }
            setHud({
              speed: s.state.speed,
              goalDist: check ? check.posM : NaN,
              headingError: headingErrorSigned,
              ghostDivM,
              posOk: check?.posOk ?? false,
              hdgOk: check?.hdgOk ?? false,
              spdOk: check?.spdOk ?? false,
              allOk: check?.passed ?? false,
              totalReplans: s.diagnostics.totalReplans,
              lastReplanMs: s.diagnostics.lastReplanMs,
              steer: s.metrics.liveControls.steer,
              throttle: s.metrics.liveControls.throttle,
              brake: s.metrics.liveControls.brake,
              targetSpeed: s.metrics.liveControls.targetSpeed,
              trackingErrorRms: s.metrics.trackingErrorRms,
              planAgeMs: s.diagnostics.planAgeMs,
              consecutiveFailedReplans: s.diagnostics.consecutiveFailedReplans,
              predErrorRms: s.diagnostics.predErrorRms,
              planLength: s.plan?.length ?? 0,
              activeSegmentIndex: s.activeSegmentIndex,
              totalSegments: s.totalSegments,
              activeSegmentGear: s.activeSegmentGear,
              finished: s.finished,
            });
          }
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
      else if (e.key === 'g' || e.key === 'G') setShowGhost((s) => !s);
      else if (e.key === 'd' || e.key === 'D') setDebug((d) => !d);
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
      scene.remove(footprintLine);
      scene.remove(clearanceGroup);
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
        {hud && (() => {
          // Pass flags come straight from `checkParkingGoal` (computed
          // when the HUD snapshot was taken). The constants below are
          // only used to label the tolerance side of each readout.
          const { posOk, hdgOk, spdOk, allOk } = hud;
          return (
            <>
              <div style={{ marginTop: 6, fontSize: 11, borderTop: '1px solid #1c2840', paddingTop: 5 }}>
                <div style={{ color: allOk ? '#4ade80' : '#55dcff', fontSize: 10, marginBottom: 3 }}>
                  GOAL {allOk ? '— ALL MET' : hud.finished ? '— INCOMPLETE' : ''}
                </div>
                <div>
                  <span style={{ color: posOk ? '#4ade80' : '#ff6b6b' }}>{posOk ? '\u2713' : '\u2717'}</span>
                  {' pos '}
                  {hud.goalDist.toFixed(2)} m
                  <span style={{ opacity: 0.5 }}> / {GOAL_POS_TOL} m</span>
                </div>
                <div>
                  <span style={{ color: hdgOk ? '#4ade80' : '#ff6b6b' }}>{hdgOk ? '\u2713' : '\u2717'}</span>
                  {' hdg '}
                  {hud.headingError >= 0 ? '+' : ''}{hud.headingError.toFixed(1)}&deg;
                  <span style={{ opacity: 0.5 }}> / &plusmn;{GOAL_HEADING_TOL_DEG.toFixed(1)}&deg;</span>
                </div>
                <div>
                  <span style={{ color: spdOk ? '#4ade80' : '#ff6b6b' }}>{spdOk ? '\u2713' : '\u2717'}</span>
                  {' spd '}
                  {Math.abs(hud.speed).toFixed(2)} m/s
                  <span style={{ opacity: 0.5 }}> / {GOAL_SPEED_TOL} m/s</span>
                </div>
              </div>
              <div style={{ opacity: 0.75, marginTop: 4, fontSize: 11 }}>
                <span>replans={hud.totalReplans}</span>
                {' · '}
                <span>plan-ms={hud.lastReplanMs.toFixed(0)}</span>
              </div>
              {/* Plan-vs-execution divergence. Compares the chassis's
                  actual pose to where the active plan PREDICTED it
                  would be NOW (the ghost car). Small → tracker is
                  faithful, any goal error is the planner's terminal
                  pose. Large → tracker is fighting the plan or the
                  plan is mid-update; the goal error isn't the
                  planner's fault. */}
              {showGhost && Number.isFinite(hud.ghostDivM) && (
                <div style={{ opacity: 0.75, marginTop: 2, fontSize: 11 }}>
                  <span style={{ color: '#ffaa33' }}>ghost</span>
                  {' div='}
                  {hud.ghostDivM.toFixed(2)} m
                  <span style={{ opacity: 0.5 }}> (plan vs exec)</span>
                </div>
              )}
            </>
          );
        })()}
        {debug && hud && (
          <div style={{ marginTop: 6, fontSize: 11, opacity: 0.8, borderTop: '1px solid #1c2840', paddingTop: 6 }}>
            <div style={{ color: '#55dcff', fontSize: 10, marginBottom: 3 }}>CONTROLS</div>
            <div>steer={((hud.steer * 180) / Math.PI).toFixed(1)}&deg; · throttle={(hud.throttle * 100).toFixed(0)}%{hud.brake > 0 && ` · brake=${(hud.brake * 100).toFixed(0)}%`}</div>
            <div>target spd={hud.targetSpeed.toFixed(1)} m/s</div>

            <div style={{ color: '#55dcff', fontSize: 10, marginTop: 6, marginBottom: 3 }}>SEGMENT</div>
            <div>seg {hud.activeSegmentIndex + 1}/{hud.totalSegments} · gear={hud.activeSegmentGear}</div>

            <div style={{ color: '#55dcff', fontSize: 10, marginTop: 6, marginBottom: 3 }}>PLAN HEALTH</div>
            <div>age={hud.planAgeMs.toFixed(0)} ms · pts={hud.planLength}</div>
            <div>track err={hud.trackingErrorRms.toFixed(3)} m · pred err={hud.predErrorRms.toFixed(3)} m</div>
            {hud.consecutiveFailedReplans > 0 && (
              <div style={{ color: '#ff6b6b' }}>failed replans={hud.consecutiveFailedReplans}</div>
            )}
          </div>
        )}
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
        <div style={{ marginTop: 8, display: 'flex', gap: 6, fontSize: 11, flexWrap: 'wrap' }}>
          <button onClick={() => setPaused((p) => !p)}>{`[p] ${paused ? 'paused' : 'running'}`}</button>
          <button onClick={() => setShowPath((s) => !s)}>{`[l] path ${showPath ? 'on' : 'off'}`}</button>
          <button onClick={() => setShowGhost((s) => !s)}>{`[g] ghost ${showGhost ? 'on' : 'off'}`}</button>
          <button onClick={() => setDebug((d) => !d)}>{`[d] debug ${debug ? 'on' : 'off'}`}</button>
          <button onClick={() => rebuildRef.current?.(scenarioId)}>[r] reset</button>
        </div>
        <div style={{ marginTop: 6, opacity: 0.5, fontSize: 10 }}>
          [1] [2] [3] scenarios · [r] reset · [p] pause · [l] path · [g] ghost · [d] debug
        </div>
      </div>
    </div>
  );
}
