'use client';

// Side-by-side primitive-library race. Two identical Rapier cars drive an
// identical waypoint loop; the only variable is the motion-primitive library
// their planner uses — kinematic-derived (the "before") vs learned from
// Rapier ground truth via `/learnprimitives` (the "after"). The course is
// designed so the kinematic library's blind spots (no understeer, no lateral
// drag, instant speed tracking, no acceleration limit) cost lap time:
// overshoots in the tight slalom and late braking into the 90° turn.

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import RAPIER from '@dimforge/rapier3d-compat';
import type { LearnedVehicleParams, VehicleState } from 'kinocat/agent';
import { DEFAULT_LEARNED_PARAMS } from 'kinocat/agent';
import { InMemoryNavWorld } from 'kinocat/environment';
import { MotionPrimitiveLibrary } from 'kinocat/primitives';
import {
  createRaycastVehicle,
  createGroundCollider,
  ensureRapier,
  planToAckermannControls,
  type CarHandle,
} from 'kinocat/adapters/rapier';
import {
  createCarMeshHelper,
  syncCarMesh,
  createGroundPlaneHelper,
} from 'kinocat/adapters/three';
import {
  buildKinematicLibrary,
  buildLearnedRaceLibrary,
  buildRaceCourse,
  emptyMetrics,
  pickNextWaypoint,
  planRace,
  RACE_AGENT,
  RACE_BOUNDS,
  RACE_PALETTE as C,
  type RaceMetrics,
} from '../lib/race-primitives-scenarios';
import {
  buildLearnedLibrary,
  createSweepWorld,
  defaultControlSets,
  DEFAULT_START_SPEEDS,
  fitParams,
  LEARN_VEHICLE_TUNING,
  runSweep,
} from '../lib/learn-primitives';

const PHYSICS_DT = 1 / 60;
const VEHICLE_SUBSTEPS = 4;
const REPLAN_INTERVAL_MS = 200;
const WHEEL_BASE = 1.6;
const TOTAL_LAPS = 2;
const RACE_TIMEOUT_S = 90;
const PARAMS_KEY = 'kinocat:learned-params';
const LIBRARY_KEY = 'kinocat:learned-library';

type Phase = 'loading' | 'no-lib' | 'learning' | 'ready' | 'racing' | 'finished';

interface CarRuntime {
  id: 'kinematic' | 'learned';
  color: number;
  pathColor: number;
  world: RAPIER.World;
  car: CarHandle;
  carMesh: ReturnType<typeof createCarMeshHelper>;
  pathLine: THREE.Line | null;
  trailLine: THREE.Line;
  trailPts: THREE.Vector3[];
  ai: {
    plan: VehicleState[] | null;
    planStartWall: number;
    loopIndex: number;
    goal: VehicleState | null;
  };
  lib: MotionPrimitiveLibrary;
  metrics: RaceMetrics;
  trackingErrorAcc: { sumSq: number; count: number };
  /** Total waypoints cleared since race start. */
  waypointsCleared: number;
  /** Last time (perf.now ms) the car moved measurably; used to detect stalls. */
  lastMoveWall: number;
  lastPos: { x: number; z: number };
  scene: THREE.Scene;
  finishWall: number | null;
}

function loadLearnedParams(): LearnedVehicleParams | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = window.localStorage.getItem(PARAMS_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as { params?: LearnedVehicleParams };
    return parsed.params ?? null;
  } catch {
    return null;
  }
}

export default function RacePrimitives() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [phase, setPhase] = useState<Phase>('loading');
  const [learnProgress, setLearnProgress] = useState({ done: 0, total: 0 });
  const [params, setParams] = useState<LearnedVehicleParams | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<{
    kinematic: RaceMetrics;
    learned: RaceMetrics;
  }>({ kinematic: emptyMetrics(), learned: emptyMetrics() });
  const [winner, setWinner] = useState<'kinematic' | 'learned' | 'tie' | null>(null);

  const sceneRef = useRef<{
    cleanup: () => void;
    start: () => void;
    reset: () => void;
  } | null>(null);

  // On mount: look for a cached learned library.
  useEffect(() => {
    const p = loadLearnedParams();
    if (p) {
      setParams(p);
      setPhase('ready');
    } else {
      setPhase('no-lib');
    }
  }, []);

  // Mount the Three.js + Rapier scene only after params are available.
  useEffect(() => {
    if (!params || phase === 'loading' || phase === 'no-lib' || phase === 'learning') return;
    const mount = containerRef.current;
    if (!mount) return;
    let disposed = false;
    let cleanup: (() => void) | null = null;
    (async () => {
      try {
        await ensureRapier();
        if (disposed) return;
        const setup = await setupScene(mount, params, {
          onMetrics: (km, lm) => setMetrics({ kinematic: km, learned: lm }),
          onFinish: (w) => {
            setWinner(w);
            setPhase('finished');
          },
        });
        sceneRef.current = setup;
        cleanup = setup.cleanup;
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      disposed = true;
      cleanup?.();
      sceneRef.current = null;
    };
  }, [params, phase === 'no-lib' || phase === 'learning' ? 'pending' : 'mounted']); // eslint-disable-line react-hooks/exhaustive-deps

  async function runInlineLearn() {
    setError(null);
    setPhase('learning');
    setLearnProgress({ done: 0, total: 0 });
    try {
      const sw = await createSweepWorld(RACE_AGENT);
      try {
        const data = await runSweep(sw, {
          agent: RACE_AGENT,
          startSpeeds: DEFAULT_START_SPEEDS,
          controlSets: defaultControlSets(RACE_AGENT),
          onProgress: (done, total) => setLearnProgress({ done, total }),
          yieldEvery: 4,
          yieldFn: () => new Promise((r) => setTimeout(r, 0)),
        });
        const fit = fitParams(data);
        const lib = buildLearnedLibrary(fit.params, { agent: RACE_AGENT });
        const cached = {
          params: fit.params,
          fit: {
            meanPosError: fit.meanPosError,
            maxPosError: fit.maxPosError,
            loss: fit.loss,
          },
          createdAt: Date.now(),
        };
        window.localStorage.setItem(PARAMS_KEY, JSON.stringify(cached));
        window.localStorage.setItem(LIBRARY_KEY, lib.toJSON());
        setParams(fit.params);
        setPhase('ready');
      } finally {
        sw.dispose();
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setPhase('no-lib');
    }
  }

  function startRace() {
    setWinner(null);
    setMetrics({ kinematic: emptyMetrics(), learned: emptyMetrics() });
    sceneRef.current?.reset();
    sceneRef.current?.start();
    setPhase('racing');
  }

  function resetRace() {
    sceneRef.current?.reset();
    setWinner(null);
    setMetrics({ kinematic: emptyMetrics(), learned: emptyMetrics() });
    setPhase('ready');
  }

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        background: '#0a0d14',
        color: '#cdd3de',
        font: '12px ui-monospace, monospace',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <TopBar
        phase={phase}
        learnProgress={learnProgress}
        winner={winner}
        params={params}
        error={error}
        onLearn={runInlineLearn}
        onStart={startRace}
        onReset={resetRace}
      />
      <div style={{ flex: 1, position: 'relative' }}>
        <div ref={containerRef} style={{ position: 'absolute', inset: 0 }} />
        <MetricsOverlay metrics={metrics} winner={winner} />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Scene setup.

interface SceneCallbacks {
  onMetrics: (k: RaceMetrics, l: RaceMetrics) => void;
  onFinish: (w: 'kinematic' | 'learned' | 'tie') => void;
}

async function setupScene(
  mount: HTMLDivElement,
  params: LearnedVehicleParams,
  cb: SceneCallbacks,
): Promise<{ cleanup: () => void; start: () => void; reset: () => void }> {
  const W = mount.clientWidth;
  const H = mount.clientHeight;

  // ---- Three.js renderer (shared) ----
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(W, H);
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setScissorTest(true);
  mount.appendChild(renderer.domElement);

  // ---- Shared course ----
  const course = buildRaceCourse();
  const navWorld = new InMemoryNavWorld(course.polygons, course.obstacles);
  const kinematicLib = buildKinematicLibrary();
  const learnedLib = buildLearnedRaceLibrary(params);

  // ---- Per-car setup ----
  function makeCar(
    id: 'kinematic' | 'learned',
    lib: MotionPrimitiveLibrary,
    color: number,
    pathColor: number,
  ): CarRuntime {
    const world = new RAPIER.World({ x: 0, y: -9.81, z: 0 });
    createGroundCollider(world, {
      bounds: RACE_BOUNDS,
      pad: 20,
      friction: 1.5,
    });
    const car = createRaycastVehicle(world, {
      id: `race-${id}`,
      position: { x: course.spawn.x, z: course.spawn.z },
      heading: course.spawn.heading,
      ...LEARN_VEHICLE_TUNING,
    });
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(C.bg);
    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const sun = new THREE.DirectionalLight(0xffffff, 0.9);
    sun.position.set(40, 100, 20);
    scene.add(sun);
    scene.add(createGroundPlaneHelper({ bounds: RACE_BOUNDS, color: 0x141a26 }));
    // Waypoint cones — shared geometry per car scene.
    for (let i = 0; i < course.waypoints.length; i++) {
      const wp = course.waypoints[i]!;
      const cone = new THREE.Mesh(
        new THREE.ConeGeometry(0.8, 2.2, 14),
        new THREE.MeshStandardMaterial({
          color: C.gate,
          emissive: C.gate,
          emissiveIntensity: 0.25,
        }),
      );
      cone.position.set(wp.x, 1.1, wp.z);
      scene.add(cone);
      // Number label via a small ring at the base.
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(1.2, 1.8, 24),
        new THREE.MeshBasicMaterial({
          color: 0xffd070,
          transparent: true,
          opacity: 0.35,
          side: THREE.DoubleSide,
        }),
      );
      ring.rotation.x = -Math.PI / 2;
      ring.position.set(wp.x, 0.05, wp.z);
      scene.add(ring);
    }
    // Start marker.
    const start = new THREE.Mesh(
      new THREE.RingGeometry(1.5, 2.2, 24),
      new THREE.MeshBasicMaterial({ color: C.startMarker, side: THREE.DoubleSide }),
    );
    start.rotation.x = -Math.PI / 2;
    start.position.set(course.spawn.x, 0.06, course.spawn.z);
    scene.add(start);
    // Car mesh.
    const carMesh = createCarMeshHelper({ color });
    scene.add(carMesh.group);
    // Actual trail line (grows during race).
    const trailGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(course.spawn.x, 0.15, course.spawn.z),
    ]);
    const trail = new THREE.Line(
      trailGeo,
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.55 }),
    );
    scene.add(trail);
    return {
      id,
      color,
      pathColor,
      world,
      car,
      carMesh,
      pathLine: null,
      trailLine: trail,
      trailPts: [new THREE.Vector3(course.spawn.x, 0.15, course.spawn.z)],
      ai: {
        plan: null,
        planStartWall: performance.now(),
        loopIndex: 0,
        goal: null,
      },
      lib,
      metrics: emptyMetrics(),
      trackingErrorAcc: { sumSq: 0, count: 0 },
      waypointsCleared: 0,
      lastMoveWall: performance.now(),
      lastPos: { x: course.spawn.x, z: course.spawn.z },
      scene,
      finishWall: null,
    };
  }

  const kinematic = makeCar('kinematic', kinematicLib, C.kinematic, C.kinematicPath);
  const learned = makeCar('learned', learnedLib, C.learned, C.learnedPath);

  // ---- Cameras (one per side) ----
  const mapCx = (RACE_BOUNDS.x0 + RACE_BOUNDS.x1) / 2;
  const mapCz = (RACE_BOUNDS.z0 + RACE_BOUNDS.z1) / 2;
  const range = 40;
  function makeOrthoCamera(): THREE.OrthographicCamera {
    const aspect = (W / 2) / H;
    const cam = new THREE.OrthographicCamera(
      -range * aspect,
      range * aspect,
      range,
      -range,
      0.1,
      400,
    );
    cam.position.set(mapCx, 120, mapCz);
    cam.up.set(0, 0, 1);
    cam.lookAt(mapCx, 0, mapCz);
    return cam;
  }
  let camK = makeOrthoCamera();
  let camL = makeOrthoCamera();

  // ---- Settle both vehicles ----
  for (const car of [kinematic, learned]) {
    for (let i = 0; i < 30; i++) {
      car.car.applyControls({ steer: 0, throttle: 0, brake: 0 });
      car.world.timestep = PHYSICS_DT;
      car.car.vehicle.updateVehicle(PHYSICS_DT);
      car.world.step();
    }
    car.car.teleport({
      x: course.spawn.x,
      z: course.spawn.z,
      heading: course.spawn.heading,
    });
  }

  // ---- State ----
  let running = false;
  let raceStartWall = 0;

  function resetCar(car: CarRuntime) {
    // Race START is the only allowed teleport — both cars are placed on the
    // start line at rest, then physically driven by their planner from there.
    car.car.teleport({
      x: course.spawn.x,
      z: course.spawn.z,
      heading: course.spawn.heading,
    });
    car.ai.plan = null;
    car.ai.loopIndex = 0;
    car.ai.goal = null;
    car.metrics = emptyMetrics();
    car.trackingErrorAcc = { sumSq: 0, count: 0 };
    car.waypointsCleared = 0;
    car.lastMoveWall = performance.now();
    car.lastPos = { x: course.spawn.x, z: course.spawn.z };
    car.finishWall = null;
    car.trailPts = [new THREE.Vector3(course.spawn.x, 0.15, course.spawn.z)];
    car.trailLine.geometry.dispose();
    car.trailLine.geometry = new THREE.BufferGeometry().setFromPoints(car.trailPts);
    if (car.pathLine) {
      car.scene.remove(car.pathLine);
      car.pathLine.geometry.dispose();
      (car.pathLine.material as THREE.Material).dispose();
      car.pathLine = null;
    }
  }

  function replan(car: CarRuntime, now: number): void {
    const state = car.car.readState(now);
    const pick = pickNextWaypoint(
      { ...state, t: 0 },
      course.waypoints,
      car.ai.loopIndex,
    );
    if (pick.advanced) {
      car.waypointsCleared++;
      car.ai.loopIndex = pick.nextIndex;
      // Lap completion = wrapping past index 0.
      if (car.waypointsCleared % course.waypoints.length === 0) {
        const lapEnd = car.metrics.raceTime;
        const lap = lapEnd - car.metrics.lapStartTime;
        car.metrics.laps++;
        car.metrics.lastLapTime = lap;
        car.metrics.bestLapTime = Number.isFinite(car.metrics.bestLapTime)
          ? Math.min(car.metrics.bestLapTime, lap)
          : lap;
        car.metrics.lapStartTime = lapEnd;
        if (car.metrics.laps >= TOTAL_LAPS && car.finishWall === null) {
          car.finishWall = now;
        }
      }
    }
    car.ai.goal = pick.goal;
    if (car.finishWall !== null) {
      // Already finished — coast.
      car.ai.plan = null;
      return;
    }
    const res = planRace({
      state: { ...state, t: 0 },
      goal: { ...pick.goal, t: 0 },
      lib: car.lib,
      polygons: course.polygons,
      obstacles: course.obstacles,
      world: navWorld,
    });
    if (res.found && res.path.length > 1) {
      car.ai.plan = res.path;
      car.ai.planStartWall = now;
      // Replace path line.
      if (car.pathLine) {
        car.scene.remove(car.pathLine);
        car.pathLine.geometry.dispose();
        (car.pathLine.material as THREE.Material).dispose();
      }
      const pts = res.path.map((p) => new THREE.Vector3(p.x, 0.4, p.z));
      car.pathLine = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineBasicMaterial({
          color: car.pathColor,
          transparent: true,
          opacity: 0.85,
        }),
      );
      car.scene.add(car.pathLine);
    }
  }

  function stepCar(car: CarRuntime, now: number, dt: number): void {
    if (car.finishWall !== null) {
      // Brake to a stop after finishing.
      car.car.applyControls({ steer: 0, throttle: 0, brake: 1 });
    } else if (car.ai.plan && car.ai.plan.length > 1) {
      const state = car.car.readState(now);
      const elapsed = (now - car.ai.planStartWall) / 1000;
      const live = trimPlan(car.ai.plan, elapsed);
      if (live.length >= 2) {
        car.car.applyControls(
          planToAckermannControls(state, live, {
            wheelBase: 2 * WHEEL_BASE,
            lookaheadMin: 3,
            lookaheadGain: 0.45,
            lookaheadMax: 14,
            maxLateralAccel: 8,
            maxAccel: 6,
            maxDecel: 8,
            cruiseSpeed: RACE_AGENT.maxSpeed,
            goalTolerance: 2,
            minTurnRadius: RACE_AGENT.minTurnRadius,
          }),
        );
      } else {
        car.car.applyControls({ steer: 0, throttle: 0.2, brake: 0 });
      }
    } else {
      car.car.applyControls({ steer: 0, throttle: 0.2, brake: 0 });
    }
    // Sub-stepped physics tick.
    const subDt = dt / VEHICLE_SUBSTEPS;
    car.world.timestep = subDt;
    const filter = RAPIER.QueryFilterFlags.EXCLUDE_DYNAMIC;
    for (let s = 0; s < VEHICLE_SUBSTEPS; s++) {
      car.car.vehicle.updateVehicle(subDt, filter);
      car.world.step();
    }
    // Update metrics.
    const after = car.car.readState(now);
    syncCarMesh(car.carMesh.group, after);
    car.metrics.peakSpeed = Math.max(car.metrics.peakSpeed, Math.abs(after.speed));
    if (running && car.finishWall === null) {
      car.metrics.raceTime += dt;
    }
    // Tracking error: planned position at current elapsed time vs actual.
    if (car.ai.plan && car.ai.plan.length > 1 && car.finishWall === null) {
      const elapsed = (now - car.ai.planStartWall) / 1000;
      const target = planAtTime(car.ai.plan, elapsed);
      if (target) {
        const dx = after.x - target.x;
        const dz = after.z - target.z;
        car.trackingErrorAcc.sumSq += dx * dx + dz * dz;
        car.trackingErrorAcc.count++;
        const n = car.trackingErrorAcc.count;
        car.metrics.trackingErrorRms = Math.sqrt(car.trackingErrorAcc.sumSq / n);
      }
    }
    car.metrics.waypointsCleared = car.waypointsCleared;
    // Trail.
    const lastPt = car.trailPts[car.trailPts.length - 1]!;
    if (Math.hypot(after.x - lastPt.x, after.z - lastPt.z) > 0.4) {
      car.trailPts.push(new THREE.Vector3(after.x, 0.15, after.z));
      if (car.trailPts.length > 1000) car.trailPts.shift();
      car.trailLine.geometry.dispose();
      car.trailLine.geometry = new THREE.BufferGeometry().setFromPoints(car.trailPts);
    }
    // Stall guard — if a car gets stuck against bounds, reset it to current waypoint pose.
    if (Math.hypot(after.x - car.lastPos.x, after.z - car.lastPos.z) > 0.5) {
      car.lastMoveWall = now;
      car.lastPos = { x: after.x, z: after.z };
    } else if (now - car.lastMoveWall > 4000 && car.finishWall === null) {
      // Stuck — bump it back to the current waypoint pose.
      const wp = course.waypoints[car.ai.loopIndex]!;
      car.car.teleportWithSpeed({ x: wp.x, z: wp.z, heading: wp.heading }, 0);
      car.ai.plan = null;
      car.lastMoveWall = now;
      car.lastPos = { x: wp.x, z: wp.z };
    }
  }

  // ---- Replan ticker ----
  const replanTimer = window.setInterval(() => {
    if (!running) return;
    const now = performance.now();
    if (kinematic.finishWall === null) replan(kinematic, now);
    if (learned.finishWall === null) replan(learned, now);
  }, REPLAN_INTERVAL_MS);

  // ---- Animation loop ----
  let stopped = false;
  let lastWall = performance.now();
  function tick() {
    if (stopped) return;
    requestAnimationFrame(tick);
    const now = performance.now();
    const dt = Math.min(0.05, (now - lastWall) / 1000);
    lastWall = now;
    if (running) {
      stepCar(kinematic, now, dt);
      stepCar(learned, now, dt);
      cb.onMetrics(kinematic.metrics, learned.metrics);
      // Race-end check.
      const kDone = kinematic.finishWall !== null;
      const lDone = learned.finishWall !== null;
      const timedOut = (now - raceStartWall) / 1000 > RACE_TIMEOUT_S;
      if ((kDone && lDone) || timedOut) {
        running = false;
        // Winner: whoever finished first; if neither finished, more waypoints
        // wins; if tied, lower tracking error wins.
        let w: 'kinematic' | 'learned' | 'tie';
        if (kDone && lDone) {
          if (kinematic.finishWall! < learned.finishWall!) w = 'kinematic';
          else if (learned.finishWall! < kinematic.finishWall!) w = 'learned';
          else w = 'tie';
        } else if (kDone) {
          w = 'kinematic';
        } else if (lDone) {
          w = 'learned';
        } else if (kinematic.waypointsCleared > learned.waypointsCleared) {
          w = 'kinematic';
        } else if (learned.waypointsCleared > kinematic.waypointsCleared) {
          w = 'learned';
        } else {
          w =
            kinematic.metrics.trackingErrorRms < learned.metrics.trackingErrorRms
              ? 'kinematic'
              : 'learned';
        }
        cb.onFinish(w);
      }
    }
    // Render split viewport.
    const w = mount.clientWidth;
    const h = mount.clientHeight;
    renderer.setScissor(0, 0, w / 2, h);
    renderer.setViewport(0, 0, w / 2, h);
    renderer.render(kinematic.scene, camK);
    renderer.setScissor(w / 2, 0, w / 2, h);
    renderer.setViewport(w / 2, 0, w / 2, h);
    renderer.render(learned.scene, camL);
  }
  tick();

  // ---- Resize ----
  const onResize = () => {
    const w = mount.clientWidth;
    const h = mount.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h);
    const aspect = (w / 2) / h;
    for (const cam of [camK, camL]) {
      cam.left = -range * aspect;
      cam.right = range * aspect;
      cam.top = range;
      cam.bottom = -range;
      cam.updateProjectionMatrix();
    }
  };
  window.addEventListener('resize', onResize);

  return {
    cleanup() {
      stopped = true;
      window.clearInterval(replanTimer);
      window.removeEventListener('resize', onResize);
      kinematic.car.dispose();
      kinematic.world.free();
      learned.car.dispose();
      learned.world.free();
      renderer.dispose();
      if (renderer.domElement.parentNode === mount) {
        mount.removeChild(renderer.domElement);
      }
    },
    start() {
      resetCar(kinematic);
      resetCar(learned);
      running = true;
      raceStartWall = performance.now();
      kinematic.ai.planStartWall = raceStartWall;
      learned.ai.planStartWall = raceStartWall;
      // Kick off an immediate plan for each.
      replan(kinematic, raceStartWall);
      replan(learned, raceStartWall);
    },
    reset() {
      running = false;
      resetCar(kinematic);
      resetCar(learned);
    },
  };
}

// ---------------------------------------------------------------------------
// Helpers.

function trimPlan(plan: VehicleState[], elapsed: number): VehicleState[] {
  let i = 0;
  while (i < plan.length - 1 && plan[i + 1]!.t <= elapsed) i++;
  return plan.slice(i);
}

function planAtTime(plan: VehicleState[], t: number): VehicleState | null {
  if (plan.length === 0) return null;
  if (t <= plan[0]!.t) return plan[0]!;
  for (let i = 1; i < plan.length; i++) {
    if (plan[i]!.t >= t) {
      const a = plan[i - 1]!;
      const b = plan[i]!;
      const span = b.t - a.t;
      const u = span > 1e-9 ? (t - a.t) / span : 0;
      return {
        x: a.x + (b.x - a.x) * u,
        z: a.z + (b.z - a.z) * u,
        heading: a.heading,
        speed: a.speed,
        t,
      };
    }
  }
  return plan[plan.length - 1]!;
}

// ---------------------------------------------------------------------------
// UI bits.

function TopBar({
  phase,
  learnProgress,
  winner,
  params,
  error,
  onLearn,
  onStart,
  onReset,
}: {
  phase: Phase;
  learnProgress: { done: number; total: number };
  winner: 'kinematic' | 'learned' | 'tie' | null;
  params: LearnedVehicleParams | null;
  error: string | null;
  onLearn: () => void;
  onStart: () => void;
  onReset: () => void;
}) {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        padding: '10px 14px',
        borderBottom: '1px solid #1f2735',
        background: '#0d1119',
      }}
    >
      <div style={{ color: '#7fd6ff', fontWeight: 700 }}>race the primitives</div>
      <div style={{ opacity: 0.65 }}>
        kinematic vs learned · same agent, same course, only the motion-primitive library differs
      </div>
      <div style={{ flex: 1 }} />
      {phase === 'loading' && <Status>loading…</Status>}
      {phase === 'no-lib' && (
        <>
          <Status warning>no learned library cached</Status>
          <Btn onClick={onLearn}>learn now (~10s)</Btn>
        </>
      )}
      {phase === 'learning' && (
        <Status>
          collecting trials {learnProgress.done}/{learnProgress.total || '?'}…
        </Status>
      )}
      {phase === 'ready' && params && (
        <>
          <Status>library ready</Status>
          <Btn onClick={onStart}>start race</Btn>
          <Btn onClick={onReset} secondary>reset</Btn>
        </>
      )}
      {phase === 'racing' && (
        <>
          <Status>racing… (best of {TOTAL_LAPS} laps)</Status>
          <Btn onClick={onReset} secondary>reset</Btn>
        </>
      )}
      {phase === 'finished' && (
        <>
          <Status>
            {winner === 'tie' ? 'tie!' : `winner: ${winner}`}
          </Status>
          <Btn onClick={onStart}>race again</Btn>
          <Btn onClick={onReset} secondary>reset</Btn>
        </>
      )}
      {error && <Status warning>err: {error}</Status>}
    </div>
  );
}

function MetricsOverlay({
  metrics,
  winner,
}: {
  metrics: { kinematic: RaceMetrics; learned: RaceMetrics };
  winner: 'kinematic' | 'learned' | 'tie' | null;
}) {
  return (
    <>
      <SideMetrics
        side="left"
        title="KINEMATIC (CARCHASE_LIB)"
        color="#ff8aa0"
        m={metrics.kinematic}
        highlight={winner === 'kinematic'}
      />
      <SideMetrics
        side="right"
        title="LEARNED (fit to Rapier)"
        color="#55dcff"
        m={metrics.learned}
        highlight={winner === 'learned'}
      />
    </>
  );
}

function SideMetrics({
  side,
  title,
  color,
  m,
  highlight,
}: {
  side: 'left' | 'right';
  title: string;
  color: string;
  m: RaceMetrics;
  highlight: boolean;
}) {
  return (
    <div
      style={{
        position: 'absolute',
        top: 12,
        [side]: 12,
        background: 'rgba(13, 17, 25, 0.85)',
        border: `1px solid ${highlight ? color : '#1f2735'}`,
        borderRadius: 8,
        padding: '10px 14px',
        minWidth: 220,
        boxShadow: highlight ? `0 0 24px ${color}66` : 'none',
        color: '#cdd3de',
      }}
    >
      <div style={{ color, fontWeight: 700, marginBottom: 6 }}>{title}</div>
      <KV k="time" v={`${m.raceTime.toFixed(2)} s`} />
      <KV k="laps" v={`${m.laps}/${TOTAL_LAPS}`} />
      <KV k="waypoints" v={`${m.waypointsCleared}`} />
      <KV
        k="best lap"
        v={Number.isFinite(m.bestLapTime) ? `${m.bestLapTime.toFixed(2)} s` : '—'}
      />
      <KV
        k="last lap"
        v={Number.isFinite(m.lastLapTime) ? `${m.lastLapTime.toFixed(2)} s` : '—'}
      />
      <KV k="tracking err (rms)" v={`${m.trackingErrorRms.toFixed(2)} m`} />
      <KV k="peak speed" v={`${m.peakSpeed.toFixed(1)} m/s`} />
    </div>
  );
}

function KV({ k, v }: { k: string; v: string }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
      <span style={{ opacity: 0.7 }}>{k}</span>
      <span style={{ color: '#cdeaff' }}>{v}</span>
    </div>
  );
}

function Status({
  children,
  warning,
}: {
  children: React.ReactNode;
  warning?: boolean;
}) {
  return (
    <div
      style={{
        padding: '4px 10px',
        border: `1px solid ${warning ? '#ff8aa0' : '#1f2735'}`,
        borderRadius: 6,
        color: warning ? '#ff8aa0' : '#cdeaff',
        background: warning ? 'rgba(255, 138, 160, 0.10)' : 'rgba(127, 214, 255, 0.10)',
      }}
    >
      {children}
    </div>
  );
}

function Btn({
  children,
  onClick,
  secondary,
}: {
  children: React.ReactNode;
  onClick: () => void;
  secondary?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        font: '11px ui-monospace, monospace',
        padding: '6px 12px',
        borderRadius: 6,
        border: '1px solid #2a3340',
        background: secondary ? 'rgba(20, 26, 38, 0.85)' : 'rgba(127, 214, 255, 0.18)',
        color: secondary ? '#8c95a4' : '#cdeaff',
        cursor: 'pointer',
        letterSpacing: 0.3,
      }}
    >
      {children}
    </button>
  );
}
