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
import { purePursuit } from 'kinocat/execute';
import {
  createRaycastVehicle,
  createGroundCollider,
  ensureRapier,
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
  fitParamsOnline,
  LEARN_VEHICLE_TUNING,
  runSweep,
  summariseKinematicGap,
  type TransitionSample,
} from '../lib/learn-primitives';

const PHYSICS_DT = 1 / 60;
const VEHICLE_SUBSTEPS = 4;
const REPLAN_INTERVAL_MS = 500;
const WHEEL_BASE = 1.6;
const TRACKER_MAX_LATERAL_ACCEL = 25;
// Online-learning buffer cap. ~60Hz × ~30s/lap = 1800 samples/lap; 4000
// covers ~2 laps of real driving, refit converges in well under a second.
const ONLINE_SAMPLE_CAP = 4000;
// Refit takes a few hundred ms on 4000 samples — long enough that doing it
// inline in the animation loop would visibly hitch. Defer to a microtask.
const REFIT_DEFER_MS = 0;
const PARAMS_KEY = 'kinocat:learned-params';
const LIBRARY_KEY = 'kinocat:learned-library';

type Phase = 'loading' | 'learning' | 'ready' | 'racing' | 'finished';

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
  /** Lap times completed in this race. */
  lapTimes: number[];
  /** Online learner — only populated for the 'learned' car. The kinematic
   *  car has no learner (control group). */
  learner?: OnlineLearnerState;
}

interface OnlineLearnerState {
  params: LearnedVehicleParams;
  samples: TransitionSample[];
  refitCount: number;
  lastFitMs: number;
  lastMeanError: number;
  /** True while a refit is in flight (debounce). */
  refitting: boolean;
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
  const [learner, setLearner] = useState<LearnerSnapshot | null>(null);
  const [winner, setWinner] = useState<'kinematic' | 'learned' | 'tie' | null>(null);

  const sceneRef = useRef<{
    cleanup: () => void;
    start: () => void;
    stop: () => void;
    reset: () => void;
  } | null>(null);

  // On mount: try to load cached pre-trained params as a warm start for the
  // online learner. If none are cached, the learner starts from
  // DEFAULT_LEARNED_PARAMS — both cars still race fine, the learned car just
  // begins with no prior knowledge and learns purely from race data.
  useEffect(() => {
    const p = loadLearnedParams();
    setParams(p ?? DEFAULT_LEARNED_PARAMS);
    setPhase('ready');
  }, []);

  // Mount the Three.js + Rapier scene as soon as initial params are decided.
  useEffect(() => {
    if (!params || phase === 'loading' || phase === 'learning') return;
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
          onLearner: (snap) => setLearner(snap),
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
  }, [params, phase === 'learning' ? 'pending' : 'mounted']); // eslint-disable-line react-hooks/exhaustive-deps

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
        const kinematic = summariseKinematicGap(data);
        const lib = buildLearnedLibrary(fit.params, { agent: RACE_AGENT });
        // Cache matches /learnprimitives' CachedRun shape so both demos can
        // share the same localStorage entry.
        const cached = {
          params: fit.params,
          fit: {
            meanPosError: fit.meanPosError,
            maxPosError: fit.maxPosError,
            loss: fit.loss,
          },
          kinematic,
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
      setPhase('ready');
    }
  }

  function startRace() {
    setWinner(null);
    setMetrics({ kinematic: emptyMetrics(), learned: emptyMetrics() });
    sceneRef.current?.reset();
    sceneRef.current?.start();
    setPhase('racing');
  }

  function stopRace() {
    sceneRef.current?.stop();
    setPhase('finished');
  }

  function resetRace() {
    sceneRef.current?.reset();
    setWinner(null);
    setMetrics({ kinematic: emptyMetrics(), learned: emptyMetrics() });
    setLearner(null);
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
        onStop={stopRace}
        onReset={resetRace}
      />
      <div style={{ flex: 1, position: 'relative' }}>
        <div ref={containerRef} style={{ position: 'absolute', inset: 0 }} />
        <MetricsOverlay metrics={metrics} winner={winner} />
        {(phase === 'racing' || phase === 'finished') && learner && (
          <LearnerPanel snap={learner} />
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Scene setup.

interface SceneCallbacks {
  onMetrics: (k: RaceMetrics, l: RaceMetrics) => void;
  onLearner: (state: LearnerSnapshot) => void;
  onFinish: (w: 'kinematic' | 'learned' | 'tie') => void;
}

interface LearnerSnapshot {
  params: LearnedVehicleParams;
  refitCount: number;
  sampleCount: number;
  lastFitMs: number;
  lastMeanError: number;
  kinematicLapTimes: number[];
  learnedLapTimes: number[];
}

async function setupScene(
  mount: HTMLDivElement,
  params: LearnedVehicleParams,
  cb: SceneCallbacks,
): Promise<{
  cleanup: () => void;
  start: () => void;
  stop: () => void;
  reset: () => void;
}> {
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
  // BOTH cars start with the kinematic library. The learned car records its
  // (state, controls, dt, next) transitions every tick and refits a 5-coef
  // dynamics model after each completed lap; its library is rebuilt from
  // the new params and swapped in. So the user sees lap 1 ≈ identical, then
  // the learned car visibly improves each lap as the model converges.
  const initialLearnerParams = params ?? DEFAULT_LEARNED_PARAMS;

  // ---- Per-car setup ----
  function makeCar(
    id: 'kinematic' | 'learned',
    lib: MotionPrimitiveLibrary,
    color: number,
    pathColor: number,
    learner: OnlineLearnerState | undefined,
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
      lapTimes: [],
      learner,
    };
  }

  const kinematic = makeCar('kinematic', kinematicLib, C.kinematic, C.kinematicPath, undefined);
  // Learned car ALSO starts with kinematic library — it learns from race data
  // and rebuilds its library each lap. First lap will be identical to the
  // control car; subsequent laps the model converges and the library improves.
  const learned = makeCar('learned', kinematicLib, C.learned, C.learnedPath, {
    params: initialLearnerParams,
    samples: [],
    refitCount: 0,
    lastFitMs: 0,
    lastMeanError: 0,
    refitting: false,
  });

  // ---- Cameras: one perspective chase-cam per car. Stays slightly above and
  // behind the chassis so the user can see the wheels, suspension, and the
  // car's relationship to upcoming waypoints — i.e. a real 3D Rapier vehicle.
  function makeChaseCamera(): THREE.PerspectiveCamera {
    const aspect = (W / 2) / H;
    const cam = new THREE.PerspectiveCamera(55, aspect, 0.3, 600);
    cam.position.set(course.spawn.x - 16, 10, course.spawn.z - 4);
    cam.lookAt(course.spawn.x, 1, course.spawn.z);
    return cam;
  }
  const camK = makeChaseCamera();
  const camL = makeChaseCamera();

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
    car.lapTimes = [];
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
        car.lapTimes.push(lap);
        // Online learning: refit the learned car's 5 coefficients from the
        // transitions accumulated this lap (plus prior buffered laps, capped
        // at ONLINE_SAMPLE_CAP for fit time). New params → rebuild library
        // → planner uses better primitives starting next replan.
        if (car.learner && car.learner.samples.length > 50) {
          scheduleRefit(car);
        }
      }
    }
    car.ai.goal = pick.goal;
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
    // Read state BEFORE the tick so we can record the transition for online
    // learning (sample = state_before, controls, dt, state_after).
    const stateBefore = car.car.readState(now);
    let recordedControls: [number, number] | null = null;

    if (car.ai.plan && car.ai.plan.length > 1) {
      const elapsed = (now - car.ai.planStartWall) / 1000;
      const live = trimPlan(car.ai.plan, elapsed);
      if (live.length >= 2) {
        // Use purePursuit directly (vs planToAckermannControls) so we get
        // back `targetSpeed` alongside steering/throttle/brake — that's the
        // second component of the (κ, v_target) control vector the learner
        // fits its 5-coef model against.
        const cmd = purePursuit(stateBefore, live, {
          lookaheadMin: 3,
          lookaheadGain: 0.45,
          lookaheadMax: 14,
          maxLateralAccel: TRACKER_MAX_LATERAL_ACCEL,
          maxAccel: 6,
          maxDecel: 8,
          cruiseSpeed: RACE_AGENT.maxSpeed,
          goalTolerance: 2,
          minTurnRadius: RACE_AGENT.minTurnRadius,
        });
        // Curvature → Ackermann wheel angle, with the kinocat ↔ Rapier yaw
        // sign-flip (planToAckermannControls applies the same flip).
        const steer = -Math.atan(cmd.steering * (2 * WHEEL_BASE));
        car.car.applyControls({ steer, throttle: cmd.throttle, brake: cmd.brake });
        recordedControls = [cmd.steering, cmd.targetSpeed];
      } else {
        car.car.applyControls({ steer: 0, throttle: 0.2, brake: 0 });
        recordedControls = [0, 5];
      }
    } else {
      car.car.applyControls({ steer: 0, throttle: 0.2, brake: 0 });
      recordedControls = [0, 5];
    }
    // Sub-stepped physics tick.
    const subDt = dt / VEHICLE_SUBSTEPS;
    car.world.timestep = subDt;
    const filter = RAPIER.QueryFilterFlags.EXCLUDE_DYNAMIC;
    for (let s = 0; s < VEHICLE_SUBSTEPS; s++) {
      car.car.vehicle.updateVehicle(subDt, filter);
      car.world.step();
    }
    const after = car.car.readState(now);

    // Online learning: append the transition to the learner's rolling buffer.
    // Only the learned car has a learner; the kinematic control car ignores.
    if (car.learner && recordedControls && running) {
      car.learner.samples.push({
        state: stateBefore,
        controls: recordedControls,
        dt,
        next: after,
      });
      while (car.learner.samples.length > ONLINE_SAMPLE_CAP) {
        car.learner.samples.shift();
      }
    }

    // Metrics + visuals.
    syncCarMesh(car.carMesh.group, after);
    car.metrics.peakSpeed = Math.max(car.metrics.peakSpeed, Math.abs(after.speed));
    if (running) car.metrics.raceTime += dt;
    if (car.ai.plan && car.ai.plan.length > 1) {
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
    car.metrics.laps = car.lapTimes.length;
    // Trail.
    const lastPt = car.trailPts[car.trailPts.length - 1]!;
    if (Math.hypot(after.x - lastPt.x, after.z - lastPt.z) > 0.4) {
      car.trailPts.push(new THREE.Vector3(after.x, 0.15, after.z));
      if (car.trailPts.length > 1000) car.trailPts.shift();
      car.trailLine.geometry.dispose();
      car.trailLine.geometry = new THREE.BufferGeometry().setFromPoints(car.trailPts);
    }
    // Stall guard. Only operates if running.
    if (running) {
      if (Math.hypot(after.x - car.lastPos.x, after.z - car.lastPos.z) > 0.5) {
        car.lastMoveWall = now;
        car.lastPos = { x: after.x, z: after.z };
      } else if (now - car.lastMoveWall > 4000) {
        const wp = course.waypoints[car.ai.loopIndex]!;
        car.car.teleport({ x: wp.x, z: wp.z, heading: wp.heading });
        car.ai.plan = null;
        car.lastMoveWall = now;
        car.lastPos = { x: wp.x, z: wp.z };
      }
    }
  }

  /** Defer the (CPU-heavy) refit to a microtask so the animation loop
   *  doesn't hitch. Only one refit at a time per car. */
  function scheduleRefit(car: CarRuntime): void {
    if (!car.learner || car.learner.refitting) return;
    car.learner.refitting = true;
    const samplesSnapshot = car.learner.samples.slice();
    const initParams = car.learner.params;
    setTimeout(() => {
      const learner = car.learner;
      if (!learner) return;
      const t0 = performance.now();
      const fit = fitParamsOnline(samplesSnapshot, RACE_AGENT, {
        init: initParams,
        maxIter: 200,
      });
      learner.params = fit.params;
      learner.refitCount++;
      learner.lastFitMs = performance.now() - t0;
      learner.lastMeanError = fit.meanPosError;
      // Rebuild the learned car's primitive library from the new params and
      // swap it in. Next replan (within REPLAN_INTERVAL_MS) picks it up.
      car.lib = buildLearnedRaceLibrary(fit.params);
      learner.refitting = false;
    }, REFIT_DEFER_MS);
  }

  // ---- Replan ticker ----
  const replanTimer = window.setInterval(() => {
    if (!running) return;
    const now = performance.now();
    replan(kinematic, now);
    replan(learned, now);
  }, REPLAN_INTERVAL_MS);

  // ---- Animation loop ----
  let stopped = false;
  let lastWall = performance.now();
  let lastLearnerEmit = 0;
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
      if (now - lastLearnerEmit > 250 && learned.learner) {
        lastLearnerEmit = now;
        cb.onLearner({
          params: learned.learner.params,
          refitCount: learned.learner.refitCount,
          sampleCount: learned.learner.samples.length,
          lastFitMs: learned.learner.lastFitMs,
          lastMeanError: learned.learner.lastMeanError,
          kinematicLapTimes: kinematic.lapTimes.slice(),
          learnedLapTimes: learned.lapTimes.slice(),
        });
      }
    }
    // Update chase cameras (smooth follow). Reads chassis pose directly from
    // Rapier — same DynamicRayCastVehicleController used in /carchase, just
    // viewed through a chase-cam per side.
    updateChaseCamera(camK, kinematic.car.readState(now));
    updateChaseCamera(camL, learned.car.readState(now));
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
      cam.aspect = aspect;
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
      // Reset learner state on a new race: clear samples but KEEP the most
      // recently fitted params so re-races can carry over what was learned.
      if (learned.learner) {
        learned.learner.samples = [];
      }
      running = true;
      raceStartWall = performance.now();
      kinematic.ai.planStartWall = raceStartWall;
      learned.ai.planStartWall = raceStartWall;
      replan(kinematic, raceStartWall);
      replan(learned, raceStartWall);
    },
    stop() {
      running = false;
    },
    reset() {
      running = false;
      resetCar(kinematic);
      resetCar(learned);
      // Full reset wipes the learner so the demo starts honestly identical.
      if (learned.learner) {
        learned.learner.samples = [];
        learned.learner.refitCount = 0;
        learned.learner.params = initialLearnerParams;
        learned.learner.lastFitMs = 0;
        learned.learner.lastMeanError = 0;
        learned.lib = kinematicLib;
      }
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

/** Smoothly move a perspective camera into a chase position behind a
 *  VehicleState. The chassis y is unknown to the planner (Y is derived in
 *  kinocat); for the cam we place it ~10m above so the wheels and the
 *  upcoming course are both visible. */
function updateChaseCamera(cam: THREE.PerspectiveCamera, s: VehicleState): void {
  const c = Math.cos(s.heading);
  const sn = Math.sin(s.heading);
  // 14m behind + 7m above the chassis, looking ~6m ahead.
  const target = new THREE.Vector3(s.x - 14 * c, 7, s.z - 14 * sn);
  cam.position.lerp(target, 0.12);
  cam.lookAt(s.x + 6 * c, 1.2, s.z + 6 * sn);
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
  onStop,
  onReset,
}: {
  phase: Phase;
  learnProgress: { done: number; total: number };
  winner: 'kinematic' | 'learned' | 'tie' | null;
  params: LearnedVehicleParams | null;
  error: string | null;
  onLearn: () => void;
  onStart: () => void;
  onStop: () => void;
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
        kinematic (control) vs online-learning · both start with the same
        library, learned car refits 5 coefficients from race data each lap
      </div>
      <div style={{ flex: 1 }} />
      {phase === 'loading' && <Status>loading…</Status>}
      {phase === 'learning' && (
        <Status>
          pre-training… collecting trials {learnProgress.done}/{learnProgress.total || '?'}
        </Status>
      )}
      {phase === 'ready' && params && (
        <>
          <Status>ready</Status>
          <Btn onClick={onStart}>start race</Btn>
          <Btn onClick={onLearn} secondary>pre-train</Btn>
          <Btn onClick={onReset} secondary>reset</Btn>
        </>
      )}
      {phase === 'racing' && (
        <>
          <Status>racing… (the learned car refits every lap)</Status>
          <Btn onClick={onStop}>stop</Btn>
        </>
      )}
      {phase === 'finished' && (
        <>
          <Status>stopped</Status>
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
        title="KINEMATIC (control)"
        color="#ff8aa0"
        m={metrics.kinematic}
        highlight={winner === 'kinematic'}
      />
      <SideMetrics
        side="right"
        title="LEARNED (online)"
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
        minWidth: 240,
        boxShadow: highlight ? `0 0 24px ${color}66` : 'none',
        color: '#cdd3de',
      }}
    >
      <div style={{ color, fontWeight: 700, marginBottom: 6 }}>{title}</div>
      <KV k="time" v={`${m.raceTime.toFixed(1)} s`} />
      <KV k="laps" v={`${m.laps}`} />
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

function LearnerPanel({ snap }: { snap: LearnerSnapshot }) {
  return (
    <div
      style={{
        position: 'absolute',
        bottom: 12,
        left: 12,
        right: 12,
        background: 'rgba(13, 17, 25, 0.88)',
        border: '1px solid #1f2735',
        borderRadius: 8,
        padding: '10px 14px',
        color: '#cdd3de',
        font: '11px ui-monospace, monospace',
        display: 'grid',
        gridTemplateColumns: '1.2fr 1fr 1fr 1fr',
        gap: 16,
      }}
    >
      <div>
        <div style={{ color: '#55dcff', fontWeight: 700, marginBottom: 6 }}>
          ONLINE LEARNER · {snap.refitCount} refit{snap.refitCount === 1 ? '' : 's'}
          {' '}· {snap.sampleCount} samples
          {snap.lastFitMs > 0 && (
            <span style={{ opacity: 0.65 }}>
              {' '}· last fit {snap.lastFitMs.toFixed(0)}ms (mean err {snap.lastMeanError.toFixed(3)}m)
            </span>
          )}
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr auto', gap: '2px 12px' }}>
          <KV k="maxAccel" v={`${snap.params.maxAccel.toFixed(2)} m/s²`} />
          <KV k="maxDecel" v={`${snap.params.maxDecel.toFixed(2)} m/s²`} />
          <KV k="accelTau" v={`${snap.params.accelTau.toFixed(3)} s`} />
          <KV k="understeerGain" v={snap.params.understeerGain.toExponential(2)} />
          <KV k="lateralDrag" v={snap.params.lateralDrag.toExponential(2)} />
        </div>
      </div>
      <LapTimeList
        title="kinematic laps"
        color="#ff8aa0"
        laps={snap.kinematicLapTimes}
      />
      <LapTimeList
        title="learned laps"
        color="#55dcff"
        laps={snap.learnedLapTimes}
      />
      <LapDeltaSpark
        kinematic={snap.kinematicLapTimes}
        learned={snap.learnedLapTimes}
      />
    </div>
  );
}

function LapTimeList({
  title,
  color,
  laps,
}: {
  title: string;
  color: string;
  laps: number[];
}) {
  const shown = laps.slice(-8);
  const best = laps.length ? Math.min(...laps) : Number.POSITIVE_INFINITY;
  return (
    <div>
      <div style={{ color, fontWeight: 700, marginBottom: 6 }}>{title}</div>
      {shown.length === 0 && <div style={{ opacity: 0.5 }}>—</div>}
      {shown.map((t, i) => {
        const lapIdx = laps.length - shown.length + i + 1;
        const isBest = t === best;
        return (
          <div
            key={lapIdx}
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              color: isBest ? color : '#cdd3de',
              fontWeight: isBest ? 700 : 400,
            }}
          >
            <span style={{ opacity: 0.7 }}>lap {lapIdx}</span>
            <span>{t.toFixed(2)} s</span>
          </div>
        );
      })}
    </div>
  );
}

function LapDeltaSpark({
  kinematic,
  learned,
}: {
  kinematic: number[];
  learned: number[];
}) {
  const n = Math.min(kinematic.length, learned.length);
  const deltas = [];
  for (let i = 0; i < n; i++) deltas.push(kinematic[i]! - learned[i]!); // positive = learned faster
  return (
    <div>
      <div style={{ color: '#7fd6ff', fontWeight: 700, marginBottom: 6 }}>
        Δ per lap (kinematic − learned)
      </div>
      {deltas.length === 0 && <div style={{ opacity: 0.5 }}>—</div>}
      {deltas.slice(-8).map((d, i) => {
        const lapIdx = deltas.length - Math.min(deltas.length, 8) + i + 1;
        const color = d > 0 ? '#55dcff' : d < 0 ? '#ff8aa0' : '#cdd3de';
        return (
          <div
            key={lapIdx}
            style={{ display: 'flex', justifyContent: 'space-between', color }}
          >
            <span style={{ opacity: 0.7 }}>lap {lapIdx}</span>
            <span>{d > 0 ? '+' : ''}{d.toFixed(2)} s</span>
          </div>
        );
      })}
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
