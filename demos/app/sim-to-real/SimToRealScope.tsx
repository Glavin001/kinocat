'use client';

// /sim-to-real — 3D scope that overlays the v2 model's open-loop
// prediction (and parametric-only + kinematic baselines) on top of the
// real Rapier raycast vehicle in the same world, at the same time.
//
// Designed to isolate the two gaps that cause model-vs-physics mismatch:
//   - Gap A (dynamics): same controls into model.forward vs Rapier.step.
//     Surfaced by Playback + Free Drive modes.
//   - Gap B (executor): planner output vs pure-pursuit-driven chassis.
//     Surfaced by Plan & Execute mode.
//
// One scene, one physics world, one real CarHandle. The "ghosts" are
// pure Three.js meshes posed each tick from open-loop rollouts — no
// rigid bodies, no collisions. The HUD on the right + overlay toggles
// on the left are React; the 60 Hz scene loop is imperative.

import { useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import RAPIER from '@dimforge/rapier3d-compat';
import type {
  LearnedVehicleModel,
  CarKinematicState,
  WheeledCarControls,
} from 'kinocat/agent';
import type { ForwardSim } from 'kinocat/primitives';
import {
  DEFAULT_LEARNED_PARAMS_V2,
  DEFAULT_LEARNABLE_CONFIG,
  buildParametricOnlyModel,
  learnedForwardSimV2,
  parametricForwardV2,
  kinematicForwardSim,
  predictWithUncertainty,
} from 'kinocat/agent';
import {
  ensureRapier,
  createGroundCollider,
  createRaycastVehicle,
  planToAckermannControls,
  stepRaycastVehicle,
  type CarHandle,
  type WheelTelemetry,
} from 'kinocat/adapters/rapier';
import {
  encodeForParametricV2,
  encodeForKinematic,
  wheeledFromNormalized,
} from 'kinocat/vehicle/car';
import {
  createCarMeshHelper,
  syncCarMesh,
  createGroundPlaneHelper,
} from 'kinocat/adapters/three';
import { planVehicleOnce } from 'kinocat/planner';
import { InMemoryNavWorld } from 'kinocat/environment';
import {
  RACE_AGENT,
  RACE_BOUNDS,
  buildLearnedRaceLibraryV2,
} from '../lib/race-primitives-scenarios';
import { loadV2Model } from '../lib/v2-model-persistence';
import {
  projectFuture,
  poseGap,
  GapAccumulator,
  FuturePredictionTracker,
  type GapSample,
} from '../lib/sim-to-real-scene';
import { DebugRecorder, type RecorderMeta } from 'kinocat/diagnostics';
import { carRecorderFormatters } from 'kinocat/vehicle/car';
import type { GhostStepResult } from 'kinocat/scene';
import {
  createGhostCar,
  createTrailRibbon,
  createFuturePolyline,
  createUncertaintyCloud,
  createErrorArrow,
  createFrictionCircles,
  type GhostCar,
  type TrailRibbon,
  type FuturePolyline,
  type UncertaintyCloud,
  type ErrorArrow,
  type FrictionCircles,
} from '../components/sim-to-real/overlays';
import { HUD, type HUDHandle, type HUDSnapshot, type SimToRealMode } from '../components/sim-to-real/HUD';
import { ModeSwitcher } from '../components/sim-to-real/ModeSwitcher';

// ---------------------------------------------------------------------------
// Constants

const PHYSICS_DT = 1 / 60;
const VEHICLE_SUBSTEPS = 4;
const FREE_DRIVE_HORIZON_SEC = 1.0;
const FREE_DRIVE_PROJECT_INTERVAL_MS = 200;
const V_MAX = 30;
const TRAIL_CAPACITY = 1500;
// Re-anchor each ghost's continuous open-loop rollout back to the real
// Rapier state every N seconds. Without this, integration error
// compounds for the entire session and the visible gap reads in
// hundreds of meters after a minute of driving — true but useless for
// debugging. Re-anchoring makes the displayed gap "model vs real over
// the last RE_ANCHOR_SEC seconds", which is the metric you actually
// want for Gap A analysis.
const RE_ANCHOR_SEC = 2.0;

interface GhostKind {
  id: 'v2-full' | 'parametric' | 'kinematic';
  label: string;
  color: number;
}

const GHOSTS: GhostKind[] = [
  { id: 'v2-full', label: 'v2 (parametric + residual)', color: 0x4dd2ff },
  { id: 'parametric', label: 'parametric-only', color: 0xb47bff },
  { id: 'kinematic', label: 'kinematic', color: 0xffcc55 },
];

export default function SimToRealScope() {
  const mountRef = useRef<HTMLDivElement>(null);
  const hudRef = useRef<HUDHandle>(null);
  const [ready, setReady] = useState(false);
  const [mode, setMode] = useState<SimToRealMode>('free-drive');
  const [showFriction, setShowFriction] = useState(true);
  const [showUncertainty, setShowUncertainty] = useState(true);
  const [matchSubsteps, setMatchSubsteps] = useState(false);
  const [hasModel, setHasModel] = useState(false);

  // Mutable refs so the imperative scene loop reads the latest UI state.
  const modeRef = useRef(mode);
  const showFrictionRef = useRef(showFriction);
  const showUncertaintyRef = useRef(showUncertainty);
  const matchSubstepsRef = useRef(matchSubsteps);
  modeRef.current = mode;
  showFrictionRef.current = showFriction;
  showUncertaintyRef.current = showUncertainty;
  matchSubstepsRef.current = matchSubsteps;

  const resetFnRef = useRef<(() => void) | null>(null);
  const exportFnRef = useRef<((fmt: 'json' | 'md') => string) | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const toastTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const flashToast = (msg: string) => {
    setToast(msg);
    if (toastTimerRef.current) clearTimeout(toastTimerRef.current);
    toastTimerRef.current = setTimeout(() => setToast(null), 1800);
  };
  const copyDebug = async (fmt: 'json' | 'md') => {
    const out = exportFnRef.current?.(fmt);
    if (!out) return;
    try {
      await navigator.clipboard.writeText(out);
      flashToast(`Copied ${fmt === 'md' ? 'Markdown' : 'JSON'} (${out.length.toLocaleString()} chars)`);
    } catch {
      flashToast('Clipboard blocked — use Download JSON');
    }
  };
  const downloadDebug = () => {
    const out = exportFnRef.current?.('json');
    if (!out) return;
    const blob = new Blob([out], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `sim-to-real-debug-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    flashToast('Downloaded JSON snapshot');
  };

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    let cancelled = false;
    let cleanup: (() => void) | null = null;
    (async () => {
      await ensureRapier();
      if (cancelled) return;
      const cached = loadV2Model();
      setHasModel(cached !== null);
      cleanup = setupScene(mount, cached?.model ?? null);
      setReady(true);
    })();
    return () => {
      cancelled = true;
      cleanup?.();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function setupScene(
    mount: HTMLDivElement,
    persistedModel: LearnedVehicleModel | null,
  ): () => void {
    // ---- model setup --------------------------------------------------
    // v2-full model: prefer the persisted one (trained in /model-lab) so
    // the visual gap reflects the user's actual model. Fall back to a
    // pure parametric model with defaults if the user hasn't trained yet.
    const v2Model: LearnedVehicleModel =
      persistedModel ?? buildParametricOnlyModel(DEFAULT_LEARNED_PARAMS_V2, DEFAULT_LEARNABLE_CONFIG);
    const paramModel: LearnedVehicleModel = buildParametricOnlyModel(
      v2Model.params, v2Model.config,
    );

    // Chassis force constants. v2/parametric take these directly as the
    // driveForce / brakeForce N in their control vector; the kinematic
    // sim derives target speed from throttle/brake instead.
    //
    // CRITICAL: read these from the v2 model's OWN embedded config, not
    // hardcode. The model was trained against a specific chassis tuning
    // (maxDriveForce / maxBrakeForce in LearnableVehicleConfig); feeding
    // it a different magnitude is a silent scale-mismatch bug. The
    // raycast vehicle here also uses the same defaults (engineForce=4000,
    // brakeForce=2000) since we don't override RaycastVehicleOptions —
    // and `deriveLearnableConfig(options)` is how /raceprimitives keeps
    // them in lockstep. We could pull the chassis's `D.engineForce` if
    // we ever start customizing, but for now the model's config is the
    // contract both sides must agree on.
    const ENGINE_FORCE_N = v2Model.config.maxDriveForce;
    const BRAKE_FORCE_N = v2Model.config.maxBrakeForce;

    const sims: Record<GhostKind['id'], ForwardSim<CarKinematicState>> = {
      'v2-full': learnedForwardSimV2(v2Model),
      'parametric': parametricForwardV2(paramModel.params, paramModel.config),
      'kinematic': kinematicForwardSim(RACE_AGENT),
    };

    // Control-vector encoding differs per forward sim:
    //   - v2-full and parametric take WheeledCarControls native form:
    //       [steer_rad, driveForce_N, brakeForce_N]
    //   - kinematic takes [curvature_1/m, target_speed_m_s] (see
    //     core/src/agent/vehicle.ts).
    //   The same applied {steer, throttle, brake} must therefore be
    //   re-encoded per sim. Without this conversion the kinematic ghost
    //   always reads driveForce=4500 as "target speed 4500 m/s", clamps
    //   to maxSpeed (30), and predicts that for every tick — making the
    //   speed gap roughly equal to (30 - real.speed) on every frame.
    //
    // STEER SIGN: `car.applyControls(c)` interprets `c.steer` in Rapier
    // native sign (handed to the wheel controller directly). The model
    // forward sims use kinocat planning sign (heading 0 = +X, +heading
    // rotates +X → +Z) — the SAME wheeled-controls convention that
    // `applyWheeledControls` adopts by explicitly negating its input.
    // We therefore negate `steer` here when encoding for the ghost
    // sims; otherwise the chassis turns left, every ghost predicts
    // right, and the user notices instantly (they did).
    // Control encoders. The steer-sign-flip rule (Rapier raycast frame ->
    // kinocat planning frame) lives once, in `kinocat/vehicle/car`'s
    // `encodeForParametricV2`. Demos no longer reinvent it inline.
    const encodeForSim: Record<GhostKind['id'], (steer: number, throttle: number, brake: number) => number[]> = {
      'v2-full': (steer, throttle, brake) =>
        encodeForParametricV2({ steer, driveForce: throttle * ENGINE_FORCE_N, brakeForce: brake * BRAKE_FORCE_N }),
      'parametric': (steer, throttle, brake) =>
        encodeForParametricV2({ steer, driveForce: throttle * ENGINE_FORCE_N, brakeForce: brake * BRAKE_FORCE_N }),
      'kinematic': (steer, throttle, brake) =>
        encodeForKinematic(
          { steer, driveForce: 0, brakeForce: 0 },
          {
            wheelBase: Math.max(0.5, 2 * RACE_AGENT.minTurnRadius),
            maxSpeed: RACE_AGENT.maxSpeed,
            throttle,
            brake,
          },
        ),
    };

    // ---- Three.js scene -----------------------------------------------
    const W0 = mount.clientWidth || window.innerWidth;
    const H0 = mount.clientHeight || window.innerHeight;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0e16);
    scene.fog = new THREE.Fog(0x0a0e16, 80, 280);
    const camera = new THREE.PerspectiveCamera(60, W0 / H0, 0.1, 800);
    camera.position.set(20, 25, 30);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(W0, H0);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);

    const orbit = new OrbitControls(camera, renderer.domElement);
    orbit.target.set(0, 0, 0);
    orbit.update();

    scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const sun = new THREE.DirectionalLight(0xffffff, 1.0);
    sun.position.set(40, 100, 30);
    scene.add(sun);

    scene.add(
      createGroundPlaneHelper({
        bounds: { x0: -80, x1: 80, z0: -50, z1: 50 },
        color: 0x1c2230,
        gridColor: 0x2a3344,
      }),
    );

    // ---- physics ------------------------------------------------------
    // Generously oversized ground: the auto-replay playback drives at
    // ~10 m/s for ~60 s before recycling, which already covers >500 m of
    // travel. Keep bounds big so the chassis never drops off the world
    // and falls into the void (the previous 80×50 bounds produced
    // catastrophic-looking sim-to-real gaps that were entirely artefacts
    // of the real chassis going airborne, with `wheelIsInContact=false`
    // and infinite-fall y in the wheel telemetry).
    const world = new RAPIER.World({ x: 0, y: -9.81, z: 0 });
    createGroundCollider(world, {
      bounds: { x0: -1000, x1: 1000, z0: -1000, z1: 1000 },
      pad: 200,
    });
    const car = createRaycastVehicle(world, {
      id: 'sim-to-real-car',
      position: { x: 0, z: 0 },
      heading: 0,
    });
    const realCarMesh = createCarMeshHelper({ color: 0xffffff });
    scene.add(realCarMesh.group);

    // ---- overlays -----------------------------------------------------
    const realTrail = createTrailRibbon(V_MAX, TRAIL_CAPACITY);
    scene.add(realTrail.line);

    const ghosts = new Map<
      GhostKind['id'],
      {
        kind: GhostKind;
        car: GhostCar;
        trail: TrailRibbon;
        future: FuturePolyline;
        cloud: UncertaintyCloud;
        arrow: ErrorArrow;
        gap: GapAccumulator;
        tracker: FuturePredictionTracker;
        /** Last instantaneous gap sample (for HUD). */
        last: GapSample;
        /** Predicted state from the OPEN-LOOP playback rollout (Playback). */
        playbackTrace: CarKinematicState[] | null;
        playbackIdx: number;
        /** Sim-time the ghost was last re-anchored to the real chassis. */
        lastAnchorT: number;
      }
    >();

    for (const k of GHOSTS) {
      const g = createGhostCar(k.color);
      scene.add(g.group);
      const trail = createTrailRibbon(V_MAX, TRAIL_CAPACITY);
      scene.add(trail.line);
      const future = createFuturePolyline(k.color);
      scene.add(future.line);
      const cloud = createUncertaintyCloud(k.color);
      cloud.setVisible(false);
      scene.add(cloud.mesh);
      const arrow = createErrorArrow(k.color);
      scene.add(arrow.arrow);
      ghosts.set(k.id, {
        kind: k,
        car: g,
        trail,
        future,
        cloud,
        arrow,
        gap: new GapAccumulator(2.0),
        tracker: new FuturePredictionTracker(),
        last: { t: 0, posErr: 0, headingErr: 0, speedErr: 0 },
        playbackTrace: null,
        playbackIdx: 0,
        lastAnchorT: 0,
      });
    }

    const friction = createFrictionCircles();
    scene.add(friction.group);

    // Rolling debug recorder. 600 frames @ ~60 Hz = 10 s of history.
    // Generic kinocat/diagnostics ring buffer + car-domain formatters.
    const recorder = new DebugRecorder<CarKinematicState, WheeledCarControls>({
      capacity: 600,
      formatters: carRecorderFormatters,
    });

    // Plan-execute overlay (cyan polyline). Built on demand.
    const planMat = new THREE.LineBasicMaterial({ color: 0x00ffff, transparent: true, opacity: 0.95 });
    const planGeom = new THREE.BufferGeometry();
    const planLine = new THREE.Line(planGeom, planMat);
    planLine.visible = false;
    scene.add(planLine);

    // Goal marker.
    const goalMarker = new THREE.Mesh(
      new THREE.RingGeometry(1.4, 1.8, 24),
      new THREE.MeshBasicMaterial({ color: 0x00ffff, side: THREE.DoubleSide, transparent: true, opacity: 0.8 }),
    );
    goalMarker.rotation.x = -Math.PI / 2;
    goalMarker.visible = false;
    scene.add(goalMarker);

    // ---- input --------------------------------------------------------
    const keys = new Set<string>();
    const onKeyDown = (e: KeyboardEvent) => {
      const k = e.key.toLowerCase();
      keys.add(k);
      if (k === 'r') resetAll();
      if (k === 'c') {
        // Snap orbit to a chase-ish position behind the car.
        const s = car.readState(0);
        camera.position.set(s.x - Math.cos(s.heading) * 12, 6, s.z + Math.sin(s.heading) * 12);
        orbit.target.set(s.x, 0, s.z);
        orbit.update();
      }
    };
    const onKeyUp = (e: KeyboardEvent) => keys.delete(e.key.toLowerCase());
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);

    // ---- click-to-goal (plan-execute) --------------------------------
    let planActive: CarKinematicState[] | null = null;
    const navWorld = new InMemoryNavWorld(
      [{
        id: 0,
        ring: [[-75, -45], [75, -45], [75, 45], [-75, 45]],
        y: 0,
      }],
      [],
    );
    const lib = buildLearnedRaceLibraryV2(v2Model);

    const onClick = (e: MouseEvent) => {
      if (modeRef.current !== 'plan-execute') return;
      // Ray-cast click into the ground plane y=0.
      const rect = renderer.domElement.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      const y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      const ray = new THREE.Raycaster();
      ray.setFromCamera(new THREE.Vector2(x, y), camera);
      const t = -ray.ray.origin.y / Math.max(1e-6, ray.ray.direction.y);
      if (t <= 0 || !Number.isFinite(t)) return;
      const p = ray.ray.origin.clone().add(ray.ray.direction.clone().multiplyScalar(t));
      goalMarker.visible = true;
      goalMarker.position.set(p.x, 0.05, p.z);
      const s = car.readState(0);
      const result = planVehicleOnce({
        start: { ...s, t: 0 },
        goal: { x: p.x, z: p.z, heading: s.heading, speed: 0, t: 0 },
        world: navWorld,
        agent: RACE_AGENT,
        lib,
        deadlineMs: 250,
      });
      if (result.found && result.path.length > 1) {
        planActive = result.path;
        const pts = result.path.map((q) => new THREE.Vector3(q.x, 0.18, q.z));
        planGeom.setFromPoints(pts);
        planLine.visible = true;
        // Also stamp ghost waypoints from the plan into each model's
        // open-loop trace so we can overlay model-predicted execution.
        const dt = result.path[1]!.t - result.path[0]!.t;
        const initial: CarKinematicState = { ...s, t: 0 };
        // The plan's controls aren't directly exported per-step here;
        // we approximate by sampling pure-pursuit deltas. For Gap B
        // analysis the plan itself is the reference, so the ghosts roll
        // open-loop using the SAME pure-pursuit commands the chassis
        // will receive each tick (recomputed inside the tick loop).
        // We pre-seed an empty trace; the ghosts get updated in lockstep
        // with Rapier inside the tick.
        void dt;
        for (const ent of ghosts.values()) {
          ent.playbackTrace = [initial];
          ent.playbackIdx = 0;
          ent.gap.reset();
          ent.trail.reset();
        }
      }
    };
    renderer.domElement.addEventListener('click', onClick);

    function resetAll() {
      car.teleportFull(
        { x: 0, z: 0, heading: 0 },
        { forwardSpeed: 0, lateralVelocity: 0, yawRate: 0 },
      );
      realTrail.reset();
      planActive = null;
      planLine.visible = false;
      goalMarker.visible = false;
      for (const g of ghosts.values()) {
        g.gap.reset();
        g.tracker.reset();
        g.trail.reset();
        g.playbackTrace = null;
        g.playbackIdx = 0;
        g.lastAnchorT = 0;
      }
      recorder.clear();
    }
    resetFnRef.current = resetAll;

    exportFnRef.current = (fmt: 'json' | 'md') => {
      const meta: RecorderMeta = {
        mode: modeRef.current,
        matchSubsteps: matchSubstepsRef.current,
        physicsDt: PHYSICS_DT,
        physicsSubsteps: VEHICLE_SUBSTEPS,
        modelDt: matchSubstepsRef.current ? PHYSICS_DT / VEHICLE_SUBSTEPS : PHYSICS_DT,
        engineForceN: ENGINE_FORCE_N,
        brakeForceN: BRAKE_FORCE_N,
        hasPersistedV2: persistedModel !== null,
      };
      return fmt === 'md' ? recorder.toMarkdown(meta) : recorder.toJSON(meta);
    };
    // Also expose on window for headless debugging via CDP.
    (window as unknown as { __simToRealExport?: (f: 'json' | 'md') => string }).__simToRealExport =
      exportFnRef.current!;

    // ---- resize -------------------------------------------------------
    const onResize = () => {
      const wW = mount.clientWidth || window.innerWidth;
      const hH = mount.clientHeight || window.innerHeight;
      renderer.setSize(wW, hH);
      camera.aspect = wW / hH;
      camera.updateProjectionMatrix();
    };
    window.addEventListener('resize', onResize);

    // ---- main loop ----------------------------------------------------
    let stopped = false;
    let simTime = 0;
    let lastProjectMs = 0;
    let lastHudMs = 0;

    function tick() {
      if (stopped) return;
      requestAnimationFrame(tick);

      const m = modeRef.current;
      const realBefore = car.readState(0);

      // ---- decide controls for this tick ----
      let appliedControls = { steer: 0, throttle: 0, brake: 0 };
      if (m === 'free-drive') {
        const accel =
          (keys.has('w') || keys.has('arrowup') ? 1 : 0) -
          (keys.has('s') || keys.has('arrowdown') ? 1 : 0);
        const steerIn =
          (keys.has('a') || keys.has('arrowleft') ? 1 : 0) -
          (keys.has('d') || keys.has('arrowright') ? 1 : 0);
        const brake = keys.has(' ') ? 1 : 0;
        appliedControls = { steer: steerIn * 0.55, throttle: accel, brake };
      } else if (m === 'plan-execute' && planActive && planActive.length > 1) {
        const cmd = planToAckermannControls(
          { ...realBefore, t: simTime },
          planActive,
          {
            wheelBase: 3.2,
            lookaheadMin: 3,
            lookaheadGain: 0.45,
            lookaheadMax: 14,
            maxLateralAccel: 8,
            maxAccel: 6,
            maxDecel: 8,
            cruiseSpeed: 12,
            goalTolerance: 1.5,
            minTurnRadius: RACE_AGENT.minTurnRadius,
          },
        );
        appliedControls = { steer: cmd.steer, throttle: cmd.throttle, brake: cmd.brake };
      } else if (m === 'playback') {
        // Auto-script in absence of a recorded trial: 2 s throttle / 4 s
        // slalom / 1 s brake / 1 s idle. The car is teleported back to
        // the origin at the start of every cycle so it never drifts off
        // the ground and we get clean repeated trials.
        const phase = (simTime % 8);
        if (phase < 2) appliedControls = { steer: 0, throttle: 1, brake: 0 };
        else if (phase < 6) appliedControls = { steer: Math.sin(phase * 2) * 0.5, throttle: 0.7, brake: 0 };
        else appliedControls = { steer: 0, throttle: 0, brake: 1 };
        // Detect cycle boundary (within a few physics ticks of phase=0).
        if (phase < PHYSICS_DT * 1.5 && simTime > 1) {
          car.teleportFull(
            { x: 0, z: 0, heading: 0 },
            { forwardSpeed: 0, lateralVelocity: 0, yawRate: 0 },
          );
          for (const g of ghosts.values()) {
            g.playbackTrace = null;
            g.lastAnchorT = 0;
            g.gap.reset();
            g.trail.reset();
          }
          realTrail.reset();
        }
      }

      // Canonical action shape: WheeledCarControls. Steer-sign flip + force
      // scaling come from `wheeledFromNormalized` so the chassis input
      // matches the headless learner and every other demo bit-for-bit.
      car.applyWheeledControls(
        wheeledFromNormalized(appliedControls, {
          engineForceN: ENGINE_FORCE_N,
          brakeForceN: BRAKE_FORCE_N,
        }),
      );
      stepRaycastVehicle(world, [car], { dt: PHYSICS_DT, substeps: VEHICLE_SUBSTEPS });
      const subDt = PHYSICS_DT / VEHICLE_SUBSTEPS;
      const realAfter: CarKinematicState = { ...car.readState(0), t: simTime + PHYSICS_DT };
      simTime += PHYSICS_DT;
      syncCarMesh(realCarMesh.group, realAfter);
      realTrail.push(realAfter);

      // ---- step ghosts open-loop with the SAME controls --------------
      // This is the Gap A measurement: same (state_t, controls_t, dt)
      // should produce realAfter; the ghost's predicted next state shows
      // exactly where the model diverges. Note: each sim has its OWN
      // control vector encoding (see `encodeForSim` above).

      // The model rollout proceeds at PHYSICS_DT (one big step) by
      // default. With matchSubsteps the rollout substeps at subDt so
      // the integrator-mismatch component of Gap A is excluded.
      const modelDt = matchSubstepsRef.current ? subDt : PHYSICS_DT;
      const innerSteps = matchSubstepsRef.current ? VEHICLE_SUBSTEPS : 1;

      for (const ent of ghosts.values()) {
        // Decide whether to re-anchor this ghost to the real chassis.
        // Anchor on first tick, every RE_ANCHOR_SEC, or whenever the
        // ghost has drifted absurdly far (>50m) so a temporary numerical
        // blow-up doesn't permanently break the visual.
        const needsAnchor =
          !ent.playbackTrace ||
          simTime - ent.lastAnchorT >= RE_ANCHOR_SEC ||
          (ent.playbackTrace.length > 0 &&
            Math.hypot(
              ent.playbackTrace[ent.playbackTrace.length - 1]!.x - realBefore.x,
              ent.playbackTrace[ent.playbackTrace.length - 1]!.z - realBefore.z,
            ) > 50);
        if (needsAnchor) {
          const seed: CarKinematicState = { ...realBefore, t: simTime - PHYSICS_DT };
          ent.playbackTrace = [seed];
          ent.lastAnchorT = simTime;
        }
        let s = ent.playbackTrace![ent.playbackTrace!.length - 1]!;
        const sim = sims[ent.kind.id];
        const cv = encodeForSim[ent.kind.id](appliedControls.steer, appliedControls.throttle, appliedControls.brake);
        for (let i = 0; i < innerSteps; i++) {
          s = sim(s, cv, modelDt);
        }
        const predicted: CarKinematicState = { ...s, t: simTime };
        ent.playbackTrace!.push(predicted);
        if (ent.playbackTrace!.length > 4096) ent.playbackTrace!.shift();
        ent.car.setPose(predicted);
        ent.trail.push(predicted);

        const gap = poseGap(realAfter, predicted);
        ent.gap.push(gap);
        ent.last = gap;
        ent.arrow.setFromTo(realAfter, predicted);

        // For Free Drive: schedule a T-second future projection from
        // the CURRENT state with the CURRENT controls; show the polyline
        // and the ghost-at-T position. Every 200ms only.
        if (m === 'free-drive' && performance.now() - lastProjectMs > FREE_DRIVE_PROJECT_INTERVAL_MS) {
          const future = projectFuture(
            realAfter,
            cv,
            sim,
            { horizonSec: FREE_DRIVE_HORIZON_SEC, stepDt: 1 / 30 },
          );
          ent.future.setPath(future);
          ent.tracker.schedule(future[future.length - 1]!, simTime, FREE_DRIVE_HORIZON_SEC);
          // Uncertainty ellipsoid at T (v2-full ghost only). predictWithUncertainty
          // takes the v2/wheeled encoding regardless of ghost.kind.id.
          if (ent.kind.id === 'v2-full' && showUncertaintyRef.current) {
            const v2CtrlVec = encodeForSim['v2-full'](appliedControls.steer, appliedControls.throttle, appliedControls.brake);
            const pred = predictWithUncertainty(v2Model, realAfter, v2CtrlVec, modelDt);
            const stdSteps = Math.sqrt(FREE_DRIVE_HORIZON_SEC / modelDt);
            ent.cloud.setAt(future[future.length - 1]!, (pred.std[0] ?? 0) * stdSteps, (pred.std[1] ?? 0) * stdSteps);
            ent.cloud.setVisible(true);
          } else {
            ent.cloud.setVisible(false);
          }
        } else if (m !== 'free-drive') {
          ent.future.setVisible(false);
          ent.cloud.setVisible(false);
        } else {
          ent.future.setVisible(true);
        }
        // Drain matured predictions into rolling RMS so Free Drive's
        // RMS is actually "what we predicted T seconds ago, vs now".
        const matured = ent.tracker.drainMatured(realAfter);
        for (const mat of matured) ent.gap.push(mat);
      }
      if (m === 'free-drive' && performance.now() - lastProjectMs > FREE_DRIVE_PROJECT_INTERVAL_MS) {
        lastProjectMs = performance.now();
      }

      // ---- friction circles ----
      const wheels = car.readWheelTelemetry();
      friction.update(wheels, PHYSICS_DT);
      friction.setVisible(showFrictionRef.current);

      // ---- debug capture ----
      // The recorder consumes `WheeledCarControls` (the v2/parametric
      // wheeled encoding). The kinematic ghost uses a different encoding
      // internally (see encodeForSim) but for the captured controls we
      // use the wheeled form, which is what the planner ultimately emits.
      const recCtrl = encodeForSim['v2-full'](appliedControls.steer, appliedControls.throttle, appliedControls.brake);
      const wheeled: WheeledCarControls = {
        steer: recCtrl[0]!,
        driveForce: recCtrl[1]!,
        brakeForce: recCtrl[2]!,
      };
      const ghostStates: GhostStepResult<CarKinematicState>[] = [];
      for (const ent of ghosts.values()) {
        const last = ent.playbackTrace![ent.playbackTrace!.length - 1]!;
        ghostStates.push({ name: ent.kind.id, state: { ...last } });
      }
      recorder.attachExtras({
        appliedRaw: { ...appliedControls },
        wheels: wheels.map((w) => ({
          inContact: w.inContact,
          contactPoint: w.contactPoint ? { ...w.contactPoint } : null,
          forwardImpulse: w.forwardImpulse,
          sideImpulse: w.sideImpulse,
          suspensionForce: w.suspensionForce,
          frictionSlip: w.frictionSlip,
        })),
      });
      recorder.capture({
        simTime,
        real: { ...realAfter },
        controls: wheeled,
        ghosts: ghostStates,
      });

      // ---- chase-cam follow when orbit is idle? Leave manual. ----

      // ---- HUD throttle ----
      if (performance.now() - lastHudMs > 100) {
        lastHudMs = performance.now();
        const snap = buildHudSnapshot(m, realAfter, ghosts, friction.lastMaxUtil(), planActive);
        hudRef.current?.update(snap);
      }

      renderer.render(scene, camera);
    }
    tick();

    return () => {
      stopped = true;
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('resize', onResize);
      renderer.domElement.removeEventListener('click', onClick);
      car.dispose();
      world.free();
      realTrail.dispose(scene);
      for (const g of ghosts.values()) {
        g.car.dispose(scene);
        g.trail.dispose(scene);
        g.future.dispose(scene);
        g.cloud.dispose(scene);
        g.arrow.dispose(scene);
      }
      friction.dispose(scene);
      planLine.geometry.dispose();
      (planLine.material as THREE.Material).dispose();
      renderer.dispose();
      if (renderer.domElement.parentElement === mount) {
        mount.removeChild(renderer.domElement);
      }
    };
  }

  return (
    <div style={{ position: 'fixed', inset: 0, background: '#0a0e16', color: '#e6e9ee' }}>
      <div ref={mountRef} style={{ position: 'absolute', inset: 0 }} />
      <ModeSwitcher
        mode={mode}
        onChange={setMode}
        onReset={() => resetFnRef.current?.()}
        showFriction={showFriction}
        onToggleFriction={setShowFriction}
        showUncertainty={showUncertainty}
        onToggleUncertainty={setShowUncertainty}
        matchSubsteps={matchSubsteps}
        onToggleSubsteps={setMatchSubsteps}
        onCopyMarkdown={() => copyDebug('md')}
        onCopyJson={() => copyDebug('json')}
        onDownloadJson={downloadDebug}
        toast={toast}
      />
      <HUD ref={hudRef} />
      <div style={headerStyle}>
        <Link href="/" style={linkStyle}>← demos</Link>
        <span style={{ opacity: 0.6 }}>·</span>
        <Link href="/model-lab" style={linkStyle}>model-lab</Link>
        <span style={{ opacity: 0.6 }}>·</span>
        <Link href="/raceprimitives" style={linkStyle}>raceprimitives</Link>
        {ready && !hasModel && (
          <span style={{ marginLeft: 12, color: '#ffcc55', fontSize: 12 }}>
            No trained v2 model in cache — using defaults. Train one in /model-lab for a real Gap A test.
          </span>
        )}
      </div>
    </div>
  );
}

function buildHudSnapshot(
  mode: SimToRealMode,
  real: CarKinematicState,
  ghosts: Map<string, {
    kind: GhostKind;
    last: GapSample;
    gap: GapAccumulator;
  }>,
  maxGripPct: number,
  planActive: CarKinematicState[] | null,
): HUDSnapshot {
  const ghostEntries = [...ghosts.values()].map((g) => ({
    label: g.kind.label,
    color: g.kind.color,
    posErr: g.last.posErr,
    headingErrDeg: g.last.headingErr * 180 / Math.PI,
    speedErr: g.last.speedErr,
    rolling: g.gap.rms(),
  }));
  const extra: string[] = [];
  if (mode === 'playback') extra.push('Auto-replay: 2s throttle / 4s slalom / 1s brake / loop');
  if (mode === 'free-drive') extra.push('WASD = drive · Space = brake · C = chase-cam · R = reset');
  extra.push(`Ghosts re-anchor every 2.0 s — gap shown is "model vs real over <=2s"`);
  if (mode === 'plan-execute') {
    if (!planActive) extra.push('Click on the ground to set a goal pose.');
    else extra.push(`Plan: ${planActive.length} states · pure-pursuit driving`);
  }
  return {
    mode,
    status: mode === 'plan-execute' && !planActive ? 'Awaiting goal click' : 'Live',
    real: {
      x: real.x, z: real.z, heading: real.heading, speed: real.speed,
      yawRate: real.yawRate ?? 0, lateralVelocity: real.lateralVelocity ?? 0,
    },
    ghosts: ghostEntries,
    maxGripPct,
    extra,
  };
}

const headerStyle: React.CSSProperties = {
  position: 'fixed',
  bottom: 16,
  left: 16,
  display: 'flex',
  gap: 8,
  alignItems: 'center',
  padding: '8px 12px',
  background: 'rgba(10, 14, 22, 0.82)',
  border: '1px solid rgba(255,255,255,0.12)',
  borderRadius: 8,
  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
  fontSize: 13,
  zIndex: 10,
};

const linkStyle: React.CSSProperties = {
  color: '#9cf',
  textDecoration: 'none',
};
