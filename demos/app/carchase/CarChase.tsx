'use client';

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import RAPIER from '@dimforge/rapier3d-compat';
import {
  PlanRegistry,
  asObstacle,
  constantVelocity,
} from 'kinocat/predict';
import type { Predict, MovingObstacle } from 'kinocat/predict';
import type { VehicleState } from 'kinocat/agent';
import {
  CARCHASE_AGENT,
  CARCHASE_BOUNDS,
  CARCHASE_PALETTE as C,
  buildCarChaseCourse,
  planCarChaseAI,
  predictRobberFromState,
  robberGoal,
  selectTacticalMode,
  tacticalGoal,
  spawnPoses,
  type CarChaseCourse,
  type CopTacticalMode,
} from '../lib/carchase-scenarios';
import {
  createCarChaseWorld,
  ensureRapier,
  planToControls,
  spawnCar,
  type CarHandle,
} from './rapierVehicle';
import {
  createBuildingHelper,
  createJumpArcHelper,
  createBoostPadHelper,
  createDriftGateHelper,
  createCarMeshHelper,
  syncCarMesh as syncCarMeshCore,
  createWaypointLoopHelper,
  createGoalMarkerHelper,
  createInflatedObstacleHelper,
  createNavBoundsHelper,
  createAgentFootprintHelper,
  createHeightfieldMeshHelper,
  createRampChevronsHelper,
  createRapierDebugRenderer,
} from 'kinocat/adapters/three';
import { rampHeightSampler } from 'kinocat/environment';

interface CopAI {
  id: string;
  color: number;
  car: CarHandle;
  plan: VehicleState[] | null;
  planStartWall: number;
  mode: CopTacticalMode;
  goal: VehicleState | null;
  lastReplanWall: number;
  lastExpansions: number;
  lastBudgetMs: number;
  capturedAtWall: number;
  // Wall-clock when the cop first entered the capture radius of the robber
  // in the current contact run. -Infinity when not currently touching. Used
  // to require sustained contact (see `CAPTURE_HOLD_MS`) before a bust —
  // a fender-bump shouldn't count as an arrest.
  contactSinceWall: number;
  // Wall-clock when speed last exceeded `STUCK_SPEED`. If we've been
  // crawling longer than `STUCK_TRIGGER_MS` the per-tick controls do a
  // reverse burst (see `UNSTICK_BURST_MS`) to back away from whatever
  // wall the chassis got pinned against.
  lastMovedWall: number;
  // Wall-clock when the current reverse burst started. -Infinity if not
  // burst-reversing. Set when stuck-trigger fires, cleared when burst
  // duration elapses.
  unstickUntilWall: number;
}

interface RobberAI {
  car: CarHandle;
  plan: VehicleState[] | null;
  planStartWall: number;
  loopIndex: number;
  goal: VehicleState | null;
  lastReplanWall: number;
  lastExpansions: number;
  lastBudgetMs: number;
  // Wall-clock when the AI robber's speed last exceeded the "stuck" floor.
  // If the AI is in charge and we haven't moved for `ROBBER_STUCK_MS`, we
  // force-advance the waypoint loop in case the current target is behind
  // a wall (or otherwise unreachable from the current pose).
  lastMovedWall: number;
  // Same role as `CopAI.unstickUntilWall` — wall-clock the active reverse
  // burst ends; -Infinity when not burst-reversing.
  unstickUntilWall: number;
}

interface Score {
  busts: number;
  round: number;
}

interface Banner {
  text: string;
  color: string;
  untilWall: number;
}

const NUM_COPS = 3;
const CAPTURE_DISTANCE = 4.5;
// Maximum vertical separation (world Y) for a capture to count. Without this
// a cop sitting directly under a robber mid-air over the jump ramp registers
// a bust, even though they are tens of metres apart vertically.
const CAPTURE_VERTICAL = 2.0;
const PHYSICS_DT = 1 / 60;
// DynamicRayCastVehicleController is twitchy at 60 Hz on its own — sub-step
// the vehicle update + world step together (vibe-land does the same with
// VEHICLE_CONTROLLER_SUBSTEPS = 4). This keeps the suspension stable and
// prevents wheels from intermittently losing contact under turning load.
const VEHICLE_SUBSTEPS = 4;
const REPLAN_INTERVAL_MS = 80;
const CAPTURE_COOLDOWN_MS = 2000;
// A bust requires the cop to stay inside the capture radius for this many
// ms. A drive-by bump only counts as a tag, not an arrest — the cop has
// to actually pin the robber. Reset when contact breaks.
const CAPTURE_HOLD_MS = 2500;
// AI-robber "stuck" detector. If `|speed| < ROBBER_STUCK_SPEED` for this
// long, advance the loopIndex to break out of an unreachable waypoint.
const ROBBER_STUCK_MS = 1800;
const ROBBER_STUCK_SPEED = 0.6;
// Unstick reverse burst — applied to ANY AI car (cop or robber) that's
// been below `STUCK_SPEED` for `STUCK_TRIGGER_MS`. The car reverses
// throttle for `UNSTICK_BURST_MS` with a small counter-steer; after the
// burst it goes back to following its plan (which has likely been
// replanned from the new pose by then).
const STUCK_SPEED = 0.6;
const STUCK_TRIGGER_MS = 1200;
const UNSTICK_BURST_MS = 700;

const COP_COLORS = [0xff5566, 0xffaa44, 0xff66dd];

export default function CarChase() {
  const mountRef = useRef<HTMLDivElement>(null);

  const [paused, setPaused] = useState(false);
  const [chase, setChase] = useState(true);
  const [showPaths, setShowPaths] = useState(true);
  const [showGoals, setShowGoals] = useState(true);
  const [showAff, setShowAff] = useState(true);
  const [showDebug, setShowDebug] = useState(false);
  const [showRapierDebug, setShowRapierDebug] = useState(false);
  const [playerDriving, setPlayerDriving] = useState(false);
  const [ready, setReady] = useState(false);
  const [hud, setHud] = useState('');
  const [copStatus, setCopStatus] = useState<string[]>([]);
  const [robberStatus, setRobberStatus] = useState('');
  const [score, setScore] = useState<Score>({ busts: 0, round: 1 });
  const [banner, setBanner] = useState<Banner | null>(null);

  const pausedRef = useRef(paused);
  const chaseRef = useRef(chase);
  const showPathsRef = useRef(showPaths);
  const showGoalsRef = useRef(showGoals);
  const showAffRef = useRef(showAff);
  const showDebugRef = useRef(showDebug);
  const showRapierDebugRef = useRef(showRapierDebug);
  const playerDrivingRef = useRef(playerDriving);
  const resetRef = useRef<(() => void) | null>(null);
  pausedRef.current = paused;
  chaseRef.current = chase;
  showPathsRef.current = showPaths;
  showGoalsRef.current = showGoals;
  showAffRef.current = showAff;
  showDebugRef.current = showDebug;
  showRapierDebugRef.current = showRapierDebug;
  playerDrivingRef.current = playerDriving;

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

  function setupScene(mount: HTMLDivElement): () => void {
    const W0 = window.innerWidth;
    const H0 = window.innerHeight;

    // ---- three.js setup --------------------------------------------------
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(C.bg);
    scene.fog = new THREE.Fog(C.fog, 160, 420);
    const camera = new THREE.PerspectiveCamera(60, W0 / H0, 0.1, 800);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(W0, H0);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);

    const mapCx = (CARCHASE_BOUNDS.x0 + CARCHASE_BOUNDS.x1) / 2;
    const mapCz = (CARCHASE_BOUNDS.z0 + CARCHASE_BOUNDS.z1) / 2;
    const orbit = new OrbitControls(camera, renderer.domElement);
    orbit.target.set(mapCx, 0, mapCz);
    camera.position.set(mapCx + 60, 90, mapCz + 120);
    orbit.update();

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const sun = new THREE.DirectionalLight(0xffffff, 1.1);
    sun.position.set(80, 180, 60);
    scene.add(sun);

    // ---- course + physics ------------------------------------------------
    const course: CarChaseCourse = buildCarChaseCourse();
    const physics = createCarChaseWorld(course);
    const world = physics.world;

    // Ground visual: heightfield mesh sampled from the same
    // `rampHeightSampler` the physics world uses, so visuals + colliders
    // match. When there are no ramps the sampler is constantly 0 and the
    // mesh degenerates to a flat slab — same look as the old plane.
    {
      const groundColor = Number.parseInt(C.ground.slice(1), 16);
      const rampColor = Number.parseInt(C.ramp.slice(1), 16);
      scene.add(
        createHeightfieldMeshHelper({
          bounds: CARCHASE_BOUNDS,
          sampler: rampHeightSampler(course.ramps),
          segmentsX: 120,
          segmentsZ: 90,
          groundColor,
          vertexColorAbove: course.ramps.length > 0 ? 0.2 : undefined,
          aboveColor: rampColor,
        }),
      );
    }

    // Buildings.
    for (const b of course.buildings) {
      scene.add(
        createBuildingHelper(b, {
          color: Number.parseInt(C.building.slice(1), 16),
          edgeColor: Number.parseInt(C.buildingEdge.slice(1), 16),
        }),
      );
    }

    // Debug overlay: shows the planner's view of the world. Each visible
    // building gets a wireframe of its INFLATED obstacle footprint (the
    // polygon the planner actually navigates around — 0.5 m larger on each
    // side than the visual building). The agent footprint of each car is
    // also drawn so any sync issue between physics pose, planner pose, and
    // visual mesh is immediately obvious. Toggled with [d].
    const debugGroup = new THREE.Group();
    debugGroup.visible = false;
    scene.add(debugGroup);
    for (const b of course.buildings) {
      // 0.5 mirrors `box(b.x, b.z, b.hx + 0.5, b.hz + 0.5)` in carchase-scenarios.
      debugGroup.add(createInflatedObstacleHelper(b, 0.5));
    }
    debugGroup.add(createNavBoundsHelper(CARCHASE_BOUNDS));

    // Jump ramps, boost pads, drift gates, waypoint loop — every affordance
    // visual via the shared kinocat/three helpers so the obstacle-course demo
    // and the car-chase demo render the same building blocks identically.
    const affordanceGroup = new THREE.Group();
    scene.add(affordanceGroup);
    // Drivable ramps: chevrons on the surface for launch-direction clarity
    // + a `createJumpArcHelper` overlay per ramp showing the affordance
    // shortcut over the planner-only gap.
    for (const r of course.ramps) affordanceGroup.add(createRampChevronsHelper(r));
    for (const j of course.jumps) {
      affordanceGroup.add(
        createJumpArcHelper(
          { launch: j.launch, land: j.land, hx: 0, hz: 0, height: j.height },
          { launchY: j.height, apexClearance: 2 },
        ),
      );
    }
    for (const p of course.boostPads) {
      affordanceGroup.add(createBoostPadHelper({ x: p.x, z: p.z }));
    }
    for (const g of course.driftGates) {
      affordanceGroup.add(createDriftGateHelper({ x: g.x, z: g.z, heading: g.heading }));
    }
    affordanceGroup.add(createWaypointLoopHelper(course.robberLoop));

    // ---- rapier debug wireframe ----
    const rapierDebug = createRapierDebugRenderer();
    rapierDebug.mesh.visible = false;
    scene.add(rapierDebug.mesh);

    // ---- spawn cars ------------------------------------------------------
    const poses = spawnPoses();
    const robberCar = spawnCar(world, {
      id: 'robber',
      position: { x: poses.robber.x, z: poses.robber.z },
      heading: poses.robber.heading,
    });
    const robber: RobberAI = {
      car: robberCar,
      plan: null,
      planStartWall: performance.now(),
      loopIndex: 0,
      goal: null,
      lastReplanWall: -Infinity,
      lastExpansions: 0,
      lastBudgetMs: 0,
      lastMovedWall: performance.now(),
      unstickUntilWall: -Infinity,
    };
    const cops: CopAI[] = [];
    for (let i = 0; i < NUM_COPS; i++) {
      const spawn = poses.cops[i % poses.cops.length]!;
      const color = COP_COLORS[i % COP_COLORS.length]!;
      const car = spawnCar(world, {
        id: `cop${i}`,
        position: { x: spawn.x, z: spawn.z },
        heading: spawn.heading,
      });
      cops.push({
        id: `cop${i}`,
        color,
        car,
        plan: null,
        planStartWall: performance.now(),
        mode: 'PURSUE',
        goal: null,
        lastReplanWall: -Infinity,
        lastExpansions: 0,
        lastBudgetMs: 0,
        capturedAtWall: -Infinity,
        contactSinceWall: -Infinity,
        lastMovedWall: performance.now(),
        unstickUntilWall: -Infinity,
      });
    }

    // Three meshes for the cars — shared helper from kinocat/adapters/three.
    const robberMesh = createCarMeshHelper({ color: C.robber });
    scene.add(robberMesh.group);
    const copMeshes = cops.map((co) => {
      const c = createCarMeshHelper({ color: co.color, withLightbar: true });
      scene.add(c.group);
      return c;
    });

    // Path lines (re-created on each replan to avoid leaking geometries).
    let robberPathLine: THREE.Line | null = null;
    const copPathLines: (THREE.Line | null)[] = cops.map(() => null);
    function replacePathLine(
      old: THREE.Line | null,
      path: VehicleState[],
      color: number,
    ): THREE.Line {
      if (old) {
        scene.remove(old);
        old.geometry.dispose();
        (old.material as THREE.Material).dispose();
      }
      const pts = path.map((p) => new THREE.Vector3(p.x, 0.3, p.z));
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.85 }),
      );
      scene.add(line);
      return line;
    }

    // Goal markers + per-car planner-footprint outlines via shared helpers.
    const robberGoalMark = createGoalMarkerHelper({ color: C.robber });
    scene.add(robberGoalMark);
    const copGoalMarks = cops.map((co) => {
      const m = createGoalMarkerHelper({ color: co.color });
      scene.add(m);
      return m;
    });
    const robberFootprint = createAgentFootprintHelper(CARCHASE_AGENT.footprint, {
      color: C.robber,
    });
    debugGroup.add(robberFootprint);
    const copFootprints = cops.map((co) => {
      const fp = createAgentFootprintHelper(CARCHASE_AGENT.footprint, { color: co.color });
      debugGroup.add(fp);
      return fp;
    });

    // ---- shared planning state -----------------------------------------
    const registry = new PlanRegistry();

    // ---- input ----------------------------------------------------------
    const keys = new Set<string>();
    const onKeyDown = (e: KeyboardEvent) => {
      const k = e.key.toLowerCase();
      keys.add(k);
      if (k === 'p') setPaused((v) => !v);
      if (k === 'c') setChase((v) => !v);
      if (k === '1') setShowPaths((v) => !v);
      if (k === '2') setShowGoals((v) => !v);
      if (k === '3') setShowAff((v) => !v);
      if (k === 'd') setShowDebug((v) => !v);
      if (k === '4') setShowRapierDebug((v) => !v);
      if (k === 't') setPlayerDriving((v) => !v);
      if (k === 'r') reset();
    };
    resetRef.current = reset;
    const onKeyUp = (e: KeyboardEvent) => keys.delete(e.key.toLowerCase());
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);

    function reset() {
      const p = spawnPoses();
      robber.car.teleport({
        x: p.robber.x,
        z: p.robber.z,
        heading: p.robber.heading,
      });
      robber.plan = null;
      robber.loopIndex = 0;
      robber.lastMovedWall = performance.now();
      robber.unstickUntilWall = -Infinity;
      for (let i = 0; i < cops.length; i++) {
        const s = p.cops[i % p.cops.length]!;
        cops[i]!.car.teleport({ x: s.x, z: s.z, heading: s.heading });
        cops[i]!.plan = null;
        cops[i]!.capturedAtWall = -Infinity;
        cops[i]!.contactSinceWall = -Infinity;
        cops[i]!.lastMovedWall = performance.now();
        cops[i]!.unstickUntilWall = -Infinity;
      }
      registry.publish('robber', []);
      for (const co of cops) registry.publish(co.id, []);
      scoreRef.busts = 0;
      scoreRef.round = 1;
      setScore({ ...scoreRef });
      flashBanner('Reset', '#9bd0ff', 800);
    }

    const scoreRef = { busts: 0, round: 1 };

    let bannerToken = 0;
    function flashBanner(text: string, color: string, ms: number) {
      const wall = performance.now();
      const token = ++bannerToken;
      setBanner({ text, color, untilWall: wall + ms });
      window.setTimeout(() => {
        if (token === bannerToken) setBanner(null);
      }, ms);
    }

    // ---- planning helpers ----------------------------------------------
    function planRobber(now: number) {
      const robberState = robber.car.readState(now);
      // Stuck detector: if the AI robber's been crawling for too long,
      // skip the current waypoint before computing a new goal. Cleared
      // every replan where we observe meaningful motion. (Player-driven
      // mode never reaches planRobber, so this only fires for the AI.)
      if (Math.abs(robberState.speed) > ROBBER_STUCK_SPEED) {
        robber.lastMovedWall = now;
      } else if (now - robber.lastMovedWall > ROBBER_STUCK_MS) {
        robber.loopIndex = (robber.loopIndex + 1) % course.robberLoop.length;
        robber.lastMovedWall = now;
      }
      const pick = robberGoal(
        robberState,
        course.robberLoop,
        robber.loopIndex,
        cops.map((c) => c.car.readState(now)),
        course.buildings,
        course,
      );
      robber.loopIndex = pick.nextIndex;
      const obstacles: MovingObstacle[] = cops.map((co) =>
        asObstacle(
          registry.predictNPC(co.id) as Predict<{ x: number; z: number }> | null
            ? (registry.predictNPC(co.id) as Predict<{ x: number; z: number }>)
            : constantVelocity(co.car.readState(now), 4),
          2.6,
        ),
      );
      const t0 = performance.now();
      const res = planCarChaseAI({
        npcId: 'robber',
        state: { ...robberState, t: 0 },
        goal: { ...pick.goal, t: 0 },
        movingObstacles: obstacles,
        registry,
        course,
      });
      const dt = performance.now() - t0;
      robber.lastBudgetMs = dt;
      robber.lastExpansions = res.stats.expansions;
      robber.lastReplanWall = now;
      robber.goal = pick.goal;
      if (res.found && res.path.length > 1) {
        robber.plan = res.path;
        robber.planStartWall = now;
        registry.publish('robber', res.path);
        if (showPathsRef.current)
          robberPathLine = replacePathLine(robberPathLine, res.path, C.robberPath);
      }
    }

    function planCop(co: CopAI, copIndex: number, now: number) {
      const robberState = robber.car.readState(now);
      const copState = co.car.readState(now);
      const mode = selectTacticalMode(robberState, copState, copIndex);
      // Predict the robber from its published plan first; constant-velocity
      // fallback so cops still have something to chase before the robber
      // finishes its first plan.
      // Horizon must exceed the tactical-goal `eta` (max 6 s) or
      // `tacticalGoal` quietly falls back to robber's CURRENT pose for
      // anything past 4 s — fine for nearby cops, but for player-driven
      // mode this stops cops from leading the robber's predicted path.
      const robberPredict: Predict<VehicleState> = (t) => {
        const p = registry.predictNPC('robber')(t) as VehicleState | null;
        return p ?? predictRobberFromState(robberState, 8)(t);
      };
      const goal = tacticalGoal(robberState, robberPredict, copState, mode, course.buildings, course);
      const siblingIds = cops
        .filter((o) => o.id !== co.id)
        .map((o) => o.id);
      const obstacles: MovingObstacle[] = [
        asObstacle(robberPredict, 2.6),
        ...siblingIds.map((sid) =>
          asObstacle(
            registry.predictNPC(sid) as Predict<{ x: number; z: number }>,
            2.6,
          ),
        ),
      ];
      const t0 = performance.now();
      const res = planCarChaseAI({
        npcId: co.id,
        state: { ...copState, t: 0 },
        goal: { ...goal, t: 0 },
        movingObstacles: obstacles,
        registry,
        course,
      });
      const dt = performance.now() - t0;
      co.mode = mode;
      co.goal = goal;
      co.lastBudgetMs = dt;
      co.lastExpansions = res.stats.expansions;
      co.lastReplanWall = now;
      if (res.found && res.path.length > 1) {
        co.plan = res.path;
        co.planStartWall = now;
        registry.publish(co.id, res.path);
        if (showPathsRef.current)
          copPathLines[copIndex] = replacePathLine(
            copPathLines[copIndex] ?? null,
            res.path,
            C.copPath,
          );
      }
    }

    // ---- replan scheduler ----------------------------------------------
    // Round-robin across robber + cops. The robber gets a slot every other
    // tick so it keeps moving smartly.
    let replanCursor = 0;
    const replanTimer = window.setInterval(() => {
      if (pausedRef.current) return;
      const now = performance.now();
      const slot = replanCursor % (NUM_COPS + 1);
      if (slot === 0) {
        if (!playerDrivingRef.current) planRobber(now);
      } else {
        const ci = (slot - 1) % NUM_COPS;
        const co = cops[ci]!;
        if (co.capturedAtWall + CAPTURE_COOLDOWN_MS < now) planCop(co, ci, now);
      }
      replanCursor += 1;
    }, REPLAN_INTERVAL_MS);

    // ---- resize ---------------------------------------------------------
    const onResize = () => {
      const w = window.innerWidth;
      const h = window.innerHeight;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    window.addEventListener('resize', onResize);

    // ---- main animation loop -------------------------------------------
    let stopped = false;
    function tick() {
      if (stopped) return;
      requestAnimationFrame(tick);
      if (pausedRef.current) {
        renderer.render(scene, camera);
        return;
      }
      const now = performance.now();

      // Read pre-step states so AI plan-tracking and player input both see
      // the post-previous-step poses (the chassis state we'll render).
      const robberState = robber.car.readState(now);
      const copStates = cops.map((co) => co.car.readState(now));

      // ---- drive each car (set wheel forces BEFORE world.step) ----
      if (playerDrivingRef.current) {
        // Player drives robber with WASD. ArrowKeys mirror WASD so the demo
        // works regardless of layout. Space = handbrake.
        const accel =
          (keys.has('w') || keys.has('arrowup') ? 1 : 0) -
          (keys.has('s') || keys.has('arrowdown') ? 1 : 0);
        const steerIn =
          (keys.has('a') || keys.has('arrowleft') ? 1 : 0) -
          (keys.has('d') || keys.has('arrowright') ? 1 : 0);
        const brake = keys.has(' ') ? 1 : 0;
        robber.car.applyControls({
          steer: steerIn * 0.55,
          throttle: accel,
          brake,
        });
      } else if (maybeUnstick(robber, robberState, now)) {
        // Reverse burst is active — controls already applied.
      } else if (robber.plan && robber.plan.length > 1) {
        const elapsed = (now - robber.planStartWall) / 1000;
        // Walk the plan with elapsed time: trim the path to states at or
        // after `elapsed` so pure-pursuit doesn't lock onto stale poses.
        const live = trimPlan(robber.plan, elapsed);
        if (live.length >= 2) {
          const cmd = planToControls(robberState, live);
          robber.car.applyControls(cmd);
        } else {
          robber.car.applyControls({ steer: 0, throttle: 0.2, brake: 0 });
        }
      } else {
        // No plan yet — coast in neutral instead of idling forward. The old
        // behaviour applied 0.15 throttle until the first plan arrived, which
        // on slopes is enough to creep into the nearest wall before the AI
        // gets going (see bug-audit notes in the plan).
        robber.car.applyControls({ steer: 0, throttle: 0, brake: 0 });
      }
      for (let i = 0; i < cops.length; i++) {
        const co = cops[i]!;
        if (co.capturedAtWall + CAPTURE_COOLDOWN_MS > now) {
          co.car.applyControls({ steer: 0, throttle: 0, brake: 1 });
          // Don't accumulate "stuck" time during the post-bust freeze.
          co.lastMovedWall = now;
          co.unstickUntilWall = -Infinity;
          continue;
        }
        if (maybeUnstick(co, copStates[i]!, now)) continue;
        if (co.plan && co.plan.length > 1) {
          const elapsed = (now - co.planStartWall) / 1000;
          const live = trimPlan(co.plan, elapsed);
          if (live.length >= 2) {
            const cmd = planToControls(copStates[i]!, live);
            co.car.applyControls(cmd);
            continue;
          }
        }
        // Fallback "dumb pursuit": planner failed or plan exhausted, but a
        // stationary cop is worse than a cop driving toward the robber.
        // Steer toward the robber's current XZ and throttle forward (or
        // reverse if the robber is behind us). The next replan tick will
        // hopefully produce a real plan; meanwhile we close distance.
        const cs = copStates[i]!;
        const dxr = robberState.x - cs.x;
        const dzr = robberState.z - cs.z;
        const bearing = Math.atan2(dzr, dxr) - cs.heading;
        // Wrap to (-pi, pi].
        const wrapped = Math.atan2(Math.sin(bearing), Math.cos(bearing));
        const forward = Math.abs(wrapped) < Math.PI / 2;
        const steer = Math.max(-1, Math.min(1, (forward ? wrapped : Math.atan2(Math.sin(Math.PI - wrapped), Math.cos(Math.PI - wrapped))) / 0.55));
        co.car.applyControls({
          steer: steer * 0.55,
          throttle: forward ? 0.6 : -0.4,
          brake: 0,
        });
      }

      // Canonical Rapier raycast-vehicle order: updateVehicle BEFORE
      // world.step so the wheel forces are integrated this tick (not the
      // next one). Sub-step both together so suspension stays stable.
      // EXCLUDE_DYNAMIC keeps wheel rays from hitting other cars' chassis
      // colliders — without it cars ride up on each other when close.
      const subDt = PHYSICS_DT / VEHICLE_SUBSTEPS;
      world.timestep = subDt;
      const wheelFilter = RAPIER.QueryFilterFlags.EXCLUDE_DYNAMIC;
      for (let s = 0; s < VEHICLE_SUBSTEPS; s++) {
        robber.car.vehicle.updateVehicle(subDt, wheelFilter);
        for (const co of cops) co.car.vehicle.updateVehicle(subDt, wheelFilter);
        world.step();
      }

      // Re-read post-step states for rendering + capture detection.
      const robberStateAfter = robber.car.readState(now);
      const copStatesAfter = cops.map((co) => co.car.readState(now));

      // ---- capture detection ----
      // A bust requires the cop to stay inside the capture radius for
      // `CAPTURE_HOLD_MS` continuous milliseconds. Each cop tracks when its
      // current contact run began (`contactSinceWall`); breaking contact
      // resets that timer. Drive-by fender bumps therefore tag but don't
      // arrest — the cop has to actually pin the robber.
      const robberY = robber.car.chassis.translation().y;
      for (let i = 0; i < cops.length; i++) {
        const co = cops[i]!;
        if (co.capturedAtWall + CAPTURE_COOLDOWN_MS > now) {
          co.contactSinceWall = -Infinity;
          continue;
        }
        const dx = robberStateAfter.x - copStatesAfter[i]!.x;
        const dz = robberStateAfter.z - copStatesAfter[i]!.z;
        // Vertical separation comes from the physics body translation, NOT
        // from VehicleState (which is the XZ planning plane). A cop directly
        // below a mid-air robber should NOT count as touching.
        const copY = cops[i]!.car.chassis.translation().y;
        const dy = robberY - copY;
        const d = Math.hypot(dx, dz);
        const touching = d < CAPTURE_DISTANCE && Math.abs(dy) < CAPTURE_VERTICAL;
        if (!touching) {
          co.contactSinceWall = -Infinity;
          continue;
        }
        if (co.contactSinceWall === -Infinity) co.contactSinceWall = now;
        if (now - co.contactSinceWall < CAPTURE_HOLD_MS) continue;
        co.capturedAtWall = now;
        co.contactSinceWall = -Infinity;
        scoreRef.busts += 1;
        scoreRef.round += 1;
        setScore({ ...scoreRef });
        flashBanner(`Busted by ${co.id}! Round ${scoreRef.round}`, '#ffd070', 1400);
        // Knock the robber back to a fresh waypoint, cops freeze briefly.
        const p = spawnPoses();
        robber.car.teleport({
          x: p.robber.x,
          z: p.robber.z,
          heading: p.robber.heading,
        });
        robber.plan = null;
        robber.loopIndex = 0;
        robber.lastMovedWall = now;
        // Also reset the OTHER cops' contact timers — the robber just
        // teleported, so any in-progress hold from a sibling is stale.
        for (const other of cops) if (other !== co) other.contactSinceWall = -Infinity;
      }

      // ---- visual sync ----
      // syncCarMeshCore only updates X, Z, and yaw. Because the ramp is a
      // real drivable heightfield surface now, cars physically climb it
      // and (with the jump affordance) launch into the air — we MUST also
      // sync world-Y and the full chassis quaternion so the mesh follows
      // the chassis through the arc, not glued to y=0.
      syncCarMeshCore(robberMesh.group, robberStateAfter);
      {
        const tr = robber.car.chassis.translation();
        const qr = robber.car.chassis.rotation();
        robberMesh.group.position.y = tr.y;
        robberMesh.group.quaternion.set(qr.x, qr.y, qr.z, qr.w);
      }
      for (let i = 0; i < cops.length; i++) {
        syncCarMeshCore(copMeshes[i]!.group, copStatesAfter[i]!);
        {
          const tr = cops[i]!.car.chassis.translation();
          const qr = cops[i]!.car.chassis.rotation();
          copMeshes[i]!.group.position.y = tr.y;
          copMeshes[i]!.group.quaternion.set(qr.x, qr.y, qr.z, qr.w);
        }
        // Blink the lightbar in pursuit.
        const bar = copMeshes[i]!.lightbar;
        if (bar) {
          const mat = bar.material as THREE.MeshStandardMaterial;
          mat.emissiveIntensity = 0.4 + 0.5 * Math.sin(now / 90 + i);
        }
      }

      // Path visibility.
      if (robberPathLine) robberPathLine.visible = showPathsRef.current;
      for (const l of copPathLines) if (l) l.visible = showPathsRef.current;
      affordanceGroup.visible = showAffRef.current;

      // Rapier physics debug wireframe.
      rapierDebug.mesh.visible = showRapierDebugRef.current;
      if (showRapierDebugRef.current) rapierDebug.update(world);

      // Debug overlay (planner navmesh + agent footprints).
      debugGroup.visible = showDebugRef.current;
      if (showDebugRef.current) {
        robberFootprint.position.set(robberStateAfter.x, 0, robberStateAfter.z);
        robberFootprint.rotation.y = -robberStateAfter.heading;
        for (let i = 0; i < cops.length; i++) {
          const s = copStatesAfter[i]!;
          copFootprints[i]!.position.set(s.x, 0, s.z);
          copFootprints[i]!.rotation.y = -s.heading;
        }
      }

      // Goal markers.
      robberGoalMark.visible = showGoalsRef.current && !!robber.goal;
      if (robber.goal) robberGoalMark.position.set(robber.goal.x, 2, robber.goal.z);
      for (let i = 0; i < cops.length; i++) {
        const co = cops[i]!;
        copGoalMarks[i]!.visible = showGoalsRef.current && !!co.goal;
        if (co.goal) copGoalMarks[i]!.position.set(co.goal.x, 2, co.goal.z);
      }

      // Chase camera tracks the robber from behind.
      if (chaseRef.current) {
        const c = Math.cos(robberStateAfter.heading);
        const s = Math.sin(robberStateAfter.heading);
        const cam = new THREE.Vector3(
          robberStateAfter.x - 14 * c,
          7,
          robberStateAfter.z - 14 * s,
        );
        camera.position.lerp(cam, 0.12);
        orbit.target.set(robberStateAfter.x, 1.5, robberStateAfter.z);
        orbit.update();
      }

      // HUD throttle (~10 Hz to avoid React thrash).
      if (now - lastHudWall > 100) {
        lastHudWall = now;
        const v = Math.abs(robberStateAfter.speed).toFixed(1);
        const hdg = ((robberStateAfter.heading * 180) / Math.PI).toFixed(0);
        setHud(
          `${playerDrivingRef.current ? 'YOU (T to release)' : 'AI ROBBER'} · v=${v} m/s · hdg=${hdg}°`,
        );
        setRobberStatus(
          `robber: plan=${robber.plan?.length ?? 0} exp=${robber.lastExpansions} budget=${robber.lastBudgetMs.toFixed(0)}ms loop=${robber.loopIndex}`,
        );
        setCopStatus(
          cops.map((co) => {
            const cooldown = co.capturedAtWall + CAPTURE_COOLDOWN_MS > now;
            const holding =
              !cooldown && co.contactSinceWall !== -Infinity
                ? ((now - co.contactSinceWall) / 1000).toFixed(1)
                : null;
            const tag = cooldown
              ? ' [COOLDOWN]'
              : holding !== null
                ? ` [HOLD ${holding}/${(CAPTURE_HOLD_MS / 1000).toFixed(1)}s]`
                : '';
            return `${co.id} · ${co.mode} · plan=${co.plan?.length ?? 0} exp=${co.lastExpansions} ${co.lastBudgetMs.toFixed(0)}ms${tag}`;
          }),
        );
      }

      renderer.render(scene, camera);
    }
    let lastHudWall = 0;
    tick();

    return () => {
      stopped = true;
      window.clearInterval(replanTimer);
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('resize', onResize);
      robber.car.dispose();
      for (const co of cops) co.car.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === mount)
        mount.removeChild(renderer.domElement);
      world.free();
    };
  }

  return (
    <div
      ref={mountRef}
      style={{
        position: 'fixed',
        inset: 0,
        background: '#0a0d14',
      }}
    >
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
          maxWidth: 420,
          lineHeight: 1.45,
        }}
      >
        <div style={{ color: '#7fd6ff', fontWeight: 700, marginBottom: 4 }}>
          car-chase — cops &amp; robbers
        </div>
        <div>{hud}</div>
        <div style={{ opacity: 0.8 }}>{robberStatus}</div>
        {copStatus.map((s, i) => (
          <div key={i} style={{ opacity: 0.8 }}>
            {s}
          </div>
        ))}
        <div style={{ opacity: 0.7, marginTop: 6 }}>
          busts: {score.busts} · round: {score.round}
        </div>
        <div
          style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: 4,
            marginTop: 8,
          }}
        >
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
            label="paths"
            shortcut="1"
            on={showPaths}
            onClick={() => setShowPaths((v) => !v)}
          />
          <ToggleButton
            label="goals"
            shortcut="2"
            on={showGoals}
            onClick={() => setShowGoals((v) => !v)}
          />
          <ToggleButton
            label="affordances"
            shortcut="3"
            on={showAff}
            onClick={() => setShowAff((v) => !v)}
          />
          <ToggleButton
            label="debug navmesh"
            shortcut="d"
            on={showDebug}
            onClick={() => setShowDebug((v) => !v)}
          />
          <ToggleButton
            label="physics"
            shortcut="4"
            on={showRapierDebug}
            onClick={() => setShowRapierDebug((v) => !v)}
          />
          <ActionButton
            label="reset"
            shortcut="r"
            onClick={() => resetRef.current?.()}
          />
        </div>
        <div style={{ opacity: 0.55, marginTop: 8, fontSize: 11 }}>
          [wasd] drive · [space] brake
        </div>
      </div>
      {banner && (
        <div
          style={{
            position: 'absolute',
            top: 26,
            left: '50%',
            transform: 'translateX(-50%)',
            background: 'rgba(10,14,22,0.85)',
            border: `1px solid ${banner.color}`,
            color: banner.color,
            padding: '10px 20px',
            borderRadius: 10,
            font: '600 14px ui-monospace, monospace',
            letterSpacing: 0.5,
            pointerEvents: 'none',
          }}
        >
          {banner.text}
        </div>
      )}
    </div>
  );
}

/** Drop path samples already passed; resample t-origin so pure-pursuit picks
 *  a fresh lookahead. */
function trimPlan(plan: VehicleState[], elapsed: number): VehicleState[] {
  let i = 0;
  while (i < plan.length - 1 && plan[i + 1]!.t <= elapsed) i++;
  return plan.slice(i);
}

/** Anything that holds a car handle plus the stuck-detector bookkeeping.
 *  Both RobberAI and CopAI satisfy this — the helper doesn't care which.
 *  Mutates `lastMovedWall` / `unstickUntilWall` on the passed-in object. */
interface UnstickTarget {
  car: CarHandle;
  lastMovedWall: number;
  unstickUntilWall: number;
}

/** Detect a wedged AI car (speed ≈ 0 for too long) and apply a short
 *  reverse burst to back away from whatever wall the chassis is pinned
 *  against. Returns true when controls were applied this tick (caller
 *  should skip its normal plan-following).
 *
 *  The burst alternates direction by car id so two cars stuck against
 *  each other don't both reverse the same way and re-wedge. */
function maybeUnstick(
  target: UnstickTarget,
  state: VehicleState,
  now: number,
): boolean {
  // Active burst — keep applying reverse controls until it elapses.
  if (target.unstickUntilWall > now) {
    // Counter-steer based on a stable hash of the car id so the squad
    // fans out instead of all reversing in the same arc.
    const sign = (target.car.id.charCodeAt(target.car.id.length - 1) % 2) * 2 - 1;
    target.car.applyControls({ steer: 0.4 * sign, throttle: -0.7, brake: 0 });
    return true;
  }
  // Not actively bursting — update the moved-recently timestamp and
  // (re)arm the burst if the wedge condition holds.
  if (Math.abs(state.speed) > STUCK_SPEED) {
    target.lastMovedWall = now;
    return false;
  }
  if (now - target.lastMovedWall > STUCK_TRIGGER_MS) {
    target.unstickUntilWall = now + UNSTICK_BURST_MS;
    // Reset the moved timer so we don't re-arm immediately after the
    // burst ends (give it some time to actually move).
    target.lastMovedWall = now;
    const sign = (target.car.id.charCodeAt(target.car.id.length - 1) % 2) * 2 - 1;
    target.car.applyControls({ steer: 0.4 * sign, throttle: -0.7, brake: 0 });
    return true;
  }
  return false;
}

// CARCHASE_AGENT is re-exported so the rapierVehicle helper can keep its
// pure-pursuit config in lockstep with the planner agent.
export { CARCHASE_AGENT };

function ToggleButton({
  label,
  shortcut,
  on,
  onClick,
}: {
  label: string;
  shortcut: string;
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
      title={`Toggle ${label} ([${shortcut}])`}
    >
      <span style={{ opacity: 0.65, marginRight: 4 }}>[{shortcut}]</span>
      {label}
      <span style={{ marginLeft: 6, opacity: 0.85, fontWeight: 600 }}>
        {on ? 'on' : 'off'}
      </span>
    </button>
  );
}

function ActionButton({
  label,
  shortcut,
  onClick,
}: {
  label: string;
  shortcut: string;
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
        border: '1px solid #3b4456',
        background: 'rgba(20, 26, 38, 0.85)',
        color: '#cdd3de',
        cursor: 'pointer',
        letterSpacing: 0.3,
      }}
      title={`${label} ([${shortcut}])`}
    >
      <span style={{ opacity: 0.65, marginRight: 4 }}>[{shortcut}]</span>
      {label}
    </button>
  );
}
