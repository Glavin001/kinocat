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
  createJumpRampHelper,
  createBoostPadHelper,
  createDriftGateHelper,
  createCarMeshHelper,
  syncCarMesh as syncCarMeshCore,
  createWaypointLoopHelper,
  createGoalMarkerHelper,
  createInflatedObstacleHelper,
  createNavBoundsHelper,
  createAgentFootprintHelper,
  createRapierDebugRenderer,
} from 'kinocat/adapters/three';

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

    // Ground visual: a flat plane matching the planning rectangle.
    {
      const w = CARCHASE_BOUNDS.x1 - CARCHASE_BOUNDS.x0;
      const d = CARCHASE_BOUNDS.z1 - CARCHASE_BOUNDS.z0;
      const g = new THREE.PlaneGeometry(w, d, 1, 1);
      g.rotateX(-Math.PI / 2);
      g.translate(mapCx, 0, mapCz);
      const m = new THREE.Mesh(
        g,
        new THREE.MeshStandardMaterial({ color: C.ground }),
      );
      scene.add(m);

      // Grid overlay for spatial reference.
      const grid = new THREE.GridHelper(Math.max(w, d), 24, 0x2a3040, 0x1a1f2c);
      grid.position.set(mapCx, 0.02, mapCz);
      scene.add(grid);
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
    for (const j of course.jumps) affordanceGroup.add(createJumpRampHelper(j));
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
      for (let i = 0; i < cops.length; i++) {
        const s = p.cops[i % p.cops.length]!;
        cops[i]!.car.teleport({ x: s.x, z: s.z, heading: s.heading });
        cops[i]!.plan = null;
        cops[i]!.capturedAtWall = -Infinity;
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
      const robberPredict: Predict<VehicleState> = (t) => {
        const p = registry.predictNPC('robber')(t) as VehicleState | null;
        return p ?? predictRobberFromState(robberState, 4)(t);
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
          continue;
        }
        if (co.plan && co.plan.length > 1) {
          const elapsed = (now - co.planStartWall) / 1000;
          const live = trimPlan(co.plan, elapsed);
          if (live.length >= 2) {
            const cmd = planToControls(copStates[i]!, live);
            co.car.applyControls(cmd);
            continue;
          }
        }
        co.car.applyControls({ steer: 0, throttle: 0, brake: 0 });
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
      for (let i = 0; i < cops.length; i++) {
        const co = cops[i]!;
        if (co.capturedAtWall + CAPTURE_COOLDOWN_MS > now) continue;
        const dx = robberStateAfter.x - copStatesAfter[i]!.x;
        const dz = robberStateAfter.z - copStatesAfter[i]!.z;
        // Vertical separation comes from the physics body translation, NOT
        // from VehicleState (which is the XZ planning plane). A cop directly
        // below a mid-air robber should NOT register a capture.
        const robberY = robber.car.chassis.translation().y;
        const copY = cops[i]!.car.chassis.translation().y;
        const dy = robberY - copY;
        const d = Math.hypot(dx, dz);
        if (d < CAPTURE_DISTANCE && Math.abs(dy) < CAPTURE_VERTICAL) {
          co.capturedAtWall = now;
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
        }
      }

      // ---- visual sync ----
      syncCarMeshCore(robberMesh.group, robberStateAfter);
      for (let i = 0; i < cops.length; i++) {
        syncCarMeshCore(copMeshes[i]!.group, copStatesAfter[i]!);
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
          cops.map(
            (co) =>
              `${co.id} · ${co.mode} · plan=${co.plan?.length ?? 0} exp=${co.lastExpansions} ${co.lastBudgetMs.toFixed(0)}ms${co.capturedAtWall + CAPTURE_COOLDOWN_MS > now ? ' [COOLDOWN]' : ''}`,
          ),
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
