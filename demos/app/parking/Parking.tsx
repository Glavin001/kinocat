'use client';

// Tight-parking demo: three progressively-harder parking scenarios driven
// by the same kinocat planner stack as /carchase and /obstaclecourse, but
// with sub-meter discretisation + the new `footprintInflate` accuracy knob
// so the car threads gaps it has no business fitting in by accident.
//
// Three scenarios cycle via the [1] / [2] / [3] keys (or HUD buttons):
//   1. forward pull-in (easy)
//   2. reverse perpendicular (medium)
//   3. parallel parking (hard)
//
// Each scenario rebuilds the parked-car + curb colliders from scratch and
// kicks off a single one-shot plan; pure-pursuit (planToAckermannControls)
// then drives the chassis. Press [r] at any time to respawn at the start.

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import RAPIER from '@dimforge/rapier3d-compat';
import { InMemoryNavWorld } from 'kinocat/environment';
import type { VehicleState } from 'kinocat/agent';
import {
  ensureRapier,
  createRaycastVehicle,
  createGroundCollider,
  createBoxCollider,
  planToAckermannControls,
} from 'kinocat/adapters/rapier';
import {
  createGroundPlaneHelper,
  createCarMeshHelper,
  syncCarMesh,
  createGoalMarkerHelper,
  createAgentFootprintHelper,
} from 'kinocat/adapters/three';
import {
  PARKING_AGENT,
  PARKING_BOUNDS,
  PARKING_PALETTE as C,
  PARKING_SCENARIOS,
  PARKING_LABELS,
  PARKING_FOOTPRINT_INFLATE,
  buildParkingScenario,
  planParking,
  type ParkingScenarioId,
  type ParkingScenario,
} from '../lib/parking-scenarios';

const PHYSICS_DT = 1 / 60;
const VEHICLE_SUBSTEPS = 4;
const WHEEL_BASE = 1.6;

export default function Parking() {
  const mountRef = useRef<HTMLDivElement>(null);

  const [scenarioId, setScenarioId] = useState<ParkingScenarioId>('forward-pullin');
  const [paused, setPaused] = useState(false);
  const [showPath, setShowPath] = useState(true);
  const [showDebug, setShowDebug] = useState(false);
  const [ready, setReady] = useState(false);
  const [hud, setHud] = useState('');
  const [status, setStatus] = useState('');
  // Two extra lines that only render when debug overlay is on.
  const [debugSeg, setDebugSeg] = useState('');
  const [debugCmd, setDebugCmd] = useState('');

  const scenarioRef = useRef(scenarioId);
  const pausedRef = useRef(paused);
  const showPathRef = useRef(showPath);
  const showDebugRef = useRef(showDebug);
  const rebuildRef = useRef<((id: ParkingScenarioId) => void) | null>(null);
  scenarioRef.current = scenarioId;
  pausedRef.current = paused;
  showPathRef.current = showPath;
  showDebugRef.current = showDebug;

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

  useEffect(() => {
    rebuildRef.current?.(scenarioId);
  }, [scenarioId]);

  function setupScene(mount: HTMLDivElement): () => void {
    const W0 = window.innerWidth;
    const H0 = window.innerHeight;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(C.bg);
    scene.fog = new THREE.Fog(C.fog, 80, 220);
    const camera = new THREE.PerspectiveCamera(60, W0 / H0, 0.1, 600);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(W0, H0);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);

    const mapCx = (PARKING_BOUNDS.x0 + PARKING_BOUNDS.x1) / 2;
    const mapCz = (PARKING_BOUNDS.z0 + PARKING_BOUNDS.z1) / 2;
    const orbit = new OrbitControls(camera, renderer.domElement);
    orbit.target.set(mapCx, 0, mapCz);
    camera.position.set(mapCx + 0, 38, mapCz + 30);
    orbit.update();

    scene.add(new THREE.AmbientLight(0xffffff, 0.65));
    const sun = new THREE.DirectionalLight(0xffffff, 1.0);
    sun.position.set(30, 80, 20);
    scene.add(sun);

    const world = new RAPIER.World({ x: 0, y: -9.81, z: 0 });

    let scenario: ParkingScenario = buildParkingScenario(scenarioRef.current);
    let navWorld = new InMemoryNavWorld(scenario.polygons, scenario.obstacles);

    let scenePhysics: RAPIER.Collider[] = [];
    let sceneVisuals = new THREE.Group();
    scene.add(sceneVisuals);
    // The obstacle-clearance overlay rings each parked car / wall with
    // the planner's footprintInflate band; rebuilt per scenario inside
    // `buildSceneFor` and toggled by the debug HUD button.
    let sceneObstacleClearance: THREE.Group = new THREE.Group();

    function buildSceneFor(s: ParkingScenario) {
      scene.remove(sceneVisuals);
      sceneVisuals.traverse((o) => {
        const mesh = o as THREE.Mesh;
        if (mesh.geometry) mesh.geometry.dispose();
        const mat = mesh.material as THREE.Material | THREE.Material[] | undefined;
        if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
        else if (mat) mat.dispose();
      });
      sceneVisuals = new THREE.Group();
      scene.add(sceneVisuals);

      for (const col of scenePhysics) {
        const body = col.parent();
        world.removeCollider(col, true);
        if (body) world.removeRigidBody(body);
      }
      scenePhysics = [];

      // Ground.
      sceneVisuals.add(
        createGroundPlaneHelper({ bounds: s.bounds, color: 0x1a2233 }),
      );
      scenePhysics.push(
        createGroundCollider(world, { bounds: s.bounds, pad: 10 }),
      );

      // Target stall: a translucent rectangle painted on the ground.
      sceneVisuals.add(createStallMarker(s.targetStall));

      // Parked cars: a low cuboid with a cabin, axis-rotated by `heading`.
      for (const car of s.parkedCars) {
        sceneVisuals.add(createParkedCarMesh(car));
        // Physics: rotate the axis-aligned half-extents into world space so
        // the collider matches the visible mesh.
        const c = Math.cos(car.heading);
        const sin = Math.sin(car.heading);
        // Approximate the rotated obstacle with an AABB whose extents are
        // the projected half-widths. (Adequate because all scenario cars
        // are either 0 or π/2 heading; for arbitrary headings replace with
        // a rotated cuboid collider.)
        const hxW = Math.abs(car.hx * c) + Math.abs(car.hz * sin);
        const hzW = Math.abs(car.hx * sin) + Math.abs(car.hz * c);
        scenePhysics.push(
          createBoxCollider(world, {
            x: car.x,
            y: 0.5,
            z: car.z,
            hx: hxW,
            hy: 0.5,
            hz: hzW,
          }),
        );
      }

      // Walls / curbs.
      for (const w of s.walls) {
        const g = new THREE.Mesh(
          new THREE.BoxGeometry(w.hx * 2, 0.4, w.hz * 2),
          new THREE.MeshStandardMaterial({ color: C.curb }),
        );
        g.position.set(w.x, 0.2, w.z);
        sceneVisuals.add(g);
        scenePhysics.push(
          createBoxCollider(world, {
            x: w.x,
            y: 0.2,
            z: w.z,
            hx: w.hx,
            hy: 0.2,
            hz: w.hz,
          }),
        );
      }

      // Debug-only: ring each obstacle with the planner's clearance band.
      // The planner refuses to put any point of the ego's footprint
      // inside this band, so the user can read off "the visible chassis
      // can touch the parked-car wall, but the planning footprint —
      // which is what the search reasoned about — kept this much margin".
      sceneObstacleClearance = new THREE.Group();
      sceneObstacleClearance.visible = showDebugRef.current;
      for (const obstacle of s.obstacles) {
        const aabb = polygonAabb(obstacle);
        sceneObstacleClearance.add(
          createObstacleClearanceRing(
            aabb,
            PARKING_FOOTPRINT_INFLATE,
            0xff66aa,
          ),
        );
      }
      sceneVisuals.add(sceneObstacleClearance);
    }

    function createStallMarker(stall: ParkingScenario['targetStall']): THREE.Group {
      // Stall extents are stored in chassis-local frame (hx along the car's
      // forward axis, hz across). Build the marker as a thin flat box so
      // the orientation rotates intuitively about world Y by `-heading`
      // (matches the syncCarMesh convention).
      const group = new THREE.Group();
      const plate = new THREE.Mesh(
        new THREE.BoxGeometry(stall.hx * 2, 0.02, stall.hz * 2),
        new THREE.MeshBasicMaterial({
          color: C.stallEmpty,
          transparent: true,
          opacity: 0.6,
        }),
      );
      const border = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(stall.hx * 2, 0.04, stall.hz * 2)),
        new THREE.LineBasicMaterial({ color: 0xffd070 }),
      );
      group.add(plate);
      group.add(border);
      group.position.set(stall.x, 0.03, stall.z);
      group.rotation.y = -stall.heading;
      return group;
    }

    function createParkedCarMesh(car: ParkingScenario['parkedCars'][number]): THREE.Group {
      const group = new THREE.Group();
      const body = new THREE.Mesh(
        new THREE.BoxGeometry(car.hx * 2, 1.0, car.hz * 2),
        new THREE.MeshStandardMaterial({
          color: C.parkedCar,
          metalness: 0.3,
          roughness: 0.6,
        }),
      );
      body.position.y = 0.5;
      group.add(body);
      const cabin = new THREE.Mesh(
        new THREE.BoxGeometry(car.hx, 0.5, car.hz * 1.6),
        new THREE.MeshStandardMaterial({
          color: 0x141822,
          metalness: 0.5,
          roughness: 0.4,
        }),
      );
      cabin.position.set(-car.hx * 0.1, 1.1, 0);
      group.add(cabin);
      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(car.hx * 2, 1.0, car.hz * 2)),
        new THREE.LineBasicMaterial({ color: 0x6c7a94 }),
      );
      edges.position.y = 0.5;
      group.add(edges);
      group.position.set(car.x, 0, car.z);
      group.rotation.y = -car.heading;
      return group;
    }

    // Ego car.
    const carMesh = createCarMeshHelper({ color: C.ego });
    scene.add(carMesh.group);
    let carHandle = createRaycastVehicle(world, {
      id: 'parking-ego',
      position: { x: scenario.spawn.x, z: scenario.spawn.z },
      heading: scenario.spawn.heading,
    });

    const goalMarker = createGoalMarkerHelper({ color: C.goal });
    scene.add(goalMarker);

    const debugGroup = new THREE.Group();
    debugGroup.visible = false;
    scene.add(debugGroup);
    const footprint = createAgentFootprintHelper(PARKING_AGENT.footprint, {
      color: C.ego,
    });
    debugGroup.add(footprint);

    let pathLine: THREE.Line | null = null;
    function replacePathLine(path: VehicleState[]): void {
      if (pathLine) {
        scene.remove(pathLine);
        pathLine.geometry.dispose();
        (pathLine.material as THREE.Material).dispose();
      }
      const pts = path.map((p) => new THREE.Vector3(p.x, 0.3, p.z));
      pathLine = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineBasicMaterial({
          color: C.egoPath,
          transparent: true,
          opacity: 0.9,
        }),
      );
      scene.add(pathLine);
    }

    const ai = {
      // The FULL plan as returned by the planner — kept for the path
      // overlay so the user sees the whole intended maneuver.
      plan: null as VehicleState[] | null,
      // The plan split at every gear cusp (forward↔reverse). Pure-pursuit
      // tracks one segment at a time, otherwise its lookahead reaches
      // past the cusp and it never executes the gear change cleanly.
      segments: [] as VehicleState[][],
      activeSegIdx: 0,
      // Pose + wall-clock time the chassis was in when the active segment
      // began. We use these to gate the "segment done, advance" check on
      // actual displacement / elapsed time, so a corrective plan whose
      // first segment endpoint happens to be within the arriveTol of the
      // current chassis pose isn't instantly skipped before pure-pursuit
      // has a chance to drive it.
      segStartX: 0,
      segStartZ: 0,
      segStartWall: 0,
      planStartWall: performance.now(),
      goal: null as VehicleState | null,
      lastExpansions: 0,
      lastBudgetMs: 0,
      // Last command we applied + tracker diagnostics, for the debug HUD.
      lastSteer: 0,
      lastThrottle: 0,
      lastBrake: 0,
      lastGear: 0 as -1 | 0 | 1,
      lastCrossTrackErr: 0,
    };

    function replan(deadlineMs: number) {
      const state = carHandle.readState(performance.now());
      ai.goal = scenario.goal;
      const t0 = performance.now();
      const res = planParking({
        scenario,
        state: { ...state, t: 0 },
        world: navWorld,
        deadlineMs,
      });
      ai.lastBudgetMs = performance.now() - t0;
      ai.lastExpansions = res.stats.expansions;
      if (res.found && res.path.length > 1) {
        ai.plan = res.path;
        ai.segments = splitAtGearCusps(res.path);
        ai.activeSegIdx = 0;
        ai.planStartWall = performance.now();
        ai.segStartX = state.x;
        ai.segStartZ = state.z;
        ai.segStartWall = performance.now();
        if (showPathRef.current) replacePathLine(res.path);
      } else if (ai.plan === null) {
        // Replan failed and we had nothing — leave it null so the brake
        // branch holds the car still.
        ai.plan = null;
        ai.segments = [];
      }
    }

    // Periodic replan corrects for pure-pursuit drift mid-maneuver. The
    // tight ~250 ms cap keeps the demo responsive after the first plan
    // (which is given a longer budget below); subsequent replans usually
    // start from a state close to the existing path and finish fast.
    const REPLAN_INTERVAL_MS = 600;
    const REPLAN_DEADLINE_MS = 250;
    // First-shot plan gets a longer budget so the harder scenarios
    // (reverse-perp, parallel) actually find a path on scene load instead
    // of stranding the car.
    const FIRST_PLAN_DEADLINE_MS = 2000;

    function rebuildScenario(id: ParkingScenarioId) {
      scenario = buildParkingScenario(id);
      navWorld = new InMemoryNavWorld(scenario.polygons, scenario.obstacles);
      buildSceneFor(scenario);
      carHandle.teleport({
        x: scenario.spawn.x,
        z: scenario.spawn.z,
        heading: scenario.spawn.heading,
      });
      ai.plan = null;
      replan(FIRST_PLAN_DEADLINE_MS);
    }
    rebuildRef.current = rebuildScenario;

    // First build.
    buildSceneFor(scenario);
    replan(FIRST_PLAN_DEADLINE_MS);

    // Input.
    const onKeyDown = (e: KeyboardEvent) => {
      const k = e.key.toLowerCase();
      if (k === 'p') setPaused((v) => !v);
      if (k === '1') setScenarioId('forward-pullin');
      if (k === '2') setScenarioId('reverse-perp');
      if (k === '3') setScenarioId('parallel');
      if (k === 'd') setShowDebug((v) => !v);
      if (k === 'l') setShowPath((v) => !v);
      if (k === 'r') {
        carHandle.teleport({
          x: scenario.spawn.x,
          z: scenario.spawn.z,
          heading: scenario.spawn.heading,
        });
        ai.plan = null;
        replan(FIRST_PLAN_DEADLINE_MS);
      }
    };
    window.addEventListener('keydown', onKeyDown);

    // Parking is a committed maneuver — the first plan is given a long
    // budget on purpose so it can find a quality multi-shunt path; once
    // it's committed, follow it through. A periodic replan that
    // re-targets every 600 ms (the old behaviour here) actively breaks
    // tight maneuvers, because at a cusp / near the goal the planner can
    // legitimately return a *different* shorter plan from the current
    // chassis pose — pure-pursuit then whips between the two and the
    // car ends up wedged against a parked car.
    //
    // Instead, only replan when the controller has nothing to chase:
    //   - no plan exists yet, or
    //   - the existing plan has been executed past its last sample but
    //     the chassis hasn't actually reached the goal.
    // tick() (below) sets `ai.plan = null` once the plan is exhausted,
    // so the periodic poll just needs to look for that null and refresh.
    const REPLAN_POLL_MS = 600;
    const replanTimer = window.setInterval(() => {
      if (pausedRef.current) return;
      if (ai.plan !== null) return;
      const state = carHandle.readState(performance.now());
      const dGoal = Math.hypot(
        state.x - scenario.goal.x,
        state.z - scenario.goal.z,
      );
      const dHead = Math.abs(angleDiff(state.heading, scenario.goal.heading));
      // Already parked — let the brake-on-no-plan branch hold us still.
      // Treat as parked once the chassis is comfortably inside the goal
      // tolerance window AND nearly stopped. The heading slack (~10°) is
      // wider than the planner's goalHeadingTol (~8°) because pure-pursuit
      // on a Rapier raycast vehicle can't reliably trim heading below
      // that under tight-radius corrective shimmies — replanning more
      // doesn't help, it just busies the loop and the user sees the path
      // line flicker.
      if (dGoal < 0.6 && dHead < 0.18 && Math.abs(state.speed) < 0.4) return;
      replan(FIRST_PLAN_DEADLINE_MS);
    }, REPLAN_POLL_MS);

    const onResize = () => {
      const w = window.innerWidth;
      const h = window.innerHeight;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    window.addEventListener('resize', onResize);

    let stopped = false;
    let lastHudWall = 0;
    function tick() {
      if (stopped) return;
      requestAnimationFrame(tick);
      if (pausedRef.current) {
        renderer.render(scene, camera);
        return;
      }
      const now = performance.now();
      const state = carHandle.readState(now);

      const seg = ai.segments[ai.activeSegIdx];
      if (seg && seg.length > 1) {
        // Track the END of the current segment (the cusp pose or the
        // final goal pose), not the whole plan. If the chassis is close
        // enough to it AND nearly stopped, advance to the next segment
        // — that performs the gear-change handoff cleanly.
        //
        // CRITICAL: also require either that the chassis has actually
        // moved >= 0.25 m from where the segment began OR that ≥ 1.2 s
        // of wall-clock has elapsed. Without this gate, a short
        // corrective segment whose endpoint is already within arriveTol
        // of the current pose advances on its first tick — before
        // pure-pursuit can fire any throttle — and the chassis sits
        // still while plan→null→replan→short-segment→… loops.
        const segEnd = seg[seg.length - 1]!;
        const dEnd = Math.hypot(state.x - segEnd.x, state.z - segEnd.z);
        const isFinalSeg = ai.activeSegIdx === ai.segments.length - 1;
        const arriveTol = isFinalSeg ? 0.25 : 0.4;
        const segMoved = Math.hypot(
          state.x - ai.segStartX,
          state.z - ai.segStartZ,
        );
        const segElapsed = (now - ai.segStartWall) / 1000;
        const segHadAChance = segMoved >= 0.25 || segElapsed >= 1.2;
        if (segHadAChance && dEnd < arriveTol && Math.abs(state.speed) < 0.3) {
          if (isFinalSeg) {
            // Whole plan executed — clear it. The poll loop will check
            // goal proximity and either stop or replan.
            ai.plan = null;
            ai.segments = [];
            ai.lastSteer = 0;
            ai.lastThrottle = 0;
            ai.lastBrake = 1;
            ai.lastGear = 0;
            carHandle.applyControls({ steer: 0, throttle: 0, brake: 1 });
          } else {
            ai.activeSegIdx++;
            ai.planStartWall = now;
            ai.segStartX = state.x;
            ai.segStartZ = state.z;
            ai.segStartWall = now;
            ai.lastSteer = 0;
            ai.lastThrottle = 0;
            ai.lastBrake = 1;
            ai.lastGear = 0;
            carHandle.applyControls({ steer: 0, throttle: 0, brake: 1 });
          }
        } else {
          const cmd = planToAckermannControls(state, seg, {
            wheelBase: 2 * WHEEL_BASE,
            // Tight lookahead — at parking speeds the chassis needs to
            // stay on the exact RS arc; a long lookahead cuts the inside
            // and clips a fender. lookaheadMax stays well BELOW any
            // reasonable segment length so it never reaches past the
            // segment's end pose.
            lookaheadMin: 0.5,
            lookaheadGain: 0.3,
            lookaheadMax: 1.5,
            maxLateralAccel: 2.5,
            maxAccel: 1.5,
            maxDecel: 4,
            cruiseSpeed: PARKING_AGENT.maxSpeed,
            goalTolerance: arriveTol,
            minTurnRadius: PARKING_AGENT.minTurnRadius,
          });
          carHandle.applyControls(cmd);
          ai.lastSteer = cmd.steer;
          ai.lastThrottle = cmd.throttle;
          ai.lastBrake = cmd.brake;
          // Pure-pursuit picks gear from the ahead-on-path speed; the
          // segments are pure-gear by construction so the segment's
          // *second* state speed is a stable indicator of intent.
          const segAhead = seg[1]?.speed ?? 0;
          ai.lastGear = segAhead > 1e-3 ? 1 : segAhead < -1e-3 ? -1 : 0;
          // Cross-track error: distance from chassis to the nearest
          // path sample. A large value means the chassis has wandered
          // off the planned curve.
          let bestD = Infinity;
          for (const p of seg) {
            const d = Math.hypot(state.x - p.x, state.z - p.z);
            if (d < bestD) bestD = d;
          }
          ai.lastCrossTrackErr = bestD;
        }
      } else {
        ai.lastSteer = 0;
        ai.lastThrottle = 0;
        ai.lastBrake = 1;
        ai.lastGear = 0;
        carHandle.applyControls({ steer: 0, throttle: 0, brake: 1 });
      }

      const subDt = PHYSICS_DT / VEHICLE_SUBSTEPS;
      world.timestep = subDt;
      const wheelFilter = RAPIER.QueryFilterFlags.EXCLUDE_DYNAMIC;
      for (let s = 0; s < VEHICLE_SUBSTEPS; s++) {
        carHandle.vehicle.updateVehicle(subDt, wheelFilter);
        world.step();
      }

      const after = carHandle.readState(now);
      syncCarMesh(carMesh.group, after);
      if (pathLine) pathLine.visible = showPathRef.current;

      debugGroup.visible = showDebugRef.current;
      sceneObstacleClearance.visible = showDebugRef.current;
      if (showDebugRef.current) {
        footprint.position.set(after.x, 0, after.z);
        footprint.rotation.y = -after.heading;
      }

      goalMarker.visible = !!ai.goal;
      if (ai.goal) goalMarker.position.set(ai.goal.x, 1.5, ai.goal.z);

      // Static top-down-ish camera; the demo is small so chase-cam is not
      // necessary. Recentre slightly above the scenario centre.
      const sx = (scenario.bounds.x0 + scenario.bounds.x1) / 2;
      const sz = (scenario.bounds.z0 + scenario.bounds.z1) / 2;
      orbit.target.lerp(new THREE.Vector3(sx, 0, sz), 0.04);
      orbit.update();

      if (now - lastHudWall > 100) {
        lastHudWall = now;
        const distToGoal = Math.hypot(
          after.x - scenario.goal.x,
          after.z - scenario.goal.z,
        );
        const headErr = (angleDiff(after.heading, scenario.goal.heading) * 180) / Math.PI;
        setHud(
          `v=${Math.abs(after.speed).toFixed(2)} m/s · goal=${distToGoal.toFixed(2)} m · Δθ=${headErr.toFixed(0)}°`,
        );
        setStatus(
          ai.plan
            ? `plan=${ai.plan.length} exp=${ai.lastExpansions} budget=${ai.lastBudgetMs.toFixed(0)}ms`
            : `no plan (exp=${ai.lastExpansions} budget=${ai.lastBudgetMs.toFixed(0)}ms)`,
        );
        if (showDebugRef.current) {
          const segs = ai.segments.length;
          const gearLabel = ai.lastGear > 0 ? 'fwd' : ai.lastGear < 0 ? 'rev' : '—';
          setDebugSeg(
            segs > 0
              ? `seg ${ai.activeSegIdx + 1}/${segs} · ${gearLabel} · xte=${ai.lastCrossTrackErr.toFixed(2)}m`
              : 'no segments',
          );
          setDebugCmd(
            `steer=${(ai.lastSteer * 180 / Math.PI).toFixed(0)}° · thr=${ai.lastThrottle.toFixed(2)} · brk=${ai.lastBrake.toFixed(2)}`,
          );
        }
      }

      renderer.render(scene, camera);
    }
    tick();

    return () => {
      stopped = true;
      window.clearInterval(replanTimer);
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('resize', onResize);
      carHandle.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === mount)
        mount.removeChild(renderer.domElement);
      world.free();
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
          background: 'rgba(10,14,22,0.75)',
          padding: '10px 14px',
          borderRadius: 8,
          border: '1px solid #1f2735',
          maxWidth: 380,
          lineHeight: 1.45,
        }}
      >
        <div style={{ color: '#7fd6ff', fontWeight: 700, marginBottom: 4 }}>
          tight parking — spatial accuracy
        </div>
        <div>{hud}</div>
        <div style={{ opacity: 0.8 }}>{status}</div>
        {showDebug && debugSeg && (
          <div style={{ opacity: 0.75, color: '#ffc466' }}>{debugSeg}</div>
        )}
        {showDebug && debugCmd && (
          <div style={{ opacity: 0.75, color: '#ffc466' }}>{debugCmd}</div>
        )}
        <div style={{ opacity: 0.6, marginTop: 6, fontSize: 11 }}>
          Sub-meter discretisation + footprintInflate clearance margin so the
          planner threads gaps a default car-chase plan would clip. Toggle
          [d] footprint to also see the planner&apos;s clearance ring around
          parked cars and the live control commands.
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginTop: 8 }}>
          {PARKING_SCENARIOS.map((id) => (
            <button
              key={id}
              type="button"
              onClick={() => setScenarioId(id)}
              style={{
                font: '11px ui-monospace, monospace',
                padding: '4px 8px',
                borderRadius: 6,
                border: `1px solid ${scenarioId === id ? '#7fd6ff' : '#1f2735'}`,
                background:
                  scenarioId === id
                    ? 'rgba(127, 214, 255, 0.18)'
                    : 'rgba(20, 26, 38, 0.85)',
                color: scenarioId === id ? '#cdeaff' : '#8c95a4',
                cursor: 'pointer',
                textAlign: 'left',
              }}
            >
              {PARKING_LABELS[id]}
            </button>
          ))}
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 6 }}>
          <ToggleButton
            label="pause"
            shortcut="p"
            on={paused}
            onClick={() => setPaused((v) => !v)}
          />
          <ToggleButton
            label="path"
            shortcut="l"
            on={showPath}
            onClick={() => setShowPath((v) => !v)}
          />
          <ToggleButton
            label="footprint"
            shortcut="d"
            on={showDebug}
            onClick={() => setShowDebug((v) => !v)}
          />
        </div>
        <div style={{ opacity: 0.55, marginTop: 8, fontSize: 11 }}>
          [1] [2] [3] scenarios · [r] reset · [p] pause · [l] path · [d] footprint
        </div>
      </div>
    </div>
  );
}

function angleDiff(a: number, b: number): number {
  let d = a - b;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return d;
}

/** Split a plan into monotonic-gear segments: each segment is purely
 *  forward OR purely reverse (no mixed gears). The cusp itself is
 *  represented as the boundary BETWEEN segments — seg1 ends with the
 *  last forward state, seg2 starts with the first reverse state. The
 *  two positions are adjacent on the path (one primitive substep apart),
 *  so the geometric handoff is seamless.
 *
 *  An earlier version of this function shared the cusp pose between the
 *  two segments. That put a reverse-speed state at the END of the
 *  forward segment, so pure-pursuit's `nearestIndex + 1` lookup near
 *  the end of the segment found the wrong-sign speed there and flipped
 *  gear mid-segment. The chassis would brake, oscillate, and overshoot;
 *  by the time my advance check fired, the chassis was metres past the
 *  cusp and pure-pursuit on the reverse segment couldn't recover.
 *  Keeping segments pure avoids that. */
function splitAtGearCusps(plan: VehicleState[]): VehicleState[][] {
  if (plan.length < 2) return [plan];
  const out: VehicleState[][] = [];
  let cur: VehicleState[] = [plan[0]!];
  for (let i = 1; i < plan.length; i++) {
    const s = plan[i]!;
    const prev = cur[cur.length - 1]!;
    const flipped =
      (prev.speed > 1e-3 && s.speed < -1e-3) ||
      (prev.speed < -1e-3 && s.speed > 1e-3);
    if (flipped) {
      // Close current segment WITHOUT the new state. New segment starts
      // at the new state.
      if (cur.length >= 2) out.push(cur);
      cur = [s];
    } else {
      cur.push(s);
    }
  }
  if (cur.length >= 2) out.push(cur);
  return out;
}

/** Axis-aligned bounding box of a 2D polygon — the planner's collision
 *  obstacles are axis-aligned rectangles already (built via the scenario
 *  `box(...)` helper) so this is exact, not a conservative envelope. */
function polygonAabb(
  poly: ReadonlyArray<readonly [number, number]>,
): { x0: number; x1: number; z0: number; z1: number } {
  let x0 = Infinity;
  let x1 = -Infinity;
  let z0 = Infinity;
  let z1 = -Infinity;
  for (const [x, z] of poly) {
    if (x < x0) x0 = x;
    if (x > x1) x1 = x;
    if (z < z0) z0 = z;
    if (z > z1) z1 = z;
  }
  return { x0, x1, z0, z1 };
}

/** Pink outline around an obstacle AABB at +`inflate` on every side —
 *  the visible representation of the planner's footprintInflate band. */
function createObstacleClearanceRing(
  aabb: { x0: number; x1: number; z0: number; z1: number },
  inflate: number,
  color: number,
): THREE.Line {
  const y = 0.12;
  const ring = [
    new THREE.Vector3(aabb.x0 - inflate, y, aabb.z0 - inflate),
    new THREE.Vector3(aabb.x1 + inflate, y, aabb.z0 - inflate),
    new THREE.Vector3(aabb.x1 + inflate, y, aabb.z1 + inflate),
    new THREE.Vector3(aabb.x0 - inflate, y, aabb.z1 + inflate),
    new THREE.Vector3(aabb.x0 - inflate, y, aabb.z0 - inflate),
  ];
  return new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(ring),
    new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.6 }),
  );
}

function trimPlan(plan: VehicleState[], elapsed: number): VehicleState[] {
  let i = 0;
  while (i < plan.length - 1 && plan[i + 1]!.t <= elapsed) i++;
  return plan.slice(i);
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
