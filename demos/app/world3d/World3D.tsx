'use client';

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { purePursuit, ReplanState } from 'kinocat/execute';
import { kinematicForwardSim } from 'kinocat/agent';
import type { CarKinematicState } from 'kinocat/agent';
import {
  world3dWorldFrom,
  planWorld3d,
  WORLD3D_DEFAULT_OBSTACLES,
  WORLD3D_BOUNDS as B,
  DEMO_AGENT as agent,
  type BoxObstacle,
} from '../lib/scenarios';

const sim = kinematicForwardSim(agent);
const PP = {
  lookaheadMin: 2,
  lookaheadGain: 0.3,
  lookaheadMax: 6,
  maxLateralAccel: 6,
  maxAccel: 8,
  maxDecel: 8,
  cruiseSpeed: agent.maxSpeed,
  goalTolerance: 1.2,
  minTurnRadius: agent.minTurnRadius,
};

type Mode = 'goal' | 'obstacle';

export default function World3D() {
  const mountRef = useRef<HTMLDivElement>(null);
  const [info, setInfo] = useState('drag to orbit · tap the ground to move the goal');
  const [mode, setMode] = useState<Mode>('goal');
  const modeRef = useRef<Mode>('goal');
  const obstaclesRef = useRef<BoxObstacle[]>(
    WORLD3D_DEFAULT_OBSTACLES.map((o) => ({ ...o })),
  );
  const goalRef = useRef<CarKinematicState>({ x: 36, z: 0, heading: 0, speed: 0, t: 0 });
  const apiRef = useRef<{ clear: () => void } | null>(null);

  useEffect(() => {
    modeRef.current = mode;
  }, [mode]);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    const W = mount.clientWidth;
    const viewH = () =>
      Math.round(Math.min(480, Math.max(260, window.innerHeight * 0.58)));
    let Hpx = viewH();

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0b0f);
    const camera = new THREE.PerspectiveCamera(55, W / Hpx, 0.1, 500);
    camera.position.set(20, 34, 42);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(W, Hpx);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(20, 0, 0);
    controls.update();

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const sun = new THREE.DirectionalLight(0xffffff, 1);
    sun.position.set(10, 30, 10);
    scene.add(sun);

    const ground = new THREE.Mesh(
      new THREE.BoxGeometry(B.x1 - B.x0, 0.5, B.z1 - B.z0),
      new THREE.MeshStandardMaterial({ color: 0x161a22 }),
    );
    ground.position.set((B.x0 + B.x1) / 2, -0.25, (B.z0 + B.z1) / 2);
    scene.add(ground);

    const mkSphere = (color: number) =>
      new THREE.Mesh(
        new THREE.SphereGeometry(0.6, 16, 16),
        new THREE.MeshStandardMaterial({ color }),
      );
    const startMarker = mkSphere(0x55ff88);
    startMarker.position.set(4, 0.6, 0);
    const goalMarker = mkSphere(0xffcc33);
    scene.add(startMarker, goalMarker);

    const vehicle = new THREE.Group();
    const body = new THREE.Mesh(
      new THREE.BoxGeometry(2.4, 1, 1.2),
      new THREE.MeshStandardMaterial({ color: 0x7fd6ff }),
    );
    body.position.y = 0.6;
    const nose = new THREE.Mesh(
      new THREE.ConeGeometry(0.35, 0.9, 12),
      new THREE.MeshStandardMaterial({ color: 0xffffff }),
    );
    nose.rotation.z = -Math.PI / 2;
    nose.position.set(1.5, 0.6, 0);
    vehicle.add(body, nose);
    scene.add(vehicle);

    const obstacleGroup = new THREE.Group();
    scene.add(obstacleGroup);
    const obMat = new THREE.MeshStandardMaterial({ color: 0x5a2230 });
    const rebuildObstacleMeshes = () => {
      for (const c of [...obstacleGroup.children]) {
        obstacleGroup.remove(c);
        ((c as THREE.Mesh).geometry as THREE.BufferGeometry).dispose();
      }
      obstaclesRef.current.forEach((o, i) => {
        const m = new THREE.Mesh(
          new THREE.BoxGeometry(o.hx * 2, 2.6, o.hz * 2),
          obMat,
        );
        m.position.set(o.x, 1.3, o.z);
        m.userData.index = i;
        obstacleGroup.add(m);
      });
    };
    rebuildObstacleMeshes();

    let world = world3dWorldFrom(obstaclesRef.current);
    let state: CarKinematicState = { x: 4, z: 0, heading: 0, speed: 0, t: 0 };
    const replan = new ReplanState({
      divergenceThresholdMeters: 2.5,
      refreshIntervalMs: 1500,
      switchCostImprovement: 0.15,
      switchCostMargin: 0.5,
    });

    let pathLine: THREE.Line | null = null;
    const drawPath = (path: CarKinematicState[]) => {
      if (pathLine) {
        scene.remove(pathLine);
        pathLine.geometry.dispose();
      }
      pathLine = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(
          path.map((s) => new THREE.Vector3(s.x, 0.12, s.z)),
        ),
        new THREE.LineBasicMaterial({ color: 0x44ddff }),
      );
      scene.add(pathLine);
    };

    const doReplan = (nowMs: number, force: boolean) => {
      if (force) replan.markDirty('edit');
      const t0 = performance.now();
      const r = planWorld3d(world, state, goalRef.current);
      if (r.found) {
        const adopted = replan.consider(r.path, r.cost, nowMs);
        if (adopted) drawPath(r.path);
        setInfo(
          `${adopted ? 'replanned' : 'kept plan'} (${(performance.now() - t0).toFixed(0)} ms) · cost ${r.cost.toFixed(1)} · ${obstaclesRef.current.length} obstacles`,
        );
      } else {
        setInfo('no plan — move the goal or remove an obstacle');
      }
    };

    const rebuildWorldAndReplan = () => {
      world = world3dWorldFrom(obstaclesRef.current);
      rebuildObstacleMeshes();
      doReplan(performance.now(), true);
    };
    apiRef.current = {
      clear: () => {
        obstaclesRef.current = [];
        rebuildWorldAndReplan();
      },
    };

    goalMarker.position.set(goalRef.current.x, 0.6, goalRef.current.z);
    doReplan(performance.now(), true);

    // ---- pointer interaction: orbit, move goal, add / drag obstacles ----
    const ray = new THREE.Raycaster();
    const ndc = new THREE.Vector2();
    const setNdc = (e: PointerEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      ndc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      ndc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    };
    const groundHit = (): THREE.Vector3 | null => {
      ray.setFromCamera(ndc, camera);
      const h = ray.intersectObject(ground)[0];
      return h ? h.point : null;
    };
    let dragIdx = -1;
    let downX = 0;
    let downY = 0;
    let moved = false;

    const onDown = (e: PointerEvent) => {
      setNdc(e);
      downX = e.clientX;
      downY = e.clientY;
      moved = false;
      ray.setFromCamera(ndc, camera);
      const hit = ray.intersectObjects(obstacleGroup.children)[0];
      if (hit) {
        dragIdx = (hit.object as THREE.Mesh).userData.index as number;
        controls.enabled = false;
        renderer.domElement.setPointerCapture(e.pointerId);
      }
    };
    const onMove = (e: PointerEvent) => {
      if (Math.hypot(e.clientX - downX, e.clientY - downY) > 4) moved = true;
      if (dragIdx < 0) return;
      setNdc(e);
      const p = groundHit();
      const o = obstaclesRef.current[dragIdx];
      if (p && o) {
        o.x = Math.max(B.x0 + o.hx, Math.min(B.x1 - o.hx, p.x));
        o.z = Math.max(B.z0 + o.hz, Math.min(B.z1 - o.hz, p.z));
        const mesh = obstacleGroup.children[dragIdx] as THREE.Mesh | undefined;
        if (mesh) mesh.position.set(o.x, 1.3, o.z);
      }
    };
    const onUp = (e: PointerEvent) => {
      if (dragIdx >= 0) {
        dragIdx = -1;
        controls.enabled = true;
        if (renderer.domElement.hasPointerCapture(e.pointerId))
          renderer.domElement.releasePointerCapture(e.pointerId);
        rebuildWorldAndReplan();
        return;
      }
      if (moved) return; // an orbit drag, not a tap
      setNdc(e);
      const p = groundHit();
      if (!p) return;
      if (modeRef.current === 'goal') {
        const g = {
          x: Math.max(1, Math.min(B.x1 - 1, p.x)),
          z: Math.max(B.z0 + 1, Math.min(B.z1 - 1, p.z)),
          heading: 0,
          speed: 0,
          t: 0,
        };
        goalRef.current = g;
        goalMarker.position.set(g.x, 0.6, g.z);
        doReplan(performance.now(), true);
      } else {
        obstaclesRef.current.push({
          x: Math.max(B.x0 + 3, Math.min(B.x1 - 3, p.x)),
          z: Math.max(B.z0 + 3, Math.min(B.z1 - 3, p.z)),
          hx: 2.5,
          hz: 2.5,
        });
        rebuildWorldAndReplan();
      }
    };
    renderer.domElement.addEventListener('pointerdown', onDown);
    renderer.domElement.addEventListener('pointermove', onMove);
    renderer.domElement.addEventListener('pointerup', onUp);

    let frame = 0;
    let lastMs = performance.now();
    const animate = () => {
      frame = requestAnimationFrame(animate);
      const now = performance.now();
      const dt = Math.min((now - lastMs) / 1000, 0.05);
      lastMs = now;
      const path = replan.currentPlan;
      if (path) {
        const cmd = purePursuit(state, path, PP);
        if (!cmd.atGoal) state = sim(state, [cmd.steering, cmd.targetSpeed], dt);
        if (replan.shouldReplan(state, now)) doReplan(now, false);
      }
      vehicle.position.set(state.x, 0, state.z);
      vehicle.rotation.y = -state.heading;
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const onResize = () => {
      const w = mount.clientWidth;
      Hpx = viewH();
      camera.aspect = w / Hpx;
      camera.updateProjectionMatrix();
      renderer.setSize(w, Hpx);
    };
    window.addEventListener('resize', onResize);

    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener('resize', onResize);
      renderer.domElement.removeEventListener('pointerdown', onDown);
      renderer.domElement.removeEventListener('pointermove', onMove);
      renderer.domElement.removeEventListener('pointerup', onUp);
      controls.dispose();
      renderer.dispose();
      mount.removeChild(renderer.domElement);
      apiRef.current = null;
    };
  }, []);

  const btn = (active: boolean) => ({
    background: active ? '#2a3550' : '#161a22',
    color: '#cdd3de',
    border: '1px solid #2a2f3a',
    borderRadius: 6,
    padding: '6px 12px',
    cursor: 'pointer',
  });

  return (
    <main
      style={{
        color: '#cdd3de',
        fontFamily: 'ui-monospace, monospace',
        padding: 'clamp(12px, 4vw, 24px)',
        maxWidth: 920,
        margin: '0 auto',
      }}
    >
      <a href="/" style={{ color: '#7fd6ff' }}>← demos</a>
      <h1 style={{ fontSize: 18 }}>3D navmesh world</h1>
      <p style={{ opacity: 0.75, marginTop: 0 }}>
        The vehicle plans (IGHA*) and tracks with pure-pursuit, with plan-switch
        hysteresis so it commits to a route instead of oscillating between
        equal-cost ones. Drag to orbit; drag any obstacle to move it; pick a
        mode then tap the ground to move the goal or drop an obstacle.
      </p>
      <div style={{ display: 'flex', gap: 8, margin: '8px 0', flexWrap: 'wrap' }}>
        <button onClick={() => setMode('goal')} style={btn(mode === 'goal')}>
          tap = move goal
        </button>
        <button onClick={() => setMode('obstacle')} style={btn(mode === 'obstacle')}>
          tap = add obstacle
        </button>
        <button onClick={() => apiRef.current?.clear()} style={btn(false)}>
          clear obstacles
        </button>
      </div>
      <div ref={mountRef} style={{ width: '100%', borderRadius: 8, overflow: 'hidden' }} />
      <p style={{ opacity: 0.8 }}>{info}</p>
    </main>
  );
}
