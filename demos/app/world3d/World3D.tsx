'use client';

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { plan } from 'kinocat/planner';
import { InMemoryNavWorld, VehicleEnvironment } from 'kinocat/environment';
import { purePursuit, ReplanState } from 'kinocat/execute';
import { kinematicForwardSim } from 'kinocat/agent';
import type { VehicleState } from 'kinocat/agent';
import { buildVehicle } from '../lib/vehicle';

const { agent, lib } = buildVehicle();
const sim = kinematicForwardSim(agent);
const OBSTACLE: [number, number, number, number] = [17, -3, 23, 3];

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

function makeWorld() {
  return new InMemoryNavWorld(
    [{ id: 1, y: 0, ring: [[0, -12], [40, -12], [40, 12], [0, 12]] }],
    [
      [
        [OBSTACLE[0], OBSTACLE[1]],
        [OBSTACLE[2], OBSTACLE[1]],
        [OBSTACLE[2], OBSTACLE[3]],
        [OBSTACLE[0], OBSTACLE[3]],
      ],
    ],
  );
}

export default function World3D() {
  const mountRef = useRef<HTMLDivElement>(null);
  const [info, setInfo] = useState('drag to orbit · click the ground to move the goal');
  const goalRef = useRef<VehicleState>({ x: 36, z: 0, heading: 0, speed: 0, t: 0 });
  const knockRef = useRef(false);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    const W = mount.clientWidth;
    const viewHeight = () =>
      Math.round(Math.min(460, Math.max(260, window.innerHeight * 0.55)));
    let Hpx = viewHeight();

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0b0f);
    const camera = new THREE.PerspectiveCamera(55, W / Hpx, 0.1, 500);
    camera.position.set(20, 34, 40);
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

    // ground (the navmesh region)
    const ground = new THREE.Mesh(
      new THREE.BoxGeometry(40, 0.5, 24),
      new THREE.MeshStandardMaterial({ color: 0x161a22 }),
    );
    ground.position.set(20, -0.25, 0);
    scene.add(ground);

    const obstacle = new THREE.Mesh(
      new THREE.BoxGeometry(OBSTACLE[2] - OBSTACLE[0], 3, OBSTACLE[3] - OBSTACLE[1]),
      new THREE.MeshStandardMaterial({ color: 0x5a2230 }),
    );
    obstacle.position.set(
      (OBSTACLE[0] + OBSTACLE[2]) / 2,
      1.5,
      (OBSTACLE[1] + OBSTACLE[3]) / 2,
    );
    scene.add(obstacle);

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

    let pathLine: THREE.Line | null = null;
    const drawPath = (path: VehicleState[]) => {
      if (pathLine) scene.remove(pathLine);
      const pts = path.map((s) => new THREE.Vector3(s.x, 0.1, s.z));
      pathLine = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineBasicMaterial({ color: 0x44ddff }),
      );
      scene.add(pathLine);
    };

    let state: VehicleState = { x: 4, z: 0, heading: 0, speed: 0, t: 0 };
    const replan = new ReplanState({ divergenceThresholdMeters: 2.5, refreshIntervalMs: 1500 });

    const doReplan = (nowMs: number) => {
      const env = new VehicleEnvironment(makeWorld(), agent, lib, {
        goalRadius: 1.5,
        goalHeadingTol: Infinity,
      });
      const t0 = performance.now();
      const r = plan(
        { start: { ...state, t: 0 }, goal: goalRef.current, environment: env, options: { maxExpansions: 200000 } },
        60,
      );
      if (r.found) {
        replan.setPlan(r.path, nowMs);
        drawPath(r.path);
        setInfo(
          `replanned in ${(performance.now() - t0).toFixed(0)} ms · cost ${r.cost.toFixed(1)} · ${r.path.length} states`,
        );
      } else {
        setInfo('no plan to that goal — pick another spot');
      }
    };

    goalMarker.position.set(goalRef.current.x, 0.6, goalRef.current.z);
    doReplan(performance.now());

    // click ground → move goal
    const ray = new THREE.Raycaster();
    const ndc = new THREE.Vector2();
    const onClick = (e: MouseEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      ndc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      ndc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      ray.setFromCamera(ndc, camera);
      const hit = ray.intersectObject(ground)[0];
      if (!hit) return;
      const gx = Math.max(1, Math.min(39, hit.point.x));
      const gz = Math.max(-11, Math.min(11, hit.point.z));
      goalRef.current = { x: gx, z: gz, heading: 0, speed: 0, t: 0 };
      goalMarker.position.set(gx, 0.6, gz);
      doReplan(performance.now());
    };
    renderer.domElement.addEventListener('click', onClick);

    let frame = 0;
    let lastMs = performance.now();
    const animate = () => {
      frame = requestAnimationFrame(animate);
      const now = performance.now();
      const dt = Math.min((now - lastMs) / 1000, 0.05);
      lastMs = now;

      if (knockRef.current) {
        state = { ...state, z: state.z + 4 };
        knockRef.current = false;
      }

      const path = replan.currentPlan;
      if (path) {
        const cmd = purePursuit(state, path, PP);
        if (!cmd.atGoal) {
          state = sim(state, [cmd.steering, cmd.targetSpeed], dt);
        }
        if (replan.shouldReplan(state, now)) doReplan(now);
      }
      vehicle.position.set(state.x, 0, state.z);
      vehicle.rotation.y = -state.heading;

      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const onResize = () => {
      const w = mount.clientWidth;
      Hpx = viewHeight();
      camera.aspect = w / Hpx;
      camera.updateProjectionMatrix();
      renderer.setSize(w, Hpx);
    };
    window.addEventListener('resize', onResize);

    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener('resize', onResize);
      renderer.domElement.removeEventListener('click', onClick);
      controls.dispose();
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
  }, []);

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
        The vehicle plans (IGHA*) and tracks the path with curvature-aware
        pure-pursuit. Drag to orbit; click the ground to move the goal;
        “knock” it off-course to see divergence-triggered replanning.
      </p>
      <div ref={mountRef} style={{ width: '100%', borderRadius: 8, overflow: 'hidden' }} />
      <div style={{ marginTop: 10 }}>
        <button
          onClick={() => {
            knockRef.current = true;
          }}
          style={{ background: '#161a22', color: '#cdd3de', border: '1px solid #2a2f3a', borderRadius: 6, padding: '6px 12px', cursor: 'pointer' }}
        >
          knock vehicle off-course
        </button>
      </div>
      <p style={{ opacity: 0.8 }}>{info}</p>
    </main>
  );
}
