'use client';

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import {
  createNavMeshHelper,
  createNavMeshOffMeshConnectionsHelper,
} from 'navcat/three';
import { purePursuit, ReplanState } from 'kinocat/execute';
import { kinematicForwardSim } from 'kinocat/agent';
import type { VehicleState } from 'kinocat/agent';
import { buildNavmesh, planNavmesh, DEMO_AGENT as agent } from '../lib/scenarios';

const sim = kinematicForwardSim(agent);
const PP = {
  lookaheadMin: 2,
  lookaheadGain: 0.3,
  lookaheadMax: 6,
  maxLateralAccel: 6,
  maxAccel: 8,
  maxDecel: 8,
  cruiseSpeed: agent.maxSpeed,
  goalTolerance: 1.4,
  minTurnRadius: agent.minTurnRadius,
};

export default function NavMeshView() {
  const mountRef = useRef<HTMLDivElement>(null);
  const [info, setInfo] = useState('building navmesh…');
  const goalRef = useRef<VehicleState>({ x: 36, z: 12, heading: 0, speed: 0, t: 0 });

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    const W = mount.clientWidth;
    const viewH = () =>
      Math.round(Math.min(480, Math.max(280, window.innerHeight * 0.6)));
    let Hpx = viewH();

    const { world, navMesh } = buildNavmesh();

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0b0f);
    const camera = new THREE.PerspectiveCamera(55, W / Hpx, 0.1, 500);
    camera.position.set(8, 28, 44);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(W, Hpx);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(20, 2, 12);
    controls.update();

    scene.add(new THREE.AmbientLight(0xffffff, 0.65));
    const sun = new THREE.DirectionalLight(0xffffff, 1);
    sun.position.set(12, 30, 8);
    scene.add(sun);

    // the navcat navmesh debug helper
    const navHelper = createNavMeshHelper(navMesh);
    const offHelper = createNavMeshOffMeshConnectionsHelper(navMesh);
    scene.add(navHelper.object, offHelper.object);

    const mkSphere = (c: number) =>
      new THREE.Mesh(new THREE.SphereGeometry(0.6, 16, 16), new THREE.MeshStandardMaterial({ color: c }));
    const startMarker = mkSphere(0x55ff88);
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

    const surfaceY = (x: number, z: number): number => {
      const p = world.polygonAt(x, z);
      return p ? world.heightAt(p, x, z) : 0;
    };

    let pathLine: THREE.Line | null = null;
    const drawPath = (path: VehicleState[]) => {
      if (pathLine) {
        scene.remove(pathLine);
        pathLine.geometry.dispose();
      }
      const pts = path.map(
        (s) => new THREE.Vector3(s.x, surfaceY(s.x, s.z) + 0.15, s.z),
      );
      pathLine = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineBasicMaterial({ color: 0x44ddff }),
      );
      scene.add(pathLine);
    };

    let state: VehicleState = { x: 4, z: 12, heading: 0, speed: 0, t: 0 };
    const replan = new ReplanState({ divergenceThresholdMeters: 2.5, refreshIntervalMs: 1800 });

    const doReplan = (nowMs: number) => {
      const t0 = performance.now();
      const r = planNavmesh(world, state, goalRef.current);
      if (r.found) {
        replan.setPlan(r.path, nowMs);
        drawPath(r.path);
        setInfo(
          `navmesh: ${(performance.now() - t0).toFixed(0)} ms plan · cost ${r.cost.toFixed(1)} · ${r.path.length} states · drag to orbit, click the mesh to move the goal`,
        );
      } else {
        setInfo('no plan to that point — try elsewhere on the mesh');
      }
    };

    const place = (m: THREE.Mesh, s: VehicleState) =>
      m.position.set(s.x, surfaceY(s.x, s.z) + 0.6, s.z);
    place(startMarker, state);
    place(goalMarker, goalRef.current);
    doReplan(performance.now());

    const ray = new THREE.Raycaster();
    const ndc = new THREE.Vector2();
    const onClick = (e: MouseEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      ndc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      ndc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      ray.setFromCamera(ndc, camera);
      const hit = ray.intersectObject(navHelper.object, true)[0];
      if (!hit) return;
      const g: VehicleState = {
        x: hit.point.x,
        z: hit.point.z,
        heading: 0,
        speed: 0,
        t: 0,
      };
      goalRef.current = g;
      place(goalMarker, g);
      doReplan(performance.now());
    };
    renderer.domElement.addEventListener('click', onClick);

    let frame = 0;
    let last = performance.now();
    const animate = () => {
      frame = requestAnimationFrame(animate);
      const now = performance.now();
      const dt = Math.min((now - last) / 1000, 0.05);
      last = now;
      const path = replan.currentPlan;
      if (path) {
        const cmd = purePursuit(state, path, PP);
        if (!cmd.atGoal) state = sim(state, [cmd.steering, cmd.targetSpeed], dt);
        if (replan.shouldReplan(state, now)) doReplan(now);
      }
      vehicle.position.set(state.x, surfaceY(state.x, state.z), state.z);
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
      renderer.domElement.removeEventListener('click', onClick);
      navHelper.dispose();
      offHelper.dispose();
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
      <h1 style={{ fontSize: 18 }}>3D navmesh debug view</h1>
      <p style={{ opacity: 0.75, marginTop: 0 }}>
        A real navcat navmesh generated in the browser (ground → ramp →
        raised platform) rendered with navcat&apos;s <code>createNavMeshHelper</code>.
        kinocat plans over the <code>NavcatWorld</code> adapter and tracks the
        path with pure-pursuit, climbing the ramp.
      </p>
      <div ref={mountRef} style={{ width: '100%', borderRadius: 8, overflow: 'hidden' }} />
      <p style={{ opacity: 0.8 }}>{info}</p>
    </main>
  );
}
