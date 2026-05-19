'use client';

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import {
  createNavMeshHelper,
  createNavMeshOffMeshConnectionsHelper,
} from 'navcat/three';
import { createPlanPathHelper } from 'kinocat/adapters/three';
import type { HumanoidState } from 'kinocat/agent';
import { buildJumpLinks } from '../lib/scenarios';

function walkPoseAt(path: HumanoidState[], t: number): { x: number; z: number } {
  if (path.length === 0) return { x: 0, z: 0 };
  const first = path[0]!;
  const last = path[path.length - 1]!;
  if (t <= first.t) return { x: first.x, z: first.z };
  if (t >= last.t) return { x: last.x, z: last.z };
  for (let i = 0; i < path.length - 1; i++) {
    const a = path[i]!;
    const b = path[i + 1]!;
    if (t >= a.t && t <= b.t) {
      const span = b.t - a.t;
      const u = span > 1e-9 ? (t - a.t) / span : 0;
      return { x: a.x + (b.x - a.x) * u, z: a.z + (b.z - a.z) * u };
    }
  }
  return { x: last.x, z: last.z };
}

export default function JumpLinks() {
  const mountRef = useRef<HTMLDivElement>(null);
  const [info, setInfo] = useState('building navmesh + annotating jump link…');

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    let res;
    try {
      res = buildJumpLinks();
    } catch {
      setInfo(
        'navcat navmesh generation is unavailable in this environment — see the headless test for the asserted behaviour.',
      );
      return;
    }
    const { navMesh, linkMeta, withLink, without } = res;
    const path = withLink.path;
    const duration = path.length ? path[path.length - 1]!.t : 1;

    const W = mount.clientWidth;
    const viewH = () =>
      Math.round(Math.min(480, Math.max(280, window.innerHeight * 0.6)));
    let Hpx = viewH();

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0b0f);
    const camera = new THREE.PerspectiveCamera(55, W / Hpx, 0.1, 500);
    camera.position.set(12, 22, 30);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(W, Hpx);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(12, 0, 4.5);
    controls.update();

    scene.add(new THREE.AmbientLight(0xffffff, 0.65));
    const sun = new THREE.DirectionalLight(0xffffff, 1);
    sun.position.set(10, 28, 8);
    scene.add(sun);

    const navHelper = createNavMeshHelper(navMesh);
    const offHelper = createNavMeshOffMeshConnectionsHelper(navMesh);
    scene.add(navHelper.object, offHelper.object);
    scene.add(createPlanPathHelper(path, { y: 0.2, color: 0x44ddff }));

    const mkSphere = (c: number) =>
      new THREE.Mesh(
        new THREE.SphereGeometry(0.45, 16, 16),
        new THREE.MeshStandardMaterial({ color: c }),
      );
    const startMarker = mkSphere(0x55ff88);
    startMarker.position.set(res.start.x, 0.45, res.start.z);
    const goalMarker = mkSphere(0xffcc33);
    goalMarker.position.set(res.goal.x, 0.45, res.goal.z);
    scene.add(startMarker, goalMarker);

    const walker = new THREE.Mesh(
      new THREE.CylinderGeometry(0.3, 0.3, 1.4, 14),
      new THREE.MeshStandardMaterial({ color: 0x7fd6ff }),
    );
    scene.add(walker);

    setInfo(
      `annotateJumpLinks registered ${linkMeta.length} off-mesh connection · ` +
        `without it: ${without.found ? 'plan found' : 'no plan (gap uncrossable)'} · ` +
        `with it: ${withLink.found ? 'plan found' : 'no plan'} · ` +
        `uses the jump: ${withLink.usedJump ? 'yes' : 'no'} · drag to orbit`,
    );

    let frame = 0;
    let last = performance.now();
    let t = 0;
    const animate = () => {
      frame = requestAnimationFrame(animate);
      const now = performance.now();
      t += (now - last) / 1000;
      if (t > duration) t = 0;
      last = now;
      const p = walkPoseAt(path, t);
      walker.position.set(p.x, 0.7, p.z);
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
      <a href="/" style={{ color: '#7fd6ff' }}>
        ← demos
      </a>
      <h1 style={{ fontSize: 18 }}>Static jump links (Mononen annotation)</h1>
      <p style={{ opacity: 0.75, marginTop: 0 }}>
        Two islands separated by a real gap on a generated navcat navmesh.{' '}
        <code>annotateJumpLinks</code> validates and registers a Mononen-style
        off-mesh connection across the gap; the humanoid planner then crosses it
        (a navcat off-mesh link a vehicle env would not consume — humanoids do).
      </p>
      <div
        ref={mountRef}
        style={{ width: '100%', borderRadius: 8, overflow: 'hidden' }}
      />
      <p style={{ opacity: 0.8 }}>{info}</p>
    </main>
  );
}
