'use client';

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { HovercraftState } from '../lib/hovercraft-domain';
import { HOVER_AGENT } from '../lib/hovercraft-domain';
import {
  buildHovercraft,
  planHovercraftLeg,
  hovercraftWorldFrom,
  HOVER_BOUNDS as B,
} from '../lib/hovercraft-scenario';

/** Interpolated pose+velocity along the committed plan at sim time t. */
function poseAt(path: HovercraftState[], t: number): HovercraftState {
  if (path.length === 0) return { x: 0, z: 0, heading: 0, vx: 0, vz: 0, t };
  const first = path[0]!;
  const last = path[path.length - 1]!;
  if (t <= first.t) return { ...first, t };
  if (t >= last.t) return { ...last, vx: 0, vz: 0, t };
  for (let i = 0; i < path.length - 1; i++) {
    const a = path[i]!;
    const b = path[i + 1]!;
    if (t >= a.t && t <= b.t) {
      const span = b.t - a.t;
      const u = span > 1e-9 ? (t - a.t) / span : 0;
      let dh = b.heading - a.heading;
      if (dh > Math.PI) dh -= 2 * Math.PI;
      if (dh < -Math.PI) dh += 2 * Math.PI;
      return {
        x: a.x + (b.x - a.x) * u,
        z: a.z + (b.z - a.z) * u,
        heading: a.heading + dh * u,
        vx: a.vx + (b.vx - a.vx) * u,
        vz: a.vz + (b.vz - a.vz) * u,
        t,
      };
    }
  }
  return { ...last, t };
}

/** Speed → color for the path line: cyan glide → amber full thrust. */
function speedColor(speed: number): THREE.Color {
  const u = Math.min(1, speed / HOVER_AGENT.maxSpeed);
  return new THREE.Color().setHSL((190 - 150 * u) / 360, 0.85, 0.6);
}

export default function Hovercraft() {
  const mountRef = useRef<HTMLDivElement>(null);
  const [info, setInfo] = useState('drag to orbit · tap the ice to retarget mid-drift');

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    const W = mount.clientWidth;
    const viewH = () =>
      Math.round(Math.min(520, Math.max(280, window.innerHeight * 0.6)));
    let Hpx = viewH();

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0e16);
    scene.fog = new THREE.Fog(0x0a0e16, 70, 160);
    const camera = new THREE.PerspectiveCamera(55, W / Hpx, 0.1, 500);
    camera.position.set(32, 42, 52);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(W, Hpx);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(32, 0, 0);
    controls.update();

    scene.add(new THREE.AmbientLight(0x8fb8ff, 0.5));
    const sun = new THREE.DirectionalLight(0xffffff, 1.1);
    sun.position.set(20, 40, 12);
    scene.add(sun);

    // Ice lagoon floor.
    const ice = new THREE.Mesh(
      new THREE.BoxGeometry(B.x1 - B.x0, 0.5, B.z1 - B.z0),
      new THREE.MeshStandardMaterial({
        color: 0x14202e,
        roughness: 0.25,
        metalness: 0.45,
      }),
    );
    ice.position.set((B.x0 + B.x1) / 2, -0.25, (B.z0 + B.z1) / 2);
    scene.add(ice);

    // The scene is built once from the shipped scenario.
    const scenario = buildHovercraft();
    const world = hovercraftWorldFrom(scenario.islands);
    const floes = scenario.floes;

    // Islands.
    const islandMat = new THREE.MeshStandardMaterial({
      color: 0x3c516b,
      roughness: 0.9,
    });
    for (const o of scenario.islands) {
      const m = new THREE.Mesh(
        new THREE.BoxGeometry(o.hx * 2, 3.2, o.hz * 2),
        islandMat,
      );
      m.position.set(o.x, 1.6, o.z);
      scene.add(m);
    }

    // Patrolling ice floes (positions driven by the SAME predictors the
    // planner saw).
    const floeMat = new THREE.MeshStandardMaterial({
      color: 0xbfe3ff,
      roughness: 0.4,
    });
    const floeMeshes = floes.map((f) => {
      const m = new THREE.Mesh(
        new THREE.CylinderGeometry(f.radius, f.radius * 1.15, 1.2, 20),
        floeMat,
      );
      m.position.y = 0.6;
      scene.add(m);
      return m;
    });

    // Goal marker.
    const goalMarker = new THREE.Mesh(
      new THREE.TorusGeometry(1.6, 0.18, 10, 32),
      new THREE.MeshStandardMaterial({ color: 0xffcc33 }),
    );
    goalMarker.rotation.x = Math.PI / 2;
    goalMarker.position.set(scenario.goal.x, 0.25, scenario.goal.z);
    scene.add(goalMarker);

    // The hovercraft: hull + skirt + tail fin; facing ≠ motion, so a
    // separate drift arrow shows the velocity vector.
    const craft = new THREE.Group();
    const skirt = new THREE.Mesh(
      new THREE.CylinderGeometry(HOVER_AGENT.radius, HOVER_AGENT.radius * 1.18, 0.5, 20),
      new THREE.MeshStandardMaterial({ color: 0x22303f, roughness: 0.8 }),
    );
    skirt.position.y = 0.3;
    const hull = new THREE.Mesh(
      new THREE.BoxGeometry(HOVER_AGENT.radius * 1.7, 0.5, HOVER_AGENT.radius * 1.2),
      new THREE.MeshStandardMaterial({ color: 0x7fd6ff }),
    );
    hull.position.y = 0.75;
    const fin = new THREE.Mesh(
      new THREE.BoxGeometry(0.5, 0.9, 0.12),
      new THREE.MeshStandardMaterial({ color: 0xffffff }),
    );
    fin.position.set(-HOVER_AGENT.radius * 0.7, 1.05, 0);
    craft.add(skirt, hull, fin);
    scene.add(craft);
    const driftArrow = new THREE.ArrowHelper(
      new THREE.Vector3(1, 0, 0),
      new THREE.Vector3(0, 0.2, 0),
      1,
      0xffaa44,
      0.8,
      0.5,
    );
    scene.add(driftArrow);

    // Speed-colored committed path.
    let pathLine: THREE.Line | null = null;
    const drawPath = (path: HovercraftState[]) => {
      if (pathLine) {
        scene.remove(pathLine);
        pathLine.geometry.dispose();
        (pathLine.material as THREE.Material).dispose();
      }
      const positions: number[] = [];
      const colors: number[] = [];
      for (const s of path) {
        positions.push(s.x, 0.3, s.z);
        const c = speedColor(Math.hypot(s.vx, s.vz));
        colors.push(c.r, c.g, c.b);
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
      geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
      pathLine = new THREE.Line(
        geo,
        new THREE.LineBasicMaterial({ vertexColors: true }),
      );
      scene.add(pathLine);
    };

    // Playback state: sim time advances with the wall clock; retargeting
    // replans FROM the interpolated state — momentum and floe timing carry
    // into the new plan (kinodynamic, time-extended).
    let path = scenario.result.path;
    let simT = path.length ? path[0]!.t : 0;
    drawPath(path);

    const retarget = (gx: number, gz: number) => {
      const here = poseAt(path, simT);
      const t0 = performance.now();
      const r = planHovercraftLeg(world, floes, here, {
        x: gx,
        z: gz,
        heading: 0,
        vx: 0,
        vz: 0,
        t: simT,
      });
      const ms = performance.now() - t0;
      if (r.found) {
        path = r.path;
        drawPath(path);
        goalMarker.position.set(gx, 0.25, gz);
        setInfo(
          `replanned in ${ms.toFixed(0)} ms · ${r.stats.expansions.toLocaleString()} expansions · ${r.cost.toFixed(1)} s to goal`,
        );
      } else {
        setInfo('no plan — that spot is unreachable right now');
      }
    };

    const ray = new THREE.Raycaster();
    const ndc = new THREE.Vector2();
    let downX = 0;
    let downY = 0;
    let moved = false;
    const onDown = (e: PointerEvent) => {
      downX = e.clientX;
      downY = e.clientY;
      moved = false;
    };
    const onMove = (e: PointerEvent) => {
      if (Math.hypot(e.clientX - downX, e.clientY - downY) > 4) moved = true;
    };
    const onUp = (e: PointerEvent) => {
      if (moved) return; // orbit drag
      const rect = renderer.domElement.getBoundingClientRect();
      ndc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      ndc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      ray.setFromCamera(ndc, camera);
      const h = ray.intersectObject(ice)[0];
      if (!h) return;
      const gx = Math.max(B.x0 + 2, Math.min(B.x1 - 2, h.point.x));
      const gz = Math.max(B.z0 + 2, Math.min(B.z1 - 2, h.point.z));
      retarget(gx, gz);
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
      simT += dt;

      const pose = poseAt(path, simT);
      craft.position.set(pose.x, 0, pose.z);
      craft.rotation.y = -pose.heading;
      const speed = Math.hypot(pose.vx, pose.vz);
      driftArrow.position.set(pose.x, 0.2, pose.z);
      if (speed > 0.2) {
        driftArrow.visible = true;
        driftArrow.setDirection(
          new THREE.Vector3(pose.vx / speed, 0, pose.vz / speed),
        );
        driftArrow.setLength(1 + speed * 0.35, 0.8, 0.5);
      } else {
        driftArrow.visible = false;
      }
      for (let i = 0; i < floes.length; i++) {
        const p = floes[i]!.predict(simT);
        if (p) floeMeshes[i]!.position.set(p.x, 0.6, p.z);
      }
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
    };
  }, []);

  return (
    <main
      style={{
        color: '#cdd3de',
        fontFamily: 'ui-monospace, monospace',
        padding: 'clamp(12px, 4vw, 24px)',
        maxWidth: 980,
        margin: '0 auto',
      }}
    >
      <a href="/" style={{ color: '#7fd6ff' }}>← demos</a>
      <h1 style={{ fontSize: 18 }}>Hovercraft lagoon</h1>
      <p style={{ opacity: 0.75, marginTop: 0 }}>
        A fifth motion body defined entirely OUTSIDE kinocat, per{' '}
        <code>docs/adding-a-domain.md</code>: inertial, thrust-vectored,
        drifting — the hull&apos;s facing (fin) is decoupled from the motion
        (orange arrow), so it slides through corners sideways. Plans are
        space-time: the patrolling floes&apos; futures are part of the search.
        Tap the ice to retarget mid-drift — the replan starts from the
        current velocity, so momentum carries into the new route. Path
        color = speed (cyan glide → amber full thrust).
      </p>
      <div ref={mountRef} style={{ width: '100%', borderRadius: 8, overflow: 'hidden' }} />
      <p style={{ opacity: 0.8 }}>{info}</p>
    </main>
  );
}
