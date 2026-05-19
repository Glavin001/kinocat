'use client';

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { AircraftState } from 'kinocat/agent';
import {
  buildWaypointCourse,
  buildCanyon,
  buildRestrictedAirspace,
  buildGauntlet,
  buildKnifeEdge,
  planInteractive,
  densifyPath,
  INTERACTIVE_BOXES,
  AIRCRAFT_AGENT,
  AIRCRAFT_BOUNDS as B,
  AIR_PALETTE as C,
  type AircraftScene,
} from '../lib/aircraft-scenarios';

type Mode =
  | 'waypoint'
  | 'canyon'
  | 'restricted'
  | 'gauntlet'
  | 'knife-edge'
  | 'interactive';

const CRUISE_Y = 32;

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}
function lerpAngle(a: number, b: number, t: number) {
  let d = ((b - a + Math.PI) % (2 * Math.PI)) - Math.PI;
  if (d < -Math.PI) d += 2 * Math.PI;
  return a + d * t;
}

/** Sample a planned path at playback time `tp` (seconds). */
function sampleAt(path: AircraftState[], tp: number) {
  if (path.length === 1) {
    const p = path[0]!;
    return { x: p.x, y: p.y, z: p.z, heading: p.heading, pitch: p.pitch, roll: p.roll };
  }
  let i = 0;
  while (i < path.length - 2 && path[i + 1]!.t < tp) i++;
  const a = path[i]!;
  const b = path[i + 1]!;
  const span = Math.max(b.t - a.t, 1e-6);
  const u = Math.min(Math.max((tp - a.t) / span, 0), 1);
  return {
    x: lerp(a.x, b.x, u),
    y: lerp(a.y, b.y, u),
    z: lerp(a.z, b.z, u),
    heading: lerpAngle(a.heading, b.heading, u),
    pitch: lerp(a.pitch, b.pitch, u),
    roll: lerpAngle(a.roll, b.roll, u),
  };
}

export default function Plane() {
  const mountRef = useRef<HTMLDivElement>(null);
  const [mode, setMode] = useState<Mode>('waypoint');
  const [info, setInfo] = useState('drag to orbit');
  const modeRef = useRef<Mode>('waypoint');
  const loadRef = useRef<((m: Mode) => void) | null>(null);

  useEffect(() => {
    modeRef.current = mode;
  }, [mode]);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    const W = mount.clientWidth;
    const viewH = () =>
      Math.round(Math.min(520, Math.max(300, window.innerHeight * 0.6)));
    let Hpx = viewH();

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(C.bg);
    scene.fog = new THREE.Fog(C.bg, 180, 360);
    const camera = new THREE.PerspectiveCamera(58, W / Hpx, 0.1, 800);
    camera.position.set(40, 90, 150);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(W, Hpx);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set((B.x0 + B.x1) / 2, 25, 0);
    controls.update();

    scene.add(new THREE.AmbientLight(0xffffff, 0.65));
    const sun = new THREE.DirectionalLight(0xffffff, 1);
    sun.position.set(60, 120, 40);
    scene.add(sun);

    const ground = new THREE.Mesh(
      new THREE.BoxGeometry(B.x1 - B.x0 + 40, 1, B.z1 - B.z0 + 40),
      new THREE.MeshStandardMaterial({ color: C.ground }),
    );
    ground.position.set((B.x0 + B.x1) / 2, -0.5, 0);
    scene.add(ground);
    const grid = new THREE.GridHelper(
      Math.max(B.x1 - B.x0, B.z1 - B.z0) + 40,
      28,
      0x223044,
      0x182334,
    );
    grid.position.set((B.x0 + B.x1) / 2, 0.02, 0);
    scene.add(grid);

    const mkSphere = (color: number, r = 2) =>
      new THREE.Mesh(
        new THREE.SphereGeometry(r, 18, 18),
        new THREE.MeshStandardMaterial({ color }),
      );
    const startMarker = mkSphere(parseInt(C.start.slice(1), 16));
    const goalMarker = mkSphere(parseInt(C.goal.slice(1), 16));
    scene.add(startMarker, goalMarker);

    // ---- the aircraft (nose toward -z so THREE.lookAt aims it) ----
    // Sized to the collision sphere (radius ≈ 1.6) so the visual footprint
    // matches what the planner actually checks — oversized wings would clip
    // walls the collision sphere clears, looking like a planner bug.
    const plane = new THREE.Group();
    const planeMat = new THREE.MeshStandardMaterial({ color: C.plane });
    const fuse = new THREE.Mesh(new THREE.CylinderGeometry(0.28, 0.28, 2.6, 12), planeMat);
    fuse.rotation.x = Math.PI / 2;
    const nose = new THREE.Mesh(new THREE.ConeGeometry(0.28, 1.0, 12), planeMat);
    nose.rotation.x = -Math.PI / 2;
    nose.position.z = -1.8;
    const wing = new THREE.Mesh(new THREE.BoxGeometry(3.0, 0.10, 0.9), planeMat);
    const tail = new THREE.Mesh(new THREE.BoxGeometry(1.2, 0.09, 0.5), planeMat);
    tail.position.z = 1.15;
    const fin = new THREE.Mesh(new THREE.BoxGeometry(0.10, 0.65, 0.5), planeMat);
    fin.position.set(0, 0.35, 1.15);
    plane.add(fuse, nose, wing, tail, fin);
    scene.add(plane);

    // ---- dynamic scene content (rebuilt per scenario) ----
    const dyn = new THREE.Group();
    scene.add(dyn);
    let scn: AircraftScene | null = null;
    let playPath: AircraftState[] = [];
    let goal: AircraftState = {
      x: 150, y: CRUISE_Y, z: 0, heading: 0, pitch: 0, roll: 0, speed: AIRCRAFT_AGENT.maxSpeed, t: 0,
    };
    let playT = 0;

    const clearDyn = () => {
      for (const c of [...dyn.children]) {
        dyn.remove(c);
        const m = c as THREE.Mesh;
        if (m.geometry) m.geometry.dispose();
      }
    };

    const drawScene = (s: AircraftScene) => {
      clearDyn();
      // walls
      const wallMat = new THREE.MeshStandardMaterial({ color: C.wall });
      for (const b of s.boxes) {
        const g = new THREE.BoxGeometry(
          b.max[0] - b.min[0],
          b.max[1] - b.min[1],
          b.max[2] - b.min[2],
        );
        const m = new THREE.Mesh(g, wallMat);
        m.position.set(
          (b.min[0] + b.max[0]) / 2,
          (b.min[1] + b.max[1]) / 2,
          (b.min[2] + b.max[2]) / 2,
        );
        dyn.add(m);
      }
      // gates
      for (const gt of s.gates) {
        const ring = new THREE.Mesh(
          new THREE.TorusGeometry(7, 0.5, 10, 28),
          new THREE.MeshStandardMaterial({ color: C.gate }),
        );
        ring.position.set(gt.x, gt.y, gt.z);
        ring.rotation.y = Math.PI / 2;
        dyn.add(ring);
      }
      // moving no-fly zone
      if (s.zoneRadius != null) {
        const zone = new THREE.Mesh(
          new THREE.SphereGeometry(s.zoneRadius, 24, 24),
          new THREE.MeshStandardMaterial({
            color: C.zone,
            transparent: true,
            opacity: 0.28,
          }),
        );
        zone.name = 'zone';
        dyn.add(zone);
      }
      // planned path
      if (s.path.length > 1) {
        const line = new THREE.Line(
          new THREE.BufferGeometry().setFromPoints(
            s.path.map((p) => new THREE.Vector3(p.x, p.y, p.z)),
          ),
          new THREE.LineBasicMaterial({ color: C.path }),
        );
        dyn.add(line);
      }
      startMarker.position.set(s.start.x, s.start.y, s.start.z);
      goalMarker.position.set(s.goal.x, s.goal.y, s.goal.z);
      playPath = s.found ? densifyPath(s.path, 12) : s.path;
      playT = 0;
      setInfo(s.found ? s.info : `${s.info} — try another scenario`);
    };

    const replanInteractive = () => {
      const start: AircraftState = {
        x: 8, y: CRUISE_Y, z: 0, heading: 0, pitch: 0, roll: 0,
        speed: AIRCRAFT_AGENT.maxSpeed, t: 0,
      };
      const r = planInteractive(INTERACTIVE_BOXES, start, goal);
      scn = {
        kind: 'canyon',
        path: r.found ? r.path : [start],
        found: r.found,
        duration: r.found ? r.path[r.path.length - 1]!.t : 0,
        start,
        goal,
        gates: [],
        boxes: INTERACTIVE_BOXES,
        info: r.found
          ? `replanned · ${r.path.length} states · cost ${r.cost.toFixed(1)}`
          : 'no plan — tap elsewhere',
      };
      drawScene(scn);
    };

    const load = (m: Mode) => {
      if (m === 'waypoint') scn = buildWaypointCourse();
      else if (m === 'canyon') scn = buildCanyon();
      else if (m === 'restricted') scn = buildRestrictedAirspace();
      else if (m === 'gauntlet') scn = buildGauntlet();
      else if (m === 'knife-edge') scn = buildKnifeEdge();
      else {
        replanInteractive();
        return;
      }
      drawScene(scn);
    };
    loadRef.current = load;
    load('waypoint');

    // ---- tap-to-retarget (interactive mode) ----
    const ray = new THREE.Raycaster();
    const ndc = new THREE.Vector2();
    let downX = 0,
      downY = 0,
      moved = false;
    const onDown = (e: PointerEvent) => {
      downX = e.clientX;
      downY = e.clientY;
      moved = false;
    };
    const onMove = (e: PointerEvent) => {
      if (Math.hypot(e.clientX - downX, e.clientY - downY) > 4) moved = true;
    };
    const onUp = (e: PointerEvent) => {
      if (moved || modeRef.current !== 'interactive') return;
      const rect = renderer.domElement.getBoundingClientRect();
      ndc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      ndc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      ray.setFromCamera(ndc, camera);
      const hit = ray.intersectObject(ground)[0];
      if (!hit) return;
      goal = {
        x: Math.max(B.x0 + 6, Math.min(B.x1 - 6, hit.point.x)),
        y: CRUISE_Y,
        z: Math.max(B.z0 + 6, Math.min(B.z1 - 6, hit.point.z)),
        heading: 0,
        pitch: 0,
        roll: 0,
        speed: AIRCRAFT_AGENT.maxSpeed,
        t: 0,
      };
      replanInteractive();
    };
    renderer.domElement.addEventListener('pointerdown', onDown);
    renderer.domElement.addEventListener('pointermove', onMove);
    renderer.domElement.addEventListener('pointerup', onUp);

    const fwd = new THREE.Vector3();
    let frame = 0;
    let lastMs = performance.now();
    const animate = () => {
      frame = requestAnimationFrame(animate);
      const now = performance.now();
      const dt = Math.min((now - lastMs) / 1000, 0.05);
      lastMs = now;
      if (scn && playPath.length > 0) {
        const dur = Math.max(scn.duration, 0.001);
        playT = playPath.length > 1 ? (playT + dt) % (dur + 1.2) : 0;
        const p = sampleAt(playPath, playT);
        plane.position.set(p.x, p.y, p.z);
        const cp = Math.cos(p.pitch);
        fwd.set(
          p.x + cp * Math.cos(p.heading),
          p.y + Math.sin(p.pitch),
          p.z + cp * Math.sin(p.heading),
        );
        plane.lookAt(fwd);
        plane.rotateZ(p.roll);
        const zoneMesh = dyn.getObjectByName('zone');
        if (zoneMesh && scn.zoneAt) {
          const c = scn.zoneAt(playT);
          if (c) {
            zoneMesh.position.set(c.x, c.y, c.z);
            zoneMesh.visible = true;
          } else {
            zoneMesh.visible = false;
          }
        }
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
      loadRef.current = null;
    };
  }, []);

  const choose = (m: Mode) => {
    setMode(m);
    loadRef.current?.(m);
  };
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
        maxWidth: 960,
        margin: '0 auto',
      }}
    >
      <a href="/" style={{ color: '#7fd6ff' }}>
        ← demos
      </a>
      <h1 style={{ fontSize: 18 }}>3D flight planner</h1>
      <p style={{ opacity: 0.75, marginTop: 0 }}>
        A fixed-wing aircraft planned by the same IGHA* core as every other
        demo — but over a genuinely 3D state (x, y, z, heading, pitch, speed,
        t). Altitude is <em>searched</em>, not derived: the plane climbs over
        ridges, weaves a canyon, threads gates, routes around a moving no-fly
        zone, and knife-edges through a slot too narrow for its wingspan by
        banking 90° (roll is a searched dimension). Drag to orbit; in
        interactive mode, tap the ground to retarget and watch it replan.
      </p>
      <div style={{ display: 'flex', gap: 8, margin: '8px 0', flexWrap: 'wrap' }}>
        <button onClick={() => choose('waypoint')} style={btn(mode === 'waypoint')}>
          waypoint course
        </button>
        <button onClick={() => choose('canyon')} style={btn(mode === 'canyon')}>
          canyon slalom
        </button>
        <button
          onClick={() => choose('restricted')}
          style={btn(mode === 'restricted')}
        >
          restricted airspace
        </button>
        <button
          onClick={() => choose('gauntlet')}
          style={btn(mode === 'gauntlet')}
        >
          grand tour
        </button>
        <button
          onClick={() => choose('knife-edge')}
          style={btn(mode === 'knife-edge')}
        >
          knife edge
        </button>
        <button
          onClick={() => choose('interactive')}
          style={btn(mode === 'interactive')}
        >
          interactive
        </button>
      </div>
      <div
        ref={mountRef}
        style={{ width: '100%', borderRadius: 8, overflow: 'hidden' }}
      />
      <p style={{ opacity: 0.8 }}>{info}</p>
    </main>
  );
}
