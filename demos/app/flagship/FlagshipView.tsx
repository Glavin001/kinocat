'use client';

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { planPoseAt } from 'kinocat/execute';
import { buildFlagship, type FlagshipResult } from '../lib/scenarios';

// Mirror of flagshipTerrain's height function so agents/markers sit on the
// undulating surface (the planner ignores Y; this is purely visual).
const terrainY = (x: number, z: number) =>
  0.5 * Math.sin(x * 0.07) + 0.4 * Math.cos(z * 0.09);

const AGENT_COLORS = [
  0x7fd6ff, 0x55ff88, 0xffcc33, 0xff6688, 0x9b6cff, 0x4dd0e1, 0xff9f43,
  0xa3e635, 0xf472b6, 0x60a5fa, 0xfbbf24, 0x34d399,
];

export default function FlagshipView() {
  const mountRef = useRef<HTMLDivElement>(null);
  const [agents, setAgents] = useState(8);
  const [clearance, setClearance] = useState(true);
  const [timeBP, setTimeBP] = useState(true);
  const [status, setStatus] = useState('Planning the multi-agent solve…');

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    let disposed = false;
    setStatus(
      `Planning ${agents} agents on the main thread (staggered round-robin)…`,
    );

    // Yield a frame so the status paints before the (heavy, synchronous)
    // navcat multi-agent solve blocks the thread.
    const kickoff = setTimeout(() => {
      if (disposed) return;
      let res: FlagshipResult;
      const t0 = performance.now();
      try {
        res = buildFlagship({
          agents,
          rounds: 2,
          clearanceBroadphase: clearance,
          timeBroadphase: timeBP,
        });
      } catch {
        setStatus(
          'navcat navmesh generation is unavailable in this environment — see the headless scenario test for the asserted behaviour.',
        );
        return;
      }
      const buildMs = performance.now() - t0;
      if (disposed) return;

      const W = mount.clientWidth;
      const viewH = () =>
        Math.round(Math.min(520, Math.max(300, window.innerHeight * 0.62)));
      let Hpx = viewH();

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x0b0b0f);
      const camera = new THREE.PerspectiveCamera(55, W / Hpx, 0.1, 600);
      const b = res.bounds;
      const cx = (b.x0 + b.x1) / 2;
      const cz = (b.z0 + b.z1) / 2;
      camera.position.set(cx, 70, b.z1 + 46);
      const renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setSize(W, Hpx);
      renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
      mount.appendChild(renderer.domElement);
      const controls = new OrbitControls(camera, renderer.domElement);
      controls.target.set(cx, 0, cz);
      controls.update();

      scene.add(new THREE.AmbientLight(0xffffff, 0.65));
      const sun = new THREE.DirectionalLight(0xffffff, 1);
      sun.position.set(cx, 60, cz - 30);
      scene.add(sun);

      // Terrain (the navcat-meshed positions/indices; holes = obstacles).
      const geo = new THREE.BufferGeometry();
      geo.setAttribute(
        'position',
        new THREE.Float32BufferAttribute(res.positions, 3),
      );
      geo.setIndex(res.indices);
      geo.computeVertexNormals();
      scene.add(
        new THREE.Mesh(
          geo,
          new THREE.MeshStandardMaterial({
            color: 0x1d2230,
            flatShading: true,
            side: THREE.DoubleSide,
          }),
        ),
      );

      const ring = (
        x: number,
        z: number,
        color: number,
      ): THREE.Mesh => {
        const m = new THREE.Mesh(
          new THREE.TorusGeometry(2.4, 0.35, 10, 28),
          new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.4 }),
        );
        m.rotation.x = Math.PI / 2;
        m.position.set(x, terrainY(x, z) + 0.4, z);
        return m;
      };
      const link = (
        a: { x: number; z: number },
        c: { x: number; z: number },
        color: number,
      ) =>
        new THREE.Line(
          new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(a.x, terrainY(a.x, a.z) + 0.5, a.z),
            new THREE.Vector3(c.x, terrainY(c.x, c.z) + 0.5, c.z),
          ]),
          new THREE.LineDashedMaterial({ color, dashSize: 1.5, gapSize: 1 }),
        );
      for (const s of res.shortcuts) {
        scene.add(ring(s.launch.x, s.launch.z, 0x55ff88));
        const l = link(s.launch, s.land, 0x55ff88);
        l.computeLineDistances();
        scene.add(l);
      }
      for (const m of res.misdirects) {
        scene.add(ring(m.launch.x, m.launch.z, 0xff5566));
        const l = link(m.launch, m.land, 0xff5566);
        l.computeLineDistances();
        scene.add(l);
      }

      const groups: THREE.Group[] = [];
      let maxT = 0.1;
      res.agents.forEach((ag, i) => {
        const color = AGENT_COLORS[i % AGENT_COLORS.length]!;
        const path = ag.path;
        maxT = Math.max(maxT, path[path.length - 1]?.t ?? 0);
        if (path.length > 1) {
          scene.add(
            new THREE.Line(
              new THREE.BufferGeometry().setFromPoints(
                path.map(
                  (p) => new THREE.Vector3(p.x, terrainY(p.x, p.z) + 0.3, p.z),
                ),
              ),
              new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.5 }),
            ),
          );
        }
        const g = new THREE.Group();
        const body = new THREE.Mesh(
          new THREE.BoxGeometry(2.2, 0.9, 1.1),
          new THREE.MeshStandardMaterial({ color }),
        );
        body.position.y = 0.55;
        const nose = new THREE.Mesh(
          new THREE.ConeGeometry(0.3, 0.8, 10),
          new THREE.MeshStandardMaterial({ color: 0xffffff }),
        );
        nose.rotation.z = -Math.PI / 2;
        nose.position.set(1.4, 0.55, 0);
        g.add(body, nose);
        scene.add(g);
        groups.push(g);
      });

      const shortcutN = res.agents.filter((a) => a.usedShortcut).length;
      const misdirectN = res.agents.filter((a) => a.usedMisdirect).length;
      setStatus(
        `${res.agents.length} agents · ${res.reached} reached · ` +
          `${shortcutN} took the boost · ${misdirectN} took the misdirect · ` +
          `solve ${buildMs.toFixed(0)} ms (clearance ${clearance ? 'on' : 'off'}, ` +
          `time-broadphase ${timeBP ? 'on' : 'off'}) · drag to orbit`,
      );

      let frame = 0;
      let last = performance.now();
      let t = 0;
      const animate = () => {
        frame = requestAnimationFrame(animate);
        const now = performance.now();
        t += (now - last) / 1000;
        if (t > maxT) t = 0;
        last = now;
        res.agents.forEach((ag, i) => {
          const pose = planPoseAt(ag.path, t);
          const grp = groups[i];
          if (pose && grp) {
            grp.position.set(pose.x, terrainY(pose.x, pose.z), pose.z);
            grp.rotation.y = -pose.heading;
          }
        });
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

      cleanup = () => {
        cancelAnimationFrame(frame);
        window.removeEventListener('resize', onResize);
        controls.dispose();
        renderer.dispose();
        if (renderer.domElement.parentNode === mount)
          mount.removeChild(renderer.domElement);
      };
    }, 60);

    let cleanup: (() => void) | null = null;
    return () => {
      disposed = true;
      clearTimeout(kickoff);
      if (cleanup) cleanup();
    };
  }, [agents, clearance, timeBP]);

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
        maxWidth: 980,
        margin: '0 auto',
      }}
    >
      <a href="/" style={{ color: '#7fd6ff' }}>
        ← demos
      </a>
      <h1 style={{ fontSize: 18 }}>Real-time multi-agent flagship</h1>
      <p style={{ opacity: 0.75, marginTop: 0 }}>
        A large procedural navcat terrain (canyon + pillars) with NPC vehicles
        planning on a staggered round-robin via the plan registry. A green
        boost is a genuine shortcut the planner adopts; the red{' '}
        <em>misdirect</em> looks tempting but its honest cost makes the planner
        reject it on its own — no special-case logic. The CompactHeightfield
        clearance broadphase and moving-obstacle broadphase are toggleable;
        the solve time in the status reflects them.
      </p>
      <div style={{ display: 'flex', gap: 8, margin: '8px 0', flexWrap: 'wrap' }}>
        {[6, 8, 10].map((n) => (
          <button key={n} onClick={() => setAgents(n)} style={btn(agents === n)}>
            {n} agents
          </button>
        ))}
        <button onClick={() => setClearance((v) => !v)} style={btn(clearance)}>
          clearance broadphase: {clearance ? 'on' : 'off'}
        </button>
        <button onClick={() => setTimeBP((v) => !v)} style={btn(timeBP)}>
          time broadphase: {timeBP ? 'on' : 'off'}
        </button>
      </div>
      <div ref={mountRef} style={{ width: '100%', borderRadius: 8, overflow: 'hidden' }} />
      <p style={{ opacity: 0.8 }}>{status}</p>
    </main>
  );
}
