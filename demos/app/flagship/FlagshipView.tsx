'use client';

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { planPoseAt } from 'kinocat/execute';
import {
  buildFlagshipWorld,
  solveFlagship,
  type FlagshipWorld,
  type FlagshipResult,
} from '../lib/scenarios';

// Mirror of flagshipTerrain's height function so agents/markers sit on the
// undulating surface (the planner ignores Y; this is purely visual).
const terrainY = (x: number, z: number) =>
  0.5 * Math.sin(x * 0.07) + 0.4 * Math.cos(z * 0.09);

const AGENT_COLORS = [
  0x7fd6ff, 0x55ff88, 0xffcc33, 0xff6688, 0x9b6cff, 0x4dd0e1, 0xff9f43,
  0xa3e635, 0xf472b6, 0x60a5fa, 0xfbbf24, 0x34d399,
];

type Mode = 'orbit' | 'retarget' | 'hazard';
type Hazard = { x: number; z: number; r: number };

export default function FlagshipView() {
  const mountRef = useRef<HTMLDivElement>(null);
  const [agents, setAgents] = useState(10);
  const [clearance, setClearance] = useState(true);
  const [timeBP, setTimeBP] = useState(true);
  const [mode, setMode] = useState<Mode>('orbit');
  const [selected, setSelected] = useState(0);
  const [paused, setPaused] = useState(false);
  const [follow, setFollow] = useState(false);
  const [hazards, setHazards] = useState<Hazard[]>([]);
  const [overrides, setOverrides] = useState<
    Record<string, { x: number; z: number }>
  >({});
  const [status, setStatus] = useState('Building the procedural navmesh…');
  const [hud, setHud] = useState('');

  // Live refs so the rAF loop / click handlers see fresh state without
  // tearing down the scene.
  const modeRef = useRef(mode);
  const selectedRef = useRef(selected);
  const pausedRef = useRef(paused);
  const followRef = useRef(follow);
  const scrubRef = useRef<number | null>(null);
  modeRef.current = mode;
  selectedRef.current = selected;
  pausedRef.current = paused;
  followRef.current = follow;

  // The expensive navmesh is built ONCE and reused across re-solves.
  const fwRef = useRef<FlagshipWorld | null>(null);
  const resRef = useRef<FlagshipResult | null>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    let disposed = false;
    let cleanup: (() => void) | null = null;

    const kickoff = setTimeout(() => {
      if (disposed) return;
      let fw: FlagshipWorld;
      try {
        fw = buildFlagshipWorld(true);
      } catch {
        setStatus(
          'navcat navmesh generation is unavailable in this environment — see the headless scenario test for the asserted behaviour.',
        );
        return;
      }
      fwRef.current = fw;

      const W = mount.clientWidth;
      const viewH = () =>
        Math.round(Math.min(560, Math.max(320, window.innerHeight * 0.64)));
      let Hpx = viewH();

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x0b0b0f);
      const camera = new THREE.PerspectiveCamera(55, W / Hpx, 0.1, 800);
      const b = fw.bounds;
      const cx = (b.x0 + b.x1) / 2;
      const cz = (b.z0 + b.z1) / 2;
      camera.position.set(cx, 78, b.z1 + 52);
      const renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setSize(W, Hpx);
      renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
      mount.appendChild(renderer.domElement);
      const controls = new OrbitControls(camera, renderer.domElement);
      controls.target.set(cx, 0, cz);
      controls.update();

      scene.add(new THREE.AmbientLight(0xffffff, 0.65));
      const sun = new THREE.DirectionalLight(0xffffff, 1);
      sun.position.set(cx, 70, cz - 30);
      scene.add(sun);

      // Terrain (the navcat-meshed positions/indices; holes = obstacles).
      const tgeo = new THREE.BufferGeometry();
      tgeo.setAttribute(
        'position',
        new THREE.Float32BufferAttribute(fw.positions, 3),
      );
      tgeo.setIndex(fw.indices);
      tgeo.computeVertexNormals();
      const terrainMesh = new THREE.Mesh(
        tgeo,
        new THREE.MeshStandardMaterial({
          color: 0x1d2230,
          flatShading: true,
          side: THREE.DoubleSide,
        }),
      );
      scene.add(terrainMesh);

      // Affordance markers (rebuilt once — they never move).
      const ring = (x: number, z: number, color: number): THREE.Mesh => {
        const m = new THREE.Mesh(
          new THREE.TorusGeometry(2.4, 0.35, 10, 28),
          new THREE.MeshStandardMaterial({
            color,
            emissive: color,
            emissiveIntensity: 0.4,
          }),
        );
        m.rotation.x = Math.PI / 2;
        m.position.set(x, terrainY(x, z) + 0.4, z);
        return m;
      };
      const link = (
        a: { x: number; z: number },
        c: { x: number; z: number },
        color: number,
      ) => {
        const l = new THREE.Line(
          new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(a.x, terrainY(a.x, a.z) + 0.5, a.z),
            new THREE.Vector3(c.x, terrainY(c.x, c.z) + 0.5, c.z),
          ]),
          new THREE.LineDashedMaterial({ color, dashSize: 1.5, gapSize: 1 }),
        );
        l.computeLineDistances();
        return l;
      };
      for (const s of fw.shortcuts) {
        scene.add(ring(s.launch.x, s.launch.z, 0x55ff88));
        scene.add(link(s.launch, s.land, 0x55ff88));
      }
      for (const m of fw.misdirects) {
        scene.add(ring(m.launch.x, m.launch.z, 0xff5566));
        scene.add(link(m.launch, m.land, 0xff5566));
      }
      for (const j of fw.jumps) {
        scene.add(ring(j.launch.x, j.launch.z, 0x33ddff));
        scene.add(link(j.launch, j.land, 0x33ddff));
      }

      // Per-agent group + path line + goal marker, created lazily and reused.
      const groups: THREE.Group[] = [];
      const pathLines: (THREE.Line | null)[] = [];
      const goalMarks: THREE.Mesh[] = [];
      const hazardGroup = new THREE.Group();
      scene.add(hazardGroup);
      let maxT = 0.1;

      const buildAgentVisuals = (res: FlagshipResult) => {
        maxT = 0.1;
        res.agents.forEach((ag, i) => {
          const color = AGENT_COLORS[i % AGENT_COLORS.length]!;
          maxT = Math.max(maxT, ag.path[ag.path.length - 1]?.t ?? 0);
          if (pathLines[i]) {
            scene.remove(pathLines[i]!);
            pathLines[i]!.geometry.dispose();
          }
          if (ag.path.length > 1) {
            const pl = new THREE.Line(
              new THREE.BufferGeometry().setFromPoints(
                ag.path.map(
                  (p) =>
                    new THREE.Vector3(p.x, terrainY(p.x, p.z) + 0.3, p.z),
                ),
              ),
              new THREE.LineBasicMaterial({
                color,
                transparent: true,
                opacity: 0.5,
              }),
            );
            scene.add(pl);
            pathLines[i] = pl;
          } else {
            pathLines[i] = null;
          }
          if (!groups[i]) {
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
            groups[i] = g;
            const gm = new THREE.Mesh(
              new THREE.ConeGeometry(0.7, 1.8, 4),
              new THREE.MeshStandardMaterial({
                color,
                emissive: color,
                emissiveIntensity: 0.5,
              }),
            );
            scene.add(gm);
            goalMarks[i] = gm;
          }
          const gm = goalMarks[i]!;
          gm.position.set(
            ag.goal.x,
            terrainY(ag.goal.x, ag.goal.z) + 1.2,
            ag.goal.z,
          );
        });
        // hide unused (lower agent count) groups
        groups.forEach((g, i) => {
          const on = i < res.agents.length;
          g.visible = on;
          if (goalMarks[i]) goalMarks[i]!.visible = on;
          if (pathLines[i]) pathLines[i]!.visible = on;
        });
      };

      const refreshHazards = (hz: Hazard[]) => {
        while (hazardGroup.children.length) {
          const c = hazardGroup.children[0]!;
          hazardGroup.remove(c);
          (c as THREE.Mesh).geometry?.dispose();
        }
        for (const h of hz) {
          const dome = new THREE.Mesh(
            new THREE.SphereGeometry(h.r, 16, 12, 0, Math.PI * 2, 0, Math.PI / 2),
            new THREE.MeshStandardMaterial({
              color: 0xff3344,
              transparent: true,
              opacity: 0.35,
              emissive: 0xff3344,
              emissiveIntensity: 0.3,
            }),
          );
          dome.position.set(h.x, terrainY(h.x, h.z), h.z);
          hazardGroup.add(dome);
        }
      };

      let solving = false;
      const resolve = (
        nAgents: number,
        cb: boolean,
        tb: boolean,
        hz: Hazard[],
        ov: Record<string, { x: number; z: number }>,
      ) => {
        if (!fwRef.current || solving) return;
        solving = true;
        setStatus(`Re-solving ${nAgents} agents on the main thread…`);
        // yield a frame so the status paints before the blocking solve
        setTimeout(() => {
          if (disposed) return;
          const t0 = performance.now();
          const res = solveFlagship(fwRef.current!, {
            agents: nAgents,
            rounds: 2,
            crossTraffic: true,
            clearanceBroadphase: cb,
            timeBroadphase: tb,
            hazards: hz,
            goalOverrides: ov,
          });
          const ms = performance.now() - t0;
          resRef.current = res;
          buildAgentVisuals(res);
          refreshHazards(hz);
          const boostN = res.agents.filter((a) => a.usedShortcut).length;
          const jumpN = res.agents.filter((a) => a.usedJump).length;
          const misN = res.agents.filter((a) => a.usedMisdirect).length;
          setStatus(
            `${res.agents.length} agents · ${res.reached} reached · ` +
              `${boostN} boost · ${jumpN} jump · ${misN} misdirect · ` +
              `${hz.length} hazard(s) · re-solve ${ms.toFixed(0)} ms ` +
              `(clearance ${cb ? 'on' : 'off'}, time-bp ${tb ? 'on' : 'off'})`,
          );
          solving = false;
        }, 30);
      };

      // initial solve
      resolve(agents, clearance, timeBP, hazards, overrides);

      // ---- picking ---------------------------------------------------------
      const raycaster = new THREE.Raycaster();
      const ndc = new THREE.Vector2();
      const pickGround = (ev: PointerEvent): { x: number; z: number } | null => {
        const rect = renderer.domElement.getBoundingClientRect();
        ndc.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
        ndc.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
        raycaster.setFromCamera(ndc, camera);
        const hit = raycaster.intersectObject(terrainMesh, false)[0];
        return hit ? { x: hit.point.x, z: hit.point.z } : null;
      };
      const pickAgent = (ev: PointerEvent): number | null => {
        const rect = renderer.domElement.getBoundingClientRect();
        ndc.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
        ndc.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
        raycaster.setFromCamera(ndc, camera);
        const vis = groups.filter((g) => g.visible);
        const hit = raycaster.intersectObjects(vis, true)[0];
        if (!hit) return null;
        let o: THREE.Object3D | null = hit.object;
        while (o && !groups.includes(o as THREE.Group)) o = o.parent;
        return o ? groups.indexOf(o as THREE.Group) : null;
      };
      const onPointerDown = (ev: PointerEvent) => {
        const m = modeRef.current;
        const aIdx = pickAgent(ev);
        if (aIdx !== null && m === 'orbit') {
          setSelected(aIdx);
          return;
        }
        const p = pickGround(ev);
        if (!p) return;
        if (m === 'retarget') {
          const res = resRef.current;
          const id = res?.agents[selectedRef.current]?.id ?? `V${selectedRef.current}`;
          setOverrides((prev) => {
            const nxt = { ...prev, [id]: { x: p.x, z: p.z } };
            resolve(agents, clearance, timeBP, hazards, nxt);
            return nxt;
          });
        } else if (m === 'hazard') {
          setHazards((prev) => {
            const nxt = [...prev, { x: p.x, z: p.z, r: 4 }];
            resolve(agents, clearance, timeBP, nxt, overrides);
            return nxt;
          });
        }
      };
      renderer.domElement.addEventListener('pointerdown', onPointerDown);

      // ---- animation -------------------------------------------------------
      let frame = 0;
      let last = performance.now();
      let t = 0;
      const animate = () => {
        frame = requestAnimationFrame(animate);
        const now = performance.now();
        const res = resRef.current;
        if (res) {
          if (scrubRef.current !== null) {
            t = scrubRef.current * maxT;
          } else if (!pausedRef.current) {
            t += (now - last) / 1000;
            if (t > maxT) t = 0;
          }
          res.agents.forEach((ag, i) => {
            const pose = planPoseAt(ag.path, t);
            const grp = groups[i];
            if (pose && grp) {
              grp.position.set(pose.x, terrainY(pose.x, pose.z), pose.z);
              grp.rotation.y = -pose.heading;
              const sel = i === selectedRef.current;
              const body = grp.children[0] as THREE.Mesh;
              (body.material as THREE.MeshStandardMaterial).emissive.setHex(
                sel ? 0xffffff : 0x000000,
              );
              (
                body.material as THREE.MeshStandardMaterial
              ).emissiveIntensity = sel ? 0.45 : 0;
            }
          });
          if (followRef.current) {
            const pose = planPoseAt(
              res.agents[selectedRef.current]?.path ?? [],
              t,
            );
            if (pose) {
              controls.target.lerp(
                new THREE.Vector3(pose.x, terrainY(pose.x, pose.z), pose.z),
                0.1,
              );
            }
          }
        }
        last = now;
        controls.update();
        renderer.render(scene, camera);
        if (res) {
          const a = res.agents[selectedRef.current];
          if (a) {
            setHud(
              `▶ ${a.id} (${selectedRef.current})  ` +
                `${a.start.x < a.goal.x ? '→ eastbound' : '← westbound'} · ` +
                `${a.found ? 'planned' : 'no plan'} · ` +
                `${a.usedShortcut ? 'boost ' : ''}` +
                `${a.usedJump ? 'jump ' : ''}` +
                `${a.usedMisdirect ? 'MISDIRECT ' : ''}` +
                `path ${a.path.length}`,
            );
          }
        }
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

      // expose resolve so the React control effect can trigger re-solves
      (mount as unknown as { __resolve?: typeof resolve }).__resolve = resolve;

      cleanup = () => {
        cancelAnimationFrame(frame);
        window.removeEventListener('resize', onResize);
        renderer.domElement.removeEventListener('pointerdown', onPointerDown);
        controls.dispose();
        renderer.dispose();
        if (renderer.domElement.parentNode === mount)
          mount.removeChild(renderer.domElement);
      };
    }, 60);

    return () => {
      disposed = true;
      clearTimeout(kickoff);
      if (cleanup) cleanup();
    };
    // Build the scene once; control changes drive re-solves via the effect
    // below (no full teardown).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Re-solve (no scene teardown) when the solve-affecting controls change.
  useEffect(() => {
    const mount = mountRef.current as unknown as {
      __resolve?: (
        n: number,
        cb: boolean,
        tb: boolean,
        hz: Hazard[],
        ov: Record<string, { x: number; z: number }>,
      ) => void;
    } | null;
    mount?.__resolve?.(agents, clearance, timeBP, hazards, overrides);
    // hazards/overrides changes already trigger resolve at the click site
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
        maxWidth: 1040,
        margin: '0 auto',
      }}
    >
      <a href="/" style={{ color: '#7fd6ff' }}>
        ← demos
      </a>
      <h1 style={{ fontSize: 18 }}>Interactive multi-agent flagship</h1>
      <p style={{ opacity: 0.75, marginTop: 0 }}>
        Opposing cross-traffic on a large procedural navcat terrain, planning
        on a staggered round-robin via the plan registry. Green = a genuine
        boost the planner adopts; cyan = a canyon jump; red = a{' '}
        <em>misdirect</em> the planner rejects on its own (honest cost, no
        special-case logic). <strong>Click a vehicle</strong> to select it,
        then use <strong>Retarget</strong> to click a new goal, or{' '}
        <strong>Hazard</strong> to drop danger zones — everyone replans live.
      </p>

      <div style={{ display: 'flex', gap: 8, margin: '8px 0', flexWrap: 'wrap' }}>
        {(['orbit', 'retarget', 'hazard'] as Mode[]).map((m) => (
          <button key={m} onClick={() => setMode(m)} style={btn(mode === m)}>
            {m === 'orbit'
              ? 'orbit / select'
              : m === 'retarget'
                ? 'retarget goal'
                : 'drop hazard'}
          </button>
        ))}
        <button onClick={() => setPaused((v) => !v)} style={btn(paused)}>
          {paused ? 'play' : 'pause'}
        </button>
        <button onClick={() => setFollow((v) => !v)} style={btn(follow)}>
          follow: {follow ? 'on' : 'off'}
        </button>
        <button
          onClick={() => {
            setHazards([]);
            setOverrides({});
            const mount = mountRef.current as unknown as {
              __resolve?: (
                n: number,
                cb: boolean,
                tb: boolean,
                hz: Hazard[],
                ov: Record<string, { x: number; z: number }>,
              ) => void;
            } | null;
            mount?.__resolve?.(agents, clearance, timeBP, [], {});
          }}
          style={btn(false)}
        >
          reset goals/hazards
        </button>
      </div>

      <div style={{ display: 'flex', gap: 8, margin: '8px 0', flexWrap: 'wrap' }}>
        {[8, 10, 12].map((n) => (
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
        <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          scrub
          <input
            type="range"
            min={0}
            max={1000}
            defaultValue={0}
            onMouseDown={() => {
              scrubRef.current = 0;
            }}
            onChange={(e) => {
              scrubRef.current = Number(e.target.value) / 1000;
            }}
            onMouseUp={() => {
              scrubRef.current = null;
            }}
          />
        </label>
      </div>

      <div
        ref={mountRef}
        style={{
          width: '100%',
          borderRadius: 8,
          overflow: 'hidden',
          cursor: mode === 'orbit' ? 'grab' : 'crosshair',
        }}
      />
      <p style={{ opacity: 0.9, margin: '8px 0 2px' }}>{hud}</p>
      <p style={{ opacity: 0.7, marginTop: 0 }}>{status}</p>
    </main>
  );
}
