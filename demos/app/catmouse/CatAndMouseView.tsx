'use client';

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import {
  buildCatAndMouseWorld,
  initCatAndMouseState,
  stepCatAndMouse,
  predictMouseFromHistory,
  type CatMouseWorld,
  type CatMouseSimState,
} from '../lib/scenarios';

// Mirror of catMouseTerrain's height function so meshes sit on the ground.
const terrainY = (x: number, z: number) =>
  0.35 * Math.sin(x * 0.08) + 0.3 * Math.cos(z * 0.1);

// Warm "cat" palette + the chosen cat is brightened by emissive at render time.
const CAT_COLORS = [0xff8844, 0xffaa66, 0xff6622, 0xcc7733];
const MOUSE_COLOR = 0xe8e8f0;
const PREDICTED_COLOR = 0x66ddff;
const PLAN_COLOR = 0x7fd6ff;
const BOOST_COLOR = 0x55ff88;
const JUMP_COLOR = 0x33ddff;

type Follow = 'orbit' | 'mouse' | 'cat';

export default function CatAndMouseView() {
  const mountRef = useRef<HTMLDivElement>(null);

  // React state for the control panel. Mirrored into refs read by the rAF
  // loop so changing a slider does NOT tear down the scene.
  const [catCount, setCatCount] = useState(2);
  const [deadlineMs, setDeadlineMs] = useState(60);
  const [horizon, setHorizon] = useState(3);
  const [predictionOn, setPredictionOn] = useState(true);
  const [showPredicted, setShowPredicted] = useState(true);
  const [showPlans, setShowPlans] = useState(false);
  const [showAffordances, setShowAffordances] = useState(true);
  const [paused, setPaused] = useState(false);
  const [follow, setFollow] = useState<Follow>('orbit');
  const [status, setStatus] = useState('Building the procedural navmesh…');
  const [hud, setHud] = useState('');

  const catCountRef = useRef(catCount);
  const deadlineRef = useRef(deadlineMs);
  const horizonRef = useRef(horizon);
  const predictionOnRef = useRef(predictionOn);
  const showPredictedRef = useRef(showPredicted);
  const showPlansRef = useRef(showPlans);
  const showAffordancesRef = useRef(showAffordances);
  const pausedRef = useRef(paused);
  const followRef = useRef(follow);
  catCountRef.current = catCount;
  deadlineRef.current = deadlineMs;
  horizonRef.current = horizon;
  predictionOnRef.current = predictionOn;
  showPredictedRef.current = showPredicted;
  showPlansRef.current = showPlans;
  showAffordancesRef.current = showAffordances;
  pausedRef.current = paused;
  followRef.current = follow;

  // Expensive navmesh + affordances built ONCE.
  const worldRef = useRef<CatMouseWorld | null>(null);
  // Live sim state. Re-inited when cat count changes (cheap; navmesh stays).
  const simRef = useRef<CatMouseSimState | null>(null);
  // Bumped by the cat-count effect; the rAF loop notices and re-inits.
  const resetTickRef = useRef(0);

  // Trigger re-init from React effects via this exposed setter.
  const reinitFn = useRef<(() => void) | null>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    let disposed = false;
    let cleanup: (() => void) | null = null;

    const kickoff = setTimeout(() => {
      if (disposed) return;
      let world: CatMouseWorld;
      try {
        world = buildCatAndMouseWorld();
      } catch {
        setStatus(
          'navcat navmesh generation is unavailable in this environment — see the headless scenario test for the asserted behaviour.',
        );
        return;
      }
      worldRef.current = world;

      const W = mount.clientWidth;
      const viewH = () =>
        Math.round(Math.min(560, Math.max(320, window.innerHeight * 0.64)));
      let Hpx = viewH();

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x0b0b0f);
      const camera = new THREE.PerspectiveCamera(48, W / Hpx, 0.1, 800);
      const b = world.bounds;
      const cx = (b.x0 + b.x1) / 2;
      const cz = (b.z0 + b.z1) / 2;
      // Camera south of the arena looking north so world +Z reads as
      // "up" in screen space: cats spawn at the bottom (south-east), mouse
      // at the top (north-west) — natural chase geometry.
      camera.position.set(cx, 48, b.z0 - 18);
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

      // Terrain (navcat-meshed positions/indices; holes = the canyon).
      const tgeo = new THREE.BufferGeometry();
      tgeo.setAttribute(
        'position',
        new THREE.Float32BufferAttribute(world.positions, 3),
      );
      tgeo.setIndex(world.indices);
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

      // Affordance markers (rebuilt once — they never move). Stored on a
      // group so a toggle can hide the whole set.
      const affordanceGroup = new THREE.Group();
      scene.add(affordanceGroup);

      const ring = (x: number, z: number, color: number, r = 2.4) => {
        const m = new THREE.Mesh(
          new THREE.TorusGeometry(r, 0.3, 10, 28),
          new THREE.MeshStandardMaterial({
            color,
            emissive: color,
            emissiveIntensity: 0.45,
          }),
        );
        m.rotation.x = Math.PI / 2;
        m.position.set(x, terrainY(x, z) + 0.35, z);
        return m;
      };
      const link = (
        a: { x: number; z: number },
        c: { x: number; z: number },
        color: number,
        ballistic = false,
      ) => {
        const pts: THREE.Vector3[] = [];
        if (ballistic) {
          // Render a parabolic arc so the canyon jump is visually obvious.
          const N = 24;
          for (let i = 0; i <= N; i++) {
            const u = i / N;
            const x = a.x + (c.x - a.x) * u;
            const z = a.z + (c.z - a.z) * u;
            const y = terrainY(x, z) + 0.5 + 4 * u * (1 - u) * 3;
            pts.push(new THREE.Vector3(x, y, z));
          }
        } else {
          pts.push(
            new THREE.Vector3(a.x, terrainY(a.x, a.z) + 0.5, a.z),
            new THREE.Vector3(c.x, terrainY(c.x, c.z) + 0.5, c.z),
          );
        }
        const l = new THREE.Line(
          new THREE.BufferGeometry().setFromPoints(pts),
          new THREE.LineDashedMaterial({ color, dashSize: 1.2, gapSize: 0.8 }),
        );
        l.computeLineDistances();
        return l;
      };
      for (const bp of world.boosts) {
        affordanceGroup.add(ring(bp.pad.x, bp.pad.z, BOOST_COLOR, bp.r));
        affordanceGroup.add(ring(bp.exit.x, bp.exit.z, BOOST_COLOR, 1.5));
        affordanceGroup.add(link(bp.pad, bp.exit, BOOST_COLOR));
      }
      for (const jp of world.jumps) {
        affordanceGroup.add(ring(jp.launch.x, jp.launch.z, JUMP_COLOR, jp.r));
        affordanceGroup.add(ring(jp.land.x, jp.land.z, JUMP_COLOR, 1.5));
        affordanceGroup.add(link(jp.launch, jp.land, JUMP_COLOR, true));
      }

      // Cat meshes — a low body + a triangular "ear-nose" cone so heading is
      // legible at any zoom. Reused across cat-count changes (extras hidden).
      const catGroups: THREE.Group[] = [];
      const buildCatMesh = (idx: number): THREE.Group => {
        const color = CAT_COLORS[idx % CAT_COLORS.length]!;
        const g = new THREE.Group();
        const body = new THREE.Mesh(
          new THREE.BoxGeometry(2.0, 0.7, 1.0),
          new THREE.MeshStandardMaterial({ color }),
        );
        body.position.y = 0.45;
        const nose = new THREE.Mesh(
          new THREE.ConeGeometry(0.35, 0.9, 10),
          new THREE.MeshStandardMaterial({ color: 0xffffff }),
        );
        nose.rotation.z = -Math.PI / 2;
        nose.position.set(1.25, 0.45, 0);
        const earL = new THREE.Mesh(
          new THREE.ConeGeometry(0.25, 0.5, 8),
          new THREE.MeshStandardMaterial({ color }),
        );
        earL.position.set(0.6, 1.0, 0.35);
        const earR = earL.clone();
        earR.position.set(0.6, 1.0, -0.35);
        g.add(body, nose, earL, earR);
        return g;
      };

      // Mouse mesh — a small light sphere/capsule.
      const mouseGroup = new THREE.Group();
      const mouseBody = new THREE.Mesh(
        new THREE.SphereGeometry(0.45, 14, 12),
        new THREE.MeshStandardMaterial({
          color: MOUSE_COLOR,
          emissive: MOUSE_COLOR,
          emissiveIntensity: 0.15,
        }),
      );
      mouseBody.position.y = 0.45;
      const mouseTail = new THREE.Mesh(
        new THREE.ConeGeometry(0.12, 0.7, 6),
        new THREE.MeshStandardMaterial({ color: 0xc0c0d0 }),
      );
      mouseTail.rotation.z = Math.PI / 2;
      mouseTail.position.set(-0.55, 0.45, 0);
      mouseGroup.add(mouseBody, mouseTail);
      scene.add(mouseGroup);

      // Predicted-mouse ghost ribbon. Geometry is rewritten each frame.
      const predictedGeom = new THREE.BufferGeometry();
      const predictedLine = new THREE.Line(
        predictedGeom,
        new THREE.LineBasicMaterial({
          color: PREDICTED_COLOR,
          transparent: true,
          opacity: 0.85,
        }),
      );
      scene.add(predictedLine);

      // One faint polyline per cat showing its committed plan.
      const planLines: (THREE.Line | null)[] = [];

      // A red flash overlay on capture — a wide transparent dome that pops.
      const captureFlash = new THREE.Mesh(
        new THREE.SphereGeometry(60, 16, 8),
        new THREE.MeshBasicMaterial({
          color: 0xff3344,
          transparent: true,
          opacity: 0,
          side: THREE.BackSide,
        }),
      );
      captureFlash.position.set(cx, 0, cz);
      scene.add(captureFlash);

      const ensureCatVisuals = (n: number) => {
        for (let i = catGroups.length; i < n; i++) {
          const g = buildCatMesh(i);
          scene.add(g);
          catGroups.push(g);
        }
        for (let i = 0; i < catGroups.length; i++) {
          catGroups[i]!.visible = i < n;
        }
        // Drop plan lines beyond n.
        for (let i = n; i < planLines.length; i++) {
          const pl = planLines[i];
          if (pl) {
            scene.remove(pl);
            pl.geometry.dispose();
            planLines[i] = null;
          }
        }
      };

      const initSim = () => {
        const w = worldRef.current;
        if (!w) return;
        const n = catCountRef.current;
        simRef.current = initCatAndMouseState(w, n, (Date.now() & 0xffff) | 1);
        ensureCatVisuals(n);
        // Reset every cat's plan polyline.
        for (let i = 0; i < planLines.length; i++) {
          const pl = planLines[i];
          if (pl) {
            scene.remove(pl);
            pl.geometry.dispose();
            planLines[i] = null;
          }
        }
        setStatus(`Pursuit live · ${n} cats vs 1 mouse`);
      };
      reinitFn.current = initSim;
      initSim();

      // ---- animation -----------------------------------------------------
      let frame = 0;
      let last = performance.now();
      let lastResetTick = resetTickRef.current;
      const animate = () => {
        frame = requestAnimationFrame(animate);
        const now = performance.now();
        if (lastResetTick !== resetTickRef.current) {
          lastResetTick = resetTickRef.current;
          initSim();
          last = now;
        }
        const w = worldRef.current;
        const sim = simRef.current;
        if (!w || !sim) return;

        if (!pausedRef.current) {
          // Clamp dt to keep the integration stable when the tab was hidden.
          const dt = Math.min(0.06, Math.max(0.001, (now - last) / 1000));
          stepCatAndMouse(w, sim, dt, {
            catCount: catCountRef.current,
            deadlineMs: deadlineRef.current,
            predictionHorizon: horizonRef.current,
            predictionEnabled: predictionOnRef.current,
            nowMs: now,
          });
        }
        last = now;

        // Mouse mesh.
        const m = sim.mouse.state;
        mouseGroup.position.set(m.x, terrainY(m.x, m.z), m.z);
        mouseGroup.rotation.y = -m.heading;

        // Cat meshes.
        for (let i = 0; i < sim.cats.length; i++) {
          const cat = sim.cats[i]!;
          const g = catGroups[i];
          if (!g) continue;
          g.position.set(cat.state.x, terrainY(cat.state.x, cat.state.z), cat.state.z);
          g.rotation.y = -cat.state.heading;
        }

        // Predicted mouse ribbon.
        if (showPredictedRef.current) {
          const pred = predictMouseFromHistory(
            sim.mouse.obsHistory,
            horizonRef.current,
          );
          const pts: THREE.Vector3[] = [];
          const step = 0.18;
          for (let t = 0; t <= horizonRef.current; t += step) {
            const p = pred(sim.simTime + t);
            if (!p) break;
            pts.push(new THREE.Vector3(p.x, terrainY(p.x, p.z) + 0.6, p.z));
          }
          predictedGeom.setFromPoints(pts);
          predictedGeom.attributes.position?.needsUpdate &&
            (predictedGeom.attributes.position.needsUpdate = true);
          predictedLine.visible = pts.length > 1;
        } else {
          predictedLine.visible = false;
        }

        // Cat plan polylines (default off — gets cluttered with 4 cats).
        for (let i = 0; i < sim.cats.length; i++) {
          const cat = sim.cats[i]!;
          const want = showPlansRef.current && cat.plan.length > 1;
          if (!want) {
            const old = planLines[i];
            if (old) {
              scene.remove(old);
              old.geometry.dispose();
              planLines[i] = null;
            }
            continue;
          }
          const old = planLines[i];
          if (old) {
            scene.remove(old);
            old.geometry.dispose();
          }
          const color = CAT_COLORS[i % CAT_COLORS.length]!;
          const pts = cat.plan.map(
            (p) => new THREE.Vector3(p.x, terrainY(p.x, p.z) + 0.3, p.z),
          );
          const pl = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(pts),
            new THREE.LineBasicMaterial({
              color,
              transparent: true,
              opacity: 0.5,
            }),
          );
          scene.add(pl);
          planLines[i] = pl;
        }

        affordanceGroup.visible = showAffordancesRef.current;

        // Capture flash fade.
        const flashLeft = sim.capturedFlashUntil - sim.simTime;
        const flashMat = captureFlash.material as THREE.MeshBasicMaterial;
        flashMat.opacity = Math.max(0, Math.min(0.45, flashLeft * 0.7));

        // Follow camera.
        const f = followRef.current;
        if (f === 'mouse') {
          controls.target.lerp(new THREE.Vector3(m.x, terrainY(m.x, m.z), m.z), 0.08);
        } else if (f === 'cat' && sim.cats[0]) {
          const c = sim.cats[0].state;
          controls.target.lerp(new THREE.Vector3(c.x, terrainY(c.x, c.z), c.z), 0.08);
        }

        controls.update();
        renderer.render(scene, camera);

        // HUD: min-distance, captures, and which cat (if any) is boosting/jumping.
        let minD = Infinity;
        let boosting = 0;
        let jumping = 0;
        for (const c of sim.cats) {
          const d = Math.hypot(c.state.x - m.x, c.state.z - m.z);
          if (d < minD) minD = d;
          if (c.usedBoost) boosting++;
          if (c.usedJump) jumping++;
        }
        setHud(
          `▶ ${sim.cats.length} cats · captures ${sim.captures} · ` +
            `closest ${minD.toFixed(1)} m · ` +
            `${boosting} on boost · ${jumping} on jump · ` +
            `t=${sim.simTime.toFixed(1)} s`,
        );
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

    return () => {
      disposed = true;
      clearTimeout(kickoff);
      if (cleanup) cleanup();
    };
    // Scene is built once; control changes drive sim re-inits via the effect
    // below (no full teardown).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Cat-count change → re-init the sim (cheap; navmesh stays cached).
  useEffect(() => {
    resetTickRef.current += 1;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [catCount]);

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
      <h1 style={{ fontSize: 18 }}>Cat &amp; Mouse pursuit</h1>
      <p style={{ opacity: 0.75, marginTop: 0 }}>
        AI cats observe a non-cooperative mouse, build a{' '}
        <code>Predict&lt;{`{x,z}`}&gt;</code> from its motion, and plan to the{' '}
        <em>interception</em> pose — where the mouse <em>will be</em> at the
        cat's arrival time, not where it <em>is</em>. The cyan ribbon shows the
        predicted future trajectory. Cats share plans via the{' '}
        <code>PlanRegistry</code> so they don't double up on the same lane —
        emergent flanking with no negotiation protocol. Both predator and prey
        can take the green boost pads; the canyon (un-meshed strip) is only
        crossable via the cyan jump affordance.
      </p>

      <div style={{ display: 'flex', gap: 8, margin: '8px 0', flexWrap: 'wrap' }}>
        {[1, 2, 3, 4].map((n) => (
          <button
            key={n}
            onClick={() => setCatCount(n)}
            style={btn(catCount === n)}
          >
            {n} cat{n > 1 ? 's' : ''}
          </button>
        ))}
        <button onClick={() => setPaused((v) => !v)} style={btn(paused)}>
          {paused ? 'play' : 'pause'}
        </button>
        <button
          onClick={() => {
            resetTickRef.current += 1;
          }}
          style={btn(false)}
        >
          reset
        </button>
        <label
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            border: '1px solid #2a2f3a',
            borderRadius: 6,
            padding: '4px 10px',
          }}
        >
          follow
          <select
            value={follow}
            onChange={(e) => setFollow(e.target.value as Follow)}
            style={{
              background: '#0b0b0f',
              color: '#cdd3de',
              border: 'none',
              outline: 'none',
            }}
          >
            <option value="orbit">orbit</option>
            <option value="mouse">mouse</option>
            <option value="cat">cat 1</option>
          </select>
        </label>
      </div>

      <div style={{ display: 'flex', gap: 8, margin: '8px 0', flexWrap: 'wrap' }}>
        <label
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            border: '1px solid #2a2f3a',
            borderRadius: 6,
            padding: '4px 10px',
          }}
        >
          deadline
          <select
            value={deadlineMs}
            onChange={(e) => setDeadlineMs(Number(e.target.value))}
            style={{
              background: '#0b0b0f',
              color: '#cdd3de',
              border: 'none',
              outline: 'none',
            }}
          >
            <option value={30}>30 ms</option>
            <option value={60}>60 ms</option>
            <option value={120}>120 ms</option>
            <option value={200}>200 ms</option>
          </select>
        </label>
        <label
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            border: '1px solid #2a2f3a',
            borderRadius: 6,
            padding: '4px 10px',
          }}
        >
          horizon
          <select
            value={horizon}
            onChange={(e) => setHorizon(Number(e.target.value))}
            style={{
              background: '#0b0b0f',
              color: '#cdd3de',
              border: 'none',
              outline: 'none',
            }}
          >
            <option value={1}>1 s</option>
            <option value={3}>3 s</option>
            <option value={6}>6 s</option>
          </select>
        </label>
        <button
          onClick={() => setPredictionOn((v) => !v)}
          style={btn(predictionOn)}
        >
          prediction: {predictionOn ? 'on' : 'naive (chase current)'}
        </button>
      </div>

      <div style={{ display: 'flex', gap: 8, margin: '8px 0', flexWrap: 'wrap' }}>
        <button
          onClick={() => setShowPredicted((v) => !v)}
          style={btn(showPredicted)}
        >
          predicted trajectory: {showPredicted ? 'on' : 'off'}
        </button>
        <button onClick={() => setShowPlans((v) => !v)} style={btn(showPlans)}>
          cat plans: {showPlans ? 'on' : 'off'}
        </button>
        <button
          onClick={() => setShowAffordances((v) => !v)}
          style={btn(showAffordances)}
        >
          affordances: {showAffordances ? 'on' : 'off'}
        </button>
      </div>

      <div
        ref={mountRef}
        style={{
          width: '100%',
          borderRadius: 8,
          overflow: 'hidden',
          cursor: 'grab',
        }}
      />
      <p style={{ opacity: 0.9, margin: '8px 0 2px' }}>{hud}</p>
      <p style={{ opacity: 0.7, marginTop: 0 }}>{status}</p>
    </main>
  );
}
