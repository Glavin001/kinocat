'use client';

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { aircraftForwardSim } from 'kinocat/agent';
import type { AircraftState } from 'kinocat/agent';
import { PlanRegistry } from 'kinocat/predict';
import {
  DOGFIGHT_AGENT,
  DOGFIGHT_BOUNDS,
  DOGFIGHT_PALETTE as C,
  dogfightAirspace,
  dogfightTerrain,
  dogfightStaticObstacles,
  dogfightMovingZones,
  dogfightBoostRings,
  playerForecast,
  selectTacticalMode,
  tacticalGoal,
  planAI,
  type TacticalMode,
} from '../lib/dogfight-scenarios';

interface AI {
  id: string;
  color: number;
  state: AircraftState;
  plan: AircraftState[] | null;
  planStartT: number;
  planStartWall: number;
  mode: TacticalMode;
  goal: AircraftState | null;
  lastReplanWall: number;
  lastExpansions: number;
  lastBudgetMs: number;
}

interface BoostState {
  endT: number;
  ringId: string | null;
}

const NUM_AIS = 3;
const PLAYER_INPUT_RATE = 1 / 60;

export default function Dogfight() {
  const mountRef = useRef<HTMLDivElement>(null);

  // UI state mirrored into refs so the rAF loop reads fresh values without
  // tearing down the scene on every toggle.
  const [paused, setPaused] = useState(false);
  const [chase, setChase] = useState(true);
  const [showPaths, setShowPaths] = useState(true);
  const [showCones, setShowCones] = useState(false);
  const [showZones, setShowZones] = useState(true);
  const [hud, setHud] = useState('');
  const [aiStatus, setAiStatus] = useState<string[]>([]);

  const pausedRef = useRef(paused);
  const chaseRef = useRef(chase);
  const showPathsRef = useRef(showPaths);
  const showConesRef = useRef(showCones);
  const showZonesRef = useRef(showZones);
  pausedRef.current = paused;
  chaseRef.current = chase;
  showPathsRef.current = showPaths;
  showConesRef.current = showCones;
  showZonesRef.current = showZones;

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const W = mount.clientWidth;
    const viewH = () =>
      Math.round(Math.min(620, Math.max(360, window.innerHeight * 0.7)));
    let Hpx = viewH();

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(C.bg);
    scene.fog = new THREE.Fog(C.fog, 220, 480);
    const camera = new THREE.PerspectiveCamera(62, W / Hpx, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(W, Hpx);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(120, 40, 0);
    camera.position.set(40, 100, 120);
    controls.update();

    scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const sun = new THREE.DirectionalLight(0xffffff, 1.1);
    sun.position.set(60, 200, 100);
    scene.add(sun);

    // ---- terrain mesh (displaced PlaneGeometry using dogfightTerrain) ------
    const tw = DOGFIGHT_BOUNDS.x1 - DOGFIGHT_BOUNDS.x0 + 40;
    const td = DOGFIGHT_BOUNDS.z1 - DOGFIGHT_BOUNDS.z0 + 40;
    const tgeo = new THREE.PlaneGeometry(tw, td, 192, 128);
    tgeo.rotateX(-Math.PI / 2);
    // World-space offset so plane spans the playable bounds.
    const tcx = (DOGFIGHT_BOUNDS.x0 + DOGFIGHT_BOUNDS.x1) / 2;
    const tcz = (DOGFIGHT_BOUNDS.z0 + DOGFIGHT_BOUNDS.z1) / 2;
    tgeo.translate(tcx, 0, tcz);
    // Displace + colour each vertex by its terrain sample.
    {
      const pos = tgeo.attributes['position'] as THREE.BufferAttribute;
      const cols = new Float32Array(pos.count * 3);
      const lo = new THREE.Color(C.terrainLow);
      const mid = new THREE.Color(C.terrainMid);
      const hi = new THREE.Color(C.terrainHigh);
      for (let i = 0; i < pos.count; i++) {
        const x = pos.getX(i);
        const z = pos.getZ(i);
        const y = dogfightTerrain(x, z);
        pos.setY(i, y);
        const t = Math.min(1, y / 28);
        const tmp = new THREE.Color();
        if (t < 0.5) tmp.copy(lo).lerp(mid, t * 2);
        else tmp.copy(mid).lerp(hi, (t - 0.5) * 2);
        cols[3 * i] = tmp.r;
        cols[3 * i + 1] = tmp.g;
        cols[3 * i + 2] = tmp.b;
      }
      tgeo.setAttribute('color', new THREE.BufferAttribute(cols, 3));
      tgeo.computeVertexNormals();
    }
    const terrainMesh = new THREE.Mesh(
      tgeo,
      new THREE.MeshStandardMaterial({
        vertexColors: true,
        flatShading: false,
        side: THREE.FrontSide,
      }),
    );
    scene.add(terrainMesh);

    // ---- static obstacles ---------------------------------------------------
    const obsGroup = new THREE.Group();
    scene.add(obsGroup);
    const wallMat = new THREE.MeshStandardMaterial({ color: C.wall });
    for (const b of dogfightStaticObstacles()) {
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
      obsGroup.add(m);
    }

    // ---- moving zones (blimp + sweeping barrier) ---------------------------
    const zones = dogfightMovingZones();
    const zoneMeshes: THREE.Mesh[] = [];
    for (let i = 0; i < zones.length; i++) {
      const z = zones[i]!;
      const mesh = new THREE.Mesh(
        new THREE.SphereGeometry(z.radius, 24, 18),
        new THREE.MeshStandardMaterial({
          color: i === 0 ? C.blimp : 0xff5577,
          transparent: true,
          opacity: i === 0 ? 0.85 : 0.32,
          emissive: i === 0 ? 0x000000 : 0xff3344,
          emissiveIntensity: i === 0 ? 0 : 0.35,
        }),
      );
      scene.add(mesh);
      zoneMeshes.push(mesh);
    }

    // ---- boost rings -------------------------------------------------------
    const rings = dogfightBoostRings();
    const ringMeshes: THREE.Mesh[] = [];
    for (const r of rings) {
      const m = new THREE.Mesh(
        new THREE.TorusGeometry(r.radius, 0.45, 12, 36),
        new THREE.MeshStandardMaterial({
          color: C.ring,
          emissive: C.ring,
          emissiveIntensity: 0.5,
          transparent: true,
          opacity: 0.85,
        }),
      );
      m.position.set(r.x, r.y, r.z);
      // Torus's hole axis is local +Z; orient it so the hole points along r.axis.
      const dir = new THREE.Vector3(r.axis.x, r.axis.y, r.axis.z).normalize();
      const q = new THREE.Quaternion().setFromUnitVectors(
        new THREE.Vector3(0, 0, 1),
        dir,
      );
      m.quaternion.copy(q);
      scene.add(m);
      ringMeshes.push(m);
    }

    // ---- player aircraft mesh ----------------------------------------------
    const buildPlane = (color: number) => {
      const group = new THREE.Group();
      const mat = new THREE.MeshStandardMaterial({ color });
      const fuse = new THREE.Mesh(
        new THREE.CylinderGeometry(0.28, 0.28, 2.6, 12),
        mat,
      );
      fuse.rotation.x = Math.PI / 2;
      const nose = new THREE.Mesh(new THREE.ConeGeometry(0.28, 1.0, 12), mat);
      nose.rotation.x = Math.PI / 2;
      nose.position.z = 1.8;
      const wing = new THREE.Mesh(
        new THREE.BoxGeometry(3.6, 0.12, 0.9),
        mat,
      );
      const tail = new THREE.Mesh(new THREE.BoxGeometry(1.2, 0.1, 0.5), mat);
      tail.position.z = -1.15;
      const fin = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.65, 0.5), mat);
      fin.position.set(0, 0.35, -1.15);
      group.add(fuse, nose, wing, tail, fin);
      return group;
    };
    const playerMesh = buildPlane(parseInt(C.player.slice(1), 16));
    scene.add(playerMesh);

    // ---- AIs ---------------------------------------------------------------
    const aiMeshes: THREE.Group[] = [];
    const aiPathLines: (THREE.Line | null)[] = [];
    const aiGoalMarks: THREE.Mesh[] = [];
    const aiCones: THREE.Mesh[] = [];

    const ais: AI[] = [];
    for (let i = 0; i < NUM_AIS; i++) {
      const color = C.enemy[i % C.enemy.length]!;
      const m = buildPlane(color);
      scene.add(m);
      aiMeshes.push(m);
      const goalMark = new THREE.Mesh(
        new THREE.OctahedronGeometry(1.6),
        new THREE.MeshStandardMaterial({
          color,
          emissive: color,
          emissiveIntensity: 0.6,
          transparent: true,
          opacity: 0.6,
        }),
      );
      scene.add(goalMark);
      aiGoalMarks.push(goalMark);
      const cone = new THREE.Mesh(
        new THREE.ConeGeometry(20, 60, 16, 1, true),
        new THREE.MeshStandardMaterial({
          color,
          transparent: true,
          opacity: 0.08,
          side: THREE.DoubleSide,
          emissive: color,
          emissiveIntensity: 0.1,
        }),
      );
      cone.visible = false;
      scene.add(cone);
      aiCones.push(cone);
      aiPathLines.push(null);
      ais.push({
        id: `AI${i}`,
        color,
        state: {
          x: 200 + (i - 1) * 10,
          y: 40 + i * 6,
          z: -30 + i * 30,
          heading: Math.PI,
          pitch: 0,
          roll: 0,
          speed: DOGFIGHT_AGENT.maxSpeed,
          t: 0,
        },
        plan: null,
        planStartT: 0,
        planStartWall: performance.now(),
        mode: 'PURSUE',
        goal: null,
        lastReplanWall: -Infinity,
        lastExpansions: 0,
        lastBudgetMs: 0,
      });
    }

    // ---- shared registry so AIs predict each other ------------------------
    const registry = new PlanRegistry();
    const airspace = dogfightAirspace();
    const sim = aircraftForwardSim(DOGFIGHT_AGENT);

    // ---- player state ------------------------------------------------------
    let player: AircraftState = {
      x: 30,
      y: 35,
      z: 0,
      heading: 0,
      pitch: 0,
      roll: 0,
      speed: DOGFIGHT_AGENT.maxSpeed * 0.9,
      t: 0,
    };
    let playerBoost: BoostState = { endT: -1, ringId: null };

    // ---- input -------------------------------------------------------------
    const keys = new Set<string>();
    const onKeyDown = (e: KeyboardEvent) => {
      const k = e.key.toLowerCase();
      keys.add(k);
      if (k === 'p') setPaused((v) => !v);
      if (k === 'c') setChase((v) => !v);
      if (k === '1') setShowPaths((v) => !v);
      if (k === '2') setShowCones((v) => !v);
      if (k === '3') setShowZones((v) => !v);
    };
    const onKeyUp = (e: KeyboardEvent) => keys.delete(e.key.toLowerCase());
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);

    // ---- planning loop: round-robin one AI per tick (~150ms) --------------
    let aiCursor = 0;
    const replanOne = () => {
      const ai = ais[aiCursor]!;
      aiCursor = (aiCursor + 1) % ais.length;
      const mode = selectTacticalMode(player, ai.state, aiCursor);
      const playerPredict = playerForecast(player, 5);
      const goal = tacticalGoal(player, playerPredict, ai.state, mode);
      const t0 = performance.now();
      const res = planAI(airspace, {
        npcId: ai.id,
        state: ai.state,
        goal,
        player,
        registry,
        otherNpcs: ais.filter((o) => o.id !== ai.id).map((o) => o.id),
        deadlineMs: 100,
        maxExpansions: 50000,
      });
      const dt = performance.now() - t0;
      ai.mode = mode;
      ai.goal = goal;
      ai.lastBudgetMs = dt;
      ai.lastExpansions = res.stats.expansions;
      ai.lastReplanWall = performance.now();
      if (res.found && res.path.length > 1) {
        // Re-base plan times so playback starts at 0.
        const t0p = res.path[0]!.t;
        ai.plan = res.path.map((p) => ({ ...p, t: p.t - t0p }));
        ai.planStartT = 0;
        ai.planStartWall = performance.now();
        registry.publish(
          ai.id,
          ai.plan.map((p) => ({ ...p, t: p.t + (performance.now() / 1000) })),
        );
      }
    };
    const planTimer = window.setInterval(() => {
      if (!pausedRef.current) replanOne();
    }, 160);

    // ---- HUD status reporter (lower-frequency setState) -------------------
    const hudTimer = window.setInterval(() => {
      const speed = player.speed;
      const alt = player.y;
      const boost = playerBoost.endT > player.t ? ' · BOOST' : '';
      setHud(
        `airspeed ${speed.toFixed(1)}  alt ${alt.toFixed(0)}  heading ${(
          (player.heading * 180) / Math.PI
        ).toFixed(0)}°  bank ${((player.roll * 180) / Math.PI).toFixed(0)}°${boost}`,
      );
      setAiStatus(
        ais.map(
          (a) =>
            `${a.id}: ${a.mode} · ${a.plan ? `path ${a.plan.length}` : 'no-plan'} · ` +
            `${a.lastExpansions} exp · ${a.lastBudgetMs.toFixed(0)} ms`,
        ),
      );
    }, 250);

    // ---- per-AI path-line refresh (when plan changes) ---------------------
    const refreshLine = (i: number) => {
      const ai = ais[i]!;
      if (aiPathLines[i]) {
        scene.remove(aiPathLines[i]!);
        aiPathLines[i]!.geometry.dispose();
        aiPathLines[i] = null;
      }
      if (!ai.plan || ai.plan.length < 2) return;
      const pts = ai.plan.map((p) => new THREE.Vector3(p.x, p.y, p.z));
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineBasicMaterial({
          color: ai.color,
          transparent: true,
          opacity: 0.55,
        }),
      );
      scene.add(line);
      aiPathLines[i] = line;
    };

    // ---- physics + render loop --------------------------------------------
    const fwd = new THREE.Vector3();
    const chaseOffset = new THREE.Vector3(-26, 8, 0);
    let last = performance.now();
    let frame = 0;
    const animate = () => {
      frame = requestAnimationFrame(animate);
      const now = performance.now();
      let dt = Math.min((now - last) / 1000, 0.05);
      last = now;
      if (pausedRef.current) dt = 0;

      // -- update player from input --
      if (dt > 0) {
        const pitch = keys.has('w') ? -0.6 : keys.has('s') ? 0.6 : 0;
        const roll = keys.has('a') ? -0.9 : keys.has('d') ? 0.9 : 0;
        const yawInput = keys.has('q') ? 1 : keys.has('e') ? -1 : 0;
        const throttle = keys.has('shift')
          ? 1
          : keys.has('control')
            ? -1
            : 0;
        // Roll commands a curvature scaled by sin(roll) — banked turns feel
        // natural without needing live yaw input.
        const targetRoll = roll * DOGFIGHT_AGENT.maxBank;
        const targetPitch = pitch * DOGFIGHT_AGENT.maxClimbAngle;
        const curvature =
          (Math.sin(targetRoll) / DOGFIGHT_AGENT.minTurnRadius) * 1.0 +
          yawInput * (0.4 / DOGFIGHT_AGENT.minTurnRadius);
        let targetSpeed = player.speed + throttle * 8 * dt;
        targetSpeed = Math.max(
          DOGFIGHT_AGENT.minSpeed,
          Math.min(DOGFIGHT_AGENT.maxSpeed, targetSpeed),
        );
        const boost = playerBoost.endT > player.t ? 1.3 : 1;
        const stepSpeed = targetSpeed * boost;
        // Use the kinematic forward sim — same model the planner uses, so
        // the AIs' predictions of the player stay calibrated.
        player = sim(
          player,
          [curvature, targetPitch, targetRoll, stepSpeed],
          dt,
        );
        // Soft horizontal bounds (snap inside the playable region).
        player.x = Math.max(
          DOGFIGHT_BOUNDS.x0 + 4,
          Math.min(DOGFIGHT_BOUNDS.x1 - 4, player.x),
        );
        player.z = Math.max(
          DOGFIGHT_BOUNDS.z0 + 4,
          Math.min(DOGFIGHT_BOUNDS.z1 - 4, player.z),
        );
        // Ground / ceiling clamp; bounce up off terrain.
        const groundY = dogfightTerrain(player.x, player.z);
        if (player.y < groundY + 4) {
          player.y = groundY + 4;
          if (player.pitch < 0) player.pitch = 0;
        }
        player.y = Math.min(player.y, DOGFIGHT_BOUNDS.ceiling - 4);

        // Ring detection
        for (const r of rings) {
          const dx = player.x - r.x;
          const dy = player.y - r.y;
          const dz = player.z - r.z;
          if (dx * dx + dy * dy + dz * dz < (r.radius + 1) * (r.radius + 1)) {
            if (playerBoost.ringId !== r.id) {
              playerBoost = { endT: player.t + 2.5, ringId: r.id };
            }
          }
        }
        if (playerBoost.endT < player.t) playerBoost.ringId = null;
      }

      // -- advance AIs along their plans --
      const wall = performance.now();
      for (let i = 0; i < ais.length; i++) {
        const ai = ais[i]!;
        const plan = ai.plan;
        if (!plan || plan.length < 2) {
          // Drift forward at level cruise until a plan arrives.
          if (dt > 0) {
            ai.state = sim(
              ai.state,
              [0, 0, 0, DOGFIGHT_AGENT.maxSpeed * 0.9],
              dt,
            );
          }
          continue;
        }
        // Playback time = wall-clock since plan committed.
        const tp = (wall - ai.planStartWall) / 1000;
        const sampled = samplePathAt(plan, tp);
        ai.state = {
          x: sampled.x,
          y: sampled.y,
          z: sampled.z,
          heading: sampled.heading,
          pitch: sampled.pitch,
          roll: sampled.roll,
          speed: sampled.speed,
          t: ai.state.t + dt,
        };
      }

      // -- update visuals --
      orientPlane(playerMesh, player, fwd);
      for (let i = 0; i < ais.length; i++) {
        const ai = ais[i]!;
        orientPlane(aiMeshes[i]!, ai.state, fwd);
        const gm = aiGoalMarks[i]!;
        gm.visible = !!ai.goal;
        if (ai.goal) gm.position.set(ai.goal.x, ai.goal.y, ai.goal.z);
        // Vision cone — origin at AI nose, pointing along its heading.
        const cone = aiCones[i]!;
        cone.visible = showConesRef.current;
        if (cone.visible) {
          const cp = Math.cos(ai.state.pitch);
          cone.position.set(
            ai.state.x + 30 * cp * Math.cos(ai.state.heading),
            ai.state.y + 30 * Math.sin(ai.state.pitch),
            ai.state.z + 30 * cp * Math.sin(ai.state.heading),
          );
          cone.lookAt(ai.state.x, ai.state.y, ai.state.z);
          cone.rotateX(Math.PI / 2);
        }
        // Refresh path line if its target reference changed.
        const line = aiPathLines[i];
        const hasPlan = ai.plan && ai.plan.length > 1;
        if (showPathsRef.current && hasPlan && line === null) refreshLine(i);
        if (line) line.visible = showPathsRef.current && !!hasPlan;
        if (line && hasPlan) {
          // Cheap check: refresh if the line vertex count is stale.
          const need = ai.plan!.length;
          const have =
            (line.geometry.attributes['position'] as THREE.BufferAttribute)
              .count;
          if (have !== need) refreshLine(i);
        }
      }

      // -- moving zones --
      for (let i = 0; i < zones.length; i++) {
        const z = zones[i]!;
        const p = z.predict(player.t);
        const mesh = zoneMeshes[i]!;
        if (p) {
          mesh.position.set(p.x, p.y, p.z);
          mesh.visible = showZonesRef.current;
        } else {
          mesh.visible = false;
        }
      }

      // -- camera (chase / orbit) --
      if (chaseRef.current) {
        controls.enabled = false;
        const ch = Math.cos(player.heading);
        const sh = Math.sin(player.heading);
        const offX = chaseOffset.x * ch - chaseOffset.z * sh;
        const offZ = chaseOffset.x * sh + chaseOffset.z * ch;
        camera.position.lerp(
          new THREE.Vector3(player.x + offX, player.y + chaseOffset.y, player.z + offZ),
          0.18,
        );
        camera.lookAt(player.x, player.y, player.z);
      } else {
        controls.enabled = true;
        controls.update();
      }

      renderer.render(scene, camera);
    };
    animate();

    // ---- resize -----------------------------------------------------------
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
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('resize', onResize);
      window.clearInterval(planTimer);
      window.clearInterval(hudTimer);
      controls.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === mount)
        mount.removeChild(renderer.domElement);
    };
    // The scene mounts once; control toggles flow through refs.
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
        maxWidth: 1080,
        margin: '0 auto',
      }}
    >
      <a href="/" style={{ color: '#7fd6ff' }}>
        ← demos
      </a>
      <h1 style={{ fontSize: 18 }}>Dogfight — interactive 3D</h1>
      <p style={{ opacity: 0.75, marginTop: 0 }}>
        Pilot the cyan aircraft with the keyboard while {NUM_AIS} kinocat-driven
        opponents pursue, intercept, and flank you through a heightfield
        terrain, sky-high pylons, moving no-fly zones, and a sweeping barrier
        between the twin peaks. Each AI rebuilds its plan every ~160 ms against
        the live <em>predicted</em> player trajectory; sibling AIs read each
        other from a shared plan registry so they don&apos;t pile up. A new{' '}
        <code>HeightfieldAirspace</code> in core gives the planner real
        ground-elevation collision — fly low through valleys to break line of
        sight.
      </p>

      <div
        style={{ display: 'flex', gap: 8, margin: '8px 0', flexWrap: 'wrap' }}
      >
        <button onClick={() => setPaused((v) => !v)} style={btn(paused)}>
          {paused ? 'play' : 'pause'} (p)
        </button>
        <button onClick={() => setChase((v) => !v)} style={btn(chase)}>
          camera: {chase ? 'chase' : 'orbit'} (c)
        </button>
        <button onClick={() => setShowPaths((v) => !v)} style={btn(showPaths)}>
          AI plans (1)
        </button>
        <button onClick={() => setShowCones((v) => !v)} style={btn(showCones)}>
          AI vision cones (2)
        </button>
        <button onClick={() => setShowZones((v) => !v)} style={btn(showZones)}>
          moving zones (3)
        </button>
      </div>

      <div
        ref={mountRef}
        style={{
          width: '100%',
          borderRadius: 8,
          overflow: 'hidden',
          outline: 'none',
        }}
        tabIndex={0}
      />

      <p style={{ opacity: 0.9, margin: '8px 0 2px' }}>{hud}</p>
      <ul
        style={{
          listStyle: 'none',
          padding: 0,
          margin: '4px 0 0',
          opacity: 0.85,
          fontSize: 13,
        }}
      >
        {aiStatus.map((s, i) => (
          <li key={i}>{s}</li>
        ))}
      </ul>
      <p style={{ opacity: 0.7, marginTop: 12, fontSize: 13 }}>
        <strong>Controls:</strong> <code>W/S</code> pitch · <code>A/D</code>{' '}
        roll · <code>Q/E</code> yaw · <code>Shift / Ctrl</code> throttle ± ·
        green rings give a temporary speed boost · <code>P</code> pause ·{' '}
        <code>C</code> camera · <code>1 2 3</code> overlays.
      </p>
    </main>
  );
}

// ---------------------------------------------------------------------------
// Helpers

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}
function lerpAngle(a: number, b: number, t: number) {
  let d = ((b - a + Math.PI) % (2 * Math.PI)) - Math.PI;
  if (d < -Math.PI) d += 2 * Math.PI;
  return a + d * t;
}
function samplePathAt(path: AircraftState[], tp: number) {
  if (path.length === 1) {
    const p = path[0]!;
    return {
      x: p.x,
      y: p.y,
      z: p.z,
      heading: p.heading,
      pitch: p.pitch,
      roll: p.roll,
      speed: p.speed,
    };
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
    speed: lerp(a.speed, b.speed, u),
  };
}
function orientPlane(
  group: THREE.Object3D,
  s: AircraftState,
  fwdScratch: THREE.Vector3,
) {
  group.position.set(s.x, s.y, s.z);
  const cp = Math.cos(s.pitch);
  fwdScratch.set(
    s.x + cp * Math.cos(s.heading),
    s.y + Math.sin(s.pitch),
    s.z + cp * Math.sin(s.heading),
  );
  group.lookAt(fwdScratch);
  group.rotateZ(-s.roll);
}
