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
  DOGFIGHT_HALF,
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
  /** Wall time at which a crashed AI may rejoin combat (-Infinity = active). */
  respawnAtWall: number;
}

interface Score {
  aiWins: number;
  crashes: number;
  round: number;
}

interface Banner {
  text: string;
  color: string;
  untilWall: number;
}

const NUM_AIS = 3;
const CAPTURE_RADIUS = 7;
const RESPAWN_INVINCIBLE_MS = 1500;
const AI_RESPAWN_DELAY_MS = 2200;

/** Player spawn points used after a crash / capture — picked far from AIs. */
const PLAYER_SPAWNS: Array<Omit<AircraftState, 'speed' | 't'>> = [
  { x: 20, y: 38, z: 0, heading: 0, pitch: 0, roll: 0 },
  { x: 20, y: 55, z: -50, heading: 0.2, pitch: 0, roll: 0 },
  { x: 30, y: 60, z: 50, heading: -0.2, pitch: 0, roll: 0 },
  { x: 10, y: 42, z: -25, heading: 0, pitch: 0, roll: 0 },
];

/** AI spawn points used after an AI crash. */
const AI_SPAWNS: Array<Omit<AircraftState, 'speed' | 't'>> = [
  { x: 225, y: 50, z: -45, heading: Math.PI, pitch: 0, roll: 0 },
  { x: 225, y: 55, z: 45, heading: Math.PI, pitch: 0, roll: 0 },
  { x: 220, y: 65, z: 0, heading: Math.PI, pitch: 0, roll: 0 },
  { x: 215, y: 45, z: -25, heading: Math.PI + 0.15, pitch: 0, roll: 0 },
];

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
  const [score, setScore] = useState<Score>({ aiWins: 0, crashes: 0, round: 1 });
  const [banner, setBanner] = useState<Banner | null>(null);

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

    const vpW = () => window.innerWidth;
    const vpH = () => window.innerHeight;
    let W = vpW();
    let H = vpH();

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(C.bg);
    scene.fog = new THREE.Fog(C.fog, 220, 540);
    const camera = new THREE.PerspectiveCamera(64, W / H, 0.1, 1200);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(W, H);
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
    const tcx = (DOGFIGHT_BOUNDS.x0 + DOGFIGHT_BOUNDS.x1) / 2;
    const tcz = (DOGFIGHT_BOUNDS.z0 + DOGFIGHT_BOUNDS.z1) / 2;
    tgeo.translate(tcx, 0, tcz);
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
      const dir = new THREE.Vector3(r.axis.x, r.axis.y, r.axis.z).normalize();
      const q = new THREE.Quaternion().setFromUnitVectors(
        new THREE.Vector3(0, 0, 1),
        dir,
      );
      m.quaternion.copy(q);
      scene.add(m);
      ringMeshes.push(m);
    }

    // ---- aircraft prefabs --------------------------------------------------
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
      const wing = new THREE.Mesh(new THREE.BoxGeometry(3.6, 0.12, 0.9), mat);
      const tail = new THREE.Mesh(new THREE.BoxGeometry(1.2, 0.1, 0.5), mat);
      tail.position.z = -1.15;
      const fin = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.65, 0.5), mat);
      fin.position.set(0, 0.35, -1.15);
      group.add(fuse, nose, wing, tail, fin);
      return group;
    };
    const playerMesh = buildPlane(parseInt(C.player.slice(1), 16));
    scene.add(playerMesh);
    const playerMaterials = (playerMesh.children as THREE.Mesh[]).map(
      (m) => m.material as THREE.MeshStandardMaterial,
    );

    // Explosion marker reused for both player and AI crashes.
    const burst = new THREE.Mesh(
      new THREE.SphereGeometry(2.5, 16, 12),
      new THREE.MeshStandardMaterial({
        color: 0xff8844,
        emissive: 0xff5522,
        emissiveIntensity: 1.2,
        transparent: true,
        opacity: 0,
      }),
    );
    scene.add(burst);
    let burstUntilWall = -Infinity;
    const triggerBurst = (x: number, y: number, z: number) => {
      burst.position.set(x, y, z);
      burstUntilWall = performance.now() + 700;
    };

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
      const spawn = AI_SPAWNS[i % AI_SPAWNS.length]!;
      ais.push({
        id: `AI${i}`,
        color,
        state: {
          ...spawn,
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
        respawnAtWall: -Infinity,
      });
    }

    const registry = new PlanRegistry();
    const airspace = dogfightAirspace();
    const sim = aircraftForwardSim(DOGFIGHT_AGENT);

    // ---- player state ------------------------------------------------------
    let player: AircraftState = makePlayerSpawn(0, []);
    let playerInvincibleUntilWall = performance.now() + RESPAWN_INVINCIBLE_MS;
    let playerCrashedUntilWall = -Infinity;
    let playerBoostUntilT = -1;
    let lastBoostRing: string | null = null;
    const scoreRef = { aiWins: 0, crashes: 0, round: 1 };

    /** Pick the player spawn farthest from any active AI. */
    function makePlayerSpawn(
      round: number,
      activeAis: AI[],
    ): AircraftState {
      let best = PLAYER_SPAWNS[0]!;
      let bestD = -Infinity;
      for (const s of PLAYER_SPAWNS) {
        let minDtoAi = Infinity;
        for (const a of activeAis) {
          const d = Math.hypot(s.x - a.state.x, s.z - a.state.z);
          if (d < minDtoAi) minDtoAi = d;
        }
        if (minDtoAi > bestD) {
          bestD = minDtoAi;
          best = s;
        }
      }
      // Round bumps the start altitude slightly so successive respawns stagger.
      const yOff = (round % 3) * 4;
      return {
        x: best.x,
        y: best.y + yOff,
        z: best.z,
        heading: best.heading,
        pitch: 0,
        roll: 0,
        speed: DOGFIGHT_AGENT.maxSpeed * 0.9,
        t: 0,
      };
    }

    function respawnPlayer(reason: 'crash' | 'caught', byAi: string | null) {
      triggerBurst(player.x, player.y, player.z);
      const wall = performance.now();
      if (reason === 'crash') {
        scoreRef.crashes += 1;
        flashBanner('CRASHED — you flew into the terrain', '#ff7777', 1800);
      } else {
        scoreRef.aiWins += 1;
        scoreRef.round += 1;
        flashBanner(
          `${byAi ?? 'An AI'} got you! Round ${scoreRef.round}`,
          '#ffaa44',
          1800,
        );
      }
      setScore({ ...scoreRef });
      player = makePlayerSpawn(scoreRef.round, ais);
      playerInvincibleUntilWall = wall + RESPAWN_INVINCIBLE_MS;
      playerCrashedUntilWall = -Infinity;
      playerBoostUntilT = -1;
      lastBoostRing = null;
      // Kick the AIs into immediate replanning toward the new spawn — clearing
      // their plans triggers a no-plan replan on the next loop tick.
      for (const a of ais) {
        a.plan = null;
        registry.remove(a.id);
      }
    }

    let bannerToken = 0;
    function flashBanner(text: string, color: string, ms: number) {
      const wall = performance.now();
      const token = ++bannerToken;
      setBanner({ text, color, untilWall: wall + ms });
      window.setTimeout(() => {
        // Only clear if no newer banner has appeared in the meantime.
        if (token === bannerToken) setBanner(null);
      }, ms);
    }

    function crashAi(ai: AI) {
      triggerBurst(ai.state.x, ai.state.y, ai.state.z);
      ai.plan = null;
      registry.remove(ai.id);
      ai.respawnAtWall = performance.now() + AI_RESPAWN_DELAY_MS;
    }

    function respawnAi(ai: AI, idx: number) {
      const spawn = AI_SPAWNS[idx % AI_SPAWNS.length]!;
      ai.state = {
        ...spawn,
        speed: DOGFIGHT_AGENT.maxSpeed,
        t: ai.state.t, // continue absolute game time so registry stays valid
      };
      ai.respawnAtWall = -Infinity;
      ai.plan = null;
    }

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

    // ---- planning loop -----------------------------------------------------
    let aiCursor = 0;
    const replanOne = () => {
      // Skip crashed AIs in the rotation.
      let tries = 0;
      while (tries < ais.length && ais[aiCursor]!.respawnAtWall !== -Infinity) {
        aiCursor = (aiCursor + 1) % ais.length;
        tries += 1;
      }
      if (tries >= ais.length) return; // all AIs respawning
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
        otherNpcs: ais
          .filter((o) => o.id !== ai.id && o.respawnAtWall === -Infinity)
          .map((o) => o.id),
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
        const t0p = res.path[0]!.t;
        ai.plan = res.path.map((p) => ({ ...p, t: p.t - t0p }));
        ai.planStartT = 0;
        ai.planStartWall = performance.now();
        registry.publish(
          ai.id,
          ai.plan.map((p) => ({ ...p, t: p.t + performance.now() / 1000 })),
        );
      }
    };
    const planTimer = window.setInterval(() => {
      if (!pausedRef.current) replanOne();
    }, 160);

    // ---- HUD reporter ------------------------------------------------------
    const hudTimer = window.setInterval(() => {
      const boost = playerBoostUntilT > player.t ? ' · BOOST' : '';
      const invinc =
        performance.now() < playerInvincibleUntilWall ? ' · INVINCIBLE' : '';
      setHud(
        `airspeed ${player.speed.toFixed(1)}  alt ${player.y.toFixed(0)}  ` +
          `heading ${((player.heading * 180) / Math.PI).toFixed(0)}°  ` +
          `bank ${((player.roll * 180) / Math.PI).toFixed(0)}°${boost}${invinc}`,
      );
      setAiStatus(
        ais.map(
          (a) =>
            `${a.id}: ${
              a.respawnAtWall !== -Infinity ? 'RESPAWNING' : a.mode
            } · ${a.plan ? `path ${a.plan.length}` : 'no-plan'} · ` +
            `${a.lastExpansions} exp · ${a.lastBudgetMs.toFixed(0)} ms`,
        ),
      );
    }, 250);

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

    function poseOf(s: AircraftState) {
      return {
        x: s.x,
        y: s.y,
        z: s.z,
        yaw: s.heading,
        pitch: s.pitch,
        roll: s.roll,
      };
    }

    const animate = () => {
      frame = requestAnimationFrame(animate);
      const now = performance.now();
      let dt = Math.min((now - last) / 1000, 0.05);
      last = now;
      if (pausedRef.current) dt = 0;

      // -- player input + integration --
      const playerAlive = now >= playerCrashedUntilWall;
      if (dt > 0 && playerAlive) {
        const pitch = keys.has('w') ? -0.6 : keys.has('s') ? 0.6 : 0;
        const roll = keys.has('a') ? -0.9 : keys.has('d') ? 0.9 : 0;
        const yawInput = keys.has('q') ? 1 : keys.has('e') ? -1 : 0;
        const throttle = keys.has('shift')
          ? 1
          : keys.has('control')
            ? -1
            : 0;
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
        const boost = playerBoostUntilT > player.t ? 1.3 : 1;
        const stepSpeed = targetSpeed * boost;
        player = sim(
          player,
          [curvature, targetPitch, targetRoll, stepSpeed],
          dt,
        );
        // Soft horizontal bounds — keep the playable area finite without
        // crashing (going to the edge is a navigation choice, not a fail).
        player.x = Math.max(
          DOGFIGHT_BOUNDS.x0 + 4,
          Math.min(DOGFIGHT_BOUNDS.x1 - 4, player.x),
        );
        player.z = Math.max(
          DOGFIGHT_BOUNDS.z0 + 4,
          Math.min(DOGFIGHT_BOUNDS.z1 - 4, player.z),
        );
        player.y = Math.min(player.y, DOGFIGHT_BOUNDS.ceiling - 4);

        // -- crash detection: solid airspace check (terrain + obstacles + zones).
        // No auto ground-clamp — fly too low and you crash.
        if (now > playerInvincibleUntilWall) {
          if (!airspace.clear(poseOf(player), DOGFIGHT_HALF, player.t)) {
            playerCrashedUntilWall = now + 1100;
            // Defer the actual respawn so the burst is visible at the impact.
            window.setTimeout(() => respawnPlayer('crash', null), 1100);
          }
        }

        // -- ring detection (visual / sim-only speed boost) --
        for (const r of rings) {
          const dx = player.x - r.x;
          const dy = player.y - r.y;
          const dz = player.z - r.z;
          if (dx * dx + dy * dy + dz * dz < (r.radius + 1) * (r.radius + 1)) {
            if (lastBoostRing !== r.id) {
              playerBoostUntilT = player.t + 2.5;
              lastBoostRing = r.id;
            }
          }
        }
        if (playerBoostUntilT < player.t) lastBoostRing = null;
      }

      // -- advance AIs along their plans + crash check + capture check --
      const wall = performance.now();
      for (let i = 0; i < ais.length; i++) {
        const ai = ais[i]!;
        // Respawn an AI whose cooldown elapsed.
        if (ai.respawnAtWall !== -Infinity && wall >= ai.respawnAtWall) {
          respawnAi(ai, i);
        }
        const respawning = ai.respawnAtWall !== -Infinity;

        if (!respawning) {
          const plan = ai.plan;
          if (!plan || plan.length < 2) {
            if (dt > 0) {
              ai.state = sim(
                ai.state,
                [0, 0, 0, DOGFIGHT_AGENT.maxSpeed * 0.9],
                dt,
              );
            }
          } else {
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

          // AI crash check — same airspace contract as the player.
          if (
            !airspace.clear(poseOf(ai.state), DOGFIGHT_HALF, ai.state.t)
          ) {
            crashAi(ai);
            continue;
          }

          // Capture check (skipped while player is invincible/crashed).
          if (playerAlive && wall > playerInvincibleUntilWall) {
            const dx = player.x - ai.state.x;
            const dy = player.y - ai.state.y;
            const dz = player.z - ai.state.z;
            if (
              dx * dx + dy * dy + dz * dz <
              CAPTURE_RADIUS * CAPTURE_RADIUS
            ) {
              playerCrashedUntilWall = wall + 1100;
              window.setTimeout(() => respawnPlayer('caught', ai.id), 1100);
              break;
            }
          }
        }
      }

      // -- update visuals --
      const alive = now < playerCrashedUntilWall ? false : true;
      playerMesh.visible = alive;
      if (alive) orientPlane(playerMesh, player, fwd);
      // Subtle invincibility shimmer — pulse emissive.
      const invinc = now < playerInvincibleUntilWall && alive;
      const pulse = invinc ? 0.4 + 0.4 * Math.sin(now / 80) : 0;
      for (const m of playerMaterials)
        m.emissiveIntensity = pulse;

      for (let i = 0; i < ais.length; i++) {
        const ai = ais[i]!;
        const respawning = ai.respawnAtWall !== -Infinity;
        aiMeshes[i]!.visible = !respawning;
        if (!respawning) orientPlane(aiMeshes[i]!, ai.state, fwd);
        const gm = aiGoalMarks[i]!;
        gm.visible = !!ai.goal && !respawning;
        if (ai.goal) gm.position.set(ai.goal.x, ai.goal.y, ai.goal.z);
        const cone = aiCones[i]!;
        cone.visible = showConesRef.current && !respawning;
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
        const line = aiPathLines[i];
        const hasPlan = ai.plan && ai.plan.length > 1;
        if (showPathsRef.current && hasPlan && line === null) refreshLine(i);
        if (line) line.visible = showPathsRef.current && !!hasPlan && !respawning;
        if (line && hasPlan) {
          const need = ai.plan!.length;
          const have =
            (line.geometry.attributes['position'] as THREE.BufferAttribute)
              .count;
          if (have !== need) refreshLine(i);
        }
      }

      // -- burst animation --
      if (now < burstUntilWall) {
        const u = 1 - (burstUntilWall - now) / 700;
        burst.scale.setScalar(1 + u * 4);
        (burst.material as THREE.MeshStandardMaterial).opacity = 1 - u;
      } else {
        (burst.material as THREE.MeshStandardMaterial).opacity = 0;
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

      // -- camera --
      if (chaseRef.current && alive) {
        controls.enabled = false;
        const ch = Math.cos(player.heading);
        const sh = Math.sin(player.heading);
        const offX = chaseOffset.x * ch - chaseOffset.z * sh;
        const offZ = chaseOffset.x * sh + chaseOffset.z * ch;
        camera.position.lerp(
          new THREE.Vector3(
            player.x + offX,
            player.y + chaseOffset.y,
            player.z + offZ,
          ),
          0.18,
        );
        camera.lookAt(player.x, player.y, player.z);
      } else if (chaseRef.current) {
        // Player dead — hold last camera and look at burst point.
        camera.lookAt(burst.position);
      } else {
        controls.enabled = true;
        controls.update();
      }

      renderer.render(scene, camera);
    };
    animate();

    const onResize = () => {
      W = vpW();
      H = vpH();
      camera.aspect = W / H;
      camera.updateProjectionMatrix();
      renderer.setSize(W, H);
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const overlayBg = 'rgba(10, 14, 22, 0.72)';
  const overlayBorder = '1px solid #2a2f3a';
  const overlayCommon: React.CSSProperties = {
    background: overlayBg,
    border: overlayBorder,
    borderRadius: 8,
    padding: '10px 12px',
    color: '#cdd3de',
    fontFamily: 'ui-monospace, monospace',
    fontSize: 12.5,
    backdropFilter: 'blur(4px)',
  };
  const btn = (active: boolean): React.CSSProperties => ({
    background: active ? '#2a3550' : '#161a22cc',
    color: '#cdd3de',
    border: '1px solid #2a2f3a',
    borderRadius: 6,
    padding: '5px 9px',
    fontFamily: 'ui-monospace, monospace',
    fontSize: 12,
    cursor: 'pointer',
  });

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        background: '#0a0e16',
        overflow: 'hidden',
      }}
    >
      <div ref={mountRef} style={{ position: 'absolute', inset: 0 }} />

      {/* Top-left: brand + back */}
      <div style={{ position: 'absolute', top: 12, left: 12, ...overlayCommon }}>
        <a href="/" style={{ color: '#7fd6ff', textDecoration: 'none' }}>
          ← demos
        </a>
        <div style={{ marginTop: 4, fontWeight: 600 }}>
          Dogfight — interactive 3D
        </div>
        <div style={{ opacity: 0.7, marginTop: 2 }}>
          {NUM_AIS} kinocat AIs · live replanning · heightfield collision
        </div>
      </div>

      {/* Top-right: scoreboard */}
      <div
        style={{
          position: 'absolute',
          top: 12,
          right: 12,
          ...overlayCommon,
          minWidth: 170,
        }}
      >
        <div style={{ fontSize: 11, opacity: 0.7, marginBottom: 2 }}>SCORE</div>
        <div style={{ fontSize: 18, fontWeight: 600 }}>
          <span style={{ color: '#ff7799' }}>AI {score.aiWins}</span>
          {'  '}
          <span style={{ opacity: 0.4 }}>·</span>{'  '}
          <span style={{ color: '#7fd6ff' }}>YOU 0</span>
        </div>
        <div style={{ marginTop: 4, fontSize: 12, opacity: 0.75 }}>
          round {score.round} · {score.crashes} crashes
        </div>
      </div>

      {/* Top-centre: control bar */}
      <div
        style={{
          position: 'absolute',
          top: 12,
          left: '50%',
          transform: 'translateX(-50%)',
          display: 'flex',
          gap: 6,
          flexWrap: 'wrap',
          justifyContent: 'center',
          ...overlayCommon,
          padding: '6px 8px',
        }}
      >
        <button onClick={() => setPaused((v) => !v)} style={btn(paused)}>
          {paused ? '▶ play' : '⏸ pause'} (P)
        </button>
        <button onClick={() => setChase((v) => !v)} style={btn(chase)}>
          camera: {chase ? 'chase' : 'orbit'} (C)
        </button>
        <button onClick={() => setShowPaths((v) => !v)} style={btn(showPaths)}>
          AI plans (1)
        </button>
        <button onClick={() => setShowCones((v) => !v)} style={btn(showCones)}>
          vision (2)
        </button>
        <button onClick={() => setShowZones((v) => !v)} style={btn(showZones)}>
          zones (3)
        </button>
      </div>

      {/* Bottom-left: HUD + AI status */}
      <div
        style={{
          position: 'absolute',
          bottom: 12,
          left: 12,
          ...overlayCommon,
          maxWidth: 380,
        }}
      >
        <div style={{ marginBottom: 4 }}>{hud}</div>
        <ul
          style={{
            listStyle: 'none',
            padding: 0,
            margin: '4px 0 0',
            opacity: 0.85,
          }}
        >
          {aiStatus.map((s, i) => (
            <li key={i}>{s}</li>
          ))}
        </ul>
      </div>

      {/* Bottom-right: control hint */}
      <div
        style={{
          position: 'absolute',
          bottom: 12,
          right: 12,
          ...overlayCommon,
          maxWidth: 320,
          opacity: 0.85,
        }}
      >
        <div style={{ fontSize: 11, opacity: 0.7, marginBottom: 4 }}>
          CONTROLS
        </div>
        <code>W/S</code> pitch · <code>A/D</code> roll · <code>Q/E</code> yaw
        <br />
        <code>Shift / Ctrl</code> throttle ± · green rings boost
      </div>

      {/* Centre banner: flashes on crash / capture */}
      {banner && (
        <div
          style={{
            position: 'absolute',
            top: '38%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            background: 'rgba(10, 14, 22, 0.88)',
            border: `1px solid ${banner.color}`,
            color: banner.color,
            padding: '14px 22px',
            borderRadius: 10,
            fontFamily: 'ui-monospace, monospace',
            fontSize: 20,
            fontWeight: 700,
            letterSpacing: 1,
            textAlign: 'center',
            pointerEvents: 'none',
            textShadow: `0 0 12px ${banner.color}`,
          }}
        >
          {banner.text}
        </div>
      )}
    </div>
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
