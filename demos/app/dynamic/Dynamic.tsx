'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { plan } from 'kinocat/planner';
import { InMemoryNavWorld, VehicleEnvironment, TimeAwareEnvironment } from 'kinocat/environment';
import { linearObstacle, asObstacle, PlanRegistry, AffordanceRegistry, createJumpAffordance } from 'kinocat/predict';
import { planPoseAt } from 'kinocat/execute';
import type { VehicleState } from 'kinocat/agent';
import { buildVehicle, PALETTE } from '../lib/vehicle';

const SCALE = 19;
const { agent, lib } = buildVehicle();
type Scenario = 'moving' | 'coop' | 'jump';

interface Built {
  bounds: { x0: number; z0: number; x1: number; z1: number };
  islands: [number, number, number, number][];
  path: VehicleState[];
  duration: number;
  start: VehicleState;
  goal: VehicleState;
  ghostAt?: (t: number) => { x: number; z: number } | null;
  ghostLabel?: string;
  affordanceHop?: [VehicleState, VehicleState] | null;
  info: string;
}

function floor(x0: number, z0: number, x1: number, z1: number) {
  return { id: 1, y: 0, ring: [[x0, z0], [x1, z0], [x1, z1], [x0, z1]] as [number, number][] };
}

function build(scn: Scenario): Built {
  const start: VehicleState = { x: 3, z: 0, heading: 0, speed: 0, t: 0 };

  if (scn === 'moving') {
    const goal: VehicleState = { x: 37, z: 0, heading: 0, speed: 0, t: 0 };
    const world = new InMemoryNavWorld([floor(0, -12, 40, 12)]);
    const obstacle = linearObstacle(20, -12, 0, 4, 2.5, 0, 60);
    const env = new TimeAwareEnvironment(
      new VehicleEnvironment(world, agent, lib, { goalRadius: 1.5, goalHeadingTol: Infinity }),
      { obstacles: [obstacle], agentRadius: 1.4 },
    );
    const r = plan({ start, goal, environment: env, options: { maxExpansions: 500000 } }, Infinity);
    const dur = r.found ? r.path[r.path.length - 1]!.t : 0;
    return {
      bounds: { x0: 0, z0: -12, x1: 40, z1: 12 },
      islands: [[0, -12, 40, 12]],
      path: r.path,
      duration: dur,
      start,
      goal,
      ghostAt: (t) => obstacle.predict(t),
      ghostLabel: 'moving obstacle (r=2.5)',
      info: r.found
        ? `avoids a linearly-moving obstacle; path time ${dur.toFixed(1)} s, ${r.path.length} states`
        : 'no plan',
    };
  }

  if (scn === 'coop') {
    const goal: VehicleState = { x: 37, z: 2, heading: 0, speed: 0, t: 0 };
    const world = new InMemoryNavWorld([floor(0, -12, 40, 12)]);
    const reg = new PlanRegistry();
    // NPC A cruises slowly straight along z=0 across the corridor.
    const aPlan: VehicleState[] = [];
    for (let i = 0; i <= 10; i++) {
      aPlan.push({ x: 6 + i * 3, z: 0, heading: 0, speed: 4, t: i * 0.75 });
    }
    reg.publish('A', aPlan);
    const env = new TimeAwareEnvironment(
      new VehicleEnvironment(world, agent, lib, { goalRadius: 1.5, goalHeadingTol: Infinity }),
      { obstacles: [asObstacle(reg.predictNPC('A'), 2.5)], agentRadius: 1.4 },
    );
    const r = plan({ start, goal, environment: env, options: { maxExpansions: 500000 } }, Infinity);
    const dur = r.found ? r.path[r.path.length - 1]!.t : 0;
    return {
      bounds: { x0: 0, z0: -12, x1: 40, z1: 12 },
      islands: [[0, -12, 40, 12]],
      path: r.path,
      duration: Math.max(dur, aPlan[aPlan.length - 1]!.t),
      start,
      goal,
      ghostAt: (t) => {
        const p = reg.predictNPC('A')(t);
        return p ? { x: p.x, z: p.z } : null;
      },
      ghostLabel: "NPC A's published plan (B routes around it)",
      info: r.found
        ? `NPC B reads NPC A's plan from the registry and weaves around it — emergent coordination, no negotiation`
        : 'no plan',
    };
  }

  // jump
  const goal: VehicleState = { x: 36, z: 0, heading: 0, speed: 0, t: 0 };
  const world = new InMemoryNavWorld([floor(0, -6, 16, 6), floor(0, -6, 16, 6)]);
  const w2 = new InMemoryNavWorld(
    [
      { id: 1, y: 0, ring: [[0, -6], [16, -6], [16, 6], [0, 6]] },
      { id: 2, y: 0, ring: [[24, -6], [40, -6], [40, 6], [24, 6]] },
    ],
    [],
  );
  const reg = new AffordanceRegistry();
  reg.add(
    createJumpAffordance({
      id: 'gap',
      launch: { x: 15, z: 0 },
      entryRadius: 3,
      land: { x: 25, z: 0, heading: 0, speed: 0, t: 0 },
      duration: 1,
      cost: 1.5,
    }),
  );
  const env = new TimeAwareEnvironment(
    new VehicleEnvironment(w2, agent, lib, { goalRadius: 1.5, goalHeadingTol: Infinity }),
    { affordances: reg, affordanceRadius: 12 },
  );
  void world;
  const r = plan({ start, goal, environment: env, options: { maxExpansions: 500000 } }, Infinity);
  const dur = r.found ? r.path[r.path.length - 1]!.t : 0;
  let hop: [VehicleState, VehicleState] | null = null;
  const hi = r.nodes.findIndex((n) => n.edge?.kind === 'affordance');
  if (hi > 0) hop = [r.path[hi - 1]!, r.path[hi]!];
  return {
    bounds: { x0: 0, z0: -6, x1: 40, z1: 6 },
    islands: [
      [0, -6, 16, 6],
      [24, -6, 40, 6],
    ],
    path: r.path,
    duration: dur,
    start,
    goal,
    affordanceHop: hop,
    info: r.found
      ? `drive primitives cannot cross the gap — the planner uses a registered jump affordance`
      : 'no plan',
  };
}

export default function Dynamic() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [scn, setScn] = useState<Scenario>('moving');
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(true);
  const raf = useRef(0);
  const built = useMemo(() => build(scn), [scn]);

  useEffect(() => {
    setT(0);
  }, [scn]);

  useEffect(() => {
    if (!playing) return;
    let last = performance.now();
    const tick = () => {
      const now = performance.now();
      setT((prev) => {
        const next = prev + (now - last) / 1000;
        return next > built.duration ? 0 : next;
      });
      last = now;
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [playing, built]);

  useEffect(() => {
    const ctx = canvasRef.current?.getContext('2d');
    if (!ctx) return;
    const b = built.bounds;
    const W = (b.x1 - b.x0) * SCALE;
    const H = (b.z1 - b.z0) * SCALE;
    const px = (x: number, z: number): [number, number] => [
      (x - b.x0) * SCALE,
      (z - b.z0) * SCALE,
    ];
    ctx.fillStyle = PALETTE.bg;
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = PALETTE.floor;
    for (const [x0, z0, x1, z1] of built.islands) {
      const [ax, ay] = px(x0, z0);
      ctx.fillRect(ax, ay, (x1 - x0) * SCALE, (z1 - z0) * SCALE);
    }
    if (built.affordanceHop) {
      ctx.strokeStyle = '#ffd166';
      ctx.setLineDash([6, 6]);
      ctx.lineWidth = 2;
      const a = px(built.affordanceHop[0].x, built.affordanceHop[0].z);
      const c = px(built.affordanceHop[1].x, built.affordanceHop[1].z);
      ctx.beginPath();
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(c[0], c[1]);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    if (built.path.length) {
      ctx.strokeStyle = PALETTE.path;
      ctx.lineWidth = 3;
      ctx.beginPath();
      built.path.forEach((s, i) => {
        const p = px(s.x, s.z);
        i === 0 ? ctx.moveTo(p[0], p[1]) : ctx.lineTo(p[0], p[1]);
      });
      ctx.stroke();
    }
    const ghost = built.ghostAt?.(t);
    if (ghost) {
      const g = px(ghost.x, ghost.z);
      ctx.fillStyle = PALETTE.ghost;
      ctx.globalAlpha = 0.5;
      ctx.beginPath();
      ctx.arc(g[0], g[1], 2.5 * SCALE, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
    }
    for (const [pt, col] of [
      [built.start, PALETTE.start],
      [built.goal, PALETTE.goal],
    ] as const) {
      const p = px(pt.x, pt.z);
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.arc(p[0], p[1], 7, 0, Math.PI * 2);
      ctx.fill();
    }
    const pose = built.path.length ? planPoseAt(built.path, t) : null;
    if (pose) {
      const p = px(pose.x, pose.z);
      ctx.fillStyle = PALETTE.agent;
      ctx.beginPath();
      ctx.arc(p[0], p[1], 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = PALETTE.agent;
      ctx.beginPath();
      ctx.moveTo(p[0], p[1]);
      ctx.lineTo(p[0] + Math.cos(pose.heading) * 16, p[1] + Math.sin(pose.heading) * 16);
      ctx.stroke();
    }
  }, [built, t]);

  const b = built.bounds;
  const cw = (b.x1 - b.x0) * SCALE;
  const ch = (b.z1 - b.z0) * SCALE;
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
      <h1 style={{ fontSize: 18 }}>Time-aware + multi-agent</h1>
      <div style={{ display: 'flex', gap: 8, margin: '8px 0', flexWrap: 'wrap' }}>
        {(['moving', 'coop', 'jump'] as Scenario[]).map((s) => (
          <button
            key={s}
            onClick={() => setScn(s)}
            style={{
              background: s === scn ? '#2a3550' : '#161a22',
              color: '#cdd3de',
              border: '1px solid #2a2f3a',
              borderRadius: 6,
              padding: '6px 12px',
              cursor: 'pointer',
            }}
          >
            {s === 'moving' ? 'Moving obstacle' : s === 'coop' ? 'Two NPCs' : 'Jump affordance'}
          </button>
        ))}
      </div>
      <canvas
        ref={canvasRef}
        width={cw}
        height={ch}
        style={{
          borderRadius: 8,
          display: 'block',
          width: '100%',
          maxWidth: cw,
          height: 'auto',
          aspectRatio: `${cw} / ${ch}`,
        }}
      />
      <div
        style={{
          display: 'flex',
          gap: 16,
          alignItems: 'center',
          marginTop: 10,
          flexWrap: 'wrap',
        }}
      >
        <button
          onClick={() => setPlaying((p) => !p)}
          style={{ background: '#161a22', color: '#cdd3de', border: '1px solid #2a2f3a', borderRadius: 6, padding: '6px 12px', cursor: 'pointer' }}
        >
          {playing ? '❚❚ pause' : '▶ play'}
        </button>
        <input
          type="range"
          min={0}
          max={Math.max(built.duration, 0.1)}
          step={0.01}
          value={Math.min(t, built.duration)}
          onChange={(e) => {
            setPlaying(false);
            setT(+e.target.value);
          }}
          style={{ flex: '1 1 200px' }}
        />
        <span style={{ width: 90 }}>t = {t.toFixed(2)} s</span>
      </div>
      <p style={{ opacity: 0.8 }}>{built.info}</p>
    </main>
  );
}
