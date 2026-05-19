'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import type { HumanoidState } from 'kinocat/agent';
import { buildHumanoid, PALETTE } from '../lib/scenarios';

const SCALE = 21;

function walkPoseAt(
  path: HumanoidState[],
  t: number,
): { x: number; z: number; heading: number } {
  if (path.length === 0) return { x: 0, z: 0, heading: 0 };
  const first = path[0]!;
  const last = path[path.length - 1]!;
  if (t <= first.t) return { x: first.x, z: first.z, heading: first.heading };
  if (t >= last.t) return { x: last.x, z: last.z, heading: last.heading };
  for (let i = 0; i < path.length - 1; i++) {
    const a = path[i]!;
    const b = path[i + 1]!;
    if (t >= a.t && t <= b.t) {
      const span = b.t - a.t;
      const u = span > 1e-9 ? (t - a.t) / span : 0;
      return {
        x: a.x + (b.x - a.x) * u,
        z: a.z + (b.z - a.z) * u,
        heading: Math.atan2(b.z - a.z, b.x - a.x),
      };
    }
  }
  return { x: last.x, z: last.z, heading: last.heading };
}

export default function Humanoid() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const built = useMemo(() => buildHumanoid(), []);
  const path = built.humanoid.path;
  const duration = path.length ? path[path.length - 1]!.t : 0.1;
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(true);
  const raf = useRef(0);

  useEffect(() => {
    if (!playing) return;
    let last = performance.now();
    const tick = () => {
      const now = performance.now();
      setT((p) => {
        const n = p + (now - last) / 1000;
        return n > duration ? 0 : n;
      });
      last = now;
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [playing, duration]);

  useEffect(() => {
    const ctx = canvasRef.current?.getContext('2d');
    if (!ctx) return;
    const b = built.bounds;
    const W = (b.x1 - b.x0) * SCALE;
    const Hp = (b.z1 - b.z0) * SCALE;
    const px = (x: number, z: number): [number, number] => [
      (x - b.x0) * SCALE,
      (z - b.z0) * SCALE,
    ];
    ctx.fillStyle = PALETTE.bg;
    ctx.fillRect(0, 0, W, Hp);
    ctx.fillStyle = PALETTE.floor;
    for (const poly of built.corridor) {
      ctx.beginPath();
      poly.forEach(([x, z], i) => {
        const [a, c] = px(x, z);
        i === 0 ? ctx.moveTo(a, c) : ctx.lineTo(a, c);
      });
      ctx.closePath();
      ctx.fill();
    }

    if (path.length > 1) {
      ctx.strokeStyle = PALETTE.path;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      path.forEach((s, i) => {
        const [a, c] = px(s.x, s.z);
        i === 0 ? ctx.moveTo(a, c) : ctx.lineTo(a, c);
      });
      ctx.stroke();
    }

    for (const [p, col] of [
      [built.start, PALETTE.start],
      [built.goal, PALETTE.goal],
    ] as const) {
      const [a, c] = px(p.x, p.z);
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.arc(a, c, 7, 0, Math.PI * 2);
      ctx.fill();
    }

    const pose = walkPoseAt(path, t);
    const [ax, ay] = px(pose.x, pose.z);
    ctx.fillStyle = PALETTE.agent;
    ctx.beginPath();
    ctx.arc(ax, ay, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = PALETTE.agent;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(
      ax + Math.cos(pose.heading) * 14,
      ay + Math.sin(pose.heading) * 14,
    );
    ctx.stroke();
  }, [built, path, t]);

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
      <a href="/" style={{ color: '#7fd6ff' }}>
        ← demos
      </a>
      <h1 style={{ fontSize: 18 }}>Humanoid vs. vehicle</h1>
      <p style={{ opacity: 0.75, marginTop: 0 }}>
        The omnidirectional humanoid threads a tight L-corridor using the same
        IGHA* core — only the <code>Environment</code> differs (no turn-radius
        constraint, no inertial speed dimension).
      </p>
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
          style={{
            background: '#161a22',
            color: '#cdd3de',
            border: '1px solid #2a2f3a',
            borderRadius: 6,
            padding: '6px 12px',
            cursor: 'pointer',
          }}
        >
          {playing ? '❚❚ pause' : '▶ play'}
        </button>
        <input
          type="range"
          min={0}
          max={Math.max(duration, 0.1)}
          step={0.01}
          value={Math.min(t, duration)}
          onChange={(e) => {
            setPlaying(false);
            setT(+e.target.value);
          }}
          style={{ flex: '1 1 200px' }}
        />
        <span style={{ width: 70 }}>t = {t.toFixed(2)}</span>
      </div>
      <p style={{ opacity: 0.8 }}>
        humanoid: {built.humanoid.found ? 'plan found' : 'no plan'} ·{' '}
        vehicle (turn-radius constrained):{' '}
        {built.vehicle.found
          ? 'plan found'
          : 'no feasible plan — cannot turn the corner'}
      </p>
    </main>
  );
}
