'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import type { MomentumHumanoidState } from 'kinocat/agent';
import { buildCrowd } from '../lib/crowd-scenario';
import { PALETTE } from '../lib/scenarios';

const SCALE = 30;

function poseAt(
  path: MomentumHumanoidState[],
  t: number,
): { x: number; z: number; heading: number; speed: number } {
  if (path.length === 0) return { x: 0, z: 0, heading: 0, speed: 0 };
  const first = path[0]!;
  const last = path[path.length - 1]!;
  const sp = (s: MomentumHumanoidState) => Math.hypot(s.vx, s.vz);
  if (t <= first.t)
    return { x: first.x, z: first.z, heading: first.heading, speed: sp(first) };
  if (t >= last.t)
    return { x: last.x, z: last.z, heading: last.heading, speed: sp(last) };
  for (let i = 0; i < path.length - 1; i++) {
    const a = path[i]!;
    const b = path[i + 1]!;
    if (t >= a.t && t <= b.t) {
      const span = b.t - a.t;
      const u = span > 1e-9 ? (t - a.t) / span : 0;
      return {
        x: a.x + (b.x - a.x) * u,
        z: a.z + (b.z - a.z) * u,
        heading: a.heading + (b.heading - a.heading) * u,
        speed: sp(a) + (sp(b) - sp(a)) * u,
      };
    }
  }
  return { x: last.x, z: last.z, heading: last.heading, speed: sp(last) };
}

/** Speed → color: walk = cool, sprint = hot. */
function speedColor(speed: number, maxSpeed: number): string {
  const u = Math.min(1, speed / maxSpeed);
  const hue = 190 - 150 * u; // 190 (cyan) → 40 (amber)
  return `hsl(${hue}, 85%, 60%)`;
}

export default function Crowd() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const scene = useMemo(() => buildCrowd(), []);
  const path = scene.result.path;
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
    const b = scene.bounds;
    const W = (b.x1 - b.x0) * SCALE;
    const Hp = (b.z1 - b.z0) * SCALE;
    const px = (x: number, z: number): [number, number] => [
      (x - b.x0) * SCALE,
      (z - b.z0) * SCALE,
    ];
    ctx.fillStyle = PALETTE.bg;
    ctx.fillRect(0, 0, W, Hp);
    ctx.fillStyle = PALETTE.floor;
    ctx.beginPath();
    scene.floor.forEach(([x, z], i) => {
      const [a, c] = px(x, z);
      i === 0 ? ctx.moveTo(a, c) : ctx.lineTo(a, c);
    });
    ctx.closePath();
    ctx.fill();

    // Committed path, colored by speed — the momentum story at a glance.
    for (let i = 1; i < path.length; i++) {
      const a = path[i - 1]!;
      const c2 = path[i]!;
      const [ax, az] = px(a.x, a.z);
      const [bx, bz] = px(c2.x, c2.z);
      ctx.strokeStyle = speedColor(
        Math.hypot(c2.vx, c2.vz),
        scene.agent.maxSpeed,
      );
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(ax, az);
      ctx.lineTo(bx, bz);
      ctx.stroke();
    }

    for (const [p, col] of [
      [scene.start, PALETTE.start],
      [scene.goal, PALETTE.goal],
    ] as const) {
      const [a, c] = px(p.x, p.z);
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.arc(a, c, 6, 0, Math.PI * 2);
      ctx.fill();
    }

    // Pedestrians at scrub time (their predicted positions — exactly what
    // the planner saw).
    for (const ped of scene.pedestrians) {
      const q = ped.predict(t);
      if (!q) continue;
      const [a, c] = px(q.x, q.z);
      ctx.fillStyle = '#e0705a';
      ctx.beginPath();
      ctx.arc(a, c, ped.radius * SCALE, 0, Math.PI * 2);
      ctx.fill();
    }

    // The runner.
    const pose = poseAt(path, t);
    const [ax, ay] = px(pose.x, pose.z);
    ctx.fillStyle = speedColor(pose.speed, scene.agent.maxSpeed);
    ctx.beginPath();
    ctx.arc(ax, ay, scene.agent.radius * SCALE, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(
      ax + Math.cos(pose.heading) * 16,
      ay + Math.sin(pose.heading) * 16,
    );
    ctx.stroke();
  }, [scene, path, t]);

  const b = scene.bounds;
  const cw = (b.x1 - b.x0) * SCALE;
  const ch = (b.z1 - b.z0) * SCALE;
  const pose = poseAt(path, t);

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
      <h1 style={{ fontSize: 18 }}>Momentum runner in a crowd</h1>
      <p style={{ opacity: 0.75, marginTop: 0 }}>
        An <em>inertial</em> person — launch/brake limits, a strafe cap, turn
        rate that degrades at sprint — crosses a plaza while pedestrians cut
        its line at exactly the wrong times. Planned in space-time by the same
        IGHA* core (<code>MomentumHumanoidEnvironment</code> +{' '}
        <code>TimeAwareEnvironment</code>). Path color = speed: cyan walk →
        amber sprint.
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
        <span style={{ width: 150 }}>
          t = {t.toFixed(2)} · {pose.speed.toFixed(1)} m/s
        </span>
      </div>
      <p style={{ opacity: 0.8 }}>
        {scene.result.found
          ? `plan found · ${scene.result.cost.toFixed(1)} s · ${scene.result.stats.expansions.toLocaleString()} expansions`
          : 'no plan'}
      </p>
    </main>
  );
}
