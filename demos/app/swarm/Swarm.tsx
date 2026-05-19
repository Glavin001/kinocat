'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { buildSwarm, PALETTE } from '../lib/scenarios';

const SCALE = 11;

export default function Swarm() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [count, setCount] = useState(4);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(true);
  const raf = useRef(0);
  const built = useMemo(
    () => buildSwarm({ agents: count, rounds: 4 }),
    [count],
  );
  const colors = useMemo(
    () =>
      built.agents.map(
        (_, i) => `hsl(${Math.round((360 * i) / built.agents.length)} 80% 65%)`,
      ),
    [built],
  );

  useEffect(() => {
    setT(0);
  }, [built]);

  useEffect(() => {
    if (!playing) return;
    let last = performance.now();
    const tick = () => {
      const now = performance.now();
      setT((p) => {
        const n = p + (now - last) / 1000;
        return n > built.duration ? 0 : n;
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
    const Hp = (b.z1 - b.z0) * SCALE;
    const px = (x: number, z: number): [number, number] => [
      (x - b.x0) * SCALE,
      (z - b.z0) * SCALE,
    ];
    ctx.fillStyle = PALETTE.floor;
    ctx.fillRect(0, 0, W, Hp);

    built.agents.forEach((ag, i) => {
      const col = colors[i]!;
      if (ag.path.length > 1) {
        ctx.strokeStyle = col;
        ctx.globalAlpha = 0.35;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ag.path.forEach((s, k) => {
          const [a, c] = px(s.x, s.z);
          k === 0 ? ctx.moveTo(a, c) : ctx.lineTo(a, c);
        });
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
      const [gx, gy] = px(ag.goal.x, ag.goal.z);
      ctx.strokeStyle = col;
      ctx.globalAlpha = 0.5;
      ctx.beginPath();
      ctx.arc(gx, gy, 6, 0, Math.PI * 2);
      ctx.stroke();
      ctx.globalAlpha = 1;

      const pose = built.registry.predictNPC(ag.id)(t) ?? ag.start;
      const [ax, ay] = px(pose.x, pose.z);
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.arc(ax, ay, 7, 0, Math.PI * 2);
      ctx.fill();
    });
  }, [built, colors, t]);

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
      <h1 style={{ fontSize: 18 }}>Multi-agent coordination (swarm)</h1>
      <p style={{ opacity: 0.75, marginTop: 0 }}>
        NPCs on a ring each drive to the antipodal point. Every round each agent
        replans treating the others&apos; published plans (the plan registry) as
        moving obstacles, then republishes. Cooperative avoidance is emergent —
        no negotiation protocol. (Changing the count re-runs the coordination,
        which takes a moment.)
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
        <label>
          agents {count}
          <input
            type="range"
            min={2}
            max={6}
            step={1}
            value={count}
            onChange={(e) => setCount(+e.target.value)}
            style={{ marginLeft: 8 }}
          />
        </label>
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
          style={{ flex: '1 1 160px' }}
        />
        <span style={{ width: 70 }}>t = {t.toFixed(2)}</span>
      </div>
      <p style={{ opacity: 0.8 }}>
        {built.agents.length} agents · {built.rounds} coordination rounds ·{' '}
        {built.reached}/{built.agents.length} reached their goal
      </p>
    </main>
  );
}
