'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { buildAnytime, PALETTE } from '../lib/scenarios';

const SCALE = 17;

export default function Anytime() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const built = useMemo(() => buildAnytime(), []);
  const [i, setI] = useState(built.steps.length - 1);
  const [playing, setPlaying] = useState(false);
  const raf = useRef(0);

  useEffect(() => {
    if (!playing) return;
    let last = performance.now();
    const tick = () => {
      const now = performance.now();
      if (now - last > 900) {
        last = now;
        setI((p) => (p + 1) % built.steps.length);
      }
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
    ctx.fillStyle = PALETTE.obstacle;
    for (const o of built.obstacles) {
      const [ax, ay] = px(o.x - o.hx, o.z - o.hz);
      ctx.fillRect(ax, ay, o.hx * 2 * SCALE, o.hz * 2 * SCALE);
    }
    const step = built.steps[i]!;
    if (step.found && step.path.length > 1) {
      ctx.strokeStyle = PALETTE.path;
      ctx.lineWidth = 3;
      ctx.beginPath();
      step.path.forEach((s, k) => {
        const [a, c] = px(s.x, s.z);
        k === 0 ? ctx.moveTo(a, c) : ctx.lineTo(a, c);
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
      ctx.arc(a, c, 8, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [built, i]);

  const b = built.bounds;
  const cw = (b.x1 - b.x0) * SCALE;
  const ch = (b.z1 - b.z0) * SCALE;
  const step = built.steps[i]!;

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
      <h1 style={{ fontSize: 18 }}>Anytime planning (budget sweep)</h1>
      <p style={{ opacity: 0.75, marginTop: 0 }}>
        The same query solved at growing expansion budgets. IGHA* is anytime:
        it always returns the best plan found so far, so a tight budget yields a
        rough (or no) plan and a generous one tightens it. The NPC always has a
        usable plan.
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
          max={built.steps.length - 1}
          step={1}
          value={i}
          onChange={(e) => {
            setPlaying(false);
            setI(+e.target.value);
          }}
          style={{ flex: '1 1 200px' }}
        />
        <span style={{ width: 110 }}>
          budget {i + 1}/{built.steps.length}
        </span>
      </div>
      <p style={{ opacity: 0.8 }}>
        {step.budget.toLocaleString()} expansion budget ·{' '}
        {step.found
          ? `plan found · cost ${step.cost.toFixed(2)} · ${step.expansions.toLocaleString()} expansions used`
          : `no plan within budget (${step.expansions.toLocaleString()} expansions) — keep the previous plan, replan next tick`}
      </p>
    </main>
  );
}
