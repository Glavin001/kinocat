'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { compareCurves, PALETTE } from '../lib/scenarios';

const B = { x0: -3, z0: -15, x1: 33, z1: 15 };
const SCALE = 16;
const W = (B.x1 - B.x0) * SCALE;
const H = (B.z1 - B.z0) * SCALE;
const RS_COLOR = '#ffd166';

interface Pt {
  x: number;
  z: number;
  heading: number;
}
type Handle = 'startPos' | 'startDir' | 'goalPos' | 'goalDir' | null;

function px(x: number, z: number): [number, number] {
  return [(x - B.x0) * SCALE, (z - B.z0) * SCALE];
}
function toWorld(cx: number, cz: number): [number, number] {
  return [cx / SCALE + B.x0, cz / SCALE + B.z0];
}

export default function Curves() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [start, setStart] = useState<Pt>({ x: 3, z: 0, heading: 0 });
  const [goal, setGoal] = useState<Pt>({ x: 24, z: 5, heading: Math.PI / 2 });
  const [radius, setRadius] = useState(4);
  const [info, setInfo] = useState('');
  const drag = useRef<Handle>(null);

  const draw = useCallback(() => {
    const ctx = canvasRef.current?.getContext('2d');
    if (!ctx) return;
    const c = compareCurves({
      sx: start.x,
      sz: start.z,
      sHeading: start.heading,
      gx: goal.x,
      gz: goal.z,
      gHeading: goal.heading,
      radius,
    });

    ctx.fillStyle = PALETTE.floor;
    ctx.fillRect(0, 0, W, H);

    const poly = (pts: [number, number][], color: string, width: number) => {
      if (pts.length < 2) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      pts.forEach(([x, z], i) => {
        const [a, b] = px(x, z);
        i === 0 ? ctx.moveTo(a, b) : ctx.lineTo(a, b);
      });
      ctx.stroke();
    };
    poly(c.reedsShepp.samples, RS_COLOR, 5);
    poly(c.dubins.samples, PALETTE.path, 2.5);

    for (const [p, col] of [
      [start, PALETTE.start],
      [goal, PALETTE.goal],
    ] as const) {
      const [a, b] = px(p.x, p.z);
      ctx.strokeStyle = col;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(a, b);
      const [ha, hb] = px(
        p.x + Math.cos(p.heading) * 4,
        p.z + Math.sin(p.heading) * 4,
      );
      ctx.lineTo(ha, hb);
      ctx.stroke();
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.arc(a, b, 8, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(ha, hb, 5, 0, Math.PI * 2);
      ctx.fill();
    }

    const segTxt = c.reedsShepp.segments
      .map((s) => `${s.steer}${s.gear < 0 ? '−' : '+'}`)
      .join(' ');
    setInfo(
      `Dubins (forward-only): ${c.dubins.word || '—'} · len ${c.dubins.length.toFixed(2)}    │    ` +
        `Reeds-Shepp (fwd+rev): ${segTxt || '—'} · len ${c.reedsShepp.length.toFixed(2)}`,
    );
  }, [start, goal, radius]);

  useEffect(() => {
    draw();
  }, [draw]);

  const evt = (e: React.PointerEvent): [number, number] => {
    const el = canvasRef.current!;
    const r = el.getBoundingClientRect();
    return toWorld(
      ((e.clientX - r.left) * el.width) / r.width,
      ((e.clientY - r.top) * el.height) / r.height,
    );
  };
  const near = (wx: number, wz: number, p: Pt) =>
    Math.hypot(wx - p.x, wz - p.z) < 1.6;
  const nearDir = (wx: number, wz: number, p: Pt) =>
    Math.hypot(
      wx - (p.x + Math.cos(p.heading) * 4),
      wz - (p.z + Math.sin(p.heading) * 4),
    ) < 1.4;

  const onDown = (e: React.PointerEvent) => {
    const [wx, wz] = evt(e);
    drag.current = nearDir(wx, wz, start)
      ? 'startDir'
      : nearDir(wx, wz, goal)
        ? 'goalDir'
        : near(wx, wz, start)
          ? 'startPos'
          : near(wx, wz, goal)
            ? 'goalPos'
            : null;
    if (drag.current) e.currentTarget.setPointerCapture(e.pointerId);
  };
  const onMove = (e: React.PointerEvent) => {
    if (!drag.current) return;
    const [wx, wz] = evt(e);
    const d = drag.current;
    if (d === 'startPos') setStart((s) => ({ ...s, x: wx, z: wz }));
    else if (d === 'goalPos') setGoal((s) => ({ ...s, x: wx, z: wz }));
    else if (d === 'startDir')
      setStart((s) => ({ ...s, heading: Math.atan2(wz - s.z, wx - s.x) }));
    else setGoal((s) => ({ ...s, heading: Math.atan2(wz - s.z, wx - s.x) }));
  };
  const onUp = (e: React.PointerEvent) => {
    drag.current = null;
    if (e.currentTarget.hasPointerCapture(e.pointerId))
      e.currentTarget.releasePointerCapture(e.pointerId);
  };

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
      <h1 style={{ fontSize: 18 }}>Reeds-Shepp vs Dubins curves</h1>
      <p style={{ opacity: 0.75, marginTop: 0 }}>
        The analytical car curves <code>kinocat/curves</code> ships — the
        Hybrid&nbsp;A* heuristic and shot-to-goal. Dubins is forward-only;
        Reeds-Shepp adds reverse (cusps), so it is never longer. Drag the dots
        to move; drag the small handles to rotate the heading.
      </p>
      <canvas
        ref={canvasRef}
        width={W}
        height={H}
        style={{
          borderRadius: 8,
          cursor: 'crosshair',
          touchAction: 'none',
          display: 'block',
          width: '100%',
          maxWidth: W,
          height: 'auto',
          aspectRatio: `${W} / ${H}`,
        }}
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onPointerCancel={onUp}
      />
      <div style={{ marginTop: 12 }}>
        <label>
          turn radius {radius.toFixed(1)} m
          <br />
          <input
            type="range"
            min={1}
            max={10}
            step={0.5}
            value={radius}
            onChange={(e) => setRadius(+e.target.value)}
          />
        </label>
      </div>
      <p style={{ opacity: 0.8 }}>{info}</p>
    </main>
  );
}
