'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { planReverse, PALETTE } from '../lib/scenarios';

const SCALE = 15;
const REV = '#ff6688';

export default function Reverse() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [reverseCost, setReverseCost] = useState(2);
  const [dirPenalty, setDirPenalty] = useState(0.5);
  const r = useMemo(
    () => planReverse({ reverseCost, dirChangePenalty: dirPenalty }),
    [reverseCost, dirPenalty],
  );

  useEffect(() => {
    const ctx = canvasRef.current?.getContext('2d');
    if (!ctx) return;
    const b = r.bounds;
    const W = (b.x1 - b.x0) * SCALE;
    const Hp = (b.z1 - b.z0) * SCALE;
    const px = (x: number, z: number): [number, number] => [
      (x - b.x0) * SCALE,
      (z - b.z0) * SCALE,
    ];
    ctx.fillStyle = PALETTE.bg;
    ctx.fillRect(0, 0, W, Hp);
    ctx.fillStyle = PALETTE.floor;
    ctx.fillRect(0, 0, W, Hp);

    for (const seg of r.segments) {
      ctx.strokeStyle = seg.reverse ? REV : PALETTE.path;
      ctx.lineWidth = 4;
      ctx.beginPath();
      const [a, c] = px(seg.from.x, seg.from.z);
      const [d, e] = px(seg.to.x, seg.to.z);
      ctx.moveTo(a, c);
      ctx.lineTo(d, e);
      ctx.stroke();
    }

    for (const [p, col] of [
      [r.start, PALETTE.start],
      [r.goal, PALETTE.goal],
    ] as const) {
      const [a, c] = px(p.x, p.z);
      ctx.strokeStyle = col;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(a, c);
      const [ha, hb] = px(
        p.x + Math.cos(p.heading) * 3,
        p.z + Math.sin(p.heading) * 3,
      );
      ctx.lineTo(ha, hb);
      ctx.stroke();
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.arc(a, c, 8, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [r]);

  const b = r.bounds;
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
      <h1 style={{ fontSize: 18 }}>Reverse maneuvers</h1>
      <p style={{ opacity: 0.75, marginTop: 0 }}>
        A corridor too narrow to turn around in, with the goal behind the start
        at the same heading. The only feasible plan is a reverse maneuver —
        IGHA* produces it with no special-case logic. Reverse segments are red.
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
      <div style={{ display: 'flex', gap: 32, marginTop: 14, flexWrap: 'wrap' }}>
        <label>
          reverse cost ×{reverseCost.toFixed(1)}
          <br />
          <input
            type="range"
            min={1}
            max={10}
            step={0.5}
            value={reverseCost}
            onChange={(e) => setReverseCost(+e.target.value)}
          />
        </label>
        <label>
          direction-change penalty {dirPenalty.toFixed(1)} s
          <br />
          <input
            type="range"
            min={0}
            max={3}
            step={0.25}
            value={dirPenalty}
            onChange={(e) => setDirPenalty(+e.target.value)}
          />
        </label>
      </div>
      <p style={{ opacity: 0.8 }}>
        {r.found
          ? `plan found · cost ${r.cost.toFixed(2)} · ${r.reverseCount} reverse segment(s) of ${r.segments.length}`
          : 'no plan'}
      </p>
    </main>
  );
}
