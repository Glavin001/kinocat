'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { buildPrimitiveFan, PALETTE } from '../lib/scenarios';

const CW = 760;
const CH = 460;
const FWD = '#7fffa8';
const REV = '#ff6688';

export default function Primitives() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [minTurnRadius, setR] = useState(3);
  const [duration, setDur] = useState(0.5);
  const [startSpeed, setSp] = useState(0);
  const fan = useMemo(
    () => buildPrimitiveFan({ minTurnRadius, duration, startSpeed }),
    [minTurnRadius, duration, startSpeed],
  );

  useEffect(() => {
    const ctx = canvasRef.current?.getContext('2d');
    if (!ctx) return;
    ctx.fillStyle = PALETTE.floor;
    ctx.fillRect(0, 0, CW, CH);

    let maxR = 1;
    for (const p of fan.primitives)
      for (const s of p.sweep) maxR = Math.max(maxR, Math.abs(s.x), Math.abs(s.z));
    const scale = Math.min(CW, CH) / (2 * (maxR + 1.5));
    const ox = CW * 0.32;
    const oy = CH / 2;
    // local +x (forward) → screen right, local +z → screen down
    const px = (x: number, z: number): [number, number] => [
      ox + x * scale,
      oy + z * scale,
    ];

    ctx.strokeStyle = '#2a2f3a';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, oy);
    ctx.lineTo(CW, oy);
    ctx.moveTo(ox, 0);
    ctx.lineTo(ox, CH);
    ctx.stroke();

    // footprint at origin (heading 0): half-extents 1.2 × 0.6
    ctx.strokeStyle = '#5566aa';
    ctx.lineWidth = 1.5;
    const fp: [number, number][] = [
      [1.2, 0.6],
      [-1.2, 0.6],
      [-1.2, -0.6],
      [1.2, -0.6],
    ];
    ctx.beginPath();
    fp.forEach(([x, z], i) => {
      const [a, b] = px(x, z);
      i === 0 ? ctx.moveTo(a, b) : ctx.lineTo(a, b);
    });
    ctx.closePath();
    ctx.stroke();

    for (const p of fan.primitives) {
      ctx.strokeStyle = p.reverse ? REV : FWD;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      p.sweep.forEach((s, i) => {
        const [a, b] = px(s.x, s.z);
        i === 0 ? ctx.moveTo(a, b) : ctx.lineTo(a, b);
      });
      ctx.stroke();
      const e = px(p.end.dx, p.end.dz);
      ctx.fillStyle = p.reverse ? REV : FWD;
      ctx.beginPath();
      ctx.arc(e[0], e[1], 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [fan]);

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
      <h1 style={{ fontSize: 18 }}>Motion-primitive characterization</h1>
      <p style={{ opacity: 0.75, marginTop: 0 }}>
        The planner&apos;s entire action set is this swept set of pre-character&shy;ized
        primitives — produced by rolling a forward model across a control grid.
        Green = forward, red = reverse, dots = end states, blue = the body
        footprint. Tune the kinematics and watch the fan re-characterize live.
      </p>
      <canvas
        ref={canvasRef}
        width={CW}
        height={CH}
        style={{
          borderRadius: 8,
          display: 'block',
          width: '100%',
          maxWidth: CW,
          height: 'auto',
          aspectRatio: `${CW} / ${CH}`,
        }}
      />
      <div style={{ display: 'flex', gap: 28, marginTop: 14, flexWrap: 'wrap' }}>
        <label>
          min turn radius {minTurnRadius.toFixed(1)} m
          <br />
          <input
            type="range"
            min={1}
            max={8}
            step={0.5}
            value={minTurnRadius}
            onChange={(e) => setR(+e.target.value)}
          />
        </label>
        <label>
          primitive duration {duration.toFixed(2)} s
          <br />
          <input
            type="range"
            min={0.2}
            max={1.5}
            step={0.1}
            value={duration}
            onChange={(e) => setDur(+e.target.value)}
          />
        </label>
        <label>
          start speed {startSpeed.toFixed(1)} m/s
          <br />
          <input
            type="range"
            min={0}
            max={8}
            step={1}
            value={startSpeed}
            onChange={(e) => setSp(+e.target.value)}
          />
        </label>
      </div>
      <p style={{ opacity: 0.8 }}>
        {fan.count} primitives · {fan.primitives.filter((p) => p.reverse).length}{' '}
        reverse · {fan.primitives.filter((p) => !p.reverse).length} forward
      </p>
    </main>
  );
}
