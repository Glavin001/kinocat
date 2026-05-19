'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { plan } from 'kinocat/planner';
import { InMemoryNavWorld, VehicleEnvironment } from 'kinocat/environment';
import type { VehicleState } from 'kinocat/agent';
import { buildVehicle, PALETTE } from '../lib/vehicle';

const BW = 44;
const BH = 22; // world z in [-11, 11]
const SCALE = 18;
const W = BW * SCALE;
const H = BH * SCALE;
const OB = 2.4; // obstacle half-size

type Obstacle = { x: number; z: number };
type Drag =
  | { kind: 'start' }
  | { kind: 'goal' }
  | { kind: 'obstacle'; i: number }
  | null;

const { agent, lib } = buildVehicle();

function toWorld(px: number, py: number): [number, number] {
  return [px / SCALE, py / SCALE - BH / 2];
}
function toPx(x: number, z: number): [number, number] {
  return [x * SCALE, (z + BH / 2) * SCALE];
}

export default function Playground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [start, setStart] = useState<VehicleState>({ x: 4, z: 0, heading: 0, speed: 0, t: 0 });
  const [goal, setGoal] = useState<VehicleState>({ x: 40, z: 0, heading: 0, speed: 0, t: 0 });
  const [obstacles, setObstacles] = useState<Obstacle[]>([{ x: 22, z: 0 }]);
  const [deadline, setDeadline] = useState(40);
  const [reverseCost, setReverseCost] = useState(2);
  const [info, setInfo] = useState('');
  const drag = useRef<Drag>(null);
  const raf = useRef(0);

  const replanAndDraw = useCallback(() => {
    const ctx = canvasRef.current?.getContext('2d');
    if (!ctx) return;
    const obPolys = obstacles.map((o) => [
      [o.x - OB, o.z - OB],
      [o.x + OB, o.z - OB],
      [o.x + OB, o.z + OB],
      [o.x - OB, o.z + OB],
    ] as [number, number][]);
    const world = new InMemoryNavWorld(
      [{ id: 1, y: 0, ring: [[0, -BH / 2], [BW, -BH / 2], [BW, BH / 2], [0, BH / 2]] }],
      obPolys,
    );
    const env = new VehicleEnvironment(
      world,
      { ...agent, reverseCostMultiplier: reverseCost },
      lib,
      { goalRadius: 1.5, goalHeadingTol: Infinity },
    );
    const t0 = performance.now();
    const r = plan(
      { start, goal, environment: env, options: { maxExpansions: 120000 } },
      deadline,
    );
    const ms = (performance.now() - t0).toFixed(1);

    ctx.fillStyle = PALETTE.bg;
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = PALETTE.floor;
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = PALETTE.obstacle;
    for (const o of obstacles) {
      const [px, py] = toPx(o.x - OB, o.z - OB);
      ctx.fillRect(px, py, OB * 2 * SCALE, OB * 2 * SCALE);
    }
    if (r.found) {
      ctx.strokeStyle = PALETTE.path;
      ctx.lineWidth = 3;
      ctx.beginPath();
      r.path.forEach((s, i) => {
        const [px, py] = toPx(s.x, s.z);
        i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      });
      ctx.stroke();
      // mark reverse segments
      ctx.strokeStyle = '#ff6688';
      ctx.lineWidth = 3;
      ctx.beginPath();
      r.nodes.forEach((n, i) => {
        if (i === 0 || n.edge?.kind !== 'drive-reverse') return;
        const a = toPx(r.path[i - 1]!.x, r.path[i - 1]!.z);
        const b = toPx(r.path[i]!.x, r.path[i]!.z);
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
      });
      ctx.stroke();
    }
    for (const [pt, c] of [
      [start, PALETTE.start],
      [goal, PALETTE.goal],
    ] as const) {
      const [px, py] = toPx(pt.x, pt.z);
      ctx.fillStyle = c;
      ctx.beginPath();
      ctx.arc(px, py, 8, 0, Math.PI * 2);
      ctx.fill();
    }
    setInfo(
      r.found
        ? `found in ${ms} ms · cost ${r.cost.toFixed(2)} · ${r.stats.expansions} expansions · ${r.path.length} states${r.stats.deadlineHit ? ' · deadline hit' : ''}`
        : `no plan (${ms} ms, ${r.stats.expansions} expansions) — try a larger deadline or move obstacles`,
    );
  }, [start, goal, obstacles, deadline, reverseCost]);

  useEffect(() => {
    replanAndDraw();
  }, [replanAndDraw]);

  const evtWorld = (e: React.PointerEvent): [number, number] => {
    const c = canvasRef.current!;
    const rect = c.getBoundingClientRect();
    const sx = c.width / rect.width;
    const sy = c.height / rect.height;
    return toWorld((e.clientX - rect.left) * sx, (e.clientY - rect.top) * sy);
  };

  const pick = (wx: number, wz: number): Drag => {
    if (Math.hypot(wx - start.x, wz - start.z) < 1.6) return { kind: 'start' };
    if (Math.hypot(wx - goal.x, wz - goal.z) < 1.2) return { kind: 'goal' };
    for (let i = 0; i < obstacles.length; i++) {
      if (Math.abs(wx - obstacles[i]!.x) < OB && Math.abs(wz - obstacles[i]!.z) < OB)
        return { kind: 'obstacle', i };
    }
    return null;
  };

  const onDown = (e: React.PointerEvent) => {
    const [wx, wz] = evtWorld(e);
    const hit = pick(wx, wz);
    if (e.shiftKey && hit?.kind === 'obstacle') {
      setObstacles((o) => o.filter((_, i) => i !== hit.i));
      return;
    }
    if (!hit) {
      setObstacles((o) => [...o, { x: wx, z: wz }]);
      return;
    }
    drag.current = hit;
    e.currentTarget.setPointerCapture(e.pointerId);
  };

  const onMove = (e: React.PointerEvent) => {
    if (!drag.current) return;
    const [wx, wz] = evtWorld(e);
    const d = drag.current;
    if (d.kind === 'start') setStart((s) => ({ ...s, x: wx, z: wz }));
    else if (d.kind === 'goal') setGoal((s) => ({ ...s, x: wx, z: wz }));
    else
      setObstacles((o) => o.map((ob, i) => (i === d.i ? { x: wx, z: wz } : ob)));
    cancelAnimationFrame(raf.current);
    raf.current = requestAnimationFrame(replanAndDraw);
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
      <a href="/" style={{ color: '#7fd6ff' }}>← demos</a>
      <h1 style={{ fontSize: 18 }}>Interactive 2D playground</h1>
      <p style={{ opacity: 0.75, marginTop: 0 }}>
        Drag the green start / yellow goal. Tap empty space to drop an obstacle;
        drag obstacles to move them; shift-tap (or use Clear) to remove. Red
        segments are reverse maneuvers.
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
      <div style={{ marginTop: 10 }}>
        <button
          onClick={() => setObstacles([])}
          style={{ background: '#161a22', color: '#cdd3de', border: '1px solid #2a2f3a', borderRadius: 6, padding: '6px 12px', cursor: 'pointer' }}
        >
          clear obstacles
        </button>
      </div>
      <div style={{ display: 'flex', gap: 32, marginTop: 12, flexWrap: 'wrap' }}>
        <label>
          anytime deadline: {deadline} ms
          <br />
          <input
            type="range"
            min={1}
            max={200}
            value={deadline}
            onChange={(e) => setDeadline(+e.target.value)}
          />
        </label>
        <label>
          reverse cost ×{reverseCost.toFixed(1)}
          <br />
          <input
            type="range"
            min={1}
            max={6}
            step={0.5}
            value={reverseCost}
            onChange={(e) => setReverseCost(+e.target.value)}
          />
        </label>
      </div>
      <p style={{ opacity: 0.8 }}>{info}</p>
    </main>
  );
}
