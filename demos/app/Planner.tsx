'use client';

import { useEffect, useRef, useState } from 'react';
import { plan } from 'kinocat/planner';
import {
  InMemoryNavWorld,
  VehicleEnvironment,
  type NavPolygon,
} from 'kinocat/environment';
import { defaultVehicleAgent, kinematicForwardSim } from 'kinocat/agent';
import { characterizeVehicle } from 'kinocat/primitives';
import type { VehicleState } from 'kinocat/agent';

const W = 720;
const H = 420;
const SCALE = 18; // world units → px

function rect(id: number, x0: number, z0: number, x1: number, z1: number): NavPolygon {
  return { id, y: 0, ring: [[x0, z0], [x1, z0], [x1, z1], [x0, z1]] };
}

export default function Planner() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [info, setInfo] = useState('planning…');

  useEffect(() => {
    const agent = defaultVehicleAgent({
      minTurnRadius: 3,
      maxSpeed: 8,
      footprint: [
        [1.2, 0.6],
        [-1.2, 0.6],
        [-1.2, -0.6],
        [1.2, -0.6],
      ],
    });
    const k = 1 / agent.minTurnRadius;
    const lib = characterizeVehicle({
      forwardSim: kinematicForwardSim(agent),
      controlSets: [[0, 6], [k, 6], [-k, 6], [k / 2, 6], [-k / 2, 6]],
      duration: 0.5,
      substeps: 6,
      startSpeeds: [0],
    });
    const obstacle = [
      [16, -3],
      [22, -3],
      [22, 3],
      [16, 3],
    ] as [number, number][];
    const world = new InMemoryNavWorld([rect(1, 0, -10, 38, 10)], [obstacle]);
    const env = new VehicleEnvironment(world, agent, lib, {
      goalRadius: 1.5,
      goalHeadingTol: Infinity,
    });
    const start: VehicleState = { x: 3, z: 0, heading: 0, speed: 0, t: 0 };
    const goal: VehicleState = { x: 34, z: 0, heading: 0, speed: 0, t: 0 };

    const t0 = performance.now();
    const r = plan(
      { start, goal, environment: env, options: { maxExpansions: 400000 } },
      Infinity,
    );
    const ms = (performance.now() - t0).toFixed(1);

    const ctx = canvasRef.current?.getContext('2d');
    if (!ctx) return;
    const toPx = (x: number, z: number): [number, number] => [
      40 + x * SCALE,
      H / 2 + z * SCALE,
    ];
    ctx.fillStyle = '#0b0b0f';
    ctx.fillRect(0, 0, W, H);
    // walkable
    ctx.fillStyle = '#161a22';
    const [wx, wy] = toPx(0, -10);
    ctx.fillRect(wx, wy, 38 * SCALE, 20 * SCALE);
    // obstacle
    ctx.fillStyle = '#5a2230';
    ctx.beginPath();
    obstacle.forEach(([x, z], i) => {
      const [px, py] = toPx(x, z);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.closePath();
    ctx.fill();
    // path
    if (r.found) {
      ctx.strokeStyle = '#44ddff';
      ctx.lineWidth = 3;
      ctx.beginPath();
      r.path.forEach((s, i) => {
        const [px, py] = toPx(s.x, s.z);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();
    }
    for (const [pt, color] of [
      [start, '#55ff88'],
      [goal, '#ffcc33'],
    ] as const) {
      const [px, py] = toPx(pt.x, pt.z);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(px, py, 6, 0, Math.PI * 2);
      ctx.fill();
    }
    setInfo(
      r.found
        ? `plan found in ${ms} ms · cost ${r.cost.toFixed(2)} · ${r.stats.expansions} expansions · ${r.path.length} states`
        : `no plan (${ms} ms, ${r.stats.expansions} expansions)`,
    );
  }, []);

  return (
    <main style={{ color: '#cdd3de', fontFamily: 'ui-monospace, monospace', padding: 24 }}>
      <h1 style={{ fontSize: 18 }}>kinocat — vehicle planner playground</h1>
      <p style={{ opacity: 0.8 }}>
        IGHA* time-extended kinodynamic plan around an obstacle, computed in the
        browser via the <code>kinocat</code> package.
      </p>
      <canvas ref={canvasRef} width={W} height={H} style={{ borderRadius: 8 }} />
      <p style={{ opacity: 0.8 }}>{info}</p>
    </main>
  );
}
