'use client';

// Endpoint overlay + disagreement-line plot. Plots BOTH libraries on a
// single shared canvas: endpoints colored per library, plus a thin grey
// line connecting kinematic↔v2 predictions for each control. The line
// lengths *are* the per-primitive disagreement — exactly the "where do
// the models predict different outcomes?" view.

import { useEffect, useMemo, useRef } from 'react';
import type { MotionPrimitive } from 'kinocat/primitives';
import type { PrimitiveMismatch } from '../lib/primitive-diagnostics';

export interface PrimitiveOverlayPlotProps {
  primitivesA: ReadonlyArray<MotionPrimitive>;
  primitivesB: ReadonlyArray<MotionPrimitive>;
  /** Pre-computed control-paired mismatches. */
  mismatches: ReadonlyArray<PrimitiveMismatch>;
  colorA: string;
  colorB: string;
  labelA: string;
  labelB: string;
  /** Shared extent used by the side-by-side fan plots above. */
  extent: { xMin: number; xMax: number; zMin: number; zMax: number };
  highlightIndex?: number;
  onHover?: (index: number | null) => void;
  aspectRatio?: number;
}

const PALETTE = {
  background: '#0d1119',
  grid: '#1a2030',
  axis: '#3a4458',
  footprint: '#5566aa',
  /** Disagreement line — neutral grey so it doesn't visually compete
   *  with the colored endpoints. */
  diff: '#666c7a',
};

const FOOTPRINT: [number, number][] = [
  [1.2, 0.6],
  [-1.2, 0.6],
  [-1.2, -0.6],
  [1.2, -0.6],
];

export function PrimitiveOverlayPlot({
  primitivesA, primitivesB, mismatches,
  colorA, colorB, labelA, labelB,
  extent, highlightIndex, onHover,
  aspectRatio = 4 / 3,
}: PrimitiveOverlayPlotProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dimsRef = useRef<{ cw: number; ch: number }>({ cw: 760, ch: 460 });

  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const observe = () => {
      const rect = cv.getBoundingClientRect();
      if (rect.width <= 0) return;
      const cw = Math.round(rect.width);
      const ch = Math.round(rect.width / aspectRatio);
      if (cw === dimsRef.current.cw && ch === dimsRef.current.ch) return;
      dimsRef.current = { cw, ch };
      cv.width = cw * (window.devicePixelRatio || 1);
      cv.height = ch * (window.devicePixelRatio || 1);
      cv.style.height = `${ch}px`;
      render();
    };
    const ro = new ResizeObserver(observe);
    ro.observe(cv);
    observe();
    return () => ro.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [aspectRatio]);

  // Map control-vector key → mismatch row, for hover lookup.
  const mismatchByA = useMemo(() => {
    const m = new Map<number, PrimitiveMismatch>();
    for (const x of mismatches) m.set(x.index, x);
    return m;
  }, [mismatches]);

  function render() {
    const cv = canvasRef.current;
    if (!cv) return;
    const ctx = cv.getContext('2d');
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const { cw, ch } = dimsRef.current;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = PALETTE.background;
    ctx.fillRect(0, 0, cw, ch);

    const xRange = extent.xMax - extent.xMin;
    const zRange = extent.zMax - extent.zMin;
    const scale = Math.min(cw / xRange, ch / zRange);
    const ox = cw / 2 - ((extent.xMin + extent.xMax) / 2) * scale;
    const oy = ch / 2 - ((extent.zMin + extent.zMax) / 2) * scale;
    const px = (x: number, z: number): [number, number] => [ox + x * scale, oy + z * scale];

    // Grid
    ctx.strokeStyle = PALETTE.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    const grid = 5;
    for (let x = Math.ceil(extent.xMin / grid) * grid; x <= extent.xMax; x += grid) {
      const [sx] = px(x, extent.zMin);
      ctx.moveTo(sx, 0);
      ctx.lineTo(sx, ch);
    }
    for (let z = Math.ceil(extent.zMin / grid) * grid; z <= extent.zMax; z += grid) {
      const [, sy] = px(extent.xMin, z);
      ctx.moveTo(0, sy);
      ctx.lineTo(cw, sy);
    }
    ctx.stroke();

    // Axes
    ctx.strokeStyle = PALETTE.axis;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    const [oxAxis, oyAxis] = px(0, 0);
    ctx.moveTo(0, oyAxis);
    ctx.lineTo(cw, oyAxis);
    ctx.moveTo(oxAxis, 0);
    ctx.lineTo(oxAxis, ch);
    ctx.stroke();

    // Footprint
    ctx.strokeStyle = PALETTE.footprint;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    FOOTPRINT.forEach(([x, z], i) => {
      const [a, b] = px(x, z);
      if (i === 0) ctx.moveTo(a, b);
      else ctx.lineTo(a, b);
    });
    ctx.closePath();
    ctx.stroke();

    // Disagreement lines first (so they sit BENEATH the endpoint dots).
    ctx.strokeStyle = PALETTE.diff;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const m of mismatches) {
      const [ax, ay] = px(m.endA.dx, m.endA.dz);
      const [bx, by] = px(m.endB.dx, m.endB.dz);
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
    }
    ctx.stroke();

    // Endpoints A (kinematic)
    ctx.fillStyle = colorA;
    for (const p of primitivesA) {
      const [a, b] = px(p.end.dx, p.end.dz);
      ctx.beginPath();
      ctx.arc(a, b, 4, 0, Math.PI * 2);
      ctx.fill();
    }
    // Endpoints B (v2)
    ctx.fillStyle = colorB;
    for (const p of primitivesB) {
      const [a, b] = px(p.end.dx, p.end.dz);
      ctx.beginPath();
      ctx.arc(a, b, 4, 0, Math.PI * 2);
      ctx.fill();
    }

    // Highlight selected primitive
    if (highlightIndex !== undefined && highlightIndex >= 0) {
      const m = mismatchByA.get(highlightIndex);
      if (m) {
        const [ax, ay] = px(m.endA.dx, m.endA.dz);
        const [bx, by] = px(m.endB.dx, m.endB.dz);
        ctx.strokeStyle = '#ffd070';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
        ctx.stroke();
        for (const [cx, cy, c] of [[ax, ay, colorA] as const, [bx, by, colorB] as const]) {
          ctx.strokeStyle = '#ffd070';
          ctx.fillStyle = c;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(cx, cy, 6, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
        }
      }
    }

    // Origin
    ctx.strokeStyle = PALETTE.axis;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(oxAxis, oyAxis, 4, 0, Math.PI * 2);
    ctx.stroke();
  }

  useEffect(render);  // eslint-disable-line react-hooks/exhaustive-deps

  function handleMouseMove(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!onHover) return;
    const cv = canvasRef.current;
    if (!cv) return;
    const rect = cv.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const { cw, ch } = dimsRef.current;
    const xRange = extent.xMax - extent.xMin;
    const zRange = extent.zMax - extent.zMin;
    const scale = Math.min(cw / xRange, ch / zRange);
    const ox = cw / 2 - ((extent.xMin + extent.xMax) / 2) * scale;
    const oy = ch / 2 - ((extent.zMin + extent.zMax) / 2) * scale;
    let bestIdx = -1;
    let bestDist = 16;
    // Match by A's index (mismatches uses A's index field).
    for (const m of mismatches) {
      // Hit-test both endpoints; closer one wins.
      const tests = [
        [ox + m.endA.dx * scale, oy + m.endA.dz * scale],
        [ox + m.endB.dx * scale, oy + m.endB.dz * scale],
      ] as const;
      for (const [tx, ty] of tests) {
        const d = Math.hypot(mx - tx, my - ty);
        if (d < bestDist) { bestDist = d; bestIdx = m.index; }
      }
    }
    onHover(bestIdx >= 0 ? bestIdx : null);
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, flexWrap: 'wrap' }}>
        <span style={{ fontWeight: 700, fontSize: 13 }}>overlay · disagreement</span>
        <Legend color={colorA} label={labelA} />
        <Legend color={colorB} label={labelB} />
        <Legend color={PALETTE.diff} label="connector = per-control mismatch" line />
      </div>
      <canvas
        ref={canvasRef}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => onHover?.(null)}
        style={{
          width: '100%',
          height: 'auto',
          aspectRatio: `${aspectRatio}`,
          borderRadius: 6,
          background: PALETTE.background,
          display: 'block',
          cursor: onHover ? 'crosshair' : 'default',
        }}
      />
    </div>
  );
}

function Legend({ color, label, line }: { color: string; label: string; line?: boolean }) {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5, fontSize: 11 }}>
      <span
        style={{
          display: 'inline-block',
          width: line ? 16 : 8,
          height: line ? 2 : 8,
          background: color,
          borderRadius: line ? 0 : '50%',
        }}
      />
      <span style={{ opacity: 0.7 }}>{label}</span>
    </span>
  );
}
