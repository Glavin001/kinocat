'use client';

// Canvas-based primitive fan plot. Renders every primitive in the supplied
// list as a swept polyline + endpoint dot in start-local frame (chassis at
// origin facing +X).
//
// Reusable for both the single-library view and as a building block for
// the overlay-and-diff view. Pure presentational — caller decides what
// primitives + colors to pass.

import { useEffect, useMemo, useRef } from 'react';
import type { MotionPrimitive } from 'kinocat/primitives';

export interface FanPlotGroundTruth {
  /** Index of the primitive this GT corresponds to. */
  index: number;
  /** End-state offset measured in Rapier (start-local frame). */
  dx: number;
  dz: number;
}

export interface FanPlotUncertainty {
  /** Index of the primitive this uncertainty applies to. */
  index: number;
  /** Radius (in meters) representing 1-sigma position spread across the
   *  ensemble. Rendered as a translucent halo around the predicted endpoint. */
  radiusM: number;
}

export interface PrimitiveFanPlotProps {
  primitives: ReadonlyArray<MotionPrimitive>;
  /** Color for forward primitives (and their endpoint dots). */
  forwardColor: string;
  /** Color for reverse primitives. Default desaturated forwardColor. */
  reverseColor?: string;
  /** Title rendered above the canvas. */
  title: string;
  /** Caption rendered below the canvas. */
  subtitle?: string;
  /** Index of a primitive to highlight (matches `primitives[i]`). */
  highlightIndex?: number;
  /** Caller-supplied callback when the user hovers over a primitive's
   *  endpoint. `null` when no hit. */
  onHover?: (index: number | null) => void;
  /** Aspect ratio (width/height). Default 4:3. */
  aspectRatio?: number;
  /** Color palette overrides. */
  palette?: {
    background?: string;
    grid?: string;
    axis?: string;
    footprint?: string;
  };
  /** Bounding-box override. When supplied, the plot uses this as its
   *  fixed extent so multiple plots can share identical axes for visual
   *  comparison. Otherwise auto-fits from the primitive sweeps. */
  fixedExtent?: { xMin: number; xMax: number; zMin: number; zMax: number };
  /** Ground-truth Rapier endpoints rendered as white-ish dots with
   *  arrows from the predicted endpoint → ground truth (= per-control
   *  open-loop error visualized). */
  groundTruth?: ReadonlyArray<FanPlotGroundTruth>;
  /** Ensemble uncertainty halos — translucent rings around each
   *  predicted endpoint sized by per-primitive 1-sigma spread. */
  uncertainty?: ReadonlyArray<FanPlotUncertainty>;
}

const DEFAULT_PALETTE = {
  background: '#0d1119',
  grid: '#1a2030',
  axis: '#3a4458',
  footprint: '#5566aa',
};

const FOOTPRINT: [number, number][] = [
  [1.2, 0.6],
  [-1.2, 0.6],
  [-1.2, -0.6],
  [1.2, -0.6],
];

export function PrimitiveFanPlot({
  primitives,
  forwardColor,
  reverseColor,
  title,
  subtitle,
  highlightIndex,
  onHover,
  aspectRatio = 4 / 3,
  palette,
  fixedExtent,
  groundTruth,
  uncertainty,
}: PrimitiveFanPlotProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dimsRef = useRef<{ cw: number; ch: number }>({ cw: 760, ch: 460 });
  const pal = { ...DEFAULT_PALETTE, ...palette };
  const reverse = reverseColor ?? desaturate(forwardColor);

  // Compute plot extent. fixedExtent takes priority for shared-axis views.
  const extent = useMemo(() => {
    if (fixedExtent) return fixedExtent;
    let xMin = -1, xMax = 1, zMin = -1, zMax = 1;
    for (const p of primitives) {
      for (const s of p.sweep) {
        if (s.x < xMin) xMin = s.x;
        if (s.x > xMax) xMax = s.x;
        if (s.z < zMin) zMin = s.z;
        if (s.z > zMax) zMax = s.z;
      }
    }
    const pad = 1.5;
    return { xMin: xMin - pad, xMax: xMax + pad, zMin: zMin - pad, zMax: zMax + pad };
  }, [primitives, fixedExtent]);

  // Re-render on resize.
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

  function render() {
    const cv = canvasRef.current;
    if (!cv) return;
    const ctx = cv.getContext('2d');
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const { cw, ch } = dimsRef.current;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = pal.background;
    ctx.fillRect(0, 0, cw, ch);

    // Compute world→screen with equal scaling and centered viewport.
    const xRange = extent.xMax - extent.xMin;
    const zRange = extent.zMax - extent.zMin;
    const scale = Math.min(cw / xRange, ch / zRange);
    const ox = cw / 2 - ((extent.xMin + extent.xMax) / 2) * scale;
    const oy = ch / 2 - ((extent.zMin + extent.zMax) / 2) * scale;
    const px = (x: number, z: number): [number, number] => [ox + x * scale, oy + z * scale];

    // Grid: 5m lines
    ctx.strokeStyle = pal.grid;
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

    // Axes through origin
    ctx.strokeStyle = pal.axis;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    const [, oyAxis] = px(0, 0);
    const [oxAxis] = px(0, 0);
    ctx.moveTo(0, oyAxis);
    ctx.lineTo(cw, oyAxis);
    ctx.moveTo(oxAxis, 0);
    ctx.lineTo(oxAxis, ch);
    ctx.stroke();

    // Footprint at origin (heading 0)
    ctx.strokeStyle = pal.footprint;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    FOOTPRINT.forEach(([x, z], i) => {
      const [a, b] = px(x, z);
      if (i === 0) ctx.moveTo(a, b);
      else ctx.lineTo(a, b);
    });
    ctx.closePath();
    ctx.stroke();

    // Primitives. Render unhighlighted first, then highlighted on top.
    const drawOne = (p: MotionPrimitive, dim: boolean) => {
      const color = p.reverse ? reverse : forwardColor;
      ctx.strokeStyle = dim ? withAlpha(color, 0.35) : color;
      ctx.lineWidth = dim ? 1.5 : 2.5;
      ctx.beginPath();
      p.sweep.forEach((s, i) => {
        const [a, b] = px(s.x, s.z);
        if (i === 0) ctx.moveTo(a, b);
        else ctx.lineTo(a, b);
      });
      ctx.stroke();
      const e = px(p.end.dx, p.end.dz);
      ctx.fillStyle = dim ? withAlpha(color, 0.45) : color;
      ctx.beginPath();
      ctx.arc(e[0], e[1], dim ? 3 : 4.5, 0, Math.PI * 2);
      ctx.fill();
    };
    const hasHighlight = highlightIndex !== undefined && highlightIndex >= 0;
    for (let i = 0; i < primitives.length; i++) {
      if (hasHighlight && i === highlightIndex) continue;
      drawOne(primitives[i]!, hasHighlight);
    }
    if (hasHighlight && highlightIndex! < primitives.length) {
      drawOne(primitives[highlightIndex!]!, false);
    }

    // Uncertainty halos — drawn UNDER the GT arrows so the arrows read
    // on top. A halo larger than the GT arrow length signals the model
    // is honest about how wrong it might be (or that the ensemble
    // hasn't seen this region).
    if (uncertainty && uncertainty.length > 0) {
      for (const u of uncertainty) {
        const p = primitives[u.index];
        if (!p) continue;
        const [ex, ey] = px(p.end.dx, p.end.dz);
        const rPx = u.radiusM * scale;
        if (rPx < 0.5) continue;
        ctx.fillStyle = withAlpha(forwardColor, 0.10);
        ctx.beginPath();
        ctx.arc(ex, ey, rPx, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = withAlpha(forwardColor, 0.35);
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(ex, ey, rPx, 0, Math.PI * 2);
        ctx.stroke();
      }
    }

    // Ground-truth dots + error arrows (predicted -> actual).
    if (groundTruth && groundTruth.length > 0) {
      ctx.lineWidth = 1.25;
      for (const g of groundTruth) {
        const p = primitives[g.index];
        if (!p) continue;
        const [px0, py0] = px(p.end.dx, p.end.dz);
        const [gx, gy] = px(g.dx, g.dz);
        // Arrow shaft.
        ctx.strokeStyle = '#ffd070';
        ctx.beginPath();
        ctx.moveTo(px0, py0);
        ctx.lineTo(gx, gy);
        ctx.stroke();
        // GT dot.
        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.arc(gx, gy, 2.5, 0, Math.PI * 2);
        ctx.fill();
        // Subtle outline so it pops against the colored prediction dot.
        ctx.strokeStyle = '#ffd070';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(gx, gy, 3.5, 0, Math.PI * 2);
        ctx.stroke();
      }
    }

    // Origin marker
    ctx.strokeStyle = pal.axis;
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
    let bestDist = 16; // px hit radius
    for (let i = 0; i < primitives.length; i++) {
      const p = primitives[i]!;
      const ex = ox + p.end.dx * scale;
      const ey = oy + p.end.dz * scale;
      const d = Math.hypot(mx - ex, my - ey);
      if (d < bestDist) { bestDist = d; bestIdx = i; }
    }
    onHover(bestIdx >= 0 ? bestIdx : null);
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
        <span style={{ color: forwardColor, fontWeight: 700, fontSize: 13 }}>{title}</span>
        {subtitle && <span style={{ opacity: 0.6, fontSize: 11 }}>{subtitle}</span>}
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
          background: pal.background,
          display: 'block',
          cursor: onHover ? 'crosshair' : 'default',
        }}
      />
    </div>
  );
}

function desaturate(hex: string): string {
  // crude darken — works for our hex palette
  return withAlpha(hex, 0.5);
}

function withAlpha(hex: string, a: number): string {
  if (hex.startsWith('#') && (hex.length === 7 || hex.length === 4)) {
    const r = parseInt(hex.length === 7 ? hex.slice(1, 3) : hex[1]! + hex[1], 16);
    const g = parseInt(hex.length === 7 ? hex.slice(3, 5) : hex[2]! + hex[2], 16);
    const b = parseInt(hex.length === 7 ? hex.slice(5, 7) : hex[3]! + hex[3], 16);
    return `rgba(${r}, ${g}, ${b}, ${a})`;
  }
  return hex;
}
