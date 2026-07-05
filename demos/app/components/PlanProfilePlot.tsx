'use client';

// 2-D profile strip for a rich Plan (kinocat/plan). Plots the plan's scalar
// reference fields against ARC LENGTH — the honest medium for a 1-D signal.
// A function of one variable (speed, steer, accel vs. distance along the
// plan) is legible as a curve and unreadable as 3-D glyphs; the 3-D overlay
// carries the spatial story (where it drives, where it reverses, where it
// stops), this carries the quantitative one.
//
// Three stacked bands share the arc-length X axis, each auto-scaled with its
// own zero line for the signed ones. Forward↔reverse cusps are drawn as
// vertical dashed lines across every band, so you can see the whole reference
// — speed braking to the stall, the steering profile, the accel — snap at the
// gear change. Pure presentational; the caller passes the live committed plan.

import { useEffect, useRef } from 'react';
import type { Plan } from 'kinocat/plan';

export interface PlanProfilePlotProps {
  plan: Plan | null;
  /** Overall pixel height (all bands + labels). Default 190. */
  height?: number;
}

interface Band {
  label: string;
  color: string;
  signed: boolean;
  value: (i: number) => number;
}

const BG = '#0d1119';
const GRID = '#1a2030';
const AXIS = '#3a4458';
const CUSP = '#ffd24a';
const TEXT = '#8fa3bf';

export function PlanProfilePlot({ plan, height = 190 }: PlanProfilePlotProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const widthRef = useRef(320);

  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const ro = new ResizeObserver(() => {
      const rect = cv.getBoundingClientRect();
      if (rect.width <= 0) return;
      widthRef.current = Math.round(rect.width);
      draw();
    });
    ro.observe(cv);
    draw();
    return () => ro.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [plan, height]);

  function draw() {
    const cv = canvasRef.current;
    if (!cv) return;
    const ctx = cv.getContext('2d');
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const cw = widthRef.current;
    const ch = height;
    cv.width = cw * dpr;
    cv.height = ch * dpr;
    cv.style.height = `${ch}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, cw, ch);

    const pts = plan?.points ?? [];
    if (pts.length < 2) {
      ctx.fillStyle = TEXT;
      ctx.font = '11px ui-monospace, monospace';
      ctx.fillText('no plan yet', 8, 18);
      return;
    }

    const bands: Band[] = [
      { label: 'speed vRef (m/s)', color: '#5fd08a', signed: false, value: (i) => pts[i]!.vRef },
      { label: 'steer δff (rad)', color: '#ffaa33', signed: true, value: (i) => pts[i]!.steerFf ?? Math.atan(pts[i]!.kappa) },
      { label: 'accel aRef (m/s²)', color: '#6aa9ff', signed: true, value: (i) => pts[i]!.aRef },
    ];

    const s0 = pts[0]!.s;
    const sTotal = Math.max(pts[pts.length - 1]!.s - s0, 1e-6);
    const padL = 6;
    const padR = 6;
    const plotW = cw - padL - padR;
    const sx = (s: number) => padL + ((s - s0) / sTotal) * plotW;

    // Cusp arc-lengths (interior segment boundaries).
    const cuspS: number[] = [];
    const segs = plan?.segments ?? [];
    for (let k = 0; k < segs.length - 1; k++) {
      const p = pts[segs[k]!.endIdx];
      if (p) cuspS.push(p.s);
    }

    const bandH = ch / bands.length;
    bands.forEach((band, bi) => {
      const top = bi * bandH + 4;
      const bot = (bi + 1) * bandH - 4;
      const h = bot - top;

      // Range.
      let lo = Infinity;
      let hi = -Infinity;
      for (let i = 0; i < pts.length; i++) {
        const v = band.value(i);
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      if (band.signed) {
        const m = Math.max(Math.abs(lo), Math.abs(hi), 1e-3);
        lo = -m;
        hi = m;
      } else {
        lo = Math.min(lo, 0);
        hi = Math.max(hi, lo + 1e-3);
      }
      const vy = (v: number) => bot - ((v - lo) / (hi - lo)) * h;

      // Band frame + baseline.
      ctx.strokeStyle = GRID;
      ctx.lineWidth = 1;
      ctx.strokeRect(padL, top, plotW, h);
      if (band.signed) {
        ctx.strokeStyle = AXIS;
        ctx.beginPath();
        ctx.moveTo(padL, vy(0));
        ctx.lineTo(padL + plotW, vy(0));
        ctx.stroke();
      }

      // Cusp verticals.
      ctx.strokeStyle = CUSP;
      ctx.globalAlpha = 0.5;
      ctx.setLineDash([3, 3]);
      for (const cs of cuspS) {
        ctx.beginPath();
        ctx.moveTo(sx(cs), top);
        ctx.lineTo(sx(cs), bot);
        ctx.stroke();
      }
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;

      // Series.
      ctx.strokeStyle = band.color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let i = 0; i < pts.length; i++) {
        const x = sx(pts[i]!.s);
        const yv = vy(band.value(i));
        if (i === 0) ctx.moveTo(x, yv);
        else ctx.lineTo(x, yv);
      }
      ctx.stroke();

      // Labels: title (left) + range (right).
      ctx.font = '10px ui-monospace, monospace';
      ctx.fillStyle = band.color;
      ctx.fillText(band.label, padL + 4, top + 11);
      ctx.fillStyle = TEXT;
      ctx.textAlign = 'right';
      ctx.fillText(`${hi.toFixed(1)} … ${lo.toFixed(1)}`, padL + plotW - 4, top + 11);
      ctx.textAlign = 'left';
    });

    // X-axis caption.
    ctx.fillStyle = TEXT;
    ctx.font = '10px ui-monospace, monospace';
    ctx.fillText(`arc length → ${sTotal.toFixed(1)} m`, padL + 4, ch - 3);
  }

  return (
    <canvas
      ref={canvasRef}
      style={{ width: '100%', height, display: 'block', borderRadius: 4 }}
      aria-label="Plan reference profiles vs arc length"
    />
  );
}
