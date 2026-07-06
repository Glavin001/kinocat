// Top-down trajectory plotter for headless race debugging.
//
// Renders a bird's-eye PNG of a race run: course bounds, walls, waypoint
// gates (numbered, with arrive-radius disks), spawn pose, and the chassis
// trajectory as a polyline coloured by speed (blue → red over 0 → vMax).
// Event markers overlay the moments that matter when diagnosing a
// controller: stuck-recovery activations (red ✕), near-stops (hollow
// circles), and optionally the planned path active at a given time (thin
// grey overlay) so overshoot vs plan is visible at a glance.
//
// Node-only (uses @napi-rs/canvas — prebuilt native binary, no GUI needed).
// Consumed by the tmp-* debug scripts and any bench that wants per-lap
// visual artifacts; deliberately independent of the scenario runner so it
// can plot anything that produces {x, z, speed} samples.

import { createCanvas, type SKRSContext2D } from '@napi-rs/canvas';
import { writeFileSync } from 'node:fs';

export interface TrajectorySample {
  t: number;
  x: number;
  z: number;
  speed: number;
}

export interface TrajectoryEvent {
  t: number;
  x: number;
  z: number;
  kind: 'recovery' | 'stop' | 'offtrack' | 'lap';
  label?: string;
}

export interface PlanOverlay {
  /** Sim time the plan was committed (for the caption only). */
  t: number;
  pts: ReadonlyArray<{ x: number; z: number }>;
  /** Optional explicit stroke (overrides the default grey). Used to colour
   *  replans by order (time gradient) so plan STABILITY is visible: plans
   *  stacked on the same line = committed/stable; a fan of colours = thrash. */
  stroke?: string;
}

export interface CourseGeometry {
  bounds: { x0: number; z0: number; x1: number; z1: number };
  waypoints: ReadonlyArray<{ x: number; z: number; heading: number }>;
  walls?: ReadonlyArray<{ x: number; z: number; hx: number; hz: number }>;
  spawn?: { x: number; z: number; heading: number };
  arriveRadius?: number;
}

export interface TrajectoryPlotOptions {
  /** Image width in pixels (height derives from the course aspect). */
  width?: number;
  /** Speed mapped to the hot end of the colour ramp. Default: max sample. */
  vMax?: number;
  title?: string;
  /** Thin grey planned-path overlays (e.g. every Nth committed plan). */
  plans?: PlanOverlay[];
  events?: TrajectoryEvent[];
}

/** Blue → cyan → green → yellow → red ramp over t ∈ [0, 1]. */
function speedColor(tRaw: number): string {
  const t = Math.max(0, Math.min(1, tRaw));
  // Piecewise-linear through 4 stops (hue 240° → 0° in HSL terms).
  const hue = 240 * (1 - t);
  return `hsl(${hue.toFixed(0)}, 90%, 50%)`;
}

class Mapper {
  readonly sx: number;
  readonly sy: number;
  readonly scale: number;
  constructor(
    readonly b: CourseGeometry['bounds'],
    readonly w: number,
    readonly h: number,
    readonly pad: number,
  ) {
    this.scale = Math.min(
      (w - 2 * pad) / (b.x1 - b.x0),
      (h - 2 * pad) / (b.z1 - b.z0),
    );
    this.sx = pad - b.x0 * this.scale;
    this.sy = pad - b.z0 * this.scale;
  }
  px(x: number): number { return x * this.scale + this.sx; }
  // World +z drawn downward — a pure top-down view; flip if you prefer
  // math convention. Keeping z-down matches the web demo's camera.
  py(z: number): number { return z * this.scale + this.sy; }
}

function drawCourse(ctx: SKRSContext2D, course: CourseGeometry, m: Mapper): void {
  // Arena bounds.
  ctx.strokeStyle = '#556';
  ctx.lineWidth = 1.5;
  ctx.strokeRect(
    m.px(course.bounds.x0), m.py(course.bounds.z0),
    (course.bounds.x1 - course.bounds.x0) * m.scale,
    (course.bounds.z1 - course.bounds.z0) * m.scale,
  );
  // Walls.
  ctx.fillStyle = '#889';
  for (const w of course.walls ?? []) {
    ctx.fillRect(
      m.px(w.x - w.hx), m.py(w.z - w.hz),
      2 * w.hx * m.scale, 2 * w.hz * m.scale,
    );
  }
  // Waypoint gates: arrive disk + index + heading tick.
  const r = (course.arriveRadius ?? 2.5) * m.scale;
  ctx.font = '11px sans-serif';
  course.waypoints.forEach((wp, i) => {
    ctx.strokeStyle = '#3a3';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(m.px(wp.x), m.py(wp.z), r, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.strokeStyle = '#3a3';
    ctx.beginPath();
    ctx.moveTo(m.px(wp.x), m.py(wp.z));
    ctx.lineTo(
      m.px(wp.x + Math.cos(wp.heading) * (r / m.scale)),
      m.py(wp.z + Math.sin(wp.heading) * (r / m.scale)),
    );
    ctx.stroke();
    ctx.fillStyle = '#2a2';
    ctx.fillText(String(i), m.px(wp.x) + 3, m.py(wp.z) - 3);
  });
  // Spawn.
  if (course.spawn) {
    ctx.fillStyle = '#06c';
    ctx.beginPath();
    ctx.arc(m.px(course.spawn.x), m.py(course.spawn.z), 4, 0, 2 * Math.PI);
    ctx.fill();
    ctx.fillText('spawn', m.px(course.spawn.x) + 5, m.py(course.spawn.z) + 4);
  }
}

function drawColorbar(
  ctx: SKRSContext2D, x: number, y: number, w: number, h: number, vMax: number,
): void {
  for (let i = 0; i < w; i++) {
    ctx.fillStyle = speedColor(i / (w - 1));
    ctx.fillRect(x + i, y, 1, h);
  }
  ctx.strokeStyle = '#333';
  ctx.strokeRect(x, y, w, h);
  ctx.fillStyle = '#333';
  ctx.font = '11px sans-serif';
  ctx.fillText('0', x, y + h + 11);
  ctx.fillText(`${vMax.toFixed(0)} m/s`, x + w - 24, y + h + 11);
}

/**
 * Render a run to a PNG. Returns the file path for convenience.
 * `samples` should be time-ordered; large gaps (teleports) break the line.
 */
export function plotTrajectory(
  outPath: string,
  course: CourseGeometry,
  samples: ReadonlyArray<TrajectorySample>,
  opts: TrajectoryPlotOptions = {},
): string {
  const width = opts.width ?? 1200;
  const pad = 40;
  const aspect =
    (course.bounds.z1 - course.bounds.z0) / (course.bounds.x1 - course.bounds.x0);
  const height = Math.round((width - 2 * pad) * aspect) + 2 * pad + 30;
  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#fafafa';
  ctx.fillRect(0, 0, width, height);
  const m = new Mapper(course.bounds, width, height - 30, pad);

  drawCourse(ctx, course, m);

  // Planned-path overlays (thin, light — beneath the executed line).
  for (const plan of opts.plans ?? []) {
    ctx.strokeStyle = plan.stroke ?? 'rgba(120,120,140,0.45)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    plan.pts.forEach((p, i) => {
      if (i === 0) ctx.moveTo(m.px(p.x), m.py(p.z));
      else ctx.lineTo(m.px(p.x), m.py(p.z));
    });
    ctx.stroke();
  }

  // Executed trajectory, speed-coloured per segment.
  const vMax =
    opts.vMax ?? Math.max(1, ...samples.map((s) => Math.abs(s.speed)));
  ctx.lineWidth = 2.5;
  ctx.lineCap = 'round';
  for (let i = 1; i < samples.length; i++) {
    const a = samples[i - 1]!;
    const b = samples[i]!;
    // Teleport / reset guard.
    if (Math.hypot(b.x - a.x, b.z - a.z) > 10) continue;
    ctx.strokeStyle = speedColor(Math.abs(b.speed) / vMax);
    ctx.beginPath();
    ctx.moveTo(m.px(a.x), m.py(a.z));
    ctx.lineTo(m.px(b.x), m.py(b.z));
    ctx.stroke();
  }

  // Direction ticks every ~5 s so loops and reversals are readable.
  ctx.fillStyle = '#222';
  ctx.font = '10px sans-serif';
  let nextTick = 0;
  for (const s of samples) {
    if (s.t >= nextTick) {
      ctx.fillText(`${s.t.toFixed(0)}s`, m.px(s.x) + 3, m.py(s.z) - 3);
      ctx.beginPath();
      ctx.arc(m.px(s.x), m.py(s.z), 1.8, 0, 2 * Math.PI);
      ctx.fill();
      nextTick += 5;
    }
  }

  // Events.
  for (const e of opts.events ?? []) {
    const x = m.px(e.x);
    const y = m.py(e.z);
    if (e.kind === 'recovery') {
      ctx.strokeStyle = '#d00';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x - 5, y - 5); ctx.lineTo(x + 5, y + 5);
      ctx.moveTo(x - 5, y + 5); ctx.lineTo(x + 5, y - 5);
      ctx.stroke();
      ctx.fillStyle = '#d00';
      ctx.fillText(e.label ?? `recov@${e.t.toFixed(0)}s`, x + 7, y + 3);
    } else if (e.kind === 'stop') {
      ctx.strokeStyle = '#a0a';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, 2 * Math.PI);
      ctx.stroke();
    } else if (e.kind === 'lap') {
      ctx.fillStyle = '#080';
      ctx.font = 'bold 12px sans-serif';
      ctx.fillText(e.label ?? `lap@${e.t.toFixed(0)}s`, x + 6, y);
      ctx.font = '10px sans-serif';
    } else {
      ctx.fillStyle = '#f80';
      ctx.fillText(e.label ?? e.kind, x + 6, y);
    }
  }

  // Title + colorbar.
  ctx.fillStyle = '#111';
  ctx.font = 'bold 13px sans-serif';
  if (opts.title) ctx.fillText(opts.title, pad, height - 12);
  drawColorbar(ctx, width - 220, height - 26, 160, 10, vMax);

  writeFileSync(outPath, canvas.toBuffer('image/png'));
  return outPath;
}

/**
 * Convenience accumulator for scenario loops: push a sample every tick and
 * events as they happen; call `save()` once (whole run) or per lap.
 */
export class TrajectoryRecorder {
  samples: TrajectorySample[] = [];
  events: TrajectoryEvent[] = [];
  plans: PlanOverlay[] = [];
  private lastRecoveryCount = 0;
  private lastLapCount = 0;
  private lastPlanRef: unknown = null;
  private stopSince = -1;

  /** Call once per tick with the car's live status. */
  record(
    t: number,
    car: {
      state: { x: number; z: number; speed: number };
      quality: { recoveryCount: number };
      laps: ReadonlyArray<unknown>;
      plan: ReadonlyArray<{ x: number; z: number }> | null;
    },
    opts: { planEveryNth?: number } = {},
  ): void {
    const { x, z, speed } = car.state;
    this.samples.push({ t, x, z, speed });
    if (car.quality.recoveryCount > this.lastRecoveryCount) {
      this.lastRecoveryCount = car.quality.recoveryCount;
      this.events.push({ t, x, z, kind: 'recovery' });
    }
    if (car.laps.length > this.lastLapCount) {
      this.lastLapCount = car.laps.length;
      this.events.push({ t, x, z, kind: 'lap', label: `lap${car.laps.length}@${t.toFixed(0)}s` });
    }
    // Near-stop marker (rising edge after 1 s stationary).
    if (Math.abs(speed) < 0.3 && t > 2) {
      if (this.stopSince < 0) this.stopSince = t;
      else if (t - this.stopSince > 1.0) {
        this.events.push({ t, x, z, kind: 'stop' });
        this.stopSince = Infinity; // once per stop episode
      }
    } else {
      this.stopSince = -1;
    }
    // Committed-plan overlays: record each new plan identity (thin grey).
    const every = opts.planEveryNth ?? 4;
    if (car.plan && car.plan !== this.lastPlanRef) {
      this.lastPlanRef = car.plan;
      if (this.plans.length % every === 0 || this.plans.length === 0) {
        this.plans.push({ t, pts: car.plan.map((p) => ({ x: p.x, z: p.z })) });
      }
    }
  }

  save(outPath: string, course: CourseGeometry, title: string, vMax?: number): string {
    return plotTrajectory(outPath, course, this.samples, {
      title,
      vMax,
      events: this.events,
      plans: this.plans,
    });
  }
}
