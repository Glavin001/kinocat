'use client';

// Pick a held-out trial; render Rapier ground-truth path, v2 full
// rollout, parametric-only rollout, and kinematic baseline side by
// side. Includes a scrubber so the user can watch divergence develop
// over the trial duration — this is the visual answer to "how does
// the model think about motion?"

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type {
  LearnedVehicleModel,
  LearnableVehicleConfig,
  CarKinematicState,
  WheeledCarControls,
} from 'kinocat/agent';
import {
  defaultVehicleAgent,
  kinematicForwardSim,
  learnedForwardSimV2,
  parametricForwardV2,
} from 'kinocat/agent';
import type { Trial } from 'kinocat/learning';

export interface RolloutTrial extends Trial<CarKinematicState, WheeledCarControls, LearnableVehicleConfig> {}

export interface RolloutPlayerProps {
  trial: RolloutTrial | null;
  model: LearnedVehicleModel | null;
  /** Optional second rollout the user can attach (e.g. for the scenario
   *  playground's "Run in Rapier" capture). */
  extraTrack?: { name: string; color: string; states: CarKinematicState[]; times: number[] } | null;
  /** Trial picker UI. */
  trials: ReadonlyArray<RolloutTrial>;
  onSelectTrial?: (t: RolloutTrial) => void;
}

type Track = {
  name: string;
  color: string;
  /** State at each sample boundary. */
  states: CarKinematicState[];
  /** Time (sec) at each sample boundary. */
  times: number[];
};

function buildTracks(
  trial: RolloutTrial,
  model: LearnedVehicleModel | null,
): Track[] {
  const sampleStride = Math.max(1, Math.round((trial.samples[1]?.t ?? trial.dt * 6) / trial.dt));
  const tracks: Track[] = [];
  // Rapier ground truth — already sampled by the harness.
  tracks.push({
    name: 'Rapier (ground truth)',
    color: '#ffffff',
    states: trial.samples.map((s) => s.state),
    times: trial.samples.map((s) => s.t),
  });
  if (model) {
    const ctrlVec = (c: WheeledCarControls) => [c.steer, c.driveForce, c.brakeForce];
    const v2Sim = learnedForwardSimV2(model);
    const paramSim = parametricForwardV2(model.params, model.config);
    const v2States: CarKinematicState[] = [trial.initialState];
    const v2Times: number[] = [0];
    const paramStates: CarKinematicState[] = [trial.initialState];
    const paramTimes: number[] = [0];
    let sV = trial.initialState;
    let sP = trial.initialState;
    for (let i = 0; i < trial.controlsTrace.length; i++) {
      const cv = ctrlVec(trial.controlsTrace[i]!);
      sV = v2Sim(sV, cv, trial.dt);
      sP = paramSim(sP, cv, trial.dt);
      if ((i + 1) % sampleStride === 0) {
        v2States.push(sV);
        v2Times.push((i + 1) * trial.dt);
        paramStates.push(sP);
        paramTimes.push((i + 1) * trial.dt);
      }
    }
    if (model.residualEnsemble.length > 0) {
      tracks.push({ name: 'v2 full', color: '#55dcff', states: v2States, times: v2Times });
      tracks.push({ name: 'parametric-only', color: '#a6e9ff', states: paramStates, times: paramTimes });
    } else {
      tracks.push({ name: 'v2 parametric', color: '#55dcff', states: v2States, times: v2Times });
    }
  }
  // Kinematic baseline — adapter approximates the legacy [curvature, targetSpeed].
  const agent = defaultVehicleAgent();
  const kinSim = kinematicForwardSim(agent);
  const wheeledToLegacy = (c: WheeledCarControls): number[] => {
    const k = Math.sin(c.steer) / (2 * 1.6); // wheelBase default
    const targetSpeed = c.driveForce > 0 ? 10 : (c.brakeForce > 0 ? 0 : 5);
    return [k, targetSpeed];
  };
  const kinStates: CarKinematicState[] = [trial.initialState];
  const kinTimes: number[] = [0];
  let sK = trial.initialState;
  for (let i = 0; i < trial.controlsTrace.length; i++) {
    sK = kinSim(sK, wheeledToLegacy(trial.controlsTrace[i]!), trial.dt);
    if ((i + 1) % sampleStride === 0) {
      kinStates.push(sK);
      kinTimes.push((i + 1) * trial.dt);
    }
  }
  tracks.push({ name: 'kinematic', color: '#ffd070', states: kinStates, times: kinTimes });
  return tracks;
}

function trialDuration(trial: RolloutTrial): number {
  const last = trial.samples[trial.samples.length - 1];
  return last?.t ?? trial.dt * trial.controlsTrace.length;
}

function trialEndError(trial: RolloutTrial, model: LearnedVehicleModel | null): number {
  if (!model) return 0;
  const tracks = buildTracks(trial, model);
  const gt = tracks[0]!;
  const v2 = tracks.find((t) => t.name.startsWith('v2'));
  if (!v2) return 0;
  const a = gt.states[gt.states.length - 1]!;
  const b = v2.states[v2.states.length - 1]!;
  return Math.hypot(a.x - b.x, a.z - b.z);
}

export function RolloutPlayer({ trial, model, trials, onSelectTrial, extraTrack }: RolloutPlayerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [sortBy, setSortBy] = useState<'error' | 'index' | 'speed'>('error');

  const tracks = useMemo(() => {
    if (!trial) return [] as Track[];
    const base = buildTracks(trial, model);
    if (extraTrack) base.push(extraTrack);
    return base;
  }, [trial, model, extraTrack]);

  const duration = trial ? trialDuration(trial) : 0;

  // Animate when playing.
  useEffect(() => {
    if (!playing || !trial) return;
    const start = performance.now() - t * 1000;
    let raf = 0;
    const tick = () => {
      const now = performance.now();
      const cur = (now - start) / 1000;
      if (cur >= duration) {
        setT(duration);
        setPlaying(false);
        return;
      }
      setT(cur);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, trial?.id]);

  const sortedTrials = useMemo(() => {
    if (!model) return trials;
    const arr = [...trials];
    if (sortBy === 'error') {
      const errs = new Map(arr.map((t) => [t.id, trialEndError(t, model)]));
      arr.sort((a, b) => (errs.get(b.id) ?? 0) - (errs.get(a.id) ?? 0));
    } else if (sortBy === 'speed') {
      arr.sort((a, b) => Math.abs(b.initialState.speed) - Math.abs(a.initialState.speed));
    }
    return arr;
  }, [trials, sortBy, model]);

  // Canvas rendering.
  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = cv.getBoundingClientRect();
      const w = rect.width;
      const h = Math.max(280, Math.round(w * 0.55));
      cv.width = w * dpr;
      cv.height = h * dpr;
      cv.style.height = `${h}px`;
      const ctx = cv.getContext('2d');
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = '#0d1119';
      ctx.fillRect(0, 0, w, h);
      if (tracks.length === 0 || !trial) {
        ctx.fillStyle = '#666';
        ctx.font = '12px ui-monospace, monospace';
        ctx.fillText('select a trial to play', 12, 22);
        return;
      }

      // World extent — union over all tracks' visible portions.
      let xMin = Infinity, xMax = -Infinity, zMin = Infinity, zMax = -Infinity;
      const localXform = (s: CarKinematicState) => {
        // Center around the trial's spawn — the harness teleports to
        // (0,0) heading 0, but the recorded samples are in world frame.
        // Trial samples already start near (0,0).
        return { x: s.x, z: s.z };
      };
      for (const tr of tracks) {
        for (const s of tr.states) {
          const p = localXform(s);
          if (p.x < xMin) xMin = p.x;
          if (p.x > xMax) xMax = p.x;
          if (p.z < zMin) zMin = p.z;
          if (p.z > zMax) zMax = p.z;
        }
      }
      const padW = Math.max(2, (xMax - xMin) * 0.1);
      const padH = Math.max(2, (zMax - zMin) * 0.1);
      xMin -= padW; xMax += padW; zMin -= padH; zMax += padH;
      const xR = xMax - xMin || 1;
      const zR = zMax - zMin || 1;
      const scale = Math.min(w / xR, h / zR);
      const ox = w / 2 - ((xMin + xMax) / 2) * scale;
      const oy = h / 2 - ((zMin + zMax) / 2) * scale;
      const px = (s: CarKinematicState): [number, number] => [ox + localXform(s).x * scale, oy + localXform(s).z * scale];

      // Grid
      ctx.strokeStyle = '#1a2030';
      ctx.lineWidth = 1;
      ctx.beginPath();
      const grid = 5;
      for (let xg = Math.ceil(xMin / grid) * grid; xg <= xMax; xg += grid) {
        const sx = ox + xg * scale;
        ctx.moveTo(sx, 0); ctx.lineTo(sx, h);
      }
      for (let zg = Math.ceil(zMin / grid) * grid; zg <= zMax; zg += grid) {
        const sy = oy + zg * scale;
        ctx.moveTo(0, sy); ctx.lineTo(w, sy);
      }
      ctx.stroke();

      // Cursor t — frame index per track.
      const curIdx = (tr: Track) => {
        let best = 0;
        for (let i = 0; i < tr.times.length; i++) {
          if (tr.times[i]! <= t) best = i; else break;
        }
        return best;
      };

      // Trails — full path dimmed; segment up to cursor brighter.
      for (const tr of tracks) {
        ctx.strokeStyle = withAlpha(tr.color, 0.25);
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        tr.states.forEach((s, i) => {
          const [a, b] = px(s);
          if (i === 0) ctx.moveTo(a, b); else ctx.lineTo(a, b);
        });
        ctx.stroke();
        const ci = curIdx(tr);
        ctx.strokeStyle = tr.color;
        ctx.lineWidth = 2.25;
        ctx.beginPath();
        for (let i = 0; i <= ci; i++) {
          const [a, b] = px(tr.states[i]!);
          if (i === 0) ctx.moveTo(a, b); else ctx.lineTo(a, b);
        }
        ctx.stroke();
      }

      // Error connectors at the cursor: link GT to each predicted track.
      const gt = tracks[0]!;
      const gtIdx = curIdx(gt);
      const [gx, gy] = px(gt.states[gtIdx]!);
      for (let ti = 1; ti < tracks.length; ti++) {
        const tr = tracks[ti]!;
        const ci = curIdx(tr);
        const [a, b] = px(tr.states[ci]!);
        ctx.strokeStyle = withAlpha(tr.color, 0.6);
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(gx, gy);
        ctx.lineTo(a, b);
        ctx.stroke();
      }
      ctx.setLineDash([]);

      // Car heading triangles at cursor.
      for (const tr of tracks) {
        const ci = curIdx(tr);
        const s = tr.states[ci]!;
        const [a, b] = px(s);
        const c = Math.cos(s.heading);
        const sn = Math.sin(s.heading);
        const len = 8;
        const w2 = 4;
        ctx.fillStyle = tr.color;
        ctx.beginPath();
        ctx.moveTo(a + c * len, b + sn * len);
        ctx.lineTo(a - c * w2 + sn * w2, b - sn * w2 - c * w2);
        ctx.lineTo(a - c * w2 - sn * w2, b - sn * w2 + c * w2);
        ctx.closePath();
        ctx.fill();
      }
    };
    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(cv);
    return () => ro.disconnect();
  }, [tracks, t, trial]);

  const onScrub = useCallback((v: number) => {
    setPlaying(false);
    setT(v);
  }, []);

  // GT-vs-v2 error at cursor.
  const errorAt = useMemo(() => {
    if (tracks.length < 2) return null;
    const gt = tracks[0]!;
    const v2 = tracks.find((tr) => tr.name.startsWith('v2'));
    if (!v2) return null;
    const idxFor = (tr: Track) => {
      let best = 0;
      for (let i = 0; i < tr.times.length; i++) {
        if (tr.times[i]! <= t) best = i; else break;
      }
      return best;
    };
    const a = gt.states[idxFor(gt)]!;
    const b = v2.states[idxFor(v2)]!;
    return Math.hypot(a.x - b.x, a.z - b.z);
  }, [tracks, t]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
        <strong style={{ color: '#7fd6ff' }}>Trial rollout</strong>
        {trial && (
          <span style={{ opacity: 0.65, fontSize: 11 }}>
            {trial.id} · v0 = {trial.initialState.speed.toFixed(1)} m/s · duration {duration.toFixed(2)}s
          </span>
        )}
        <span style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as typeof sortBy)}
            style={selectStyle}
          >
            <option value="error">sort: worst error first</option>
            <option value="speed">sort: highest speed first</option>
            <option value="index">sort: collection order</option>
          </select>
          <select
            value={trial?.id ?? ''}
            onChange={(e) => {
              const next = sortedTrials.find((t) => t.id === e.target.value);
              if (next) onSelectTrial?.(next);
            }}
            style={selectStyle}
          >
            <option value="" disabled>{trials.length === 0 ? 'no trials in store' : 'pick trial'}</option>
            {sortedTrials.map((tr) => (
              <option key={tr.id} value={tr.id}>
                {tr.id} · v0={tr.initialState.speed.toFixed(1)}
              </option>
            ))}
          </select>
        </span>
      </div>
      <canvas ref={canvasRef} style={{ width: '100%', borderRadius: 6 }} />
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <button onClick={() => setPlaying((p) => !p)} disabled={!trial} style={btnStyle}>
          {playing ? '⏸ pause' : '▶ play'}
        </button>
        <button onClick={() => { setT(0); setPlaying(false); }} disabled={!trial} style={btnStyle}>⟲</button>
        <input
          type="range"
          min={0}
          max={duration || 1}
          step={duration / 200 || 0.01}
          value={t}
          onChange={(e) => onScrub(Number(e.target.value))}
          style={{ flex: 1 }}
          disabled={!trial}
        />
        <span style={{ fontSize: 11, opacity: 0.7, minWidth: 90, textAlign: 'right' }}>
          t = {t.toFixed(2)}s
        </span>
        {errorAt !== null && (
          <span style={{ fontSize: 11, color: '#ffd070', minWidth: 110, textAlign: 'right' }}>
            v2 err: {errorAt.toFixed(3)} m
          </span>
        )}
      </div>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', fontSize: 11 }}>
        {tracks.map((tr) => (
          <span key={tr.name} style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
            <span style={{
              width: 12, height: 3, background: tr.color, display: 'inline-block', borderRadius: 1,
            }} />
            <span style={{ color: tr.color }}>{tr.name}</span>
          </span>
        ))}
      </div>
    </div>
  );
}

function withAlpha(hex: string, a: number): string {
  if (hex.startsWith('#') && (hex.length === 7 || hex.length === 4)) {
    const r = parseInt(hex.length === 7 ? hex.slice(1, 3) : hex[1]! + hex[1], 16);
    const g = parseInt(hex.length === 7 ? hex.slice(3, 5) : hex[2]! + hex[2], 16);
    const b = parseInt(hex.length === 7 ? hex.slice(5, 7) : hex[3]! + hex[3], 16);
    return `rgba(${r},${g},${b},${a})`;
  }
  return hex;
}

const selectStyle: React.CSSProperties = {
  background: '#0d1119', color: '#cdd3de', border: '1px solid #1f2735',
  font: '11px ui-monospace, monospace', borderRadius: 4, padding: '4px 6px',
};

const btnStyle: React.CSSProperties = {
  background: '#1a2030', color: '#cdd3de', border: '1px solid #1f2735',
  font: '11px ui-monospace, monospace', borderRadius: 4, padding: '4px 10px',
  cursor: 'pointer',
};

// Re-export so the dashboard can also reuse buildTracks logic if needed.
export { buildTracks };
