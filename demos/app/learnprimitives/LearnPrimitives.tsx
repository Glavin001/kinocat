'use client';

// Autonomous motion-primitive learner UI. On mount it spawns a Rapier vehicle,
// drives it through 48 deterministic open-loop trials on flat ground, fits a
// 5-coefficient parametric dynamics model to the recorded trajectories, and
// produces a drop-in `MotionPrimitiveLibrary`. The fitted coefficients and
// derived library are persisted to localStorage so re-visits load instantly.
//
// All physics + fitting logic lives in `demos/app/lib/learn-primitives.ts`;
// this file is just the React shell + a top-down Three.js arc visualisation.

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import type {
  LearnedVehicleParams,
  CarKinematicState,
  VehicleAgent,
} from 'kinocat/agent';
import { kinematicForwardSim } from 'kinocat/agent';
import {
  buildLearnedLibrary,
  createSweepWorld,
  defaultControlSets,
  DEFAULT_START_SPEEDS,
  fitParams,
  PRIMITIVE_DURATION,
  PRIMITIVE_SUBSTEPS,
  runSweep,
  summariseKinematicGap,
  type DiscrepancySummary,
  type FitResult,
  type SweepData,
} from '../lib/learn-primitives';
import { CARCHASE_AGENT } from '../lib/carchase-scenarios';

// Schema-versioned cache key. Bump the version suffix any time the model
// formulation (e.g. learnedForwardSim) or the parameter bounds change in
// a way that would silently re-interpret old cached coefficients. The old
// keys become unreachable; the user transparently re-fits.
//
// :v2 — switched learnedForwardSim from clamp(speedErr/tau) to
//       sat*tanh(speedErr/(sat*tau)); also tightened PARAM_LO/HI which
//       would otherwise clamp old fits onto the new bounds and produce
//       a permanently-pinned prior.
const PARAMS_KEY = 'kinocat:learned-params:v2';
const LIBRARY_KEY = 'kinocat:learned-library:v2';

type Phase = 'idle' | 'collecting' | 'fitting' | 'done' | 'error';

interface CachedRun {
  params: LearnedVehicleParams;
  fit: { meanPosError: number; maxPosError: number; loss: number };
  /** Optional — older caches (or the race demo's inline learn) may omit
   *  this. Render gracefully when missing. */
  kinematic?: DiscrepancySummary;
  libraryJSON: string;
  createdAt: number;
}

function loadCached(): CachedRun | null {
  if (typeof window === 'undefined') return null;
  try {
    const params = window.localStorage.getItem(PARAMS_KEY);
    const lib = window.localStorage.getItem(LIBRARY_KEY);
    if (!params || !lib) return null;
    return { ...JSON.parse(params), libraryJSON: lib } as CachedRun;
  } catch {
    return null;
  }
}

function saveCached(run: CachedRun) {
  if (typeof window === 'undefined') return;
  const { libraryJSON, ...meta } = run;
  window.localStorage.setItem(PARAMS_KEY, JSON.stringify(meta));
  window.localStorage.setItem(LIBRARY_KEY, libraryJSON);
}

function clearCached() {
  if (typeof window === 'undefined') return;
  window.localStorage.removeItem(PARAMS_KEY);
  window.localStorage.removeItem(LIBRARY_KEY);
}

export default function LearnPrimitives() {
  const [phase, setPhase] = useState<Phase>('idle');
  const [progress, setProgress] = useState({ done: 0, total: 0 });
  const [run, setRun] = useState<CachedRun | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sweep, setSweep] = useState<SweepData | null>(null);
  const mountRef = useRef<HTMLDivElement>(null);
  const startedRef = useRef(false);

  // On mount: if cached results exist, surface them immediately; else kick the
  // sweep off automatically. Either way the user can hit "re-learn" to redo.
  useEffect(() => {
    if (startedRef.current) return;
    startedRef.current = true;
    const cached = loadCached();
    if (cached) {
      setRun(cached);
      setPhase('done');
    } else {
      void runLearn();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function runLearn() {
    setError(null);
    setPhase('collecting');
    setProgress({ done: 0, total: 0 });
    setSweep(null);
    try {
      const sw = await createSweepWorld(CARCHASE_AGENT);
      try {
        const sweepData = await runSweep(sw, {
          startSpeeds: DEFAULT_START_SPEEDS,
          controlSets: defaultControlSets(CARCHASE_AGENT),
          onProgress: (p) => setProgress({ done: p.done, total: p.total }),
          yieldEvery: 4,
          yieldFn: () => new Promise<void>((r) => setTimeout(r, 0)),
        });
        setSweep(sweepData);
        setPhase('fitting');
        await new Promise((r) => setTimeout(r, 0));
        const fit = fitParams(sweepData);
        const kinematic = summariseKinematicGap(sweepData);
        const lib = buildLearnedLibrary(fit.params, { agent: CARCHASE_AGENT });
        const next: CachedRun = {
          params: fit.params,
          fit: {
            meanPosError: fit.meanPosError,
            maxPosError: fit.maxPosError,
            loss: fit.loss,
          },
          kinematic,
          libraryJSON: lib.toJSON(),
          createdAt: Date.now(),
        };
        saveCached(next);
        setRun(next);
        setPhase('done');
      } finally {
        sw.dispose();
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setPhase('error');
    }
  }

  function onReLearn() {
    clearCached();
    setRun(null);
    void runLearn();
  }

  function onDownloadParams() {
    if (!run) return;
    downloadBlob(
      `learned-params-${run.createdAt}.json`,
      JSON.stringify(run, null, 2),
    );
  }

  function onDownloadLibrary() {
    if (!run) return;
    downloadBlob(
      `learned-library-${run.createdAt}.json`,
      run.libraryJSON,
    );
  }

  // Visualisation: top-down ortho scene with one line per trial. Drawn from
  // the most recent sweep (collected this session) or — if loaded from cache —
  // the recovered library's local sweeps.
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    const W = mount.clientWidth;
    const H = mount.clientHeight;
    if (W === 0 || H === 0) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0d14);
    const range = 14;
    const aspect = W / H;
    const camera = new THREE.OrthographicCamera(
      -range * aspect,
      range * aspect,
      range,
      -range,
      0.1,
      100,
    );
    camera.position.set(0, 20, 0);
    camera.up.set(0, 0, 1);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);

    // Grid.
    const grid = new THREE.GridHelper(30, 30, 0x223044, 0x141a26);
    scene.add(grid);

    // Origin marker.
    const origin = new THREE.Mesh(
      new THREE.RingGeometry(0.3, 0.5, 24),
      new THREE.MeshBasicMaterial({ color: 0xffd070, side: THREE.DoubleSide }),
    );
    origin.rotation.x = -Math.PI / 2;
    scene.add(origin);

    function buildLines() {
      const group = new THREE.Group();
      if (sweep) {
        for (const tr of sweep.trials) {
          const pts = tr.samples.map((s) => new THREE.Vector3(s.x, 0.1, s.z));
          const color = tr.controls[1] >= 0 ? 0x55b8ff : 0xffa030;
          const line = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(pts),
            new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.85 }),
          );
          group.add(line);
        }
        if (run) {
          // Kinematic ghost — dashed gray, current params (or default).
          for (const tr of sweep.trials) {
            const sim = kinematicForwardSim(sweep.agent as VehicleAgent);
            let s: CarKinematicState = {
              x: 0,
              z: 0,
              heading: 0,
              speed: tr.startSpeed,
              t: 0,
            };
            const pts: THREE.Vector3[] = [new THREE.Vector3(0, 0.05, 0)];
            const dt = PRIMITIVE_DURATION / PRIMITIVE_SUBSTEPS;
            for (let k = 0; k < PRIMITIVE_SUBSTEPS; k++) {
              s = sim(s, tr.controls, dt);
              pts.push(new THREE.Vector3(s.x, 0.05, s.z));
            }
            const mat = new THREE.LineDashedMaterial({
              color: 0x7a8398,
              dashSize: 0.3,
              gapSize: 0.2,
              transparent: true,
              opacity: 0.6,
            });
            const line = new THREE.Line(
              new THREE.BufferGeometry().setFromPoints(pts),
              mat,
            );
            line.computeLineDistances();
            group.add(line);
          }
        }
      }
      return group;
    }

    let lineGroup = buildLines();
    scene.add(lineGroup);

    let stopped = false;
    function tick() {
      if (stopped) return;
      requestAnimationFrame(tick);
      renderer.render(scene, camera);
    }
    tick();

    const onResize = () => {
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      if (!w || !h) return;
      renderer.setSize(w, h);
      const a = w / h;
      camera.left = -range * a;
      camera.right = range * a;
      camera.updateProjectionMatrix();
    };
    window.addEventListener('resize', onResize);

    return () => {
      stopped = true;
      window.removeEventListener('resize', onResize);
      scene.remove(lineGroup);
      lineGroup.traverse((o) => {
        const m = o as THREE.Line;
        if (m.geometry) m.geometry.dispose();
        const mat = m.material as THREE.Material | undefined;
        if (mat) mat.dispose();
      });
      renderer.dispose();
      if (renderer.domElement.parentNode === mount) {
        mount.removeChild(renderer.domElement);
      }
    };
  }, [sweep, run]);

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        background: '#0a0d14',
        color: '#cdd3de',
        font: '12px ui-monospace, monospace',
        display: 'flex',
      }}
    >
      <div ref={mountRef} style={{ flex: 1, position: 'relative' }} />
      <div
        style={{
          width: 380,
          padding: 16,
          borderLeft: '1px solid #1f2735',
          overflowY: 'auto',
        }}
      >
        <h1 style={{ color: '#7fd6ff', fontSize: 16, margin: 0 }}>
          learn motion primitives
        </h1>
        <p style={{ opacity: 0.7, marginTop: 6 }}>
          Drive a Rapier vehicle through 48 open-loop trials, fit a 5-coefficient
          dynamics model, derive a drop-in motion-primitive library.
        </p>
        <PhaseBanner phase={phase} progress={progress} error={error} />
        {run && (
          <>
            <h2 style={{ fontSize: 12, opacity: 0.65, marginTop: 16, marginBottom: 4, textTransform: 'uppercase', letterSpacing: 1 }}>
              Learned coefficients
            </h2>
            <ParamsTable params={run.params} />
            <h2 style={{ fontSize: 12, opacity: 0.65, marginTop: 16, marginBottom: 4, textTransform: 'uppercase', letterSpacing: 1 }}>
              Fit quality (vs Rapier ground truth)
            </h2>
            <KV k="mean pos error" v={`${run.fit.meanPosError.toFixed(3)} m`} />
            <KV k="max pos error" v={`${run.fit.maxPosError.toFixed(3)} m`} />
            <KV k="loss" v={run.fit.loss.toFixed(3)} />
            <h2 style={{ fontSize: 12, opacity: 0.65, marginTop: 16, marginBottom: 4, textTransform: 'uppercase', letterSpacing: 1 }}>
              Kinematic-model gap (baseline)
            </h2>
            <KV
              k="mean pos error"
              v={run.kinematic ? `${run.kinematic.meanPosError.toFixed(3)} m` : '—'}
            />
            <KV
              k="max pos error"
              v={run.kinematic ? `${run.kinematic.maxPosError.toFixed(3)} m` : '—'}
            />
            <KV
              k="mean speed error"
              v={run.kinematic ? `${run.kinematic.meanSpeedError.toFixed(3)} m/s` : '—'}
            />
            {!run.kinematic && (
              <p style={{ opacity: 0.55, marginTop: 4, fontSize: 11 }}>
                Cache was written by another flow without the kinematic
                baseline. Click <em>re-learn</em> below to recompute.
              </p>
            )}
            <div style={{ display: 'flex', gap: 6, marginTop: 14, flexWrap: 'wrap' }}>
              <Button onClick={onDownloadParams}>download params</Button>
              <Button onClick={onDownloadLibrary}>download library</Button>
              <Button onClick={onReLearn} disabled={phase === 'collecting' || phase === 'fitting'}>
                re-learn
              </Button>
            </div>
            <p style={{ opacity: 0.55, marginTop: 10, fontSize: 11 }}>
              Cached in <code>localStorage[{PARAMS_KEY}]</code>.
            </p>
          </>
        )}
        {!run && phase === 'error' && (
          <Button onClick={() => void runLearn()}>retry</Button>
        )}
        <div style={{ marginTop: 20, opacity: 0.55, fontSize: 11, borderTop: '1px solid #1f2735', paddingTop: 10 }}>
          <div>blue = forward · orange = reverse · dashed gray = kinematic ghost</div>
        </div>
      </div>
    </div>
  );
}

function PhaseBanner({
  phase,
  progress,
  error,
}: {
  phase: Phase;
  progress: { done: number; total: number };
  error: string | null;
}) {
  let label: string;
  switch (phase) {
    case 'idle':
      label = 'Initialising…';
      break;
    case 'collecting':
      label = `Collecting trials… ${progress.done}/${progress.total || '?'}`;
      break;
    case 'fitting':
      label = 'Fitting parameters…';
      break;
    case 'done':
      label = 'Done — coefficients ready';
      break;
    case 'error':
      label = `Error: ${error ?? 'unknown'}`;
      break;
  }
  const pct = progress.total > 0 ? (progress.done / progress.total) * 100 : 0;
  return (
    <div style={{ marginTop: 12 }}>
      <div
        style={{
          padding: '6px 10px',
          background: phase === 'done' ? 'rgba(85, 220, 140, 0.12)' : 'rgba(127, 214, 255, 0.10)',
          border: '1px solid #1f2735',
          borderRadius: 6,
          color: phase === 'error' ? '#ff8aa0' : '#cdeaff',
        }}
      >
        {label}
      </div>
      {phase === 'collecting' && progress.total > 0 && (
        <div style={{ height: 4, background: '#1f2735', marginTop: 6, borderRadius: 2 }}>
          <div
            style={{
              width: `${pct}%`,
              height: '100%',
              background: '#7fd6ff',
              borderRadius: 2,
              transition: 'width 80ms linear',
            }}
          />
        </div>
      )}
    </div>
  );
}

function ParamsTable({ params }: { params: LearnedVehicleParams }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr auto', gap: '2px 12px' }}>
      <KV k="maxAccel" v={`${params.maxAccel.toFixed(3)} m/s²`} />
      <KV k="maxDecel" v={`${params.maxDecel.toFixed(3)} m/s²`} />
      <KV k="accelTau" v={`${params.accelTau.toFixed(4)} s`} />
      <KV k="understeerGain" v={params.understeerGain.toExponential(3)} />
      <KV k="lateralDrag" v={params.lateralDrag.toExponential(3)} />
    </div>
  );
}

function KV({ k, v }: { k: string; v: string }) {
  return (
    <>
      <span style={{ opacity: 0.7 }}>{k}</span>
      <span style={{ color: '#cdeaff', textAlign: 'right' }}>{v}</span>
    </>
  );
}

function Button({
  children,
  onClick,
  disabled,
}: {
  children: React.ReactNode;
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      style={{
        font: '11px ui-monospace, monospace',
        padding: '6px 10px',
        borderRadius: 6,
        border: '1px solid #2a3340',
        background: disabled ? 'rgba(20, 26, 38, 0.5)' : 'rgba(127, 214, 255, 0.14)',
        color: disabled ? '#5a6577' : '#cdeaff',
        cursor: disabled ? 'not-allowed' : 'pointer',
        letterSpacing: 0.3,
      }}
    >
      {children}
    </button>
  );
}

function downloadBlob(filename: string, content: string) {
  const blob = new Blob([content], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
