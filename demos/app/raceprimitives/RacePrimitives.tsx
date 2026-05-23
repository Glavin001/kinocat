'use client';

// Side-by-side primitive-library race. Two identical Rapier cars drive an
// identical waypoint loop; the only variable is the motion-primitive library
// their planner uses — kinematic-derived (the "before") vs learned from
// Rapier ground truth via `/learnprimitives` (the "after"). The course is
// designed so the kinematic library's blind spots (no understeer, no lateral
// drag, instant speed tracking, no acceleration limit) cost lap time:
// overshoots in the tight slalom and late braking into the 90° turn.

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import RAPIER from '@dimforge/rapier3d-compat';
import type { LearnedVehicleParams, VehicleState } from 'kinocat/agent';
import { DEFAULT_LEARNED_PARAMS } from 'kinocat/agent';
import { InMemoryNavWorld } from 'kinocat/environment';
import { MotionPrimitiveLibrary } from 'kinocat/primitives';
import { purePursuit } from 'kinocat/execute';
import {
  createRaycastVehicle,
  createGroundCollider,
  ensureRapier,
  type CarHandle,
} from 'kinocat/adapters/rapier';
import {
  createCarMeshHelper,
  syncCarMesh,
  createGroundPlaneHelper,
} from 'kinocat/adapters/three';
import {
  buildKinematicLibrary,
  buildLearnedRaceLibrary,
  buildLearnedRaceLibraryV2,
  buildRaceCourse,
  emptyMetrics,
  pickNextWaypoint,
  planThroughWaypoints,
  planRaceMultiGoal,
  RACE_AGENT,
  RACE_BOUNDS,
  RACE_PALETTE as C,
  RACE_REPLAN_BUDGET_MS,
  RACE_START_SPEEDS,
  raceControlSets,
  type RaceMetrics,
} from '../lib/race-primitives-scenarios';
import {
  buildLearnedLibrary,
  createSweepWorld,
  fitParams,
  fitParamsOnline,
  LEARN_VEHICLE_TUNING,
  runSweep,
  summariseKinematicGap,
  type TransitionSample,
} from '../lib/learn-primitives';
import { ModelLab } from '../components/ModelLab';
import {
  loadV2Model,
  saveV2Model,
  clearV2Model,
  buildV2ModelDownloadUrl,
  type PersistedV2Model,
} from '../lib/v2-model-persistence';
import type { LearnedVehicleModel } from 'kinocat/agent';

const PHYSICS_DT = 1 / 60;
const VEHICLE_SUBSTEPS = 4;
// 300ms tick (~3.3 Hz). Both cars plan in the SAME tick callback so they
// always plan at the same wall-time with the same per-car budget — fair
// by construction. CPU = 2 × RACE_REPLAN_BUDGET_MS (120) / 300 = 80%
// (was 120% with the old 500ms × 300 budget, which was over-saturated
// and animation visibly choked).
const REPLAN_INTERVAL_MS = 300;
const WHEEL_BASE = 1.6;
// Real tire grip on dry asphalt is ~9-10 m/s² (~1g). 12 keeps the cars
// physically plausible — pure-pursuit won't follow a plan that demands more
// than this, so a kinematic plan saying "take this turn at 16 m/s" gets
// clipped to ~7 m/s by the tracker. The LEARNED planner knows about this
// and plans entry speeds the tracker WON'T need to clip, so its trajectory
// executes cleanly. With the previous TRACKER_MAX_LATERAL_ACCEL=25 the
// tracker would let the car attempt impossible turns and slide off-map.
const TRACKER_MAX_LATERAL_ACCEL = 12;
// Online-learning buffer cap. ~60Hz × ~30s/lap = 1800 samples/lap; 4000
// covers ~2 laps of real driving, refit converges in well under a second.
const ONLINE_SAMPLE_CAP = 4000;
// Refit takes a few hundred ms on 4000 samples — long enough that doing it
// inline in the animation loop would visibly hitch. Defer to a microtask.
const REFIT_DEFER_MS = 0;
// Schema-versioned cache key — must match the value in
// /learnprimitives so the two demos share a single cache. Bump the
// version suffix whenever the model formulation or parameter bounds
// change in a way that would silently re-interpret old cached
// coefficients. See LearnPrimitives.tsx for the change history.
const PARAMS_KEY = 'kinocat:learned-params:v2';
const LIBRARY_KEY = 'kinocat:learned-library:v2';

type Phase = 'loading' | 'learning' | 'ready' | 'racing' | 'finished';

interface CarRuntime {
  id: 'kinematic' | 'learned';
  color: number;
  pathColor: number;
  world: RAPIER.World;
  car: CarHandle;
  carMesh: ReturnType<typeof createCarMeshHelper>;
  pathLine: THREE.Line | null;
  /** Dashed straight reference line from chassis to the next-uncleared
   *  waypoint — lets the viewer compare the planner's actual curve to the
   *  naïve "drive in a straight line to the cone" trajectory. */
  idealLine: THREE.Line;
  /** Small sphere at the pure-pursuit lookahead point — the spot on the
   *  plan the tracker is actively chasing this tick. Visualizes the
   *  difference between "where the plan goes long-term" (pathLine) and
   *  "where the car is steering toward right now" (lookaheadMarker). */
  lookaheadMarker: THREE.Mesh;
  trailLine: THREE.Line;
  trailPts: THREE.Vector3[];
  ai: {
    plan: VehicleState[] | null;
    planStartWall: number;
    loopIndex: number;
    goal: VehicleState | null;
    /** Predicted end state of the plan's first primitive (t = 0.55s ahead)
     *  recorded at plan-install time. When wall-time reaches the predicted
     *  time, we compare predicted vs actual to get the per-primitive
     *  prediction error — the honest measure of dynamics-model accuracy
     *  (unlike the position-vs-elapsed-time metric which rewards
     *  confidence over correctness on long straights). */
    predictedEnd: { state: VehicleState; dueWall: number } | null;
  };
  lib: MotionPrimitiveLibrary;
  metrics: RaceMetrics;
  /** Per-primitive prediction error (plan said you'd be at X in 0.55s,
   *  actually you ended up at Y). Honest dynamics-model accuracy metric. */
  predErrorAcc: { sumSq: number; count: number };
  /** Total waypoints cleared since race start. */
  waypointsCleared: number;
  /** Last time (perf.now ms) the car moved measurably; used to detect stalls. */
  lastMoveWall: number;
  lastPos: { x: number; z: number };
  scene: THREE.Scene;
  finishWall: number | null;
  /** Lap times completed in this race. */
  lapTimes: number[];
  /** Per-lap × per-waypoint cumulative time within the lap. After each
   *  completed lap, `sectorTimes[lapIdx][i]` = wall time the i-th gate
   *  of that lap was crossed, relative to the lap's start. */
  sectorTimes: number[][];
  /** Sector crossings recorded so far in the IN-PROGRESS lap. Pushed
   *  into `sectorTimes` on lap completion and reset. */
  currentLapSectors: number[];
  /** True when this car has finished its current lap and is holding at
   *  the start line waiting for the other car to catch up. While held,
   *  `raceTime` does not accumulate (to keep sector + lap timing pure). */
  holdingForSync: boolean;
  /** Online learner — only populated for the 'learned' car. The kinematic
   *  car has no learner (control group). */
  learner?: OnlineLearnerState;
}

interface OnlineLearnerState {
  /** Latest refit's coefficients — shown in the panel for transparency. */
  params: LearnedVehicleParams;
  /** L2-regularization anchor. If the user pre-trained offline this is the
   *  offline fit; online refits can deviate from it but only when race data
   *  provides strong evidence. Without a pre-train this is just the
   *  conservative defaults — that's why pre-training matters: it produces
   *  a meaningful prior the online learner can lean on. */
  priorParams: LearnedVehicleParams;
  /** Params CURRENTLY baked into `car.lib` (what's actually racing). May
   *  equal `params` (latest fit adopted) OR `bestParams` (rolled back
   *  after a bad refit). Updating `car.lib` MUST also update this. */
  libParams: LearnedVehicleParams;
  /** Params that produced the best lap time so far. Always reflects a
   *  configuration we KNOW works well. The car races with this whenever
   *  the latest refit is significantly worse. */
  bestParams: LearnedVehicleParams;
  /** Best lap time recorded with this learner instance. NaN until the
   *  first lap completes. */
  bestLapTime: number;
  /** Lap number (1-indexed) that produced `bestLapTime`. */
  bestLapNumber: number;
  /** True when `car.lib` was just reverted to `bestParams` because the
   *  most recent lap was significantly worse than the best so far. The
   *  HUD surfaces this with a "USING BEST" badge so the user knows the
   *  live coef panel values aren't what's racing. Cleared when the
   *  latest refit is adopted again. */
  rollbackActive: boolean;
  samples: TransitionSample[];
  refitCount: number;
  /** Refits where the rollback fired (latest refit hurt → reverted). */
  rollbackCount: number;
  lastFitMs: number;
  lastMeanError: number;
  /** True while a refit is in flight (debounce). */
  refitting: boolean;
}

function loadLearnedParams(): LearnedVehicleParams | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = window.localStorage.getItem(PARAMS_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as { params?: LearnedVehicleParams };
    return parsed.params ?? null;
  } catch {
    return null;
  }
}

export default function RacePrimitives() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [phase, setPhase] = useState<Phase>('loading');
  const [learnProgress, setLearnProgress] = useState<{
    done: number;
    total: number;
    /** Last trial parameters, for the overlay's "currently testing" line. */
    curvature?: number;
    targetSpeed?: number;
    startSpeed?: number;
  }>({ done: 0, total: 0 });
  const [params, setParams] = useState<LearnedVehicleParams | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<{
    kinematic: RaceMetrics;
    learned: RaceMetrics;
  }>({ kinematic: emptyMetrics(), learned: emptyMetrics() });
  const [learner, setLearner] = useState<LearnerSnapshot | null>(null);
  const [winner, setWinner] = useState<'kinematic' | 'learned' | 'tie' | null>(null);
  // v2 model state (Phase-2 addition). When `useV2 && v2Model != null`, the
  // learned car's library is built from v2 instead of legacy.
  const [v2Model, setV2Model] = useState<LearnedVehicleModel | null>(null);
  const [v2Meta, setV2Meta] = useState<PersistedV2Model['meta'] | null>(null);
  const [useV2, setUseV2State] = useState(false);
  // Persist the toggle so the user doesn't have to re-enable v2 on every
  // page reload after they've trained a model.
  const setUseV2 = (v: boolean) => {
    setUseV2State(v);
    if (typeof window !== 'undefined') {
      try {
        window.localStorage.setItem('kinocat:v2-toggle:v1', v ? '1' : '0');
      } catch { /* quota; ignore */ }
    }
  };
  const v2Active = useV2 && v2Model !== null;

  const sceneRef = useRef<{
    cleanup: () => void;
    start: () => void;
    stop: () => void;
    reset: () => void;
  } | null>(null);

  // On mount: try to load cached pre-trained params as a warm start for the
  // online learner. If none are cached, the learner starts from
  // DEFAULT_LEARNED_PARAMS — both cars still race fine, the learned car just
  // begins with no prior knowledge and learns purely from race data.
  useEffect(() => {
    const p = loadLearnedParams();
    setParams(p ?? DEFAULT_LEARNED_PARAMS);
    setPhase('ready');
    // Also try to load a previously-trained v2 model so the user can
    // immediately toggle "use v2" without retraining each visit.
    const v2 = loadV2Model();
    if (v2) {
      setV2Model(v2.model);
      setV2Meta(v2.meta);
    }
    if (typeof window !== 'undefined') {
      try {
        const v = window.localStorage.getItem('kinocat:v2-toggle:v1');
        if (v === '1') setUseV2State(true);
      } catch { /* ignore */ }
    }
  }, []);

  // Mount the Three.js + Rapier scene as soon as initial params are decided.
  // Also re-mounts when the v2 toggle changes (rebuilds the learned car's
  // primitive library from the v2 model or back to the legacy path).
  useEffect(() => {
    if (!params || phase === 'loading' || phase === 'learning') return;
    const mount = containerRef.current;
    if (!mount) return;
    let disposed = false;
    let cleanup: (() => void) | null = null;
    // Clear stale online-learner UI state from a previous race (the v2
    // scene won't emit onLearner so any prior snapshot would otherwise
    // linger).
    setLearner(null);
    (async () => {
      try {
        await ensureRapier();
        if (disposed) return;
        const learnedLibraryOverride = v2Active
          ? buildLearnedRaceLibraryV2(v2Model!)
          : undefined;
        const setup = await setupScene(mount, params, {
          onMetrics: (km, lm) => setMetrics({ kinematic: km, learned: lm }),
          onLearner: (snap) => setLearner(snap),
          onFinish: (w) => {
            setWinner(w);
            setPhase('finished');
          },
        }, { learnedLibraryOverride });
        sceneRef.current = setup;
        cleanup = setup.cleanup;
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      disposed = true;
      cleanup?.();
      sceneRef.current = null;
    };
    // Re-mount when params, v2 toggle, or v2 model identity change.
  }, [params, useV2, v2Model, phase === 'learning' ? 'pending' : 'mounted']); // eslint-disable-line react-hooks/exhaustive-deps

  async function runInlineLearn() {
    setError(null);
    setPhase('learning');
    setLearnProgress({ done: 0, total: 0 });
    try {
      const sw = await createSweepWorld(RACE_AGENT);
      try {
        const data = await runSweep(sw, {
          agent: RACE_AGENT,
          startSpeeds: RACE_START_SPEEDS,
          controlSets: raceControlSets(RACE_AGENT),
          onProgress: (p) =>
            setLearnProgress({
              done: p.done,
              total: p.total,
              curvature: p.curvature,
              targetSpeed: p.targetSpeed,
              startSpeed: p.startSpeed,
            }),
          yieldEvery: 4,
          yieldFn: () => new Promise((r) => setTimeout(r, 0)),
        });
        const fit = fitParams(data);
        const kinematic = summariseKinematicGap(data);
        const lib = buildLearnedRaceLibrary(fit.params);
        // Cache matches /learnprimitives' CachedRun shape so both demos can
        // share the same localStorage entry.
        const cached = {
          params: fit.params,
          fit: {
            meanPosError: fit.meanPosError,
            maxPosError: fit.maxPosError,
            loss: fit.loss,
          },
          kinematic,
          createdAt: Date.now(),
        };
        window.localStorage.setItem(PARAMS_KEY, JSON.stringify(cached));
        window.localStorage.setItem(LIBRARY_KEY, lib.toJSON());
        setParams(fit.params);
        setPhase('ready');
      } finally {
        sw.dispose();
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setPhase('ready');
    }
  }

  function startRace() {
    setWinner(null);
    setMetrics({ kinematic: emptyMetrics(), learned: emptyMetrics() });
    sceneRef.current?.reset();
    sceneRef.current?.start();
    setPhase('racing');
  }

  function stopRace() {
    sceneRef.current?.stop();
    setPhase('finished');
  }

  function resetRace() {
    sceneRef.current?.reset();
    setWinner(null);
    setMetrics({ kinematic: emptyMetrics(), learned: emptyMetrics() });
    setLearner(null);
    setPhase('ready');
  }

  /** Wipe localStorage cache and rebuild the scene with default priors.
   *  Use this when the live coefs are pinned to bounds matching the cached
   *  prior (a sign the prior is stale / poisoned, e.g. after a bounds or
   *  model change), or just to start fresh. */
  function clearCache() {
    if (typeof window !== 'undefined') {
      window.localStorage.removeItem(PARAMS_KEY);
      window.localStorage.removeItem(LIBRARY_KEY);
    }
    setParams(DEFAULT_LEARNED_PARAMS);
    setWinner(null);
    setMetrics({ kinematic: emptyMetrics(), learned: emptyMetrics() });
    setLearner(null);
    setPhase('ready');
  }

  function onV2Trained(model: LearnedVehicleModel, diag: import('kinocat/learning').ModelDiagnostics, trialsUsed: number) {
    const mid1s = diag.openLoopDivergence.find((r) => r.tSec >= 1.0);
    const legacyMid = diag.baselines['legacyV1']?.find((r) => r.tSec >= 1.0);
    const kinMid = diag.baselines['kinematic']?.find((r) => r.tSec >= 1.0);
    const meta: PersistedV2Model['meta'] = {
      trialsUsed,
      openLoopRmsAt1s: mid1s?.posRms ?? 0,
      legacyRmsAt1s: legacyMid?.posRms,
      kinematicRmsAt1s: kinMid?.posRms,
      createdAt: Date.now(),
    };
    saveV2Model(model, meta);
    setV2Model(model);
    setV2Meta(meta);
  }

  function onV2Clear() {
    clearV2Model();
    setV2Model(null);
    setV2Meta(null);
    setUseV2(false);
  }

  function onV2Export() {
    if (!v2Model || !v2Meta) return;
    const url = buildV2ModelDownloadUrl(v2Model, v2Meta);
    const a = document.createElement('a');
    a.href = url;
    a.download = `kinocat-v2-model-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        background: '#0a0d14',
        color: '#cdd3de',
        font: '12px ui-monospace, monospace',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <TopBar
        phase={phase}
        learnProgress={learnProgress}
        winner={winner}
        params={params}
        error={error}
        v2Active={v2Active}
        onLearn={runInlineLearn}
        onStart={startRace}
        onStop={stopRace}
        onReset={resetRace}
        onClearCache={clearCache}
      />
      <div style={{ flex: 1, position: 'relative' }}>
        <div ref={containerRef} style={{ position: 'absolute', inset: 0 }} />
        <MetricsOverlay
          metrics={metrics}
          winner={winner}
          holding={{
            kinematic: learner?.kinematicHolding ?? false,
            learned: learner?.learnedHolding ?? false,
          }}
          lapTimes={{
            kinematic: learner?.kinematicLapTimes ?? [],
            learned: learner?.learnedLapTimes ?? [],
          }}
          rollbackActive={learner?.rollbackActive ?? false}
          bestLapNumber={learner?.bestLapNumber ?? 0}
        />
        {(phase === 'racing' || phase === 'finished') && learner && (
          <LearnerPanel snap={learner} v2Active={v2Active} v2Meta={v2Meta} />
        )}
        {phase === 'learning' && (
          <PretrainOverlay progress={learnProgress} />
        )}
        <ModelLab
          onTrained={onV2Trained}
          loadedMeta={v2Meta}
          onClearLoaded={v2Meta ? onV2Clear : undefined}
          onExport={v2Model ? onV2Export : undefined}
          useV2={useV2}
          onToggleUseV2={setUseV2}
          hasV2Model={v2Model !== null}
        />
      </div>
    </div>
  );
}

function PretrainOverlay({
  progress,
}: {
  progress: {
    done: number;
    total: number;
    curvature?: number;
    targetSpeed?: number;
    startSpeed?: number;
  };
}) {
  const pct = progress.total > 0 ? (progress.done / progress.total) * 100 : 0;
  const trial =
    progress.curvature !== undefined && progress.targetSpeed !== undefined
      ? `κ = ${progress.curvature >= 0 ? '+' : ''}${progress.curvature.toFixed(3)} rad/m · target speed ${progress.targetSpeed.toFixed(1)} m/s · start ${(progress.startSpeed ?? 0).toFixed(1)} m/s`
      : 'initialising…';
  return (
    <div
      style={{
        position: 'absolute',
        inset: 0,
        background: 'rgba(10, 13, 20, 0.85)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 50,
      }}
    >
      <div
        style={{
          minWidth: 420,
          padding: 24,
          background: '#0d1119',
          border: '1px solid #1f2735',
          borderRadius: 10,
          color: '#cdd3de',
          font: '12px ui-monospace, monospace',
        }}
      >
        <div style={{ color: '#7fd6ff', fontWeight: 700, fontSize: 14, marginBottom: 8 }}>
          Pre-training the learned model (offline)
        </div>
        <div style={{ opacity: 0.7, marginBottom: 14, lineHeight: 1.5 }}>
          The learner is driving a Rapier vehicle through a deliberate sweep
          of {progress.total || '…'} controlled trials (brake to stop →
          accelerate to target speed → apply test controls → record) so it
          covers braking, hard cornering, and reverse — situations that
          race driving alone wouldn't sample. Result feeds the online
          learner as a regularization prior.
        </div>
        <div style={{ marginBottom: 8 }}>
          <span style={{ color: '#cdeaff' }}>
            trial {progress.done}/{progress.total || '?'}
          </span>
          {progress.total > 0 && (
            <span style={{ opacity: 0.55, marginLeft: 8 }}>
              ({pct.toFixed(0)}%)
            </span>
          )}
        </div>
        <div
          style={{
            height: 8,
            background: '#1f2735',
            borderRadius: 4,
            overflow: 'hidden',
            marginBottom: 12,
          }}
        >
          <div
            style={{
              width: `${pct}%`,
              height: '100%',
              background: '#7fd6ff',
              transition: 'width 100ms linear',
            }}
          />
        </div>
        <div style={{ opacity: 0.7, fontSize: 11 }}>
          currently testing: <span style={{ color: '#cdeaff' }}>{trial}</span>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Scene setup.

interface SceneCallbacks {
  onMetrics: (k: RaceMetrics, l: RaceMetrics) => void;
  onLearner: (state: LearnerSnapshot) => void;
  onFinish: (w: 'kinematic' | 'learned' | 'tie') => void;
}

interface LearnerSnapshot {
  /** Latest refit (shown in the live coef panel). */
  params: LearnedVehicleParams;
  /** Anchor the live coefficients drifted FROM (pre-train fit, or
   *  DEFAULT_LEARNED_PARAMS without pre-train). Each online refit starts
   *  from here, not from `params`, to prevent drift accumulation. */
  priorParams: LearnedVehicleParams;
  /** Params that produced the best lap so far. When `rollbackActive` is
   *  true, this is what's actually racing (not `params`). */
  bestParams: LearnedVehicleParams;
  bestLapTime: number;
  bestLapNumber: number;
  /** True when `car.lib` is built from `bestParams` because the last lap
   *  was significantly slower than the best. */
  rollbackActive: boolean;
  rollbackCount: number;
  refitCount: number;
  sampleCount: number;
  lastFitMs: number;
  lastMeanError: number;
  kinematicLapTimes: number[];
  learnedLapTimes: number[];
  /** Per-lap cumulative sector times (sectorTimes[lap][gate]). */
  kinematicSectors: number[][];
  learnedSectors: number[][];
  /** Number of waypoints in one full lap — for sector-count labelling. */
  sectorsPerLap: number;
  kinematicHolding: boolean;
  learnedHolding: boolean;
}

interface SceneOptions {
  /** When supplied, overrides the legacy `buildLearnedRaceLibrary(params)`
   *  for the LEARNED car's primitive library. Used to demo the v2 model
   *  with the same race pipeline. Online refitting is suppressed while this
   *  override is active (the v2 model is trained offline; mixing online
   *  refits of the legacy 5-param model would race a stale v1 library while
   *  the v2-derived primitive library is what the planner is searching). */
  learnedLibraryOverride?: MotionPrimitiveLibrary;
}

async function setupScene(
  mount: HTMLDivElement,
  params: LearnedVehicleParams,
  cb: SceneCallbacks,
  options: SceneOptions = {},
): Promise<{
  cleanup: () => void;
  start: () => void;
  stop: () => void;
  reset: () => void;
}> {
  const W = mount.clientWidth;
  const H = mount.clientHeight;

  // ---- Three.js renderer (shared) ----
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(W, H);
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setScissorTest(true);
  mount.appendChild(renderer.domElement);

  // ---- Shared course ----
  const course = buildRaceCourse();
  const navWorld = new InMemoryNavWorld(course.polygons, course.obstacles);
  const kinematicLib = buildKinematicLibrary();
  const initialLearnerParams = params ?? DEFAULT_LEARNED_PARAMS;
  // If pre-train ran (params differs from DEFAULT_LEARNED_PARAMS),
  // give the learned car its pre-trained library from lap 1 — otherwise
  // the first 5 online refits would have to rediscover what pre-train
  // already learned, and pre-train would look like it does nothing. With
  // no pre-train, both cars start identical (kinematicLib) and the
  // learned car learns from race data alone.
  const hasPreTrain =
    initialLearnerParams !== DEFAULT_LEARNED_PARAMS &&
    !paramsEqual(initialLearnerParams, DEFAULT_LEARNED_PARAMS);
  const initialLearnedLib = options.learnedLibraryOverride
    ?? (hasPreTrain ? buildLearnedRaceLibrary(initialLearnerParams) : kinematicLib);
  const v2Override = Boolean(options.learnedLibraryOverride);

  // ---- Per-car setup ----
  function makeCar(
    id: 'kinematic' | 'learned',
    lib: MotionPrimitiveLibrary,
    color: number,
    pathColor: number,
    learner: OnlineLearnerState | undefined,
  ): CarRuntime {
    const world = new RAPIER.World({ x: 0, y: -9.81, z: 0 });
    createGroundCollider(world, {
      bounds: RACE_BOUNDS,
      pad: 20,
      friction: 1.5,
    });
    const car = createRaycastVehicle(world, {
      id: `race-${id}`,
      position: { x: course.spawn.x, z: course.spawn.z },
      heading: course.spawn.heading,
      ...LEARN_VEHICLE_TUNING,
    });
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(C.bg);
    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const sun = new THREE.DirectionalLight(0xffffff, 0.9);
    sun.position.set(40, 100, 20);
    scene.add(sun);
    scene.add(createGroundPlaneHelper({ bounds: RACE_BOUNDS, color: 0x141a26 }));
    // Waypoint cones — shared geometry per car scene.
    for (let i = 0; i < course.waypoints.length; i++) {
      const wp = course.waypoints[i]!;
      const cone = new THREE.Mesh(
        new THREE.ConeGeometry(0.8, 2.2, 14),
        new THREE.MeshStandardMaterial({
          color: C.gate,
          emissive: C.gate,
          emissiveIntensity: 0.25,
        }),
      );
      cone.position.set(wp.x, 1.1, wp.z);
      scene.add(cone);
      // Number label via a small ring at the base.
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(1.2, 1.8, 24),
        new THREE.MeshBasicMaterial({
          color: 0xffd070,
          transparent: true,
          opacity: 0.35,
          side: THREE.DoubleSide,
        }),
      );
      ring.rotation.x = -Math.PI / 2;
      ring.position.set(wp.x, 0.05, wp.z);
      scene.add(ring);
    }
    // Start marker.
    const start = new THREE.Mesh(
      new THREE.RingGeometry(1.5, 2.2, 24),
      new THREE.MeshBasicMaterial({ color: C.startMarker, side: THREE.DoubleSide }),
    );
    start.rotation.x = -Math.PI / 2;
    start.position.set(course.spawn.x, 0.06, course.spawn.z);
    scene.add(start);
    // Car mesh.
    const carMesh = createCarMeshHelper({ color });
    scene.add(carMesh.group);
    // Actual trail line (grows during race).
    const trailGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(course.spawn.x, 0.15, course.spawn.z),
    ]);
    const trail = new THREE.Line(
      trailGeo,
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.55 }),
    );
    scene.add(trail);
    // Ideal/reference line — dashed straight line from car to next waypoint.
    // Pure visual reference; never followed.
    const idealMat = new THREE.LineDashedMaterial({
      color: 0xffffff,
      dashSize: 1.0,
      gapSize: 0.8,
      transparent: true,
      opacity: 0.45,
    });
    const idealGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(course.spawn.x, 0.25, course.spawn.z),
      new THREE.Vector3(course.spawn.x, 0.25, course.spawn.z),
    ]);
    const ideal = new THREE.Line(idealGeo, idealMat);
    ideal.computeLineDistances();
    scene.add(ideal);
    // Pure-pursuit lookahead marker — small bright sphere in the car's
    // color, sitting at the spot on the plan the tracker is steering
    // toward this tick.
    const lookaheadMarker = new THREE.Mesh(
      new THREE.SphereGeometry(0.5, 12, 8),
      new THREE.MeshBasicMaterial({ color: pathColor, transparent: true, opacity: 0.85 }),
    );
    lookaheadMarker.position.set(course.spawn.x, 0.5, course.spawn.z);
    scene.add(lookaheadMarker);
    return {
      id,
      color,
      pathColor,
      world,
      car,
      carMesh,
      pathLine: null,
      idealLine: ideal,
      lookaheadMarker,
      trailLine: trail,
      trailPts: [new THREE.Vector3(course.spawn.x, 0.15, course.spawn.z)],
      ai: {
        plan: null,
        planStartWall: performance.now(),
        loopIndex: 0,
        goal: null,
        predictedEnd: null,
      },
      lib,
      metrics: emptyMetrics(),
      predErrorAcc: { sumSq: 0, count: 0 },
      waypointsCleared: 0,
      lastMoveWall: performance.now(),
      lastPos: { x: course.spawn.x, z: course.spawn.z },
      scene,
      finishWall: null,
      lapTimes: [],
      sectorTimes: [],
      currentLapSectors: [],
      holdingForSync: false,
      learner,
    };
  }

  const kinematic = makeCar('kinematic', kinematicLib, C.kinematic, C.kinematicPath, undefined);
  // Learned car starts with the pre-trained library when available, else
  // with the kinematic library. Either way, it refines per-lap online
  // using race data via fitParamsOnline. When the v2 override is active,
  // online refitting is suppressed (passing `undefined`) so the offline-
  // trained v2 library is what's actually driving — mixing in an online
  // refit of the legacy 5-param model would silently shadow the v2 lib.
  const learnerState: OnlineLearnerState | undefined = v2Override
    ? undefined
    : {
        params: initialLearnerParams,
        priorParams: initialLearnerParams,
        libParams: initialLearnerParams,
        bestParams: initialLearnerParams,
        bestLapTime: Number.NaN,
        bestLapNumber: 0,
        rollbackActive: false,
        samples: [],
        refitCount: 0,
        rollbackCount: 0,
        lastFitMs: 0,
        lastMeanError: 0,
        refitting: false,
      };
  const learned = makeCar('learned', initialLearnedLib, C.learned, C.learnedPath, learnerState);

  // ---- Cameras: one perspective chase-cam per car. Stays slightly above and
  // behind the chassis so the user can see the wheels, suspension, and the
  // car's relationship to upcoming waypoints — i.e. a real 3D Rapier vehicle.
  function makeChaseCamera(): THREE.PerspectiveCamera {
    const aspect = (W / 2) / H;
    const cam = new THREE.PerspectiveCamera(55, aspect, 0.3, 600);
    cam.position.set(course.spawn.x - 16, 10, course.spawn.z - 4);
    cam.lookAt(course.spawn.x, 1, course.spawn.z);
    return cam;
  }
  const camK = makeChaseCamera();
  const camL = makeChaseCamera();

  // ---- Settle both vehicles ----
  for (const car of [kinematic, learned]) {
    for (let i = 0; i < 30; i++) {
      car.car.applyControls({ steer: 0, throttle: 0, brake: 0 });
      car.world.timestep = PHYSICS_DT;
      car.car.vehicle.updateVehicle(PHYSICS_DT);
      car.world.step();
    }
    car.car.teleport({
      x: course.spawn.x,
      z: course.spawn.z,
      heading: course.spawn.heading,
    });
  }

  // ---- State ----
  let running = false;
  let raceStartWall = 0;

  function resetCar(car: CarRuntime) {
    // Race START is the only allowed teleport — both cars are placed on the
    // start line at rest, then physically driven by their planner from there.
    car.car.teleport({
      x: course.spawn.x,
      z: course.spawn.z,
      heading: course.spawn.heading,
    });
    car.ai.plan = null;
    car.ai.loopIndex = 0;
    car.ai.goal = null;
    car.ai.predictedEnd = null;
    car.metrics = emptyMetrics();
    car.predErrorAcc = { sumSq: 0, count: 0 };
    car.waypointsCleared = 0;
    car.lapTimes = [];
    car.sectorTimes = [];
    car.currentLapSectors = [];
    car.holdingForSync = false;
    car.lastMoveWall = performance.now();
    car.lastPos = { x: course.spawn.x, z: course.spawn.z };
    car.finishWall = null;
    car.trailPts = [new THREE.Vector3(course.spawn.x, 0.15, course.spawn.z)];
    car.trailLine.geometry.dispose();
    car.trailLine.geometry = new THREE.BufferGeometry().setFromPoints(car.trailPts);
    if (car.pathLine) {
      car.scene.remove(car.pathLine);
      car.pathLine.geometry.dispose();
      (car.pathLine.material as THREE.Material).dispose();
      car.pathLine = null;
    }
    car.lookaheadMarker.visible = false;
  }

  /** Replan = planning only (no waypoint advancement, no lap detection —
   *  those happen at 60Hz in `stepCar` so lap times aren't quantized to
   *  the 500ms replan interval). Reads the car's current state and the
   *  current `loopIndex` (which stepCar maintains), runs a SINGLE multi-
   *  goal A* through the next N waypoints, and records the predicted
   *  end of the first primitive for the prediction-error metric. */
  function replan(car: CarRuntime, now: number): void {
    if (car.holdingForSync) return;
    const state = car.car.readState(now);
    car.ai.goal = course.waypoints[car.ai.loopIndex]!;
    // Lookahead: number of consecutive gates the planner sees per replan.
    // Multi-goal A* solves ONE search over the (chassis × gate-index) joint
    // state space, so the planner GLOBALLY trades off entries to gate i
    // against exits toward gate i+1, i+2 — proper racing-line behavior
    // (wide entry / late apex / early exit) emerges from the time-cost
    // search across all N gates simultaneously.
    //
    // Trade-off: the joint state space is ~N× larger than single-gate, so
    // 3 gates @ 120ms is roughly the practical cap on this course. Bumping
    // to 5 caused per-replan timeouts in testing (the chained-per-gate
    // approach handled 5 because each segment was an independent smaller
    // search; the global version is strictly more work per node).
    const PLAN_LOOKAHEAD_COUNT = 3;
    const gates: VehicleState[] = [];
    for (let i = 0; i < PLAN_LOOKAHEAD_COUNT; i++) {
      const idx = (car.ai.loopIndex + i) % course.waypoints.length;
      gates.push({ ...course.waypoints[idx]!, t: 0 });
    }
    const res = planRaceMultiGoal({
      state: { ...state, t: 0 },
      gates,
      lib: car.lib,
      polygons: course.polygons,
      obstacles: course.obstacles,
      world: navWorld,
      deadlineMs: RACE_REPLAN_BUDGET_MS,
    });
    if (res.found && res.path.length > 1) {
      car.ai.plan = res.path;
      car.ai.planStartWall = now;
      // Record the predicted state at the FIRST primitive's end (t≈0.55s
      // ahead in plan time). When wall-time reaches that boundary, the
      // step loop computes the prediction error: |actual - predicted|.
      // This is the honest dynamics-model accuracy metric — kinematic
      // overestimates how far the car will travel in 0.55s, learned
      // matches it. (The OLD "tracking error" rewarded confidence over
      // correctness on long straights because pure-pursuit's lookahead
      // chases position regardless of planned speed.)
      const firstEnd = res.path.find((p) => p.t > 0.05) ?? res.path[res.path.length - 1]!;
      car.ai.predictedEnd = {
        state: firstEnd,
        dueWall: now + firstEnd.t * 1000,
      };
      // Replace path line.
      if (car.pathLine) {
        car.scene.remove(car.pathLine);
        car.pathLine.geometry.dispose();
        (car.pathLine.material as THREE.Material).dispose();
      }
      const pts = res.path.map((p: VehicleState) => new THREE.Vector3(p.x, 0.4, p.z));
      car.pathLine = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineBasicMaterial({
          color: car.pathColor,
          transparent: true,
          opacity: 0.85,
        }),
      );
      car.scene.add(car.pathLine);
    }
  }

  function stepCar(car: CarRuntime, now: number, dt: number): void {
    // Read state BEFORE the tick so we can record the transition for online
    // learning (sample = state_before, controls, dt, state_after) and detect
    // waypoint crossings precisely at 60Hz.
    const stateBefore = car.car.readState(now);

    // ---- Waypoint advance + lap detection (60Hz, was in replan @ 2Hz).
    // Detecting at 60Hz means lap times are accurate to ~16ms instead of
    // ~500ms, which kills the consistent ±0.50s delta artifact.
    if (running && !car.holdingForSync) {
      const pick = pickNextWaypoint(
        { ...stateBefore, t: 0 },
        course.waypoints,
        car.ai.loopIndex,
      );
      if (pick.advanced) {
        car.waypointsCleared++;
        car.ai.loopIndex = pick.nextIndex;
        // F1-style sector timing: record cumulative time since lap start.
        const sectorTime = car.metrics.raceTime - car.metrics.lapStartTime;
        car.currentLapSectors.push(sectorTime);
        // Lap completion = wrapping past index 0.
        if (car.waypointsCleared % course.waypoints.length === 0) {
          const lapEnd = car.metrics.raceTime;
          const lap = lapEnd - car.metrics.lapStartTime;
          car.metrics.laps++;
          car.metrics.lastLapTime = lap;
          car.metrics.bestLapTime = Number.isFinite(car.metrics.bestLapTime)
            ? Math.min(car.metrics.bestLapTime, lap)
            : lap;
          car.metrics.lapStartTime = lapEnd;
          car.lapTimes.push(lap);
          car.sectorTimes.push(car.currentLapSectors.slice());
          car.currentLapSectors = [];
          // Hold at the next waypoint until the other car catches up.
          car.holdingForSync = true;
          // Online learning: refit on the lap's accumulated transitions.
          if (car.learner && car.learner.samples.length > 50) {
            scheduleRefit(car);
          }
        }
      }
    }

    // ---- Apply controls (pure-pursuit, OR sync-hold brake).
    let recordedControls: [number, number] | null = null;
    if (car.holdingForSync) {
      // Brake firmly to a stop and hold. Sync hold should be brief — the
      // other car catches up within seconds — but firm braking ensures the
      // car doesn't drift through the next gate during the wait.
      car.car.applyControls({ steer: 0, throttle: 0, brake: 1 });
      car.metrics.liveControls = { steer: 0, throttle: 0, brake: 1, targetSpeed: 0 };
    } else if (car.ai.plan && car.ai.plan.length > 1) {
      const elapsed = (now - car.ai.planStartWall) / 1000;
      const live = trimPlan(car.ai.plan, elapsed);
      if (live.length >= 2) {
        // purePursuit returns targetSpeed (which planToAckermannControls
        // strips) — needed as the second component of (κ, v_target)
        // for online learning samples.
        const cmd = purePursuit(stateBefore, live, {
          lookaheadMin: 3,
          lookaheadGain: 0.45,
          lookaheadMax: 14,
          maxLateralAccel: TRACKER_MAX_LATERAL_ACCEL,
          maxAccel: 6,
          maxDecel: 8,
          cruiseSpeed: RACE_AGENT.maxSpeed,
          goalTolerance: 2,
          minTurnRadius: RACE_AGENT.minTurnRadius,
        });
        const steer = -Math.atan(cmd.steering * (2 * WHEEL_BASE));
        car.car.applyControls({ steer, throttle: cmd.throttle, brake: cmd.brake });
        recordedControls = [cmd.steering, cmd.targetSpeed];
        car.metrics.liveControls = {
          steer: cmd.steering, throttle: cmd.throttle, brake: cmd.brake,
          targetSpeed: cmd.targetSpeed,
        };
        // Lookahead marker — bright sphere at the spot the tracker is
        // chasing this tick.
        car.lookaheadMarker.position.set(cmd.lookahead.x, 0.5, cmd.lookahead.z);
        car.lookaheadMarker.visible = true;
      } else {
        car.car.applyControls({ steer: 0, throttle: 0.2, brake: 0 });
        recordedControls = [0, 5];
        car.metrics.liveControls = { steer: 0, throttle: 0.2, brake: 0, targetSpeed: 5 };
        car.lookaheadMarker.visible = false;
      }
    } else {
      car.car.applyControls({ steer: 0, throttle: 0.2, brake: 0 });
      recordedControls = [0, 5];
      car.metrics.liveControls = { steer: 0, throttle: 0.2, brake: 0, targetSpeed: 5 };
      car.lookaheadMarker.visible = false;
    }

    // ---- Sub-stepped physics.
    const subDt = dt / VEHICLE_SUBSTEPS;
    car.world.timestep = subDt;
    const filter = RAPIER.QueryFilterFlags.EXCLUDE_DYNAMIC;
    for (let s = 0; s < VEHICLE_SUBSTEPS; s++) {
      car.car.vehicle.updateVehicle(subDt, filter);
      car.world.step();
    }
    const after = car.car.readState(now);

    // ---- Online learning: append transition (only while actually driving;
    // skip sync holds where controls are artificial braking).
    if (car.learner && recordedControls && running && !car.holdingForSync) {
      car.learner.samples.push({
        state: stateBefore,
        controls: recordedControls,
        dt,
        next: after,
      });
      while (car.learner.samples.length > ONLINE_SAMPLE_CAP) {
        car.learner.samples.shift();
      }
    }

    // ---- Prediction-error metric: when wall-time reaches the predicted
    // primitive end, compare actual vs predicted. This is the HONEST
    // dynamics-model accuracy: how well does the plan predict where the
    // car will physically be 0.55s ahead? (Replaces the previous
    // "tracking error" which mostly measured pure-pursuit's geometric
    // chase and rewarded confident plans.)
    if (car.ai.predictedEnd && now >= car.ai.predictedEnd.dueWall) {
      const p = car.ai.predictedEnd.state;
      const dx = after.x - p.x;
      const dz = after.z - p.z;
      car.predErrorAcc.sumSq += dx * dx + dz * dz;
      car.predErrorAcc.count++;
      const n = car.predErrorAcc.count;
      car.metrics.trackingErrorRms = Math.sqrt(car.predErrorAcc.sumSq / n);
      car.ai.predictedEnd = null;
    }

    // ---- Metrics + visuals.
    syncCarMesh(car.carMesh.group, after);
    {
      const wp = course.waypoints[car.ai.loopIndex]!;
      const pts = [
        new THREE.Vector3(after.x, 0.25, after.z),
        new THREE.Vector3(wp.x, 0.25, wp.z),
      ];
      car.idealLine.geometry.dispose();
      car.idealLine.geometry = new THREE.BufferGeometry().setFromPoints(pts);
      car.idealLine.computeLineDistances();
    }
    car.metrics.peakSpeed = Math.max(car.metrics.peakSpeed, Math.abs(after.speed));
    // Race-time accumulates only while DRIVING; sync holds are paused so
    // they don't pollute lap-time measurement.
    if (running && !car.holdingForSync) car.metrics.raceTime += dt;
    car.metrics.waypointsCleared = car.waypointsCleared;
    car.metrics.laps = car.lapTimes.length;
    // Trail.
    const lastPt = car.trailPts[car.trailPts.length - 1]!;
    if (Math.hypot(after.x - lastPt.x, after.z - lastPt.z) > 0.4) {
      car.trailPts.push(new THREE.Vector3(after.x, 0.15, after.z));
      if (car.trailPts.length > 1000) car.trailPts.shift();
      car.trailLine.geometry.dispose();
      car.trailLine.geometry = new THREE.BufferGeometry().setFromPoints(car.trailPts);
    }
    // Stall guard: only when actually driving (not holding for sync).
    if (running && !car.holdingForSync) {
      if (Math.hypot(after.x - car.lastPos.x, after.z - car.lastPos.z) > 0.5) {
        car.lastMoveWall = now;
        car.lastPos = { x: after.x, z: after.z };
      } else if (now - car.lastMoveWall > 4000) {
        const wp = course.waypoints[car.ai.loopIndex]!;
        car.car.teleport({ x: wp.x, z: wp.z, heading: wp.heading });
        car.ai.plan = null;
        car.lastMoveWall = now;
        car.lastPos = { x: wp.x, z: wp.z };
      }
    }
  }

  /** Defer the (CPU-heavy) refit to a microtask so the animation loop
   *  doesn't hitch. Only one refit at a time per car.
   *
   *  Three things happen each call:
   *  1. **Lap evaluation**: did the just-finished lap (driven with
   *     `libParams`) beat the best so far? If yes, promote `libParams`
   *     to `bestParams`/`bestLapTime`.
   *  2. **Refit**: fit the 5 coefficients to the current sample buffer,
   *     ALWAYS starting from `priorParams` (not the drifted current
   *     params — that's how drift used to accumulate across laps).
   *  3. **Adopt or rollback**: if the just-finished lap was much worse
   *     than the best, revert `car.lib` to the best-known params so
   *     racing performance never regresses. Otherwise adopt the new fit. */
  function scheduleRefit(car: CarRuntime): void {
    if (!car.learner || car.learner.refitting) return;
    car.learner.refitting = true;
    const samplesSnapshot = car.learner.samples.slice();
    const learner = car.learner;
    // Evaluate the lap that JUST ENDED (driven with `libParams`) BEFORE
    // running the refit. This is the only honest moment to credit the
    // params that produced this lap.
    const justFinishedLap =
      car.lapTimes.length > 0 ? car.lapTimes[car.lapTimes.length - 1]! : Number.NaN;
    if (
      Number.isFinite(justFinishedLap) &&
      (!Number.isFinite(learner.bestLapTime) || justFinishedLap < learner.bestLapTime)
    ) {
      // New best — record the params that produced it.
      learner.bestLapTime = justFinishedLap;
      learner.bestParams = learner.libParams;
      learner.bestLapNumber = car.lapTimes.length;
    }
    // Hysteresis: 0.5s slop prevents oscillation between "adopt" and
    // "rollback" from lap-to-lap noise.
    const ROLLBACK_HYSTERESIS_S = 0.5;
    const wasMuchWorse =
      Number.isFinite(justFinishedLap) &&
      Number.isFinite(learner.bestLapTime) &&
      justFinishedLap > learner.bestLapTime + ROLLBACK_HYSTERESIS_S;

    setTimeout(() => {
      if (!car.learner) return;
      const t0 = performance.now();
      const fit = fitParamsOnline(samplesSnapshot, RACE_AGENT, {
        // Init from PRIOR (not previous params) so drift from lap N
        // doesn't poison lap N+1's starting point. Each refit explores
        // from the same anchor; only genuinely informative gradients
        // move it.
        init: learner.priorParams,
        prior: learner.priorParams,
        maxIter: 200,
      });
      learner.params = fit.params;
      learner.refitCount++;
      learner.lastFitMs = performance.now() - t0;
      learner.lastMeanError = fit.meanPosError;
      // Adoption vs rollback: race with whichever lib we trust more.
      if (wasMuchWorse) {
        // Recent lap was significantly slower than best → revert.
        car.lib = buildLearnedRaceLibrary(learner.bestParams);
        learner.libParams = learner.bestParams;
        learner.rollbackActive = true;
        learner.rollbackCount++;
      } else {
        // First lap (no best yet) or within hysteresis of best → try
        // the latest refit. If it underperforms, the next lap's check
        // will roll back.
        car.lib = buildLearnedRaceLibrary(fit.params);
        learner.libParams = fit.params;
        learner.rollbackActive = false;
      }
      learner.refitting = false;
    }, REFIT_DEFER_MS);
  }

  // ---- Replan ticker ----
  const replanTimer = window.setInterval(() => {
    if (!running) return;
    const now = performance.now();
    replan(kinematic, now);
    replan(learned, now);
  }, REPLAN_INTERVAL_MS);

  // ---- Animation loop ----
  let stopped = false;
  let lastWall = performance.now();
  let lastLearnerEmit = 0;
  function tick() {
    if (stopped) return;
    requestAnimationFrame(tick);
    const now = performance.now();
    const dt = Math.min(0.05, (now - lastWall) / 1000);
    lastWall = now;
    if (running) {
      stepCar(kinematic, now, dt);
      stepCar(learned, now, dt);
      // Sync release: when BOTH cars have just finished the same lap,
      // release both holds simultaneously so the next lap starts head-to-
      // head. Until then, the leader holds at the start line.
      if (kinematic.holdingForSync && learned.holdingForSync) {
        kinematic.holdingForSync = false;
        learned.holdingForSync = false;
        // Force an immediate replan so neither car coasts on its stale
        // pre-hold plan.
        kinematic.ai.plan = null;
        learned.ai.plan = null;
        replan(kinematic, now);
        replan(learned, now);
      }
      cb.onMetrics(kinematic.metrics, learned.metrics);
      // Emit the comparison snapshot in BOTH modes (legacy + v2). The lap-
      // times / sector-deltas are universal; only the per-coef refit fields
      // are legacy-only and get safe defaults when v2 is driving.
      if (now - lastLearnerEmit > 250) {
        lastLearnerEmit = now;
        const lz = learned.learner;
        cb.onLearner({
          params: lz?.params ?? DEFAULT_LEARNED_PARAMS,
          priorParams: lz?.priorParams ?? DEFAULT_LEARNED_PARAMS,
          bestParams: lz?.bestParams ?? DEFAULT_LEARNED_PARAMS,
          bestLapTime: lz?.bestLapTime ?? Number.NaN,
          bestLapNumber: lz?.bestLapNumber ?? 0,
          rollbackActive: lz?.rollbackActive ?? false,
          rollbackCount: lz?.rollbackCount ?? 0,
          refitCount: lz?.refitCount ?? 0,
          sampleCount: lz?.samples.length ?? 0,
          lastFitMs: lz?.lastFitMs ?? 0,
          lastMeanError: lz?.lastMeanError ?? 0,
          kinematicLapTimes: kinematic.lapTimes.slice(),
          learnedLapTimes: learned.lapTimes.slice(),
          kinematicSectors: kinematic.sectorTimes.map((s) => s.slice()),
          learnedSectors: learned.sectorTimes.map((s) => s.slice()),
          sectorsPerLap: course.waypoints.length,
          kinematicHolding: kinematic.holdingForSync,
          learnedHolding: learned.holdingForSync,
        });
      }
    }
    // Update chase cameras (smooth follow). Reads chassis pose directly from
    // Rapier — same DynamicRayCastVehicleController used in /carchase, just
    // viewed through a chase-cam per side.
    updateChaseCamera(camK, kinematic.car.readState(now));
    updateChaseCamera(camL, learned.car.readState(now));
    // Render split viewport.
    const w = mount.clientWidth;
    const h = mount.clientHeight;
    renderer.setScissor(0, 0, w / 2, h);
    renderer.setViewport(0, 0, w / 2, h);
    renderer.render(kinematic.scene, camK);
    renderer.setScissor(w / 2, 0, w / 2, h);
    renderer.setViewport(w / 2, 0, w / 2, h);
    renderer.render(learned.scene, camL);
  }
  tick();

  // ---- Resize ----
  const onResize = () => {
    const w = mount.clientWidth;
    const h = mount.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h);
    const aspect = (w / 2) / h;
    for (const cam of [camK, camL]) {
      cam.aspect = aspect;
      cam.updateProjectionMatrix();
    }
  };
  window.addEventListener('resize', onResize);

  return {
    cleanup() {
      stopped = true;
      window.clearInterval(replanTimer);
      window.removeEventListener('resize', onResize);
      kinematic.car.dispose();
      kinematic.world.free();
      learned.car.dispose();
      learned.world.free();
      renderer.dispose();
      if (renderer.domElement.parentNode === mount) {
        mount.removeChild(renderer.domElement);
      }
    },
    start() {
      resetCar(kinematic);
      resetCar(learned);
      // Reset learner state on a new race: clear samples but KEEP the most
      // recently fitted params so re-races can carry over what was learned.
      if (learned.learner) {
        learned.learner.samples = [];
      }
      running = true;
      raceStartWall = performance.now();
      kinematic.ai.planStartWall = raceStartWall;
      learned.ai.planStartWall = raceStartWall;
      replan(kinematic, raceStartWall);
      replan(learned, raceStartWall);
    },
    stop() {
      running = false;
    },
    reset() {
      running = false;
      resetCar(kinematic);
      resetCar(learned);
      // Full reset wipes the learner so the demo starts honestly identical.
      if (learned.learner) {
        learned.learner.samples = [];
        learned.learner.refitCount = 0;
        learned.learner.rollbackCount = 0;
        learned.learner.params = initialLearnerParams;
        learned.learner.priorParams = initialLearnerParams;
        learned.learner.libParams = initialLearnerParams;
        learned.learner.bestParams = initialLearnerParams;
        learned.learner.bestLapTime = Number.NaN;
        learned.learner.bestLapNumber = 0;
        learned.learner.rollbackActive = false;
        learned.learner.lastFitMs = 0;
        learned.learner.lastMeanError = 0;
        learned.lib = initialLearnedLib;
      }
    },
  };
}

// ---------------------------------------------------------------------------
// Helpers.

function paramsEqual(a: LearnedVehicleParams, b: LearnedVehicleParams): boolean {
  const EPS = 1e-9;
  return (
    Math.abs(a.maxAccel - b.maxAccel) < EPS &&
    Math.abs(a.maxDecel - b.maxDecel) < EPS &&
    Math.abs(a.accelTau - b.accelTau) < EPS &&
    Math.abs(a.understeerGain - b.understeerGain) < EPS &&
    Math.abs(a.lateralDrag - b.lateralDrag) < EPS
  );
}

function trimPlan(plan: VehicleState[], elapsed: number): VehicleState[] {
  let i = 0;
  while (i < plan.length - 1 && plan[i + 1]!.t <= elapsed) i++;
  return plan.slice(i);
}

/** Smoothly move a perspective camera into a chase position behind a
 *  VehicleState. The chassis y is unknown to the planner (Y is derived in
 *  kinocat); for the cam we place it ~10m above so the wheels and the
 *  upcoming course are both visible. */
function updateChaseCamera(cam: THREE.PerspectiveCamera, s: VehicleState): void {
  const c = Math.cos(s.heading);
  const sn = Math.sin(s.heading);
  // 14m behind + 7m above the chassis, looking ~6m ahead.
  const target = new THREE.Vector3(s.x - 14 * c, 7, s.z - 14 * sn);
  cam.position.lerp(target, 0.12);
  cam.lookAt(s.x + 6 * c, 1.2, s.z + 6 * sn);
}

function planAtTime(plan: VehicleState[], t: number): VehicleState | null {
  if (plan.length === 0) return null;
  if (t <= plan[0]!.t) return plan[0]!;
  for (let i = 1; i < plan.length; i++) {
    if (plan[i]!.t >= t) {
      const a = plan[i - 1]!;
      const b = plan[i]!;
      const span = b.t - a.t;
      const u = span > 1e-9 ? (t - a.t) / span : 0;
      return {
        x: a.x + (b.x - a.x) * u,
        z: a.z + (b.z - a.z) * u,
        heading: a.heading,
        speed: a.speed,
        t,
      };
    }
  }
  return plan[plan.length - 1]!;
}

// ---------------------------------------------------------------------------
// UI bits.

function TopBar({
  phase,
  learnProgress,
  winner,
  params,
  error,
  v2Active,
  onLearn,
  onStart,
  onStop,
  onReset,
  onClearCache,
}: {
  phase: Phase;
  learnProgress: {
    done: number;
    total: number;
    curvature?: number;
    targetSpeed?: number;
    startSpeed?: number;
  };
  winner: 'kinematic' | 'learned' | 'tie' | null;
  params: LearnedVehicleParams | null;
  error: string | null;
  /** True when the learned car is driving with the offline-trained v2
   *  library — online refitting is disabled in that mode, so the status
   *  text changes accordingly. */
  v2Active: boolean;
  onLearn: () => void;
  onStart: () => void;
  onStop: () => void;
  onReset: () => void;
  onClearCache: () => void;
}) {
  const subtitle = v2Active
    ? 'kinematic (control) vs offline-trained v2 model · online refit disabled while v2 is active'
    : 'kinematic (control) vs online-learning · both start with the same library, learned car refits 5 coefficients from race data each lap';
  const racingStatus = v2Active
    ? 'racing… (learned car using v2 library — offline-trained, no online refit)'
    : 'racing… (the learned car refits every lap)';
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        padding: '10px 14px',
        borderBottom: '1px solid #1f2735',
        background: '#0d1119',
      }}
    >
      <div style={{ color: '#7fd6ff', fontWeight: 700 }}>race the primitives</div>
      <div style={{ opacity: 0.65 }}>{subtitle}</div>
      <div style={{ flex: 1 }} />
      {phase === 'loading' && <Status>loading…</Status>}
      {phase === 'learning' && (
        <Status>
          pre-training… collecting trials {learnProgress.done}/{learnProgress.total || '?'}
        </Status>
      )}
      {phase === 'ready' && params && (
        <>
          <Status>ready{v2Active ? ' · v2 active' : ''}</Status>
          <Btn onClick={onStart}>start race</Btn>
          <Btn onClick={onLearn} secondary>pre-train</Btn>
          <Btn onClick={onReset} secondary>reset</Btn>
          <Btn onClick={onClearCache} secondary>clear cache</Btn>
        </>
      )}
      {phase === 'racing' && (
        <>
          <Status>{racingStatus}</Status>
          <Btn onClick={onStop}>stop</Btn>
        </>
      )}
      {phase === 'finished' && (
        <>
          <Status>stopped{v2Active ? ' · v2 active' : ''}</Status>
          <Btn onClick={onStart}>race again</Btn>
          <Btn onClick={onReset} secondary>reset</Btn>
        </>
      )}
      {error && <Status warning>err: {error}</Status>}
    </div>
  );
}

function MetricsOverlay({
  metrics,
  winner,
  holding,
  lapTimes,
  rollbackActive,
  bestLapNumber,
}: {
  metrics: { kinematic: RaceMetrics; learned: RaceMetrics };
  winner: 'kinematic' | 'learned' | 'tie' | null;
  holding: { kinematic: boolean; learned: boolean };
  lapTimes: { kinematic: number[]; learned: number[] };
  rollbackActive: boolean;
  bestLapNumber: number;
}) {
  return (
    <>
      <SideMetrics
        side="left"
        title="KINEMATIC (control)"
        color="#ff8aa0"
        m={metrics.kinematic}
        highlight={winner === 'kinematic'}
        holding={holding.kinematic}
        recentLaps={lapTimes.kinematic}
        rollbackBadge={null}
      />
      <SideMetrics
        side="right"
        title="LEARNED (online)"
        color="#55dcff"
        m={metrics.learned}
        highlight={winner === 'learned'}
        holding={holding.learned}
        recentLaps={lapTimes.learned}
        rollbackBadge={rollbackActive ? `USING BEST (lap ${bestLapNumber})` : null}
      />
    </>
  );
}

function SideMetrics({
  side,
  title,
  color,
  m,
  highlight,
  holding,
  recentLaps,
  rollbackBadge,
}: {
  side: 'left' | 'right';
  title: string;
  color: string;
  m: RaceMetrics;
  highlight: boolean;
  holding: boolean;
  recentLaps: number[];
  rollbackBadge: string | null;
}) {
  // Stability over the last 5 laps — low std-dev means the car has
  // settled into a consistent racing line, high means it's still
  // oscillating between fits or fighting the course.
  const last5 = recentLaps.slice(-5);
  const mean5 =
    last5.length > 0 ? last5.reduce((a, b) => a + b, 0) / last5.length : Number.NaN;
  const std5 =
    last5.length > 1
      ? Math.sqrt(
          last5.reduce((a, b) => a + (b - mean5) * (b - mean5), 0) / last5.length,
        )
      : Number.NaN;
  return (
    <div
      style={{
        position: 'absolute',
        top: 12,
        [side]: 12,
        background: 'rgba(13, 17, 25, 0.85)',
        border: `1px solid ${holding ? '#ffd070' : highlight ? color : '#1f2735'}`,
        borderRadius: 8,
        padding: '10px 14px',
        minWidth: 240,
        boxShadow: highlight ? `0 0 24px ${color}66` : 'none',
        color: '#cdd3de',
      }}
    >
      <div style={{ color, fontWeight: 700, marginBottom: 6 }}>
        {title}
        {holding && (
          <span
            style={{
              marginLeft: 8,
              color: '#ffd070',
              fontWeight: 700,
              fontSize: 10,
              padding: '2px 6px',
              border: '1px solid #ffd070',
              borderRadius: 4,
            }}
          >
            WAITING…
          </span>
        )}
        {rollbackBadge && (
          <span
            style={{
              marginLeft: 8,
              color: '#a6e9ff',
              fontWeight: 700,
              fontSize: 10,
              padding: '2px 6px',
              border: '1px solid #55dcff',
              borderRadius: 4,
            }}
          >
            {rollbackBadge}
          </span>
        )}
      </div>
      <KV k="time" v={`${m.raceTime.toFixed(2)} s`} />
      <KV k="laps" v={`${m.laps}`} />
      <KV k="waypoints" v={`${m.waypointsCleared}`} />
      <KV
        k="best lap"
        v={Number.isFinite(m.bestLapTime) ? `${m.bestLapTime.toFixed(2)} s` : '—'}
      />
      <KV
        k="last lap"
        v={Number.isFinite(m.lastLapTime) ? `${m.lastLapTime.toFixed(2)} s` : '—'}
      />
      <KV
        k="mean (last 5)"
        v={Number.isFinite(mean5) ? `${mean5.toFixed(2)} s` : '—'}
      />
      <KV
        k="std-dev (last 5)"
        v={Number.isFinite(std5) ? `${std5.toFixed(3)} s` : '—'}
      />
      <KV k="0.55s pred err (rms)" v={`${m.trackingErrorRms.toFixed(2)} m`} />
      <KV k="peak speed" v={`${m.peakSpeed.toFixed(1)} m/s`} />
      <div style={{ marginTop: 6, paddingTop: 6, borderTop: '1px solid #1f2735', opacity: 0.85 }}>
        <div style={{ fontSize: 10, opacity: 0.7, marginBottom: 2 }}>LIVE CONTROLS</div>
        <KV
          k="steer"
          v={`${m.liveControls.steer >= 0 ? '+' : ''}${m.liveControls.steer.toFixed(3)} rad`}
        />
        <KV k="throttle" v={`${(m.liveControls.throttle * 100).toFixed(0)}%`} />
        <KV k="brake" v={`${(m.liveControls.brake * 100).toFixed(0)}%`} />
        <KV k="target spd" v={`${m.liveControls.targetSpeed.toFixed(1)} m/s`} />
      </div>
    </div>
  );
}

function LearnerPanel({ snap, v2Active, v2Meta }: {
  snap: LearnerSnapshot;
  v2Active: boolean;
  v2Meta?: PersistedV2Model['meta'] | null;
}) {
  return (
    <div
      style={{
        position: 'absolute',
        bottom: 12,
        left: 12,
        right: 12,
        background: 'rgba(13, 17, 25, 0.88)',
        border: '1px solid #1f2735',
        borderRadius: 8,
        padding: '10px 14px',
        color: '#cdd3de',
        font: '11px ui-monospace, monospace',
        display: 'flex',
        flexDirection: 'column',
        gap: 10,
      }}
    >
      <LapTimeChart
        kinematic={snap.kinematicLapTimes}
        learned={snap.learnedLapTimes}
      />
      <SectorDeltaStrip
        kinematicSectors={snap.kinematicSectors}
        learnedSectors={snap.learnedSectors}
        sectorsPerLap={snap.sectorsPerLap}
      />
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1.4fr 1fr 1fr 1fr',
          gap: 16,
          borderTop: '1px solid #1f2735',
          paddingTop: 10,
        }}
      >
        {v2Active ? (
          <div>
            <div style={{ color: '#55dcff', fontWeight: 700, marginBottom: 6 }}>
              V2 LEARNED MODEL · offline-trained
              <span style={{ opacity: 0.65 }}>
                {' '}· online refit disabled
              </span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              {v2Meta && (
                <>
                  <KV k="trials used" v={`${v2Meta.trialsUsed}`} />
                  <KV k="open-loop RMS @ 1s" v={`${v2Meta.openLoopRmsAt1s.toFixed(3)} m`} />
                  {v2Meta.legacyRmsAt1s !== undefined && (
                    <KV
                      k="vs legacy 5-param"
                      v={`${v2Meta.legacyRmsAt1s.toFixed(3)} m (${((1 - v2Meta.openLoopRmsAt1s / v2Meta.legacyRmsAt1s) * 100).toFixed(1)}% better)`}
                    />
                  )}
                  {v2Meta.kinematicRmsAt1s !== undefined && (
                    <KV
                      k="vs kinematic"
                      v={`${v2Meta.kinematicRmsAt1s.toFixed(3)} m`}
                    />
                  )}
                  <KV k="trained" v={new Date(v2Meta.createdAt).toLocaleString()} />
                </>
              )}
              {!v2Meta && (
                <span style={{ opacity: 0.65 }}>v2 model active (no meta available)</span>
              )}
            </div>
          </div>
        ) : (
          <div>
            <div style={{ color: '#55dcff', fontWeight: 700, marginBottom: 6 }}>
              ONLINE LEARNER · {snap.refitCount} refit{snap.refitCount === 1 ? '' : 's'}
              {snap.rollbackCount > 0 && (
                <span style={{ color: '#ffd070' }}>
                  {' '}· {snap.rollbackCount} rollback{snap.rollbackCount === 1 ? '' : 's'}
                </span>
              )}
              {' '}· {snap.sampleCount} samples
              {snap.lastFitMs > 0 && (
                <span style={{ opacity: 0.65 }}>
                  {' '}· last fit {snap.lastFitMs.toFixed(0)}ms (mean err {snap.lastMeanError.toFixed(3)}m)
                </span>
              )}
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'auto auto auto auto', gap: '2px 12px' }}>
              <span style={{ opacity: 0.55 }}>coef</span>
              <span style={{ opacity: 0.55, textAlign: 'right' }}>live</span>
              <span style={{ opacity: 0.55, textAlign: 'right' }}>
                best{Number.isFinite(snap.bestLapTime) ? ` (l${snap.bestLapNumber})` : ''}
              </span>
              <span style={{ opacity: 0.55, textAlign: 'right' }}>prior</span>
              <ParamRow k="maxAccel" v={snap.params.maxAccel} best={snap.bestParams.maxAccel} prior={snap.priorParams.maxAccel} unit="m/s²" />
              <ParamRow k="maxDecel" v={snap.params.maxDecel} best={snap.bestParams.maxDecel} prior={snap.priorParams.maxDecel} unit="m/s²" />
              <ParamRow k="accelTau" v={snap.params.accelTau} best={snap.bestParams.accelTau} prior={snap.priorParams.accelTau} unit="s" digits={3} />
              <ParamRow k="understeerGain" v={snap.params.understeerGain} best={snap.bestParams.understeerGain} prior={snap.priorParams.understeerGain} exp />
              <ParamRow k="lateralDrag" v={snap.params.lateralDrag} best={snap.bestParams.lateralDrag} prior={snap.priorParams.lateralDrag} exp />
            </div>
          </div>
        )}
        <LapTimeList
          title="kinematic laps"
          color="#ff8aa0"
          laps={snap.kinematicLapTimes}
        />
        <LapTimeList
          title={v2Active ? 'learned (v2) laps' : 'learned laps'}
          color="#55dcff"
          laps={snap.learnedLapTimes}
        />
        <LapDeltaSpark
          kinematic={snap.kinematicLapTimes}
          learned={snap.learnedLapTimes}
        />
      </div>
    </div>
  );
}

function ParamRow({
  k,
  v,
  best,
  prior,
  unit,
  digits,
  exp,
}: {
  k: string;
  v: number;
  best: number;
  prior: number;
  unit?: string;
  digits?: number;
  exp?: boolean;
}) {
  const fmt = (x: number) =>
    exp ? x.toExponential(2) : x.toFixed(digits ?? 2);
  return (
    <>
      <span style={{ opacity: 0.7 }}>{k}</span>
      <span style={{ color: '#cdeaff', textAlign: 'right' }}>
        {fmt(v)}
        {unit ? ` ${unit}` : ''}
      </span>
      <span style={{ color: '#a6e9ff', textAlign: 'right' }}>{fmt(best)}</span>
      <span style={{ opacity: 0.45, textAlign: 'right' }}>{fmt(prior)}</span>
    </>
  );
}

/** Per-gate sector-delta bars for the most recently completed lap.
 *  Each bar = (kinematic_sector_time − learned_sector_time) for that
 *  gate-to-gate segment. Cyan bar above the line = learned was faster
 *  on that segment; pink bar below = kinematic was faster. Reveals
 *  WHERE on the course each library wins (e.g., "learned wins the
 *  tight slalom but loses the long straight"). */
function SectorDeltaStrip({
  kinematicSectors,
  learnedSectors,
  sectorsPerLap,
}: {
  kinematicSectors: number[][];
  learnedSectors: number[][];
  sectorsPerLap: number;
}) {
  const lapsCompared = Math.min(kinematicSectors.length, learnedSectors.length);
  if (lapsCompared === 0) {
    return (
      <div style={{ opacity: 0.5, padding: '4px 0' }}>
        sector deltas appear after both cars complete a lap…
      </div>
    );
  }
  // sector i = time from previous gate to gate i (gate 0 from lap start).
  function sectorDeltas(cum: number[]): number[] {
    const out: number[] = [];
    for (let i = 0; i < cum.length; i++) {
      out.push(i === 0 ? cum[0]! : cum[i]! - cum[i - 1]!);
    }
    return out;
  }
  // Per-sector best across ALL completed laps (per car). Surfaced as
  // a small ghost outline next to the current bar so the user can see
  // whether THIS lap matched or beat the historical best for each gate
  // — a "running personal best" view, independent of the
  // current-lap vs current-lap delta.
  function perSectorBests(allLaps: number[][]): number[] {
    if (allLaps.length === 0) return [];
    const m = allLaps[0]!.length;
    const bests = new Array(m).fill(Number.POSITIVE_INFINITY) as number[];
    for (const lap of allLaps) {
      const deltas = sectorDeltas(lap);
      for (let i = 0; i < Math.min(m, deltas.length); i++) {
        if (deltas[i]! < bests[i]!) bests[i] = deltas[i]!;
      }
    }
    return bests;
  }
  const kLap = kinematicSectors[lapsCompared - 1]!;
  const lLap = learnedSectors[lapsCompared - 1]!;
  const kSec = sectorDeltas(kLap);
  const lSec = sectorDeltas(lLap);
  const kBest = perSectorBests(kinematicSectors);
  const lBest = perSectorBests(learnedSectors);
  const n = Math.min(kSec.length, lSec.length, sectorsPerLap);
  const deltas: number[] = [];
  const bestDeltas: number[] = [];
  for (let i = 0; i < n; i++) {
    deltas.push(kSec[i]! - lSec[i]!); // positive = learned faster THIS lap
    bestDeltas.push(kBest[i]! - lBest[i]!); // best-vs-best (the "ceiling")
  }
  const maxAbs = Math.max(
    0.1,
    ...deltas.map(Math.abs),
    ...bestDeltas.map(Math.abs),
  );
  const W = 600;
  const H = 78;
  const padL = 38;
  const padR = 8;
  const padT = 6;
  const padB = 16;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const midY = padT + plotH / 2;
  const barW = (plotW / n) * 0.78;
  const stride = plotW / n;
  return (
    <div>
      <div style={{ color: '#7fd6ff', fontWeight: 700, marginBottom: 4 }}>
        sector deltas · lap {lapsCompared} (filled) vs best-vs-best (outline) · cyan = learned faster · pink = kinematic faster
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        style={{ width: '100%', height: H, display: 'block' }}
      >
        <text x={4} y={midY + 4} fill="#6b7280" fontSize={10}>{`±${maxAbs.toFixed(2)}s`}</text>
        <line x1={padL} y1={midY} x2={W - padR} y2={midY} stroke="#1f2735" strokeWidth={1} />
        {deltas.map((d, i) => {
          const h = (Math.abs(d) / maxAbs) * (plotH / 2);
          const x = padL + i * stride + (stride - barW) / 2;
          const color = d > 0 ? '#55dcff' : '#ff8aa0';
          const y = d > 0 ? midY - h : midY;
          // Best-vs-best ghost outline at the same x position.
          const bd = bestDeltas[i] ?? 0;
          const bh = (Math.abs(bd) / maxAbs) * (plotH / 2);
          const bColor = bd > 0 ? '#55dcff' : '#ff8aa0';
          const by = bd > 0 ? midY - bh : midY;
          return (
            <g key={i}>
              {/* current-lap filled bar */}
              <rect x={x} y={y} width={barW} height={h} fill={color} opacity={0.85} />
              {/* best-vs-best outline (no fill) — the "ceiling" achievable so far */}
              <rect
                x={x - 1}
                y={by}
                width={barW + 2}
                height={bh}
                fill="none"
                stroke={bColor}
                strokeWidth={1}
                strokeDasharray="2 2"
                opacity={0.7}
              />
              <text
                x={x + barW / 2}
                y={H - 4}
                fill="#6b7280"
                fontSize={9}
                textAnchor="middle"
              >
                s{i + 1}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

/** Inline SVG line chart of lap times for both cars over the last 24 laps.
 *  Lower is better; the gap between the two lines IS the learning gap. */
function LapTimeChart({
  kinematic,
  learned,
}: {
  kinematic: number[];
  learned: number[];
}) {
  const WINDOW = 24;
  const kSeries = kinematic.slice(-WINDOW);
  const lSeries = learned.slice(-WINDOW);
  const n = Math.max(kSeries.length, lSeries.length);
  if (n === 0) {
    return (
      <div style={{ height: 90, opacity: 0.5, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        chart appears after the first lap completes…
      </div>
    );
  }
  const all = [...kSeries, ...lSeries];
  const minT = Math.min(...all);
  const maxT = Math.max(...all);
  const pad = Math.max(0.05 * (maxT - minT || 1), 0.5);
  const y0 = minT - pad;
  const y1 = maxT + pad;
  const W = 600;
  const H = 110;
  const padL = 38;
  const padR = 8;
  const padT = 8;
  const padB = 18;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const xAt = (i: number) => padL + (n <= 1 ? plotW / 2 : (i / (n - 1)) * plotW);
  const yAt = (t: number) => padT + plotH * (1 - (t - y0) / (y1 - y0));
  function polyline(series: number[]): string {
    return series.map((t, i) => `${i === 0 ? 'M' : 'L'}${xAt(i).toFixed(1)},${yAt(t).toFixed(1)}`).join(' ');
  }
  return (
    <div>
      <div style={{ color: '#7fd6ff', fontWeight: 700, marginBottom: 4 }}>
        lap-time progression · lower is better · {n} lap{n === 1 ? '' : 's'} shown
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        style={{ width: '100%', height: 110, display: 'block' }}
      >
        {/* y-axis labels */}
        <text x={4} y={padT + 8} fill="#6b7280" fontSize={10}>{y1.toFixed(1)}s</text>
        <text x={4} y={H - padB + 2} fill="#6b7280" fontSize={10}>{y0.toFixed(1)}s</text>
        {/* gridlines */}
        <line x1={padL} y1={padT} x2={W - padR} y2={padT} stroke="#1f2735" strokeWidth={1} />
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="#1f2735" strokeWidth={1} />
        <line x1={padL} y1={(padT + H - padB) / 2} x2={W - padR} y2={(padT + H - padB) / 2} stroke="#1f2735" strokeDasharray="2 4" strokeWidth={1} />
        {/* lines + points */}
        <path d={polyline(kSeries)} fill="none" stroke="#ff8aa0" strokeWidth={2} />
        {kSeries.map((t, i) => (
          <circle key={`k${i}`} cx={xAt(i)} cy={yAt(t)} r={2.5} fill="#ff8aa0" />
        ))}
        <path d={polyline(lSeries)} fill="none" stroke="#55dcff" strokeWidth={2} />
        {lSeries.map((t, i) => (
          <circle key={`l${i}`} cx={xAt(i)} cy={yAt(t)} r={2.5} fill="#55dcff" />
        ))}
        {/* x-axis labels */}
        <text x={padL} y={H - 4} fill="#6b7280" fontSize={10}>
          lap {Math.max(1, (kinematic.length || learned.length) - n + 1)}
        </text>
        <text x={W - padR - 28} y={H - 4} fill="#6b7280" fontSize={10}>
          lap {kinematic.length || learned.length}
        </text>
      </svg>
    </div>
  );
}

function LapTimeList({
  title,
  color,
  laps,
}: {
  title: string;
  color: string;
  laps: number[];
}) {
  const shown = laps.slice(-8);
  const best = laps.length ? Math.min(...laps) : Number.POSITIVE_INFINITY;
  return (
    <div>
      <div style={{ color, fontWeight: 700, marginBottom: 6 }}>{title}</div>
      {shown.length === 0 && <div style={{ opacity: 0.5 }}>—</div>}
      {shown.map((t, i) => {
        const lapIdx = laps.length - shown.length + i + 1;
        const isBest = t === best;
        return (
          <div
            key={lapIdx}
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              color: isBest ? color : '#cdd3de',
              fontWeight: isBest ? 700 : 400,
            }}
          >
            <span style={{ opacity: 0.7 }}>lap {lapIdx}</span>
            <span>{t.toFixed(2)} s</span>
          </div>
        );
      })}
    </div>
  );
}

function LapDeltaSpark({
  kinematic,
  learned,
}: {
  kinematic: number[];
  learned: number[];
}) {
  const n = Math.min(kinematic.length, learned.length);
  const deltas = [];
  for (let i = 0; i < n; i++) deltas.push(kinematic[i]! - learned[i]!); // positive = learned faster
  return (
    <div>
      <div style={{ color: '#7fd6ff', fontWeight: 700, marginBottom: 6 }}>
        Δ per lap (kinematic − learned)
      </div>
      {deltas.length === 0 && <div style={{ opacity: 0.5 }}>—</div>}
      {deltas.slice(-8).map((d, i) => {
        const lapIdx = deltas.length - Math.min(deltas.length, 8) + i + 1;
        const color = d > 0 ? '#55dcff' : d < 0 ? '#ff8aa0' : '#cdd3de';
        return (
          <div
            key={lapIdx}
            style={{ display: 'flex', justifyContent: 'space-between', color }}
          >
            <span style={{ opacity: 0.7 }}>lap {lapIdx}</span>
            <span>{d > 0 ? '+' : ''}{d.toFixed(2)} s</span>
          </div>
        );
      })}
    </div>
  );
}

function KV({ k, v }: { k: string; v: string }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
      <span style={{ opacity: 0.7 }}>{k}</span>
      <span style={{ color: '#cdeaff' }}>{v}</span>
    </div>
  );
}

function Status({
  children,
  warning,
}: {
  children: React.ReactNode;
  warning?: boolean;
}) {
  return (
    <div
      style={{
        padding: '4px 10px',
        border: `1px solid ${warning ? '#ff8aa0' : '#1f2735'}`,
        borderRadius: 6,
        color: warning ? '#ff8aa0' : '#cdeaff',
        background: warning ? 'rgba(255, 138, 160, 0.10)' : 'rgba(127, 214, 255, 0.10)',
      }}
    >
      {children}
    </div>
  );
}

function Btn({
  children,
  onClick,
  secondary,
}: {
  children: React.ReactNode;
  onClick: () => void;
  secondary?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        font: '11px ui-monospace, monospace',
        padding: '6px 12px',
        borderRadius: 6,
        border: '1px solid #2a3340',
        background: secondary ? 'rgba(20, 26, 38, 0.85)' : 'rgba(127, 214, 255, 0.18)',
        color: secondary ? '#8c95a4' : '#cdeaff',
        cursor: 'pointer',
        letterSpacing: 0.3,
      }}
    >
      {children}
    </button>
  );
}
