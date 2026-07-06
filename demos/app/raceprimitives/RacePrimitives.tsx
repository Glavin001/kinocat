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
import type { LearnedVehicleParams, CarKinematicState } from 'kinocat/agent';
import {
  DEFAULT_LEARNED_PARAMS,
  DEFAULT_LEARNABLE_CONFIG,
  KINEMATIC_NATIVE_PARAMS,
  learnedForwardSimV2,
  parametricForwardV2,
} from 'kinocat/agent';
import { InMemoryNavWorld } from 'kinocat/environment';
import { MotionPrimitiveLibrary } from 'kinocat/primitives';
import { purePursuit } from 'kinocat/execute';
import {
  createRaycastVehicle,
  createGroundCollider,
  ensureRapier,
  stepRaycastVehicle,
  type CarHandle,
} from 'kinocat/adapters/rapier';
import {
  trimPlan,
  wheeledFromNormalized,
  type CarForceTuning,
} from 'kinocat/vehicle/car';
import {
  createCarMeshHelper,
  syncCarMesh,
  createGroundPlaneHelper,
  createBuildingHelper,
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
  RACE_PLANNER_GATE_RADIUS,
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
import { EvalHUD } from '../components/EvalHUD';
import { createEvalProbe, type EvalSnapshot } from '../lib/eval-probe';
import { limitsFromAgent } from 'kinocat/eval';
import { useIsMobile } from '../lib/use-is-mobile';
import {
  loadV2Model,
  loadV2ModelFromUrl,
  saveV2Model,
  clearV2Model,
  buildV2ModelDownloadUrl,
  type PersistedV2Model,
} from '../lib/v2-model-persistence';
import {
  buildDebugReport,
  copyToClipboard,
  downloadMarkdown,
} from '../lib/debug-report';
import type { LearnedVehicleModel } from 'kinocat/agent';

// SINGLE SOURCE OF TRUTH: all race-simulation tunables live in
// `../lib/race-scenario.ts`. Importing them here (rather than redeclaring
// inline) guarantees the React demo + the headless CLI cannot drift on
// physics dt, replan cadence, pure-pursuit gain, steer-angle formula,
// engine/brake forces, etc. If you need to change any value, change it
// in `race-scenario.ts` and BOTH consumers update automatically.
//
// Behavioral note: until the React tick loop also routes through
// `createRaceScenario` (planned follow-up), this file still has its own
// `replan` + `stepCar` implementations that read these constants. They
// mirror `race-scenario.ts` line-for-line so behavior is identical.
import {
  PHYSICS_DT,
  VEHICLE_SUBSTEPS,
  REPLAN_INTERVAL_MS,
  WHEEL_BASE,
  ENGINE_FORCE_N,
  BRAKE_FORCE_N,
  TRACKER_MAX_LATERAL_ACCEL as SCENARIO_TRACKER_MAX_LATERAL_ACCEL,
  PLAN_LOOKAHEAD_COUNT as SCENARIO_PLAN_LOOKAHEAD_COUNT,
  createRaceScenario,
  type RaceCarStatus,
} from '../lib/race-scenario';

const RACE_FORCE_TUNING: CarForceTuning = {
  engineForceN: ENGINE_FORCE_N,
  brakeForceN: BRAKE_FORCE_N,
};
const toWheeled = (cmd: { steer: number; throttle: number; brake: number }) =>
  wheeledFromNormalized(cmd, RACE_FORCE_TUNING);
// Re-export the scenario-shared constants under the names the rest of this
// file uses. Real tire grip on dry asphalt is ~9-10 m/s² (~1g); the value
// in race-scenario.ts (12) keeps the cars physically plausible.
const TRACKER_MAX_LATERAL_ACCEL = SCENARIO_TRACKER_MAX_LATERAL_ACCEL;
const PLAN_LOOKAHEAD_COUNT = SCENARIO_PLAN_LOOKAHEAD_COUNT;
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

type TrackerMode = 'pure-pursuit' | 'mpc';
type CourseVariant = 'open' | 'technical';
/** Which primitive library / rollout model the LEARNED car drives with. */
type LearnedModelChoice = 'v2' | 'kinematic';

/** Path-tracking executor this mount runs. The GUI (top-bar Race Setup) is
 *  the primary control; the `?tracker=mpc` query param only SEEDS the initial
 *  value so a run is shareable/deep-linkable. MPPI runs each car's own forward
 *  model in the loop — the fidelity-becomes-control-quality mode (roadmap
 *  WS-3). */
function trackerFromUrl(): TrackerMode {
  if (typeof window === 'undefined') return 'pure-pursuit';
  return new URLSearchParams(window.location.search).get('tracker') === 'mpc'
    ? 'mpc'
    : 'pure-pursuit';
}

/** Course variant seed (GUI-primary; `?course=technical` seeds it). */
function courseFromUrl(): CourseVariant {
  if (typeof window === 'undefined') return 'open';
  return new URLSearchParams(window.location.search).get('course') === 'technical'
    ? 'technical'
    : 'open';
}

/** Control-feedforward seed (GUI-primary; `?ff=1` seeds it ON). Under MPPI the
 *  tracker warm-starts its prior from the plan's own primitive controls
 *  (WS-1½) instead of re-deriving them from geometry — a no-op under
 *  pure-pursuit. */
function feedforwardFromUrl(): boolean {
  if (typeof window === 'undefined') return false;
  return new URLSearchParams(window.location.search).get('ff') === '1';
}

/** Reflect a Race Setup choice back into the URL (via replaceState, no
 *  navigation) so the current configuration stays shareable. The GUI state
 *  is the source of truth — this is a convenience mirror, not the config
 *  path. */
function syncSetupParam(key: string, value: string, isDefault: boolean): void {
  if (typeof window === 'undefined') return;
  const url = new URL(window.location.href);
  if (isDefault) url.searchParams.delete(key);
  else url.searchParams.set(key, value);
  window.history.replaceState(null, '', url.toString());
}

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
    plan: CarKinematicState[] | null;
    planStartWall: number;
    loopIndex: number;
    goal: CarKinematicState | null;
    /** Predicted end state of the plan's first primitive (t = 0.55s ahead)
     *  recorded at plan-install time. When wall-time reaches the predicted
     *  time, we compare predicted vs actual to get the per-primitive
     *  prediction error — the honest measure of dynamics-model accuracy
     *  (unlike the position-vs-elapsed-time metric which rewards
     *  confidence over correctness on long straights). */
    predictedEnd: { state: CarKinematicState; dueWall: number } | null;
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
  const isMobile = useIsMobile(820); // 820 leaves room for desktop side-by-side
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
  const [evalSnap, setEvalSnap] = useState<{
    kinematic: EvalSnapshot | null;
    learned: EvalSnapshot | null;
  }>({ kinematic: null, learned: null });
  const [winner, setWinner] = useState<'kinematic' | 'learned' | 'tie' | null>(null);
  // Model Lab drawer open/close (launched from the toolbar).
  const [modelLabOpen, setModelLabOpen] = useState(false);
  // Race Setup (GUI-primary). Executor + course seed from the URL on mount
  // (so `?tracker=mpc&course=technical` deep-links still work) but the GUI
  // selectors in the top bar are the source of truth thereafter; changing
  // one re-mounts the scene (same as the existing v2-library toggle). Read
  // the URL seeds in an effect, not at render, so SSR + hydration agree.
  const [trackerMode, setTrackerModeState] = useState<TrackerMode>('pure-pursuit');
  const [courseVariant, setCourseVariantState] = useState<CourseVariant>('open');
  const [feedforward, setFeedforwardState] = useState(false);
  useEffect(() => {
    setTrackerModeState(trackerFromUrl());
    setCourseVariantState(courseFromUrl());
    setFeedforwardState(feedforwardFromUrl());
  }, []);
  const setTrackerMode = (m: TrackerMode) => {
    setTrackerModeState(m);
    syncSetupParam('tracker', m, m === 'pure-pursuit');
  };
  const setCourseVariant = (c: CourseVariant) => {
    setCourseVariantState(c);
    syncSetupParam('course', c, c === 'open');
  };
  const setFeedforward = (on: boolean) => {
    setFeedforwardState(on);
    syncSetupParam('ff', on ? '1' : '0', !on);
  };
  // v2 model state (Phase-2 addition). When `useV2 && v2Model != null`, the
  // learned car's library is built from v2 instead of legacy.
  const [v2Model, setV2Model] = useState<LearnedVehicleModel | null>(null);
  const [v2Meta, setV2Meta] = useState<PersistedV2Model['meta'] | null>(null);
  const [useV2, setUseV2State] = useState(false);
  // True iff `/models/v2-default.json` is reachable (the preloaded
  // artifact `pnpm run train` writes). Drives the "Reset to default"
  // button in Model Lab.
  const [hasPreloadedDefault, setHasPreloadedDefault] = useState(false);
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
    // Toggle preference: '1' = explicitly on, '0' = explicitly off,
    // null = no explicit choice. When the user has never chosen, v2
    // defaults ON as soon as a trained model is available — otherwise a
    // fresh visitor watches kinematic vs the legacy 5-param model and
    // the trained v2 library (the page's whole point) sits unused.
    let toggle: string | null = null;
    if (typeof window !== 'undefined') {
      try {
        toggle = window.localStorage.getItem('kinocat:v2-toggle:v1');
      } catch { /* ignore */ }
    }
    if (toggle === '1') setUseV2State(true);
    // Try the localStorage cache first — if the user trained or imported
    // a model in a previous session it stays sticky.
    const cached = loadV2Model();
    let cancelled = false;
    if (cached) {
      setV2Model(cached.model);
      setV2Meta(cached.meta);
      if (toggle === null) setUseV2State(true);
    }
    // Always probe the preloaded artifact in the background — both so
    // we can light up the "Reset to default" button, and so a fresh
    // visitor (no cache) gets the CLI-trained model without waiting on
    // an inline retrain. Preloaded never overrides a cached model
    // (otherwise the user's training would be silently discarded on
    // reload).
    void loadV2ModelFromUrl().then((res) => {
      if (cancelled) return;
      if (!res) return;
      setHasPreloadedDefault(true);
      if (!cached) {
        setV2Model(res.model);
        setV2Meta(res.meta);
        if (toggle === null) setUseV2State(true);
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);

  // Reset action wired into the Model Lab "Reset to default" button.
  // Discards any cached model + reloads the preloaded artifact.
  async function resetToPreloadedDefault(): Promise<void> {
    const res = await loadV2ModelFromUrl();
    if (!res) return;
    clearV2Model();
    setV2Model(res.model);
    setV2Meta(res.meta);
  }

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
        const learnedForwardModel = v2Active ? learnedForwardSimV2(v2Model!) : undefined;
        const setup = await setupScene(mount, params, {
          onMetrics: (km, lm) => setMetrics({ kinematic: km, learned: lm }),
          onLearner: (snap) => setLearner(snap),
          onEval: (k, l) => setEvalSnap({ kinematic: k, learned: l }),
          onFinish: (w) => {
            setWinner(w);
            setPhase('finished');
          },
        }, { learnedLibraryOverride, learnedForwardModel, tracker: trackerMode, courseVariant, feedforward });
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
    // Re-mount when params, v2 toggle, v2 model identity, or a Race Setup
    // selector (tracker / course / feedforward) change — each rebuilds the scenario.
  }, [params, useV2, v2Model, trackerMode, courseVariant, feedforward, phase === 'learning' ? 'pending' : 'mounted']); // eslint-disable-line react-hooks/exhaustive-deps

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

  // ms timestamp when the "Export debug" button last fired — used to
  // show a brief "copied + downloaded" confirmation toast.
  const [debugExportedAt, setDebugExportedAt] = useState(0);

  async function onExportDebug() {
    // Build the kinematic + v2 libraries on demand to capture their
    // current shape (controls + per-bucket durations) in the report. We
    // could re-use the in-scene libraries but rebuilding from the same
    // sources guarantees the report matches what /primitive-explorer
    // would show right now.
    const kinematicLibrary = (await import('../lib/race-primitives-scenarios'))
      .buildKinematicLibrary();
    const learnedLibrary = v2Model
      ? (await import('../lib/race-primitives-scenarios')).buildLearnedRaceLibraryV2(v2Model)
      : null;
    const course = (await import('../lib/race-primitives-scenarios')).buildRaceCourse();
    const md = buildDebugReport({
      phase,
      useV2,
      v2Active,
      winner,
      v2Model,
      v2Meta,
      kinematicMetrics: metrics.kinematic,
      learnedMetrics: metrics.learned,
      kinematicLapTimes: learner?.kinematicLapTimes ?? [],
      learnedLapTimes: learner?.learnedLapTimes ?? [],
      kinematicSectors: learner?.kinematicSectors ?? [],
      learnedSectors: learner?.learnedSectors ?? [],
      waypointCount: course.waypoints.length,
      kinematicLibrary,
      learnedLibrary,
      startSpeeds: RACE_START_SPEEDS,
      plannerConfig: {
        lookaheadCount: PLAN_LOOKAHEAD_COUNT,
        replanIntervalMs: REPLAN_INTERVAL_MS,
        perCarBudgetMs: RACE_REPLAN_BUDGET_MS,
        plannerGateRadius: RACE_PLANNER_GATE_RADIUS,
        advanceRadius: 2.5,
        trackerMaxLateralAccel: TRACKER_MAX_LATERAL_ACCEL,
      },
    });
    const filename = `kinocat-raceprimitives-debug-${new Date().toISOString().replace(/[:.]/g, '-')}.md`;
    // Always download; also try to copy so the user can paste straight
    // into a chat for diagnosis.
    downloadMarkdown(md, filename);
    await copyToClipboard(md);
    setDebugExportedAt(Date.now());
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
        trackerMode={trackerMode}
        onTrackerMode={setTrackerMode}
        courseVariant={courseVariant}
        onCourseVariant={setCourseVariant}
        learnedModel={v2Active ? 'v2' : 'kinematic'}
        onLearnedModel={(m) => setUseV2(m === 'v2')}
        canUseV2={v2Model !== null}
        feedforward={feedforward}
        onFeedforward={setFeedforward}
        phase={phase}
        learnProgress={learnProgress}
        winner={winner}
        params={params}
        error={error}
        v2Active={v2Active}
        isMobile={isMobile}
        onLearn={runInlineLearn}
        onStart={startRace}
        onStop={stopRace}
        onReset={resetRace}
        onClearCache={clearCache}
        onExportDebug={onExportDebug}
        onOpenModelLab={() => setModelLabOpen(true)}
        debugExportedAt={debugExportedAt}
      />
      <div style={{ flex: 1, position: 'relative' }}>
        <div ref={containerRef} style={{ position: 'absolute', inset: 0 }} />
        <MetricsOverlay
          metrics={metrics}
          winner={winner}
          isMobile={isMobile}
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
          stacks={{
            kinematic: {
              tracker: trackerMode === 'mpc' ? 'MPPI (progress)' : 'pure-pursuit',
              library: 'kinematic bicycle',
              rolloutModel: trackerMode === 'mpc' ? 'kinematic bicycle' : undefined,
            },
            learned: {
              tracker: trackerMode === 'mpc' ? 'MPPI (progress)' : 'pure-pursuit',
              library: v2Active ? 'v2 learned' : 'online-refit (legacy)',
              rolloutModel:
                trackerMode === 'mpc'
                  ? (v2Active ? 'v2 learned' : 'v2 default (shared)')
                  : undefined,
            },
          }}
        />
        {(phase === 'racing' || phase === 'finished') && learner && (
          <LearnerPanel snap={learner} v2Active={v2Active} v2Meta={v2Meta} isMobile={isMobile} />
        )}
        {(phase === 'racing' || phase === 'finished') && !isMobile && (
          <EvalHUD
            entries={[
              { label: 'kinematic', color: C.kinematic, snap: evalSnap.kinematic },
              { label: 'learned', color: C.learned, snap: evalSnap.learned },
            ]}
          />
        )}
        {phase === 'learning' && (
          <PretrainOverlay progress={learnProgress} />
        )}
        <ModelLab
          open={modelLabOpen}
          onOpenChange={setModelLabOpen}
          onTrained={onV2Trained}
          loadedMeta={v2Meta}
          onClearLoaded={v2Meta ? onV2Clear : undefined}
          onExport={v2Model ? onV2Export : undefined}
          useV2={useV2}
          onToggleUseV2={setUseV2}
          hasV2Model={v2Model !== null}
          onResetToDefault={resetToPreloadedDefault}
          hasPreloadedDefault={hasPreloadedDefault}
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
  onEval: (k: EvalSnapshot, l: EvalSnapshot) => void;
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
  /** Forward dynamics model the LEARNED car's MPPI tracker rolls when the
   *  tracker is `'mpc'` (the trained v2 sim — its fidelity reaching the
   *  wheels). Unused under pure-pursuit. */
  learnedForwardModel?: import('kinocat/primitives').ForwardSim<CarKinematicState>;
  /** Path-tracking executor for both cars. Chosen in the Race Setup GUI
   *  (top bar). Defaults to pure-pursuit. */
  tracker?: TrackerMode;
  /** Course layout. Chosen in the Race Setup GUI (top bar). Defaults to the
   *  open flat pad. */
  courseVariant?: CourseVariant;
  /** WS-1½ control feedforward (MPPI only). When true, the tracker warm-starts
   *  its prior from the plan's own primitive controls instead of re-deriving
   *  them from geometry. Chosen in the Race Setup GUI (top bar). Default off. */
  feedforward?: boolean;
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
  // Course variant comes from the Race Setup GUI (top bar). The walled /
  // chicane / thread-the-gate `technical` layout turns corner overshoot into
  // a physical wall strike (see race-primitives-scenarios.ts); `open` is the
  // flat pad. The `?course=` param only seeds the GUI's initial value.
  const courseVariant: CourseVariant = options.courseVariant ?? courseFromUrl();
  const course = buildRaceCourse(courseVariant);
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

  // Build the shared RaceScenario that owns the simulation (per-car
  // Rapier world, planner, pure-pursuit, lap detection, sync hold,
  // stall + off-track recovery). The React component below is purely
  // a renderer — it consumes scenario.status() each frame to update
  // meshes / trail / lookahead marker. Online learning is intentionally
  // not part of the scenario: per-lap legacy 5-param refits were
  // removed in favor of the offline-trained v2 model (see
  // `pnpm run train` / Model Lab).
  // Tracker from the Race Setup GUI (top bar). Under MPPI each car tracks
  // with its OWN forward model — the kinematic car with the naive idealised-
  // bicycle params, the learned car with the trained v2 sim — so model
  // fidelity reaches the wheels, not just the plan. `?tracker=` seeds it.
  const tracker: TrackerMode = options.tracker ?? trackerFromUrl();
  const scenario = await createRaceScenario({
    entries: [
      {
        name: 'kinematic',
        lib: kinematicLib,
        forwardModel: parametricForwardV2(KINEMATIC_NATIVE_PARAMS, DEFAULT_LEARNABLE_CONFIG),
      },
      {
        name: 'learned',
        lib: initialLearnedLib,
        forwardModel: options.learnedForwardModel,
      },
    ],
    syncHold: true,
    offTrackRecovery: 'waypoint',
    course,
    tuning: { tracker, controlFeedforward: options.feedforward ?? false },
  });

  // ---- Per-car setup ----
  function makeCar(
    id: 'kinematic' | 'learned',
    lib: MotionPrimitiveLibrary,
    color: number,
    pathColor: number,
    _learner: OnlineLearnerState | undefined, // kept for shape compat; always undefined
  ): CarRuntime {
    const world = scenario.getWorld(id)!;
    const car = scenario.getCarHandle(id)!;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(C.bg);
    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const sun = new THREE.DirectionalLight(0xffffff, 0.9);
    sun.position.set(40, 100, 20);
    scene.add(sun);
    scene.add(createGroundPlaneHelper({ bounds: RACE_BOUNDS, color: 0x141a26 }));
    // Technical-course walls (empty on the open course). Rendered as slate
    // blocks with edge wireframes; these are the same boxes the planner sees
    // (inflated) as obstacles and the physics world has as colliders.
    for (const w of course.walls ?? []) {
      scene.add(
        createBuildingHelper(
          { x: w.x, z: w.z, hx: w.hx, hz: w.hz, height: w.height },
          { color: 0x3a4458, edgeColor: 0x8fa2c0 },
        ),
      );
    }
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
      learner: _learner,
    };
  }

  const kinematic = makeCar('kinematic', kinematicLib, C.kinematic, C.kinematicPath, undefined);
  // Learned car starts with the pre-trained library when available, else
  // with the kinematic library. Either way, it refines per-lap online
  // using race data via fitParamsOnline. When the v2 override is active,
  // online refitting is suppressed (passing `undefined`) so the offline-
  // trained v2 library is what's actually driving — mixing in an online
  // refit of the legacy 5-param model would silently shadow the v2 lib.
  // Online learning is intentionally OFF: the project now trains v2
  // offline via `pnpm run train` / Model Lab, so per-lap legacy 5-param
  // refits are no longer wired in. The learner field is kept on
  // CarRuntime for backward compatibility with downstream code (snapshot
  // emission, LearnerPanel) but is always undefined, so all online-refit
  // branches are dead code.
  const learnerState: OnlineLearnerState | undefined = undefined;
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
  // (Suspension settle + spawn snap happens inside `createRaceScenario`.)

  // ---- State ----
  let running = false;
  let raceStartWall = 0;

  // ---- Eval probes (read-only observability; never feed back into the sim) ----
  // Score each car's plan-vs-execution with the SAME kinocat/eval functions the
  // CLI harness + tests use. `frictionLimit` is the race tracker's lateral-accel
  // budget so the friction-circle reading matches the controller's envelope.
  const evalLimits = limitsFromAgent(RACE_AGENT, {
    frictionLimit: TRACKER_MAX_LATERAL_ACCEL,
    maxAccel: 6,
    maxDecel: 8,
  });
  const makeProbe = () =>
    createEvalProbe({ footprint: RACE_AGENT.footprint, dt: PHYSICS_DT, limits: evalLimits });
  let kProbe = makeProbe();
  let lProbe = makeProbe();

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
  /** Mirror a scenario per-car status into the React-side `CarRuntime`
   *  visual + metrics fields. Pure: never touches Rapier (the scenario
   *  owns the chassis); only updates Three.js meshes + the demo's
   *  in-React mirror of lap state. */
  function mirrorStatus(car: CarRuntime, status: RaceCarStatus, now: number): void {
    const after = status.state;
    syncCarMesh(car.carMesh.group, after);
    // Ideal/reference line to the current target waypoint.
    const wp = course.waypoints[status.loopIndex]!;
    const pts = [
      new THREE.Vector3(after.x, 0.25, after.z),
      new THREE.Vector3(wp.x, 0.25, wp.z),
    ];
    car.idealLine.geometry.dispose();
    car.idealLine.geometry = new THREE.BufferGeometry().setFromPoints(pts);
    car.idealLine.computeLineDistances();
    // Plan polyline (if a plan is available). Trim the segment we've
    // already executed, then prepend the chassis position so the line
    // starts exactly where the car is and follows ONLY the unexecuted
    // future. With the commit-window stitch the plan's first sample
    // may be the predicted future state (not the live chassis) — the
    // prepended chassis point bridges that small gap so the rendered
    // line is "from where I am, along what I plan to do" rather than
    // including a stale tail or a phantom future-start segment.
    if (status.plan && status.plan.length >= 2) {
      if (car.pathLine) {
        car.scene.remove(car.pathLine);
        car.pathLine.geometry.dispose();
        (car.pathLine.material as THREE.Material).dispose();
      }
      const elapsed = Math.max(0, scenario.simTime() - status.planStartSimTime);
      const tail = trimPlan(status.plan, elapsed);
      const pl: THREE.Vector3[] = [new THREE.Vector3(after.x, 0.4, after.z)];
      for (const p of tail) pl.push(new THREE.Vector3(p.x, 0.4, p.z));
      car.pathLine = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(pl),
        new THREE.LineBasicMaterial({ color: car.pathColor, transparent: true, opacity: 0.85 }),
      );
      car.scene.add(car.pathLine);
    }
    // Lookahead marker is hidden in the scenario-driven path — the
    // scenario doesn't expose it (it's an internal pure-pursuit detail).
    car.lookaheadMarker.visible = false;
    // Mirror lap state into runtime fields the existing UI panels read.
    car.metrics = status.metrics;
    car.waypointsCleared = status.metrics.waypointsCleared;
    car.holdingForSync = status.holdingForSync;
    car.lapTimes = status.laps.map((l) => l.duration);
    car.sectorTimes = status.laps.map((l) => l.sectors);
    car.currentLapSectors = []; // not surfaced from scenario; ok for visuals
    car.ai.loopIndex = status.loopIndex;
    car.ai.plan = status.plan;
    car.ai.planStartWall = now - Math.max(0, (scenario.simTime() - status.planStartSimTime) * 1000);
    car.ai.predictedEnd = null;
    // Trail.
    const lastPt = car.trailPts[car.trailPts.length - 1]!;
    if (Math.hypot(after.x - lastPt.x, after.z - lastPt.z) > 0.4) {
      car.trailPts.push(new THREE.Vector3(after.x, 0.15, after.z));
      if (car.trailPts.length > 1000) car.trailPts.shift();
      car.trailLine.geometry.dispose();
      car.trailLine.geometry = new THREE.BufferGeometry().setFromPoints(car.trailPts);
    }
  }

  /** Replan + stepCar are kept as no-ops because the shared scenario
   *  drives both internally. They remain as named functions because the
   *  rest of this file references them in deprecated control flow
   *  (e.g. the sync-hold release path); turning them into no-ops keeps
   *  the surrounding code shape stable. */
  function replan(_car: CarRuntime, _now: number): void { /* handled by scenario */ }
  function stepCar(_car: CarRuntime, _now: number, _dt: number): void { /* handled by scenario */ }

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
  // The shared scenario auto-replans every REPLAN_INTERVAL_MS internally.
  // No explicit replan timer needed in the React layer anymore. The
  // `replanTimer` variable is retained as a no-op so the cleanup path
  // (clearInterval below) doesn't need to be re-shaped.
  const replanTimer: number = 0;

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
      // Advance the shared simulation. The scenario steps every car's
      // Rapier world, runs the planner on the REPLAN_INTERVAL_MS cadence,
      // detects waypoint advances + laps, applies sync-hold, runs the
      // stall + off-track recovery. Returns per-car status which the
      // React layer mirrors into Three.js meshes.
      const r = scenario.tick(dt);
      // r.cars order matches the `entries` array we passed to
      // createRaceScenario: [kinematic, learned].
      mirrorStatus(kinematic, r.cars[0]!, now);
      mirrorStatus(learned, r.cars[1]!, now);
      cb.onMetrics(kinematic.metrics, learned.metrics);
      // Feed the eval probes every tick (read-only; cannot affect determinism).
      // RaceCarStatus is structurally a MonitorSample with a richer plan. Pass
      // the real frame dt — the scenario steps physics by this variable amount,
      // so the probe's finite differences must use it, not a fixed 1/60.
      kProbe.sample(r.cars[0]!, dt);
      lProbe.sample(r.cars[1]!, dt);
      // Emit the comparison snapshot in BOTH modes (legacy + v2). The lap-
      // times / sector-deltas are universal; only the per-coef refit fields
      // are legacy-only and get safe defaults when v2 is driving.
      if (now - lastLearnerEmit > 250) {
        lastLearnerEmit = now;
        cb.onEval(kProbe.snapshot(), lProbe.snapshot());
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
      scenario.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === mount) {
        mount.removeChild(renderer.domElement);
      }
    },
    start() {
      resetCar(kinematic);
      resetCar(learned);
      // Fresh probes per race so rolling RMSE / peaks don't carry over.
      kProbe = makeProbe();
      lProbe = makeProbe();
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
      kProbe = makeProbe();
      lProbe = makeProbe();
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

/** Smoothly move a perspective camera into a chase position behind a
 *  CarKinematicState. The chassis y is unknown to the planner (Y is derived in
 *  kinocat); for the cam we place it ~10m above so the wheels and the
 *  upcoming course are both visible. */
function updateChaseCamera(cam: THREE.PerspectiveCamera, s: CarKinematicState): void {
  const c = Math.cos(s.heading);
  const sn = Math.sin(s.heading);
  // 14m behind + 7m above the chassis, looking ~6m ahead.
  const target = new THREE.Vector3(s.x - 14 * c, 7, s.z - 14 * sn);
  cam.position.lerp(target, 0.12);
  cam.lookAt(s.x + 6 * c, 1.2, s.z + 6 * sn);
}

function planAtTime(plan: CarKinematicState[], t: number): CarKinematicState | null {
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
  trackerMode,
  onTrackerMode,
  courseVariant,
  onCourseVariant,
  learnedModel,
  onLearnedModel,
  canUseV2,
  feedforward,
  onFeedforward,
  isMobile,
  onLearn,
  onStart,
  onStop,
  onReset,
  onClearCache,
  onExportDebug,
  onOpenModelLab,
  debugExportedAt,
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
  /** Race Setup: which path-tracking executor this mount runs. */
  trackerMode: TrackerMode;
  onTrackerMode: (m: TrackerMode) => void;
  /** Race Setup: which course layout. */
  courseVariant: CourseVariant;
  onCourseVariant: (c: CourseVariant) => void;
  /** Race Setup: the LEARNED car's plan library / rollout model. */
  learnedModel: LearnedModelChoice;
  onLearnedModel: (m: LearnedModelChoice) => void;
  /** Whether a trained v2 model is available to select. */
  canUseV2: boolean;
  /** Race Setup: WS-1½ control feedforward (MPPI only). */
  feedforward: boolean;
  onFeedforward: (on: boolean) => void;
  isMobile: boolean;
  onLearn: () => void;
  onStart: () => void;
  onStop: () => void;
  onReset: () => void;
  onClearCache: () => void;
  /** Generate + download + clipboard-copy a Markdown debug report of
   *  the live page state (model, race, libraries, planner config). */
  onExportDebug: () => void;
  /** Open the Model Lab drawer (train / load / inspect the v2 model). */
  onOpenModelLab: () => void;
  /** ms-since-epoch the last export fired — used to show a brief
   *  "copied + downloaded" confirmation. */
  debugExportedAt: number;
}) {
  const justExported = debugExportedAt > 0 && Date.now() - debugExportedAt < 3000;
  const subtitle = v2Active
    ? 'kinematic vs offline-trained v2 · online refit off'
    : 'kinematic vs online-learning · learned car refits each lap';
  // Compact one-glance summary of the current setup, shown on the Setup
  // dropdown trigger so the config is visible without opening it.
  const setupSummary = [
    trackerMode === 'mpc' ? 'MPPI' : 'PP',
    learnedModel === 'v2' ? 'v2' : 'kin',
    courseVariant === 'technical' ? 'Tech' : 'Open',
    ...(feedforward && trackerMode === 'mpc' ? ['FF'] : []),
  ].join(' · ');

  // Single-row toolbar. Everything that used to overflow now lives behind two
  // dropdowns (Setup + overflow ⋯) and a Model Lab launcher, so the header is
  // one line at every width. On narrow screens the brand + subtitle collapse
  // and an overflowX guard keeps it usable.
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        padding: isMobile ? '6px 8px' : '8px 14px',
        borderBottom: '1px solid #1f2735',
        background: '#0d1119',
        // No overflow clip here: a scroll container (overflow-x:auto forces
        // overflow-y:auto too) would clip the Setup / overflow dropdown panels
        // that drop BELOW the bar. The compact dropdown design keeps the row
        // within ~360px, so clipping is unnecessary.
      }}
    >
      <div style={{ color: '#7fd6ff', fontWeight: 700, whiteSpace: 'nowrap' }}>
        {isMobile ? 'race' : 'race the primitives'}
      </div>
      {!isMobile && (
        <div style={{ opacity: 0.5, fontSize: 12, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: 300, flexShrink: 1 }}>
          {subtitle}
        </div>
      )}

      {/* Setup dropdown — the three run-defining selectors. */}
      <Popover
        align="left"
        triggerTitle="Race setup — tracker, learned-car model, course"
        triggerLabel={
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            <span aria-hidden>⚙</span>
            {!isMobile && <span>Setup</span>}
            <span style={{ opacity: 0.7, color: '#7fd6ff' }}>{setupSummary}</span>
          </span>
        }
      >
        {() => (
          <RaceSetup
            trackerMode={trackerMode}
            onTrackerMode={onTrackerMode}
            courseVariant={courseVariant}
            onCourseVariant={onCourseVariant}
            learnedModel={learnedModel}
            onLearnedModel={onLearnedModel}
            canUseV2={canUseV2}
            feedforward={feedforward}
            onFeedforward={onFeedforward}
          />
        )}
      </Popover>

      <div style={{ flex: 1, minWidth: 8 }} />

      {/* Phase status + the single primary action for the current phase. */}
      {phase === 'loading' && <Status>loading…</Status>}
      {phase === 'learning' && (
        <Status>
          {isMobile
            ? `pre-train ${learnProgress.done}/${learnProgress.total || '?'}`
            : `pre-training… ${learnProgress.done}/${learnProgress.total || '?'}`}
        </Status>
      )}
      {phase === 'ready' && params && (
        <>
          {!isMobile && <Status>ready{v2Active ? ' · v2' : ''}</Status>}
          <Btn onClick={onStart}>start race</Btn>
        </>
      )}
      {phase === 'racing' && (
        <>
          {!isMobile && <Status>racing…</Status>}
          <Btn onClick={onStop}>stop</Btn>
        </>
      )}
      {phase === 'finished' && (
        <>
          {!isMobile && <Status>stopped{v2Active ? ' · v2' : ''}</Status>}
          <Btn onClick={onStart}>{isMobile ? 'race' : 'race again'}</Btn>
        </>
      )}
      {error && <Status warning>err</Status>}

      {/* Model Lab launcher — opens the drawer (train / load / inspect the v2
          model). Replaces the old floating top-right panel that overlapped
          the LEARNED metrics. */}
      <Btn onClick={onOpenModelLab} secondary title="Open Model Lab — train, load, or inspect the v2 model">
        {isMobile ? 'Lab' : 'Model Lab'}
        {canUseV2 && <span style={{ marginLeft: 6, color: '#55dcff' }}>●</span>}
      </Btn>

      {/* Overflow menu — secondary actions that don't need to be one click. */}
      <Popover align="right" triggerLabel={<span aria-hidden style={{ fontSize: 16, lineHeight: 1 }}>⋯</span>} triggerTitle="More actions">
        {(close) => (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4, minWidth: 170 }}>
            <MenuItem onClick={() => { onLearn(); close(); }} disabled={phase === 'learning'}>Pre-train v2…</MenuItem>
            <MenuItem onClick={() => { onReset(); close(); }}>Reset race</MenuItem>
            <MenuItem onClick={() => { onClearCache(); close(); }}>Clear cached model</MenuItem>
            <div style={{ height: 1, background: '#223044', margin: '2px 0' }} />
            <MenuItem onClick={() => { onExportDebug(); close(); }}>
              {justExported ? '✓ copied · saved' : '🐛 Export debug report'}
            </MenuItem>
          </div>
        )}
      </Popover>
    </div>
  );
}

/** A full-width row button used inside the overflow / setup popovers. */
function MenuItem({
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
      role="menuitem"
      onClick={onClick}
      disabled={disabled}
      style={{
        textAlign: 'left',
        padding: '7px 10px',
        background: 'transparent',
        border: 'none',
        borderRadius: 4,
        color: disabled ? '#4a5364' : '#cdd3de',
        font: '12px ui-monospace, monospace',
        cursor: disabled ? 'not-allowed' : 'pointer',
        whiteSpace: 'nowrap',
      }}
      onMouseEnter={(e) => { if (!disabled) e.currentTarget.style.background = '#1b2740'; }}
      onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; }}
    >
      {children}
    </button>
  );
}

/** Human-readable description of each car's live control stack: which
 *  path tracker executes the plan, and which planning/rollout model the
 *  car reasons with. Rendered in the HUD so a screenshot of a run is
 *  never ambiguous about WHAT was being tested. */
interface StackInfo {
  /** Path-tracking executor. */
  tracker: string;
  /** Planner primitive library (what A* searches). */
  library: string;
  /** MPPI rollout model (only meaningful under the mpc tracker). */
  rolloutModel?: string;
}

function MetricsOverlay({
  metrics,
  winner,
  isMobile,
  holding,
  lapTimes,
  rollbackActive,
  bestLapNumber,
  stacks,
}: {
  metrics: { kinematic: RaceMetrics; learned: RaceMetrics };
  winner: 'kinematic' | 'learned' | 'tie' | null;
  isMobile: boolean;
  holding: { kinematic: boolean; learned: boolean };
  lapTimes: { kinematic: number[]; learned: number[] };
  rollbackActive: boolean;
  bestLapNumber: number;
  stacks: { kinematic: StackInfo; learned: StackInfo };
}) {
  // Mobile: compact stacked summary row at the top of the viewport. Tap a
  // card to expand its full stats (LIVE CONTROLS + tracking error + …).
  if (isMobile) {
    return (
      <div
        style={{
          position: 'absolute',
          top: 8,
          left: 8,
          right: 8,
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: 6,
          zIndex: 30,
          pointerEvents: 'none',
        }}
      >
        <CompactCarCard
          title="KINEMATIC"
          color="#ff8aa0"
          m={metrics.kinematic}
          highlight={winner === 'kinematic'}
          holding={holding.kinematic}
          recentLaps={lapTimes.kinematic}
          rollbackBadge={null}
          stack={stacks.kinematic}
        />
        <CompactCarCard
          title="LEARNED"
          color="#55dcff"
          m={metrics.learned}
          highlight={winner === 'learned'}
          holding={holding.learned}
          recentLaps={lapTimes.learned}
          rollbackBadge={rollbackActive ? `BEST l${bestLapNumber}` : null}
          stack={stacks.learned}
        />
      </div>
    );
  }
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
        stack={stacks.kinematic}
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
        stack={stacks.learned}
      />
    </>
  );
}

/** Tightly-packed per-car card for mobile. Tap to expand for full stats. */
function CompactCarCard({
  title, color, m, highlight, holding, recentLaps, rollbackBadge, stack,
}: {
  title: string; color: string; m: RaceMetrics; highlight: boolean;
  holding: boolean; recentLaps: number[]; rollbackBadge: string | null;
  stack: StackInfo;
}) {
  const [open, setOpen] = useState(false);
  const last5 = recentLaps.slice(-5);
  const mean5 = last5.length > 0 ? last5.reduce((a, b) => a + b, 0) / last5.length : Number.NaN;
  return (
    <div
      onClick={() => setOpen((v) => !v)}
      style={{
        background: 'rgba(13, 17, 25, 0.92)',
        border: `1px solid ${holding ? '#ffd070' : highlight ? color : '#1f2735'}`,
        borderRadius: 6,
        padding: '6px 8px',
        color: '#cdd3de',
        font: '11px ui-monospace, monospace',
        pointerEvents: 'auto',
        cursor: 'pointer',
        boxShadow: highlight ? `0 0 12px ${color}66` : 'none',
        overflow: 'hidden',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
        <span style={{ color, fontWeight: 700, fontSize: 10, letterSpacing: 0.5 }}>{title}</span>
        {holding && <span style={{ color: '#ffd070', fontSize: 9, padding: '1px 4px', border: '1px solid #ffd070', borderRadius: 3 }}>WAIT</span>}
        {rollbackBadge && <span style={{ color: '#a6e9ff', fontSize: 9, padding: '1px 4px', border: '1px solid #55dcff', borderRadius: 3 }}>{rollbackBadge}</span>}
        <span style={{ marginLeft: 'auto', opacity: 0.6, fontSize: 9 }}>{open ? '▼' : '▶'}</span>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', columnGap: 6, rowGap: 1 }}>
        <span style={{ opacity: 0.6 }}>t</span>
        <span style={{ textAlign: 'right' }}>{m.raceTime.toFixed(1)}s</span>
        <span style={{ opacity: 0.6 }}>lap</span>
        <span style={{ textAlign: 'right' }}>{m.laps} · {Number.isFinite(m.bestLapTime) ? `${m.bestLapTime.toFixed(2)}s` : '—'}</span>
        {open && (
          <>
            <span style={{ opacity: 0.6 }}>tracker</span>
            <span style={{ textAlign: 'right', color }}>{stack.tracker}</span>
            <span style={{ opacity: 0.6 }}>lib</span>
            <span style={{ textAlign: 'right' }}>{stack.library}</span>
            {stack.rolloutModel && (
              <>
                <span style={{ opacity: 0.6 }}>model</span>
                <span style={{ textAlign: 'right' }}>{stack.rolloutModel}</span>
              </>
            )}
            <span style={{ opacity: 0.6 }}>last</span>
            <span style={{ textAlign: 'right' }}>{Number.isFinite(m.lastLapTime) ? `${m.lastLapTime.toFixed(2)}s` : '—'}</span>
            <span style={{ opacity: 0.6 }}>mean5</span>
            <span style={{ textAlign: 'right' }}>{Number.isFinite(mean5) ? `${mean5.toFixed(2)}s` : '—'}</span>
            <span style={{ opacity: 0.6 }}>wp</span>
            <span style={{ textAlign: 'right' }}>{m.waypointsCleared}</span>
            <span style={{ opacity: 0.6 }}>spd</span>
            <span style={{ textAlign: 'right' }}>{Math.abs(m.liveControls.targetSpeed).toFixed(1)} → {m.peakSpeed.toFixed(1)} m/s</span>
            <span style={{ opacity: 0.6 }}>thr</span>
            <span style={{ textAlign: 'right' }}>
              {(m.liveControls.throttle * 100).toFixed(0)}%
              {m.liveControls.brake > 0 && ` · brk ${(m.liveControls.brake * 100).toFixed(0)}%`}
            </span>
            <span style={{ opacity: 0.6 }}>steer</span>
            <span style={{ textAlign: 'right' }}>
              {m.liveControls.steer >= 0 ? '+' : ''}{m.liveControls.steer.toFixed(2)}
            </span>
            <span style={{ opacity: 0.6 }}>err</span>
            <span style={{ textAlign: 'right' }}>{m.trackingErrorRms.toFixed(2)}m</span>
            <span style={{ opacity: 0.6 }}>plan</span>
            <span style={{
              textAlign: 'right',
              color: !m.planDiagnostics.lastReplanFound ? '#ff5566'
                : m.planDiagnostics.lastReplanMs > 100 ? '#ffd070'
                : '#cdd3de',
            }}>
              {m.planDiagnostics.lastReplanMs.toFixed(0)}ms{m.planDiagnostics.lastReplanFound ? '' : ' FAIL'}
            </span>
            <span style={{ opacity: 0.6 }}>age</span>
            <span style={{
              textAlign: 'right',
              color: m.planDiagnostics.planAgeMs > 800 ? '#ff5566'
                : m.planDiagnostics.planAgeMs > 400 ? '#ffd070'
                : '#cdd3de',
            }}>
              {m.planDiagnostics.planAgeMs.toFixed(0)}ms
            </span>
          </>
        )}
      </div>
    </div>
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
  stack,
}: {
  side: 'left' | 'right';
  title: string;
  color: string;
  m: RaceMetrics;
  highlight: boolean;
  holding: boolean;
  recentLaps: number[];
  rollbackBadge: string | null;
  stack: StackInfo;
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
        <div style={{ fontSize: 10, opacity: 0.7, marginBottom: 2 }}>CONTROL STACK</div>
        <KV k="tracker" v={<span style={{ color }}>{stack.tracker}</span>} />
        <KV k="plan library" v={stack.library} />
        {stack.rolloutModel && <KV k="MPPI model" v={stack.rolloutModel} />}
        {m.mpcSolveCount > 0 && (
          <KV k="MPPI solve" v={`${m.mpcSolveMsAvg.toFixed(1)} ms · ${m.mpcSolveCount}`} />
        )}
      </div>
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
      <div style={{ marginTop: 6, paddingTop: 6, borderTop: '1px solid #1f2735', opacity: 0.85 }}>
        <div style={{ fontSize: 10, opacity: 0.7, marginBottom: 2 }}>PLANNER</div>
        {(() => {
          const d = m.planDiagnostics;
          const ok = d.successfulReplans / Math.max(1, d.totalReplans);
          const replanColor =
            !d.lastReplanFound ? '#ff5566'
              : d.lastReplanMs > 100 ? '#ffd070'
              : '#cdd3de';
          const ageColor =
            d.planAgeMs > 800 ? '#ff5566'
              : d.planAgeMs > 400 ? '#ffd070'
              : '#cdd3de';
          return (
            <>
              <KV
                k="last replan"
                v={
                  <span style={{ color: replanColor }}>
                    {d.lastReplanMs.toFixed(0)}ms · {d.lastReplanFound ? 'ok' : 'FAIL'}
                  </span>
                }
              />
              <KV
                k="plan age"
                v={<span style={{ color: ageColor }}>{d.planAgeMs.toFixed(0)} ms</span>}
              />
              <KV
                k="success rate"
                v={`${(ok * 100).toFixed(0)}% (${d.successfulReplans}/${d.totalReplans})`}
              />
              {d.consecutiveFailedReplans > 0 && (
                <KV
                  k="failed streak"
                  v={<span style={{ color: '#ff5566' }}>{d.consecutiveFailedReplans}</span>}
                />
              )}
            </>
          );
        })()}
      </div>
    </div>
  );
}

function LearnerPanel({ snap, v2Active, v2Meta, isMobile }: {
  snap: LearnerSnapshot;
  v2Active: boolean;
  v2Meta?: PersistedV2Model['meta'] | null;
  isMobile: boolean;
}) {
  // On mobile the panel is collapsible (chart + sector deltas only when
  // expanded) so it doesn't eat the viewport. The expand toggle defaults
  // to closed because the per-car cards at the top already show laps.
  const [open, setOpen] = useState(!isMobile);
  return (
    <div
      style={{
        position: 'absolute',
        bottom: isMobile ? 0 : 12,
        left: isMobile ? 0 : 12,
        right: isMobile ? 0 : 12,
        background: 'rgba(13, 17, 25, 0.94)',
        border: '1px solid #1f2735',
        borderTopLeftRadius: 8,
        borderTopRightRadius: 8,
        borderBottomLeftRadius: isMobile ? 0 : 8,
        borderBottomRightRadius: isMobile ? 0 : 8,
        padding: isMobile ? '6px 10px 10px' : '10px 14px',
        color: '#cdd3de',
        font: '11px ui-monospace, monospace',
        display: 'flex',
        flexDirection: 'column',
        gap: isMobile ? 6 : 10,
        maxHeight: isMobile ? '60vh' : 'none',
        overflowY: isMobile ? 'auto' : 'visible',
        zIndex: 20,
      }}
    >
      {isMobile && (
        <button
          onClick={() => setOpen((v) => !v)}
          style={{
            background: 'transparent', border: 'none', color: '#cdd3de',
            font: 'inherit', cursor: 'pointer', textAlign: 'left',
            padding: '2px 0', display: 'flex', alignItems: 'center', gap: 8,
          }}
        >
          <span style={{ opacity: 0.6 }}>{open ? '▼' : '▶'}</span>
          <span style={{ color: '#55dcff', fontWeight: 700 }}>
            {v2Active ? 'V2 MODEL' : 'ONLINE LEARNER'}
          </span>
          {!open && Number.isFinite(snap.kinematicLapTimes[snap.kinematicLapTimes.length - 1] ?? NaN) && (
            <span style={{ opacity: 0.7, marginLeft: 'auto' }}>
              {snap.kinematicLapTimes.length} laps · tap to expand
            </span>
          )}
        </button>
      )}
      {open && (
        <>
          <LapTimeChart
            kinematic={snap.kinematicLapTimes}
            learned={snap.learnedLapTimes}
          />
          <SectorDeltaStrip
            kinematicSectors={snap.kinematicSectors}
            learnedSectors={snap.learnedSectors}
            sectorsPerLap={snap.sectorsPerLap}
          />
        </>
      )}
      {open && (
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: isMobile
            ? 'minmax(0, 1fr)'
            : 'minmax(0, 1.4fr) minmax(0, 1fr) minmax(0, 1fr) minmax(0, 1fr)',
          gap: isMobile ? 10 : 16,
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
      )}
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

function KV({ k, v }: { k: string; v: string | React.ReactNode }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
      <span style={{ opacity: 0.7 }}>{k}</span>
      <span style={{ color: '#cdeaff' }}>{v}</span>
    </div>
  );
}

/** Race Setup selectors, laid out vertically for the top-bar "Setup"
 *  dropdown. The GUI source of truth for the three run-defining choices:
 *  which path tracker executes the plan, which plan library / rollout model
 *  the LEARNED car drives with, and which course layout to race. Changing any
 *  selector re-mounts the scene (resets the race), matching the v2-library
 *  toggle. URL query params only SEED these on first load. */
function RaceSetup({
  trackerMode,
  onTrackerMode,
  courseVariant,
  onCourseVariant,
  learnedModel,
  onLearnedModel,
  canUseV2,
  feedforward,
  onFeedforward,
}: {
  trackerMode: TrackerMode;
  onTrackerMode: (m: TrackerMode) => void;
  courseVariant: CourseVariant;
  onCourseVariant: (c: CourseVariant) => void;
  learnedModel: LearnedModelChoice;
  onLearnedModel: (m: LearnedModelChoice) => void;
  canUseV2: boolean;
  feedforward: boolean;
  onFeedforward: (on: boolean) => void;
}) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, minWidth: 210 }}>
      <Segmented
        label="tracker"
        value={trackerMode}
        onChange={onTrackerMode}
        options={[
          { value: 'pure-pursuit', label: 'Pure-pursuit', title: 'Geometric path tracker — fast, reactive, no dynamics model.' },
          { value: 'mpc', label: 'MPPI', title: 'Sampling MPC that rolls each car’s OWN forward model in the loop (model fidelity → control quality).' },
        ]}
      />
      <Segmented
        label="learned car model"
        value={learnedModel}
        onChange={onLearnedModel}
        options={[
          { value: 'kinematic', label: 'Kinematic', title: 'Learned car uses the kinematic-bicycle library (baseline — same as the control car).' },
          {
            value: 'v2',
            label: 'v2 learned',
            title: canUseV2
              ? 'Learned car uses the offline-trained v2 library + (under MPPI) the v2 rollout model.'
              : 'Train or load a v2 model first (Model Lab).',
            disabled: !canUseV2,
          },
        ]}
      />
      <Segmented
        label="course"
        value={courseVariant}
        onChange={onCourseVariant}
        options={[
          { value: 'open', label: 'Open', title: 'Flat pad — pure dynamics + waypoint chase.' },
          { value: 'technical', label: 'Technical', title: 'Walled chicane — corner overshoot becomes a physical wall strike.' },
        ]}
      />
      <Segmented
        label="control feedforward"
        value={feedforward ? 'on' : 'off'}
        onChange={(v) => onFeedforward(v === 'on')}
        options={[
          { value: 'off', label: 'Off', title: 'MPPI re-derives controls from plan geometry each tick (baseline).' },
          {
            value: 'on',
            label: 'On',
            title: trackerMode === 'mpc'
              ? 'MPPI warm-starts its prior from the plan’s OWN primitive controls (WS-1½) — a faithful model’s plan drives its proven controls. Best with the v3/learned model.'
              : 'Feedforward applies under MPPI only — switch the tracker to MPPI to see an effect.',
            disabled: trackerMode !== 'mpc',
          },
        ]}
      />
      <div style={{ fontSize: 10, opacity: 0.5, lineHeight: 1.4 }}>
        Changing a setting restarts the race.
      </div>
    </div>
  );
}

/** Compact segmented (radio-group) selector — a label above a row of
 *  mutually-exclusive pill buttons. Keyboard + pointer accessible via native
 *  buttons; fills its container so it reads cleanly inside a dropdown. */
function Segmented<T extends string>({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: T;
  onChange: (v: T) => void;
  options: Array<{ value: T; label: string; title?: string; disabled?: boolean }>;
}) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }} role="group" aria-label={label}>
      <span style={{ fontSize: 10, opacity: 0.6, textTransform: 'uppercase', letterSpacing: 0.4 }}>{label}</span>
      <div style={{ display: 'flex', border: '1px solid #223044', borderRadius: 5, overflow: 'hidden' }}>
        {options.map((opt, i) => {
          const active = opt.value === value;
          return (
            <button
              key={opt.value}
              type="button"
              onClick={() => !opt.disabled && !active && onChange(opt.value)}
              disabled={opt.disabled}
              title={opt.title}
              aria-pressed={active}
              style={{
                flex: 1,
                padding: '5px 10px',
                border: 'none',
                borderLeft: i > 0 ? '1px solid #223044' : 'none',
                background: active ? '#55dcff' : 'transparent',
                color: opt.disabled ? '#4a5364' : active ? '#0a0d14' : '#cdd3de',
                font: '11px ui-monospace, monospace',
                fontWeight: active ? 700 : 400,
                cursor: opt.disabled ? 'not-allowed' : active ? 'default' : 'pointer',
                whiteSpace: 'nowrap',
              }}
            >
              {opt.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

/** Lightweight popover: a trigger button + a floating panel that closes on
 *  outside-click or Escape. No dependency; used for the toolbar's Setup and
 *  overflow menus so the header stays a single row on every viewport. */
function Popover({
  triggerLabel,
  triggerTitle,
  align = 'right',
  badge,
  children,
}: {
  triggerLabel: React.ReactNode;
  triggerTitle?: string;
  align?: 'left' | 'right';
  badge?: React.ReactNode;
  children: (close: () => void) => React.ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', onDoc);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDoc);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);
  return (
    <div ref={ref} style={{ position: 'relative', display: 'inline-flex' }}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        title={triggerTitle}
        aria-haspopup="menu"
        aria-expanded={open}
        style={{
          padding: '6px 10px',
          background: open ? '#1b2740' : 'transparent',
          border: '1px solid #223044',
          borderRadius: 4,
          color: '#cdd3de',
          font: 'inherit',
          cursor: 'pointer',
          display: 'inline-flex',
          alignItems: 'center',
          gap: 6,
          whiteSpace: 'nowrap',
        }}
      >
        {triggerLabel}
        {badge}
        <span style={{ opacity: 0.6, fontSize: 10 }}>{open ? '▲' : '▼'}</span>
      </button>
      {open && (
        <div
          role="menu"
          style={{
            position: 'absolute',
            top: 'calc(100% + 6px)',
            [align]: 0,
            zIndex: 80,
            background: '#0d1119',
            border: '1px solid #223044',
            borderRadius: 8,
            padding: 12,
            boxShadow: '0 8px 24px rgba(0, 0, 0, 0.5)',
            maxWidth: 'calc(100vw - 24px)',
          }}
        >
          {children(() => setOpen(false))}
        </div>
      )}
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
  title,
}: {
  children: React.ReactNode;
  onClick: () => void;
  secondary?: boolean;
  title?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      style={{
        font: '11px ui-monospace, monospace',
        padding: '6px 12px',
        borderRadius: 6,
        border: '1px solid #2a3340',
        background: secondary ? 'rgba(20, 26, 38, 0.85)' : 'rgba(127, 214, 255, 0.18)',
        color: secondary ? '#8c95a4' : '#cdeaff',
        cursor: 'pointer',
        letterSpacing: 0.3,
        whiteSpace: 'nowrap',
      }}
    >
      {children}
    </button>
  );
}
