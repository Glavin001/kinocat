// Demo-side orchestration glue: combines the headless Rapier trial harness
// (kinocat/adapters/rapier) with the generic learning helpers
// (kinocat/learning) to train the v2 vehicle model offline. Streams
// progress events so the Model Lab UI (RacePrimitives.tsx) can render live
// loss / coverage diagnostics.
//
// This is the use-case-specific consumer that wires the agnostic core
// pieces together for the /raceprimitives demo.

import type { WheeledCarControls } from 'kinocat/agent';
import type { CarKinematicState, LearnableVehicleConfig, LearnedVehicleParamsV2 } from 'kinocat/agent';
import {
  DEFAULT_LEARNED_PARAMS_V2,
  parametricForwardV2,
  paramsV2ToVec,
  paramsV2FromVec,
  PARAMS_V2_ORDER,
  buildParametricOnlyModel,
  type LearnedVehicleModel,
  buildMLPInput,
  MLP_INPUT_DIM,
  MLP_OUTPUT_DIM,
  learnedForwardSimV2,
  kinematicForwardSim,
  learnedForwardSim,
  DEFAULT_LEARNED_PARAMS,
  defaultVehicleAgent,
} from 'kinocat/agent';
import {
  createTrialStore,
  type Trial,
  type TrialStore,
  runParametricFit,
  runResidualMLPFit,
  evaluateModel,
  type ModelDiagnostics,
  type OpenLoopRow,
  type FitProgressEvent,
  proposeNextBatch,
  type ExplorationCell,
  assignSplit,
} from 'kinocat/learning';
import {
  createHeadlessTrialHarness,
  deriveLearnableConfig,
  type TrialSpec,
  type HeadlessTrialHarness,
} from 'kinocat/adapters/rapier';
import {
  defaultManeuverBundle,
  CAR_COVERAGE_AXES,
  carCoverageProjection,
  type ManeuverLimits,
  type ManeuverSpec,
} from 'kinocat/vehicle/car';
import {
  buildControlsTrace,
  createCoverageMeter,
  type CoverageCellSummary,
} from 'kinocat/training';
import type { ForwardSim } from 'kinocat/primitives';
import type {
  TrainingPipeline,
  TrainingContext,
  TrainedModel,
} from 'kinocat/training';

// ---------------------------------------------------------------------------
// Defaults — match the race-primitives chassis tuning.

export const DEFAULT_VEHICLE_OPTS = {
  chassisHalf: { x: 2.4, y: 0.5, z: 1.0 },
  chassisDensity: 60,
  wheelBase: 1.6,
  wheelTrack: 0.85,
  wheelRadius: 0.35,
  engineForce: 4000,
  brakeForce: 2000,
  maxSteerAngle: 0.6,
  driveTrain: 'rwd' as const,
};

// ---------------------------------------------------------------------------
// Event stream

export type TrainingEvent =
  | { type: 'round-start'; round: number; trialsBeforeRound: number }
  | { type: 'trial-batch'; round: number; collected: number; discarded: number }
  | { type: 'fit-progress'; round: number; phase: 'parametric' | 'residual'; event: FitProgressEvent }
  | { type: 'evaluation'; round: number; diagnostics: ModelDiagnostics }
  | { type: 'coverage'; round: number; cells: CoverageCellSummary[] }
  | { type: 'round-end'; round: number; trainedModel: LearnedVehicleModel; params: LearnedVehicleParamsV2; diagnostics: ModelDiagnostics; trialsAfter: number }
  | { type: 'done'; totalTrials: number; finalModel: LearnedVehicleModel; finalDiagnostics: ModelDiagnostics };

// ---------------------------------------------------------------------------
// Trial spec construction

interface CellSpec {
  startSpeed: number;
  steer: number;
  driveForce: number;
  brakeForce: number;
}

function specFor(cell: CellSpec, ticks: number, sampleEveryNTicks: number, id: string): TrialSpec {
  return {
    pose: { x: 0, z: 0, heading: 0 },
    kin: { forwardSpeed: cell.startSpeed },
    controlsTrace: Array.from({ length: ticks }, () => ({
      steer: cell.steer,
      driveForce: cell.driveForce,
      brakeForce: cell.brakeForce,
    })),
    sampleEveryNTicks,
    id,
  };
}

/** Build the seed grid (round 0): broad coverage.
 *
 *  IMPORTANT: includes high-speed (16, 20, 24, 28 m/s) trials so the fit
 *  has direct evidence for understeer / friction-circle behavior at race
 *  speeds. The earlier 0/4/8/12 grid trained a model that was essentially
 *  default-extrapolated at the speeds where the race actually happens —
 *  visible in /primitive-explorer as a collapsed v2 fan at 20+ m/s. */
export function buildSeedGrid(): CellSpec[] {
  const lowSpeeds = [0, 4, 8, 12];
  const highSpeeds = [16, 20, 24, 28];
  const driveStrong = DEFAULT_VEHICLE_OPTS.engineForce * 0.85;
  const driveMid = DEFAULT_VEHICLE_OPTS.engineForce * 0.5;
  const driveLow = DEFAULT_VEHICLE_OPTS.engineForce * 0.25;
  const brakeMid = DEFAULT_VEHICLE_OPTS.brakeForce * 0.7;
  const brakeLow = DEFAULT_VEHICLE_OPTS.brakeForce * 0.3;
  const maxSt = DEFAULT_VEHICLE_OPTS.maxSteerAngle;
  const cells: CellSpec[] = [];
  // Low-speed grid: full set of throttle / brake / turn variations
  for (const v of lowSpeeds) {
    cells.push({ startSpeed: v, steer: 0, driveForce: driveStrong, brakeForce: 0 });
    cells.push({ startSpeed: v, steer: 0, driveForce: 0, brakeForce: brakeMid });
    cells.push({ startSpeed: v, steer: 0, driveForce: 0, brakeForce: 0 }); // coast
    cells.push({ startSpeed: v, steer: +maxSt * 0.5, driveForce: driveMid, brakeForce: 0 });
    cells.push({ startSpeed: v, steer: -maxSt * 0.5, driveForce: driveMid, brakeForce: 0 });
    cells.push({ startSpeed: v, steer: +maxSt, driveForce: driveMid * 0.7, brakeForce: 0 });
    cells.push({ startSpeed: v, steer: -maxSt, driveForce: driveMid * 0.7, brakeForce: 0 });
    cells.push({ startSpeed: v, steer: +maxSt * 0.5, driveForce: 0, brakeForce: brakeMid * 0.6 });
  }
  // High-speed grid: gentle-turn + brake-into-corner trials. Tight turns
  // at high speed saturate the friction circle and don't add information
  // about understeer slope (they all clamp); GENTLE turns at high speed
  // are the diagnostic regime for the understeer coefficient.
  for (const v of highSpeeds) {
    cells.push({ startSpeed: v, steer: 0, driveForce: driveStrong, brakeForce: 0 });   // top-speed cruise
    cells.push({ startSpeed: v, steer: 0, driveForce: 0, brakeForce: brakeMid });      // brake from speed
    cells.push({ startSpeed: v, steer: 0, driveForce: 0, brakeForce: 0 });             // coast from speed
    cells.push({ startSpeed: v, steer: +maxSt * 0.2, driveForce: driveLow, brakeForce: 0 });  // VERY gentle right turn
    cells.push({ startSpeed: v, steer: -maxSt * 0.2, driveForce: driveLow, brakeForce: 0 });  // VERY gentle left turn
    cells.push({ startSpeed: v, steer: +maxSt * 0.4, driveForce: 0, brakeForce: 0 });          // moderate-coast turn
    cells.push({ startSpeed: v, steer: -maxSt * 0.4, driveForce: 0, brakeForce: 0 });
    cells.push({ startSpeed: v, steer: +maxSt * 0.3, driveForce: 0, brakeForce: brakeLow });   // trail-brake gentle
    cells.push({ startSpeed: v, steer: -maxSt * 0.3, driveForce: 0, brakeForce: brakeLow });
  }
  return cells;
}

/** Extreme-input probes that exercise saturation regimes. Include some
 *  high-speed scenarios so the model sees data at race speeds. */
export function extremeProbes(): CellSpec[] {
  return [
    { startSpeed: 8,  steer: DEFAULT_VEHICLE_OPTS.maxSteerAngle, driveForce: DEFAULT_VEHICLE_OPTS.engineForce, brakeForce: 0 },
    { startSpeed: 12, steer: -DEFAULT_VEHICLE_OPTS.maxSteerAngle, driveForce: DEFAULT_VEHICLE_OPTS.engineForce, brakeForce: 0 },
    { startSpeed: 10, steer: DEFAULT_VEHICLE_OPTS.maxSteerAngle, driveForce: 0, brakeForce: DEFAULT_VEHICLE_OPTS.brakeForce },
    { startSpeed: 12, steer: 0, driveForce: -DEFAULT_VEHICLE_OPTS.engineForce * 0.5, brakeForce: 0 },
    // High-speed extreme probes — the friction-circle saturation regime
    // the v2 race library specifically needs grounded.
    { startSpeed: 20, steer: DEFAULT_VEHICLE_OPTS.maxSteerAngle * 0.3, driveForce: 0, brakeForce: DEFAULT_VEHICLE_OPTS.brakeForce * 0.5 },
    { startSpeed: 24, steer: 0, driveForce: 0, brakeForce: DEFAULT_VEHICLE_OPTS.brakeForce },           // full brake from high
    { startSpeed: 28, steer: DEFAULT_VEHICLE_OPTS.maxSteerAngle * 0.15, driveForce: 0, brakeForce: 0 }, // gentle top-speed turn
    { startSpeed: 28, steer: -DEFAULT_VEHICLE_OPTS.maxSteerAngle * 0.15, driveForce: 0, brakeForce: 0 },
  ];
}

// ---------------------------------------------------------------------------
// Active-exploration cells (state × controls grid for the active-explorer)

interface SpeedSteerCellKey {
  speedBin: number; // [0..3]
  steerBin: number; // [0..2] = [neg, ~0, pos]
}

function cellKeyOf(c: CellSpec): string {
  const sb = c.startSpeed <= 2 ? 0 : c.startSpeed <= 6 ? 1 : c.startSpeed <= 10 ? 2 : 3;
  const tb = c.steer < -0.1 ? 0 : c.steer > 0.1 ? 2 : 1;
  return `s${sb}-t${tb}`;
}

// ---------------------------------------------------------------------------
// Trial collection

async function collectTrialBatch(
  harness: HeadlessTrialHarness,
  cells: CellSpec[],
  ticks: number,
  sampleEveryNTicks: number,
  startId: number,
): Promise<{ collected: Trial<CarKinematicState, WheeledCarControls, LearnableVehicleConfig>[]; discarded: number }> {
  const collected: Trial<CarKinematicState, WheeledCarControls, LearnableVehicleConfig>[] = [];
  let discarded = 0;
  let idx = startId;
  for (const c of cells) {
    const spec = specFor(c, ticks, sampleEveryNTicks, `t-${idx++}`);
    const result = harness.runTrial(spec);
    if (!result.ok) { discarded++; continue; }
    const t = result.trial;
    collected.push({
      id: t.id,
      initialState: t.samples[0]!,
      controlsTrace: spec.controlsTrace,
      dt: t.dt,
      samples: t.samples.map((s, i) => ({ t: i * sampleEveryNTicks * t.dt, state: s })),
      config: t.config,
      configKey: 'rwd-default',
    });
  }
  return { collected, discarded };
}

// ---------------------------------------------------------------------------
// Loss + evaluation

function stateDeltaForFit(pred: CarKinematicState, act: CarKinematicState): number {
  const dx = pred.x - act.x;
  const dz = pred.z - act.z;
  let dh = pred.heading - act.heading;
  while (dh > Math.PI) dh -= 2 * Math.PI;
  while (dh < -Math.PI) dh += 2 * Math.PI;
  const ds = pred.speed - act.speed;
  return dx * dx + dz * dz + 5 * dh * dh + ds * ds;
}

function controlsToVec(c: WheeledCarControls): number[] {
  return [c.steer, c.driveForce, c.brakeForce];
}

/** Build a coverage summary over the current store contents using the
 *  canonical car coverage axes. Pure: makes a fresh meter each time so
 *  callers don't accidentally accumulate across rounds. */
export function buildCoverageSummary(
  store: TrialStore<CarKinematicState, WheeledCarControls, LearnableVehicleConfig>,
): CoverageCellSummary[] {
  const meter = createCoverageMeter<CarKinematicState, WheeledCarControls, LearnableVehicleConfig>({
    axes: CAR_COVERAGE_AXES,
    project: carCoverageProjection,
    controlsToVec,
  });
  for (const t of store.all()) meter.record(t);
  return meter.summary();
}

export { CAR_COVERAGE_AXES };

function evaluate(
  store: TrialStore<CarKinematicState, WheeledCarControls, LearnableVehicleConfig>,
  model: LearnedVehicleModel,
): ModelDiagnostics {
  const all = store.all();
  if (all.length === 0) {
    return { openLoopDivergence: [], perStateRms: [], coverage: [], baselines: {} };
  }
  // Split-aware partitioning — Phase 0 of the training-dataset plan.
  // Falls back to the legacy last-25% slice when no trial in the store
  // carries a `split` field (e.g. the constant-hold pipeline being
  // exercised by `training-pipeline-equivalence.test.ts`).
  const hasSplit = all.some((t) => t.split !== undefined);
  let heldOut: ReadonlyArray<typeof all[number]>;
  let trainSet: ReadonlyArray<typeof all[number]>;
  let valSet: ReadonlyArray<typeof all[number]>;
  let testSet: ReadonlyArray<typeof all[number]>;
  if (hasSplit) {
    trainSet = store.all('train');
    valSet = store.all('val');
    testSet = store.all('test');
    // Headline numbers report the TEST set (frozen, never trained on);
    // when test is empty (small N) fall back to val, then to all trials.
    heldOut = testSet.length > 0 ? testSet : valSet.length > 0 ? valSet : all;
  } else {
    const cut = Math.max(1, Math.floor(all.length * 0.75));
    heldOut = all.slice(cut);
    trainSet = all.slice(0, cut);
    valSet = [];
    testSet = heldOut;
  }
  const horizons = [0.5, 1.0, 1.6];
  const agent = defaultVehicleAgent();
  const wheeledToLegacy = (c: WheeledCarControls): number[] => {
    const k = Math.sin(c.steer) / (2 * DEFAULT_VEHICLE_OPTS.wheelBase);
    const targetSpeed = c.driveForce > 0 ? 10 : (c.brakeForce > 0 ? 0 : 5);
    return [k, targetSpeed];
  };
  const composedSim = (
    inner: ForwardSim<CarKinematicState>,
    encode: (c: WheeledCarControls) => number[],
  ): ForwardSim<CarKinematicState> => (s, controls, dt) => inner(s, encode({
    steer: controls[0] ?? 0, driveForce: controls[1] ?? 0, brakeForce: controls[2] ?? 0,
  }), dt);

  const wrap = (a: number): number => {
    let d = a;
    while (d > Math.PI) d -= 2 * Math.PI;
    while (d < -Math.PI) d += 2 * Math.PI;
    return d;
  };
  // Headline diagnostics: full learnedForwardSimV2 vs baselines, evaluated
  // on the held-out (test / val / all) set.
  const headline = evaluateModel<CarKinematicState, WheeledCarControls, LearnableVehicleConfig>({
    trials: heldOut,
    horizons,
    controlsToVec,
    extractMetricFields: (s) => ({ x: s.x, z: s.z, heading: s.heading, speed: s.speed }),
    perStateRmsFields: [
      { name: 'heading', sqError: (p, a) => wrap(p.heading - a.heading) ** 2 },
      { name: 'speed', sqError: (p, a) => (p.speed - a.speed) ** 2 },
      { name: 'yawRate', sqError: (p, a) => ((p.yawRate ?? 0) - (a.yawRate ?? 0)) ** 2 },
      { name: 'lateralVelocity', sqError: (p, a) => ((p.lateralVelocity ?? 0) - (a.lateralVelocity ?? 0)) ** 2 },
    ],
    model: { make: () => learnedForwardSimV2(model) },
    baselines: {
      kinematic: { make: () => composedSim(kinematicForwardSim(agent), wheeledToLegacy) },
      legacyV1: { make: () => composedSim(learnedForwardSim(DEFAULT_LEARNED_PARAMS, agent), wheeledToLegacy) },
      parametricOnly: {
        make: () => parametricForwardV2(model.params, model.config),
      },
    },
  });
  // Per-split open-loop only — cheaper than full headline per split, and
  // it's the column users actually read in the Phase 0 train / val / test
  // RMS table.
  const evalOpenLoop = (trials: ReadonlyArray<typeof all[number]>): OpenLoopRow[] | undefined => {
    if (trials.length === 0) return undefined;
    return evaluateModel<CarKinematicState, WheeledCarControls, LearnableVehicleConfig>({
      trials, horizons, controlsToVec,
      extractMetricFields: (s) => ({ x: s.x, z: s.z, heading: s.heading, speed: s.speed }),
      model: { make: () => learnedForwardSimV2(model) },
    }).openLoopDivergence;
  };
  const perSplit = hasSplit
    ? {
        train: evalOpenLoop(trainSet),
        val: evalOpenLoop(valSet),
        test: evalOpenLoop(testSet),
      }
    : undefined;
  return { ...headline, perSplit };
}

// ---------------------------------------------------------------------------
// Active exploration: turn per-cell error stats from the diagnostics into
// next-round trial specs.

function buildExplorationCells(
  cellErrors: Map<string, { error: number; count: number }>,
  rng: () => number,
): ExplorationCell<CellSpec>[] {
  const out: ExplorationCell<CellSpec>[] = [];
  // Match the broadened seed grid: include race-relevant speeds 16-28
  // so active rounds also probe high-speed regimes.
  const speeds = [0, 4, 8, 12, 16, 20, 24, 28];
  const steerLevels = [-0.5, 0, 0.5];
  const driveMid = DEFAULT_VEHICLE_OPTS.engineForce * 0.5;
  for (let si = 0; si < speeds.length; si++) {
    for (let ti = 0; ti < steerLevels.length; ti++) {
      const id = `s${si}-t${ti}`;
      const stat = cellErrors.get(id) ?? { error: 0.5, count: 0 };
      const v = speeds[si]!;
      const stMag = DEFAULT_VEHICLE_OPTS.maxSteerAngle * steerLevels[ti]!;
      out.push({
        id, errorRms: stat.error, count: stat.count,
        sample: (r) => {
          const jitterV = (r() - 0.5) * 1.5;
          const jitterSt = (r() - 0.5) * 0.05;
          return {
            startSpeed: Math.max(0, v + jitterV),
            steer: stMag + jitterSt,
            driveForce: driveMid * (0.7 + r() * 0.3),
            brakeForce: 0,
          };
        },
      });
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Main driver

export interface RunOfflineTrainingOptions {
  rounds?: number;
  seedTrials?: CellSpec[];
  trialsPerActiveRound?: number;
  trialTicks?: number;
  sampleEveryNTicks?: number;
  seed?: number;
  onEvent?: (e: TrainingEvent) => void;
  vehicleOptions?: typeof DEFAULT_VEHICLE_OPTS;
  /** When true, the harness is RETURNED to the caller (not disposed)
   *  so downstream UI (scenario playground) can keep running Rapier
   *  trials against the same vehicle config. Caller MUST dispose. */
  keepHarness?: boolean;
}

export interface RunOfflineTrainingResult {
  model: LearnedVehicleModel;
  trials: TrialStore<CarKinematicState, WheeledCarControls, LearnableVehicleConfig>;
  finalDiagnostics: ModelDiagnostics;
  /** Only populated when `keepHarness: true`. Caller owns disposal. */
  harness?: HeadlessTrialHarness;
  /** The vehicle config the trials were collected against. */
  config: LearnableVehicleConfig;
  /** Sample interval in seconds = sampleEveryNTicks / 60. */
  sampleDt: number;
}

/** Per-parameter regularization scales used by the parametric fit.
 *  Hand-tuned to keep weakly-constrained coefficients near the default
 *  prior while still letting strong evidence move them. */
const REG_SCALES: LearnedVehicleParamsV2 = {
  engineScale: 0.10,
  reverseEffScale: 0.15,
  brakeScale: 0.30,
  accelTau: 0.10,
  gripScale: 0.15,
  frictionCircleSlack: 0.10,
  steerRatio: 0.15,
  understeerOffThrottle: 0.005,
  understeerPowerOn: 0.005,
  yawRateTau: 0.08,
  lateralDamping: 2.5,
  lateralFromSteer: 0.30,
  slipDrag: 0.30,
  loadTransferCoeff: 0.02,
  driveDeadzone: 80,
  rollingResistance: 0.05,
};

/** Concrete car-v2 implementation of the generic `TrainingPipeline`
 *  contract. The same class is consumed by:
 *
 *  - the demo's `runOfflineTraining` function (preserves the existing
 *    demo-shape event stream + return shape for backward compatibility),
 *  - the core's `runOfflineTraining` orchestrator (proves an alternative
 *    vehicle pipeline could plug in identically; covered by the
 *    equivalence test).
 *
 *  Carries its own Rapier harness, trial store config, and round-by-
 *  round mutable state (current model, last diagnostics for the
 *  active explorer's cell-error map). */
export class CarV2TrainingPipeline
  implements TrainingPipeline<CarKinematicState, WheeledCarControls, LearnedVehicleParamsV2, LearnableVehicleConfig>
{
  readonly name = 'car-v2';
  readonly harness: HeadlessTrialHarness;
  readonly config: LearnableVehicleConfig;
  readonly sampleDt: number;

  private readonly seedTrials: CellSpec[];
  private readonly trialsPerActive: number;
  private readonly ticks: number;
  private readonly sampleEveryN: number;
  private readonly rounds: number;
  private readonly seed: number;
  private readonly onParametricProgress?: (round: number, e: FitProgressEvent) => void;
  private readonly onResidualProgress?: (round: number, e: FitProgressEvent) => void;

  private model: LearnedVehicleModel;
  private lastDiag: ModelDiagnostics = { openLoopDivergence: [], perStateRms: [], coverage: [], baselines: {} };
  private trialIdx = 0;
  private rngState: number;

  private constructor(args: {
    harness: HeadlessTrialHarness;
    config: LearnableVehicleConfig;
    seedTrials: CellSpec[];
    trialsPerActive: number;
    ticks: number;
    sampleEveryN: number;
    rounds: number;
    seed: number;
    initialModel: LearnedVehicleModel;
    onParametricProgress?: (round: number, e: FitProgressEvent) => void;
    onResidualProgress?: (round: number, e: FitProgressEvent) => void;
  }) {
    this.harness = args.harness;
    this.config = args.config;
    this.seedTrials = args.seedTrials;
    this.trialsPerActive = args.trialsPerActive;
    this.ticks = args.ticks;
    this.sampleEveryN = args.sampleEveryN;
    this.rounds = args.rounds;
    this.seed = args.seed;
    this.sampleDt = args.sampleEveryN / 60;
    this.model = args.initialModel;
    this.rngState = args.seed | 0;
    this.onParametricProgress = args.onParametricProgress;
    this.onResidualProgress = args.onResidualProgress;
  }

  static async create(opts: {
    rounds?: number;
    trialsPerActiveRound?: number;
    trialTicks?: number;
    sampleEveryNTicks?: number;
    seed?: number;
    seedTrials?: CellSpec[];
    vehicleOptions?: typeof DEFAULT_VEHICLE_OPTS;
    onParametricProgress?: (round: number, e: FitProgressEvent) => void;
    onResidualProgress?: (round: number, e: FitProgressEvent) => void;
  } = {}): Promise<CarV2TrainingPipeline> {
    const veh = opts.vehicleOptions ?? DEFAULT_VEHICLE_OPTS;
    const harness = await createHeadlessTrialHarness({
      vehicleOptions: veh,
      groundBounds: { x0: -500, x1: 500, z0: -500, z1: 500 },
    });
    const config = deriveLearnableConfig({
      id: 'driver', position: { x: 0, z: 0 }, heading: 0, ...veh,
    });
    return new CarV2TrainingPipeline({
      harness,
      config,
      seedTrials: opts.seedTrials ?? [...buildSeedGrid(), ...extremeProbes()],
      trialsPerActive: opts.trialsPerActiveRound ?? 48,
      ticks: opts.trialTicks ?? 120,
      sampleEveryN: opts.sampleEveryNTicks ?? 6,
      rounds: opts.rounds ?? 3,
      seed: opts.seed ?? 42,
      initialModel: buildParametricOnlyModel(DEFAULT_LEARNED_PARAMS_V2, config),
      onParametricProgress: opts.onParametricProgress,
      onResidualProgress: opts.onResidualProgress,
    });
  }

  totalRounds(): number {
    return this.rounds;
  }

  /** Current learned model (parametric coefficients + optional residual
   *  ensemble + reference dt). Mutates round-by-round. */
  currentLearnedModel(): LearnedVehicleModel {
    return this.model;
  }

  dispose(): void {
    this.harness.dispose();
  }

  private rng = (): number => {
    this.rngState = (this.rngState + 0x6d2b79f5) | 0;
    let t = this.rngState;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  /** Round 0 → seed grid; rounds > 0 → active exploration sampled from
   *  the previous round's per-cell error map. */
  private nextCells(round: number): CellSpec[] {
    if (round === 0) return this.seedTrials;
    const cellErrors = new Map<string, { error: number; count: number }>();
    for (const c of this.lastDiag.coverage) {
      cellErrors.set(c.binId, { error: c.errorRms, count: c.count });
    }
    const explorationCells = buildExplorationCells(cellErrors, this.rng);
    const proposed = proposeNextBatch({
      cells: explorationCells, budget: this.trialsPerActive, seed: this.seed + round,
      alwaysInclude: extremeProbes(),
    });
    return proposed.map((p) => p.spec);
  }

  async collectTrials(
    ctx: TrainingContext<CarKinematicState, WheeledCarControls, LearnableVehicleConfig>,
  ): Promise<{ collected: Trial<CarKinematicState, WheeledCarControls, LearnableVehicleConfig>[]; discarded: number }> {
    const cells = this.nextCells(ctx.round);
    const result = await collectTrialBatch(this.harness, cells, this.ticks, this.sampleEveryN, this.trialIdx);
    this.trialIdx += cells.length;
    return result;
  }

  async fitParametric(
    ctx: TrainingContext<CarKinematicState, WheeledCarControls, LearnableVehicleConfig>,
    onProgress?: (e: FitProgressEvent) => void,
  ): Promise<LearnedVehicleParamsV2> {
    const fit = runParametricFit<LearnedVehicleParamsV2, CarKinematicState, WheeledCarControls, LearnableVehicleConfig>({
      init: this.model.params,
      encode: paramsV2ToVec,
      decode: paramsV2FromVec,
      makeSim: (p, cfg) => parametricForwardV2(p, cfg),
      stateDelta: stateDeltaForFit,
      trials: ctx.store.all(),
      controlsToVec,
      maxIter: ctx.round === 0 ? 200 : 120,
      onProgress: (e) => {
        onProgress?.(e);
        this.onParametricProgress?.(ctx.round, e);
      },
      // Regularize toward DEFAULT_LEARNED_PARAMS_V2 to keep weakly-
      // constrained coefficients from pinning to bounds.
      regularization: {
        strength: 0.05,
        priorVec: paramsV2ToVec(DEFAULT_LEARNED_PARAMS_V2),
        scales: paramsV2ToVec(REG_SCALES),
      },
    });
    this.model = { ...this.model, params: fit.params };
    return fit.params;
  }

  async fitResidual(
    ctx: TrainingContext<CarKinematicState, WheeledCarControls, LearnableVehicleConfig>,
    params: LearnedVehicleParamsV2,
    onProgress?: (e: FitProgressEvent) => void,
  ): Promise<TrainedModel<CarKinematicState, LearnedVehicleParamsV2>> {
    // Only fit residual on the final round, and only if we have enough
    // trials. The parametric needs to be at its final fit BEFORE residual
    // training — fitting residual against an intermediate parametric
    // bakes stale corrections in.
    const isFinalRound = ctx.round === this.rounds - 1;
    if (isFinalRound && ctx.store.size() >= 16) {
      const baselineSim = parametricForwardV2(params, this.config);
      const residualFit = runResidualMLPFit<CarKinematicState, WheeledCarControls, LearnableVehicleConfig>({
        trials: ctx.store.all(),
        makeBaselineSim: () => baselineSim,
        encodeInput: (s, ctrl, cfg) => buildMLPInput(s, ctrl, cfg),
        encodeResidual: (actual, baseline) => [
          actual.x - baseline.x,
          actual.z - baseline.z,
          wrapAngleResidual(actual.heading - baseline.heading),
          actual.speed - baseline.speed,
          (actual.yawRate ?? 0) - (baseline.yawRate ?? 0),
          (actual.lateralVelocity ?? 0) - (baseline.lateralVelocity ?? 0),
        ],
        controlsToVec,
        mlpShape: { inputDim: MLP_INPUT_DIM, hiddenDims: [32, 32], outputDim: MLP_OUTPUT_DIM },
        ensembleSize: 3,
        seed: 42,
        epochs: 200,
        batchSize: 64,
        learningRate: 1e-3,
        valSplit: 0.2,
        fitSubstepsPerSample: 6,
        onProgress: (e) => {
          const ev: FitProgressEvent = {
            iter: e.epoch,
            loss: e.trainLoss,
            valLoss: e.valLoss,
            perComponent: undefined,
          };
          onProgress?.(ev);
          this.onResidualProgress?.(ctx.round, ev);
        },
      });
      this.model = {
        ...this.model,
        params,
        residualEnsemble: residualFit.ensemble,
        residualReferenceDt: this.sampleEveryN / 60,
      };
    } else {
      this.model = { ...this.model, params };
    }
    return {
      params: this.model.params,
      forwardSim: learnedForwardSimV2(this.model),
    };
  }

  evaluate(
    ctx: TrainingContext<CarKinematicState, WheeledCarControls, LearnableVehicleConfig>,
    _trained: TrainedModel<CarKinematicState, LearnedVehicleParamsV2>,
  ): ModelDiagnostics {
    let diag = evaluate(ctx.store, this.model);
    diag = withCellBinning(diag, ctx.store);
    this.lastDiag = diag;
    return diag;
  }
}

export async function runOfflineTraining(
  opts: RunOfflineTrainingOptions = {},
): Promise<RunOfflineTrainingResult> {
  const pipeline = await CarV2TrainingPipeline.create({
    rounds: opts.rounds,
    trialsPerActiveRound: opts.trialsPerActiveRound,
    trialTicks: opts.trialTicks,
    sampleEveryNTicks: opts.sampleEveryNTicks,
    seed: opts.seed,
    seedTrials: opts.seedTrials,
    vehicleOptions: opts.vehicleOptions,
  });
  const store = createTrialStore<CarKinematicState, WheeledCarControls, LearnableVehicleConfig>();
  let finalDiag: ModelDiagnostics = { openLoopDivergence: [], perStateRms: [], coverage: [], baselines: {} };
  for (let round = 0; round < pipeline.totalRounds(); round++) {
    opts.onEvent?.({ type: 'round-start', round, trialsBeforeRound: store.size() });
    const ctx: TrainingContext<CarKinematicState, WheeledCarControls, LearnableVehicleConfig> = { round, store };
    const { collected, discarded } = await pipeline.collectTrials(ctx);
    for (const t of collected) store.add(t);
    opts.onEvent?.({ type: 'trial-batch', round, collected: collected.length, discarded });
    const params = await pipeline.fitParametric(ctx, (event) =>
      opts.onEvent?.({ type: 'fit-progress', round, phase: 'parametric', event }),
    );
    const trained = await pipeline.fitResidual(ctx, params, (event) =>
      opts.onEvent?.({ type: 'fit-progress', round, phase: 'residual', event }),
    );
    finalDiag = pipeline.evaluate(ctx, trained);
    opts.onEvent?.({ type: 'evaluation', round, diagnostics: finalDiag });
    opts.onEvent?.({
      type: 'round-end',
      round,
      trainedModel: pipeline.currentLearnedModel(),
      params: pipeline.currentLearnedModel().params,
      diagnostics: finalDiag,
      trialsAfter: store.size(),
    });
  }
  const finalModel = pipeline.currentLearnedModel();
  if (!opts.keepHarness) pipeline.dispose();
  opts.onEvent?.({ type: 'done', totalTrials: store.size(), finalModel, finalDiagnostics: finalDiag });
  return {
    model: finalModel,
    trials: store,
    finalDiagnostics: finalDiag,
    harness: opts.keepHarness ? pipeline.harness : undefined,
    config: pipeline.config,
    sampleDt: pipeline.sampleDt,
  };
}

function wrapAngleResidual(a: number): number {
  let d = a;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return d;
}

/** Recompute coverage with `(speedBin, steerBin)` cell keys so the active
 *  explorer can index into them next round. The generic evaluateModel
 *  doesn't know about our binning scheme; we attach it here. */
function withCellBinning(
  diag: ModelDiagnostics,
  store: TrialStore<CarKinematicState, WheeledCarControls, LearnableVehicleConfig>,
): ModelDiagnostics {
  // Rebuild a coverage list keyed by (speed, steer) bin from the held-out
  // trials' initial state + first control.
  const map = new Map<string, { sq: number; n: number }>();
  const all = store.all();
  const cut = Math.max(1, Math.floor(all.length * 0.75));
  const heldOut = all.slice(cut);
  for (const tr of heldOut) {
    const cell: CellSpec = {
      startSpeed: tr.initialState.speed,
      steer: tr.controlsTrace[0]?.steer ?? 0,
      driveForce: tr.controlsTrace[0]?.driveForce ?? 0,
      brakeForce: tr.controlsTrace[0]?.brakeForce ?? 0,
    };
    const key = cellKeyOf(cell);
    const last = tr.samples[tr.samples.length - 1]!;
    // approximate error from open-loop divergence at last sample (cheap: use
    // diag's perStateRms.pos if available)
    const rmsApprox = diag.openLoopDivergence.find((r) => r.tSec >= 1.0)?.posRms ?? 0.5;
    const cur = map.get(key) ?? { sq: 0, n: 0 };
    cur.sq += rmsApprox * rmsApprox;
    cur.n += 1;
    map.set(key, cur);
  }
  const coverage = [...map.entries()].map(([binId, v]) => ({
    binId,
    count: v.n,
    errorRms: Math.sqrt(v.sq / v.n),
  }));
  return { ...diag, coverage };
}

// Re-export for the demo UI.
export { PARAMS_V2_ORDER };

/** Build a standalone Rapier harness for the Model Lab's scenario
 *  playground. Mirrors the vehicle config used by `runOfflineTraining`
 *  so trajectories are directly comparable. Caller owns disposal. */
export async function createScenarioHarness(): Promise<{
  harness: HeadlessTrialHarness;
  config: LearnableVehicleConfig;
}> {
  const harness = await createHeadlessTrialHarness({
    vehicleOptions: DEFAULT_VEHICLE_OPTS,
    groundBounds: { x0: -500, x1: 500, z0: -500, z1: 500 },
  });
  const config = deriveLearnableConfig({
    id: 'driver', position: { x: 0, z: 0 }, heading: 0, ...DEFAULT_VEHICLE_OPTS,
  });
  return { harness, config };
}

/** Roll a forward sim through a controls trace and return the predicted
 *  state at every recorded sample boundary. Useful for the RolloutPlayer
 *  to show the model's open-loop prediction next to the Rapier truth. */
export function rolloutForwardSim(
  sim: import('kinocat/primitives').ForwardSim<CarKinematicState>,
  initialState: CarKinematicState,
  controlsTrace: ReadonlyArray<WheeledCarControls>,
  dt: number,
  sampleEveryNTicks: number,
): { states: CarKinematicState[]; times: number[] } {
  const states: CarKinematicState[] = [{ ...initialState }];
  const times: number[] = [0];
  let s: CarKinematicState = { ...initialState };
  for (let i = 0; i < controlsTrace.length; i++) {
    const c = controlsTrace[i]!;
    s = sim(s, [c.steer, c.driveForce, c.brakeForce], dt);
    if ((i + 1) % sampleEveryNTicks === 0) {
      states.push({ ...s });
      times.push((i + 1) * dt);
    }
  }
  return { states, times };
}

// ===========================================================================
// Maneuver-based trial collection (Phase 1 of the training-dataset plan).
//
// Drop-in replacement for `collectTrialBatch`: takes a `ManeuverSpec[]`
// (each producing a time-varying controls trace) and returns trials whose
// `controlsTrace` came from the maneuver factories — OU random walks,
// transition probes, panic / identification maneuvers, …
//
// Each emitted trial is tagged with `maneuverId` + `maneuverParams` so the
// hash-based split policy + coverage meter index them consistently. The
// initial state for every maneuver is the zero-velocity origin, matching
// the existing harness; Phase 2's `state-conditioner` work will diversify
// initial conditions on top of this.

export interface ManeuverTrialOptions {
  ticks: number;
  sampleEveryNTicks: number;
  /** Initial forward speed (m/s). Defaults to 0. */
  startSpeed?: number;
}

function carManeuverLimits(): ManeuverLimits {
  return {
    maxSteerAngle: DEFAULT_VEHICLE_OPTS.maxSteerAngle,
    maxDriveForce: DEFAULT_VEHICLE_OPTS.engineForce,
    maxBrakeForce: DEFAULT_VEHICLE_OPTS.brakeForce,
  };
}

/** Build the default maneuver bundle sized for one training round. */
export function buildDefaultManeuverBundle(args: {
  count: number;
  seed?: number;
}): ManeuverSpec[] {
  return defaultManeuverBundle({
    limits: carManeuverLimits(),
    count: args.count,
    seed: args.seed,
  });
}

/** Collect a batch of trials by running each spec's driver through the
 *  headless trial harness (no real body needed — drivers in the default
 *  bundle are state-independent so the trace pre-rolls correctly). */
export async function collectManeuverBatch(
  harness: HeadlessTrialHarness,
  specs: ManeuverSpec[],
  opts: ManeuverTrialOptions,
  startId: number,
  startSpeedSchedule?: number[],
): Promise<{
  collected: Trial<CarKinematicState, WheeledCarControls, LearnableVehicleConfig>[];
  discarded: number;
}> {
  const collected: Trial<CarKinematicState, WheeledCarControls, LearnableVehicleConfig>[] = [];
  let discarded = 0;
  let id = startId;
  const limits = carManeuverLimits();
  const dt = 1 / 60;
  const speeds = startSpeedSchedule && startSpeedSchedule.length > 0
    ? startSpeedSchedule
    : [opts.startSpeed ?? 0];
  for (const spec of specs) {
    const driver = spec.build(limits, 0);
    const trace = buildControlsTrace(driver, {
      state: { x: 0, z: 0, heading: 0, speed: 0, t: 0 } as CarKinematicState,
      dt,
      steps: opts.ticks,
    });
    // Rotate through the start-speed schedule so the maneuvers are
    // exercised across the full speed envelope.
    const startSpeed = speeds[id % speeds.length]!;
    const trialSpec: TrialSpec = {
      pose: { x: 0, z: 0, heading: 0 },
      kin: { forwardSpeed: startSpeed },
      controlsTrace: trace,
      sampleEveryNTicks: opts.sampleEveryNTicks,
      id: `m-${id}`,
    };
    id++;
    const result = harness.runTrial(trialSpec);
    if (!result.ok) { discarded++; continue; }
    const t = result.trial;
    const trial: Trial<CarKinematicState, WheeledCarControls, LearnableVehicleConfig> = {
      id: t.id,
      initialState: t.samples[0]!,
      controlsTrace: trialSpec.controlsTrace,
      dt: t.dt,
      samples: t.samples.map((s, i) => ({ t: i * opts.sampleEveryNTicks * t.dt, state: s })),
      config: t.config,
      configKey: 'rwd-default',
      maneuverId: spec.id,
      maneuverParams: { ...spec.params, startSpeed },
    };
    trial.split = assignSplit(trial);
    collected.push(trial);
  }
  return { collected, discarded };
}

// ---------------------------------------------------------------------------
// Maneuver-based training driver — entry point used by `pnpm run train`.

export interface ManeuverTrainingOptions {
  /** Total trials per round. Default 200. */
  trialsPerRound?: number;
  rounds?: number;
  trialTicks?: number;
  sampleEveryNTicks?: number;
  seed?: number;
  vehicleOptions?: typeof DEFAULT_VEHICLE_OPTS;
  /** Forward-speed schedule cycled across maneuvers. Default
   *  `[0, 4, 8, 12, 16, 20, 24, 28]`. */
  startSpeedSchedule?: number[];
  onEvent?: (e: TrainingEvent) => void;
  /** Phase 3 DAgger mode: starting at round `daggerStartRound`, race the
   *  currently-trained v2 model against the track for `daggerLapsPerRound`
   *  laps and mix the collected closed-loop trials into the next round's
   *  training set. Disabled by default. */
  daggerStartRound?: number;
  daggerLapsPerRound?: number;
  daggerMaxSimTime?: number;
  daggerWindowSec?: number;
}

const DEFAULT_SPEED_SCHEDULE = [0, 4, 8, 12, 16, 20, 24, 28];

/** Orchestrate maneuver-based offline training. Same pipeline shape as
 *  `runOfflineTraining` but the trial sourcing is the Phase 1 maneuver
 *  library instead of the constant-hold grid. */
export async function runManeuverTraining(
  opts: ManeuverTrainingOptions = {},
): Promise<RunOfflineTrainingResult> {
  const trialsPerRound = opts.trialsPerRound ?? 200;
  const rounds = opts.rounds ?? 3;
  const ticks = opts.trialTicks ?? 120;
  const sampleEveryN = opts.sampleEveryNTicks ?? 6;
  const seed = opts.seed ?? 42;
  const veh = opts.vehicleOptions ?? DEFAULT_VEHICLE_OPTS;
  const startSpeedSchedule = opts.startSpeedSchedule ?? DEFAULT_SPEED_SCHEDULE;

  // Reuse the same pipeline class as `runOfflineTraining` so the fit /
  // evaluation logic is shared, but override trial collection with the
  // maneuver-based collector.
  const pipeline = await CarV2TrainingPipeline.create({
    rounds, trialsPerActiveRound: trialsPerRound, trialTicks: ticks,
    sampleEveryNTicks: sampleEveryN, seed, vehicleOptions: veh,
  });
  // Replace cell-based trial sourcing with maneuver-based on every round.
  // (Round 0 in the default pipeline uses the seed grid; rounds 1+ use
  // active exploration. We replace BOTH with fresh maneuver bundles so
  // the dataset is uniformly maneuver-sourced.)
  const harness = pipeline.harness;
  const store = createTrialStore<CarKinematicState, WheeledCarControls, LearnableVehicleConfig>();
  let finalDiag: ModelDiagnostics = { openLoopDivergence: [], perStateRms: [], coverage: [], baselines: {} };
  let trialIdx = 0;
  const daggerStartRound = opts.daggerStartRound ?? Infinity;
  const daggerLapsPerRound = opts.daggerLapsPerRound ?? 2;
  const daggerMaxSimTime = opts.daggerMaxSimTime ?? 120;
  const daggerWindowSec = opts.daggerWindowSec ?? 1.0;

  for (let round = 0; round < rounds; round++) {
    opts.onEvent?.({ type: 'round-start', round, trialsBeforeRound: store.size() });
    const bundle = buildDefaultManeuverBundle({
      count: trialsPerRound,
      seed: seed + round * 17,
    });
    const { collected, discarded } = await collectManeuverBatch(
      harness,
      bundle,
      { ticks, sampleEveryNTicks: sampleEveryN },
      trialIdx,
      startSpeedSchedule,
    );
    trialIdx += bundle.length;
    for (const t of collected) store.add(t);
    opts.onEvent?.({ type: 'trial-batch', round, collected: collected.length, discarded });
    // Phase 3 DAgger: race the current v2 model on the actual track and
    // append the recorded (state, controls, next_state) trials. Lifted
    // behind a feature flag so the existing maneuver-only pipeline
    // remains the default — DAgger needs a halfway-decent starting model.
    if (round >= daggerStartRound) {
      const { collectFromRaceScenario } = await import('./race-scenario-collect');
      const { buildLearnedRaceLibraryV2 } = await import('./race-primitives-scenarios');
      const lib = buildLearnedRaceLibraryV2(pipeline.currentLearnedModel());
      const race = await collectFromRaceScenario({
        lib,
        targetLaps: daggerLapsPerRound,
        maxSimTime: daggerMaxSimTime,
        windowSec: daggerWindowSec,
        sampleEveryNTicks: sampleEveryN,
        scenarioId: `dagger-round${round}`,
      });
      for (const t of race.trials) store.add(t);
      opts.onEvent?.({
        type: 'trial-batch',
        round,
        collected: race.trials.length,
        discarded: 0,
      });
    }
    const ctx = { round, store };
    const params = await pipeline.fitParametric(ctx, (event) =>
      opts.onEvent?.({ type: 'fit-progress', round, phase: 'parametric', event }),
    );
    await pipeline.fitResidual(ctx, params, (event) =>
      opts.onEvent?.({ type: 'fit-progress', round, phase: 'residual', event }),
    );
    const trained = { params, forwardSim: learnedForwardSimV2(pipeline.currentLearnedModel()) };
    finalDiag = pipeline.evaluate(ctx, trained);
    opts.onEvent?.({ type: 'evaluation', round, diagnostics: finalDiag });
    opts.onEvent?.({ type: 'coverage', round, cells: buildCoverageSummary(store) });
    opts.onEvent?.({
      type: 'round-end',
      round,
      trainedModel: pipeline.currentLearnedModel(),
      params: pipeline.currentLearnedModel().params,
      diagnostics: finalDiag,
      trialsAfter: store.size(),
    });
  }
  const finalModel = pipeline.currentLearnedModel();
  pipeline.dispose();
  opts.onEvent?.({ type: 'done', totalTrials: store.size(), finalModel, finalDiagnostics: finalDiag });
  return {
    model: finalModel,
    trials: store,
    finalDiagnostics: finalDiag,
    config: pipeline.config,
    sampleDt: pipeline.sampleDt,
  };
}
