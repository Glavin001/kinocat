// Demo-side orchestration glue: combines the headless Rapier trial harness
// (kinocat/adapters/rapier) with the generic learning helpers
// (kinocat/learning) to train the v2 vehicle model offline. Streams
// progress events so the Model Lab UI (RacePrimitives.tsx) can render live
// loss / coverage diagnostics.
//
// This is the use-case-specific consumer that wires the agnostic core
// pieces together for the /raceprimitives demo.

import type { WheeledControls } from 'kinocat/agent';
import type { VehicleState, LearnableVehicleConfig, LearnedVehicleParamsV2 } from 'kinocat/agent';
import {
  DEFAULT_LEARNED_PARAMS_V2,
  parametricForwardV2,
  paramsV2ToVec,
  paramsV2FromVec,
  PARAMS_V2_ORDER,
  buildParametricOnlyModel,
  type LearnedVehicleModel,
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
  evaluateModel,
  type ModelDiagnostics,
  type FitProgressEvent,
  proposeNextBatch,
  type ExplorationCell,
} from 'kinocat/learning';
import {
  createHeadlessTrialHarness,
  deriveLearnableConfig,
  type TrialSpec,
  type HeadlessTrialHarness,
} from 'kinocat/adapters/rapier';
import type { ForwardSim } from 'kinocat/primitives';

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
  | { type: 'fit-progress'; round: number; phase: 'parametric'; event: FitProgressEvent }
  | { type: 'evaluation'; round: number; diagnostics: ModelDiagnostics }
  | { type: 'round-end'; round: number; trainedModel: LearnedVehicleModel }
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
): Promise<{ collected: Trial<VehicleState, WheeledControls, LearnableVehicleConfig>[]; discarded: number }> {
  const collected: Trial<VehicleState, WheeledControls, LearnableVehicleConfig>[] = [];
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

function stateDeltaForFit(pred: VehicleState, act: VehicleState): number {
  const dx = pred.x - act.x;
  const dz = pred.z - act.z;
  let dh = pred.heading - act.heading;
  while (dh > Math.PI) dh -= 2 * Math.PI;
  while (dh < -Math.PI) dh += 2 * Math.PI;
  const ds = pred.speed - act.speed;
  return dx * dx + dz * dz + 5 * dh * dh + ds * ds;
}

function controlsToVec(c: WheeledControls): number[] {
  return [c.steer, c.driveForce, c.brakeForce];
}

function evaluate(
  store: TrialStore<VehicleState, WheeledControls, LearnableVehicleConfig>,
  model: LearnedVehicleModel,
): ModelDiagnostics {
  // Last 25% of trials are held-out evaluation set.
  const all = store.all();
  const cut = Math.max(1, Math.floor(all.length * 0.75));
  const heldOut = all.slice(cut);
  if (heldOut.length === 0) {
    return { openLoopDivergence: [], perStateRms: [], coverage: [], baselines: {} };
  }
  const horizons = [0.5, 1.0, 1.6];
  const agent = defaultVehicleAgent();
  const wheeledToLegacy = (c: WheeledControls): number[] => {
    const k = Math.sin(c.steer) / (2 * DEFAULT_VEHICLE_OPTS.wheelBase);
    const targetSpeed = c.driveForce > 0 ? 10 : (c.brakeForce > 0 ? 0 : 5);
    return [k, targetSpeed];
  };
  const composedSim = (
    inner: ForwardSim<VehicleState>,
    encode: (c: WheeledControls) => number[],
  ): ForwardSim<VehicleState> => (s, controls, dt) => inner(s, encode({
    steer: controls[0] ?? 0, driveForce: controls[1] ?? 0, brakeForce: controls[2] ?? 0,
  }), dt);

  return evaluateModel<VehicleState, WheeledControls, LearnableVehicleConfig>({
    trials: heldOut,
    horizons,
    controlsToVec,
    extractMetricFields: (s) => ({ x: s.x, z: s.z, heading: s.heading, speed: s.speed }),
    model: { make: (cfg) => parametricForwardV2(model.params, cfg) },
    baselines: {
      kinematic: { make: () => composedSim(kinematicForwardSim(agent), wheeledToLegacy) },
      legacyV1: { make: () => composedSim(learnedForwardSim(DEFAULT_LEARNED_PARAMS, agent), wheeledToLegacy) },
    },
  });
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
}

export async function runOfflineTraining(
  opts: RunOfflineTrainingOptions = {},
): Promise<{ model: LearnedVehicleModel; trials: TrialStore<VehicleState, WheeledControls, LearnableVehicleConfig>; finalDiagnostics: ModelDiagnostics }> {
  const rounds = opts.rounds ?? 3;
  const trialsPerActive = opts.trialsPerActiveRound ?? 48;
  const ticks = opts.trialTicks ?? 120;
  const sampleEveryN = opts.sampleEveryNTicks ?? 6;
  const veh = opts.vehicleOptions ?? DEFAULT_VEHICLE_OPTS;
  const seedTrials = opts.seedTrials ?? [...buildSeedGrid(), ...extremeProbes()];
  const seed = opts.seed ?? 42;

  const harness = await createHeadlessTrialHarness({
    vehicleOptions: veh,
    groundBounds: { x0: -500, x1: 500, z0: -500, z1: 500 },
  });
  const config = deriveLearnableConfig({
    id: 'driver', position: { x: 0, z: 0 }, heading: 0, ...veh,
  });
  const store = createTrialStore<VehicleState, WheeledControls, LearnableVehicleConfig>();
  let model = buildParametricOnlyModel(DEFAULT_LEARNED_PARAMS_V2, config);
  let trialIdx = 0;

  // Seedable RNG for active exploration jitter.
  let rngState = seed | 0;
  const rng = (): number => {
    rngState = (rngState + 0x6d2b79f5) | 0;
    let t = rngState;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  let finalDiag: ModelDiagnostics = { openLoopDivergence: [], perStateRms: [], coverage: [], baselines: {} };
  for (let round = 0; round < rounds; round++) {
    opts.onEvent?.({ type: 'round-start', round, trialsBeforeRound: store.size() });
    // Round 0 uses seed grid; subsequent rounds use active exploration.
    let cells: CellSpec[];
    if (round === 0) {
      cells = seedTrials;
    } else {
      // Build cell-error map from coverage diagnostics of previous evaluation.
      const cellErrors = new Map<string, { error: number; count: number }>();
      for (const c of finalDiag.coverage) {
        cellErrors.set(c.binId, { error: c.errorRms, count: c.count });
      }
      const explorationCells = buildExplorationCells(cellErrors, rng);
      const proposed = proposeNextBatch({
        cells: explorationCells, budget: trialsPerActive, seed: seed + round,
        alwaysInclude: extremeProbes(),
      });
      cells = proposed.map((p) => p.spec);
    }
    const { collected, discarded } = await collectTrialBatch(harness, cells, ticks, sampleEveryN, trialIdx);
    trialIdx += cells.length;
    for (const t of collected) store.add(t);
    opts.onEvent?.({ type: 'trial-batch', round, collected: collected.length, discarded });

    // Fit on accumulated trials.
    const fit = runParametricFit<LearnedVehicleParamsV2, VehicleState, WheeledControls, LearnableVehicleConfig>({
      init: model.params,
      encode: paramsV2ToVec,
      decode: paramsV2FromVec,
      makeSim: (p, cfg) => parametricForwardV2(p, cfg),
      stateDelta: stateDeltaForFit,
      trials: store.all(),
      controlsToVec,
      maxIter: round === 0 ? 200 : 120,
      onProgress: (e) => opts.onEvent?.({ type: 'fit-progress', round, phase: 'parametric', event: e }),
    });
    model = { ...model, params: fit.params };

    // Evaluate.
    finalDiag = evaluate(store, model);
    // Populate coverage with bin keys we'll use for active exploration in the next round.
    finalDiag = withCellBinning(finalDiag, store);
    opts.onEvent?.({ type: 'evaluation', round, diagnostics: finalDiag });
    opts.onEvent?.({ type: 'round-end', round, trainedModel: model });
  }
  harness.dispose();
  opts.onEvent?.({ type: 'done', totalTrials: store.size(), finalModel: model, finalDiagnostics: finalDiag });
  return { model, trials: store, finalDiagnostics: finalDiag };
}

/** Recompute coverage with `(speedBin, steerBin)` cell keys so the active
 *  explorer can index into them next round. The generic evaluateModel
 *  doesn't know about our binning scheme; we attach it here. */
function withCellBinning(
  diag: ModelDiagnostics,
  store: TrialStore<VehicleState, WheeledControls, LearnableVehicleConfig>,
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
