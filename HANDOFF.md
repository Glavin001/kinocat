# MPC Tracker & Race Scenario Handoff

## What was done

### Problem
The v2 trained model (`demos/public/models/v2-default.json`) produces excellent trajectory
curves during training, but at runtime the car overshoots turns, shows red dots (sharp
planned curvature), replans with progressively worse paths. Root cause: planning-execution
dynamics mismatch -- the planner builds primitives using `learnedForwardSimV2(model)` with
trained dynamics (steerRatio=1.31, yawRateTau=0.098, etc.), but the pure-pursuit tracker
follows paths using kinematic bicycle-model assumptions.

### Changes made (branch: claude/serene-shannon-Z0FUr)

1. **`core/src/execute/mpc-tracker.ts`**
   - Added `minReferenceSpeed` config to prevent MPC braking at waypoints
   - Fixed `buildReference` to use static floor only (removed `currentSpeed` floor that caused identical trajectories for all cars)
   - Added drive/brake mutual exclusion after MPPI averaging (MPPI blends accelerate+brake samples, causing the car to fight itself with simultaneous throttle+brake)

2. **`demos/app/lib/race-scenario.ts`**
   - Added per-car MPC forward sim (`mpcForwardSimFor`) so each car's MPC uses its own dynamics model
   - Added `model?: LearnedVehicleModel` to `RaceEntry` interface
   - Added `tracker: 'pure-pursuit' | 'mpc'` toggle in `RaceTuning`
   - Tuned MPC weights for racing (original was parking-oriented: wSpeed=10, wLateral=2)
   - `DEFAULT_TUNING.tracker` remains `'pure-pursuit'` (MPC not yet competitive)

3. **`demos/app/lib/headless-race.ts`**
   - `v2Entry()` and `parametricOnlyEntry()` now pass `model` on the `RaceEntry`
   - Progress output includes peak speed, throttle%, brake% for diagnostics

4. **`demos/scripts/race.ts`**
   - Added `--tracker=pure-pursuit|mpc` CLI flag (default: pure-pursuit)

5. **`demos/app/raceprimitives/RacePrimitives.tsx`**
   - Passes `learnedModel` through to `createRaceScenario` entry
   - Aligned options with CLI: `syncHold: false`, `offTrackRecovery: 'spawn'`

### IMPORTANT: core/ requires rebuild
Changes to `core/src/` are NOT picked up until `pnpm build` runs in `core/`.
Both CLI and web resolve `kinocat/*` to `core/dist/` (compiled JS), not source TS.
This caused hours of debugging where edits appeared to have no effect.

## Known regression: kinematic DNF

**Before**: kinematic finished 3/3 laps with pure-pursuit (avg ~45s)
**After**: kinematic DNFs at 2/3 laps (avg ~61s)

The pure-pursuit code path was NOT modified -- this regression likely comes from the
`core/dist/` rebuild picking up previously-uncommitted source changes in core/. The
kinematic car doesn't use any model or MPC code, so the regression source is unclear.
Needs investigation.

## MPC tracker status: NOT ready for default use

MPC was explored as a solution but is not yet competitive with pure-pursuit for racing:

- **Drive/brake conflict**: MPPI's importance-weighted averaging blends opposing control
  samples (some want throttle, some want brake), producing both simultaneously. The
  mutual-exclusion fix resolves this but makes control less smooth.
- **Forward sim mismatch**: The parametric backbone doesn't match Rapier well enough for
  accurate multi-step rollouts. Using `learnedForwardSimV2` (with MLPs) was worse because
  MLP errors compound over the 10-step horizon.
- **Sample count**: 64-128 samples is sparse for 3D control space. Production MPPI uses
  1000-10000 samples, not feasible at 60fps in browser.

Best results with MPC: v2 beats kinematic by 6-20% but both cars are ~2x slower than
pure-pursuit.

## What needs to happen next

1. **Fix kinematic regression**: Investigate why kinematic DNFs with pure-pursuit after
   core rebuild. Compare `core/dist/` before/after. May need to `git stash` core changes
   and rebuild to isolate.

2. **Improve pure-pursuit for v2**: The steerRatio correction approach (dividing steer
   by model.params.steerRatio) was tried and made v2 much worse (103s vs 40s). The
   relationship between steerRatio and Rapier's actual response needs more analysis.
   Consider: is steerRatio compensating for wheelbase or tire model differences?

3. **MPC improvements if revisited**:
   - Try single-axis longitudinal control (positive=drive, negative=brake) instead of
     independent drive+brake sampling to eliminate the conflict at the source
   - Try shorter stepDt matching physics dt (1/60) with proportionally longer horizon
   - Consider hybrid: MPC for steering only, pure-pursuit PID for speed

4. **Web demo syncHold**: Changed from `true` to `false` to match CLI. If visual
   side-by-side comparison suffers, consider adding a UI toggle.

## Key file map

- `core/src/execute/mpc-tracker.ts` -- MPPI sampling MPC
- `demos/app/lib/race-scenario.ts` -- single source of truth for simulation (used by both CLI and web)
- `demos/app/lib/headless-race.ts` -- CLI race runner + entry builders
- `demos/scripts/race.ts` -- CLI entry point
- `demos/app/raceprimitives/RacePrimitives.tsx` -- web demo (React)
- `demos/public/models/v2-default.json` -- trained v2 model (steerRatio=1.31, 3600 trials)
- `core/src/agent/vehicle-model.ts` -- v2 parametric backbone + MLP ensemble

## Race results (current, pure-pursuit, seed=42, 3 laps)

```
v2-default    OK  3/3  40.08s  57.75s  72.87s  best=40.08s  avg=56.90s
kinematic     DNF 2/3  35.02s  87.30s  ---     best=35.02s  avg=61.16s
v2-default beats kinematic by 7.0% (avg)
```
