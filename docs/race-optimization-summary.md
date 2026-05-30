# Race Optimization — Engineering Summary

## Goal

Optimize the v2 learned model for racing on the `/raceprimitives`
course so it beats the kinematic baseline, while keeping the same
controller competent at parking. Reduce the visible "red dots" (sharp
steering events) and the planner-thrash that the user observed.

## Outcome

| Metric (best post-fix race vs baseline) | Before | After |
|---|---|---|
| v2 avg lap | 47.9 s | **42.5 s** (best run; 45 s typical) |
| v2 lap 3 (warm) | 41 s | **36–38 s** |
| v2 sharp-steer ticks | 2638 | **1717–1981** |
| v2 lap-completion (3/3) | sometimes DNF | always 3/3 |
| Planner deadline hits | 270 | 145 |
| Controller-bench pass rate | 4/4 | 4/4 (race + 3 parking) |

`v2` is now consistently faster on its WARM laps (lap 2/3) than the
baseline; the run-to-run average is dragged by the cold-start lap 1
(chassis from rest through the slalom) and by inherent
wall-clock-bound planner deadline variance.

`v2 vs kinematic`: kinematic still wins on a single 3-lap race by
≈20 %. The remaining gap is the planner-execution dynamics
mismatch the handoff first flagged — v2's trained `steerRatio=1.31`
doesn't match Rapier's actual response, so the planner asks the
controller to take arcs the chassis doesn't quite execute, the
chassis drifts off the plan, the planner replans. Closing this
gap requires retraining v2 against Rapier-generated data (offline)
or a fundamentally different controller — both out of scope for
this iteration. Both empirical attempts in the handoff
(`steer/steerRatio` correction; the MPPI tracker) made things
worse, which we confirmed.

## What was fixed (engineering)

### Pure-pursuit (`core/src/execute/pure-pursuit.ts`)

1. **Unified brake-to-target.** Brake-to-goal targets the plan's
   terminal speed instead of zero: `vGoal = sqrt(v_term² + 2·a·d)`.
   Collapses to the classic brake-to-stop when the plan asks for a
   stop (parking, v_term ≈ 0); stays high when the plan asks for
   drive-through (racing, v_term ≈ cruise).
2. **`atGoal` brake gated on terminal speed.** The full-brake-on-
   arrival behaviour fires only when the plan's terminal speed is
   stop-intent; drive-through plans pass through their endpoint at
   the planner's intended speed.
3. **vPath brake-distance backward sweep.** When `respectPathSpeed`
   is on, the path-speed cap is the brake-distance-aware speed at
   each forward sample (not the raw min). Old behaviour stall-trapped
   the chassis on any zero-speed plan sample 14 m ahead.
4. **`minPathSpeed` floor.** Lets racing scenarios ignore the
   planner's slow-start primitives in the brake-distance pass
   without affecting parking's stop intents.
5. **`lookaheadCurvature` (opt-in).** Brake-distance backward sweep
   over the polyline's per-sample curvature, used for the racing
   `vCurve` cap (off by default — the on-line min-sweep proved too
   sensitive to smoother artifacts; kept as an opt-in for future
   experimentation).

### Race scenario (`demos/app/lib/race-scenario.ts`)

1. **Racing terminal-speed override.** If the planner picked a
   low-speed ending primitive at a race gate (`waypoints.length > 1`,
   raw terminal speed < 0.5·cruise), the segment fed to the controller
   is rewritten with terminal speed = cruise. Otherwise the chassis
   brakes to a halt at the gate and re-accelerates, costing 2–10 s
   per lap.
2. **Cusp-stop signalling.** Non-last segments of a multi-segment
   plan (gear-cusp boundaries) have their terminal speed overridden
   to 0 so the chassis comes to a full stop before flipping
   direction. Without this, multi-cusp parallel-park back-and-forths
   degrade into a curved coast.
3. **No teleport in racing.** Removed the stall-guard's
   teleport-to-current-waypoint behaviour for racing scenarios. The
   old behaviour was producing fake gate clearings (stalled controller
   → chassis warped to gate → loopIndex advanced → next gate
   stalls too → "lap" of 11 teleports completed in 22 s). Racing
   chassis now stay stuck where they get stuck; parking keeps the
   rescue because the controller occasionally fails to engage
   reverse after a forward-segment cusp.
4. **Per-trigger replan reasons.** Every replan records its reason
   (`cadence` | `lateral-error` | `waypoint-advance` | `failure-retry`
   | `manual`) on the snapshot. The breakdown answers "what's actually
   triggering the planner?" without staring at the raw history.
5. **Cumulative diagnostics.** Replan reason counts, planner-ms
   totals, deadline-hit counts, sharp-steer-tick counts all
   accumulated for the entire run (the 30-snapshot ring buffer
   covered only the last few seconds).
6. **Heavier path-consistency cost** (`consistencyWeight 0.08 →
   0.2`). The planner now prefers paths close to the
   previously-committed one — a fresh plan only wins when
   meaningfully better. Cut planner deadline hits ~50 % and
   reduced the chassis's tendency to oscillate between two
   marginal racing lines.

## Tooling

### `pnpm run race -- --debug-dir=.race-debug`

Writes a timestamped sub-directory at the end of every run with:
- `summary.json` — high-level result + diagnostics per car
- `replan-history.json` — last 30 per-replan snapshots
- `traces.json` — per-tick (0.1 s stride) chassis state + control
  output + lateral error to plan

Multiple runs append (don't overwrite); each run gets its own
timestamped directory.

### `pnpm exec tsx scripts/analyze-race-debug.ts <run-dir>`

Reads a debug bundle (or the newest sub-directory if given the
root) and prints per-car:
- speed / lateral-error / sharp-steer percentile stats
- the 5 worst 2-s windows for lateral error and sharp-steer
  (with positions, so you can identify "where on the lap is
  this happening?")
- replan-reason breakdown plus planner-ms percentiles

### `pnpm exec tsx scripts/controller-bench.ts`

Runs the race scenario PLUS 3 parking sub-scenarios
(forward-pullin, reverse-perp, parallel) through the SAME
`createRaceScenario` runner. Pass/fail table answers "did
the controller break anything?" after any change.

## Tests

- `demos/test/race-regression.test.ts` — kinematic and v2 must
  complete their target laps with off-track ≤ 5.
- `demos/test/parking-precision.test.ts` — stricter than the
  controller-bench (sub-meter position + single-digit-degree
  heading on most scenarios).
- `core/test/execute/pure-pursuit.test.ts` — unit tests for the
  unified brake-to-target + brake-distance vPath pass.

All passing.

## Web/CLI parity

`demos/app/raceprimitives/RacePrimitives.tsx` and
`demos/app/parking/Parking.tsx` both consume `createRaceScenario`
from `demos/app/lib/race-scenario.ts`. Every fix in this iteration
lands automatically in both surfaces — the React demo is a thin
renderer, not a fork.

## What needs to happen next

To close the residual v2 gap to kinematic, the model itself needs
retraining against Rapier-generated rollouts so the planner's
predicted curvature matches the chassis's actual response. The
handoff identified this; the empirical fixes I attempted
(`steer/steerRatio` correction, lookahead curvature) confirmed
that no controller-side trick is going to close it. The right
work is:

1. Generate Rapier-only training data (no kinematic prior).
2. Retrain v2 with a residual head that learns the
   `steerRatio` and `yawRateTau` from Rapier's actual response.
3. Validate via `pnpm run race` — predErrorRMS should drop
   below ~0.3 m (currently ~0.9), lateral-error replans should
   drop from ~70 % of all replans to under 30 %.

After that, the existing controller stack (unified pure-pursuit
+ brake-distance vPath + terminal-speed override) is the right
shape for racing AND parking, and v2 should naturally take its
place ahead of kinematic.
