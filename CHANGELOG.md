# Changelog

## 0.1.0 (unreleased)

First cut planned for npm. The theme of this release is **domain
convergence**: kinocat's four agent domains (car, humanoid, momentum
humanoid, aircraft) now share one set of documented, conformance-tested
seams instead of three parallel ad-hoc implementations.

### Added

- **`kinocat/testing` — domain conformance kit.** `runConformance(harness)`
  proves any `Environment<State>` satisfies the planner's contract:
  heuristic consistency + admissibility, successor invariants (positive
  costs, monotone time, g/h/f bookkeeping, dominance-key arity), hash
  stability, cross-instance determinism, anytime monotonicity, and budgeted
  scenario solvability. Framework-agnostic (structured report, no
  test-runner dependency); excluded from the size-gated runtime bundle.
- **Momentum humanoid — fourth agent domain.** Inertial person with human
  movement envelopes (sprint along the facing, strafe cap, launch/brake
  asymmetry, speed-degraded turning): `MomentumHumanoidState`,
  `MomentumHumanoidAgent`, `momentumHumanoidForwardSim`,
  `MomentumHumanoidEnvironment`. Built exclusively on public seams (zero
  planner-core edits). New `/crowd` demo: a runner weaving through
  pedestrians timed to cut its line, planned in space-time.
- **Generic `characterize<S>` primitive-rollout harness** with `crossRuns`,
  shared by car, aircraft, and momentum-humanoid primitive builders; the
  local-frame equivariance contract is now documented on the function.
- **`AircraftEnvOptions.timeInHash`** — drop the aircraft env's internal
  time bucket when composing with `TimeAwareEnvironment`.
- **Docs:** `docs/architecture.md` (the five seams + contract fine print)
  and `docs/adding-a-domain.md` — the "define a new controllable motion
  body" API guide, built around an EXECUTABLE worked example
  (`core/test/examples/hovercraft.test.ts`): a complete drifting,
  thrust-vectored hovercraft in one file, proven by the conformance battery
  with exact fidelity and a body-typed boost affordance. CI runs the
  example, so the documentation cannot rot.
- **`checkSuccessorFidelity`** in `kinocat/testing`: re-simulates each
  successor edge from the ACTUAL parent state (harness supplies
  `resimulate` + a tolerance, with `angularFields` compared on the circle)
  and reports per-field deviation — exact for live-rollout environments,
  a declared bucket-teleport bound for cached-primitive ones. Wired into
  the vehicle, aircraft, and momentum-humanoid conformance harnesses.

### Changed

- **Aircraft attitude is evolving state, not a snap-to output.**
  `AircraftAgent` gains `maxRollRate` / `maxPitchRate`;
  `aircraftForwardSim` integrates pitch and roll toward their setpoints
  (`Infinity` recovers the legacy quasi-static model bit-exactly). A
  primitive commanding a bank gets `maxRollRate·dt` of it per step — so
  knife-edge maneuvers are a matter of TIMING: the plane begins its roll
  before the slot, holds the bank through a double slot whose gap is too
  short to unroll and re-roll, and relaxes to level when the gap affords
  it (all emergent from search; pinned by tests).
- **`AircraftEnvironment` rolls primitives live through the sim** from each
  node's actual state instead of rigid-transforming cached canonical
  rollouts — with rate-limited attitude a cache misrepresents every
  mid-ramp start, and the sim is a rounding error next to the OBB
  narrowphase it feeds. `FlyEdgeData` now carries the full control quad
  (incl. `v`) so any edge can be re-simulated; the analytic shot fires only
  when settling to the segment attitude is a small fraction of the flight.
- demos `densifyPath` re-integrates at the planner's certified substep
  resolution (finer densities are linear subdivisions of the certified
  polygon), recovers rate-aware attitude setpoints, and reproduces
  analytic-shot segments linearly as certified — the rendered trajectory
  can no longer drift onto paths the planner never collision-checked.

- **Dynamic-world layer generalized beyond ground vehicles.**
  `MovingObstacle` predictions may carry an optional `y`; when both the
  obstacle and the wrapped state have altitude the collision proxy is a 3D
  sphere (y-less pairings keep the exact planar behavior, bit-for-bit).
  The time-aware broadphase gains sound altitude rejection.
  `Affordance<S>` / `AffordanceRegistry<S>` / `AffordanceUseResult<S>` are
  generic over the agent state (default `CarKinematicState`, so existing
  car code compiles unchanged); the `'speed' in state` vehicle gate and
  casts in `TimeAwareEnvironment` are gone. `MovingZone` is now structurally
  a `MovingObstacle`, so the same zone works at either fidelity level.

### Fixed

- `TimeAwareEnvironment` now forwards the optional `level` argument to the
  base environment's `succ` (per-level primitive sets were silently
  disabled under composition) and conditionally forwards the `progress`
  hook per the `Environment` contract.
