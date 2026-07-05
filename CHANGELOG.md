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
  and `docs/adding-a-domain.md` (eight-step recipe; aircraft as the worked
  example).

### Changed

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
