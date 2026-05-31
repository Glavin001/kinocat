# AV Library Scenario Testing — Handoff & Next Steps

PR: [Glavin001/kinocat#26](https://github.com/Glavin001/kinocat/pull/26)
(branch `claude/av-library-scenario-testing-2Fhtk`)

This document is the handoff for the headless scenario-testing work. It is
self-contained — it assumes no prior context from the PR conversation, and
it ends with a prioritized list of next steps.

---

## TL;DR

The demo scenarios (`/obstaclecourse`, `/ramp`, race, parking) used to rely
on **teleportation** to keep agents "on track" — when a controller got
stuck or drifted, the runner would silently snap the vehicle back to a valid
pose. That made every scenario *look* green even when the underlying planner
was failing.

This PR removes that crutch and makes the scenarios **honest and headless**:

- A telemetry monitor + geometry helpers record what actually happened
  (collisions, replan-failure ratio, teleport count, final pose).
- Race and parking now run as deterministic, headless factories with
  invariant tests — no browser, bit-exact reproducible.
- **All teleportation is gone** from the shared race/parking runner; a stuck
  controller now fails the run instead of being rescued.
- `/obstaclecourse` and `/ramp` were migrated onto the same deterministic
  headless factories, so the pages and the tests run the identical code path.
- A **golden bench** (`controller-bench`) gates the unified scenarios in CI
  with a recorded invariant summary (`collide=no, teleport=0`).

**End state:** `pnpm test` → 462 passed / 0 failed, 2 skipped (pre-existing
carchase flake — see below). Typecheck clean. Determinism bit-exact. Bench
4/4 with `collide=no, teleport=0`.

**The honesty paid off immediately** — removing teleportation exposed real
failures that were previously masked:

1. A **v2 race DNF** (the car does not finish under the honest runner).
2. **Reverse-perpendicular and parallel parking** both fail — the planner
   enters a replan-failure storm instead of parking.

These are documented as `it.fails` tests (see "How failures are encoded"),
so they are exercised on every run and will alert us the moment they get
fixed.

---

## What shipped (commit by commit)

| Commit | Summary |
| --- | --- |
| `29424b0` | Telemetry monitor + geometry helpers for headless scenario testing |
| `92b6e7e` | Headless scenario factories + determinism tests (race & parking) |
| `c74841c` | Remove teleportation masking from parking; fail honestly on timeout |
| `9807db1` | Unify `/obstaclecourse` onto a headless, deterministic scenario |
| `e05993e` | Unify `/ramp` onto a headless, deterministic scenario |
| `05f27e0` | Remove **all** teleportation from the shared race/parking runner |
| `ba67327` | Wire `controller-bench` as a golden gate over the unified scenarios |

---

## Key files & how to run things

### Tests (Vitest, in `demos/test/`)

- `parking-invariants.test.ts` — forward-pullin (passes), reverse-perp &
  parallel (`it.fails`, documented broken).
- `race-invariants.test.ts`, `headless-race.test.ts` — race honesty +
  the exposed v2 DNF.
- `obstaclecourse-invariants.test.ts`, `ramp-invariants.test.ts` — invariants
  for the newly-unified pages.
- `determinism.test.ts` — bit-exact reproducibility of the headless runs.

### Bench (golden gate)

```bash
pnpm run controller-bench               # human-readable, 4/4 expected
pnpm run controller-bench:json          # writes docs/controller-bench/latest.json
```

The bench records the invariant summary (`collide`, `teleport`, final pose)
for the unified scenarios and is the artifact CI compares against.

### Full local check

```bash
pnpm test          # 462 passed / 0 failed / 2 skipped
pnpm typecheck     # clean
```

---

## How failures are encoded (read this before "fixing" a green test)

Two distinct mechanisms — don't confuse them:

- **`it.fails(...)`** — the test *runs*, the correct-behaviour assertion
  *throws as expected*, and Vitest counts that expected failure as a **pass**.
  Used for `reverse-perp` and `parallel` parking. When someone fixes the
  planner, these tests will start passing unexpectedly → Vitest flips them to
  **failing** → that is the signal to delete the `.fails` marker. This is
  intentional: broken behaviour stays exercised and self-documenting instead
  of being skipped and forgotten.
- **`describe.skip` / `it.skip`** — the test does **not** run at all. Only the
  carchase suite uses this, and it predates this PR.

So: the broken parking scenarios are **not skipped** — they run on every CI
pass as expected-failures.

---

## Known issues / deferred work

### 1. Carchase migration (deferred)
The carchase demo (`/carchase`) was **not** migrated onto the headless
factories — deferred to a focused follow-up. It carries the project's other
masked failure: `demos/test/carchase-scenarios.test.ts` has a
`describe.skip` ("pre-existing flake") where 2 of 3 spawn-cops hit the
25000-expansion budget cap in PURSUE mode. See
`docs/v2-model-handoff.md` § "Other open issues".

### 2. v2 race DNF (exposed, not fixed)
With teleportation removed, the v2 car does not finish the race. The DNF is
captured by the race invariants; root-causing the controller/planner is out
of scope for this PR.

### 3. Reverse-perp & parallel parking (exposed, not fixed)
Both enter a replan-failure storm (`failedReplanRatio` exceeds threshold)
rather than parking. Encoded as `it.fails`.

---

## Next steps (prioritized)

1. **Carchase migration + un-skip.** Migrate `/carchase` onto the headless
   factories like obstaclecourse/ramp, and convert the `describe.skip` to
   either a real passing test (if retuning/budget fixes it) or an `it.fails`
   so it stops being silently skipped. This is the natural next PR — it closes
   the last teleportation/masking gap.

2. **Fix the parking planner (reverse-perp, parallel).** Diagnose the
   replan-failure storm. When fixed, the `it.fails` tests will flip to red —
   remove the `.fails` markers as part of the fix.

3. **Root-cause the v2 race DNF.** Now that the runner is honest, the DNF is
   reproducible headlessly and bit-exactly — a good substrate for debugging
   the v2 controller without browser flake.

4. **(Optional) Make parking loud instead of green.** Today reverse-perp /
   parallel are green-but-documented via `it.fails`. If you'd rather CI be
   *red* until they're fixed, convert them to plain failing tests. Trade-off:
   louder signal vs. a non-green main branch.

5. **Extend the bench.** As carchase joins the unified factories, add it to
   `controller-bench` and the golden JSON so every scenario shares one gate.

---

## Honesty notes (the meta-point of this PR)

The whole theme here is *trusting scenarios from headless data alone*. Two
anti-patterns to keep stamping out:

- **Teleportation rescue** — masked planner failures by snapping poses.
  Removed from race/parking; still implicitly absent from the migrated pages.
- **Skip-to-stay-green** — the carchase `describe.skip` is the same
  anti-pattern in a different costume. Prefer `it.fails` over `.skip` so
  broken behaviour stays visible and self-heals its own signal when fixed.
