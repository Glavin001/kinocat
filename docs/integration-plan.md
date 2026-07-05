# kinocat integration plan — motion feasibility & execution layer

Status assessment against the integration charter (kinocat is feature-frozen;
integration work only), plus the ordered work plan for what remains.

Charter requirements:

1. Anytime `plan(req, deadline)`
2. Executor accepting external invalidation events (tiles changed under the
   active trajectory) and replanning
3. Cheap ETA/feasibility oracle ("can I reach region R, roughly how long") in
   ~1–2 ms for HTN operator costs
4. Consume new tiles and off-mesh links from layer 2 (navcat) without restart
5. 3–4 concurrent agents within frame budget

Done-when gates: replan-after-rebuild < 100 ms; ETA oracle < 2 ms/query;
4 agents at 60 fps; torture test — destroy the ground under an agent's
committed path and it recovers without teleporting or freezing.

---

## Part A — un-merged work audit

9 open PRs and 8 PR-less unmerged branches were reviewed against the charter.

### Recommend merging (advances the charter)

| Work | What it is | Relevance | State |
|---|---|---|---|
| **PR #18** (`claude/determined-galileo-Am4mi`) | Uniform-grid spatial index for `InMemoryNavWorld` collision queries; per-agent planner **worker pool** for carchase (replaces the single shared worker); replan interval 80 → 25 ms; `carchase.bench.ts` measuring per-replan cost | Directly serves req 5 (4 agents in frame budget) and the replan-latency gate. The worker-pool pattern is what Phase 4 below promotes into core. | 77 commits behind main; one small conflict in `core/src/internal/geom.ts`; `nav-world.ts` untouched on main since, so the bulk rebases cleanly. Rebase, re-run bench + carchase tests, merge. |
| **`claude/fix-pure-pursuit-stop`** (no PR) | Fixes pure-pursuit to stop reliably at goal terminals; converts `pure-pursuit-stop-known-bugs.test.ts` into passing `pure-pursuit-stop.test.ts` | Executor correctness — "recovers without freezing" depends on the tracker terminating cleanly. | Small (3 files), 15 commits behind. Verify against main's newer pure-pursuit follow-controller changes (PR #38 touched the same file), then merge. |

### Not needed for this charter

Everything else is feature/demo/training work, out of scope under the freeze.
None of it blocks the charter, and none of it helps it:

- **PR #39** (parking goal via scenario layer), **#36** (Stanley heading term),
  **#34** (rich Plan structure), **#24** (dense sweep visualization) — demo /
  controller / visualization quality for parking & racing.
- **PR #37** (eval harness), **#33** (model comparison tooling), **#30** (JAX
  trainer, stacked on #24's branch) — model-training and evaluation stack.
- **PR #25** (gitignore chore) — harmless housekeeping; merge or close at leisure.
- Branches `claude/kind-gates-R5Lyf`, `fix/racing`, `claude/fervent-cori-KXMEy`,
  `claude/funny-faraday-eddWk` (ramp wedge collision), dogfight/speed-primitive
  branches, `claude/av-library-scenario-testing-*` — demo experiments, training
  handoffs, or superseded fixes. Defer or close.

---

## Part B — requirement status (verified against source)

### 1. Anytime `plan(req, deadline)` — DONE

`plan<State>(req, deadlineMs)` at `core/src/planner/ighastar.ts:43`. Wall-clock
deadline checked every `deadlineCheckEvery` (64) expansions; true anytime loop
(keeps improving the incumbent, `solutionHistory` records every improvement);
best-progress partial fallback when no solution reached. Game-facing wrapper
`planVehicleOnce` (`plan-vehicle.ts:78`, default 120 ms). No work needed.

### 2. Executor + external invalidation — PARTIAL

`ReplanState` (`core/src/execute/replan.ts:34`) has the hook: `markDirty(reason)`
forces replan and bypasses plan-switch hysteresis. `markTileRebuilt(world,
affectedReplanStates)` (`core/src/adapters/navcat/tile-rebuild.ts:10`) is the
tile-event entry point. **Gaps:** the caller must decide which agents are
affected — there is no "does this committed trajectory cross the changed
region" test in the library; `bumpRevision()` is a dead counter (nothing in
core reads `world.revision`, and `NavcatWorld`'s memoized goal-distance field
and clearance field are *not* invalidated by it); `markTileRebuilt` never swaps
in rebuilt geometry.

### 3. ETA/feasibility oracle — MISSING as an API (building blocks exist)

No packaged `eta(start, region)` query. The pieces are all there and fast in
the lookup regime: `VehicleEnvironment.heuristic()` returns an admissible
time-to-goal lower bound (`vehicle-environment.ts:413`) combining a Reeds-Shepp
LUT and `NavcatWorld.buildGoalLowerBound()` (`adapters/navcat/index.ts:101`) —
a per-goal grid-Dijkstra over the CompactHeightfield with O(1) lookups after a
one-time build. The build (not the lookup) is the real cost, is unmeasured in
isolation, is memoized for only one goal at a time, and takes a point goal, not
a region.

### 4. New tiles + off-mesh links without restart — PARTIAL

Off-mesh links can be added live (`annotateJumpLinks` → `world.addOffLink`,
consumed via `offMeshFrom` / `AffordanceRegistry`). Tiles: only full
from-scratch world rebuilds exist (`navWorldFromTriangleMesh`, demo pattern in
`World3D.tsx:164`: rebuild whole world → `markDirty('edit')`). **Gaps:** no way
to hand `NavcatWorld` a rebuilt navmesh/CHF without constructing a new world;
revision-keyed cache invalidation not implemented; the planner **worker
protocol** (`core/src/worker/protocol.ts`) supports only `init` and `plan` —
workers cannot consume new tiles or links without a full re-init (a restart in
all but name).

### 5. 3–4 concurrent agents in frame budget — PARTIAL

Carchase runs 4 agents (robber + 3 cops) via one worker and a staggered
round-robin (`REPLAN_INTERVAL_MS = 80`, one agent per tick), with in-flight
dedup and stale-result rejection — all in the demo (`CarChase.tsx:696-780`),
not core. PR #18 upgrades this to a per-agent worker pool. **Gaps:** no
core-level pool/scheduler abstraction; no per-frame budget governor; no
automated 60 fps acceptance measurement.

### Done-when gates — all currently unmeasured

No replan-after-rebuild latency bench; no standalone ETA-query bench; no
4-agents-at-60fps measurement; no torture test (only unit tests:
`replan.test.ts:57`, `navcat.test.ts:84`).

---

## Part C — work plan

Ordered so each phase lands the API the next one consumes. All phases are
integration/plumbing — no planner-feature work.

### Phase 0 — merge hygiene (½ day)

- Rebase PR #18 onto main (resolve `geom.ts` conflict), re-run
  `carchase.bench.ts` + demo tests, merge.
- Verify `claude/fix-pure-pursuit-stop` against main's current pure-pursuit,
  merge if green. Optionally merge #25; close/defer the rest.

### Phase 1 — region-scoped external invalidation (req 2 gap; ~1–2 days)

1. Define `TileChange` in the navcat adapter: `{ bounds: Aabb2 | polygonIds,
   revision }`.
2. Add a cheap committed-trajectory–vs–region intersection test (swept
   footprint-inflated `PlanPath` segments against the AABB) in
   `core/src/execute/` — this is the missing "tiles changed under the *active
   trajectory*" logic.
3. Extend `markTileRebuilt(world, change, agents)` to auto-filter agents by
   that test (keep the current explicit-list signature for compatibility).
4. Make revision real: key `NavcatWorld`'s `goalLB` and clearance-field caches
   by revision so `bumpRevision()` actually invalidates; add
   `NavcatWorld.swapNavMesh(navMesh, chf?)` so the game can hand in rebuilt
   geometry without constructing a new world (preserving off-mesh links and
   registered state).
5. Tests: only agents whose committed path crosses the changed region get
   dirty; stale goal-field regression test (replan to same goal after rebuild
   sees new geometry).

### Phase 2 — live world updates through the worker seam (req 4 gap; ~1–2 days)

1. Add a `world-update` message to `core/src/worker/protocol.ts` carrying new
   polygons / serialized navmesh delta, off-mesh links, and revision; handle in
   `planner-worker.ts` without re-init.
2. Promote the demos' `markDirty('off-mesh')` pattern into a core helper next
   to `markTileRebuilt`.
3. **Gate bench:** `core/bench/replan-after-rebuild.bench.ts` — rebuild tile →
   world-update → dirty → replan; assert p95 end-to-end < 100 ms.

### Phase 3 — ETA/feasibility oracle (req 3; ~2 days)

1. New `core/src/predict/eta-oracle.ts`: `createEtaOracle(world, agent)` →
   `eta(start, region) => { reachable, seconds } | null`.
2. Implementation: extend `ChfGoalDistanceField` to accept region seeds
   (multi-source Dijkstra from all cells in R), LRU-memoize a handful of
   region fields (HTN operators re-query the same regions), invalidate on
   revision bump (Phase 1). Query = O(1) field lookup, floored by the
   Reeds-Shepp/Euclid kinematic bound, divided by `maxSpeed`; `reachable` =
   lookup non-null.
3. Field builds happen off the query path (prebuild per HTN operator region /
   amortized), so steady-state queries are microseconds.
4. **Gate bench:** `core/bench/eta-oracle.bench.ts` — steady-state query
   < 2 ms (expect ≪), report per-region build cost separately so the HTN layer
   can budget it.

### Phase 4 — multi-agent scheduling in core (req 5; ~2 days)

1. Extract the proven demo patterns (PR #18 pool + CarChase stagger, dedup,
   stale-rejection, emergency slot-steal) into `core/src/worker/pool.ts`:
   `PlannerPool(n)` with per-agent request slots and the `world-update`
   broadcast from Phase 2.
2. Add a light frame-budget governor for the main-thread side (plan adoption,
   tracker, sync fallback): skip/defer work past an X-ms-per-frame cap.
3. **Gate measurement:** scripted carchase run (4 agents, Phase-0 bench
   extended) asserting frame time ≤ 16.6 ms on the reference scene, and each
   agent replanning at its target cadence.

### Phase 5 — the torture test (done-when gate; ~1 day)

Automated headless test (vitest, `demos/test/` or `core/test/adapters/`):
agent mid-execution on a committed path across a platform; destroy the tiles
under the path via `swapNavMesh` + `markTileRebuilt(change)`. Assert:

- the agent is auto-detected as affected (no manual list) and marked dirty;
- a valid replacement plan is adopted within the 100 ms budget;
- **no teleport** — consecutive executed poses within `maxSpeed × dt`;
- **no freeze** — controller keeps emitting controls and the agent keeps
  making progress toward the goal;
- the new path avoids the destroyed region.

### Acceptance mapping

| Gate | Where it's proven |
|---|---|
| replan-after-rebuild < 100 ms | Phase 2 bench |
| ETA oracle < 2 ms/query | Phase 3 bench |
| 4 agents at 60 fps | Phase 4 measurement |
| torture test | Phase 5 test |

Estimated total: ~7–9 working days including review cycles.
