# kinocat (core)

The kinocat library. Pure TypeScript, tree-shakeable, zero runtime
dependencies; navcat / Rapier / three.js are **optional peer dependencies**
used only by the `adapters/*` subpaths.

## Quick start

```ts
import { plan } from 'kinocat/planner';
import { InMemoryNavWorld, VehicleEnvironment } from 'kinocat/environment';
import { defaultVehicleAgent, kinematicForwardSim } from 'kinocat/agent';
import { characterizeVehicle } from 'kinocat/primitives';

const agent = defaultVehicleAgent();
const lib = characterizeVehicle({
  forwardSim: kinematicForwardSim(agent),
  controlSets: [[0, 6], [1 / agent.minTurnRadius, 6], [-1 / agent.minTurnRadius, 6], [0, -4]],
  duration: 0.5, substeps: 6, startSpeeds: [0],
});
const world = new InMemoryNavWorld([{ id: 1, y: 0, ring: [[0,-10],[40,-10],[40,10],[0,10]] }]);
const env = new VehicleEnvironment(world, agent, lib, { goalRadius: 1.5, goalHeadingTol: Infinity });

const result = plan(
  {
    start: { x: 2, z: 0, heading: 0, speed: 0, t: 0 },
    goal: { x: 30, z: 4, heading: 0, speed: 0, t: 0 },
    environment: env,
  },
  50, // anytime deadline (ms)
);
// result.found, result.path (VehicleState[]), result.cost, result.stats
```

Wrap `env` in `TimeAwareEnvironment` (from `kinocat/environment`) to add
predicted-obstacle avoidance and lazy affordance edges; track the plan with
`purePursuit` from `kinocat/execute`.

## Layout

```
src/
  curves/        Reeds-Shepp & Dubins analytical curves
  primitives/    motion-primitive library + characterization harness
  agent/         vehicle / humanoid agent metadata
  planner/       IGHA* anytime time-extended planner core
  environment/   Environment interface, NavWorld seam, env impls, R2 oracle
  predict/       Predict<T> factories, affordance & plan registries
  execute/       curvature-aware pure-pursuit tracker + replan logic
  adapters/      navcat / rapier / three integrations (optional peers)
  internal/      heap, planar math, asserts, JSON serialization
```

## Decoupling seam

The core never imports navcat. `VehicleEnvironment` / `HumanoidEnvironment`
consume a kinocat-owned `NavWorld` / `PolygonRef` interface
(`environment/nav-world.ts`). `InMemoryNavWorld` (polygon soup, zero deps)
makes the entire algorithmic core unit-testable without any external runtime.
`adapters/navcat` implements `NavWorld` against navcat's public API.

## No linter / formatter — intentional

This package ships **no ESLint/Prettier/Biome**, by design. The shared
`tsconfig.base.json` (`strict`, `noUncheckedIndexedAccess`, `isolatedModules`,
`forceConsistentCasingInFileNames`) plus `tsc --noEmit` and vitest are the
correctness gates. This keeps the toolchain minimal and web-native, avoids a
large dependency tree, and keeps diffs deterministic. If a low-cost guard is
ever wanted, prefer `tsc`'s `noUnusedLocals`/`noUnusedParameters` (zero new
deps) over a linter.

## Build & gates

- `pnpm --filter kinocat build` — tsup emits ESM + `.d.ts` per subpath, plus a
  minified size-check artifact (peers externalized).
- `pnpm --filter kinocat size` — asserts core + navcat-adapter < 100 KB
  minified (spec §15.8); logs gzip too.
- `pnpm --filter kinocat typecheck` — `tsc -p tsconfig.json --noEmit`.
- Tests run from the repo root via the shared `vitest.config.ts`; coverage gate
  is 80% on `core/src/**` excluding barrels, `types.ts`, `adapters/**`, and the
  `r2-environment.ts` validation harness.
