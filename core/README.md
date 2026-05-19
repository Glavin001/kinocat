# kinocat (core)

The kinocat library. Pure TypeScript, tree-shakeable, zero runtime
dependencies; navcat / Rapier / three.js are **optional peer dependencies**
used only by the `adapters/*` subpaths.

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
