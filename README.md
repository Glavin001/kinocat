# kinocat

**Time-extended kinodynamic motion planning for the web.**

kinocat is a pure-TypeScript, tree-shakeable, web-native motion planner that
gives browser-based 3D games F.E.A.R.-grade emergent NPC navigation: reverse
maneuvers, ballistic jumps, anticipatory routing over *predicted* future world
states, opportunistic affordance use, and emergent multi-agent cooperation.

The planner core is a TypeScript port of **IGHA\*** (Incremental Generalized
Hybrid A\*, Talia/Salzman/Srinivasa, RA-L 2025), extended with **time as a
state dimension** participating in multi-resolution dominance — the novel
kinocat contribution. It builds on
[navcat](https://github.com/isaac-mason/navcat) as its topology/collision
substrate, but the core never imports a physics or rendering library.

> Status: under active implementation. Algorithmic core (curves, planner,
> environments, prediction, execution) is built and tested with zero external
> dependencies; navcat / Rapier / three.js integrations live behind optional
> adapters.

## Packages

| Package | Path | What |
|---|---|---|
| `kinocat` | `core/` | The library (multi-entry, subpath exports) |
| `@kinocat/three` | `three/` | three.js debug helpers (demo-oriented) |
| `@kinocat/demos` | `demos/` | Next.js demo app |

## `kinocat` module map (subpath exports)

| Import | Responsibility |
|---|---|
| `kinocat/curves` | Reeds-Shepp & Dubins analytical curves |
| `kinocat/primitives` | Motion-primitive library + characterization harness |
| `kinocat/agent` | Vehicle / humanoid agent metadata |
| `kinocat/planner` | IGHA\* anytime, time-extended planner core |
| `kinocat/environment` | `Environment` interface, `NavWorld` seam, env impls |
| `kinocat/predict` | `Predict<T>` factories, affordance & plan registries |
| `kinocat/execute` | Curvature-aware pure-pursuit tracker + replan logic |
| `kinocat/adapters/navcat` | `NavWorld` over a navcat `NavMesh` (optional peer) |
| `kinocat/adapters/rapier` | Rapier `ForwardSim` wrapper (optional peer) |
| `kinocat/adapters/three` | Debug visualization (optional peer) |

`sideEffects: false` + per-subpath entry points: importing `kinocat/curves`
pulls zero planner/environment code. The core is decoupled from navcat behind a
small `NavWorld`/`PolygonRef` seam, so it is fully unit-testable with an
in-memory polygon world and no external runtime.

## Develop

```sh
pnpm install
pnpm test            # vitest (core)
pnpm test:coverage   # 80% gate on core algorithm + environments
pnpm typecheck       # tsc, all workspaces
pnpm --filter kinocat build   # tsup -> ESM + .d.ts
pnpm size            # asserts core + navcat-adapter < 100 KB minified
pnpm dev             # build core, then run the Next.js demo
```

Requires Node ≥ 22 and pnpm. No bundler-specific globals; the core is
browser/Node agnostic.

## References

- Talia, Salzman, Srinivasa — *Incremental Generalized Hybrid A\** (RA-L 2025)
- Reeds & Shepp (1990); Dubins (1957) — optimal car curves
- Dolgov et al. (2010) — Hybrid A\*
- Mononen — Recast Navigation auto-annotation (Paris Game AI 2011)
- Orkin — *Three States and a Plan: The A.I. of F.E.A.R.* (GDC 2006)
- navcat — Isaac Mason, https://github.com/isaac-mason/navcat
