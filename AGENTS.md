# AGENTS.md

## Cursor Cloud specific instructions

kinocat is a **pnpm monorepo** (Node ≥ 22, pnpm 10.33). Three workspace packages: `core/` (the `kinocat` library), `three/` (`@kinocat/three` stub), and `demos/` (`@kinocat/demos`, a Next.js 15 demo app). There is **no backend, database, or external service** — everything runs in-process / in the browser. Standard scripts and setup are documented in `README.md` (§Develop) and root `package.json`; the notes below only cover non-obvious things.

### Running the demo app
- `pnpm dev` builds the core library, then starts the Next.js dev server at http://localhost:3000. To skip the core rebuild when it is already built, run `pnpm --filter @kinocat/demos dev` directly.
- The landing page lists ~24 demos. The flagship demos are `/carchase`, `/ramp`, `/raceprimitives`, and `/parking`.

### Build ordering (non-obvious)
- The core must be built before typechecking or running the app, but you rarely invoke this yourself: `pnpm dev`, `pnpm typecheck`, and `pnpm verify` each run `pnpm --filter kinocat build` first automatically.
- `pnpm test` (vitest) runs against **source** — the root `vitest.config.ts` aliases `kinocat` → `core/src`, so tests do **not** require a prior build.

### Testing / gates
- No ESLint/Prettier/Biome by design — `tsc --noEmit` + vitest are the only correctness gates.
- Full CI gate is `pnpm verify` (typecheck + test + build + size). The test suite is ~700 tests and takes roughly 2 minutes.

### 3D demo performance in the cloud VM (important)
- The 3D physics demos (`/carchase`, `/ramp`, `/raceprimitives`, `/parking`, `/dogfight`, `/world3d`, etc.) use three.js + Rapier WASM. The cloud VM has **no GPU** (software WebGL) and 4 CPUs, so these scenes render and run their planning/physics logic correctly but **animate slowly/choppily** — the planner's per-plan expansion time can be 1–2+ s. This is an environment limitation, not a bug; on-screen stats/phase indicators still advance and there are no console errors.
- The 2D demos (`/playground`, `/curves`, `/primitives`, `/anytime`, etc.) render on a plain 2D canvas and are smooth — prefer `/playground` for a quick, responsive end-to-end check that the core planner works (drag the goal marker; the path replans live).
