// Generic active-exploration sampler.
//
// Given the current model's prediction error map (caller-supplied
// `ExplorationCell` list — each cell is a region of state-control space with
// a measured error and visit count), propose a batch of trials to run next
// to maximally reduce the model's uncertainty / error.
//
// The caller is responsible for:
//   - Defining what a "cell" means (e.g. (speed × curvature) bins).
//   - Computing per-cell error/uncertainty from the current trial set.
//   - Synthesizing concrete trial specs from chosen cells.
//
// Reference scoring:  cell_score = errorRms × log(1 + uncertaintyStd) / sqrt(1 + count)
//   - High error → sample more
//   - High uncertainty → sample more (epistemic, reducible by data)
//   - Low count → sample more (low confidence in current per-cell stats)

export interface ExplorationCell<Spec> {
  /** Stable cell id (caller-supplied; usually a flat-encoded bin index). */
  id: string;
  /** Current model RMS prediction error in this cell (consumer-computed). */
  errorRms: number;
  /** Optional epistemic uncertainty in this cell (e.g. ensemble std). */
  uncertaintyStd?: number;
  /** Number of existing trials/samples in this cell. */
  count: number;
  /** Function to synthesize a concrete trial spec from this cell. Called
   *  `numSamples` times, returning fresh (jittered) specs each call. */
  sample: (rng: () => number) => Spec;
}

export interface ProposedTrial<Spec> {
  cellId: string;
  /** The concrete trial to run (consumer-defined shape). */
  spec: Spec;
  /** Score assigned to the parent cell. Higher = more valuable. */
  cellScore: number;
}

export interface ActiveExplorerOptions<Spec> {
  cells: ExplorationCell<Spec>[];
  /** Number of new trials to propose total. */
  budget: number;
  /** Always-include probes (e.g. extreme-input safety probes) prepended to
   *  the proposed list, regardless of cell scoring. Subtract from `budget`. */
  alwaysInclude?: Spec[];
  /** Random seed for jittering. */
  seed?: number;
  /** Floor for uncertainty in scoring (avoids 0 → 0 multiplicative wipeout
   *  when no ensemble has been trained yet). Default 0.01. */
  uncertaintyFloor?: number;
}

export function proposeNextBatch<Spec>(
  opts: ActiveExplorerOptions<Spec>,
): ProposedTrial<Spec>[] {
  const uFloor = opts.uncertaintyFloor ?? 0.01;
  const rng = makeRng(opts.seed ?? 1);

  const proposed: ProposedTrial<Spec>[] = [];
  // Always-include first.
  if (opts.alwaysInclude) {
    for (const spec of opts.alwaysInclude) {
      proposed.push({ cellId: '__always_include__', spec, cellScore: Infinity });
    }
  }
  let remaining = Math.max(0, opts.budget - proposed.length);
  if (remaining <= 0 || opts.cells.length === 0) return proposed;

  // Compute scores.
  const scored = opts.cells.map((c) => {
    const u = (c.uncertaintyStd ?? 0) + uFloor;
    const score = c.errorRms * Math.log(1 + u) / Math.sqrt(1 + c.count);
    return { cell: c, score };
  });
  const totalScore = scored.reduce((a, b) => a + Math.max(0, b.score), 0);
  if (totalScore <= 0) {
    // Uniform fallback.
    while (remaining > 0) {
      const c = opts.cells[Math.floor(rng() * opts.cells.length)]!;
      proposed.push({ cellId: c.id, spec: c.sample(rng), cellScore: 0 });
      remaining--;
    }
    return proposed;
  }
  // Weighted sampling with replacement.
  while (remaining > 0) {
    const r = rng() * totalScore;
    let acc = 0;
    let picked: typeof scored[number] | null = null;
    for (const s of scored) {
      acc += Math.max(0, s.score);
      if (r <= acc) { picked = s; break; }
    }
    if (!picked) picked = scored[scored.length - 1]!;
    proposed.push({
      cellId: picked.cell.id,
      spec: picked.cell.sample(rng),
      cellScore: picked.score,
    });
    remaining--;
  }
  return proposed;
}

function makeRng(seed: number): () => number {
  let state = (seed | 0) || 1;
  return (): number => {
    state = (state + 0x6d2b79f5) | 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
