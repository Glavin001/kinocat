// Verify the v2 library builds successfully and produces primitives the
// planner can consume. Lightweight (no Rapier dependency at fit time —
// uses default v2 params).

import { describe, it, expect } from 'vitest';
import { buildLearnedLibraryV2 } from '../app/lib/learned-library-v2';

describe('buildLearnedLibraryV2 — drop-in MotionPrimitiveLibrary', () => {
  it('coarse tier produces 5 actions × 4 start speeds = 20 primitives', () => {
    const lib = buildLearnedLibraryV2({ tier: 'coarse' });
    expect(lib.primitives.length).toBe(20);
    expect(lib.startSpeeds).toEqual([0, 4, 8, 12]);
  });

  it('fine tier produces a denser library', () => {
    const lib = buildLearnedLibraryV2({ tier: 'fine' });
    expect(lib.primitives.length).toBeGreaterThan(80);
  });

  it('lookup(speed) returns primitives applicable from that bucket', () => {
    const lib = buildLearnedLibraryV2({ tier: 'coarse' });
    const atRest = lib.lookup(0);
    expect(atRest.length).toBe(5);
    const at8 = lib.lookup(8);
    expect(at8.length).toBe(5);
  });

  it('each primitive has the expected end-state offset shape', () => {
    const lib = buildLearnedLibraryV2({ tier: 'coarse' });
    for (const p of lib.primitives) {
      expect(p.end).toHaveProperty('dx');
      expect(p.end).toHaveProperty('dz');
      expect(p.end).toHaveProperty('dHeading');
      expect(p.end).toHaveProperty('speed');
      expect(p.sweep.length).toBe(7); // 1 start + 6 substeps
    }
  });
});
