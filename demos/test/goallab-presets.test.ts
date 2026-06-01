// GoalLab preset catalog tests — every preset must compile, validate clean, and
// be planned by the real ScenarioEnvironment product search (full or partial).

import { describe, it, expect } from 'vitest';
import { compile, validate, collectScenarioRegions } from 'kinocat/scenario';
import { goalLabPresets } from '../app/lib/goallab-presets';

describe('GoalLab presets', () => {
  const presets = goalLabPresets();

  it('exposes a non-empty catalog with unique ids', () => {
    expect(presets.length).toBeGreaterThanOrEqual(4);
    expect(new Set(presets.map((p) => p.id)).size).toBe(presets.length);
  });

  for (const preset of goalLabPresets()) {
    it(`${preset.id}: compiles, validates clean, and plans`, () => {
      const automaton = compile(preset.scenario.goal);
      expect(automaton.states.length).toBeGreaterThan(0);

      const errors = validate(preset.scenario, { posCell: 0.3 }).filter(
        (d) => d.severity === 'error',
      );
      expect(errors).toEqual([]);

      // The scenario references at least one objective region to visualize.
      const regions = collectScenarioRegions(preset.scenario);
      expect(regions.objective.length).toBeGreaterThan(0);

      const result = preset.plan();
      expect(result.raw.found).toBe(true);
      expect(result.path.length).toBeGreaterThanOrEqual(2);
    });
  }
});
