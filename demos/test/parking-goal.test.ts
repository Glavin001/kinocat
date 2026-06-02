// Verifies the parking page's goal is now DESCRIBED in the canonical
// kinocat/scenario layer and routed to the planner via the ScenarioEnvironment
// bridge — forward-pullin + reverse-perp are bridge-enabled; parallel stays on
// the legacy planner (documented parity gap in the cramped two-car slot).
import { describe, it, expect } from 'vitest';
import { compile, validate, goalRegions } from 'kinocat/scenario';
import { parkingCourse, buildParkingScenario, parkingPlannerGoal } from '../app/lib/parking-scenarios';

describe('parking goal via the scenario layer', () => {
  it('bridge-enabled scenarios carry a canonical goal; parallel falls back to legacy', () => {
    expect(parkingCourse('forward-pullin').goal).toBeDefined();
    expect(parkingCourse('reverse-perp').goal).toBeDefined();
    expect(parkingCourse('parallel').goal).toBeUndefined();
  });

  it('the parking goal compiles + validates clean and aims at the stall pose', () => {
    const s = buildParkingScenario('reverse-perp');
    const spec = parkingPlannerGoal(s);
    const automaton = compile(spec.goal);
    expect(automaton.accepting.length).toBeGreaterThan(0);
    // One objective region (the at-pose goal), aimed at the stall.
    const regions = goalRegions(spec.goal);
    expect(regions).toHaveLength(1);
    const rep = regions[0]!.representative();
    expect(Math.hypot(rep.x - s.goal.x, rep.z - s.goal.z)).toBeLessThan(1e-6);
    // Validation (margins vs the 0.3 m parking grid) is clean.
    const errors = validate(
      { name: 'park', start: s.spawn, goal: spec.goal, invariants: spec.invariants },
      { posCell: 0.3 },
    ).filter((d) => d.severity === 'error');
    expect(errors).toEqual([]);
  });
});
