import { describe, it, expect } from 'vitest';
import {
  buildRampCourse,
  buildRampSnapshot,
  rampHeightSampler,
} from '../app/lib/ramp-scenarios';

describe('ramp + affordance demo', () => {
  it('builds a course with one ramp, one gap, one jump', () => {
    const c = buildRampCourse();
    expect(c.ramps.length).toBe(1);
    expect(c.gaps.length).toBe(1);
    expect(c.jumps.length).toBe(1);
    const j = c.jumps[0]!;
    // Launch should be at the crest of the ramp (base.x + length/2 for heading=0).
    const r = c.ramps[0]!;
    expect(j.launch.x).toBeCloseTo(r.base.x + r.length / 2, 5);
    expect(j.launch.z).toBeCloseTo(r.base.z, 5);
    // Land should be on the far side of the gap.
    expect(j.land.x).toBeGreaterThan(c.gaps[0]!.x + c.gaps[0]!.hx);
  });

  it('rampHeightSampler is 0 off-ramp, height at the crest, drops past the crest', () => {
    const c = buildRampCourse();
    const sampler = rampHeightSampler(c.ramps);
    const r = c.ramps[0]!;
    expect(sampler(-50, 0)).toBe(0); // pre-ramp ground
    expect(sampler(r.base.x + r.length / 2, 0)).toBeCloseTo(r.height, 1);
    // Foot of the up-slope is at ground level.
    expect(sampler(r.base.x - r.length / 2, 0)).toBeCloseTo(0, 5);
    // Past the crest the sampler returns 0 — heightfield sample-step gives
    // the lip the car launches off.
    expect(sampler(r.base.x + r.length / 2 + 1, 0)).toBe(0);
    // Way out laterally is flat ground too.
    expect(sampler(0, 20)).toBe(0);
  });

  it('plan reaches the goal with the affordance enabled', () => {
    const s = buildRampSnapshot({ withAffordance: true });
    expect(s.result.found).toBe(true);
    expect(s.result.path.length).toBeGreaterThanOrEqual(2);
  });

  it('plan reaches the goal without the affordance (detours around the gap)', () => {
    const s = buildRampSnapshot({ withAffordance: false });
    expect(s.result.found).toBe(true);
    expect(s.result.path.length).toBeGreaterThanOrEqual(2);
  });

  it('affordance plan is strictly cheaper than the detour plan', () => {
    const withAff = buildRampSnapshot({ withAffordance: true });
    const noAff = buildRampSnapshot({ withAffordance: false });
    expect(withAff.result.found).toBe(true);
    expect(noAff.result.found).toBe(true);
    const lastWith = withAff.result.path[withAff.result.path.length - 1]!;
    const lastNo = noAff.result.path[noAff.result.path.length - 1]!;
    // g-cost is monotone in time-on-path for this agent; compare arrival t.
    expect(lastWith.t).toBeLessThan(lastNo.t);
  });

  it('without the gap the detour is unnecessary and the no-affordance plan still solves', () => {
    const s = buildRampSnapshot({ withAffordance: false, withGap: false });
    expect(s.result.found).toBe(true);
    expect(s.course.gaps.length).toBe(0);
  });
});
