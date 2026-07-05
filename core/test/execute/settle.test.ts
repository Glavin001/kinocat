import { describe, expect, it } from 'vitest';
import { createSettleLatch } from '../../src/execute/settle';

const DT = 1 / 60;

function feed(latch: ReturnType<typeof createSettleLatch>, ok: boolean, speed: number, seconds: number) {
  const n = Math.round(seconds / DT);
  for (let i = 0; i < n; i++) latch.update({ ok, speed }, DT);
  return latch.state;
}

describe('createSettleLatch', () => {
  it('latches only after the condition holds continuously for holdSeconds', () => {
    const latch = createSettleLatch({ holdSeconds: 2, speedTol: 0.05 });
    feed(latch, true, 0, 1.5);
    expect(latch.state.settled).toBe(false);
    feed(latch, true, 0, 0.6);
    expect(latch.state.settled).toBe(true);
    expect(latch.state.violations).toBe(0);
  });

  it('a transient drive-through does not latch (the bench break-on-first-true bug)', () => {
    const latch = createSettleLatch({ holdSeconds: 2, speedTol: 0.05 });
    // In the region but still moving — the historic "parked at |v|=0.15" case.
    feed(latch, true, 0.15, 10);
    expect(latch.state.settled).toBe(false);
    expect(latch.state.heldFor).toBe(0);
  });

  it('a mid-shunt pause shorter than the hold window resets, not latches', () => {
    const latch = createSettleLatch({ holdSeconds: 2, speedTol: 0.05 });
    feed(latch, true, 0, 1.0); // pause in region
    feed(latch, false, 1.2, 0.5); // shunts back out
    expect(latch.state.settled).toBe(false);
    expect(latch.state.heldFor).toBe(0);
    feed(latch, true, 0, 2.1); // final, real park
    expect(latch.state.settled).toBe(true);
  });

  it('timeToSettled is when the successful hold began, not when it latched', () => {
    const latch = createSettleLatch({ holdSeconds: 2, speedTol: 0.05 });
    feed(latch, false, 2, 5); // 5 s of driving
    feed(latch, true, 0, 2.5); // rest begins at t=5
    expect(latch.state.settled).toBe(true);
    expect(latch.state.timeToSettled!).toBeCloseTo(5, 1);
  });

  it('post-latch creep-out counts violations and never unlatches', () => {
    const latch = createSettleLatch({ holdSeconds: 1, speedTol: 0.05 });
    feed(latch, true, 0, 1.5);
    expect(latch.state.settled).toBe(true);
    feed(latch, true, 0.3, 0.5); // creeps at 0.3 m/s while "in" the region
    expect(latch.state.settled).toBe(true);
    expect(latch.state.violations).toBe(Math.round(0.5 / DT));
    feed(latch, true, 0, 1);
    expect(latch.state.violations).toBe(Math.round(0.5 / DT)); // stable again
  });

  it('signed speed is compared by magnitude (reverse creep is not "at rest")', () => {
    const latch = createSettleLatch({ holdSeconds: 1, speedTol: 0.05 });
    feed(latch, true, -0.8, 3); // reversing through the region
    expect(latch.state.settled).toBe(false);
  });
});
