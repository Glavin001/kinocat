import { describe, expect, it } from 'vitest';
import { createHardExampleMiner } from 'kinocat/training';

interface S { x: number }
interface C { u: number }
interface Cfg { id: string }

describe('createHardExampleMiner', () => {
  it('emits a trial covering the window around a triggering frame', () => {
    const miner = createHardExampleMiner<S, C, Cfg>({
      gapPredicate: (f) => f.state.x > 5,
      windowTicks: 2,
      dt: 0.1,
      sampleEveryNTicks: 1,
      config: { id: 'A' },
      configKey: 'A',
    });
    // Feed 6 frames; trigger at frame index 2 (state.x = 6).
    const xs = [1, 2, 6, 7, 8, 9];
    let emitted = null;
    for (let i = 0; i < xs.length; i++) {
      const t = miner.observe({
        simTime: i * 0.1,
        state: { x: xs[i]! },
        controls: { u: 0 },
      });
      if (t) emitted = t;
    }
    expect(emitted).not.toBeNull();
    // windowTicks=2 → trial spans 5 frames; controls = 4
    expect(emitted!.controlsTrace.length).toBe(4);
    expect(emitted!.samples.length).toBe(5);
    expect(emitted!.maneuverId).toBe('mined');
    expect(emitted!.split).toBeDefined();
  });

  it('respects cooldown to avoid spam', () => {
    const miner = createHardExampleMiner<S, C, Cfg>({
      gapPredicate: () => true,
      windowTicks: 1,
      dt: 0.1,
      sampleEveryNTicks: 1,
      config: { id: 'A' },
      configKey: 'A',
      cooldownTicks: 5,
    });
    let count = 0;
    for (let i = 0; i < 10; i++) {
      const t = miner.observe({
        simTime: i * 0.1,
        state: { x: 0 },
        controls: { u: 0 },
      });
      if (t) count++;
    }
    expect(count).toBeGreaterThan(0);
    expect(count).toBeLessThan(10);
  });

  it('does not emit until the post-window has filled', () => {
    const miner = createHardExampleMiner<S, C, Cfg>({
      gapPredicate: (f) => f.simTime > 0.05 && f.simTime < 0.15,
      windowTicks: 3,
      dt: 0.1,
      sampleEveryNTicks: 1,
      config: { id: 'A' },
      configKey: 'A',
    });
    // Feed two frames — even though the trigger fires at frame 1, the post-
    // window needs 3 more, so nothing emits yet.
    expect(miner.observe({ simTime: 0, state: { x: 0 }, controls: { u: 0 } })).toBeNull();
    expect(miner.observe({ simTime: 0.1, state: { x: 0 }, controls: { u: 0 } })).toBeNull();
    expect(miner.emittedCount()).toBe(0);
  });
});
