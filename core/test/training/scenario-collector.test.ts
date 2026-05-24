import { describe, expect, it } from 'vitest';
import { createScenarioCollector } from 'kinocat/training';

interface S { x: number }
interface C { u: number }

describe('createScenarioCollector', () => {
  it('emits non-overlapping trials at each window boundary', () => {
    const collector = createScenarioCollector<S, C, unknown>({
      scenarioId: 'unit',
      dt: 0.1,
      sampleEveryNTicks: 1,
      windowSec: 0.5, // 5 ticks per window
      config: undefined,
      configKey: 'A',
    });
    const trials = [];
    let simTime = 0;
    for (let i = 0; i < 20; i++) {
      const t = collector.record(simTime, { x: i }, { u: 0 }, { x: i + 1 });
      simTime += 0.1;
      if (t) trials.push(t);
    }
    // 20 ticks / 5 per window = 4 trials.
    expect(trials.length).toBe(4);
    for (const t of trials) {
      expect(t.controlsTrace.length).toBe(5);
      // initialState should equal samples[0].state.
      expect(t.initialState).toEqual(t.samples[0]!.state);
      expect(t.scenarioId).toBe('unit');
      expect(t.maneuverId).toBe('scenario');
      expect(t.split).toBeDefined();
    }
  });

  it('flush emits the partial window', () => {
    const collector = createScenarioCollector<S, C, unknown>({
      scenarioId: 'unit',
      dt: 0.1,
      sampleEveryNTicks: 1,
      windowSec: 10, // never auto-emit in this test
      config: undefined,
      configKey: 'A',
    });
    for (let i = 0; i < 5; i++) {
      collector.record(i * 0.1, { x: i }, { u: 0 }, { x: i + 1 });
    }
    const t = collector.flush();
    expect(t).not.toBeNull();
    expect(t!.controlsTrace.length).toBe(5);
  });

  it('flush returns null when buffer is empty', () => {
    const collector = createScenarioCollector<S, C, unknown>({
      scenarioId: 'unit',
      dt: 0.1,
      sampleEveryNTicks: 1,
      windowSec: 1,
      config: undefined,
      configKey: 'A',
    });
    expect(collector.flush()).toBeNull();
  });
});
