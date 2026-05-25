import { describe, expect, it } from 'vitest';
import { createTrialStore, serializeTrials, deserializeTrials } from 'kinocat/learning';

describe('TrialStore — serialize / deserialize round trip', () => {
  it('preserves trial content byte-for-byte after one round trip', () => {
    const store = createTrialStore<{ x: number; v: number }, { e: number }, { id: string }>();
    store.add({
      id: 't1', initialState: { x: 0, v: 0 }, controlsTrace: [{ e: 0.5 }, { e: 0.7 }],
      dt: 1 / 60, samples: [{ t: 0, state: { x: 0, v: 0 } }, { t: 1 / 30, state: { x: 0.1, v: 0.6 } }],
      config: { id: 'cfgA' }, configKey: 'cfgA',
    });
    store.add({
      id: 't2', initialState: { x: 0, v: 0 }, controlsTrace: [{ e: -0.3 }],
      dt: 1 / 60, samples: [{ t: 0, state: { x: 0, v: 0 } }],
      config: { id: 'cfgB' }, configKey: 'cfgB',
    });

    const json = serializeTrials(store);
    const restored = deserializeTrials<{ x: number; v: number }, { e: number }, { id: string }>(json);
    expect(restored.size()).toBe(2);
    expect(restored.all()).toEqual(store.all());
    expect(restored.byConfig('cfgB')).toHaveLength(1);
  });

  it('rejects unknown version', () => {
    const bad = JSON.stringify({ version: 99, trials: [] });
    expect(() => deserializeTrials(bad)).toThrow(/version/);
  });
});
