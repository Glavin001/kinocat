import { describe, expect, it } from 'vitest';
import {
  assignSplit,
  createTrialStore,
  hashString,
  trialSplitKey,
  DEFAULT_SPLIT_POLICY,
  type Trial,
} from 'kinocat/learning';

interface S { x: number }
interface C { u: number }
interface Cfg { id: string }

function mk(id: string, maneuverId?: string, params?: Record<string, number>): Trial<S, C, Cfg> {
  return {
    id,
    initialState: { x: 0 },
    controlsTrace: [{ u: 0 }],
    dt: 1 / 60,
    samples: [{ t: 0, state: { x: 0 } }],
    config: { id: 'A' },
    configKey: 'A',
    maneuverId,
    maneuverParams: params,
  };
}

describe('trial-store split policy', () => {
  it('hashString is deterministic and non-zero for non-empty input', () => {
    expect(hashString('foo')).toBe(hashString('foo'));
    expect(hashString('foo')).not.toBe(hashString('bar'));
  });

  it('assignSplit is deterministic for the same key', () => {
    const a = mk('t1', 'ou', { sig: 0.2 });
    const b = mk('different-id-same-key', 'ou', { sig: 0.2 });
    expect(assignSplit(a)).toBe(assignSplit(b));
  });

  it('produces roughly the policy ratio at scale', () => {
    let train = 0, val = 0, test = 0;
    for (let i = 0; i < 4000; i++) {
      const t = mk(`t${i}`, 'ou', { idx: i });
      const s = assignSplit(t);
      if (s === 'train') train++;
      else if (s === 'val') val++;
      else test++;
    }
    expect(train / 4000).toBeGreaterThan(DEFAULT_SPLIT_POLICY.train - 0.05);
    expect(train / 4000).toBeLessThan(DEFAULT_SPLIT_POLICY.train + 0.05);
    expect(val / 4000).toBeGreaterThan(DEFAULT_SPLIT_POLICY.val - 0.05);
    expect(val / 4000).toBeLessThan(DEFAULT_SPLIT_POLICY.val + 0.05);
    const expectedTest = 1 - DEFAULT_SPLIT_POLICY.train - DEFAULT_SPLIT_POLICY.val;
    expect(test / 4000).toBeGreaterThan(expectedTest - 0.05);
    expect(test / 4000).toBeLessThan(expectedTest + 0.05);
  });

  it('TrialStore.all(split) filters by partition', () => {
    const store = createTrialStore<S, C, Cfg>();
    for (let i = 0; i < 50; i++) {
      const t = mk(`t${i}`, 'ou', { idx: i });
      t.split = assignSplit(t);
      store.add(t);
    }
    const total = store.all().length;
    const sum =
      store.all('train').length + store.all('val').length + store.all('test').length;
    expect(sum).toBe(total);
  });

  it('trialSplitKey reflects all keying fields', () => {
    const a = mk('t1', 'ou', { sig: 0.2 });
    const b = mk('t1', 'ou', { sig: 0.4 });
    expect(trialSplitKey(a)).not.toBe(trialSplitKey(b));
  });
});
