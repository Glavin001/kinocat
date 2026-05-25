// Unit tests for the generic DebugRecorder. Uses a synthetic 1-D state /
// control type to prove the recorder is domain-agnostic.

import { describe, expect, it } from 'vitest';

import { DebugRecorder } from '../../src/diagnostics';

interface S1D { x: number; v: number; t: number }
interface C1D { a: number }

const formatters = {
  formatState: (s: S1D) => ({ x: s.x, v: s.v, t: s.t }),
  formatControls: (c: C1D) => ({ a: c.a }),
  gapMetrics: (real: S1D, pred: S1D) => ({
    dx: pred.x - real.x,
    dv: pred.v - real.v,
  }),
};

describe('DebugRecorder', () => {
  it('rings the buffer at capacity', () => {
    const rec = new DebugRecorder<S1D, C1D>({ capacity: 2, formatters });
    rec.capture({ simTime: 0, real: { x: 0, v: 0, t: 0 }, controls: { a: 0 }, ghosts: [] });
    rec.capture({ simTime: 1, real: { x: 1, v: 1, t: 1 }, controls: { a: 0 }, ghosts: [] });
    rec.capture({ simTime: 2, real: { x: 2, v: 2, t: 2 }, controls: { a: 0 }, ghosts: [] });
    expect(rec.size()).toBe(2);
    expect(rec.frames()[0]!.simTime).toBe(1);
    expect(rec.frames()[1]!.simTime).toBe(2);
  });

  it('aggregates per-ghost mean + RMS gaps', () => {
    const rec = new DebugRecorder<S1D, C1D>({ formatters });
    const real0: S1D = { x: 0, v: 0, t: 0 };
    const ghost0: S1D = { x: 1, v: 0, t: 0 }; // dx=+1
    const real1: S1D = { x: 0, v: 0, t: 1 };
    const ghost1: S1D = { x: -1, v: 0, t: 1 }; // dx=-1
    rec.capture({ simTime: 0, real: real0, controls: { a: 0 }, ghosts: [{ name: 'g', state: ghost0 }] });
    rec.capture({ simTime: 1, real: real1, controls: { a: 0 }, ghosts: [{ name: 'g', state: ghost1 }] });
    const s = rec.stats();
    // dx values: +1, -1 -> mean=0, rms=1
    const dx = s.perGhost['g']!['dx']!;
    expect(dx.mean).toBe(0);
    expect(dx.rms).toBe(1);
    expect(dx.final).toBe(-1);
  });

  it('computes controls range', () => {
    const rec = new DebugRecorder<S1D, C1D>({ formatters });
    rec.capture({ simTime: 0, real: { x: 0, v: 0, t: 0 }, controls: { a: -3 }, ghosts: [] });
    rec.capture({ simTime: 1, real: { x: 0, v: 0, t: 1 }, controls: { a: 7 }, ghosts: [] });
    rec.capture({ simTime: 2, real: { x: 0, v: 0, t: 2 }, controls: { a: 0 }, ghosts: [] });
    const s = rec.stats();
    expect(s.controlsRange['a']).toEqual({ min: -3, max: 7 });
  });

  it('exports JSON and Markdown with the meta + stats + frames', () => {
    const rec = new DebugRecorder<S1D, C1D>({ formatters });
    rec.capture({ simTime: 0, real: { x: 0, v: 0, t: 0 }, controls: { a: 1 }, ghosts: [{ name: 'g', state: { x: 0.1, v: 0, t: 0 } }] });
    const json = rec.toJSON({ mode: 'unit-test', physicsDt: 1 });
    const obj = JSON.parse(json);
    expect(obj.meta.mode).toBe('unit-test');
    expect(obj.frames.length).toBe(1);
    expect(obj.frames[0].real.x).toBe(0);
    expect(obj.frames[0].ghosts.g.x).toBe(0.1);
    const md = rec.toMarkdown({ mode: 'unit-test', physicsDt: 1 }, 5);
    expect(md).toContain('# Debug snapshot — unit-test');
    expect(md).toContain('Per-ghost gap');
  });

  it('attaches per-frame extras', () => {
    const rec = new DebugRecorder<S1D, C1D>({ formatters });
    rec.attachExtras({ wheelImpulse: [10, 20] });
    rec.capture({ simTime: 0, real: { x: 0, v: 0, t: 0 }, controls: { a: 0 }, ghosts: [] });
    rec.capture({ simTime: 1, real: { x: 0, v: 0, t: 1 }, controls: { a: 0 }, ghosts: [] });
    expect(rec.frames()[0]!.extras).toEqual({ wheelImpulse: [10, 20] });
    expect(rec.frames()[1]!.extras).toBeUndefined();
  });
});
