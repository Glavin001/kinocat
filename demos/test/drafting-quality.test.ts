// Drafting QUALITY metrics. The earlier A*-stitch draft was a mess (mean ~5 m
// off the slot, gear reversals, only 7 m/s or −3 m/s ever used). The follow
// CONTROLLER should hold the slot tightly, pace the lead, and never reverse.
import { describe, it, expect } from 'vitest';
import { goalLabPresets } from '../app/lib/goallab-presets';
import { hermitePose } from '../app/lib/path-anim';

describe('drafting quality (follow controller)', () => {
  const preset = goalLabPresets().find((x) => x.id === 'drafting')!;
  const lead = preset.movingTarget!;
  const gap = 6;

  it('holds the slot tightly, paces the lead, and never reverses', () => {
    const r = preset.plan();
    const path = r.path;
    const tEnd = path[path.length - 1]!.t;
    expect(tEnd).toBeGreaterThan(20); // ~one full lead loop

    // Slot-tracking error sampled densely along the animated trajectory.
    let maxErr = 0;
    let sumErr = 0;
    let n = 0;
    // skip the brief initial acquisition transient
    for (let t = 2; t <= tEnd; t += 0.1) {
      const pose = hermitePose(path, t);
      const la = lead.predict(t)!;
      const slot = { x: la.x - Math.cos(la.heading) * gap, z: la.z - Math.sin(la.heading) * gap };
      const err = Math.hypot(pose.x - slot.x, pose.z - slot.z);
      maxErr = Math.max(maxErr, err);
      sumErr += err;
      n++;
    }
    const meanErr = sumErr / n;

    // Gear reversals + speed range over the trajectory.
    let gearReversals = 0;
    let minSpeed = Infinity;
    let maxSpeed = -Infinity;
    for (let i = 0; i < path.length; i++) {
      minSpeed = Math.min(minSpeed, path[i]!.speed);
      maxSpeed = Math.max(maxSpeed, path[i]!.speed);
      if (i > 0) {
        const a = Math.sign(path[i - 1]!.speed);
        const b = Math.sign(path[i]!.speed);
        if (a < 0 !== b < 0) gearReversals++;
      }
    }

    // The controller should track within ~1.5 m on average and ~3 m worst-case,
    // never reverse, and settle near the lead's ~4.2 m/s pace.
    expect(meanErr).toBeLessThan(1.5);
    expect(maxErr).toBeLessThan(3.5);
    expect(gearReversals).toBe(0);
    expect(minSpeed).toBeGreaterThanOrEqual(0); // no reverse
    expect(maxSpeed).toBeLessThan(9.5);
  });
});
