// Explicit, readable catalog of in-scope scenario expressions — each authored
// inline so the test doubles as documentation of "what the code looks like",
// and planned through the real ScenarioEnvironment product search to prove it
// works. Deterministic (expansion-bounded), so it is stable under coverage.

import { describe, it, expect } from 'vitest';
import { InMemoryNavWorld } from 'kinocat/environment';
import { planVehicleScenario } from 'kinocat/planner';
import type { CarKinematicState } from 'kinocat/agent';
import {
  defineScenario,
  reach,
  near,
  behind,
  cone,
  avoid,
  maintain,
  stayInside,
  distanceFrom,
  closingSpeed,
  gte,
  inRange,
  minTime,
} from 'kinocat/scenario';
import type { RegionAgent } from 'kinocat/scenario';
import { demoVehicle } from '../app/lib/scenarios';
import {
  authorDrafting,
  authorDraftingHold,
  planDrafting,
} from '../app/lib/scenario-goals';

const FIELD = { x0: -40, x1: 70, z0: -25, z1: 25 };

// ---------------------------------------------------------------------------
// DRAFTING — follow closely behind a MOVING car (the requested example).

describe('catalog: drafting behind a moving car', () => {
  // The lead cruises +x at 3 m/s; we want to sit ~6 m off its tail.
  const lead: RegionAgent = {
    id: 'lead',
    predict: (t) => ({ x: 10 + 3 * t, z: 0, heading: 0, speed: 3, t }),
  };
  const start: CarKinematicState = { x: -10, z: -10, heading: 0, speed: 0, t: 0 };

  it('the authored AST reads like the intent', () => {
    // This is the whole expression — objective + constraints + preference:
    const scenario = authorDrafting({ start, lead, gap: 6, tol: 2, safe: 2 });
    //   goal:       reach(behind(lead, 6, 2))          // get into the slipstream slot
    //   invariants: maintain(distanceFrom(lead, >= 2)) // never rear-end it
    //               stayInside(field)
    //   prefer:     minTime, smooth
    expect(scenario.goal.kind).toBe('reach');
    expect(scenario.invariants?.length).toBe(2);
    expect(scenario.agents?.[0]?.id).toBe('lead');
  });

  it('INTERCEPTS: ends up behind where the lead WILL be, not where it started', () => {
    const r = planDrafting({ start, lead, gap: 6, tol: 2, safe: 2 });
    expect(r.raw.found).toBe(true);

    const finalEgo = r.path[r.path.length - 1]!;
    const leadAtArrival = lead.predict(finalEgo.t)!;
    const slot = { x: leadAtArrival.x - 6, z: 0 };

    // Ego reached the (moving) slot, and is genuinely BEHIND the lead.
    expect(Math.hypot(finalEgo.x - slot.x, finalEgo.z - slot.z)).toBeLessThanOrEqual(2.5);
    expect(finalEgo.x).toBeLessThan(leadAtArrival.x);
    // Interception, not tail-chasing: the slot moved well past its t=0 position
    // (10-6=4), so a planner aiming at the *current* pose would land near x≈4.
    expect(finalEgo.x).toBeGreaterThan(6);
  });

  it('NEVER breaches the safe gap (the maintain invariant is enforced)', () => {
    const r = planDrafting({ start, lead, gap: 6, tol: 2, safe: 2 });
    for (const ego of r.path) {
      const a = lead.predict(ego.t)!;
      expect(Math.hypot(ego.x - a.x, ego.z - a.z)).toBeGreaterThanOrEqual(2 - 1e-6);
    }
  });

  it('SUSTAINED hold: drafts over a horizon, holding station within the band', () => {
    // Start already near the lead's path so the (time-augmented, more
    // expensive) repeat search reaches the slot cheaply + deterministically.
    const holdStart: CarKinematicState = { x: 2, z: -4, heading: 0, speed: 0, t: 0 };
    const r = planDrafting(
      { start: holdStart, lead, gap: 6, tol: 2.5, safe: 2 },
      { hold: true, horizonSeconds: 4, maxExpansions: 60_000 },
    );
    // A repeat (progress) objective never "completes" -> best-progress partial.
    expect(r.raw.found).toBe(true);
    expect(r.raw.partial).toBe(true);
    expect(r.path.length).toBeGreaterThan(2);

    // It actually gets into the draft slot at least once (drafts the lead)...
    const gotIntoSlot = r.path.some((ego) => {
      const a = lead.predict(ego.t)!;
      return Math.hypot(ego.x - (a.x - 6), ego.z - a.z) <= 4;
    });
    expect(gotIntoSlot).toBe(true);
    // ...and the safe-gap invariant holds for every committed state (the
    // tight match-pace band is `.while`-scoped to the slot, so it only binds
    // once drafting, not during the approach).
    for (const ego of r.path) {
      const a = lead.predict(ego.t)!;
      expect(Math.hypot(ego.x - a.x, ego.z - a.z)).toBeGreaterThanOrEqual(2 - 1e-6);
    }
  });

  it('hold-variant AST uses repeat + closingSpeed (match-pace) invariant', () => {
    const s = authorDraftingHold({ start, lead });
    expect(s.goal.kind).toBe('repeat');
    // distanceFrom band + closingSpeed band + stayInside.
    expect(s.invariants?.length).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// EVADE — reach safety while keeping away from a moving threat's vision cone.

describe('catalog: evade to safety past a threat cone', () => {
  it('reaches the exit while never entering the guard cone or breaching distance', () => {
    const guard: RegionAgent = {
      id: 'guard',
      // Stationary guard near the middle, looking +x.
      predict: (t) => ({ x: 0, z: 0, heading: 0, speed: 0, t }),
    };
    const start: CarKinematicState = { x: -30, z: -12, heading: 0, speed: 0, t: 0 };
    const scenario = defineScenario('evade', {
      start,
      agents: [guard],
      goal: reach(near({ x: 30, z: -12 }, 2.5)),
      invariants: [
        avoid(cone(guard, Math.PI / 6, 22)), // keep out of the vision wedge
        maintain(distanceFrom(guard, gte(6))),
        stayInside([
          [FIELD.x0, FIELD.z0],
          [FIELD.x1, FIELD.z0],
          [FIELD.x1, FIELD.z1],
          [FIELD.x0, FIELD.z1],
        ]),
      ],
      prefer: [minTime(1)],
    });

    const { agent, lib } = demoVehicle({ maxSpeed: 10 });
    const r = planVehicleScenario({
      start,
      goal: scenario.goal,
      invariants: scenario.invariants,
      prefer: scenario.prefer,
      world: new InMemoryNavWorld([{ id: 1, y: 0, ring: [
        [FIELD.x0, FIELD.z0], [FIELD.x1, FIELD.z0], [FIELD.x1, FIELD.z1], [FIELD.x0, FIELD.z1],
      ] }], []),
      agent,
      lib,
      envOptions: { posCell: 1, headingBuckets: 16, goalRadius: 2.5 },
      deadlineMs: Infinity,
      maxExpansions: 120_000,
    });

    expect(r.raw.found).toBe(true);
    const coneR = cone(guard, Math.PI / 6, 22);
    for (const ego of r.path) {
      expect(coneR.contains(ego, ego.t)).toBe(false); // never seen
      expect(Math.hypot(ego.x - 0, ego.z - 0)).toBeGreaterThanOrEqual(6 - 1e-6);
    }
    const last = r.path[r.path.length - 1]!;
    expect(Math.hypot(last.x - 30, last.z + 12)).toBeLessThanOrEqual(3);
    // closingSpeed is available too — assert the new primitive composes.
    expect(closingSpeed(guard, inRange(-99, 99)).contains(last, last.t)).toBe(true);
  });
});
