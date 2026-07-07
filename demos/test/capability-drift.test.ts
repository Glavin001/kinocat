// Containment invariants between the demo agents' assumed capability
// envelopes and the physical plant's derived envelope:
//
//   planner envelope ⊂ tracker envelope ⊂ plant envelope
//
// A planner that assumes a tighter turn than the chassis can execute
// emits arcs no tracker can follow — the resulting steady lateral
// residual is the documented root of the parking shunt cycle (E2 in the
// production-readiness review). Known-inverted cases stay red via
// `it.fails` until the owning fix lands.

import { describe, expect, it } from 'vitest';
import {
  DEFAULT_LEARNABLE_CONFIG,
  deriveVehicleCapabilities,
} from 'kinocat/agent';
import { RACE_AGENT } from '../app/lib/race-primitives-scenarios';
import { PARKING_AGENT } from '../app/lib/parking-scenarios';

const plant = deriveVehicleCapabilities(DEFAULT_LEARNABLE_CONFIG);

describe('agent capability drift vs derived plant envelope', () => {
  // MEASURED INVERSION, kept deliberately (see the agents' comments):
  // planning slightly tighter than the plant can execute is intentional
  // over-command that closed-loop feedback exploits. Enforcing strict
  // containment (radius ≈ 4.91 m from plannerVehicleCapabilities)
  // REGRESSED reality: kinematic lap 32.6 → 49.7 s, v2 DNF, parking
  // reverse-perp invariant broken (stall geometry authored for 3.5 m).
  // These stay `it.fails` as an honest record of the gap between the
  // containment ideal and today's feedback-only executor; flip to `it`
  // when curvature feedforward makes feasible-radius plans the faster
  // ones — verify via closed-loop-race-benchmark + parking invariants.
  it.fails('race agent plans at a turn radius the plant can execute', () => {
    expect(RACE_AGENT.minTurnRadius).toBeGreaterThanOrEqual(plant.minTurnRadius);
  });

  it.fails('parking agent plans at a turn radius the plant can execute', () => {
    expect(PARKING_AGENT.minTurnRadius).toBeGreaterThanOrEqual(plant.minTurnRadius);
  });

  it('race agent max speed is a deliberate policy cap (plant has no intrinsic ceiling)', () => {
    // CORRECTED (WS-0, plant-envelope.test.ts): 30 m/s is NOT a physical
    // ceiling. Rapier models no aerodynamic drag, so under full drive force
    // the chassis keeps accelerating (~11 m/s² even at 28 m/s; terminal
    // speed in a 12 s launch is ~97 m/s). `RACE_AGENT.maxSpeed` is therefore
    // a POLICY choice — the planner's action-space ceiling — not a plant
    // limit. Pinned so raising it is a conscious edit that also extends the
    // primitive lattice (RACE_START_SPEEDS + control sets), never a silent
    // drift.
    expect(RACE_AGENT.maxSpeed).toBeLessThanOrEqual(30);
  });
});
