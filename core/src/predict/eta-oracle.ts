// Cheap ETA / feasibility oracle for tactical layers (HTN operator costs):
// "can I reach region R, roughly how long?" in well under the ~2 ms budget —
// WITHOUT running a plan.
//
// The expensive part — a multi-source obstacle-aware distance field seeded
// from the region (`NavWorld.buildRegionLowerBound`) — is built once per
// (region, world revision) and LRU-cached; steady-state queries are one O(1)
// field lookup floored by the region's own admissible kinematic bound
// (`Region.costToGo`, Reeds-Shepp-aware), divided by the agent's max speed.
// Tactical layers that know their operator regions up front should call
// `prebuild` off the query path.
//
// `seconds` is an admissible LOWER bound on travel time (never an
// overestimate) — exactly what an HTN wants for operator cost/pruning.

import type { NavWorld } from '../environment/nav-world';
import type { Region, ScenarioState } from '../scenario/types';

export interface EtaResult {
  /** False is DEFINITIVE only when a distance field exists for the region:
   *  it means no obstacle-avoiding route connects the start to the region on
   *  the nav grid (or the start itself is off-grid). When the world exposes
   *  no `buildRegionLowerBound`, the region is dynamic, or it covers no
   *  walkable cell, the oracle cannot prove unreachability and reports
   *  `reachable: true` with the kinematic bound alone. */
  reachable: boolean;
  /** Admissible lower bound on travel time (s); Infinity when unreachable. */
  seconds: number;
}

export interface EtaOracle {
  eta(start: ScenarioState, region: Region): EtaResult;
  /** Build (and cache) the region's distance field off the query path.
   *  Returns true when a field is available for the region. */
  prebuild(region: Region): boolean;
}

type Lookup = (x: number, z: number, y?: number) => number | null;

export interface EtaOracleOptions {
  /** Max cached region fields (one Float32/64Array each). Default 8. */
  lruSize?: number;
}

export function createEtaOracle(
  world: NavWorld,
  agent: { maxSpeed: number },
  opts: EtaOracleOptions = {},
): EtaOracle {
  const lruSize = opts.lruSize ?? 8;
  // Insertion-ordered Map as LRU. Null values are cached too — a region with
  // no field (dynamic / off-mesh / unsupported world) shouldn't retry the
  // build on every query. Keyed by the region's stable structural key; the
  // whole cache drops when the world revision moves (a rebuilt tile changes
  // every field).
  const fields = new Map<string, Lookup | null>();
  let seenRevision = world.revision;

  function fieldFor(region: Region): Lookup | null {
    if (world.revision !== seenRevision) {
      fields.clear();
      seenRevision = world.revision;
    }
    const key = region.key;
    if (fields.has(key)) {
      const hit = fields.get(key)!;
      // Refresh recency.
      fields.delete(key);
      fields.set(key, hit);
      return hit;
    }
    let lookup: Lookup | null = null;
    // A dynamic region's membership moves over time — a static field would
    // be wrong by the time it's queried. Kinematic bound only.
    if (!region.dynamic && typeof world.buildRegionLowerBound === 'function') {
      const rep = region.representative();
      // Probe membership at cell centres; heading/speed/t come from the
      // representative pose so oriented conjuncts (e.g. `at`'s dheading)
      // don't reject purely positional membership.
      const probe: ScenarioState = {
        x: 0,
        z: 0,
        heading: rep.heading,
        speed: rep.speed,
        t: rep.t,
      };
      lookup = world.buildRegionLowerBound((x, z, cellHalfDiag = 0) => {
        probe.x = x;
        probe.z = z;
        if (region.contains(probe, rep.t)) return true;
        // A region smaller than a grid cell can fall between cell centres —
        // conservatively seed the cell holding the representative pose (the
        // ≤ half-diagonal shift is within the field's discretisation slack).
        return (
          cellHalfDiag > 0 &&
          Math.hypot(rep.x - x, rep.z - z) <= cellHalfDiag
        );
      });
    }
    fields.set(region.key, lookup);
    if (fields.size > lruSize) {
      const oldest = fields.keys().next().value!;
      fields.delete(oldest);
    }
    return lookup;
  }

  return {
    prebuild(region: Region): boolean {
      return fieldFor(region) !== null;
    },
    eta(start: ScenarioState, region: Region): EtaResult {
      const kinematicSec = region.costToGo(start, start.t) / agent.maxSpeed;
      const lookup = fieldFor(region);
      if (!lookup) return { reachable: true, seconds: kinematicSec };
      const d = lookup(start.x, start.z);
      if (d === null) return { reachable: false, seconds: Infinity };
      return {
        reachable: true,
        seconds: Math.max(d / agent.maxSpeed, kinematicSec),
      };
    },
  };
}
