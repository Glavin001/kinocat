// Narrow typed surface of navcat that the adapter depends on. navcat is an
// OPTIONAL peer dependency — only consumers importing `kinocat/adapters/navcat`
// need it installed. Re-exporting the types here localizes the coupling and
// gives the type-pin test (adapters/navcat.test.ts) one place to assert
// against navcat's real .d.ts.

export type {
  Vec3,
  NavMesh,
  NodeRef,
  QueryFilter,
  FindNearestPolyResult,
  GetClosestPointOnPolyResult,
  RaycastResult,
  OffMeshConnectionParams,
} from 'navcat';

import type { OffMeshLink, PolygonRef } from '../../environment/nav-world';

/** Rich per-connection data stored alongside a navcat off-mesh connection,
 *  keyed by the navcat connection id. */
export interface StaticAffordanceMetadata {
  connectionId: number;
  link: OffMeshLink;
}

export interface JumpCandidate {
  from: readonly [number, number];
  to: readonly [number, number];
  cost?: number;
  kind?: PolygonRef extends never ? never : 'jump' | 'drop' | 'climb';
}
