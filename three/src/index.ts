// Foundation stub for @kinocat/three.
//
// This is NOT the kinocat library. It exists only to prove the monorepo
// build system: a workspace package that depends on three.js and is
// consumed by the demos app across a workspace:* link. The real three.js
// debug helpers (plan / primitive / jump-trajectory visualizers, per
// README section 6) will replace this file later.

import * as THREE from 'three';

export const KINOCAT_THREE_VERSION = '0.0.0';

/**
 * Returns the simplest possible kinocat-themed mesh: a unit cube with a
 * normal material. Placeholder for future debug-helper factories.
 */
export function createHelloCube(): THREE.Mesh {
  const geometry = new THREE.BoxGeometry(1, 1, 1);
  const material = new THREE.MeshNormalMaterial();
  return new THREE.Mesh(geometry, material);
}
