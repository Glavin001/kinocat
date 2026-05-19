import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  createPlanPathHelper,
  createFootprintHelper,
  createMotionPrimitiveHelper,
} from '../../src/adapters/three/index';

describe('three debug helpers', () => {
  it('createPlanPathHelper builds a Line with one vertex per path point', () => {
    const path = [
      { x: 0, z: 0 },
      { x: 1, z: 2 },
      { x: 4, z: 2 },
    ];
    const line = createPlanPathHelper(path, { y: 0.1, color: 0x123456 });
    expect(line).toBeInstanceOf(THREE.Line);
    expect(line.geometry.getAttribute('position').count).toBe(3);
  });

  it('createFootprintHelper builds a closed loop', () => {
    const fp: [number, number][] = [
      [0, 0],
      [2, 0],
      [2, 1],
      [0, 1],
    ];
    const loop = createFootprintHelper(fp);
    expect(loop).toBeInstanceOf(THREE.LineLoop);
    expect(loop.geometry.getAttribute('position').count).toBe(4);
  });

  it('createMotionPrimitiveHelper builds one line per primitive', () => {
    const lib = {
      primitives: [
        { sweep: [{ x: 0, z: 0 }, { x: 1, z: 0 }] },
        { sweep: [{ x: 0, z: 0 }, { x: 0, z: 1 }, { x: 1, z: 1 }] },
      ],
    };
    const group = createMotionPrimitiveHelper(lib);
    expect(group).toBeInstanceOf(THREE.Group);
    expect(group.children.length).toBe(2);
    const first = group.children[0] as THREE.Line;
    expect(first.geometry.getAttribute('position').count).toBe(2);
  });
});
