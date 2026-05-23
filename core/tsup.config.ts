import { defineConfig } from 'tsup';

const external = ['navcat', 'mathcat', '@dimforge/rapier3d-compat', 'three'];

const entry = {
  index: 'src/index.ts',
  'curves/index': 'src/curves/index.ts',
  'primitives/index': 'src/primitives/index.ts',
  'agent/index': 'src/agent/index.ts',
  'planner/index': 'src/planner/index.ts',
  'environment/index': 'src/environment/index.ts',
  'predict/index': 'src/predict/index.ts',
  'execute/index': 'src/execute/index.ts',
  'learning/index': 'src/learning/index.ts',
  'adapters/navcat/index': 'src/adapters/navcat/index.ts',
  'adapters/rapier/index': 'src/adapters/rapier/index.ts',
  'adapters/three/index': 'src/adapters/three/index.ts',
  'worker/index': 'src/worker/index.ts',
};

export default defineConfig([
  {
    entry,
    format: ['esm'],
    dts: true,
    sourcemap: true,
    clean: true,
    splitting: true,
    treeshake: true,
    target: 'es2022',
    external,
  },
  {
    // Size-gate-only artifact: minified core + core+navcat-adapter, peers external.
    entry: {
      'sizecheck/core': 'src/index.ts',
      'sizecheck/core-navcat': 'src/adapters/navcat/index.ts',
    },
    format: ['esm'],
    dts: false,
    sourcemap: false,
    clean: false,
    minify: true,
    splitting: false,
    treeshake: true,
    target: 'es2022',
    external,
  },
]);
