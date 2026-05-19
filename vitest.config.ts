import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';

const root = fileURLToPath(new URL('.', import.meta.url));

export default defineConfig({
  resolve: {
    // Headless demo tests import the `kinocat` package by its public subpaths;
    // resolve them to source so they run without a build and stay fast.
    alias: [
      { find: /^kinocat$/, replacement: `${root}core/src/index.ts` },
      { find: /^kinocat\/(.*)$/, replacement: `${root}core/src/$1/index.ts` },
    ],
  },
  test: {
    environment: 'node',
    include: ['core/test/**/*.test.ts', 'demos/test/**/*.test.ts'],
    testTimeout: 20000,
    coverage: {
      provider: 'v8',
      reporter: ['text', 'lcov', 'html'],
      include: ['core/src/**'],
      exclude: [
        'core/src/**/index.ts',
        'core/src/**/types.ts',
        'core/src/adapters/**',
        'core/src/environment/r2-environment.ts',
      ],
      thresholds: {
        lines: 80,
        functions: 80,
        branches: 80,
        statements: 80,
      },
    },
  },
});
