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
    // 30 s test timeout. `training-driver.test.ts` legitimately runs
    // ~18 s on a warm dev machine; CI runners under load have been
    // observed to push it past the previous 20 s ceiling, surfacing
    // as an `onTimeoutError` from vitest's RPC layer (intermittent CI
    // failure on PR #23). 30 s gives headroom without masking any
    // test that's actually broken.
    testTimeout: 30000,
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
