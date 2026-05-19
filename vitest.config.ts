import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'node',
    include: ['core/test/**/*.test.ts'],
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
