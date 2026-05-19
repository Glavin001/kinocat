// Asserts the spec §15.8 budget: core + navcat-adapter < 100 KB minified.
// Peers (navcat/three/rapier) are externalized, so this measures only the
// bytes kinocat itself ships. Logs gzip too for a reality check.
import { gzipSync } from 'node:zlib';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const dist = join(here, '..', 'dist', 'sizecheck');

const targets = [
  ['core', join(dist, 'core.js')],
  ['core+navcat-adapter', join(dist, 'core-navcat.js')],
];

const LIMIT_MIN_BYTES = 100 * 1024;
let totalMin = 0;
let missing = false;

console.log('kinocat size check (peers externalized)\n');
for (const [label, file] of targets) {
  if (!existsSync(file)) {
    console.error(`  MISSING  ${label}: ${file} (run \`pnpm --filter kinocat build\`)`);
    missing = true;
    continue;
  }
  const buf = readFileSync(file);
  const min = buf.byteLength;
  const gz = gzipSync(buf).byteLength;
  totalMin += min;
  console.log(
    `  ${label.padEnd(22)} ${(min / 1024).toFixed(2)} KB min  /  ${(gz / 1024).toFixed(2)} KB gzip`,
  );
}

if (missing) process.exit(1);

console.log(`\n  total minified: ${(totalMin / 1024).toFixed(2)} KB (limit ${LIMIT_MIN_BYTES / 1024} KB)`);
if (totalMin > LIMIT_MIN_BYTES) {
  console.error(`\nFAIL: core + navcat-adapter exceeds ${LIMIT_MIN_BYTES / 1024} KB minified.`);
  process.exit(1);
}
console.log('\nOK');
