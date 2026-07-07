// Bake the generated (dispersion-designed, dense-bucket) v3 race library to a
// shipped JSON artifact. Building it at runtime rolls the v3 MLP through
// designControlSet for 15 speed buckets — ~10 s of synchronous compute, which
// froze the browser on v3 select. It's deterministic from the model, so bake
// it once here and load the artifact in the app instead.
//
// usage: npx tsx scripts/bake-v3-lib.mts
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildLearnedRaceLibraryV3 } from '../app/lib/race-primitives-scenarios';
import { v3FromJson } from 'kinocat/agent';

const root = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const model = v3FromJson(JSON.parse(readFileSync(resolve(root, 'demos/public/models/v3-default.json'), 'utf-8')));
const t0 = performance.now();
const lib = buildLearnedRaceLibraryV3(model, { generatedControls: true });
const out = resolve(root, 'demos/public/models/v3-generated-lib.json');
writeFileSync(out, lib.toJSON());
console.log(`baked ${lib.primitives.length} primitives in ${((performance.now() - t0) / 1000).toFixed(1)}s → ${out}`);
