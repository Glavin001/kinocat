// Architecture-invariant tests. Codify "agnostic core" so future
// contributions can't accidentally leak Rapier or scenario specifics into
// the domain-agnostic core surface.
//
// Allowed exceptions:
//   - core/src/adapters/* (whole directory is the adapter layer)
//   - this test file itself
//   - changelog / comments referencing rapier inside adapter folders are OK
//     because the grep is scoped outside adapters

import { describe, expect, it } from 'vitest';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative, sep } from 'node:path';

const ROOT = join(__dirname, '..', 'src');

function walk(dir: string): string[] {
  const out: string[] = [];
  for (const ent of readdirSync(dir)) {
    const p = join(dir, ent);
    const st = statSync(p);
    if (st.isDirectory()) {
      out.push(...walk(p));
    } else if (ent.endsWith('.ts')) {
      out.push(p);
    }
  }
  return out;
}

function isInAdapter(p: string): boolean {
  const rel = relative(ROOT, p).split(sep).join('/');
  return rel.startsWith('adapters/');
}

const FORBIDDEN_TOKENS_IN_CORE = ['Rapier', 'rapier', 'race-primitives', 'carchase'];

// Identify comment lines — these are documentation, not coupling.
// Strip line comments and skip lines fully inside block comments.
// A token in a comment is OK; the constraint is on CODE only.
function stripComments(src: string): string[] {
  const out: string[] = [];
  let inBlock = false;
  for (const raw of src.split('\n')) {
    let line = raw;
    if (inBlock) {
      const end = line.indexOf('*/');
      if (end < 0) { out.push(''); continue; }
      line = line.slice(end + 2);
      inBlock = false;
    }
    // Collapse all block /* ... */ and line // ... comments on this line.
    while (true) {
      const bs = line.indexOf('/*');
      const ls = line.indexOf('//');
      if (bs >= 0 && (ls < 0 || bs < ls)) {
        const be = line.indexOf('*/', bs + 2);
        if (be < 0) { line = line.slice(0, bs); inBlock = true; break; }
        line = line.slice(0, bs) + ' ' + line.slice(be + 2);
        continue;
      }
      if (ls >= 0) { line = line.slice(0, ls); break; }
      break;
    }
    out.push(line);
  }
  return out;
}

describe('core/ remains domain-agnostic', () => {
  const files = walk(ROOT).filter((p) => !isInAdapter(p));

  it('no Rapier / race / carchase tokens leak into core CODE (outside adapters/)', () => {
    const offences: { file: string; token: string; line: number; snippet: string }[] = [];
    for (const f of files) {
      const codeLines = stripComments(readFileSync(f, 'utf-8'));
      for (let i = 0; i < codeLines.length; i++) {
        const line = codeLines[i]!;
        for (const tok of FORBIDDEN_TOKENS_IN_CORE) {
          if (line.includes(tok)) {
            offences.push({ file: relative(ROOT, f), token: tok, line: i + 1, snippet: line.trim().slice(0, 120) });
          }
        }
      }
    }
    if (offences.length > 0) {
      const msg = offences
        .map((o) => `  ${o.file}:${o.line} [${o.token}] ${o.snippet}`)
        .join('\n');
      throw new Error(`Forbidden domain-specific tokens in core CODE (move to adapters/ or rename):\n${msg}`);
    }
    expect(offences).toEqual([]);
  });

  it('no @dimforge/rapier3d-compat import in core (outside adapters/)', () => {
    const offences: { file: string; line: number }[] = [];
    for (const f of files) {
      const src = readFileSync(f, 'utf-8').split('\n');
      for (let i = 0; i < src.length; i++) {
        const line = src[i]!;
        if (/from ['"]@dimforge\/rapier3d-compat['"]/.test(line)) {
          offences.push({ file: relative(ROOT, f), line: i + 1 });
        }
      }
    }
    expect(offences).toEqual([]);
  });
});
