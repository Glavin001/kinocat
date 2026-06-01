// Render a CompiledAutomaton as a Mermaid stateDiagram-v2 (matching the design
// spec's worked-example diagram). Used by docs, tests, and the visualizer's
// schematic overlay. Deterministic given the automaton.

import type { CompiledAutomaton } from './automaton';

function label(region: { kind: string }): string {
  return region.kind;
}

export function toMermaid(automaton: CompiledAutomaton): string {
  const lines: string[] = ['stateDiagram-v2'];
  lines.push(`  [*] --> q${automaton.start}`);
  for (const st of automaton.states) {
    for (const tr of st.transitions) {
      lines.push(`  q${st.id} --> q${tr.target}: ${label(tr.guard.region)}`);
    }
  }
  for (const f of automaton.accepting) {
    lines.push(`  q${f} --> [*]`);
  }
  if (automaton.progress) {
    lines.push('  note right of q' + automaton.start + ': progress (repeat)');
  }
  return lines.join('\n');
}
