'use client';

// Reusable scenario-goal progress visualizer. Given a compiled goal automaton
// and a deterministic progress snapshot (from `evaluateProgress`), it renders
// the phase bar + the automaton with the live state highlighted — the planner's
// internal objective state, made visible. Used by GoalLab and the upgraded
// scenario demos (e.g. /parking).

import type { CompiledAutomaton, ProgressSnapshot } from 'kinocat/scenario';

export interface GoalProgressPanelProps {
  automaton: CompiledAutomaton;
  snapshot: ProgressSnapshot;
  /** Optional one-line description of the goal (the AST sketch). */
  description?: string;
  /** Max automaton transitions to list (default 10). */
  maxRows?: number;
}

export function GoalProgressPanel({
  automaton,
  snapshot,
  description,
  maxRows = 10,
}: GoalProgressPanelProps) {
  const { q, depth, maxDepth, done, laps } = snapshot;
  const pct = maxDepth > 0 ? Math.round((depth / maxDepth) * 100) : done ? 100 : 0;
  const transitions: { from: number; to: number; label: string }[] = [];
  for (const st of automaton.states) {
    for (const tr of st.transitions) {
      transitions.push({ from: st.id, to: tr.target, label: tr.guard.region.kind });
    }
  }
  return (
    <div style={{ font: '12px ui-monospace, monospace', color: '#cfe', lineHeight: 1.5 }}>
      {description && <div style={{ color: '#9bd', marginBottom: 4 }}>{description}</div>}
      <div>
        phase <b>{depth}</b> / {maxDepth}
        {laps > 0 && <> · laps {laps}</>}
        {done && <span style={{ color: '#6f9' }}> · DONE ✓</span>}
      </div>
      <div style={{ height: 8, background: '#1a2330', borderRadius: 4, overflow: 'hidden', marginTop: 3 }}>
        <div style={{ width: `${pct}%`, height: '100%', background: done ? '#6f9' : '#4df' }} />
      </div>
      <div style={{ marginTop: 6, color: '#789' }}>automaton (current = q{q}):</div>
      {transitions.slice(0, maxRows).map((tr, i) => (
        <div key={i} style={{ color: tr.from === q ? '#4df' : '#566' }}>
          q{tr.from} →{' '}
          {tr.to === q ? <b style={{ color: '#4df' }}>q{tr.to}</b> : <>q{tr.to}</>} : {tr.label}
        </div>
      ))}
    </div>
  );
}
