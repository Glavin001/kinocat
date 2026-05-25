'use client';

// Two views in one: (1) a small table of per-round RMS / trials so the
// user can read off concrete numbers, and (2) a line chart of the
// open-loop divergence at multiple horizons across rounds. Together
// they make "is training actually helping?" answerable at a glance —
// previously you only saw the final round's bar chart.

import { useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import type { RoundSnapshot } from '../../lib/model-lab-store';

export interface RoundEvolutionTableProps {
  rounds: RoundSnapshot[];
}

function at(rows: { tSec: number; posRms: number }[] | undefined, t: number): number | null {
  if (!rows) return null;
  return rows.find((r) => r.tSec >= t)?.posRms ?? null;
}

export function RoundEvolutionTable({ rounds }: RoundEvolutionTableProps) {
  const data = useMemo(() => rounds.map((r) => ({
    round: r.round + 1,
    rms05: at(r.diagnostics.openLoopDivergence, 0.5) ?? 0,
    rms10: at(r.diagnostics.openLoopDivergence, 1.0) ?? 0,
    rms16: at(r.diagnostics.openLoopDivergence, 1.6) ?? 0,
    parametricOnly: at(r.diagnostics.baselines['parametricOnly'], 1.0) ?? 0,
    legacy: at(r.diagnostics.baselines['legacyV1'], 1.0) ?? 0,
    trials: r.trialsAfter,
  })), [rounds]);

  if (rounds.length === 0) {
    return (
      <div style={emptyStyle}>
        Round-by-round evolution appears once training starts. Each round expands
        the training set with active exploration and refits.
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div style={{ height: 200 }}>
        <ResponsiveContainer>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2735" />
            <XAxis dataKey="round" stroke="#cdd3de" fontSize={11} label={{ value: 'round', position: 'insideBottom', offset: -2, fill: '#cdd3de', fontSize: 11 }} />
            <YAxis stroke="#cdd3de" fontSize={11} label={{ value: 'pos RMS (m)', angle: -90, position: 'insideLeft', fill: '#cdd3de', fontSize: 11 }} />
            <Tooltip contentStyle={{ background: '#141a26', border: '1px solid #1f2735', fontSize: 11 }} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Line type="monotone" dataKey="rms05" name="v2 @ 0.5s" stroke="#a6e9ff" dot strokeWidth={1.5} />
            <Line type="monotone" dataKey="rms10" name="v2 @ 1.0s" stroke="#55dcff" dot strokeWidth={2} />
            <Line type="monotone" dataKey="rms16" name="v2 @ 1.6s" stroke="#7fd6ff" dot strokeWidth={1.5} />
            <Line type="monotone" dataKey="parametricOnly" name="parametric-only @ 1.0s" stroke="#ffd070" dot strokeDasharray="4 3" strokeWidth={1.5} />
            <Line type="monotone" dataKey="legacy" name="legacy @ 1.0s" stroke="#ff8aa0" dot strokeDasharray="4 3" strokeWidth={1.5} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table style={tableStyle}>
          <thead>
            <tr>
              <Th>round</Th>
              <Th align="right">trials</Th>
              <Th align="right">RMS @0.5s</Th>
              <Th align="right">RMS @1.0s</Th>
              <Th align="right">RMS @1.6s</Th>
              <Th align="right">parametric-only @1.0s</Th>
              <Th align="right">residual gain</Th>
            </tr>
          </thead>
          <tbody>
            {data.map((row, i) => {
              const gain = row.parametricOnly > 0 ? (1 - row.rms10 / row.parametricOnly) * 100 : 0;
              return (
                <tr key={i}>
                  <Td>{row.round}</Td>
                  <Td align="right">{row.trials}</Td>
                  <Td align="right">{row.rms05.toFixed(3)}</Td>
                  <Td align="right" highlight>{row.rms10.toFixed(3)}</Td>
                  <Td align="right">{row.rms16.toFixed(3)}</Td>
                  <Td align="right">{row.parametricOnly ? row.parametricOnly.toFixed(3) : '—'}</Td>
                  <Td align="right" color={gain > 0 ? '#55ff88' : '#ff5566'}>
                    {gain ? `${gain >= 0 ? '+' : ''}${gain.toFixed(1)}%` : '—'}
                  </Td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Th({ children, align = 'left' }: { children: React.ReactNode; align?: 'left' | 'right' }) {
  return <th style={{ textAlign: align, opacity: 0.7, padding: '4px 8px', fontWeight: 500, fontSize: 11, borderBottom: '1px solid #1f2735' }}>{children}</th>;
}

function Td({ children, align = 'left', highlight, color }: { children: React.ReactNode; align?: 'left' | 'right'; highlight?: boolean; color?: string }) {
  return (
    <td style={{
      textAlign: align,
      padding: '4px 8px',
      color: color ?? (highlight ? '#cdeaff' : '#cdd3de'),
      fontWeight: highlight ? 700 : 400,
      borderBottom: '1px solid #141a26',
    }}>
      {children}
    </td>
  );
}

const tableStyle: React.CSSProperties = {
  width: '100%', borderCollapse: 'collapse', fontSize: 11,
  font: '11px ui-monospace, monospace',
  background: '#0d1119',
  borderRadius: 6,
  overflow: 'hidden',
};

const emptyStyle: React.CSSProperties = {
  padding: '14px 18px',
  borderRadius: 6,
  background: 'rgba(13, 17, 25, 0.85)',
  border: '1px solid #1f2735',
  opacity: 0.7,
  fontSize: 12,
};
