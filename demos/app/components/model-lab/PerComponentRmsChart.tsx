'use client';

// Position RMS is the headline number, but a model can be wrong in many
// distinct ways — bad heading, lagged speed, oversteer (yawRate),
// uncaptured sideslip (lateralVelocity). The training driver now emits
// per-component RMS via `perStateRmsFields` so we can show them next
// to each other.

import { BarChart, Bar, CartesianGrid, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import type { ModelDiagnostics } from 'kinocat/learning';

const UNITS: Record<string, string> = {
  heading: 'rad',
  speed: 'm/s',
  yawRate: 'rad/s',
  lateralVelocity: 'm/s',
};

export interface PerComponentRmsChartProps {
  diag: ModelDiagnostics | null;
}

export function PerComponentRmsChart({ diag }: PerComponentRmsChartProps) {
  if (!diag || diag.perStateRms.length === 0) {
    return (
      <div style={emptyStyle}>Per-component RMS appears after training completes.</div>
    );
  }
  const data = diag.perStateRms.map((r) => ({ name: r.name, rms: r.rms, unit: UNITS[r.name] ?? '' }));
  return (
    <div style={{ height: 180 }}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 4, right: 12, left: 4, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2735" />
          <XAxis dataKey="name" stroke="#cdd3de" fontSize={11} />
          <YAxis stroke="#cdd3de" fontSize={11} />
          <Tooltip
            contentStyle={{ background: '#141a26', border: '1px solid #1f2735', fontSize: 11 }}
            formatter={(v, _n, p) => {
              const unit = (p?.payload as { unit?: string } | undefined)?.unit ?? '';
              const num = typeof v === 'number' ? v : Number(v);
              return [`${num.toFixed(4)} ${unit}`, 'RMS'];
            }}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Bar dataKey="rms" name="per-component RMS" fill="#55dcff" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

const emptyStyle: React.CSSProperties = {
  padding: '14px 18px', borderRadius: 6,
  background: 'rgba(13, 17, 25, 0.85)', border: '1px solid #1f2735',
  opacity: 0.7, fontSize: 12,
};
