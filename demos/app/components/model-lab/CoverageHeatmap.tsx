'use client';

// (speedBin, steerBin) heatmap of held-out RMS error, with sample
// counts. Wired to ModelDiagnostics.coverage which the training driver
// populates via `withCellBinning`. Clicking a cell jumps the fan plot
// to the matching start-speed bucket and (optionally) hints which
// steer band to inspect.
//
// Cell key format from training-driver: `s{0..3}-t{0..2}`
//   speedBin: 0 = ≤2 m/s, 1 = ≤6, 2 = ≤10, 3 = >10
//   steerBin: 0 = steer<-0.1, 1 = ≈0, 2 = steer>0.1

import type { ModelDiagnostics, CoverageCell } from 'kinocat/learning';

export interface CoverageHeatmapProps {
  diag: ModelDiagnostics | null;
  onSelect?: (speedBin: number, steerBin: number) => void;
}

const SPEED_LABELS = ['0–2', '2–6', '6–10', '10+'];
const STEER_LABELS = ['left', 'straight', 'right'];

function parseKey(id: string): { speedBin: number; steerBin: number } | null {
  const m = /^s(\d+)-t(\d+)$/.exec(id);
  if (!m) return null;
  return { speedBin: Number(m[1]), steerBin: Number(m[2]) };
}

function colorFor(value: number, max: number): string {
  if (max <= 0) return '#1f2735';
  const t = Math.min(1, value / max);
  // Cool (low) -> hot (high): #163040 -> #ff5566
  const r = Math.round(22 + (255 - 22) * t);
  const g = Math.round(48 + (85 - 48) * t);
  const b = Math.round(64 + (102 - 64) * t);
  return `rgb(${r},${g},${b})`;
}

export function CoverageHeatmap({ diag, onSelect }: CoverageHeatmapProps) {
  const cells = diag?.coverage ?? [];
  if (cells.length === 0) {
    return (
      <div style={emptyStyle}>
        Coverage heatmap appears after training completes. Each cell shows the
        held-out error and how many trials landed in that (speed, steer) bin —
        red = where the model is worst, dark cells = under-explored.
      </div>
    );
  }
  const maxErr = Math.max(0.001, ...cells.map((c) => c.errorRms));
  const grid: (CoverageCell | null)[][] = Array.from({ length: 4 }, () => [null, null, null]);
  for (const c of cells) {
    const k = parseKey(c.binId);
    if (!k) continue;
    if (grid[k.speedBin]) grid[k.speedBin]![k.steerBin] = c;
  }
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div style={{
        display: 'grid',
        gridTemplateColumns: '60px repeat(3, minmax(0, 1fr))',
        gridTemplateRows: '20px repeat(4, minmax(60px, 1fr))',
        gap: 2,
        fontSize: 11,
      }}>
        <div />
        {STEER_LABELS.map((l) => (
          <div key={l} style={headerStyle}>{l}</div>
        ))}
        {SPEED_LABELS.map((sl, si) => (
          <span key={sl} style={{ display: 'contents' }}>
            <div style={{ ...headerStyle, justifyContent: 'flex-end', paddingRight: 6 }}>{sl}&nbsp;m/s</div>
            {STEER_LABELS.map((_, ti) => {
              const c = grid[si]?.[ti] ?? null;
              const bg = c ? colorFor(c.errorRms, maxErr) : '#0d1119';
              return (
                <button
                  key={`${si}-${ti}`}
                  onClick={() => onSelect?.(si, ti)}
                  style={{
                    background: bg,
                    border: '1px solid #1f2735',
                    color: '#cdd3de',
                    fontFamily: 'inherit',
                    fontSize: 11,
                    padding: 6,
                    textAlign: 'left',
                    cursor: onSelect ? 'pointer' : 'default',
                    minHeight: 60,
                  }}
                  title={c ? `${c.errorRms.toFixed(3)} m · ${c.count} samples` : 'no trials'}
                  disabled={!c}
                >
                  {c ? (
                    <>
                      <div style={{ fontWeight: 700 }}>{c.errorRms.toFixed(2)} m</div>
                      <div style={{ opacity: 0.7 }}>n={c.count}</div>
                    </>
                  ) : (
                    <div style={{ opacity: 0.4 }}>—</div>
                  )}
                </button>
              );
            })}
          </span>
        ))}
      </div>
      <div style={{ fontSize: 10, opacity: 0.5 }}>
        max RMS: {maxErr.toFixed(2)} m · darker = lower error / under-sampled · brighter red = high error
      </div>
    </div>
  );
}

const headerStyle: React.CSSProperties = {
  display: 'flex', alignItems: 'center', justifyContent: 'center',
  fontSize: 10, opacity: 0.7,
};

const emptyStyle: React.CSSProperties = {
  padding: '14px 18px', borderRadius: 6,
  background: 'rgba(13, 17, 25, 0.85)', border: '1px solid #1f2735',
  opacity: 0.7, fontSize: 12,
};
