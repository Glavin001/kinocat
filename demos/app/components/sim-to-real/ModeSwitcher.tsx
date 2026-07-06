'use client';

import type { SimToRealMode } from './HUD';

export interface ModeSwitcherProps {
  mode: SimToRealMode;
  onChange: (m: SimToRealMode) => void;
  onReset: () => void;
  showFriction: boolean;
  onToggleFriction: (v: boolean) => void;
  showUncertainty: boolean;
  onToggleUncertainty: (v: boolean) => void;
  matchSubsteps: boolean;
  onToggleSubsteps: (v: boolean) => void;
  onCopyMarkdown: () => void;
  onCopyJson: () => void;
  onDownloadJson: () => void;
  /** Optional toast text shown briefly after a copy. */
  toast?: string | null;
  /** Ordered list of visibility IDs ('real' + ghost ids). */
  ghostIds?: string[];
  /** Current visibility per ID. */
  visibility?: Record<string, boolean>;
  /** Toggle a single ID on/off. */
  onToggleVisibility?: (id: string) => void;
}

const GHOST_LABELS: Record<string, string> = {
  'real': 'Real car (white)',
  'v2-full': 'v2 (parametric + residual)',
  'v2-default': 'v2-default.json (preloaded)',
  'v2-trained': 'v2-trained (localStorage)',
  'parametric': 'parametric-only',
  'kinematic': 'kinematic',
};

const GHOST_COLORS: Record<string, string> = {
  'real': '#ffffff',
  'v2-full': '#4dd2ff',
  'v2-default': '#4dd2ff',
  'v2-trained': '#55ff99',
  'parametric': '#b47bff',
  'kinematic': '#ffcc55',
};

const modes: { id: SimToRealMode; label: string; hint: string }[] = [
  { id: 'playback', label: 'Playback', hint: 'Replay a recorded trial; open-loop model vs Rapier (Gap A)' },
  { id: 'free-drive', label: 'Free Drive', hint: 'WASD; ghosts predict 1s ahead (Gap A, interactive)' },
  { id: 'plan-execute', label: 'Plan & Execute', hint: 'Click goal; pure-pursuit drives the plan (Gap B)' },
];

export function ModeSwitcher(props: ModeSwitcherProps) {
  return (
    <div style={containerStyle}>
      <div style={{ display: 'flex', gap: 4, marginBottom: 8 }}>
        {modes.map((m) => (
          <button
            key={m.id}
            onClick={() => props.onChange(m.id)}
            style={{
              ...btnStyle,
              background: props.mode === m.id ? '#2b6cff' : 'rgba(255,255,255,0.06)',
              fontWeight: props.mode === m.id ? 600 : 400,
            }}
            title={m.hint}
          >
            {m.label}
          </button>
        ))}
        <button onClick={props.onReset} style={{ ...btnStyle, background: 'rgba(255,255,255,0.06)' }}>
          Reset (R)
        </button>
      </div>
      {props.ghostIds && props.ghostIds.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, fontSize: 11, marginBottom: 6 }}>
          {props.ghostIds.map((id) => (
            <label key={id} style={{ cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 3, color: GHOST_COLORS[id] ?? '#e6e9ee' }}>
              <input
                type="checkbox"
                checked={props.visibility?.[id] !== false}
                onChange={() => props.onToggleVisibility?.(id)}
              />
              <span>{GHOST_LABELS[id] ?? id}</span>
            </label>
          ))}
        </div>
      )}
      <div style={{ display: 'flex', gap: 14, fontSize: 12, opacity: 0.85, marginBottom: 6 }}>
        <Toggle label="Friction circles" v={props.showFriction} onChange={props.onToggleFriction} />
        <Toggle label="Uncertainty cloud" v={props.showUncertainty} onChange={props.onToggleUncertainty} />
        <Toggle label="Match Rapier sub-steps" v={props.matchSubsteps} onChange={props.onToggleSubsteps} />
      </div>
      <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
        <button onClick={props.onCopyMarkdown} style={btnStyle} title="Copy a markdown summary + JSON tail of the last ~10s">
          Copy debug (Markdown)
        </button>
        <button onClick={props.onCopyJson} style={btnStyle} title="Copy the full JSON snapshot to the clipboard">
          Copy JSON
        </button>
        <button onClick={props.onDownloadJson} style={btnStyle} title="Download the full JSON snapshot as a file">
          Download JSON
        </button>
        {props.toast && (
          <span style={{ fontSize: 11, opacity: 0.9, color: '#9cf', marginLeft: 6 }}>{props.toast}</span>
        )}
      </div>
    </div>
  );
}

function Toggle({ label, v, onChange }: { label: string; v: boolean; onChange: (b: boolean) => void }) {
  return (
    <label style={{ cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 4 }}>
      <input type="checkbox" checked={v} onChange={(e) => onChange(e.target.checked)} />
      <span>{label}</span>
    </label>
  );
}

const containerStyle: React.CSSProperties = {
  position: 'fixed',
  top: 16,
  left: 16,
  padding: '10px 12px',
  background: 'rgba(10, 14, 22, 0.82)',
  color: '#e6e9ee',
  border: '1px solid rgba(255,255,255,0.12)',
  borderRadius: 8,
  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
  fontSize: 13,
  zIndex: 10,
  backdropFilter: 'blur(4px)',
};

const btnStyle: React.CSSProperties = {
  padding: '5px 10px',
  borderRadius: 6,
  border: '1px solid rgba(255,255,255,0.12)',
  color: '#fff',
  cursor: 'pointer',
  fontSize: 12,
  fontFamily: 'inherit',
};
