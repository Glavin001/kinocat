'use client';

// HUD overlay for the /sim-to-real scope. Renders the real chassis
// state, per-ghost gap (instantaneous + rolling RMS), and a small
// friction summary. Updated each frame from the scene component via a
// ref-callback (we keep React out of the 60Hz loop).

import { forwardRef, useImperativeHandle, useState } from 'react';
import type { GapRms } from '../../lib/sim-to-real-scene';

export type SimToRealMode = 'playback' | 'free-drive' | 'plan-execute';

export interface HUDGhostEntry {
  label: string;
  color: number;
  posErr: number;
  headingErrDeg: number;
  speedErr: number;
  rolling: GapRms;
}

export interface HUDSnapshot {
  mode: SimToRealMode;
  status: string;
  real: { x: number; z: number; heading: number; speed: number; yawRate: number; lateralVelocity: number };
  ghosts: HUDGhostEntry[];
  maxGripPct: number;
  /** Mode-specific extras (e.g. plan-vs-actual at horizons). Plain text. */
  extra?: string[];
}

export interface HUDHandle {
  update(snapshot: HUDSnapshot): void;
}

function colorHex(c: number): string {
  return '#' + c.toString(16).padStart(6, '0');
}

export const HUD = forwardRef<HUDHandle>(function HUD(_, ref) {
  const [snap, setSnap] = useState<HUDSnapshot | null>(null);
  useImperativeHandle(ref, () => ({ update: setSnap }), []);
  if (!snap) {
    return (
      <div style={panelStyle}>
        <div style={{ opacity: 0.7 }}>Loading sim-to-real scope…</div>
      </div>
    );
  }
  return (
    <div style={panelStyle}>
      <div style={{ fontWeight: 600, marginBottom: 6 }}>
        Mode: <span style={{ color: '#9cf' }}>{labelForMode(snap.mode)}</span>
      </div>
      <div style={{ opacity: 0.85, marginBottom: 8, fontSize: 12 }}>{snap.status}</div>

      <Section title="Real chassis (Rapier)">
        <Row k="pos" v={`(${snap.real.x.toFixed(2)}, ${snap.real.z.toFixed(2)})`} />
        <Row k="heading" v={`${(snap.real.heading * 180 / Math.PI).toFixed(1)}°`} />
        <Row k="speed" v={`${snap.real.speed.toFixed(2)} m/s`} />
        <Row k="yawRate" v={`${(snap.real.yawRate * 180 / Math.PI).toFixed(1)}°/s`} />
        <Row k="lat vel" v={`${snap.real.lateralVelocity.toFixed(2)} m/s`} />
      </Section>

      <Section title="Sim-to-real gap (per model)">
        {snap.ghosts.length === 0 && (
          <div style={{ opacity: 0.6 }}>No active ghosts.</div>
        )}
        {snap.ghosts.map((g) => (
          <div key={g.label} style={{ marginBottom: 6 }}>
            <div style={{ color: colorHex(g.color), fontWeight: 600 }}>{g.label}</div>
            <Row k="Δpos" v={`${g.posErr.toFixed(2)} m`} />
            <Row k="Δheading" v={`${g.headingErrDeg.toFixed(1)}°`} />
            <Row k="Δspeed" v={`${g.speedErr.toFixed(2)} m/s`} />
            <Row
              k="RMS@2s"
              v={`p ${g.rolling.posRms.toFixed(2)} m · h ${(g.rolling.headingRms * 180 / Math.PI).toFixed(1)}° · v ${g.rolling.speedRms.toFixed(2)} m/s`}
            />
          </div>
        ))}
      </Section>

      <Section title="Tires">
        <Row
          k="max grip used"
          v={`${(snap.maxGripPct * 100).toFixed(0)}%`}
          warn={snap.maxGripPct > 1}
        />
      </Section>

      {snap.extra && snap.extra.length > 0 && (
        <Section title="Mode metrics">
          {snap.extra.map((e, i) => (
            <div key={i} style={{ fontSize: 12, opacity: 0.9 }}>{e}</div>
          ))}
        </Section>
      )}
    </div>
  );
});

function labelForMode(m: SimToRealMode): string {
  switch (m) {
    case 'playback': return 'Playback (Gap A)';
    case 'free-drive': return 'Free Drive (Gap A)';
    case 'plan-execute': return 'Plan & Execute (Gap B)';
  }
}

const panelStyle: React.CSSProperties = {
  position: 'fixed',
  top: 16,
  right: 16,
  width: 320,
  padding: '12px 14px',
  background: 'rgba(10, 14, 22, 0.82)',
  color: '#e6e9ee',
  border: '1px solid rgba(255,255,255,0.12)',
  borderRadius: 8,
  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
  fontSize: 13,
  zIndex: 10,
  backdropFilter: 'blur(4px)',
};

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 8, paddingTop: 6, borderTop: '1px solid rgba(255,255,255,0.08)' }}>
      <div style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: 0.5, opacity: 0.6, marginBottom: 3 }}>{title}</div>
      {children}
    </div>
  );
}

function Row({ k, v, warn }: { k: string; v: string; warn?: boolean }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
      <span style={{ opacity: 0.65 }}>{k}</span>
      <span style={{ color: warn ? '#ff8080' : '#e6e9ee' }}>{v}</span>
    </div>
  );
}
