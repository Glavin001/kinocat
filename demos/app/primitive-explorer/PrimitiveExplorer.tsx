'use client';

// Side-by-side comparison of the kinematic and v2-learned motion-primitive
// libraries that drive the cars on /raceprimitives. Built to answer the
// question: "Why does the v2 car drive the course differently?"
//
// Both libraries are built from the IDENTICAL set of 76 controls (19 ×
// 4 start speeds). What differs is purely the predicted endpoint /
// sweep shape for each control. Comparing them side by side reveals
// exactly which controls the two models disagree about — and at high
// speed, the kinematic library plans tight curves the chassis can't
// actually take, while v2 plans the honest wider arcs.

import { useEffect, useMemo, useState } from 'react';
import type { MotionPrimitive } from 'kinocat/primitives';
import {
  buildKinematicLibrary,
  buildLearnedRaceLibraryV2,
  RACE_START_SPEEDS,
} from '../lib/race-primitives-scenarios';

// CSS colors matching the Three.js RACE_PALETTE (which uses 0x… hex
// numbers for Three.js material consumption). Keep these in sync.
const KINEMATIC_CSS = '#ff8aa0';
const LEARNED_CSS = '#55dcff';
const GATE_CSS = '#ffd070';
import { loadV2Model, loadV2ModelFromUrl, type PersistedV2Model } from '../lib/v2-model-persistence';
import { diagnoseLibrary, type LibraryDiagnostics } from '../lib/primitive-diagnostics';
import { PrimitiveFanPlot } from '../components/PrimitiveFanPlot';
import { PrimitiveOverlayPlot } from '../components/PrimitiveOverlayPlot';
import { useIsMobile } from '../lib/use-is-mobile';
import type { LearnedVehicleModel } from 'kinocat/agent';

export default function PrimitiveExplorer() {
  const isMobile = useIsMobile(820);
  const [selectedSpeed, setSelectedSpeed] = useState<number>(RACE_START_SPEEDS[2]!); // 20 m/s default — where divergence matters most
  const [v2Model, setV2Model] = useState<LearnedVehicleModel | null>(null);
  const [v2Meta, setV2Meta] = useState<PersistedV2Model['meta'] | null>(null);
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    const cached = loadV2Model();
    if (cached) {
      setV2Model(cached.model);
      setV2Meta(cached.meta);
      return;
    }
    // No cached model — fall back to the preloaded artifact the
    // `pnpm run train` CLI ships with the project.
    void loadV2ModelFromUrl().then((res) => {
      if (cancelled || !res) return;
      setV2Model(res.model);
      setV2Meta(res.meta);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const kinematicLib = useMemo(() => buildKinematicLibrary(), []);
  const v2Lib = useMemo(
    () => (v2Model ? buildLearnedRaceLibraryV2(v2Model) : null),
    [v2Model],
  );

  const kinAtSpeed = useMemo(() => kinematicLib.lookup(selectedSpeed), [kinematicLib, selectedSpeed]);
  const v2AtSpeed = useMemo(
    () => (v2Lib ? v2Lib.lookup(selectedSpeed) : null),
    [v2Lib, selectedSpeed],
  );

  // Diagnostics — kinematic always; v2 only when present.
  const kinDiag: LibraryDiagnostics = useMemo(
    () => diagnoseLibrary(kinAtSpeed),
    [kinAtSpeed],
  );
  const v2Diag: LibraryDiagnostics | null = useMemo(
    () => (v2AtSpeed ? diagnoseLibrary(v2AtSpeed) : null),
    [v2AtSpeed],
  );
  // The kinematic-vs-v2 comparison (mismatches) is computed against both.
  const cmpDiag: LibraryDiagnostics | null = useMemo(
    () => (v2AtSpeed ? diagnoseLibrary(kinAtSpeed, v2AtSpeed) : null),
    [kinAtSpeed, v2AtSpeed],
  );

  // Shared plot extent so both fan plots use identical axes. Union of
  // both libraries' sweeps so nothing gets clipped.
  const extent = useMemo(() => {
    let xMin = -1, xMax = 1, zMin = -1, zMax = 1;
    const consume = (prims: ReadonlyArray<MotionPrimitive>) => {
      for (const p of prims) {
        for (const s of p.sweep) {
          if (s.x < xMin) xMin = s.x;
          if (s.x > xMax) xMax = s.x;
          if (s.z < zMin) zMin = s.z;
          if (s.z > zMax) zMax = s.z;
        }
      }
    };
    consume(kinAtSpeed);
    if (v2AtSpeed) consume(v2AtSpeed);
    const pad = 1.5;
    return { xMin: xMin - pad, xMax: xMax + pad, zMin: zMin - pad, zMax: zMax + pad };
  }, [kinAtSpeed, v2AtSpeed]);

  const hoveredKin = hoveredIdx !== null ? kinAtSpeed[hoveredIdx] : undefined;
  const hoveredV2 = hoveredIdx !== null && v2AtSpeed ? v2AtSpeed[hoveredIdx] : undefined;
  const hoveredMismatch = hoveredIdx !== null && cmpDiag?.pairedMismatches
    ? cmpDiag.pairedMismatches.find((m) => m.index === hoveredIdx)
    : undefined;

  return (
    <main
      style={{
        color: '#cdd3de',
        fontFamily: 'ui-monospace, monospace',
        background: '#0a0d14',
        minHeight: '100vh',
        padding: isMobile ? '12px 10px 24px' : '18px 24px 32px',
      }}
    >
      <header style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 16 }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'baseline', gap: 12 }}>
          <a href="/" style={{ color: '#7fd6ff' }}>← demos</a>
          <a href="/raceprimitives" style={{ color: '#7fd6ff' }}>/raceprimitives</a>
          <h1 style={{ fontSize: 18, margin: 0 }}>primitive · action-space explorer</h1>
        </div>
        <p style={{ opacity: 0.7, margin: 0, maxWidth: 720, fontSize: 12, lineHeight: 1.45 }}>
          Side-by-side comparison of the kinematic and v2-learned motion-
          primitive libraries that drive the two cars on /raceprimitives. Both
          libraries use the IDENTICAL set of 76 controls (19 × 4 start speeds);
          they differ only in where each model predicts a given control will
          take the chassis. The overlay-and-diff view below shows exactly
          where the two models disagree — that disagreement is why v2 drives
          a different racing line.
        </p>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, alignItems: 'center' }}>
          <span style={{ opacity: 0.6, fontSize: 11, marginRight: 4 }}>START SPEED</span>
          {RACE_START_SPEEDS.map((s) => (
            <button
              key={s}
              onClick={() => setSelectedSpeed(s)}
              style={{
                padding: '4px 10px',
                background: selectedSpeed === s ? '#1a2030' : 'transparent',
                border: `1px solid ${selectedSpeed === s ? '#55dcff' : '#1f2735'}`,
                borderRadius: 4,
                color: selectedSpeed === s ? '#55dcff' : '#cdd3de',
                font: 'inherit',
                cursor: 'pointer',
              }}
            >
              {s} m/s
            </button>
          ))}
          <span style={{ flex: 1 }} />
          {v2Meta && (
            <span style={{ opacity: 0.65, fontSize: 11 }}>
              v2: {v2Meta.trialsUsed} trials · open-loop {v2Meta.openLoopRmsAt1s.toFixed(2)} m @ 1 s
            </span>
          )}
          {!v2Meta && (
            <span style={{ color: '#ffd070', fontSize: 11 }}>
              no v2 trained · <a href="/raceprimitives" style={{ color: '#ffd070' }}>train one →</a>
            </span>
          )}
        </div>
      </header>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: isMobile ? '1fr' : 'minmax(0, 1fr) minmax(0, 1fr)',
          gap: 12,
        }}
      >
        <PrimitiveFanPlot
          primitives={kinAtSpeed}
          forwardColor={KINEMATIC_CSS}
          title={`KINEMATIC · ${selectedSpeed} m/s`}
          subtitle={`${kinDiag.forwardCount} fwd · ${kinDiag.reverseCount} rev · hull ${kinDiag.hullAreaM2.toFixed(1)} m²`}
          highlightIndex={hoveredIdx ?? undefined}
          onHover={setHoveredIdx}
          fixedExtent={extent}
        />
        {v2AtSpeed && v2Diag && (
          <PrimitiveFanPlot
            primitives={v2AtSpeed}
            forwardColor={LEARNED_CSS}
            title={`V2 LEARNED · ${selectedSpeed} m/s`}
            subtitle={`${v2Diag.forwardCount} fwd · ${v2Diag.reverseCount} rev · hull ${v2Diag.hullAreaM2.toFixed(1)} m²`}
            highlightIndex={hoveredIdx ?? undefined}
            onHover={setHoveredIdx}
            fixedExtent={extent}
          />
        )}
        {!v2AtSpeed && (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              border: '1px dashed #1f2735',
              borderRadius: 6,
              padding: 24,
              opacity: 0.6,
              fontSize: 12,
              textAlign: 'center',
              minHeight: 200,
            }}
          >
            v2-learned library is empty.
            <br />
            Train a v2 model on{' '}
            <a href="/raceprimitives" style={{ color: '#7fd6ff' }}>/raceprimitives</a>
            {' '}then return here.
          </div>
        )}
      </div>

      {v2AtSpeed && cmpDiag?.pairedMismatches && (
        <div style={{ marginTop: 16 }}>
          <PrimitiveOverlayPlot
            primitivesA={kinAtSpeed}
            primitivesB={v2AtSpeed}
            mismatches={cmpDiag.pairedMismatches}
            colorA={KINEMATIC_CSS}
            colorB={LEARNED_CSS}
            labelA="kinematic"
            labelB="v2 learned"
            extent={extent}
            highlightIndex={hoveredIdx ?? undefined}
            onHover={setHoveredIdx}
          />
        </div>
      )}

      <DiagnosticsRow
        kinDiag={kinDiag}
        v2Diag={v2Diag}
        cmpDiag={cmpDiag}
        isMobile={isMobile}
      />

      {hoveredIdx !== null && (
        <HoverDetails
          index={hoveredIdx}
          kin={hoveredKin}
          v2={hoveredV2}
          mismatchM={hoveredMismatch?.distance}
        />
      )}
    </main>
  );
}

function DiagnosticsRow({ kinDiag, v2Diag, cmpDiag, isMobile }: {
  kinDiag: LibraryDiagnostics;
  v2Diag: LibraryDiagnostics | null;
  cmpDiag: LibraryDiagnostics | null;
  isMobile: boolean;
}) {
  return (
    <div
      style={{
        marginTop: 16,
        display: 'grid',
        gridTemplateColumns: isMobile ? '1fr' : v2Diag ? '1fr 1fr 1fr' : '1fr 1fr',
        gap: 8,
      }}
    >
      <DiagBlock
        label="kinematic · resolution"
        color="#ff8aa0"
        items={[
          ['primitives', `${kinDiag.count} (${kinDiag.forwardCount} fwd / ${kinDiag.reverseCount} rev)`],
          ['max angular gap', `${kinDiag.maxAngularGapDeg.toFixed(1)}°`],
          ['reachable hull', `${kinDiag.hullAreaM2.toFixed(2)} m²`],
          ['forward span x', `${kinDiag.forwardEndpointBBox.xMax.toFixed(2)} m`],
          ['forward span z', `±${Math.max(Math.abs(kinDiag.forwardEndpointBBox.zMin), Math.abs(kinDiag.forwardEndpointBBox.zMax)).toFixed(2)} m`],
        ]}
      />
      {v2Diag && (
        <DiagBlock
          label="v2 learned · resolution"
          color="#55dcff"
          items={[
            ['primitives', `${v2Diag.count} (${v2Diag.forwardCount} fwd / ${v2Diag.reverseCount} rev)`],
            ['max angular gap', `${v2Diag.maxAngularGapDeg.toFixed(1)}°`],
            ['reachable hull', `${v2Diag.hullAreaM2.toFixed(2)} m²`],
            ['forward span x', `${v2Diag.forwardEndpointBBox.xMax.toFixed(2)} m`],
            ['forward span z', `±${Math.max(Math.abs(v2Diag.forwardEndpointBBox.zMin), Math.abs(v2Diag.forwardEndpointBBox.zMax)).toFixed(2)} m`],
          ]}
        />
      )}
      {cmpDiag && cmpDiag.pairedMismatches && cmpDiag.largestMismatch && (
        <DiagBlock
          label="kinematic vs v2"
          color="#ffd070"
          items={[
            ['controls paired', `${cmpDiag.pairedMismatches.length}`],
            ['mean mismatch', `${cmpDiag.meanMismatch!.toFixed(3)} m`],
            ['max mismatch', `${cmpDiag.maxMismatch!.toFixed(3)} m`],
            ['worst control', `[${cmpDiag.largestMismatch.controls.map((c) => c.toFixed(2)).join(', ')}]`],
            ['hull ratio v2/kin', `${v2Diag && kinDiag.hullAreaM2 > 0 ? (v2Diag.hullAreaM2 / kinDiag.hullAreaM2).toFixed(2) : '—'}`],
          ]}
        />
      )}
    </div>
  );
}

function DiagBlock({ label, color, items }: {
  label: string;
  color: string;
  items: Array<[string, string]>;
}) {
  return (
    <div
      style={{
        background: '#0d1119',
        border: '1px solid #1f2735',
        borderRadius: 6,
        padding: '8px 12px',
        font: '11px ui-monospace, monospace',
      }}
    >
      <div style={{ color, fontWeight: 700, marginBottom: 6, fontSize: 10, letterSpacing: 0.5 }}>{label.toUpperCase()}</div>
      <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', columnGap: 10, rowGap: 3 }}>
        {items.map(([k, v]) => (
          <span key={k} style={{ display: 'contents' }}>
            <span style={{ opacity: 0.6 }}>{k}</span>
            <span style={{ textAlign: 'right', color: '#cdeaff' }}>{v}</span>
          </span>
        ))}
      </div>
    </div>
  );
}

function HoverDetails({ index, kin, v2, mismatchM }: {
  index: number;
  kin: MotionPrimitive | undefined;
  v2: MotionPrimitive | undefined;
  mismatchM: number | undefined;
}) {
  return (
    <div
      style={{
        marginTop: 12,
        background: '#0d1119',
        border: '1px solid #1f2735',
        borderRadius: 6,
        padding: '8px 12px',
        font: '11px ui-monospace, monospace',
      }}
    >
      <div style={{ fontSize: 10, opacity: 0.7, marginBottom: 6, letterSpacing: 0.5 }}>
        SELECTED PRIMITIVE · INDEX {index}
        {kin && (
          <span style={{ marginLeft: 12 }}>
            controls [{kin.controls.map((c) => c.toFixed(3)).join(', ')}] · duration {kin.duration.toFixed(2)} s
            {kin.reverse && <span style={{ color: '#ff8aa0' }}> · REVERSE</span>}
          </span>
        )}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr 1fr', columnGap: 12, rowGap: 3 }}>
        <span style={{ opacity: 0.6 }}>end</span>
        <span style={{ color: '#ff8aa0' }}>
          {kin
            ? `kin: dx=${kin.end.dx.toFixed(3)}  dz=${kin.end.dz.toFixed(3)}  dHead=${kin.end.dHeading.toFixed(3)}  spd=${kin.end.speed.toFixed(2)}`
            : '—'}
        </span>
        <span style={{ color: '#55dcff' }}>
          {v2
            ? `v2: dx=${v2.end.dx.toFixed(3)}  dz=${v2.end.dz.toFixed(3)}  dHead=${v2.end.dHeading.toFixed(3)}  spd=${v2.end.speed.toFixed(2)}`
            : '(no v2)'}
        </span>
        {mismatchM !== undefined && (
          <>
            <span style={{ opacity: 0.6 }}>mismatch</span>
            <span style={{ color: '#ffd070', gridColumn: 'span 2' }}>{mismatchM.toFixed(3)} m</span>
          </>
        )}
      </div>
    </div>
  );
}
