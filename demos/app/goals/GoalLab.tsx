'use client';

// GoalLab — a dedicated 3D surface for AUTHORING, VISUALIZING, and DEBUGGING
// canonical scenario goals. Pick a goal expressed in the `kinocat/scenario` AST;
// GoalLab compiles it, plans toward it with the real ScenarioEnvironment product
// search, renders every region (objective / avoid / bounds) color-coded, and
// animates the car along the plan while the compiled automaton lights up
// phase-by-phase — the planner's internal objective state, made visible.

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import {
  createGroundPlaneHelper,
  createCarMeshHelper,
  syncCarMesh,
  createPlanPathHelper,
  createRegionHelper,
  createGoalMarkerHelper,
  REGION_COLORS,
} from 'kinocat/adapters/three';
import {
  compile,
  evaluateProgress,
  collectScenarioRegions,
  validate,
  type CompiledAutomaton,
  type ScenarioState,
} from 'kinocat/scenario';
import type { CarKinematicState } from 'kinocat/agent';
import { goalLabPresets, type GoalPreset } from '../lib/goallab-presets';

interface HudState {
  preset: string;
  q: number;
  depth: number;
  maxDepth: number;
  done: boolean;
  laps: number;
  partial: boolean;
  cost: number;
  expansions: number;
  diagnostics: string[];
  transitions: { from: number; to: number; label: string }[];
  accepting: number[];
  start: number;
}

const PRESETS = goalLabPresets();

export default function GoalLab() {
  const mountRef = useRef<HTMLDivElement>(null);
  const [presetId, setPresetId] = useState(PRESETS[0]!.id);
  const [paused, setPaused] = useState(false);
  const [hud, setHud] = useState<HudState | null>(null);

  const presetIdRef = useRef(presetId);
  const pausedRef = useRef(paused);
  presetIdRef.current = presetId;
  pausedRef.current = paused;
  const loadRef = useRef<((id: string) => void) | null>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    const W = mount.clientWidth || window.innerWidth;
    const H = mount.clientHeight || 520;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0e14);
    const camera = new THREE.PerspectiveCamera(55, W / H, 0.1, 800);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);

    const orbit = new OrbitControls(camera, renderer.domElement);
    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const sun = new THREE.DirectionalLight(0xffffff, 1.0);
    sun.position.set(40, 120, 30);
    scene.add(sun);

    const car = createCarMeshHelper({ color: 0x4488ff });
    scene.add(car.group);
    const targetMarker = createGoalMarkerHelper({ color: 0xffcc33 });
    targetMarker.visible = false;
    scene.add(targetMarker);

    let content = new THREE.Group();
    scene.add(content);

    // Per-preset animation state.
    let automaton: CompiledAutomaton | null = null;
    let path: CarKinematicState[] = [];
    let preset: GoalPreset | null = null;
    let animClock = 0;
    let planCost = 0;
    let planExpansions = 0;
    let planPartial = false;
    let diagnostics: string[] = [];

    function clear(g: THREE.Group) {
      g.traverse((o) => {
        const m = o as THREE.Mesh;
        if (m.geometry) m.geometry.dispose();
        const mat = m.material as THREE.Material | THREE.Material[] | undefined;
        if (Array.isArray(mat)) mat.forEach((x) => x.dispose());
        else mat?.dispose();
      });
      scene.remove(g);
    }

    function load(id: string) {
      preset = PRESETS.find((p) => p.id === id) ?? PRESETS[0]!;
      clear(content);
      content = new THREE.Group();
      scene.add(content);

      const b = preset.bounds;
      content.add(createGroundPlaneHelper({ bounds: b, color: 0x141a26, gridDivisions: 24, gridColor: 0x223044, gridSubColor: 0x1a2330 }));

      // Regions, color-coded by plane.
      const regions = collectScenarioRegions(preset.scenario);
      for (const r of regions.objective) content.add(createRegionHelper(r, { color: REGION_COLORS.objective, y: 0.08 }));
      for (const r of regions.avoid) content.add(createRegionHelper(r, { color: REGION_COLORS.avoid, y: 0.08 }));
      for (const r of regions.maintain) content.add(createRegionHelper(r, { color: 0x556677, y: 0.04 }));

      // Compile + validate + plan.
      automaton = compile(preset.scenario.goal);
      diagnostics = validate(preset.scenario, { posCell: 0.3 }).map((d) => `[${d.severity}] ${d.check}: ${d.message}`);
      const result = preset.plan();
      path = result.path; // already projected to inner chassis states
      planCost = result.raw.cost;
      planExpansions = result.raw.stats.expansions;
      planPartial = result.raw.partial ?? false;

      if (path.length > 0) content.add(createPlanPathHelper(path, { color: 0x66ffaa, y: 0.12 }));
      targetMarker.visible = !!preset.movingTarget;

      // Camera frame.
      const cx = (b.x0 + b.x1) / 2;
      const cz = (b.z0 + b.z1) / 2;
      const span = Math.max(b.x1 - b.x0, b.z1 - b.z0);
      orbit.target.set(cx, 0, cz);
      camera.position.set(cx, span * 1.1, cz + span * 0.9);
      orbit.update();

      animClock = path.length ? path[0]!.t : 0;
      if (path.length) syncCarMesh(car.group, path[0]!);
    }
    loadRef.current = load;
    // NB: initial load is driven by the [presetId] effect below (which fires on
    // mount), so we don't call load() here — doing both double-plans on mount.

    function poseAt(tSim: number): CarKinematicState {
      if (path.length === 0) return { x: 0, z: 0, heading: 0, speed: 0, t: 0 };
      if (tSim <= path[0]!.t) return path[0]!;
      for (let i = 0; i + 1 < path.length; i++) {
        const a = path[i]!;
        const c = path[i + 1]!;
        if (tSim >= a.t && tSim <= c.t) {
          const u = c.t > a.t ? (tSim - a.t) / (c.t - a.t) : 0;
          return {
            x: a.x + (c.x - a.x) * u,
            z: a.z + (c.z - a.z) * u,
            heading: a.heading,
            speed: a.speed,
            t: tSim,
          };
        }
      }
      return path[path.length - 1]!;
    }

    function prefixIndex(tSim: number): number {
      let idx = 0;
      for (let i = 0; i < path.length; i++) {
        if (path[i]!.t <= tSim) idx = i;
        else break;
      }
      return idx;
    }

    let raf = 0;
    let last = performance.now();
    let lastHud = 0;
    function tick() {
      raf = requestAnimationFrame(tick);
      const now = performance.now();
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;

      const tEnd = path.length ? path[path.length - 1]!.t : 0;
      if (!pausedRef.current && path.length) {
        animClock += dt;
        if (animClock > tEnd + 1.2) animClock = path[0]!.t; // loop with a pause
      }

      const pose = poseAt(animClock);
      syncCarMesh(car.group, pose);

      // Dynamic target marker.
      if (preset?.movingTarget) {
        const tp = preset.movingTarget.predict(Math.min(animClock, tEnd));
        if (tp) targetMarker.position.set(tp.x, 0.6, tp.z);
      }

      // Deterministic progress via the shared evaluator. Throttle the React
      // state update to ~10 Hz so the render loop doesn't thrash 60 setState/s.
      if (automaton && now - lastHud >= 100) {
        lastHud = now;
        const idx = prefixIndex(animClock);
        const prefix: ScenarioState[] = path.slice(0, Math.max(1, idx + 1));
        const p = evaluateProgress(automaton, prefix);
        const transitions: HudState['transitions'] = [];
        for (const st of automaton.states) {
          for (const tr of st.transitions) {
            transitions.push({ from: st.id, to: tr.target, label: tr.guard.region.kind });
          }
        }
        setHud({
          preset: preset?.title ?? '',
          q: p.q,
          depth: p.depth,
          maxDepth: p.maxDepth,
          done: p.done,
          laps: p.laps,
          partial: planPartial,
          cost: planCost,
          expansions: planExpansions,
          diagnostics,
          transitions,
          accepting: automaton.accepting,
          start: automaton.start,
        });
      }

      orbit.update();
      renderer.render(scene, camera);
    }
    tick();

    function onResize() {
      const el = mountRef.current;
      if (!el) return;
      const w = el.clientWidth || window.innerWidth;
      const h = el.clientHeight || 520;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    }
    window.addEventListener('resize', onResize);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', onResize);
      orbit.dispose();
      renderer.dispose();
      if (renderer.domElement.parentElement === mount) mount.removeChild(renderer.domElement);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Reload when the preset changes.
  useEffect(() => {
    loadRef.current?.(presetId);
  }, [presetId]);

  return (
    <div style={{ position: 'relative', width: '100%', height: '85vh', minHeight: 520 }}>
      <div ref={mountRef} style={{ position: 'absolute', inset: 0 }} />
      <div
        style={{
          position: 'absolute',
          top: 12,
          left: 12,
          padding: 12,
          background: 'rgba(10,14,20,0.82)',
          color: '#cfe',
          font: '12px ui-monospace, monospace',
          borderRadius: 8,
          maxWidth: 360,
          lineHeight: 1.5,
        }}
      >
        <div style={{ marginBottom: 8 }}>
          <label>
            Goal:{' '}
            <select value={presetId} onChange={(e) => setPresetId(e.target.value)}>
              {PRESETS.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.title}
                </option>
              ))}
            </select>
          </label>{' '}
          <button onClick={() => setPaused((p) => !p)}>{paused ? '▶ play' : '⏸ pause'}</button>
        </div>
        {hud && (
          <>
            <div style={{ color: '#9bd', marginBottom: 6 }}>
              {PRESETS.find((p) => p.id === presetId)?.description}
            </div>
            <ProgressView hud={hud} />
            <div style={{ marginTop: 6 }}>
              plan: cost {hud.cost === Infinity ? '∞' : hud.cost.toFixed(1)} · {hud.expansions} exp
              {hud.partial && <span style={{ color: '#fc6' }}> · best-progress (partial)</span>}
            </div>
            <AutomatonView hud={hud} />
            {hud.diagnostics.length > 0 && (
              <div style={{ marginTop: 6, color: '#fc8' }}>
                {hud.diagnostics.map((d, i) => (
                  <div key={i}>{d}</div>
                ))}
              </div>
            )}
          </>
        )}
        <div style={{ marginTop: 8, color: '#789' }}>
          <span style={{ color: '#4df' }}>■</span> objective &nbsp;
          <span style={{ color: '#f46' }}>■</span> avoid &nbsp;
          <span style={{ color: '#6fa' }}>■</span> plan
        </div>
      </div>
    </div>
  );
}

function ProgressView({ hud }: { hud: HudState }) {
  const pct = hud.maxDepth > 0 ? Math.round((hud.depth / hud.maxDepth) * 100) : 0;
  return (
    <div>
      <div>
        phase <b>{hud.depth}</b> / {hud.maxDepth}
        {hud.laps > 0 && <> · laps {hud.laps}</>}
        {hud.done && <span style={{ color: '#6f9' }}> · DONE ✓</span>}
      </div>
      <div style={{ height: 8, background: '#1a2330', borderRadius: 4, overflow: 'hidden', marginTop: 3 }}>
        <div style={{ width: `${pct}%`, height: '100%', background: hud.done ? '#6f9' : '#4df' }} />
      </div>
    </div>
  );
}

function AutomatonView({ hud }: { hud: HudState }) {
  // Compact textual automaton with the live state highlighted.
  return (
    <div style={{ marginTop: 6, color: '#9ab' }}>
      <div style={{ color: '#789' }}>automaton (current = q{hud.q}):</div>
      {hud.transitions.slice(0, 12).map((tr, i) => (
        <div key={i} style={{ color: tr.from === hud.q ? '#4df' : '#566' }}>
          q{tr.from} →{tr.to === hud.q ? <b style={{ color: '#4df' }}> q{tr.to}</b> : <> q{tr.to}</>} : {tr.label}
        </div>
      ))}
    </div>
  );
}
