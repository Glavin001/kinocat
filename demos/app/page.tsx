'use client';

import Link from 'next/link';
import { useMemo, useState } from 'react';

type Tag =
  | '2D'
  | '3D'
  | 'Interactive'
  | 'Multi-agent'
  | 'Physics'
  | 'Time-aware'
  | 'Learning'
  | 'Diagnostic';

// Importance tier. Drives visual prominence and ordering:
//  - 'flagship' : the must-see demos, surfaced in the "Start here" strip.
//  - 'standard' : the bulk of the catalog, grouped by theme.
//  - 'tool'     : low-level developer / diagnostic tooling, de-emphasized
//                 in a collapsed section so it does not crowd the showcase.
type Level = 'flagship' | 'standard' | 'tool';

type CategoryId = 'showcase' | 'learning' | 'fundamentals' | 'spatial' | 'tools';

interface Demo {
  href: string;
  title: string;
  desc: string;
  tags: readonly Tag[];
  category: CategoryId;
  level: Level;
}

interface Category {
  id: CategoryId;
  heading: string;
  blurb: string;
}

const CATEGORIES: readonly Category[] = [
  {
    id: 'showcase',
    heading: 'Showcase',
    blurb: 'Drive, fly, and chase — the planner working in real Rapier physics.',
  },
  {
    id: 'learning',
    heading: 'Learning the dynamics',
    blurb: 'Learn a vehicle model from physics, then race it and park with it.',
  },
  {
    id: 'fundamentals',
    heading: 'Fundamentals',
    blurb: 'Small, focused diagrams of the core ideas — curves, primitives, search.',
  },
  {
    id: 'spatial',
    heading: '3D worlds & flight',
    blurb: 'Navmesh adapters, off-mesh jump links, and a kinodynamic flight planner.',
  },
  {
    id: 'tools',
    heading: 'Developer & diagnostic tools',
    blurb: 'Low-level dashboards for training, inspecting, and debugging the planner internals.',
  },
];

const DEMOS: readonly Demo[] = [
  // ── Showcase ──────────────────────────────────────────────────────────
  {
    href: '/carchase',
    title: 'Car chase',
    desc: 'Three AI police cars pursue an AI evader through a downtown grid with ramps, drift slalom, and boost pads. Press T to take the wheel of the robber.',
    tags: ['3D', 'Interactive', 'Multi-agent'],
    category: 'showcase',
    level: 'flagship',
  },
  {
    href: '/ramp',
    title: 'Ramp & ballistic jump',
    desc: 'A drivable heightfield ramp paired with a ballistic-jump affordance. Toggle the affordance to watch the plan swap from detour to launch.',
    tags: ['3D', 'Interactive', 'Physics'],
    category: 'showcase',
    level: 'flagship',
  },
  {
    href: '/dogfight',
    title: 'Dogfight',
    desc: 'Pilot a fixed-wing fighter while three AI opponents pursue and flank you across heightfield terrain, pylons, and a drifting blimp.',
    tags: ['3D', 'Interactive', 'Multi-agent'],
    category: 'showcase',
    level: 'standard',
  },
  {
    href: '/flagship',
    title: 'Multi-agent flagship',
    desc: 'Opposing cross-traffic on a procedural city. Click any vehicle to retarget it or drop hazards — every NPC replans live.',
    tags: ['3D', 'Interactive', 'Multi-agent'],
    category: 'showcase',
    level: 'standard',
  },
  {
    href: '/obstaclecourse',
    title: 'Obstacle course',
    desc: 'One car, every kinocat building block toggleable from the HUD — heightfield, ramps, boost pads, drift slalom, waypoint loops.',
    tags: ['3D', 'Interactive', 'Physics'],
    category: 'showcase',
    level: 'standard',
  },
  {
    href: '/catmouse',
    title: 'Cat & mouse',
    desc: 'AI cats predict a mouse’s future path and plan to where it will be at arrival time, not where it is now. Toggle naive mode to see the difference.',
    tags: ['3D', 'Multi-agent'],
    category: 'showcase',
    level: 'standard',
  },

  // ── Learning the dynamics ─────────────────────────────────────────────
  {
    href: '/raceprimitives',
    title: 'Race the primitives',
    desc: 'Two identical Rapier cars race the same loop. One uses the kinematic library, the other the learned one. The gap is visible and measurable.',
    tags: ['3D', 'Learning', 'Physics'],
    category: 'learning',
    level: 'flagship',
  },
  {
    href: '/parking',
    title: 'Tight parking',
    desc: 'Three progressively harder parking scenarios — forward pull-in, reverse perpendicular, parallel — driven through sub-meter clearances by the same planner the racing demos use.',
    tags: ['3D', 'Physics', 'Interactive'],
    category: 'learning',
    level: 'flagship',
  },
  {
    href: '/learnprimitives',
    title: 'Primitive learner',
    desc: 'Drive a Rapier car using only throttle/steer/brake. A 5-coefficient dynamics model is fit live and emitted as a drop-in primitive library.',
    tags: ['3D', 'Learning', 'Physics'],
    category: 'learning',
    level: 'standard',
  },

  // ── Fundamentals ──────────────────────────────────────────────────────
  {
    href: '/playground',
    title: 'Sandbox',
    desc: 'Drag the start, goal, and obstacles. Tune the anytime deadline and reverse cost — the planner reruns instantly.',
    tags: ['2D', 'Interactive'],
    category: 'fundamentals',
    level: 'standard',
  },
  {
    href: '/curves',
    title: 'Reeds-Shepp & Dubins',
    desc: 'The analytical car curves underpinning the Hybrid A* heuristic. Drag poses and headings; watch the maneuver words and lengths update.',
    tags: ['2D', 'Interactive'],
    category: 'fundamentals',
    level: 'standard',
  },
  {
    href: '/primitives',
    title: 'Motion primitives',
    desc: 'The planner’s full action set, swept live from a forward model across a control grid as you tune the kinematics.',
    tags: ['2D', 'Interactive'],
    category: 'fundamentals',
    level: 'standard',
  },
  {
    href: '/anytime',
    title: 'Anytime planning',
    desc: 'The same query at growing expansion budgets — IGHA* always returns the best plan found so far.',
    tags: ['2D'],
    category: 'fundamentals',
    level: 'standard',
  },
  {
    href: '/reverse',
    title: 'Reverse maneuvers',
    desc: 'A corridor with the goal behind you. The only feasible plan reverses — and it falls out of the search, with no special case.',
    tags: ['2D'],
    category: 'fundamentals',
    level: 'standard',
  },
  {
    href: '/humanoid',
    title: 'Humanoid vs vehicle',
    desc: 'An omnidirectional walker threads a tight L-corridor that a turn-radius-limited car can’t fit through. Same IGHA* core.',
    tags: ['2D'],
    category: 'fundamentals',
    level: 'standard',
  },
  {
    href: '/dynamic',
    title: 'Time-aware planning',
    desc: 'A moving obstacle on a time scrubber, plus a second NPC and a jump affordance. Planning across time, not just space.',
    tags: ['2D', 'Time-aware'],
    category: 'fundamentals',
    level: 'standard',
  },
  {
    href: '/swarm',
    title: 'Swarm coordination',
    desc: 'NPCs on a ring cross to the far side. Emergent avoidance from shared plans alone — no negotiation protocol.',
    tags: ['2D', 'Multi-agent'],
    category: 'fundamentals',
    level: 'standard',
  },

  // ── 3D worlds & flight ────────────────────────────────────────────────
  {
    href: '/world3d',
    title: '3D navmesh world',
    desc: 'A vehicle plans and tracks a 3D path with pure-pursuit and plan-switch hysteresis. Tap to retarget or drop obstacles.',
    tags: ['3D', 'Interactive'],
    category: 'spatial',
    level: 'standard',
  },
  {
    href: '/plane',
    title: 'Fixed-wing flight planner',
    desc: 'A fixed-wing aircraft in a genuinely 3D state with OBB collision and searched roll. Knife-edges, ridges, canyons, gates.',
    tags: ['3D', 'Interactive'],
    category: 'spatial',
    level: 'standard',
  },
  {
    href: '/jumplinks',
    title: 'Off-mesh jump links',
    desc: 'Two islands and a gap. annotateJumpLinks registers a Mononen-style off-mesh connection the humanoid planner crosses.',
    tags: ['3D'],
    category: 'spatial',
    level: 'standard',
  },

  // ── Developer & diagnostic tools ──────────────────────────────────────
  {
    href: '/model-lab',
    title: 'Model Lab',
    desc: 'Train the v2 dynamics model and inspect it visually — fan plots, coverage heatmaps, rollout playback, per-component RMS.',
    tags: ['2D', 'Learning', 'Diagnostic'],
    category: 'tools',
    level: 'tool',
  },
  {
    href: '/sim-to-real',
    title: 'Sim-to-real scope',
    desc: 'Overlay model predictions onto live Rapier in 3D. Playback, free drive, or click-to-goal — see exactly where model and physics diverge.',
    tags: ['3D', 'Diagnostic', 'Physics'],
    category: 'tools',
    level: 'tool',
  },
  {
    href: '/primitive-explorer',
    title: 'Primitive explorer',
    desc: 'Side-by-side fan plots for kinematic vs learned primitives, with overlay endpoints showing exactly which controls the two models disagree about.',
    tags: ['2D', 'Diagnostic'],
    category: 'tools',
    level: 'tool',
  },
  {
    href: '/goals',
    title: 'Goal Lab',
    desc: 'Author a goal in the canonical scenario AST (reach/seq/all/any/repeat + invariants), watch the planner drive toward it, and see the compiled automaton light up phase-by-phase in real time.',
    tags: ['3D', 'Diagnostic', 'Time-aware'],
    category: 'tools',
    level: 'tool',
  },
  {
    href: '/navmesh',
    title: 'Navmesh debug view',
    desc: 'A real navcat navmesh generated in-browser (ground → ramp → platform), with the planner running over the NavcatWorld adapter.',
    tags: ['3D', 'Diagnostic'],
    category: 'tools',
    level: 'tool',
  },
];

const TAG_COLORS: Record<Tag, { fg: string; bg: string; bd: string }> = {
  '2D':           { fg: '#a8b3c7', bg: 'rgba(168, 179, 199, 0.08)', bd: 'rgba(168, 179, 199, 0.28)' },
  '3D':           { fg: '#cfd6e4', bg: 'rgba(207, 214, 228, 0.08)', bd: 'rgba(207, 214, 228, 0.28)' },
  Interactive:    { fg: '#ffd27f', bg: 'rgba(255, 210, 127, 0.10)', bd: 'rgba(255, 210, 127, 0.32)' },
  'Multi-agent':  { fg: '#ff9aa8', bg: 'rgba(255, 154, 168, 0.10)', bd: 'rgba(255, 154, 168, 0.30)' },
  Physics:        { fg: '#7fd6ff', bg: 'rgba(127, 214, 255, 0.10)', bd: 'rgba(127, 214, 255, 0.35)' },
  'Time-aware':   { fg: '#f1d77a', bg: 'rgba(241, 215, 122, 0.10)', bd: 'rgba(241, 215, 122, 0.30)' },
  Learning:       { fg: '#c79bff', bg: 'rgba(199, 155, 255, 0.10)', bd: 'rgba(199, 155, 255, 0.32)' },
  Diagnostic:     { fg: '#7fffd4', bg: 'rgba(127, 255, 212, 0.10)', bd: 'rgba(127, 255, 212, 0.30)' },
};

const ALL_TAGS: readonly Tag[] = [
  '2D',
  '3D',
  'Interactive',
  'Multi-agent',
  'Physics',
  'Time-aware',
  'Learning',
  'Diagnostic',
];

const TOOL_DEMOS = DEMOS.filter((d) => d.category === 'tools');
const TOOL_COUNT = TOOL_DEMOS.length;

// Tags shared by every demo in a group are redundant on the cards (the group
// already communicates them) — we surface those once in the section header and
// only render the *differentiating* tags on the cards themselves.
function commonTagsOf(demos: readonly Demo[]): readonly Tag[] {
  if (demos.length === 0) return [];
  return ALL_TAGS.filter((t) => demos.every((d) => d.tags.includes(t)));
}

function TagChip({ token, muted }: { token: Tag; muted?: boolean }) {
  const c = TAG_COLORS[token];
  return (
    <span
      style={{
        fontSize: 10.5,
        fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
        letterSpacing: 0.3,
        color: c.fg,
        background: c.bg,
        border: `1px solid ${c.bd}`,
        borderRadius: 999,
        padding: '2px 8px',
        whiteSpace: 'nowrap',
        opacity: muted ? 0.75 : 1,
      }}
    >
      {token}
    </span>
  );
}

function TagFilterChip({
  token,
  active,
  onClick,
}: {
  token: Tag;
  active: boolean;
  onClick: () => void;
}) {
  const c = TAG_COLORS[token];
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      style={{
        fontSize: 10.5,
        fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
        letterSpacing: 0.3,
        color: c.fg,
        background: active ? c.bg.replace(/0\.\d+\)/, '0.24)') : c.bg,
        border: `1px solid ${active ? c.fg : c.bd}`,
        borderRadius: 999,
        padding: '3px 9px',
        lineHeight: 1.4,
        whiteSpace: 'nowrap',
        cursor: 'pointer',
        transition: 'background 120ms ease, border-color 120ms ease',
      }}
    >
      {token}
    </button>
  );
}

function DemoCard({
  demo,
  hiddenTags,
}: {
  demo: Demo;
  hiddenTags?: readonly Tag[];
}) {
  const isFlagship = demo.level === 'flagship';
  const visibleTags = hiddenTags ? demo.tags.filter((t) => !hiddenTags.includes(t)) : demo.tags;
  return (
    <Link
      href={demo.href}
      className="kc-card"
      data-flagship={isFlagship ? 'true' : undefined}
    >
      {(isFlagship || visibleTags.length > 0) && (
        <div className="kc-card-tags">
          {isFlagship && <span className="kc-flag">★ Start here</span>}
          {visibleTags.map((t) => (
            <TagChip key={t} token={t} />
          ))}
        </div>
      )}
      <h3 className="kc-card-title">{demo.title}</h3>
      <p className="kc-card-desc">{demo.desc}</p>
      <span className="kc-card-cta">Open demo →</span>
    </Link>
  );
}

function DemoSection({
  id,
  heading,
  blurb,
  demos,
}: {
  id?: string;
  heading: string;
  blurb: string;
  demos: readonly Demo[];
}) {
  const common = commonTagsOf(demos);
  return (
    <section id={id} className="kc-section">
      <header className="kc-section-header">
        <div className="kc-section-bar" aria-hidden />
        <div className="kc-section-headtext">
          <div className="kc-section-titlerow">
            <h2 className="kc-section-heading">{heading}</h2>
            {common.map((t) => (
              <TagChip key={t} token={t} muted />
            ))}
          </div>
          <p className="kc-section-blurb">{blurb}</p>
        </div>
        <span className="kc-section-count">
          {demos.length} {demos.length === 1 ? 'demo' : 'demos'}
        </span>
      </header>
      <div className="kc-grid kc-grid-compact">
        {demos.map((d) => (
          <DemoCard key={d.href} demo={d} hiddenTags={common} />
        ))}
      </div>
    </section>
  );
}

function matchesQuery(demo: Demo, q: string): boolean {
  if (!q) return true;
  const haystack = `${demo.title} ${demo.desc} ${demo.tags.join(' ')}`.toLowerCase();
  return q
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean)
    .every((word) => haystack.includes(word));
}

export default function Home() {
  const [query, setQuery] = useState('');
  const [activeTags, setActiveTags] = useState<readonly Tag[]>([]);
  const [showTools, setShowTools] = useState(false);

  const toggleTag = (t: Tag) =>
    setActiveTags((prev) => (prev.includes(t) ? prev.filter((x) => x !== t) : [...prev, t]));

  const isFiltering = query.trim().length > 0 || activeTags.length > 0;

  const filtered = useMemo(
    () =>
      DEMOS.filter(
        (d) => matchesQuery(d, query.trim()) && activeTags.every((t) => d.tags.includes(t)),
      ),
    [query, activeTags],
  );

  const clearFilters = () => {
    setQuery('');
    setActiveTags([]);
  };

  return (
    <>
      <style>{CSS}</style>
      <div className="kc-bg" aria-hidden />
      <main className="kc-main">
        <header className="kc-hero">
          <div className="kc-eyebrow">
            <span className="kc-dot" /> kinocat · {DEMOS.length} demos
          </div>
          <h1 className="kc-title">
            Kinodynamic planning,{' '}
            <span className="kc-title-accent">live in your browser.</span>
          </h1>
          <p className="kc-lede">
            IGHA* with Reeds-Shepp curves and motion primitives — anytime,
            time-aware, multi-agent planning with affordances, navmesh
            integration, and learned dynamics. Powered by the{' '}
            <code className="kc-code">kinocat</code> package. New here? Look for
            the <span className="kc-lede-flag">★ Start here</span> demos.
          </p>
        </header>

        {/* Control bar: search + tag facets. */}
        <div className="kc-controls">
          <div className="kc-search">
            <span className="kc-search-icon" aria-hidden>
              ⌕
            </span>
            <input
              type="search"
              className="kc-search-input"
              placeholder="Search demos…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              aria-label="Search demos"
            />
          </div>
          <div className="kc-facets" role="group" aria-label="Filter by tag">
            {ALL_TAGS.map((t) => (
              <TagFilterChip
                key={t}
                token={t}
                active={activeTags.includes(t)}
                onClick={() => toggleTag(t)}
              />
            ))}
            {isFiltering && (
              <button type="button" className="kc-clear" onClick={clearFilters}>
                Clear
              </button>
            )}
          </div>
        </div>

        {isFiltering ? (
          <section className="kc-section" aria-label="Search results">
            <header className="kc-section-header">
              <div className="kc-section-bar" aria-hidden />
              <div className="kc-section-headtext">
                <div className="kc-section-titlerow">
                  <h2 className="kc-section-heading">Results</h2>
                </div>
                <p className="kc-section-blurb">
                  {filtered.length} {filtered.length === 1 ? 'demo matches' : 'demos match'} your
                  filters.
                </p>
              </div>
            </header>
            {filtered.length > 0 ? (
              <div className="kc-grid kc-grid-compact">
                {filtered.map((d) => (
                  <DemoCard key={d.href} demo={d} />
                ))}
              </div>
            ) : (
              <p className="kc-empty">
                No demos match.{' '}
                <button type="button" className="kc-clear" onClick={clearFilters}>
                  Clear filters
                </button>
              </p>
            )}
          </section>
        ) : (
          <>
            {/* Themed categories, most important first, excluding the
                low-level tools. Each demo appears exactly once; the
                recommended starting points carry a "★ Start here" marker
                in place rather than being duplicated into a separate
                section. */}
            {CATEGORIES.filter((c) => c.id !== 'tools').map((cat) => {
              const demos = DEMOS.filter((d) => d.category === cat.id);
              if (demos.length === 0) return null;
              return (
                <DemoSection
                  key={cat.id}
                  id={cat.id}
                  heading={cat.heading}
                  blurb={cat.blurb}
                  demos={demos}
                />
              );
            })}

            {/* Developer tools: de-emphasized, collapsed by default. */}
            <section id="tools" className="kc-section kc-section-tools">
              <button
                type="button"
                className="kc-tools-toggle"
                onClick={() => setShowTools((v) => !v)}
                aria-expanded={showTools}
              >
                <span className="kc-tools-caret" data-open={showTools ? 'true' : undefined}>
                  ▸
                </span>
                <span className="kc-tools-title">Developer &amp; diagnostic tools</span>
                <span className="kc-section-count">{TOOL_COUNT}</span>
                <span className="kc-tools-hint">
                  {showTools ? 'Hide' : 'Low-level dashboards for debugging the planner — show'}
                </span>
              </button>
              {showTools && (
                <div className="kc-grid kc-grid-compact kc-tools-grid">
                  {TOOL_DEMOS.map((d) => (
                    <DemoCard key={d.href} demo={d} hiddenTags={commonTagsOf(TOOL_DEMOS)} />
                  ))}
                </div>
              )}
            </section>
          </>
        )}

        <footer className="kc-footer">
          <span>Built with kinocat, Rapier, Three.js, and Next.js.</span>
        </footer>
      </main>
    </>
  );
}

const CSS = `
:root {
  --kc-bg: #07080c;
  --kc-panel: #0f1219;
  --kc-panel-2: #141823;
  --kc-border: #232838;
  --kc-border-soft: #1a1e2a;
  --kc-text: #e3e7ef;
  --kc-text-dim: #98a1b3;
  --kc-text-muted: #6b7385;
  --kc-accent: #7fd6ff;
  --kc-accent-2: #c79bff;
}

html, body { background: var(--kc-bg); }

.kc-bg {
  position: fixed;
  inset: 0;
  z-index: -1;
  background:
    radial-gradient(900px 600px at 15% -10%, rgba(127, 214, 255, 0.10), transparent 60%),
    radial-gradient(900px 600px at 110% 10%, rgba(199, 155, 255, 0.08), transparent 55%),
    radial-gradient(700px 500px at 50% 110%, rgba(127, 255, 212, 0.05), transparent 60%);
  pointer-events: none;
}

.kc-main {
  color: var(--kc-text);
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Inter, sans-serif;
  max-width: 1120px;
  margin: 0 auto;
  padding: clamp(16px, 3vw, 28px) clamp(16px, 4vw, 32px) 72px;
  line-height: 1.5;
}

.kc-hero { padding: 4px 0 0; }

.kc-eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11px;
  letter-spacing: 1.1px;
  text-transform: uppercase;
  color: var(--kc-text-dim);
  padding: 5px 11px;
  border: 1px solid var(--kc-border);
  background: rgba(20, 24, 35, 0.6);
  border-radius: 999px;
}
.kc-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--kc-accent);
  box-shadow: 0 0 12px var(--kc-accent);
}

.kc-title {
  font-size: clamp(26px, 4vw, 38px);
  line-height: 1.08;
  letter-spacing: -0.02em;
  margin: 12px 0 8px;
  font-weight: 600;
}
.kc-title-accent {
  background: linear-gradient(90deg, var(--kc-accent), var(--kc-accent-2));
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}

.kc-lede {
  color: var(--kc-text-dim);
  font-size: clamp(13.5px, 1.4vw, 15px);
  max-width: 680px;
  margin: 0;
}
.kc-code {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.92em;
  background: rgba(127, 214, 255, 0.10);
  color: var(--kc-accent);
  padding: 1px 6px;
  border-radius: 4px;
}

/* ── Control bar ─────────────────────────────────────────────── */
.kc-controls {
  position: sticky;
  top: 0;
  z-index: 5;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px 12px;
  margin: 16px 0 4px;
  padding: 10px 0;
  background: linear-gradient(180deg, var(--kc-bg) 72%, rgba(7, 8, 12, 0));
  backdrop-filter: blur(4px);
}
.kc-search {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  flex: 0 1 300px;
  min-width: 200px;
  padding: 7px 12px;
  border: 1px solid var(--kc-border);
  border-radius: 10px;
  background: rgba(15, 18, 25, 0.7);
  transition: border-color 120ms ease, background 120ms ease;
}
.kc-search:focus-within {
  border-color: rgba(127, 214, 255, 0.5);
  background: rgba(127, 214, 255, 0.06);
}
.kc-search-icon { color: var(--kc-text-muted); font-size: 15px; }
.kc-search-input {
  flex: 1;
  border: none;
  outline: none;
  background: transparent;
  color: var(--kc-text);
  font-size: 14px;
  font-family: inherit;
}
.kc-search-input::placeholder { color: var(--kc-text-muted); }

.kc-facets {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
}
.kc-clear {
  font-size: 12px;
  font-family: inherit;
  color: var(--kc-text-dim);
  background: transparent;
  border: 1px solid var(--kc-border);
  border-radius: 999px;
  padding: 3px 10px;
  cursor: pointer;
  transition: color 120ms ease, border-color 120ms ease;
}
.kc-clear:hover { color: var(--kc-accent); border-color: rgba(127, 214, 255, 0.4); }

.kc-empty { color: var(--kc-text-dim); font-size: 14px; }

.kc-section { margin-top: 34px; scroll-margin-top: 68px; }

.kc-section-header {
  display: flex;
  align-items: center;
  gap: 14px;
  margin-bottom: 16px;
}
.kc-section-bar {
  width: 3px;
  align-self: stretch;
  background: linear-gradient(180deg, var(--kc-accent), transparent);
  border-radius: 2px;
}
.kc-section-headtext { min-width: 0; }
.kc-section-titlerow {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
}
.kc-section-heading {
  font-size: 20px;
  font-weight: 600;
  letter-spacing: -0.01em;
  margin: 0;
}
.kc-section-blurb {
  margin: 2px 0 0;
  color: var(--kc-text-muted);
  font-size: 13.5px;
}
.kc-section-count {
  margin-left: auto;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  color: var(--kc-text-muted);
  border: 1px solid var(--kc-border);
  border-radius: 999px;
  padding: 2px 10px;
  background: rgba(15, 18, 25, 0.7);
  white-space: nowrap;
}

.kc-grid { display: grid; gap: 14px; }
.kc-grid-showcase { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.kc-grid-compact  { grid-template-columns: repeat(3, minmax(0, 1fr)); }
@media (max-width: 880px) {
  .kc-grid-showcase,
  .kc-grid-compact { grid-template-columns: 1fr; }
}
@media (min-width: 881px) and (max-width: 1080px) {
  .kc-grid-compact { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}

.kc-card {
  position: relative;
  display: flex;
  flex-direction: column;
  gap: 9px;
  padding: 16px;
  border: 1px solid var(--kc-border);
  border-radius: 14px;
  background:
    linear-gradient(180deg, rgba(20, 24, 35, 0.85), rgba(15, 18, 25, 0.85));
  text-decoration: none;
  color: inherit;
  transition: transform 140ms ease, border-color 140ms ease, box-shadow 140ms ease, background 140ms ease;
  overflow: hidden;
}
.kc-card::before {
  content: "";
  position: absolute;
  inset: 0;
  border-radius: inherit;
  padding: 1px;
  background: linear-gradient(140deg, rgba(127, 214, 255, 0.35), transparent 40%, transparent 60%, rgba(199, 155, 255, 0.25));
  -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  -webkit-mask-composite: xor;
          mask-composite: exclude;
  opacity: 0;
  transition: opacity 160ms ease;
  pointer-events: none;
}
.kc-card:hover {
  transform: translateY(-2px);
  border-color: rgba(127, 214, 255, 0.45);
  box-shadow: 0 12px 40px -16px rgba(127, 214, 255, 0.35), 0 2px 8px rgba(0, 0, 0, 0.4);
  background: linear-gradient(180deg, rgba(24, 30, 44, 0.9), rgba(17, 21, 30, 0.9));
}
.kc-card:hover::before { opacity: 1; }

/* Flagship / recommended cards keep a subtle warm accent plus a small
   "★ Start here" marker, so they stand out in place instead of being
   duplicated into a separate section. */
.kc-card[data-flagship="true"] {
  border-color: rgba(255, 210, 127, 0.28);
  background: linear-gradient(180deg, rgba(28, 27, 34, 0.9), rgba(18, 18, 23, 0.9));
}
.kc-card[data-flagship="true"]:hover {
  border-color: rgba(255, 210, 127, 0.55);
  box-shadow: 0 12px 40px -16px rgba(255, 210, 127, 0.35), 0 2px 8px rgba(0, 0, 0, 0.4);
}

.kc-flag {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 10.5px;
  letter-spacing: 0.3px;
  color: #ffd27f;
  background: rgba(255, 210, 127, 0.12);
  border: 1px solid rgba(255, 210, 127, 0.42);
  border-radius: 999px;
  padding: 2px 8px;
  white-space: nowrap;
}
.kc-lede-flag {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.86em;
  color: #ffd27f;
  white-space: nowrap;
}

.kc-card-tags { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.kc-card-title {
  font-size: 15.5px;
  font-weight: 600;
  margin: 0;
  color: var(--kc-text);
  letter-spacing: -0.005em;
}
.kc-card-desc {
  font-size: 13.5px;
  color: var(--kc-text-dim);
  margin: 0;
  display: -webkit-box;
  -webkit-line-clamp: 4;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.kc-card[data-featured="true"] .kc-card-desc { -webkit-line-clamp: 4; }
.kc-card-cta {
  margin-top: auto;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  color: var(--kc-accent);
  letter-spacing: 0.3px;
  opacity: 0.85;
  transition: opacity 140ms ease, transform 140ms ease;
}
.kc-card:hover .kc-card-cta { opacity: 1; transform: translateX(2px); }

/* ── Developer tools (de-emphasized) ─────────────────────────── */
.kc-section-tools { margin-top: 34px; }
.kc-tools-toggle {
  display: flex;
  align-items: center;
  gap: 12px;
  width: 100%;
  text-align: left;
  padding: 13px 16px;
  border: 1px dashed var(--kc-border);
  border-radius: 12px;
  background: rgba(15, 18, 25, 0.4);
  color: var(--kc-text-dim);
  cursor: pointer;
  font-family: inherit;
  transition: border-color 120ms ease, background 120ms ease, color 120ms ease;
}
.kc-tools-toggle:hover {
  border-color: rgba(127, 214, 255, 0.35);
  background: rgba(127, 214, 255, 0.04);
  color: var(--kc-text);
}
.kc-tools-caret {
  display: inline-block;
  transition: transform 140ms ease;
  color: var(--kc-text-muted);
  font-size: 12px;
}
.kc-tools-caret[data-open="true"] { transform: rotate(90deg); }
.kc-tools-title { font-size: 15px; font-weight: 600; color: var(--kc-text); }
.kc-tools-hint {
  margin-left: auto;
  font-size: 12px;
  color: var(--kc-text-muted);
}
.kc-tools-grid { margin-top: 14px; }

.kc-footer {
  margin-top: 56px;
  padding-top: 20px;
  border-top: 1px solid var(--kc-border-soft);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  color: var(--kc-text-muted);
  text-align: center;
}
`;
