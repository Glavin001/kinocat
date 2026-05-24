import Link from 'next/link';

type Tag =
  | '2D'
  | '3D'
  | 'Interactive'
  | 'Multi-agent'
  | 'Physics'
  | 'Time-aware'
  | 'Learning'
  | 'Diagnostic';

interface Demo {
  href: string;
  title: string;
  desc: string;
  tags: readonly Tag[];
}

interface DemoGroup {
  id: string;
  heading: string;
  blurb: string;
  demos: readonly Demo[];
}

const GROUPS: readonly DemoGroup[] = [
  {
    id: 'showcase',
    heading: 'Showcase',
    blurb: 'Drive, fly, chase, and watch the planner work in real Rapier physics.',
    demos: [
      {
        href: '/carchase',
        title: 'Car chase',
        desc: 'Three AI police cars pursue an AI evader through a downtown grid with ramps, drift slalom, and boost pads. Press T to take the wheel of the robber.',
        tags: ['3D', 'Interactive', 'Multi-agent'],
      },
      {
        href: '/dogfight',
        title: 'Dogfight',
        desc: 'Pilot a fixed-wing fighter while three AI opponents pursue and flank you across heightfield terrain, pylons, and a drifting blimp.',
        tags: ['3D', 'Interactive', 'Multi-agent'],
      },
      {
        href: '/flagship',
        title: 'Multi-agent flagship',
        desc: 'Opposing cross-traffic on a procedural city. Click any vehicle to retarget it or drop hazards — every NPC replans live.',
        tags: ['3D', 'Interactive', 'Multi-agent'],
      },
      {
        href: '/ramp',
        title: 'Ramp & ballistic jump',
        desc: 'A drivable heightfield ramp paired with a ballistic-jump affordance. Toggle the affordance to watch the plan swap from detour to launch.',
        tags: ['3D', 'Interactive', 'Physics'],
      },
      {
        href: '/obstaclecourse',
        title: 'Obstacle course',
        desc: 'One car, every kinocat building block toggleable from the HUD — heightfield, ramps, boost pads, drift slalom, waypoint loops.',
        tags: ['3D', 'Interactive', 'Physics'],
      },
      {
        href: '/catmouse',
        title: 'Cat & mouse',
        desc: 'AI cats predict a mouse’s future path and plan to where it will be at arrival time, not where it is now. Toggle naive mode to see the difference.',
        tags: ['3D', 'Multi-agent'],
      },
    ],
  },
  {
    id: 'fundamentals',
    heading: 'Fundamentals',
    blurb: 'Small, focused diagrams of the core ideas — curves, primitives, search, coordination.',
    demos: [
      {
        href: '/curves',
        title: 'Reeds-Shepp & Dubins',
        desc: 'The analytical car curves underpinning the Hybrid A* heuristic. Drag poses and headings; watch the maneuver words and lengths update.',
        tags: ['2D', 'Interactive'],
      },
      {
        href: '/primitives',
        title: 'Motion primitives',
        desc: 'The planner’s full action set, swept live from a forward model across a control grid as you tune the kinematics.',
        tags: ['2D', 'Interactive'],
      },
      {
        href: '/anytime',
        title: 'Anytime planning',
        desc: 'The same query at growing expansion budgets — IGHA* always returns the best plan found so far.',
        tags: ['2D'],
      },
      {
        href: '/reverse',
        title: 'Reverse maneuvers',
        desc: 'A corridor with the goal behind you. The only feasible plan reverses — and it falls out of the search, with no special case.',
        tags: ['2D'],
      },
      {
        href: '/humanoid',
        title: 'Humanoid vs vehicle',
        desc: 'An omnidirectional walker threads a tight L-corridor that a turn-radius-limited car can’t fit through. Same IGHA* core.',
        tags: ['2D'],
      },
      {
        href: '/playground',
        title: 'Sandbox',
        desc: 'Drag the start, goal, and obstacles. Tune the anytime deadline and reverse cost — the planner reruns instantly.',
        tags: ['2D', 'Interactive'],
      },
      {
        href: '/dynamic',
        title: 'Time-aware planning',
        desc: 'A moving obstacle on a time scrubber, plus a second NPC and a jump affordance. Planning across time, not just space.',
        tags: ['2D', 'Time-aware'],
      },
      {
        href: '/swarm',
        title: 'Swarm coordination',
        desc: 'NPCs on a ring cross to the far side. Emergent avoidance from shared plans alone — no negotiation protocol.',
        tags: ['2D', 'Multi-agent'],
      },
    ],
  },
  {
    id: 'spatial',
    heading: '3D worlds & flight',
    blurb: 'Navmesh adapters, off-mesh jump links, and a kinodynamic flight planner.',
    demos: [
      {
        href: '/world3d',
        title: '3D navmesh world',
        desc: 'A vehicle plans and tracks a 3D path with pure-pursuit and plan-switch hysteresis. Tap to retarget or drop obstacles.',
        tags: ['3D', 'Interactive'],
      },
      {
        href: '/navmesh',
        title: 'Navmesh debug view',
        desc: 'A real navcat navmesh generated in-browser (ground → ramp → platform), with the planner running over the NavcatWorld adapter.',
        tags: ['3D', 'Diagnostic'],
      },
      {
        href: '/jumplinks',
        title: 'Off-mesh jump links',
        desc: 'Two islands and a gap. annotateJumpLinks registers a Mononen-style off-mesh connection the humanoid planner crosses.',
        tags: ['3D'],
      },
      {
        href: '/plane',
        title: 'Fixed-wing flight planner',
        desc: 'A fixed-wing aircraft in a genuinely 3D state with OBB collision and searched roll. Knife-edges, ridges, canyons, gates.',
        tags: ['3D', 'Interactive'],
      },
    ],
  },
  {
    id: 'learning',
    heading: 'Learning the dynamics',
    blurb: 'Learn a vehicle model from physics, then race it, inspect it, and scope its errors.',
    demos: [
      {
        href: '/learnprimitives',
        title: 'Primitive learner',
        desc: 'Drive a Rapier car using only throttle/steer/brake. A 5-coefficient dynamics model is fit live and emitted as a drop-in primitive library.',
        tags: ['3D', 'Learning', 'Physics'],
      },
      {
        href: '/raceprimitives',
        title: 'Race the primitives',
        desc: 'Two identical Rapier cars race the same loop. One uses the kinematic library, the other the learned one. The gap is visible and measurable.',
        tags: ['3D', 'Learning', 'Physics'],
      },
      {
        href: '/model-lab',
        title: 'Model Lab',
        desc: 'Train the v2 dynamics model and inspect it visually — fan plots, coverage heatmaps, rollout playback, per-component RMS.',
        tags: ['2D', 'Learning', 'Diagnostic'],
      },
      {
        href: '/sim-to-real',
        title: 'Sim-to-real scope',
        desc: 'Overlay model predictions onto live Rapier in 3D. Playback, free drive, or click-to-goal — see exactly where model and physics diverge.',
        tags: ['3D', 'Diagnostic', 'Physics'],
      },
      {
        href: '/primitive-explorer',
        title: 'Primitive explorer',
        desc: 'Side-by-side fan plots for kinematic vs learned primitives, with overlay endpoints showing exactly which controls the two models disagree about.',
        tags: ['2D', 'Diagnostic'],
      },
    ],
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

const totalDemos = GROUPS.reduce((n, g) => n + g.demos.length, 0);

function TagChip({ token }: { token: Tag }) {
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
      }}
    >
      {token}
    </span>
  );
}

function DemoCard({ demo, featured = false }: { demo: Demo; featured?: boolean }) {
  return (
    <Link href={demo.href} className="kc-card" data-featured={featured ? 'true' : undefined}>
      <div className="kc-card-tags">
        {demo.tags.map((t) => (
          <TagChip key={t} token={t} />
        ))}
      </div>
      <h3 className="kc-card-title">{demo.title}</h3>
      <p className="kc-card-desc">{demo.desc}</p>
      <span className="kc-card-cta">Open demo →</span>
    </Link>
  );
}

function DemoSection({ group }: { group: DemoGroup }) {
  const isShowcase = group.id === 'showcase';
  return (
    <section id={group.id} className="kc-section">
      <header className="kc-section-header">
        <div className="kc-section-bar" aria-hidden />
        <div>
          <h2 className="kc-section-heading">{group.heading}</h2>
          <p className="kc-section-blurb">{group.blurb}</p>
        </div>
        <span className="kc-section-count">
          {group.demos.length} {group.demos.length === 1 ? 'demo' : 'demos'}
        </span>
      </header>
      <div className={isShowcase ? 'kc-grid kc-grid-showcase' : 'kc-grid kc-grid-compact'}>
        {group.demos.map((d) => (
          <DemoCard key={d.href} demo={d} featured={isShowcase} />
        ))}
      </div>
    </section>
  );
}

const TAG_LEGEND: readonly Tag[] = [
  'Interactive',
  'Multi-agent',
  'Physics',
  'Time-aware',
  'Learning',
  'Diagnostic',
];

export default function Home() {
  return (
    <>
      <style>{CSS}</style>
      <div className="kc-bg" aria-hidden />
      <main className="kc-main">
        <header className="kc-hero">
          <div className="kc-eyebrow">
            <span className="kc-dot" /> kinocat · {totalDemos} demos
          </div>
          <h1 className="kc-title">
            Kinodynamic planning,
            <br />
            <span className="kc-title-accent">live in your browser.</span>
          </h1>
          <p className="kc-lede">
            IGHA* with Reeds-Shepp curves and motion primitives — anytime
            search, time-aware multi-agent planning, affordances, navmesh
            integration, and learned dynamics. All powered by the{' '}
            <code className="kc-code">kinocat</code> package.
          </p>
          <nav className="kc-nav" aria-label="Sections">
            {GROUPS.map((g) => (
              <a key={g.id} href={`#${g.id}`} className="kc-nav-link">
                {g.heading}
                <span className="kc-nav-count">{g.demos.length}</span>
              </a>
            ))}
          </nav>
          <div className="kc-legend" aria-label="Tag legend">
            <span className="kc-legend-label">Tags</span>
            {TAG_LEGEND.map((t) => (
              <TagChip key={t} token={t} />
            ))}
          </div>
        </header>

        {GROUPS.map((group) => (
          <DemoSection key={group.id} group={group} />
        ))}

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
  padding: clamp(24px, 5vw, 56px) clamp(16px, 4vw, 32px) 80px;
  line-height: 1.55;
}

.kc-hero { padding: 24px 0 8px; }

.kc-eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  letter-spacing: 1.2px;
  text-transform: uppercase;
  color: var(--kc-text-dim);
  padding: 6px 12px;
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
  font-size: clamp(32px, 5.5vw, 52px);
  line-height: 1.05;
  letter-spacing: -0.02em;
  margin: 18px 0 14px;
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
  font-size: clamp(15px, 1.6vw, 17px);
  max-width: 720px;
  margin: 0 0 24px;
}
.kc-code {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.92em;
  background: rgba(127, 214, 255, 0.10);
  color: var(--kc-accent);
  padding: 1px 6px;
  border-radius: 4px;
}

.kc-nav {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 8px 0 16px;
}
.kc-nav-link {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-family: ui-sans-serif, system-ui, sans-serif;
  font-size: 13px;
  font-weight: 500;
  letter-spacing: -0.005em;
  color: var(--kc-text);
  text-decoration: none;
  padding: 8px 12px;
  border-radius: 8px;
  border: 1px solid var(--kc-border);
  background: rgba(15, 18, 25, 0.6);
  backdrop-filter: blur(6px);
  transition: color 120ms ease, border-color 120ms ease, background 120ms ease, transform 120ms ease;
}
.kc-nav-link:hover {
  color: var(--kc-accent);
  border-color: rgba(127, 214, 255, 0.45);
  background: rgba(127, 214, 255, 0.08);
  transform: translateY(-1px);
}
.kc-nav-count {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11px;
  color: var(--kc-text-muted);
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid var(--kc-border);
  border-radius: 999px;
  padding: 1px 7px;
  line-height: 1.4;
}
.kc-nav-link:hover .kc-nav-count {
  color: var(--kc-accent);
  border-color: rgba(127, 214, 255, 0.35);
}

.kc-legend {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  padding: 12px 0 0;
  border-top: 1px solid var(--kc-border-soft);
}
.kc-legend-label {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11px;
  letter-spacing: 1.2px;
  text-transform: uppercase;
  color: var(--kc-text-muted);
  margin-right: 4px;
}

.kc-section { margin-top: 56px; scroll-margin-top: 16px; }

.kc-section-header {
  display: flex;
  align-items: center;
  gap: 14px;
  margin-bottom: 18px;
}
.kc-section-bar {
  width: 3px;
  align-self: stretch;
  background: linear-gradient(180deg, var(--kc-accent), transparent);
  border-radius: 2px;
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
  gap: 10px;
  padding: 18px;
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
.kc-card[data-featured="true"] { padding: 22px; }
.kc-card[data-featured="true"] .kc-card-title { font-size: 17px; }

.kc-card-tags { display: flex; flex-wrap: wrap; gap: 6px; }
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
.kc-card[data-featured="true"] .kc-card-desc { -webkit-line-clamp: 6; }
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

.kc-footer {
  margin-top: 64px;
  padding-top: 20px;
  border-top: 1px solid var(--kc-border-soft);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  color: var(--kc-text-muted);
  text-align: center;
}
`;
