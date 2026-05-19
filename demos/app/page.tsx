import Link from 'next/link';

interface Demo {
  href: string;
  title: string;
  desc: string;
  tag: string;
}

const groups: { heading: string; demos: Demo[] }[] = [
  {
    heading: 'Flagship',
    demos: [
      {
        href: '/flagship',
        title: 'Interactive multi-agent flagship',
        desc: 'Opposing cross-traffic on a large procedural navcat terrain. Click a vehicle to select it, retarget its goal or drop hazards and watch every NPC replan live. A genuine boost & canyon jump the planner adopts, a misdirect it rejects on its own. Toggleable clearance & moving-obstacle broadphase.',
        tag: '3D · interactive · multi-agent',
      },
      {
        href: '/catmouse',
        title: 'Cat & Mouse pursuit',
        desc: 'AI cats observe a non-cooperative mouse, build a predictor of its motion, and plan to the interception pose — where the mouse WILL be at arrival time, not where it is. Plans are shared via PlanRegistry so cats flank instead of pile up. Both species can take boost pads; a canyon split forces one cat to detour while another takes the jump affordance. Toggle "naive mode" to watch the cats lag the mouse without prediction.',
        tag: '3D · prediction · pursuit',
      },
    ],
  },
  {
    heading: 'Algorithms & curves',
    demos: [
      {
        href: '/curves',
        title: 'Reeds-Shepp vs Dubins',
        desc: 'The analytical car curves underpinning the Hybrid A* heuristic and shot-to-goal. Drag poses and headings; watch the words and lengths.',
        tag: '2D · curves',
      },
      {
        href: '/primitives',
        title: 'Motion-primitive characterization',
        desc: 'The planner’s entire action set, swept live from a forward model across a control grid as you tune the kinematics.',
        tag: '2D · primitives',
      },
      {
        href: '/anytime',
        title: 'Anytime planning',
        desc: 'The same query at growing expansion budgets — IGHA* always returns the best plan found so far.',
        tag: '2D · planner',
      },
      {
        href: '/reverse',
        title: 'Reverse maneuvers',
        desc: 'A corridor with the goal behind: the only feasible plan reverses. No special-case logic — it falls out of the search.',
        tag: '2D · kinodynamic',
      },
    ],
  },
  {
    heading: 'Dynamic & multi-agent',
    demos: [
      {
        href: '/dynamic',
        title: 'Time-aware + multi-agent',
        desc: 'A moving obstacle with a time scrubber, a second NPC via the plan registry, and a jump affordance across a gap.',
        tag: '2D · time-aware',
      },
      {
        href: '/swarm',
        title: 'Multi-agent coordination',
        desc: 'NPCs on a ring crossing to the far side; emergent avoidance via shared plans, no negotiation protocol.',
        tag: '2D · plan registry',
      },
      {
        href: '/playground',
        title: 'Interactive playground',
        desc: 'Drag the start/goal, add and move obstacles, tune the anytime deadline and reverse cost — replans instantly.',
        tag: '2D · interactive',
      },
    ],
  },
  {
    heading: 'Spatial / navmesh (3D)',
    demos: [
      {
        href: '/world3d',
        title: '3D navmesh world',
        desc: 'A vehicle plans and tracks the path with pure-pursuit and plan-switch hysteresis. Orbit; tap to move the goal or drop obstacles.',
        tag: '3D · pure-pursuit',
      },
      {
        href: '/navmesh',
        title: '3D navmesh debug view',
        desc: 'A real navcat navmesh generated in-browser (ground → ramp → platform); kinocat plans over the NavcatWorld adapter.',
        tag: '3D · navcat',
      },
      {
        href: '/jumplinks',
        title: 'Static jump links',
        desc: 'Two islands and a gap; annotateJumpLinks registers a Mononen-style off-mesh connection the humanoid planner crosses.',
        tag: '3D · navcat',
      },
    ],
  },
  {
    heading: 'Humanoid',
    demos: [
      {
        href: '/humanoid',
        title: 'Humanoid vs. vehicle',
        desc: 'The omnidirectional humanoid threads a tight L-corridor a turn-radius-constrained vehicle cannot — same IGHA* core.',
        tag: '2D · humanoid',
      },
    ],
  },
];

export default function Home() {
  return (
    <main
      style={{
        color: '#cdd3de',
        fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
        maxWidth: 820,
        margin: '0 auto',
        padding: 'clamp(16px, 5vw, 32px)',
      }}
    >
      <h1 style={{ fontSize: 22, marginBottom: 4 }}>kinocat demos</h1>
      <p style={{ opacity: 0.75, marginTop: 0 }}>
        Time-extended kinodynamic planning (IGHA*) running entirely in the
        browser via the <code>kinocat</code> package — curves, motion
        primitives, anytime search, time-aware &amp; multi-agent planning,
        affordances, navmesh integration, and humanoid agents.
      </p>
      {groups.map((g) => (
        <section key={g.heading}>
          <h2
            style={{
              fontSize: 14,
              textTransform: 'uppercase',
              letterSpacing: 1,
              opacity: 0.6,
              marginTop: 28,
              marginBottom: 6,
            }}
          >
            {g.heading}
          </h2>
          <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
            {g.demos.map((d) => (
              <li
                key={d.href}
                style={{
                  border: '1px solid #2a2f3a',
                  borderRadius: 10,
                  padding: 16,
                  margin: '10px 0',
                  background: '#12151c',
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'baseline',
                    gap: 12,
                    flexWrap: 'wrap',
                  }}
                >
                  <Link
                    href={d.href}
                    style={{
                      color: '#7fd6ff',
                      fontSize: 17,
                      textDecoration: 'none',
                    }}
                  >
                    {d.title} →
                  </Link>
                  <span
                    style={{
                      fontSize: 11,
                      opacity: 0.6,
                      border: '1px solid #2a2f3a',
                      borderRadius: 999,
                      padding: '2px 8px',
                    }}
                  >
                    {d.tag}
                  </span>
                </div>
                <p style={{ opacity: 0.75, margin: '6px 0 0' }}>{d.desc}</p>
              </li>
            ))}
          </ul>
        </section>
      ))}
    </main>
  );
}
