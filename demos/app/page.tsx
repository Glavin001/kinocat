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
        href: '/carchase',
        title: 'Car-chase — Rapier raycast vehicles + multi-AI pursuit',
        desc: 'A cops-and-robbers stunt arena driven by Rapier.js DynamicRayCastVehicleControllers: three police cars pursue an AI evader through a downtown grid, an alley shortcut, a ramped jump gap, drift slalom, and boost pads. Each cop replans against the robber\'s published plan (PlanRegistry) and reads its siblings to fan out instead of stacking; the robber takes affordances (BoostPad / BallisticJump) opportunistically. Press T to take over the robber yourself — the cops re-target the human-driven chassis with no code change. Real Rapier raycast suspension + kinocat IGHA* on Reeds-Shepp curves.',
        tag: '3D · interactive · rapier · multi-agent',
      },
      {
        href: '/dogfight',
        title: 'Dogfight — interactive 3D pursuit',
        desc: 'Pilot a fixed-wing aircraft with the keyboard while three kinocat-driven opponents pursue, intercept, and flank you through a continuous heightfield terrain, tall pylons, a sweeping barrier between twin peaks, and a drifting blimp. Each AI replans against the live predicted player trajectory; sibling AIs read each other from a shared plan registry. Demonstrates the new HeightfieldAirspace (real ground-elevation collision), time-aware multi-agent planning, anytime replanning, and a GOAP-style tactical layer above kinocat.',
        tag: '3D · interactive · flight · multi-agent',
      },
      {
        href: '/flagship',
        title: 'Interactive multi-agent flagship',
        desc: 'Opposing cross-traffic on a large procedural navcat terrain. Click a vehicle to select it, retarget its goal or drop hazards and watch every NPC replan live. A genuine boost & canyon jump the planner adopts, a misdirect it rejects on its own. Toggleable clearance & moving-obstacle broadphase.',
        tag: '3D · interactive · multi-agent',
      },
      {
        href: '/ramp',
        title: 'Ramp + Affordance — drivable ramp vs ballistic shortcut',
        desc: 'A real drivable heightfield ramp the car physically climbs and launches off, paired with a BallisticJump Affordance the planner can take as a shortcut across a planner-only gap. Toggle the affordance to watch the planned path swap from the long detour to the jump. Execution is always real Rapier physics — the car always drives, never poses along the arc.',
        tag: '3D · interactive · affordances',
      },
      {
        href: '/obstaclecourse',
        title: 'Obstacle course — kinocat building blocks',
        desc: 'A single car on a small course with every kinocat building block toggleable from the HUD: heightfield terrain, cuboid buildings, jump ramp + affordance, boost pad, drift slalom, and a waypoint loop. Built entirely on the new core APIs (`kinocat/adapters/rapier`, `kinocat/adapters/three`, `planVehicleOnce`, `nudgeGoalClear`). Use it to sanity-check each piece in isolation before graduating to /carchase.',
        tag: '3D · interactive · building-blocks',
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
      {
        href: '/plane',
        title: '3D flight planner',
        desc: 'A fixed-wing aircraft over a genuinely 3D state with OBB collision and searched roll — knife-edges through narrow slots, climbs ridges, weaves canyons, threads gates, dodges moving zones, or all of it at once.',
        tag: '3D · flight · kinodynamic',
      },
    ],
  },
  {
    heading: 'Tools',
    demos: [
      {
        href: '/learnprimitives',
        title: 'Autonomous motion-primitive learner',
        desc: 'Spawn a Rapier vehicle on flat ground and DRIVE it like a human — only steer/throttle/brake, no teleports between trials. For each (start speed, controls) pair the car brakes itself to a stop, accelerates physically to the target speed, then applies the test controls for 0.55s while sample poses are recorded. A 5-coefficient parametric dynamics model is least-squares fit to the recorded trajectories and emitted as a drop-in MotionPrimitiveLibrary the planner uses directly. Closes the kinematic-vs-physics sim-to-real gap (inertia, suspension, tire slip, finite accel) with ~5 numbers persisted to localStorage. Downloadable JSON.',
        tag: '3D · autonomous · learning',
      },
      {
        href: '/raceprimitives',
        title: 'Race the primitives — kinematic vs learned, side by side',
        desc: 'A time trial. Two identical Rapier vehicles start on the same line and race the same waypoint loop (tight high-speed slalom + hard 90° turn into a stop). The ONLY difference: one planner uses the kinematic-derived primitive library, the other uses the library learned in /learnprimitives. The kinematic car overshoots the slalom (no understeer in its model) and brakes too late into the corner; the learned car threads the gates and stops on the mark. Split-viewport render with per-car lap times, tracking error, and lap counter — the gap closure is visible and measurable. If no learned library is cached, click "learn now" to fit one in ~10s.',
        tag: '3D · interactive · before-after',
      },
      {
        href: '/model-lab',
        title: 'Model Lab — train and inspect the v2 dynamics model',
        desc: 'First-class observability for the v2 learned vehicle model. Train (with live loss + per-round evolution), then visually inspect: action-space fan plots with Rapier ground-truth dots and ensemble uncertainty halos; coverage heatmap of held-out error by (speed × steer); trial rollout playback comparing Rapier vs v2 vs parametric-only vs kinematic; per-component RMS (heading / speed / yawRate / lateral velocity); and a scenario playground for "what would the model do here?" with optional on-demand Rapier ground truth.',
        tag: '2D · training · diagnostic',
      },
      {
        href: '/primitive-explorer',
        title: 'Primitive / action-space explorer — see what each planner can actually do',
        desc: 'A diagnostic tool for the /raceprimitives racing libraries. Side-by-side fan plots of the kinematic and v2-learned primitive libraries at each start-speed bucket. Below: an overlay view connecting kinematic↔v2 endpoint predictions per control so you can see exactly which controls the two models disagree about. Plus resolution diagnostics — primitive count, max angular gap, reachable-area hull — to verify the action space is dense enough. The honest answer to "why does the v2 car drive a different racing line?"',
        tag: '2D · diagnostic',
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
