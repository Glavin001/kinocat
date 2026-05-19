import Link from 'next/link';

const demos = [
  {
    href: '/world3d',
    title: '3D navmesh world',
    desc: 'three.js scene: a vehicle plans through a 3D world and drives the path with pure-pursuit. Orbit/zoom; click to move the goal.',
  },
  {
    href: '/dynamic',
    title: 'Time-aware + multi-agent',
    desc: 'A moving obstacle with a time scrubber, a second NPC coordinated via the plan registry, and a jump affordance across a gap.',
  },
  {
    href: '/playground',
    title: 'Interactive 2D playground',
    desc: 'Drag the start/goal, add and move obstacles, tune the anytime deadline and reverse cost — replans instantly.',
  },
];

export default function Home() {
  return (
    <main
      style={{
        color: '#cdd3de',
        fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
        maxWidth: 760,
        margin: '0 auto',
        padding: 'clamp(16px, 5vw, 32px)',
      }}
    >
      <h1 style={{ fontSize: 22, marginBottom: 4 }}>kinocat demos</h1>
      <p style={{ opacity: 0.75, marginTop: 0 }}>
        Time-extended kinodynamic planning (IGHA*) running entirely in the
        browser via the <code>kinocat</code> package.
      </p>
      <ul style={{ listStyle: 'none', padding: 0 }}>
        {demos.map((d) => (
          <li
            key={d.href}
            style={{
              border: '1px solid #2a2f3a',
              borderRadius: 10,
              padding: 16,
              margin: '14px 0',
              background: '#12151c',
            }}
          >
            <Link
              href={d.href}
              style={{ color: '#7fd6ff', fontSize: 17, textDecoration: 'none' }}
            >
              {d.title} →
            </Link>
            <p style={{ opacity: 0.75, margin: '6px 0 0' }}>{d.desc}</p>
          </li>
        ))}
      </ul>
    </main>
  );
}
