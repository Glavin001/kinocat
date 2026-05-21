#!/usr/bin/env node
// Real-browser smoke test for /carchase. Boots the Next dev server (if one
// isn't already running on $E2E_BASE_URL), loads the page in headless
// chromium, waits for Rapier to initialize, and asserts:
//
//   1. The HUD reports a non-trivial robber speed after the AI has had
//      a few seconds to plan + drive — cars actually move.
//   2. The AI cops produce real (non-zero) plan lengths in the HUD —
//      multi-agent planning is alive.
//   3. Pressing 't' transfers control to the player and holding 'w'
//      gets the robber up to speed — player override works.
//   4. No console errors are emitted in any phase.
//
// Run from repo root via `pnpm e2e:carchase` (defined in demos/package.json).

// Resolve playwright from either the workspace install or the global
// install (/opt/node22 in the remote-exec sandbox). Top-level await is
// fine in an .mjs.
import fs from 'node:fs';
const playwrightCandidates = [
  new URL('../../node_modules/playwright/index.js', import.meta.url),
  new URL('../node_modules/playwright/index.js', import.meta.url),
  new URL('file:///opt/node22/lib/node_modules/playwright/index.js'),
];
let playwrightUrl = null;
for (const u of playwrightCandidates) {
  if (fs.existsSync(decodeURIComponent(new URL(u).pathname))) {
    playwrightUrl = u.href;
    break;
  }
}
if (!playwrightUrl) {
  throw new Error('playwright not found in workspace or global modules');
}
const _pw = await import(playwrightUrl);
const chromium = _pw.chromium ?? _pw.default?.chromium;
if (!chromium) {
  throw new Error('playwright module did not expose chromium');
}
import { spawn } from 'node:child_process';
import { setTimeout as sleep } from 'node:timers/promises';

const BASE_URL = process.env.E2E_BASE_URL || 'http://localhost:3000';
const PATH = '/carchase';
const HUD_SELECTOR = 'div[style*="ui-monospace"]';

let serverProc = null;

async function waitForUrl(url, timeoutMs = 30_000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(url);
      if (res.ok) return true;
    } catch {
      // ignore — server not ready yet
    }
    await sleep(500);
  }
  return false;
}

async function ensureServer() {
  // If something is already serving BASE_URL, use it.
  if (await waitForUrl(BASE_URL, 1500)) return;
  console.log('[e2e] starting `next dev` …');
  serverProc = spawn('pnpm', ['--filter', '@kinocat/demos', 'dev'], {
    cwd: new URL('../..', import.meta.url).pathname,
    stdio: ['ignore', 'pipe', 'pipe'],
    detached: false,
  });
  serverProc.stdout.on('data', (chunk) => {
    process.stderr.write(`[next] ${chunk}`);
  });
  serverProc.stderr.on('data', (chunk) => {
    process.stderr.write(`[next ERR] ${chunk}`);
  });
  const ok = await waitForUrl(BASE_URL, 60_000);
  if (!ok) throw new Error('Next dev server failed to start within 60s');
}

function killServer() {
  if (serverProc && !serverProc.killed) {
    try {
      process.kill(-serverProc.pid, 'SIGTERM');
    } catch {
      try {
        serverProc.kill('SIGTERM');
      } catch {}
    }
  }
}

/** Parse a number out of the HUD line, e.g. "v=3.4 m/s · hdg=12°". */
function parseSpeed(text) {
  const m = text.match(/v=([-\d.]+)\s*m\/s/);
  return m ? Number(m[1]) : NaN;
}
function parsePlan(line) {
  const m = line.match(/plan=(\d+)/);
  return m ? Number(m[1]) : 0;
}

async function main() {
  await ensureServer();

  const browser = await chromium.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--use-gl=swiftshader'],
  });
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 800 } });
  const page = await ctx.newPage();

  const consoleErrors = [];
  const pageErrors = [];
  page.on('console', (msg) => {
    if (msg.type() === 'error') consoleErrors.push(msg.text());
  });
  page.on('pageerror', (err) => pageErrors.push(String(err)));

  const url = `${BASE_URL}${PATH}`;
  console.log(`[e2e] loading ${url}`);
  await page.goto(url, { waitUntil: 'domcontentloaded' });

  // Wait for the HUD to render (which happens after Rapier WASM init).
  console.log('[e2e] waiting for HUD…');
  await page.waitForSelector(HUD_SELECTOR, { timeout: 30_000 });

  // Let the AI run for ~6 s so all cops + robber have replanned a few times.
  console.log('[e2e] giving the AI 6 s to plan + drive…');
  await page.waitForTimeout(6000);

  let hud = await page.locator(HUD_SELECTOR).first().textContent();
  console.log('[e2e] HUD snapshot after AI warmup:\n' + hud);

  const speedAi = parseSpeed(hud || '');
  if (!Number.isFinite(speedAi)) {
    throw new Error('HUD did not contain a robber speed line');
  }

  // Pull out plan lengths for each cop line. textContent on the HUD div
  // collapses children into one string with no separators, so we scan for
  // "copN · MODE · plan=N exp=N N ms" runs directly.
  const copRuns = [...(hud || '').matchAll(/cop\d[^·]*·\s*[A-Z_]+\s*·\s*plan=(\d+)\s+exp=(\d+)/g)];
  if (copRuns.length < 3) {
    throw new Error(
      `expected ≥3 cop status runs in HUD, got ${copRuns.length}:\n${hud}`,
    );
  }
  const copPlanLens = copRuns.map((m) => Number(m[1]));
  console.log('[e2e] cop plan lengths:', copPlanLens);

  // Robber should be moving (>0.3 m/s) under AI control after 6 s.
  if (speedAi < 0.3) {
    throw new Error(
      `robber not moving under AI: v=${speedAi} m/s (HUD:\n${hud})`,
    );
  }
  console.log(`[e2e] ✓ robber moves under AI (v=${speedAi.toFixed(2)} m/s)`);

  // At least one cop must have produced a multi-state plan.
  const planCount = copPlanLens.filter((n) => n >= 2).length;
  if (planCount < 1) {
    throw new Error(
      `no cops produced a multi-state plan after 6 s; HUD:\n${hud}`,
    );
  }
  console.log(`[e2e] ✓ ${planCount}/3 cops have a real plan`);

  // ---- player override ----
  console.log('[e2e] pressing T to take over the robber…');
  await page.keyboard.press('t');
  await page.waitForTimeout(150);
  hud = await page.locator(HUD_SELECTOR).first().textContent();
  if (!/YOU \(T to release\)/.test(hud || '')) {
    throw new Error(
      `T toggle did not switch HUD to player mode; HUD:\n${hud}`,
    );
  }
  console.log('[e2e] ✓ T toggle put the HUD into player mode');

  console.log('[e2e] holding W for 2 s to accelerate…');
  await page.keyboard.down('w');
  await page.waitForTimeout(2000);
  hud = await page.locator(HUD_SELECTOR).first().textContent();
  await page.keyboard.up('w');

  const speedPlayer = parseSpeed(hud || '');
  console.log(`[e2e] player-driven speed = ${speedPlayer} m/s`);
  if (!(speedPlayer > 1)) {
    throw new Error(
      `player WASD did not produce motion: v=${speedPlayer} m/s (HUD:\n${hud})`,
    );
  }
  console.log(`[e2e] ✓ player WASD moves the robber (v=${speedPlayer.toFixed(2)} m/s)`);

  // ---- no console errors anywhere ----
  if (consoleErrors.length > 0) {
    console.log('[e2e] ⚠ console errors:\n  ' + consoleErrors.join('\n  '));
  }
  if (pageErrors.length > 0) {
    throw new Error(
      'page error(s):\n  ' + pageErrors.join('\n  '),
    );
  }

  // Save a screenshot for the PR description / debugging.
  const shotPath = '/tmp/carchase-e2e.png';
  await page.screenshot({ path: shotPath, fullPage: false });
  console.log(`[e2e] screenshot → ${shotPath}`);

  await browser.close();
  console.log('[e2e] ALL CHECKS PASSED');
}

main()
  .catch((err) => {
    console.error('[e2e] FAILED:', err);
    process.exitCode = 1;
  })
  .finally(() => {
    killServer();
  });
