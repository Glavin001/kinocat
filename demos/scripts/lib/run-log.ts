import { appendFileSync, mkdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * Open a unique, live-tailable log file for a long-running debug script.
 *
 * Node buffers piped stdout until the process exits, so a long headless run
 * shows NO progress while it's alive — you can't tell if it's working or hung.
 * This writes every `log()` line straight to a unique file with `appendFileSync`
 * (each call flushes), so `tail -f <path>` shows progress in real time. The
 * chosen path is printed once (to stderr, which is unbuffered) so the runner
 * knows where to look. Console output still happens too (mirror = true) for
 * interactive runs.
 *
 * Logs land in `demos/scripts/logs/` (git-ignored) with a per-run unique name
 * (`<tag>-<ISO timestamp>-<pid>.log`), so concurrent or repeated runs never
 * clobber each other.
 */
export function openRunLog(
  tag: string,
  opts: { mirror?: boolean } = {},
): { path: string; log: (message: string) => void } {
  const dir = resolve(fileURLToPath(new URL('../logs', import.meta.url)));
  mkdirSync(dir, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const path = resolve(dir, `${tag}-${stamp}-${process.pid}.log`);
  const mirror = opts.mirror ?? true;
  const log = (message: string): void => {
    appendFileSync(path, message + '\n');
    if (mirror) process.stderr.write(message + '\n');
  };
  process.stderr.write(`[run-log] writing progress to ${path}\n`);
  return { path, log };
}
