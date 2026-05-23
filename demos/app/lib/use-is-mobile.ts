'use client';

// Hook returning true when the viewport is narrower than `breakpoint` px.
// Used to swap absolute-positioned desktop overlays for stacked / bottom-
// sheet mobile layouts in the race-primitives demo. SSR-safe (returns
// false during render; updates on mount + window resize).

import { useEffect, useState } from 'react';

export function useIsMobile(breakpoint = 768): boolean {
  const [isMobile, setIsMobile] = useState(false);
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const mq = window.matchMedia(`(max-width: ${breakpoint - 1}px)`);
    const update = () => setIsMobile(mq.matches);
    update();
    // matchMedia.addEventListener is the modern API; addListener is the
    // legacy form, still required by older Safari. Both noop'd inside
    // matchMedia's contract.
    if (typeof mq.addEventListener === 'function') {
      mq.addEventListener('change', update);
      return () => mq.removeEventListener('change', update);
    } else {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (mq as any).addListener(update);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      return () => (mq as any).removeListener(update);
    }
  }, [breakpoint]);
  return isMobile;
}
