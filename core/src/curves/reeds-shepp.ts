// Reeds-Shepp curves (Reeds & Shepp 1990): shortest forward+reverse path for a
// car with a minimum turning radius. Faithful port of OMPL's
// ReedsSheppStateSpace (CSC / CCC / CCCC / CCSC / CCSCC families with the
// timeflip, reflect and backwards symmetries that cover all 48 path words).

import type { CurvePath, Pose } from './types';
import {
  HALF_PI,
  PI,
  ZERO,
  buildPath,
  mod2pi,
  polar,
  toLocal,
  type SegType,
} from './internal';

const N: SegType = 'N';
const L: SegType = 'L';
const S: SegType = 'S';
const R: SegType = 'R';

const TYPES: readonly (readonly SegType[])[] = [
  [L, R, L, N, N], // 0
  [R, L, R, N, N], // 1
  [L, R, L, R, N], // 2
  [R, L, R, L, N], // 3
  [L, R, S, L, N], // 4
  [R, L, S, R, N], // 5
  [L, S, R, L, N], // 6
  [R, S, L, R, N], // 7
  [L, R, S, R, N], // 8
  [R, L, S, L, N], // 9
  [R, S, R, L, N], // 10
  [L, S, L, R, N], // 11
  [L, S, R, N, N], // 12
  [R, S, L, N, N], // 13
  [L, S, L, N, N], // 14
  [R, S, R, N, N], // 15
  [L, R, S, L, R], // 16
  [R, L, S, R, L], // 17
];

interface Cand {
  ok: boolean;
  t: number;
  u: number;
  v: number;
}
const FAIL: Cand = { ok: false, t: 0, u: 0, v: 0 };

interface Best {
  typeIndex: number;
  values: number[];
  length: number;
}

function pathLen(typeIndex: number, values: readonly number[]): number {
  const tmpl = TYPES[typeIndex]!;
  let len = 0;
  for (let i = 0; i < tmpl.length; i++) {
    if (tmpl[i] !== N) len += Math.abs(values[i] ?? 0);
  }
  return len;
}

function consider(best: Best, typeIndex: number, values: number[]): void {
  const len = pathLen(typeIndex, values);
  if (len < best.length) {
    best.typeIndex = typeIndex;
    best.values = values;
    best.length = len;
  }
}

// ---- formula 8.1 / 8.2 (CSC) -------------------------------------------------

function LpSpLp(x: number, y: number, phi: number): Cand {
  const { r: u, theta: t } = polar(x - Math.sin(phi), y - 1 + Math.cos(phi));
  if (t >= -ZERO) {
    const v = mod2pi(phi - t);
    if (v >= -ZERO) return { ok: true, t, u, v };
  }
  return FAIL;
}

function LpSpRp(x: number, y: number, phi: number): Cand {
  const { r: u1r, theta: t1 } = polar(x + Math.sin(phi), y - 1 - Math.cos(phi));
  const u1 = u1r * u1r;
  if (u1 >= 4) {
    const f = Math.sqrt(u1 - 4);
    const theta = Math.atan2(2, f);
    const t = mod2pi(t1 + theta);
    const u = f;
    const v = mod2pi(t - phi);
    if (t >= -ZERO && v >= -ZERO) return { ok: true, t, u, v };
  }
  return FAIL;
}

function CSC(best: Best, x: number, y: number, phi: number): void {
  let c = LpSpLp(x, y, phi);
  if (c.ok) consider(best, 14, [c.t, c.u, c.v]);
  c = LpSpLp(-x, y, -phi);
  if (c.ok) consider(best, 14, [-c.t, -c.u, -c.v]);
  c = LpSpLp(x, -y, -phi);
  if (c.ok) consider(best, 15, [c.t, c.u, c.v]);
  c = LpSpLp(-x, -y, phi);
  if (c.ok) consider(best, 15, [-c.t, -c.u, -c.v]);

  c = LpSpRp(x, y, phi);
  if (c.ok) consider(best, 12, [c.t, c.u, c.v]);
  c = LpSpRp(-x, y, -phi);
  if (c.ok) consider(best, 12, [-c.t, -c.u, -c.v]);
  c = LpSpRp(x, -y, -phi);
  if (c.ok) consider(best, 13, [c.t, c.u, c.v]);
  c = LpSpRp(-x, -y, phi);
  if (c.ok) consider(best, 13, [-c.t, -c.u, -c.v]);
}

// ---- formula 8.3 / 8.4 (CCC) -------------------------------------------------

function LpRmL(x: number, y: number, phi: number): Cand {
  const xi = x - Math.sin(phi);
  const eta = y - 1 + Math.cos(phi);
  const { r: u1, theta } = polar(xi, eta);
  if (u1 <= 4) {
    const u = -2 * Math.asin(0.25 * u1);
    const t = mod2pi(theta + 0.5 * u + PI);
    const v = mod2pi(phi - t + u);
    if (t >= -ZERO && u <= ZERO) return { ok: true, t, u, v };
  }
  return FAIL;
}

function CCC(best: Best, x: number, y: number, phi: number): void {
  let c = LpRmL(x, y, phi);
  if (c.ok) consider(best, 0, [c.t, c.u, c.v]);
  c = LpRmL(-x, y, -phi);
  if (c.ok) consider(best, 0, [-c.t, -c.u, -c.v]);
  c = LpRmL(x, -y, -phi);
  if (c.ok) consider(best, 1, [c.t, c.u, c.v]);
  c = LpRmL(-x, -y, phi);
  if (c.ok) consider(best, 1, [-c.t, -c.u, -c.v]);

  // backwards
  const xb = x * Math.cos(phi) + y * Math.sin(phi);
  const yb = x * Math.sin(phi) - y * Math.cos(phi);
  c = LpRmL(xb, yb, phi);
  if (c.ok) consider(best, 0, [c.v, c.u, c.t]);
  c = LpRmL(-xb, yb, -phi);
  if (c.ok) consider(best, 0, [-c.v, -c.u, -c.t]);
  c = LpRmL(xb, -yb, -phi);
  if (c.ok) consider(best, 1, [c.v, c.u, c.t]);
  c = LpRmL(-xb, -yb, phi);
  if (c.ok) consider(best, 1, [-c.v, -c.u, -c.t]);
}

// ---- formula 8.7 / 8.8 (CCCC) ------------------------------------------------

function tauOmega(
  u: number,
  v: number,
  xi: number,
  eta: number,
  phi: number,
): { tau: number; omega: number } {
  const delta = mod2pi(u - v);
  const A = Math.sin(u) - Math.sin(delta);
  const B = Math.cos(u) - Math.cos(delta) - 1;
  const t1 = Math.atan2(eta * A - xi * B, xi * A + eta * B);
  const t2 = 2 * (Math.cos(delta) - Math.cos(v) - Math.cos(u)) + 3;
  const tau = t2 < 0 ? mod2pi(t1 + PI) : mod2pi(t1);
  const omega = mod2pi(tau - u + v - phi);
  return { tau, omega };
}

function LpRupLumRm(x: number, y: number, phi: number): Cand {
  const xi = x + Math.sin(phi);
  const eta = y - 1 - Math.cos(phi);
  const rho = 0.25 * (2 + Math.sqrt(xi * xi + eta * eta));
  if (rho <= 1) {
    const u = Math.acos(rho);
    const { tau: t, omega: v } = tauOmega(u, -u, xi, eta, phi);
    if (t >= -ZERO && v <= ZERO) return { ok: true, t, u, v };
  }
  return FAIL;
}

function LpRumLumRp(x: number, y: number, phi: number): Cand {
  const xi = x + Math.sin(phi);
  const eta = y - 1 - Math.cos(phi);
  const rho = (20 - xi * xi - eta * eta) / 16;
  if (rho >= 0 && rho <= 1) {
    const u = -Math.acos(rho);
    if (u >= -HALF_PI) {
      const { tau: t, omega: v } = tauOmega(u, u, xi, eta, phi);
      if (t >= -ZERO && v >= -ZERO) return { ok: true, t, u, v };
    }
  }
  return FAIL;
}

function CCCC(best: Best, x: number, y: number, phi: number): void {
  let c = LpRupLumRm(x, y, phi);
  if (c.ok) consider(best, 2, [c.t, c.u, -c.u, c.v]);
  c = LpRupLumRm(-x, y, -phi);
  if (c.ok) consider(best, 2, [-c.t, -c.u, c.u, -c.v]);
  c = LpRupLumRm(x, -y, -phi);
  if (c.ok) consider(best, 3, [c.t, c.u, -c.u, c.v]);
  c = LpRupLumRm(-x, -y, phi);
  if (c.ok) consider(best, 3, [-c.t, -c.u, c.u, -c.v]);

  c = LpRumLumRp(x, y, phi);
  if (c.ok) consider(best, 2, [c.t, c.u, c.u, c.v]);
  c = LpRumLumRp(-x, y, -phi);
  if (c.ok) consider(best, 2, [-c.t, -c.u, -c.u, -c.v]);
  c = LpRumLumRp(x, -y, -phi);
  if (c.ok) consider(best, 3, [c.t, c.u, c.u, c.v]);
  c = LpRumLumRp(-x, -y, phi);
  if (c.ok) consider(best, 3, [-c.t, -c.u, -c.u, -c.v]);
}

// ---- formula 8.9 (CCSC) ------------------------------------------------------

function LpRmSmLm(x: number, y: number, phi: number): Cand {
  const xi = x - Math.sin(phi);
  const eta = y - 1 + Math.cos(phi);
  const { r: rho, theta } = polar(xi, eta);
  if (rho >= 2) {
    const r = Math.sqrt(rho * rho - 4);
    const u = 2 - r;
    const t = mod2pi(theta + Math.atan2(r, -2));
    const v = mod2pi(phi - HALF_PI - t);
    if (t >= -ZERO && u <= ZERO && v <= ZERO) return { ok: true, t, u, v };
  }
  return FAIL;
}

function LpRmSmRm(x: number, y: number, phi: number): Cand {
  const xi = x + Math.sin(phi);
  const eta = y - 1 - Math.cos(phi);
  const { r: rho, theta } = polar(-eta, xi);
  if (rho >= 2) {
    const t = theta;
    const u = 2 - rho;
    const v = mod2pi(t + HALF_PI - phi);
    if (t >= -ZERO && u <= ZERO && v <= ZERO) return { ok: true, t, u, v };
  }
  return FAIL;
}

function CCSC(best: Best, x: number, y: number, phi: number): void {
  const Q = HALF_PI;
  let c = LpRmSmLm(x, y, phi);
  if (c.ok) consider(best, 4, [c.t, -Q, c.u, c.v]);
  c = LpRmSmLm(-x, y, -phi);
  if (c.ok) consider(best, 4, [-c.t, Q, -c.u, -c.v]);
  c = LpRmSmLm(x, -y, -phi);
  if (c.ok) consider(best, 5, [c.t, -Q, c.u, c.v]);
  c = LpRmSmLm(-x, -y, phi);
  if (c.ok) consider(best, 5, [-c.t, Q, -c.u, -c.v]);

  c = LpRmSmRm(x, y, phi);
  if (c.ok) consider(best, 8, [c.t, -Q, c.u, c.v]);
  c = LpRmSmRm(-x, y, -phi);
  if (c.ok) consider(best, 8, [-c.t, Q, -c.u, -c.v]);
  c = LpRmSmRm(x, -y, -phi);
  if (c.ok) consider(best, 9, [c.t, -Q, c.u, c.v]);
  c = LpRmSmRm(-x, -y, phi);
  if (c.ok) consider(best, 9, [-c.t, Q, -c.u, -c.v]);

  // backwards
  const xb = x * Math.cos(phi) + y * Math.sin(phi);
  const yb = x * Math.sin(phi) - y * Math.cos(phi);
  c = LpRmSmLm(xb, yb, phi);
  if (c.ok) consider(best, 6, [c.v, c.u, -Q, c.t]);
  c = LpRmSmLm(-xb, yb, -phi);
  if (c.ok) consider(best, 6, [-c.v, -c.u, Q, -c.t]);
  c = LpRmSmLm(xb, -yb, -phi);
  if (c.ok) consider(best, 7, [c.v, c.u, -Q, c.t]);
  c = LpRmSmLm(-xb, -yb, phi);
  if (c.ok) consider(best, 7, [-c.v, -c.u, Q, -c.t]);

  c = LpRmSmRm(xb, yb, phi);
  if (c.ok) consider(best, 10, [c.v, c.u, -Q, c.t]);
  c = LpRmSmRm(-xb, yb, -phi);
  if (c.ok) consider(best, 10, [-c.v, -c.u, Q, -c.t]);
  c = LpRmSmRm(xb, -yb, -phi);
  if (c.ok) consider(best, 11, [c.v, c.u, -Q, c.t]);
  c = LpRmSmRm(-xb, -yb, phi);
  if (c.ok) consider(best, 11, [-c.v, -c.u, Q, -c.t]);
}

// ---- formula 8.11 (CCSCC) ----------------------------------------------------

function LpRmSLmRp(x: number, y: number, phi: number): Cand {
  const xi = x + Math.sin(phi);
  const eta = y - 1 - Math.cos(phi);
  const { r: rho } = polar(xi, eta);
  if (rho >= 2) {
    const u = 4 - Math.sqrt(rho * rho - 4);
    if (u <= ZERO) {
      const t = mod2pi(
        Math.atan2((4 - u) * xi - 2 * eta, -2 * xi + (u - 4) * eta),
      );
      const v = mod2pi(t - phi);
      if (t >= -ZERO && v >= -ZERO) return { ok: true, t, u, v };
    }
  }
  return FAIL;
}

function CCSCC(best: Best, x: number, y: number, phi: number): void {
  const Q = HALF_PI;
  let c = LpRmSLmRp(x, y, phi);
  if (c.ok) consider(best, 16, [c.t, -Q, c.u, -Q, c.v]);
  c = LpRmSLmRp(-x, y, -phi);
  if (c.ok) consider(best, 16, [-c.t, Q, -c.u, Q, -c.v]);
  c = LpRmSLmRp(x, -y, -phi);
  if (c.ok) consider(best, 17, [c.t, -Q, c.u, -Q, c.v]);
  c = LpRmSLmRp(-x, -y, phi);
  if (c.ok) consider(best, 17, [-c.t, Q, -c.u, Q, -c.v]);
}

function reedsShepp(x: number, y: number, phi: number): Best {
  const best: Best = { typeIndex: 14, values: [0, 0, 0], length: Infinity };
  CSC(best, x, y, phi);
  CCC(best, x, y, phi);
  CCCC(best, x, y, phi);
  CCSC(best, x, y, phi);
  CCSCC(best, x, y, phi);
  return best;
}

/** Shortest Reeds-Shepp path (forward + reverse) from `start` to `goal`. */
export function reedsSheppShortestPath(
  start: Pose,
  goal: Pose,
  radius: number,
): CurvePath {
  const { x, y, phi } = toLocal(start, goal, radius);
  const best = reedsShepp(x, y, phi);
  return buildPath('reeds-shepp', TYPES[best.typeIndex]!, best.values, radius);
}
