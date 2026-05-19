import type { HumanoidAgent } from './types';

export function defaultHumanoidAgent(overrides: Partial<HumanoidAgent> = {}): HumanoidAgent {
  return { kind: 'humanoid', radius: 0.35, maxSpeed: 4, ...overrides };
}
