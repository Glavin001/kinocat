import type { MovingObstacle } from '../predict/types';
import type { ObstacleDescriptor } from './protocol';
import { PlanRegistry } from '../predict/plan-registry';
import { asObstacle, constantVelocity } from '../predict/factories';

export function rehydrateObstacle(desc: ObstacleDescriptor): MovingObstacle {
  if (desc.kind === 'plan') {
    const reg = new PlanRegistry();
    reg.publish('_', desc.path);
    return asObstacle(reg.predictNPC('_'), desc.radius);
  }
  return asObstacle(constantVelocity(desc.state, desc.horizon), desc.radius);
}

export function rehydrateObstacles(descs: ObstacleDescriptor[]): MovingObstacle[] {
  return descs.map(rehydrateObstacle);
}
