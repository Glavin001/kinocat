import { bench, describe } from 'vitest';
import { reedsSheppShortestPath } from '../src/curves/reeds-shepp';
import { dubinsShortestPath } from '../src/curves/dubins';

const start = { x: 0, y: 0, theta: 0 };
const goal = { x: 7.3, y: -4.1, theta: 2.2 };

describe('analytical curves', () => {
  bench('reedsSheppShortestPath', () => {
    reedsSheppShortestPath(start, goal, 3);
  });
  bench('dubinsShortestPath', () => {
    dubinsShortestPath(start, goal, 3);
  });
});
