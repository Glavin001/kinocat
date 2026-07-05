import { describe, expect, it } from 'vitest';
import { buildDebugReport } from '../app/lib/debug-report';
import {
  buildKinematicLibrary,
  buildLearnedRaceLibraryV2,
  emptyMetrics,
  RACE_START_SPEEDS,
} from '../app/lib/race-primitives-scenarios';
import {
  buildParametricOnlyModel,
  DEFAULT_LEARNED_PARAMS_V2,
  DEFAULT_LEARNABLE_CONFIG,
} from 'kinocat/agent';

describe('buildDebugReport', () => {
  const kinLib = buildKinematicLibrary();
  const v2Model = buildParametricOnlyModel(DEFAULT_LEARNED_PARAMS_V2, DEFAULT_LEARNABLE_CONFIG);
  const v2Lib = buildLearnedRaceLibraryV2(v2Model);

  it('produces markdown with every major section when v2 is loaded', () => {
    const md = buildDebugReport({
      phase: 'racing',
      useV2: true,
      v2Active: true,
      winner: null,
      v2Model,
      v2Meta: {
        trialsUsed: 516,
        openLoopRmsAt1s: 0.922,
        legacyRmsAt1s: 2.421,
        kinematicRmsAt1s: 4.385,
        createdAt: Date.UTC(2026, 4, 23, 14, 29, 13),
      },
      kinematicMetrics: emptyMetrics(),
      learnedMetrics: emptyMetrics(),
      kinematicLapTimes: [40.66],
      learnedLapTimes: [],
      kinematicSectors: [[3.2, 7.5, 12.1, 18.9, 25.0, 30.1, 35.8, 40.66]],
      learnedSectors: [],
      waypointCount: 11,
      kinematicLibrary: kinLib,
      learnedLibrary: v2Lib,
      startSpeeds: RACE_START_SPEEDS,
      plannerConfig: {
        lookaheadCount: 2,
        replanIntervalMs: 300,
        perCarBudgetMs: 120,
        plannerGateRadius: 1.8,
        advanceRadius: 2.5,
        trackerMaxLateralAccel: 12,
      },
      note: 'cyan circles waypoints; pink finishes laps',
    });

    // Has all section headers
    for (const h of [
      '# kinocat /raceprimitives debug report',
      '## Phase',
      '## v2 model',
      '### v2 parameters',
      '### v2 vehicle config',
      '## Race',
      '### KINEMATIC (pink)',
      '### LEARNED (cyan, v2)',
      '## Primitive libraries (action-space resolution)',
      '### Kinematic library',
      '### v2-learned library',
      '## Planner config',
      '## System',
    ]) {
      expect(md, `section "${h}" missing`).toContain(h);
    }
    // Note is included
    expect(md).toContain('cyan circles waypoints; pink finishes laps');
    // Hull-ratio block surfaces the v2-vs-kinematic comparison
    expect(md).toContain('### v2 vs kinematic hull ratio');
    // Lap history is serialized as the expected JSON-array-ish form
    expect(md).toContain('[40.66]');
  });

  it('handles the "no v2 loaded" case gracefully', () => {
    const md = buildDebugReport({
      phase: 'ready',
      useV2: false,
      v2Active: false,
      winner: null,
      v2Model: null,
      v2Meta: null,
      kinematicMetrics: emptyMetrics(),
      learnedMetrics: emptyMetrics(),
      kinematicLapTimes: [],
      learnedLapTimes: [],
      kinematicSectors: [],
      learnedSectors: [],
      waypointCount: 11,
      kinematicLibrary: kinLib,
      learnedLibrary: null,
      startSpeeds: RACE_START_SPEEDS,
      plannerConfig: {
        lookaheadCount: 2,
        replanIntervalMs: 300,
        perCarBudgetMs: 120,
        plannerGateRadius: 1.8,
        advanceRadius: 2.5,
        trackerMaxLateralAccel: 12,
      },
    });
    expect(md).toContain('**Loaded:** no');
    // No "v2-learned library" section when v2 isn't loaded
    expect(md).not.toContain('### v2-learned library');
  });

  it('includes the planner-diagnostics summary per car', () => {
    const m = emptyMetrics();
    m.planDiagnostics = {
      lastReplanMs: 42,
      lastReplanFound: true,
      consecutiveFailedReplans: 0,
      planAgeMs: 250,
      successfulReplans: 188,
      totalReplans: 188,
    };
    const md = buildDebugReport({
      phase: 'racing',
      useV2: false,
      v2Active: false,
      winner: null,
      v2Model: null,
      v2Meta: null,
      kinematicMetrics: m,
      learnedMetrics: emptyMetrics(),
      kinematicLapTimes: [],
      learnedLapTimes: [],
      kinematicSectors: [],
      learnedSectors: [],
      waypointCount: 11,
      kinematicLibrary: kinLib,
      learnedLibrary: null,
      startSpeeds: RACE_START_SPEEDS,
      plannerConfig: {
        lookaheadCount: 2, replanIntervalMs: 300, perCarBudgetMs: 120,
        plannerGateRadius: 1.8, advanceRadius: 2.5, trackerMaxLateralAccel: 12,
      },
    });
    expect(md).toContain('last replan 42 ms (ok)');
    expect(md).toContain('plan age 250 ms');
    expect(md).toContain('success 100% (188/188)');
  });
});
