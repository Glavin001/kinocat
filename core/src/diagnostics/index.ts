// kinocat/diagnostics — generic debug capture + export.
//
// `DebugRecorder<S, C>` is a ring buffer of (real, controls, ghosts) frames
// plus optional domain-specific `extras` (e.g. wheel telemetry for a car,
// aero forces for an airplane). It plugs into any `SceneController<S, C>`
// via the `RecorderHook` interface. JSON / Markdown serialization is
// generic; the caller supplies format projections per domain.

export type { RawFrame, RecorderFormatters, RecorderMeta, DebugStats } from './debug-recorder';
export { DebugRecorder } from './debug-recorder';
