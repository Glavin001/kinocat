// Trial-bundle npz writer for the JAX trainer handoff.
//
// Schema must match `demos/scripts/python/trial_io.py`. Bumps to the
// `version` field require touching both sides in lockstep.
//
// .npz is a zip (PKZIP STORED, no compression) of .npy files. Each .npy
// has a small header + raw little-endian array bytes. This module
// implements the minimum needed; we deliberately don't add a dependency.

import { Buffer } from 'node:buffer';
import { writeFileSync } from 'node:fs';
import { crc32 } from 'node:zlib';
import type { Trial } from 'kinocat/learning';
import { encodeConfigOneHot } from 'kinocat/agent';
import type { CarKinematicState, LearnableVehicleConfig, WheeledCarControls } from 'kinocat/agent';

// Bump on incompatible schema changes — must match TRIAL_NPZ_VERSION in
// demos/scripts/python/trial_io.py. v2: config layout changed from 3
// fields (chassisMass, wheelBase, frictionSlip) to the full 13-dim
// `encodeConfigOneHot` vector so the MLP featurisation can match
// `buildMLPInput` exactly on the Python side.
export const TRIAL_NPZ_VERSION = 2;
export const CONFIG_DIM = 13;

type CarTrial = Trial<CarKinematicState, WheeledCarControls, LearnableVehicleConfig>;

// ---- .npy encoding ---------------------------------------------------------

type Dtype = '<f8' | '<i4';
const DTYPE_BPE: Record<Dtype, number> = { '<f8': 8, '<i4': 4 };

function npyHeader(dtype: Dtype, shape: number[]): Buffer {
  const descr = dtype;
  const shapeStr = shape.length === 0 ? '()' : `(${shape.join(', ')}${shape.length === 1 ? ',' : ''})`;
  let header = `{'descr': '${descr}', 'fortran_order': False, 'shape': ${shapeStr}, }`;
  // Pad so the full header (magic+vers+len+header+\n) aligns to 64 bytes.
  const preamble = 10; // \x93NUMPY\x01\x00<2-byte len>
  let total = preamble + header.length + 1;
  const pad = (64 - (total % 64)) % 64;
  header = header + ' '.repeat(pad) + '\n';
  const headerLen = header.length;
  const buf = Buffer.alloc(preamble + headerLen);
  buf.write('\x93NUMPY', 0, 'binary');
  buf.writeUInt8(1, 6);              // major version
  buf.writeUInt8(0, 7);              // minor version
  buf.writeUInt16LE(headerLen, 8);
  buf.write(header, 10, 'binary');
  return buf;
}

function npyBody(dtype: Dtype, data: ArrayLike<number>): Buffer {
  const bpe = DTYPE_BPE[dtype];
  const buf = Buffer.alloc(bpe * data.length);
  if (dtype === '<f8') {
    for (let i = 0; i < data.length; i++) buf.writeDoubleLE(data[i]!, i * 8);
  } else if (dtype === '<i4') {
    for (let i = 0; i < data.length; i++) buf.writeInt32LE(data[i]! | 0, i * 4);
  }
  return buf;
}

function encodeNpy(dtype: Dtype, shape: number[], data: ArrayLike<number>): Buffer {
  return Buffer.concat([npyHeader(dtype, shape), npyBody(dtype, data)]);
}

// ---- minimal PKZIP STORED (no compression) writer --------------------------

interface ZipEntry {
  name: string;
  data: Buffer;
}

function dosDateTime(): { date: number; time: number } {
  // Use a fixed epoch (2026-01-01 00:00) for reproducibility.
  return { date: ((2026 - 1980) << 9) | (1 << 5) | 1, time: 0 };
}

function writeZip(entries: ZipEntry[]): Buffer {
  const { date, time } = dosDateTime();
  const localHeaders: Buffer[] = [];
  const fileDatas: Buffer[] = [];
  const centralHeaders: Buffer[] = [];
  let offset = 0;
  for (const e of entries) {
    const nameBuf = Buffer.from(e.name, 'utf8');
    const crc = crc32(e.data);
    const size = e.data.length;

    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0); // local file header signature
    local.writeUInt16LE(20, 4);          // version needed
    local.writeUInt16LE(0, 6);           // flags
    local.writeUInt16LE(0, 8);           // method = STORED
    local.writeUInt16LE(time, 10);
    local.writeUInt16LE(date, 12);
    local.writeUInt32LE(crc, 14);
    local.writeUInt32LE(size, 18);
    local.writeUInt32LE(size, 22);
    local.writeUInt16LE(nameBuf.length, 26);
    local.writeUInt16LE(0, 28);          // extra field length
    localHeaders.push(Buffer.concat([local, nameBuf]));
    fileDatas.push(e.data);

    const central = Buffer.alloc(46);
    central.writeUInt32LE(0x02014b50, 0);
    central.writeUInt16LE(20, 4); central.writeUInt16LE(20, 6);
    central.writeUInt16LE(0, 8);  central.writeUInt16LE(0, 10);
    central.writeUInt16LE(time, 12); central.writeUInt16LE(date, 14);
    central.writeUInt32LE(crc, 16);
    central.writeUInt32LE(size, 20);
    central.writeUInt32LE(size, 24);
    central.writeUInt16LE(nameBuf.length, 28);
    central.writeUInt16LE(0, 30); central.writeUInt16LE(0, 32);
    central.writeUInt16LE(0, 34); central.writeUInt16LE(0, 36);
    central.writeUInt32LE(0, 38);
    central.writeUInt32LE(offset, 42);
    centralHeaders.push(Buffer.concat([central, nameBuf]));

    offset += local.length + nameBuf.length + size;
  }
  const cdStart = offset;
  const cd = Buffer.concat(centralHeaders);
  const cdSize = cd.length;
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0);
  end.writeUInt16LE(0, 4); end.writeUInt16LE(0, 6);
  end.writeUInt16LE(entries.length, 8);
  end.writeUInt16LE(entries.length, 10);
  end.writeUInt32LE(cdSize, 12);
  end.writeUInt32LE(cdStart, 16);
  end.writeUInt16LE(0, 20);
  const parts: Buffer[] = [];
  for (let i = 0; i < entries.length; i++) {
    parts.push(localHeaders[i]!, fileDatas[i]!);
  }
  parts.push(cd, end);
  return Buffer.concat(parts);
}

// ---- trial → flat arrays ---------------------------------------------------

interface FlatTrials {
  N: number; T: number; S: number;
  initStates: Float64Array;     // N×7
  controlsTrace: Float64Array;  // N×T×3
  samples: Float64Array;        // N×S×7
  sampleTimes: Float64Array;    // N×S
  config: Float64Array;         // N×3
  split: Int32Array;            // N
  dt: number;
  sampleEvery: number;
}

/** Pack a heterogeneous list of trials into uniform tensors. Pads
 *  shorter controlsTrace / samples arrays with the last value so all
 *  trials line up to a common T,S. */
export function flattenTrials(trials: CarTrial[]): FlatTrials {
  if (trials.length === 0) {
    throw new Error('flattenTrials: empty list');
  }
  const N = trials.length;
  // Use the longest sample list / controls trace as the common shape.
  let T = 0, S = 0;
  for (const t of trials) {
    if (t.controlsTrace.length > T) T = t.controlsTrace.length;
    if (t.samples.length > S) S = t.samples.length;
  }
  const initStates = new Float64Array(N * 7);
  const controlsTrace = new Float64Array(N * T * 3);
  const samples = new Float64Array(N * S * 7);
  const sampleTimes = new Float64Array(N * S);
  const config = new Float64Array(N * CONFIG_DIM);
  const split = new Int32Array(N);
  let dt = trials[0]!.dt;
  // Sample every: infer from the first trial's sample stride.
  const t0 = trials[0]!;
  const sampleEvery = t0.samples.length >= 2
    ? Math.round((t0.samples[1]!.t - t0.samples[0]!.t) / Math.max(t0.dt, 1e-9))
    : 1;

  for (let i = 0; i < N; i++) {
    const t = trials[i]!;
    const init = (t.initialState as unknown as { state?: CarKinematicState }).state
      ?? (t.initialState as CarKinematicState);
    initStates[i * 7 + 0] = init.x;
    initStates[i * 7 + 1] = init.z;
    initStates[i * 7 + 2] = init.heading;
    initStates[i * 7 + 3] = init.speed;
    initStates[i * 7 + 4] = init.yawRate ?? 0;
    initStates[i * 7 + 5] = init.lateralVelocity ?? 0;
    initStates[i * 7 + 6] = init.t ?? 0;

    // Controls trace, padded with the last value if shorter than T.
    for (let k = 0; k < T; k++) {
      const c = t.controlsTrace[Math.min(k, t.controlsTrace.length - 1)]!;
      const base = (i * T + k) * 3;
      controlsTrace[base + 0] = c.steer;
      controlsTrace[base + 1] = c.driveForce;
      controlsTrace[base + 2] = c.brakeForce;
    }
    // Samples + sample times, padded with the last sample if shorter than S.
    for (let k = 0; k < S; k++) {
      const sm = t.samples[Math.min(k, t.samples.length - 1)]!;
      const st = sm.state;
      const base = (i * S + k) * 7;
      samples[base + 0] = st.x;
      samples[base + 1] = st.z;
      samples[base + 2] = st.heading;
      samples[base + 3] = st.speed;
      samples[base + 4] = st.yawRate ?? 0;
      samples[base + 5] = st.lateralVelocity ?? 0;
      samples[base + 6] = st.t ?? sm.t;
      sampleTimes[i * S + k] = sm.t;
    }
    // Full 13-dim config encoding — mirrors encodeConfigOneHot exactly
    // so the Python side reproduces buildMLPInput's 21-dim layout.
    const cfgVec = encodeConfigOneHot(t.config);
    for (let k = 0; k < CONFIG_DIM; k++) {
      config[i * CONFIG_DIM + k] = cfgVec[k] ?? 0;
    }
    split[i] = t.split === 'val' ? 1 : t.split === 'test' ? 2 : 0;
    if (t.dt > 0) dt = t.dt;
  }
  return { N, T, S, initStates, controlsTrace, samples, sampleTimes, config, split, dt, sampleEvery };
}

/** Write trials to disk as an npz the JAX trainer can consume. */
export function writeTrialsNpz(path: string, trials: CarTrial[]): void {
  const f = flattenTrials(trials);
  const entries: ZipEntry[] = [
    { name: 'init_states.npy', data: encodeNpy('<f8', [f.N, 7], f.initStates) },
    { name: 'controls_trace.npy', data: encodeNpy('<f8', [f.N, f.T, 3], f.controlsTrace) },
    { name: 'samples.npy', data: encodeNpy('<f8', [f.N, f.S, 7], f.samples) },
    { name: 'sample_times.npy', data: encodeNpy('<f8', [f.N, f.S], f.sampleTimes) },
    { name: 'config.npy', data: encodeNpy('<f8', [f.N, CONFIG_DIM], f.config) },
    { name: 'split.npy', data: encodeNpy('<i4', [f.N], f.split) },
    { name: 'dt.npy', data: encodeNpy('<f8', [], [f.dt]) },
    { name: 'sample_every.npy', data: encodeNpy('<i4', [], [f.sampleEvery]) },
    { name: 'version.npy', data: encodeNpy('<i4', [], [TRIAL_NPZ_VERSION]) },
  ];
  const buf = writeZip(entries);
  writeFileSync(path, buf);
}
