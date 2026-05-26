// Reader for the residual MLP ensemble npz emitted by the JAX trainer
// (`demos/scripts/python/train_fit.py`).
//
// On-disk layout (matches `trial_io.save_mlp_ensemble`):
//   n_members.npy : int32 ()
//   n_layers.npy  : int32 ()
//   version.npy   : int32 ()
//   m{i}_l{j}_W.npy : float64 (out, in)
//   m{i}_l{j}_b.npy : float64 (out,)

import { Buffer } from 'node:buffer';
import { readFileSync } from 'node:fs';
import { inflateRawSync } from 'node:zlib';
import type { MLP } from 'kinocat/agent';

// We hand-roll a minimal PKZIP reader because:
//   - the trainer writes STORED (uncompressed) entries (see trial-npz.ts),
//     so DEFLATE support is needed only for npz files numpy might emit
//     compressed (`np.savez_compressed`); we support both.
//   - avoids adding a dependency.

function readZipEntries(buf: Uint8Array): Record<string, Uint8Array> {
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  // Find end-of-central-directory record (search backwards).
  let eocd = -1;
  for (let i = buf.length - 22; i >= Math.max(0, buf.length - 65557); i--) {
    if (view.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
  }
  if (eocd < 0) throw new Error('npz: end-of-central-directory not found');
  const cdEntries = view.getUint16(eocd + 10, true);
  const cdOffset = view.getUint32(eocd + 16, true);
  const out: Record<string, Uint8Array> = {};
  let p = cdOffset;
  for (let i = 0; i < cdEntries; i++) {
    if (view.getUint32(p, true) !== 0x02014b50) throw new Error('npz: bad central dir signature');
    const method = view.getUint16(p + 10, true);
    const compSize = view.getUint32(p + 20, true);
    const uncompSize = view.getUint32(p + 24, true);
    const nameLen = view.getUint16(p + 28, true);
    const extraLen = view.getUint16(p + 30, true);
    const commentLen = view.getUint16(p + 32, true);
    const localOff = view.getUint32(p + 42, true);
    const name = Buffer.from(buf.slice(p + 46, p + 46 + nameLen)).toString('utf8');
    p += 46 + nameLen + extraLen + commentLen;
    // Read local header to locate file data start.
    if (view.getUint32(localOff, true) !== 0x04034b50) throw new Error('npz: bad local header');
    const lNameLen = view.getUint16(localOff + 26, true);
    const lExtraLen = view.getUint16(localOff + 28, true);
    const dataStart = localOff + 30 + lNameLen + lExtraLen;
    const raw = buf.slice(dataStart, dataStart + compSize);
    if (method === 0) {
      out[name] = raw;
    } else if (method === 8) {
      out[name] = new Uint8Array(inflateRawSync(Buffer.from(raw)));
      if (out[name]!.length !== uncompSize) throw new Error(`npz: inflate size mismatch for ${name}`);
    } else {
      throw new Error(`npz: unsupported compression method ${method} for ${name}`);
    }
  }
  return out;
}

function parseNpy(buf: Uint8Array): { dtype: string; shape: number[]; data: Float64Array | Int32Array } {
  // Magic: \x93NUMPY
  if (buf[0] !== 0x93 || buf[1] !== 0x4e) throw new Error('not a .npy file');
  const major = buf[6]!;
  const headerLen = major === 1
    ? buf[8]! | (buf[9]! << 8)
    : (buf[8]! | (buf[9]! << 8) | (buf[10]! << 16) | (buf[11]! << 24));
  const headerStart = major === 1 ? 10 : 12;
  const headerStr = Buffer.from(buf.slice(headerStart, headerStart + headerLen)).toString('binary');
  // crude parse: looking for descr and shape
  const descrMatch = headerStr.match(/'descr':\s*'([^']+)'/);
  const shapeMatch = headerStr.match(/'shape':\s*\(([^)]*)\)/);
  if (!descrMatch || !shapeMatch) throw new Error('npy header parse failed');
  const dtype = descrMatch[1]!;
  const shape = shapeMatch[1]!
    .split(',')
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map((s) => parseInt(s, 10));
  const bodyStart = headerStart + headerLen;
  const body = buf.slice(bodyStart);
  let data: Float64Array | Int32Array;
  if (dtype === '<f8') {
    data = new Float64Array(body.buffer, body.byteOffset, body.byteLength / 8);
  } else if (dtype === '<i4') {
    data = new Int32Array(body.buffer, body.byteOffset, body.byteLength / 4);
  } else {
    throw new Error(`unsupported dtype ${dtype}`);
  }
  return { dtype, shape, data };
}

interface RawEnsemble {
  members: number;
  layers: number;
  layerArrays: { W: Float64Array; b: Float64Array; outDim: number; inDim: number }[][];
}

function readEnsembleFromNpz(path: string): RawEnsemble {
  const raw = readFileSync(path);
  const entries = readZipEntries(new Uint8Array(raw));
  function get(name: string) {
    const b = entries[name];
    if (!b) throw new Error(`missing entry in npz: ${name}`);
    return parseNpy(b);
  }
  const n_members = Number((get('n_members.npy').data as Int32Array)[0]);
  const n_layers = Number((get('n_layers.npy').data as Int32Array)[0]);
  const layerArrays: RawEnsemble['layerArrays'] = [];
  for (let i = 0; i < n_members; i++) {
    const member: RawEnsemble['layerArrays'][number] = [];
    for (let j = 0; j < n_layers; j++) {
      const W = get(`m${i}_l${j}_W.npy`);
      const b = get(`m${i}_l${j}_b.npy`);
      member.push({
        W: W.data as Float64Array,
        b: b.data as Float64Array,
        outDim: W.shape[0]!,
        inDim: W.shape[1]!,
      });
    }
    layerArrays.push(member);
  }
  return { members: n_members, layers: n_layers, layerArrays };
}

/** Construct an MLP from per-layer (W, b) arrays. Layout matches the
 *  hand-coded MLP in `core/src/internal/mlp.ts`: weights row-major
 *  `[out, in]`, biases `[out]`, ReLU hidden + linear output. */
export function deserializeMLPFromArrays(
  layers: { W: Float64Array; b: Float64Array; outDim: number; inDim: number }[],
): MLP {
  if (layers.length === 0) throw new Error('deserializeMLPFromArrays: no layers');
  const inputDim = layers[0]!.inDim;
  const outputDim = layers[layers.length - 1]!.outDim;
  const hiddenDims = layers.slice(0, -1).map((l) => l.outDim);
  return {
    config: { inputDim, hiddenDims, outputDim },
    layers: layers.map((l) => ({
      weights: new Float64Array(l.W),
      biases: new Float64Array(l.b),
      outDim: l.outDim,
      inDim: l.inDim,
    })),
  } as MLP;
}

/** Top-level: read the entire ensemble. */
export function readResidualEnsembleNpz(path: string): MLP[] {
  const raw = readEnsembleFromNpz(path);
  return raw.layerArrays.map((layers) => deserializeMLPFromArrays(layers));
}
