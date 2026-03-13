// ── ONNX Runtime Web — Neural Network Inference ──────────────────────────────
// Loads a chess policy+value ResNet from an .onnx file and runs inference
// inside the Web Worker.
//
// Model I/O (matching train/model.py):
//   Input:  "input"   — Float32[1, 19, 8, 8]   (19 spatial planes)
//   Output: "policy"  — Float32[1, 4672]        (move logits)
//           "value"   — Float32[1]              (position value, tanh)

import * as ort from 'onnxruntime-web';
import { encodeChessEngine, moveToIndex, N_PLANES } from './input';
import type { NNOutput } from '../mcts/mcts';
import type { ChessEngine } from '../core/chess_engine';

// Point ONNX Runtime to its WASM files
ort.env.wasm.wasmPaths = '/node_modules/onnxruntime-web/dist/';

// ── Singleton session ─────────────────────────────────────────────────────────

let _session: ort.InferenceSession | null = null;
let _modelPath: string | null = null;
let _loading: Promise<void> | null = null;

export async function loadModel(modelPath: string): Promise<void> {
  if (_session && _modelPath === modelPath) return;
  if (_loading) { await _loading; return; }

  _loading = (async () => {
    try {
      console.log(`[NN] Loading model: ${modelPath}`);
      _session = await ort.InferenceSession.create(modelPath, {
        executionProviders: ['webgpu', 'wasm'],
        graphOptimizationLevel: 'all',
      });
      _modelPath = modelPath;
      console.log(`[NN] Model loaded. Inputs: ${_session.inputNames}`);
    } catch (err) {
      console.error('[NN] Failed to load model:', err);
      _session = null;
    } finally {
      _loading = null;
    }
  })();

  await _loading;
}

export function isModelLoaded(): boolean { return _session !== null; }

// ── Inference ─────────────────────────────────────────────────────────────────

export async function runInference(
  chess: ChessEngine,
  legalMoves: string[], // LAN moves e.g. "e2e4"
): Promise<NNOutput> {
  if (!_session) throw new Error('Model not loaded');

  // Encode position as spatial planes [1, 19, 8, 8]
  const features    = encodeChessEngine(chess as any);
  const inputTensor = new ort.Tensor('float32', features, [1, N_PLANES, 8, 8]);

  const results = await _session.run({ input: inputTensor });

  const policyLogits = results['policy']?.data as Float32Array;
  const valueData    = results['value']?.data  as Float32Array;

  // Masked softmax over legal moves
  const legalIndices = legalMoves.map(lan => moveToIndex(lan));
  const logits = legalIndices.map(i => (i >= 0 && policyLogits ? policyLogits[i] ?? 0 : 0));
  const maxLogit = Math.max(...logits);
  const exps   = logits.map(l => Math.exp(l - maxLogit));
  const sumExp = exps.reduce((a, b) => a + b, 1e-9);

  const policy: Record<string, number> = {};
  legalMoves.forEach((lan, i) => { policy[lan] = exps[i] / sumExp; });

  const value = valueData ? valueData[0] : 0;
  return { policy, value };
}

// ── NNInference adapter for MCTS ──────────────────────────────────────────────

export async function onnxInference(chess: ChessEngine): Promise<NNOutput> {
  if (!isModelLoaded()) throw new Error('ONNX model not loaded');
  const moves = chess.moves({ verbose: true });
  return runInference(chess, moves.map((m: any) => m.lan));
}
