// ── Web Worker: Chess Engine Search ──────────────────────────────────────────
// Supports three modes:
//   'ab'   — Alpha-Beta (proven, ~2400-2800 ELO) ← current default
//   'mcts' — MCTS + eval fallback (building block, used without NN)
//   'nn'   — MCTS + ONNX neural net (target, ~3000-3500 ELO once model trained)

import { getBestMove, abortSearch } from './search';
import { mctsSearch, abortMCTS }   from './mcts/mcts';
import { loadModel, onnxInference, isModelLoaded } from './nn/network';

interface WorkerRequest {
  fen:        string;
  timeLimitMs: number;
  maxDepth:   number;
  mode?:      'ab' | 'mcts' | 'nn'; // default = 'ab'
  abort?:     boolean;
  loadModel?: string; // path to .onnx model, triggers model loading
}

// Try to load the model at worker init (if public/model.onnx exists)
const MODEL_PATH = '/model.onnx';
loadModel(MODEL_PATH).catch(() => {
  // Normal — model file doesn't exist yet (before training)
  console.log('[Worker] No NN model found at', MODEL_PATH, '— using Alpha-Beta');
});

self.onmessage = async (e: MessageEvent<WorkerRequest>) => {
  const { fen, timeLimitMs, maxDepth, mode = 'ab', abort, loadModel: modelPath } = e.data;

  // Abort signal
  if (abort) { abortSearch(); abortMCTS(); return; }

  // Model load request
  if (modelPath) {
    try {
      await loadModel(modelPath);
      self.postMessage({ modelLoaded: true, path: modelPath });
    } catch {
      self.postMessage({ modelLoaded: false, path: modelPath });
    }
    return;
  }

  try {
    // Choose engine based on mode (auto-upgrade to NN if model is ready)
    const effectiveMode = (mode === 'nn' || (mode === 'mcts' && isModelLoaded()))
      ? 'nn'
      : mode;

    if (effectiveMode === 'nn') {
      // MCTS + Neural Network
      const result = await mctsSearch(fen, timeLimitMs, onnxInference);
      self.postMessage({ move: result.move, score: result.score, depth: result.depth, nodes: result.simulations, thinkMs: result.thinkMs });

    } else if (effectiveMode === 'mcts') {
      // MCTS + eval fallback (no NN)
      const result = await mctsSearch(fen, timeLimitMs);
      self.postMessage({ move: result.move, score: result.score, depth: result.depth, nodes: result.simulations, thinkMs: result.thinkMs });

    } else {
      // Alpha-Beta (default, strongest without NN)
      const result = getBestMove(fen, timeLimitMs, maxDepth);
      self.postMessage(result);
    }

  } catch (err) {
    console.error('[Worker] Engine error, falling back to AB:', err);
    const result = getBestMove(fen, Math.min(timeLimitMs, 2000), maxDepth);
    self.postMessage(result);
  }
};
