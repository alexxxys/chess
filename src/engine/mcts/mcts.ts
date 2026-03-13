// ── MCTS Search (AlphaZero/LC0 style) ────────────────────────────────────────
// Uses PUCT for selection, eval function for leaf value (replaced by NN later).
// The core loop: Selection → Expansion → Evaluation → Backpropagation.

import { MCTSNode } from './node';
import { ChessEngine } from '../core/chess_engine';
import { evaluateFull } from '../eval';
import { bookMove } from '../openingBook';

// ── Hyperparameters ──────────────────────────────────────────────────────────

/** Exploration constant. Higher = explore more, lower = exploit more. */
const C_PUCT = 1.5;

/**
 * Convert centipawn eval to [-1, +1] value from the perspective of the
 * CURRENT side to move. (+1 = current side wins, -1 = current side loses)
 */
function evalToValue(chess: ChessEngine): number {
  const cp = evaluateFull(chess as any);
  // evaluateFull returns from WHITE's perspective; adjust for current player
  const sign = chess.turn() === 'w' ? 1 : -1;
  return Math.tanh((sign * cp) / 600);
}

// ── Neural network interface (stub — replaced by real NN in Phase 3b) ────────
export interface NNOutput {
  /** Prior probability distribution over legal moves (move.lan → probability) */
  policy: Record<string, number>;
  /** Value estimate from current side's perspective [-1, +1] */
  value: number;
}

export type NNInference = (chess: ChessEngine) => Promise<NNOutput> | NNOutput;

/** Default: uniform policy + tanh(eval) value — used until real NN is loaded */
function defaultNN(chess: ChessEngine): NNOutput {
  const moves = chess.moves({ verbose: true });
  const uniform = 1 / Math.max(moves.length, 1);
  const policy: Record<string, number> = {};
  for (const m of moves) policy[m.lan] = uniform;
  return { policy, value: evalToValue(chess) };
}

// ── MCTS state (kept between iterations for virtual tree reuse) ──────────────

let _mctsRoot: MCTSNode | null = null;
let _mctsRootFen: string | null = null;
let _nodeCount = 0;
let _abortFlag = false;

export function abortMCTS() { _abortFlag = true; }

// ── Core MCTS functions ──────────────────────────────────────────────────────

/** Select leaf node by traversing tree with PUCT. Returns move path taken. */
function select(root: MCTSNode, chess: ChessEngine): MCTSNode[] {
  const path: MCTSNode[] = [root];
  let node = root;

  while (node.children !== null && node.children.length > 0 && !node.isTerminal) {
    const child = node.selectChild(C_PUCT);
    chess.move(child.move!);
    path.push(child);
    node = child;
  }
  return path;
}

/** Expand a leaf node: create child nodes with priors from NN/policy. */
function expand(node: MCTSNode, chess: ChessEngine, nnOutput: NNOutput): void {
  if (node.isTerminal) return;

  const moves = chess.moves({ verbose: true });
  if (moves.length === 0) {
    node.isTerminal = true;
    node.terminalValue = chess.inCheck() ? -1 : 0; // checkmate or stalemate
    return;
  }

  if (chess.isDraw()) {
    node.isTerminal = true;
    node.terminalValue = 0;
    return;
  }

  // Normalise priors from policy output
  let priorSum = 0;
  const priors: number[] = moves.map(m => {
    const p = nnOutput.policy[m.lan] ?? (1 / moves.length);
    priorSum += p;
    return p;
  });

  node.children = moves.map((m, i) =>
    new MCTSNode(m.lan, node, priorSum > 0 ? priors[i] / priorSum : 1 / moves.length)
  );
}


// ── Main MCTS search loop ────────────────────────────────────────────────────

export interface MCTSResult {
  move: string | null;
  score: number;         // value estimate in centipawns
  simulations: number;
  depth: number;
  thinkMs: number;
}

export async function mctsSearch(
  fen: string,
  timeLimitMs: number,
  nnInference: NNInference = defaultNN,
): Promise<MCTSResult> {
  _abortFlag = false;
  _nodeCount = 0;
  const startTime = performance.now();

  // Opening book
  const book = bookMove(fen);
  if (book) return { move: book, score: 0, simulations: 0, depth: 0, thinkMs: 0 };

  // Reuse tree if same position (persistent tree between moves)
  let root: MCTSNode;
  if (_mctsRoot && _mctsRootFen === fen) {
    root = _mctsRoot;
  } else {
    root = new MCTSNode(null, null, 1.0);
    _mctsRootFen = fen;
  }
  _mctsRoot = root;

  const rootChess = new ChessEngine(fen);
  const legalMoves = rootChess.moves({ verbose: true });

  if (legalMoves.length === 0) {
    return { move: null, score: 0, simulations: 0, depth: 0, thinkMs: 0 };
  }

  if (legalMoves.length === 1) {
    return { move: legalMoves[0].lan, score: 0, simulations: 1, depth: 1, thinkMs: 0 };
  }

  // Initial expansion of root if not done
  if (root.children === null) {
    const nnOut = await Promise.resolve(nnInference(rootChess));
    expand(root, rootChess, nnOut);
  }

  let simulations = 0;
  let maxDepth = 0;

  // Main simulation loop — runs until time expires
  while (!_abortFlag) {
    const elapsed = performance.now() - startTime;
    if (elapsed >= timeLimitMs) break;

    // Work on fresh chess instance per simulation (make moves down the tree)
    const chess = new ChessEngine(fen);
    _nodeCount++;

    // 1. Selection: traverse tree along PUCT
    const path = select(root, chess);
    const leaf = path[path.length - 1];
    if (path.length > maxDepth) maxDepth = path.length;

    // 2. Evaluation
    let value: number;

    if (leaf.isTerminal) {
      value = leaf.terminalValue;
    } else if (leaf.visits === 0) {
      // Fresh leaf: evaluate with NN/eval
      const nnOut = await Promise.resolve(nnInference(chess));
      value = nnOut.value;

      // Expand if we'll visit again
      if (root.visits < 10000) {
        expand(leaf, chess, nnOut);
      }
    } else {
      // Already visited: need to expand first, then pick a child
      if (leaf.children === null) {
        const nnOut = await Promise.resolve(nnInference(chess));
        expand(leaf, chess, nnOut);
        value = nnOut.value;
      } else if (leaf.children.length > 0) {
        const child = leaf.selectChild(C_PUCT);
        chess.move(child.move!);
        path.push(child);
        const nnOut = await Promise.resolve(nnInference(chess));
        value = nnOut.value;
        if (child.children === null) expand(child, chess, nnOut);
      } else {
        value = leaf.terminalValue;
      }
    }

    // 3. Backpropagation (path is already in order from root to leaf)
    // We need to flip value alternately for each ply
    let v = value;
    for (let i = path.length - 1; i >= 0; i--) {
      path[i].visits++;
      path[i].totalValue += v;
      v = -v; // flip perspective at each ply
    }

    simulations++;
  }

  // Pick best move: most visited child of root
  if (!root.children || root.children.length === 0) {
    return {
      move: legalMoves[0].lan,
      score: 0, simulations, depth: maxDepth, thinkMs: Math.round(performance.now() - startTime),
    };
  }

  const bestChild = root.bestChild();

  // Convert Q value back to centipawns for display
  const qValue = bestChild.q; // [-1,+1]
  const scoreCp = Math.round(Math.atanh(Math.max(-0.999, Math.min(0.999, qValue))) * 600);

  // Update root for future reuse (tree recycling)
  _mctsRoot = bestChild;
  _mctsRoot.parent = null;
  _mctsRootFen = null; // Will be set correctly on next call

  return {
    move: bestChild.move,
    score: scoreCp,
    simulations,
    depth: maxDepth,
    thinkMs: Math.round(performance.now() - startTime),
  };
}
