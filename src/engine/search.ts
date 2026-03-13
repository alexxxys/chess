import { ChessEngine, makeNullMoveEngine } from './core/chess_engine';
import { evaluateFull } from './eval';
import { bookMove } from './openingBook';
import type { VMove } from './core/movegen';

// ── Constants ────────────────────────────────────────────────────────────────

const INFINITY   = 1_000_000;
const MATE_SCORE = 900_000;

// Null-move
const NULL_MOVE_R         = 3;
const NULL_MOVE_MIN_DEPTH = 3;

// Late Move Reduction
const LMR_MIN_DEPTH  = 3;
const LMR_FULL_MOVES = 4;

// Futility Pruning margins — FIXED (was too aggressive at depth 3)
const FUTILITY_MARGIN = [0, 150, 300]; // index = depth ≤ 2

// Razoring — only at depth 1
const RAZOR_MARGIN_D1 = 350;

// Delta Pruning in QSearch
const DELTA_MARGIN = 200;

// Aspiration window
const ASP_WINDOW = 40;

// TT
const TT_MAX_SIZE = 500_000;

// Extensions
const CHECK_EXTENSION = 1;

// ── Transposition Table (keyed by Zobrist hash, not FEN string) ──────────────

interface TTEntry {
  hash:     number;
  depth:    number;
  score:    number;
  flag:     'exact' | 'lower' | 'upper';
  bestMove: string | null;
  age:      number;
}

const TT = new Map<number, TTEntry>();
let _searchAge = 0;

function ttGet(hash: number): TTEntry | undefined {
  const e = TT.get(hash);
  if (!e || e.hash !== hash) return undefined; // hash collision guard
  return e;
}

function ttStore(hash: number, depth: number, score: number, flag: TTEntry['flag'], bestMove: string | null) {
  const existing = TT.get(hash);
  if (!existing || existing.depth <= depth || existing.age < _searchAge - 4) {
    if (TT.size >= TT_MAX_SIZE) {
      for (const [k, v] of TT) {
        if (v.age < _searchAge - 4) TT.delete(k);
        if (TT.size < TT_MAX_SIZE * 0.8) break;
      }
    }
    TT.set(hash, { hash, depth, score, flag, bestMove, age: _searchAge });
  }
}

// ── Killer Moves ─────────────────────────────────────────────────────────────

const MAX_PLY = 64;
const killers: Array<[string | null, string | null]> =
  Array.from({ length: MAX_PLY + 1 }, () => [null, null]);

function storeKiller(ply: number, move: string) {
  if (killers[ply][0] !== move) { killers[ply][1] = killers[ply][0]; killers[ply][0] = move; }
}

// ── History Heuristic ─────────────────────────────────────────────────────────

const PIECE_INDEX: Record<string, number> = {
  wp: 0, wn: 1, wb: 2, wr: 3, wq: 4, wk: 5,
  bp: 6, bn: 7, bb: 8, br: 9, bq: 10, bk: 11,
};
const history: number[][] = Array.from({ length: 12 }, () => new Array(64).fill(0));

function histIdx(color: string, piece: string) { return PIECE_INDEX[`${color}${piece}`] ?? 0; }
function sqIdx(sq: string) { return (8 - parseInt(sq[1])) * 8 + (sq.charCodeAt(0) - 97); }
function getHistory(m: VMove) { return history[histIdx(m.color, m.piece)][sqIdx(m.to)]; }
function addHistory(m: VMove, depth: number) {
  const pi = histIdx(m.color, m.piece), si = sqIdx(m.to);
  history[pi][si] += depth * depth;
  if (history[pi][si] > 1_000_000) {
    for (let i = 0; i < 12; i++) history[i] = history[i].map(v => v >> 1);
  }
}

// ── Countermove Table ─────────────────────────────────────────────────────────

const counterMoves: Record<string, string | null> = {};
function getCounter(lastMove: string | null) { return lastMove ? (counterMoves[lastMove] ?? null) : null; }
function storeCounter(lastMove: string | null, reply: string) { if (lastMove) counterMoves[lastMove] = reply; }

// ── Material values for SEE ordering ─────────────────────────────────────────

const PIECE_VALUE: Record<string, number> = { p: 100, n: 320, b: 330, r: 500, q: 900, k: 20000 };

// ── Move ordering ─────────────────────────────────────────────────────────────

function scoreMove(m: VMove, ttBest: string | null, ply: number, lastMove: string | null): number {
  if (m.lan === ttBest) return 10_000_000;
  if (m.captured) {
    const seeScore = (PIECE_VALUE[m.captured] ?? 0) - (PIECE_VALUE[m.piece] ?? 0);
    return seeScore >= 0 ? 5_000_000 + seeScore : 1_000_000 + seeScore;
  }
  if (m.promotion) return 4_000_000;
  if (killers[ply]?.[0] === m.lan) return 3_000_000;
  if (killers[ply]?.[1] === m.lan) return 2_900_000;
  if (m.lan === getCounter(lastMove)) return 2_800_000;
  return getHistory(m);
}

function orderMoves(moves: VMove[], ttBest: string | null, ply: number, lastMove: string | null): VMove[] {
  return moves.sort((a, b) => scoreMove(b, ttBest, ply, lastMove) - scoreMove(a, ttBest, ply, lastMove));
}

// ── Quiescence Search ─────────────────────────────────────────────────────────

function quiesce(chess: ChessEngine, alpha: number, beta: number): number {
  if (_abortFlag) return 0;
  _nodeCount++;

  const stand = evaluateFull(chess as any); // eval uses board() which we implement
  if (stand >= beta) return beta;
  if (stand + DELTA_MARGIN + 900 < alpha) return alpha;
  if (stand > alpha) alpha = stand;

  const moves = chess.moves({ verbose: true });
  const good = moves
    .filter(m => m.captured || m.promotion)
    .filter(m => {
      const gain = PIECE_VALUE[m.captured ?? ''] ?? 0;
      const promoGain = m.promotion ? (PIECE_VALUE[m.promotion] ?? 0) - 100 : 0;
      return stand + gain + promoGain + DELTA_MARGIN >= alpha;
    });
  good.sort((a, b) => (PIECE_VALUE[b.captured ?? ''] ?? 0) - (PIECE_VALUE[a.captured ?? ''] ?? 0));

  for (const move of good) {
    chess.move(move);
    const score = -quiesce(chess, -beta, -alpha);
    chess.undo();
    if (score >= beta) return beta;
    if (score > alpha) alpha = score;
  }
  return alpha;
}

// ── Alpha-Beta with all optimizations + Phase 1 bug fixes ────────────────────

function alphaBeta(
  chess: ChessEngine,
  depth: number,
  alpha: number,
  beta: number,
  isPV: boolean,
  nullOk: boolean,
  ply: number,
  lastMove: string | null,
): number {
  if (_abortFlag) return 0;
  _nodeCount++;

  // ─ Draw detection (NEW: was missing) ──────────────────────────────────────
  if (chess.isDraw()) return 0; // 50-move / repetition / stalemate

  const hash = chess.posHash(); // Zobrist hash, not FEN string!

  // ─ Check Extension BEFORE TT probe (BUG FIX) ──────────────────────────────
  const inCheck = chess.inCheck();
  if (inCheck) depth += CHECK_EXTENSION;

  if (depth <= 0) return quiesce(chess, alpha, beta);

  // ─ TT lookup ──────────────────────────────────────────────────────────────
  const ttEntry = ttGet(hash);
  const ttBest = ttEntry?.bestMove ?? null;

  if (ttEntry && ttEntry.depth >= depth && !isPV) {
    const s = ttEntry.score;
    if (ttEntry.flag === 'exact') return s;
    if (ttEntry.flag === 'lower' && s > alpha) alpha = s;
    if (ttEntry.flag === 'upper' && s < beta)  beta  = s;
    if (alpha >= beta) return s;
  }

  const moves = chess.moves({ verbose: true });
  if (moves.length === 0) return inCheck ? -MATE_SCORE + ply : 0;

  // ─ Static eval (computed once, not twice) ──────────────────────────────────
  // Cache it from TT if we have it; otherwise compute once
  const staticEval = !inCheck ? evaluateFull(chess as any) : -INFINITY;

  // ─ Razoring (only depth 1, fixed from depth ≤ 3) ──────────────────────────
  if (!isPV && !inCheck && depth === 1 && staticEval + RAZOR_MARGIN_D1 < alpha) {
    const q = quiesce(chess, alpha, beta);
    if (q < alpha) return q;
  }

  // ─ Null Move Pruning ───────────────────────────────────────────────────────
  if (nullOk && !isPV && !inCheck && depth >= NULL_MOVE_MIN_DEPTH) {
    const mat = chess.nonPawnMaterial();
    if (mat > 800) { // has at least some piece (not bare king)
      const nullChess = makeNullMoveEngine(chess);
      if (nullChess) {
        const R = depth >= 6 ? NULL_MOVE_R + 1 : NULL_MOVE_R;
        const nullScore = -alphaBeta(nullChess, depth - 1 - R, -beta, -beta + 1, false, false, ply + 1, null);
        if (nullScore >= beta) return beta;
      }
    }
  }

  // ─ Internal Iterative Deepening ────────────────────────────────────────────
  if (isPV && !ttBest && depth >= 6) {
    alphaBeta(chess, depth - 2, alpha, beta, true, false, ply, lastMove);
  }

  const ordered = orderMoves(moves, ttBest ?? (ttGet(hash)?.bestMove ?? null), ply, lastMove);

  let bestScore = -INFINITY;
  let bestMove: string | null = null;
  const origAlpha = alpha;
  let moveCount = 0;

  for (const move of ordered) {
    if (_abortFlag) break;
    moveCount++;

    const isCapture = !!move.captured;
    const isPromo   = !!move.promotion;

    // ─ Futility Pruning (depth ≤ 2 only, tighter margins) ─────────────────
    if (
      !isPV && !inCheck && !isCapture && !isPromo &&
      depth <= 2 && depth >= 1 &&
      staticEval + (FUTILITY_MARGIN[depth] ?? 150) <= alpha
    ) {
      continue;
    }

    chess.move(move);
    const givesCheck = chess.inCheck();
    let score: number;

    // ─ LMR (adaptive) ──────────────────────────────────────────────────────
    const canReduce =
      depth >= LMR_MIN_DEPTH && moveCount > LMR_FULL_MOVES &&
      !isPV && !isCapture && !isPromo && !inCheck && !givesCheck;

    if (canReduce) {
      const reduction = moveCount > 12 ? 3 : moveCount > 6 ? 2 : 1;
      score = -alphaBeta(chess, depth - 1 - reduction, -alpha - 1, -alpha, false, true, ply + 1, move.lan);
      if (score > alpha) score = -alphaBeta(chess, depth - 1, -beta, -alpha, false, true, ply + 1, move.lan);
    } else if (!isPV || moveCount > 1) {
      score = -alphaBeta(chess, depth - 1, -alpha - 1, -alpha, false, true, ply + 1, move.lan);
      if (score > alpha && score < beta) score = -alphaBeta(chess, depth - 1, -beta, -alpha, true, true, ply + 1, move.lan);
    } else {
      score = -alphaBeta(chess, depth - 1, -beta, -alpha, true, true, ply + 1, move.lan);
    }

    chess.undo();

    if (score > bestScore) { bestScore = score; bestMove = move.lan; }
    if (score > alpha) alpha = score;

    if (alpha >= beta) {
      if (!isCapture && move.lan) {
        storeKiller(ply, move.lan);
        addHistory(move, depth);
        storeCounter(lastMove, move.lan);
      }
      break;
    }
  }

  const flag: TTEntry['flag'] =
    bestScore <= origAlpha ? 'upper' : bestScore >= beta ? 'lower' : 'exact';
  ttStore(hash, depth, bestScore, flag, bestMove);
  return bestScore;
}

// ── Public API ───────────────────────────────────────────────────────────────

export interface EngineResult {
  move: string | null;
  score: number;
  depth: number;
  nodes: number;
  thinkMs: number;
}

let _abortFlag = false;
let _nodeCount = 0;

export function abortSearch() { _abortFlag = true; }

export function getBestMove(fen: string, timeLimitMs: number, maxDepth = 64): EngineResult {
  _abortFlag = false;
  _nodeCount = 0;
  _searchAge++;
  const startTime = performance.now();

  // Reset killers
  for (const k of killers) { k[0] = null; k[1] = null; }

  // Opening book
  const book = bookMove(fen);
  if (book) return { move: book, score: 0, depth: 0, nodes: 0, thinkMs: 0 };

  // Use our custom engine
  const root = new ChessEngine(fen);
  if (root.moves().length === 0) return { move: null, score: 0, depth: 0, nodes: 0, thinkMs: 0 };

  let bestMove: string | null = null;
  let bestScore = 0;
  let completedDepth = 0;

  for (let depth = 1; depth <= maxDepth; depth++) {
    if (_abortFlag) break;
    const elapsed = performance.now() - startTime;
    if (elapsed >= timeLimitMs * 0.6 && depth > 1) break;

    const iterChess = new ChessEngine(fen);
    const moves = iterChess.moves({ verbose: true });
    if (!moves.length) break;
    const ordered = orderMoves(moves, bestMove, 0, null);

    let iterBest: string | null = null;
    let iterScore = -INFINITY;

    // Aspiration Windows
    let alphaAsp = depth > 4 ? bestScore - ASP_WINDOW : -INFINITY;
    let betaAsp  = depth > 4 ? bestScore + ASP_WINDOW :  INFINITY;
    let aspRetries = 0;

    while (true) {
      let alpha = alphaAsp;
      const beta = betaAsp;
      let localBest: string | null = null;
      let localScore = -INFINITY;

      for (const move of ordered) {
        if (_abortFlag) break;
        iterChess.move(move);
        _nodeCount++;
        const score = -alphaBeta(iterChess, depth - 1, -beta, -alpha, true, true, 1, move.lan);
        iterChess.undo();
        if (score > localScore) { localScore = score; localBest = move.lan; }
        if (score > alpha) alpha = score;
        if (alpha >= beta) break;
      }

      if (!_abortFlag) {
        if (localScore <= alphaAsp && aspRetries < 2) {
          alphaAsp = aspRetries === 0 ? alphaAsp - ASP_WINDOW * 3 : -INFINITY;
          aspRetries++; continue;
        }
        if (localScore >= betaAsp && aspRetries < 2) {
          betaAsp = aspRetries === 0 ? betaAsp + ASP_WINDOW * 3 : INFINITY;
          aspRetries++; continue;
        }
      }
      iterBest = localBest; iterScore = localScore; break;
    }

    if (!_abortFlag && iterBest) {
      bestMove = iterBest;
      bestScore = iterScore;
      completedDepth = depth;
    }

    if (Math.abs(bestScore) >= MATE_SCORE - 100) break;
  }

  return {
    move: bestMove,
    score: bestScore,
    depth: completedDepth,
    nodes: _nodeCount,
    thinkMs: Math.round(performance.now() - startTime),
  };
}
