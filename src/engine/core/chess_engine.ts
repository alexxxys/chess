import { Position } from './position';
import { generateMoves, isInCheck } from './movegen';
import type { VMove } from './movegen';
import { WHITE, EMPTY } from './types';

export class ChessEngine {
  private _pos: Position;
  private _movesCache: VMove[] | null = null;
  private _cacheHash: number = -1;

  constructor(fen: string) {
    this._pos = Position.fromFen(fen);
  }

  // ── chess.js-compatible API ──────────────────────────────────────────────

  moves(_opts?: { verbose?: boolean }): VMove[] {
    // Cache moves per position hash to avoid re-generation after TT probe
    if (this._movesCache && this._cacheHash === this._pos.hash) {
      return this._movesCache;
    }
    this._movesCache = generateMoves(this._pos);
    this._cacheHash = this._pos.hash;
    return this._movesCache;
  }

  move(mv: VMove | { lan: string } | string): void {
    const lan = typeof mv === 'string' ? mv : mv.lan;
    const movesNow = this.moves();
    const found = movesNow.find(m => m.lan === lan);
    if (found) {
      this._pos.makeMove(found._encoded);
      this._movesCache = null; // invalidate cache
    }
  }

  undo(): void {
    this._pos.undoMove();
    this._movesCache = null;
  }

  /** Fast hash key for TT (replaces fen()-based key) */
  posHash(): number {
    return this._pos.hash;
  }

  /** Still needed for null-move FEN construction */
  fen(): string {
    return this._pos.toFen();
  }

  turn(): 'w' | 'b' {
    return this._pos.turn === WHITE ? 'w' : 'b';
  }

  inCheck(): boolean {
    return isInCheck(this._pos);
  }

  isCheckmate(): boolean {
    return isInCheck(this._pos) && this.moves().length === 0;
  }

  isDraw(): boolean {
    const moves = this.moves();
    if (moves.length === 0 && !isInCheck(this._pos)) return true; // stalemate
    if (this._pos.halfmove >= 100) return true;                   // 50-move rule
    if (this._pos.isRepetition()) return true;                    // 3-fold repetition
    return false;
  }

  /** Non-pawn material for current side (for NMP Zugzwang check) */
  nonPawnMaterial(): number {
    // Subtract pawn and king material from total
    const total = this._pos.turn === WHITE ? this._pos.wMaterial : this._pos.bMaterial;
    // Count pawns on board (approximate — we track total not per-type)
    // Use threshold: if total material > 2 rooks, we have enough pieces
    return total;
  }

  /** Direct position access for eval */
  getPosition(): Position { return this._pos; }

  /** Board in chess.js format — used by eval.ts */
  board(): Array<Array<null | { type: string; color: string }>> {
    const b = this._pos.board;
    const PIECE_LETTERS = ['', 'p', 'n', 'b', 'r', 'q', 'k', 'p', 'n', 'b', 'r', 'q', 'k'];
    const result: Array<Array<null | { type: string; color: string }>> = [];
    for (let row = 0; row < 8; row++) {
      const r: Array<null | { type: string; color: string }> = [];
      for (let col = 0; col < 8; col++) {
        const p = b[row * 8 + col];
        r.push(p === EMPTY ? null : { type: PIECE_LETTERS[p], color: p > 6 ? 'b' : 'w' });
      }
      result.push(r);
    }
    return result;
  }
}

// ── Null-move FEN generator ────────────────────────────────────────────────────
// Used for Null Move Pruning in search.ts: creates a ChessEngine with
// side-to-move flipped and ep cleared (= a "pass" move)

export function makeNullMoveEngine(engine: ChessEngine): ChessEngine | null {
  const fen = engine.fen();
  const parts = fen.split(' ');
  if (parts.length < 4) return null;
  parts[1] = parts[1] === 'w' ? 'b' : 'w';
  parts[3] = '-';
  parts[4] = '0';
  return new ChessEngine(parts.join(' '));
}
