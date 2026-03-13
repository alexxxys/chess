// ── Position: Board State + Make/Undo Move ────────────────────────────────────
// Core data structure for the custom engine. No chess.js dependency.
// Uses Uint8Array board + incremental Zobrist hash for speed.

import {
  EMPTY, WHITE, BLACK,
  W_PAWN, W_QUEEN, W_KING,
  B_PAWN, B_QUEEN, B_KING,
  CR_WK, CR_WQ, CR_BK, CR_BQ,
  MF_DOUBLE, MF_EN_PASSANT, MF_CASTLE_K, MF_CASTLE_Q,
  MATERIAL_VALUE, CHAR_TO_PIECE, PIECE_CHAR,
  moveFrom, moveTo, movePiece, moveCaptured, moveFlag, isPromotion, promoType,
  sqFromAlg, sqToAlg,
  coloredPiece,
} from './types';
import { computeHash, pieceSquareKey as zPiece, sideKey, castleKey, epKey } from './zobrist';

export interface HistoryEntry {
  move:      number; // encoded move
  castling:  number;
  epSq:      number;
  halfmove:  number;
  captured:  number; // piece captured (including en passant)
  hash:      number;
  wMaterial: number;
  bMaterial: number;
}

export class Position {
  board:     Uint8Array;   // [64] piece on each square (0=empty)
  turn:      number;       // WHITE or BLACK
  castling:  number;       // CR_* bit flags
  epSq:      number;       // en passant target square (-1 if none)
  halfmove:  number;       // 50-move clock
  fullmove:  number;
  hash:      number;       // current Zobrist hash
  wKing:     number;       // white king square
  bKing:     number;       // black king square
  wMaterial: number;       // non-king material for white (for NMP)
  bMaterial: number;       // non-king material for black

  private _history: HistoryEntry[] = [];

  constructor() {
    this.board     = new Uint8Array(64);
    this.turn      = WHITE;
    this.castling  = 0;
    this.epSq      = -1;
    this.halfmove  = 0;
    this.fullmove  = 1;
    this.hash      = 0;
    this.wKing     = 4;    // e1 = sq 60 ... wait, sq 0=a8, so e1 = row7*8+4 = 60
    this.bKing     = 4;
    this.wMaterial = 0;
    this.bMaterial = 0;
  }

  static fromFen(fen: string): Position {
    const pos = new Position();
    pos._parseFen(fen);
    return pos;
  }

  private _parseFen(fen: string): void {
    const parts = fen.trim().split(/\s+/);
    const ranks = parts[0].split('/');

    this.board.fill(EMPTY);
    this.wMaterial = 0;
    this.bMaterial = 0;

    let sq = 0;
    for (const rank of ranks) {
      for (const ch of rank) {
        if (ch >= '1' && ch <= '8') {
          sq += parseInt(ch);
        } else {
          const piece = CHAR_TO_PIECE[ch];
          if (piece !== undefined) {
            this.board[sq] = piece;
            if (piece === W_KING) this.wKing = sq;
            if (piece === B_KING) this.bKing = sq;
            if (piece >= W_PAWN && piece <= W_QUEEN) this.wMaterial += MATERIAL_VALUE[piece];
            if (piece >= B_PAWN && piece <= B_QUEEN) this.bMaterial += MATERIAL_VALUE[piece];
            sq++;
          }
        }
      }
    }

    this.turn     = (parts[1] ?? 'w') === 'w' ? WHITE : BLACK;
    this.castling = 0;
    const cr = parts[2] ?? '-';
    if (cr.includes('K')) this.castling |= CR_WK;
    if (cr.includes('Q')) this.castling |= CR_WQ;
    if (cr.includes('k')) this.castling |= CR_BK;
    if (cr.includes('q')) this.castling |= CR_BQ;

    const epStr = parts[3] ?? '-';
    this.epSq     = epStr === '-' ? -1 : sqFromAlg(epStr);
    this.halfmove = parseInt(parts[4] ?? '0') || 0;
    this.fullmove = parseInt(parts[5] ?? '1') || 1;
    this.hash     = computeHash(this.board, this.turn, this.castling, this.epSq);
    this._history = [];
  }

  toFen(): string {
    let fen = '';
    for (let row = 0; row < 8; row++) {
      let empty = 0;
      for (let col = 0; col < 8; col++) {
        const p = this.board[row * 8 + col];
        if (p === EMPTY) { empty++; }
        else { if (empty) { fen += empty; empty = 0; } fen += PIECE_CHAR[p]; }
      }
      if (empty) fen += empty;
      if (row < 7) fen += '/';
    }
    fen += ' ' + (this.turn === WHITE ? 'w' : 'b');
    let cr = '';
    if (this.castling & CR_WK) cr += 'K';
    if (this.castling & CR_WQ) cr += 'Q';
    if (this.castling & CR_BK) cr += 'k';
    if (this.castling & CR_BQ) cr += 'q';
    fen += ' ' + (cr || '-');
    fen += ' ' + (this.epSq >= 0 ? sqToAlg(this.epSq) : '-');
    fen += ' ' + this.halfmove;
    fen += ' ' + this.fullmove;
    return fen;
  }

  /** Number of plies played (for repetition detection) */
  get plyCount(): number { return this._history.length; }

  /** Undo stack depth */
  get historyLength(): number { return this._history.length; }

  /** Get hash at ply i from start */
  getHistoryHash(idx: number): number { return this._history[idx]?.hash ?? NaN; }

  // ── Make move ──────────────────────────────────────────────────────────────
  makeMove(m: number): void {
    const from    = moveFrom(m);
    const to      = moveTo(m);
    const piece   = movePiece(m);
    const cap     = moveCaptured(m);
    const flag    = moveFlag(m);

    // Save state for undo
    this._history.push({
      move:      m,
      castling:  this.castling,
      epSq:      this.epSq,
      halfmove:  this.halfmove,
      captured:  cap,
      hash:      this.hash,
      wMaterial: this.wMaterial,
      bMaterial: this.bMaterial,
    });

    // Update hash: remove moving piece from 'from'
    this.hash ^= zPiece(piece, from);

    // Remove captured piece (if any) from hash
    if (cap) {
      if (flag === MF_EN_PASSANT) {
        // En passant: captured pawn is one square behind to
        const capSq = to + (this.turn === WHITE ? 8 : -8);
        this.hash ^= zPiece(cap, capSq);
        this.board[capSq] = EMPTY;
      } else {
        this.hash ^= zPiece(cap, to);
      }
      // Update material
      if (this.turn === WHITE) this.bMaterial -= MATERIAL_VALUE[cap];
      else                     this.wMaterial -= MATERIAL_VALUE[cap];
    }

    // Remove old EP key
    if (this.epSq >= 0) { this.hash ^= epKey(this.epSq & 7); }

    // Remove old castling key
    const oldCastle = this.castling;
    if (cap && cap !== EMPTY) {
      // If moving to a rook square, update castling
    }

    // Update board: move piece
    this.board[from] = EMPTY;

    if (isPromotion(m)) {
      const promoPiece = coloredPiece(promoType(m), this.turn);
      this.board[to] = promoPiece;
      this.hash ^= zPiece(promoPiece, to);
      if (this.turn === WHITE) { this.wMaterial += MATERIAL_VALUE[promoPiece] - MATERIAL_VALUE[piece]; }
      else                     { this.bMaterial += MATERIAL_VALUE[promoPiece] - MATERIAL_VALUE[piece]; }
    } else {
      this.board[to] = piece;
      this.hash ^= zPiece(piece, to);
    }

    // Castling: move the rook too
    if (flag === MF_CASTLE_K) {
      const rookFrom = to + 1; // h-file
      const rookTo   = to - 1; // f-file
      const rook = this.board[rookFrom];
      this.hash ^= zPiece(rook, rookFrom);
      this.board[rookFrom] = EMPTY;
      this.board[rookTo]   = rook;
      this.hash ^= zPiece(rook, rookTo);
    } else if (flag === MF_CASTLE_Q) {
      const rookFrom = to - 2; // a-file
      const rookTo   = to + 1; // d-file
      const rook = this.board[rookFrom];
      this.hash ^= zPiece(rook, rookFrom);
      this.board[rookFrom] = EMPTY;
      this.board[rookTo]   = rook;
      this.hash ^= zPiece(rook, rookTo);
    }

    // Update king position
    if (piece === W_KING) this.wKing = to;
    if (piece === B_KING) this.bKing = to;

    // Update castling rights
    this.castling &= CASTLING_RIGHTS_MASK[from] & CASTLING_RIGHTS_MASK[to];

    // Update castling hash
    const newCastle = this.castling;
    for (let bit = 1; bit <= 8; bit <<= 1) {
      const had = oldCastle & bit;
      const has = newCastle & bit;
      if (had !== has) this.hash ^= castleKey(bit);
    }

    // Set new EP square
    if (flag === MF_DOUBLE) {
      const newEp = to + (this.turn === WHITE ? 8 : -8);
      this.epSq = newEp;
      this.hash ^= epKey(newEp & 7);
    } else {
      this.epSq = -1;
    }

    // 50-move clock
    this.halfmove = (cap || piece === W_PAWN || piece === B_PAWN) ? 0 : this.halfmove + 1;

    // Full move number
    if (this.turn === BLACK) this.fullmove++;

    // Flip side
    this.turn ^= 1;
    this.hash ^= sideKey();
  }

  // ── Undo move ─────────────────────────────────────────────────────────────
  undoMove(): void {
    const h = this._history.pop();
    if (!h) return;

    const m    = h.move;
    const from = moveFrom(m);
    const to   = moveTo(m);
    const flag = moveFlag(m);

    // Flip side back
    this.turn ^= 1;

    // Restore state
    this.hash      = h.hash;
    this.castling  = h.castling;
    this.epSq      = h.epSq;
    this.halfmove  = h.halfmove;
    this.wMaterial = h.wMaterial;
    this.bMaterial = h.bMaterial;
    if (this.turn === BLACK) this.fullmove--;

    // Get the piece that moved (handle promotion)
    const movedPiece = movePiece(m);
    const cap        = h.captured;

    // Restore moving piece to 'from'
    this.board[from] = movedPiece;
    this.board[to]   = EMPTY;

    // Restore captured piece
    if (cap) {
      if (flag === MF_EN_PASSANT) {
        const capSq = to + (this.turn === WHITE ? 8 : -8);
        this.board[capSq] = cap;
      } else {
        this.board[to] = cap;
      }
    }

    // Undo castling rook move
    if (flag === MF_CASTLE_K) {
      const rookTo   = to - 1;
      const rookFrom = to + 1;
      this.board[rookFrom] = this.board[rookTo];
      this.board[rookTo]   = EMPTY;
    } else if (flag === MF_CASTLE_Q) {
      const rookTo   = to + 1;
      const rookFrom = to - 2;
      this.board[rookFrom] = this.board[rookTo];
      this.board[rookTo]   = EMPTY;
    }

    // Restore king position
    if (movedPiece === W_KING) this.wKing = from;
    if (movedPiece === B_KING) this.bKing = from;
  }

  /** Check if a repetition has occurred (3-fold) */
  isRepetition(): boolean {
    let count = 0;
    for (let i = this._history.length - 2; i >= 0; i -= 2) {
      if (this._history[i]?.hash === this.hash) {
        count++;
        if (count >= 2) return true; // 3rd occurrence (current = 1st)
      }
      // Stop at irreversible moves
      const h = this._history[i];
      if (h && (h.captured || movePiece(h.move) === W_PAWN || movePiece(h.move) === B_PAWN)) break;
    }
    return false;
  }
}

// Castling rights mask: when a piece moves from/to these squares, clear the bit
// Value: which rights to KEEP after move (AND mask)
const CASTLING_RIGHTS_MASK: Uint8Array = new Uint8Array(64).fill(0xFF);
CASTLING_RIGHTS_MASK[0]  &= ~CR_BQ;  // a8 rook
CASTLING_RIGHTS_MASK[7]  &= ~CR_BK;  // h8 rook
CASTLING_RIGHTS_MASK[56] &= ~CR_WQ;  // a1 rook
CASTLING_RIGHTS_MASK[63] &= ~CR_WK;  // h1 rook
CASTLING_RIGHTS_MASK[4]  &= ~(CR_BK | CR_BQ); // e8 king
CASTLING_RIGHTS_MASK[60] &= ~(CR_WK | CR_WQ); // e1 king
