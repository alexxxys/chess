// ── Legal Move Generator ──────────────────────────────────────────────────────
// Generates pseudo-legal moves, then filters for legality (king not in check).
// Returns encoded 32-bit integers for speed, plus a VMove-compatible array.

import {
  EMPTY, WHITE, BLACK,
  W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
  B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING,
  CR_WK, CR_WQ, CR_BK, CR_BQ,
  MF_NORMAL, MF_DOUBLE, MF_EN_PASSANT, MF_CASTLE_K, MF_CASTLE_Q,
  MF_PROMO_N, MF_PROMO_B, MF_PROMO_R, MF_PROMO_Q,
  encodeMove, moveFrom, moveTo, movePiece, moveFlag,
  sqToAlg, sqRow, sqCol, pieceType,
} from './types';
import { Position } from './position';
import { KNIGHT_ATTACKS, KING_ATTACKS, PAWN_ATTACKS, ROOK_RAYS, BISHOP_RAYS } from './tables';

// ── VMove interface (compatible with existing search.ts) ──────────────────────
export interface VMove {
  lan: string;
  san: string;
  piece: string;
  from: string;
  to: string;
  color: string;
  captured?: string;
  promotion?: string;
  flags: string;
  _encoded: number; // our fast integer form
}

const PIECE_LETTER = ['', 'p', 'n', 'b', 'r', 'q', 'k', 'p', 'n', 'b', 'r', 'q', 'k'];
const PROMO_FLAG_CHAR: Record<number, string> = {
  5: 'n', 6: 'b', 7: 'r', 8: 'q',
};

function makeVMove(pos: Position, encoded: number): VMove {
  const from  = moveFrom(encoded);
  const to    = moveTo(encoded);
  const piece = movePiece(encoded);
  const cap   = pos.board[to]; // note: for ep, captured square is different
  const flag  = moveFlag(encoded);
  const color = pos.turn === WHITE ? 'w' : 'b';

  const fromAlg = sqToAlg(from);
  const toAlg   = sqToAlg(to);
  const promoChar = PROMO_FLAG_CHAR[flag] ?? '';
  const lan = fromAlg + toAlg + promoChar;

  let capturePiece: string | undefined;
  if (flag === MF_EN_PASSANT) capturePiece = 'p';
  else if (cap && cap !== EMPTY) capturePiece = PIECE_LETTER[cap];

  return {
    lan, san: lan, // SAN generation is expensive; use LAN as fallback
    piece: PIECE_LETTER[piece],
    from: fromAlg,
    to: toAlg,
    color,
    captured: capturePiece,
    promotion: promoChar || undefined,
    flags: flag === MF_CASTLE_K ? 'k' : flag === MF_CASTLE_Q ? 'q' :
           flag === MF_EN_PASSANT ? 'e' : cap ? 'c' : 'n',
    _encoded: encoded,
  };
}

// ── Attack detection ──────────────────────────────────────────────────────────
// Returns true if square `sq` is attacked by `byColor`

export function isAttacked(board: Uint8Array, sq: number, byColor: number): boolean {
  const byPawn   = byColor === WHITE ? W_PAWN   : B_PAWN;
  const byKnight = byColor === WHITE ? W_KNIGHT : B_KNIGHT;
  const byBishop = byColor === WHITE ? W_BISHOP : B_BISHOP;
  const byRook   = byColor === WHITE ? W_ROOK   : B_ROOK;
  const byQueen  = byColor === WHITE ? W_QUEEN  : B_QUEEN;
  const byKing   = byColor === WHITE ? W_KING   : B_KING;

  // Pawns: check which squares would attack 'sq' from byColor pawns
  // (white pawn on sq attacks row-1, so a white pawn ATTACKING sq comes from PAWN_ATTACKS[WHITE][sq])
  for (const pSq of PAWN_ATTACKS[byColor][sq]) {
    if (board[pSq] === byPawn) return true;
  }

  // Knights
  for (const nSq of KNIGHT_ATTACKS[sq]) {
    if (board[nSq] === byKnight) return true;
  }

  // King
  for (const kSq of KING_ATTACKS[sq]) {
    if (board[kSq] === byKing) return true;
  }

  // Rook / Queen (orthogonal rays)
  for (const ray of ROOK_RAYS[sq]) {
    for (const rSq of ray) {
      const p = board[rSq];
      if (p !== EMPTY) {
        if (p === byRook || p === byQueen) return true;
        break; // blocked
      }
    }
  }

  // Bishop / Queen (diagonal rays)
  for (const ray of BISHOP_RAYS[sq]) {
    for (const bSq of ray) {
      const p = board[bSq];
      if (p !== EMPTY) {
        if (p === byBishop || p === byQueen) return true;
        break; // blocked
      }
    }
  }

  return false;
}

// ── Pseudo-legal move generation ──────────────────────────────────────────────

function addPawnMoves(pos: Position, sq: number, pseudo: number[]): void {
  const color = pos.turn;
  const piece  = color === WHITE ? W_PAWN : B_PAWN;
  const dir    = color === WHITE ? -1 : 1; // row direction
  const startRow = color === WHITE ? 6 : 1;
  const promoRow = color === WHITE ? 0 : 7;

  const r = sqRow(sq);
  const c = sqCol(sq);

  // Forward one square
  const fwdRow = r + dir;
  if (fwdRow >= 0 && fwdRow < 8) {
    const fwdSq = fwdRow * 8 + c;
    if (pos.board[fwdSq] === EMPTY) {
      if (fwdRow === promoRow) {
        // Promotion
        for (const flag of [MF_PROMO_Q, MF_PROMO_R, MF_PROMO_B, MF_PROMO_N]) {
          pseudo.push(encodeMove(sq, fwdSq, piece, EMPTY, flag));
        }
      } else {
        pseudo.push(encodeMove(sq, fwdSq, piece, EMPTY, MF_NORMAL));
        // Double push from starting row
        if (r === startRow) {
          const dblRow = fwdRow + dir;
          const dblSq  = dblRow * 8 + c;
          if (pos.board[dblSq] === EMPTY) {
            pseudo.push(encodeMove(sq, dblSq, piece, EMPTY, MF_DOUBLE));
          }
        }
      }
    }

    // Captures (including en passant)
    for (const dc of [-1, 1]) {
      const nc = c + dc;
      if (nc < 0 || nc > 7) continue;
      const capSq = fwdRow * 8 + nc;

      // En passant
      if (capSq === pos.epSq) {
        const capPiece = color === WHITE ? B_PAWN : W_PAWN;
        pseudo.push(encodeMove(sq, capSq, piece, capPiece, MF_EN_PASSANT));
      } else {
        const capPiece = pos.board[capSq];
        if (capPiece !== EMPTY && (capPiece > 6) === (color === WHITE)) {
          // Enemy piece
          if (fwdRow === promoRow) {
            for (const flag of [MF_PROMO_Q, MF_PROMO_R, MF_PROMO_B, MF_PROMO_N]) {
              pseudo.push(encodeMove(sq, capSq, piece, capPiece, flag));
            }
          } else {
            pseudo.push(encodeMove(sq, capSq, piece, capPiece, MF_NORMAL));
          }
        }
      }
    }
  }
}

function addKnightMoves(pos: Position, sq: number, pseudo: number[]): void {
  const piece = pos.turn === WHITE ? W_KNIGHT : B_KNIGHT;
  for (const tgt of KNIGHT_ATTACKS[sq]) {
    const p = pos.board[tgt];
    if (p === EMPTY || (p > 6) === (pos.turn === WHITE)) {
      pseudo.push(encodeMove(sq, tgt, piece, p, MF_NORMAL));
    }
  }
}

function addSliderMoves(pos: Position, sq: number, rays: number[][][], piece: number, pseudo: number[]): void {
  for (const ray of rays[sq]) {
    for (const tgt of ray) {
      const p = pos.board[tgt];
      if (p === EMPTY) {
        pseudo.push(encodeMove(sq, tgt, piece, EMPTY, MF_NORMAL));
      } else {
        // Capture enemy
        if ((p > 6) === (pos.turn === WHITE)) {
          pseudo.push(encodeMove(sq, tgt, piece, p, MF_NORMAL));
        }
        break; // blocked regardless
      }
    }
  }
}

function addKingMoves(pos: Position, sq: number, pseudo: number[]): void {
  const piece = pos.turn === WHITE ? W_KING : B_KING;
  for (const tgt of KING_ATTACKS[sq]) {
    const p = pos.board[tgt];
    if (p === EMPTY || (p > 6) === (pos.turn === WHITE)) {
      pseudo.push(encodeMove(sq, tgt, piece, p, MF_NORMAL));
    }
  }

  // Castling
  const opp = 1 - pos.turn;
  if (pos.turn === WHITE) {
    // Kingside: e1-g1 (sq 60, to 62)
    if ((pos.castling & CR_WK) && pos.board[61] === EMPTY && pos.board[62] === EMPTY) {
      if (!isAttacked(pos.board, 60, opp) && !isAttacked(pos.board, 61, opp) && !isAttacked(pos.board, 62, opp)) {
        pseudo.push(encodeMove(60, 62, W_KING, EMPTY, MF_CASTLE_K));
      }
    }
    // Queenside: e1-c1 (sq 60, to 58)
    if ((pos.castling & CR_WQ) && pos.board[59] === EMPTY && pos.board[58] === EMPTY && pos.board[57] === EMPTY) {
      if (!isAttacked(pos.board, 60, opp) && !isAttacked(pos.board, 59, opp) && !isAttacked(pos.board, 58, opp)) {
        pseudo.push(encodeMove(60, 58, W_KING, EMPTY, MF_CASTLE_Q));
      }
    }
  } else {
    // Black kingside: e8-g8 (sq 4, to 6)
    if ((pos.castling & CR_BK) && pos.board[5] === EMPTY && pos.board[6] === EMPTY) {
      if (!isAttacked(pos.board, 4, opp) && !isAttacked(pos.board, 5, opp) && !isAttacked(pos.board, 6, opp)) {
        pseudo.push(encodeMove(4, 6, B_KING, EMPTY, MF_CASTLE_K));
      }
    }
    // Black queenside: e8-c8 (sq 4, to 2)
    if ((pos.castling & CR_BQ) && pos.board[3] === EMPTY && pos.board[2] === EMPTY && pos.board[1] === EMPTY) {
      if (!isAttacked(pos.board, 4, opp) && !isAttacked(pos.board, 3, opp) && !isAttacked(pos.board, 2, opp)) {
        pseudo.push(encodeMove(4, 2, B_KING, EMPTY, MF_CASTLE_Q));
      }
    }
  }
}

// ── Full legal move generation ─────────────────────────────────────────────────

export function generateLegalMoves(pos: Position): number[] {
  const pseudo: number[] = [];

  for (let sq = 0; sq < 64; sq++) {
    const p = pos.board[sq];
    if (p === EMPTY) continue;
    // Only our pieces
    if ((p > 6) !== (pos.turn === BLACK)) continue;

    const pt = pieceType(p);
    switch (pt) {
      case 1: addPawnMoves(pos, sq, pseudo); break;   // PAWN
      case 2: addKnightMoves(pos, sq, pseudo); break; // KNIGHT
      case 3: addSliderMoves(pos, sq, BISHOP_RAYS, p, pseudo); break;
      case 4: addSliderMoves(pos, sq, ROOK_RAYS, p, pseudo); break;
      case 5: // QUEEN = rook + bishop rays
        addSliderMoves(pos, sq, ROOK_RAYS, p, pseudo);
        addSliderMoves(pos, sq, BISHOP_RAYS, p, pseudo);
        break;
      case 6: addKingMoves(pos, sq, pseudo); break;
    }
  }

  // Filter: remove moves that leave our king in check
  const legal: number[] = [];

  for (const m of pseudo) {
    pos.makeMove(m);
    // After makeMove, turn has flipped. Our king = side that just moved = opposite of current turn
    const ourKingNow = pos.turn === WHITE ? pos.bKing : pos.wKing;
    if (!isAttacked(pos.board, ourKingNow, pos.turn)) {
      legal.push(m);
    }
    pos.undoMove();
  }

  return legal;
}

// ── Public API: generate VMove array (compatible with existing search.ts) ─────

export function generateMoves(pos: Position): VMove[] {
  const encoded = generateLegalMoves(pos);
  return encoded.map(m => makeVMove(pos, m));
}

// ── Utilities ─────────────────────────────────────────────────────────────────

export function isInCheck(pos: Position): boolean {
  const kingSq = pos.turn === WHITE ? pos.wKing : pos.bKing;
  return isAttacked(pos.board, kingSq, 1 - pos.turn);
}

export function isInCheckAfterMove(pos: Position): boolean {
  // After makeMove(), our king = opposite of current turn
  const kingSq = pos.turn === WHITE ? pos.bKing : pos.wKing;
  return isAttacked(pos.board, kingSq, pos.turn);
}
