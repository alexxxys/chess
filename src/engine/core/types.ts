// ── Piece constants ───────────────────────────────────────────────────────────
// Square 0 = a8 (top-left), square 63 = h1 (bottom-right) — chess.js convention
// row = sq >> 3, col = sq & 7; row 0 = rank8, row 7 = rank1

export const EMPTY    = 0;
export const W_PAWN   = 1;  export const W_KNIGHT = 2;  export const W_BISHOP = 3;
export const W_ROOK   = 4;  export const W_QUEEN  = 5;  export const W_KING   = 6;
export const B_PAWN   = 7;  export const B_KNIGHT = 8;  export const B_BISHOP = 9;
export const B_ROOK   = 10; export const B_QUEEN  = 11; export const B_KING   = 12;

export const WHITE = 0;
export const BLACK = 1;

// Piece type without color (1-6)
export function pieceType(p: number): number { return p > 6 ? p - 6 : p; }
export function pieceColor(p: number): number { return p > 6 ? BLACK : WHITE; }
export function coloredPiece(type: number, color: number): number {
  return color === WHITE ? type : type + 6;
}

// Castling right bits
export const CR_WK = 1; // White kingside
export const CR_WQ = 2; // White queenside
export const CR_BK = 4; // Black kingside
export const CR_BQ = 8; // Black queenside

// Move flags
export const MF_NORMAL     = 0;
export const MF_DOUBLE     = 1; // double pawn push
export const MF_EN_PASSANT = 2;
export const MF_CASTLE_K   = 3;
export const MF_CASTLE_Q   = 4;
export const MF_PROMO_N    = 5;
export const MF_PROMO_B    = 6;
export const MF_PROMO_R    = 7;
export const MF_PROMO_Q    = 8;

// ── Move encoding (32-bit integer) ────────────────────────────────────────────
// bits  0-5:  from square (0-63)
// bits  6-11: to square (0-63)
// bits 12-15: moved piece (0-12)
// bits 16-19: captured piece (0-12)
// bits 20-23: move flag (0-8)

export function encodeMove(from: number, to: number, piece: number, captured: number, flag: number): number {
  return from | (to << 6) | (piece << 12) | (captured << 16) | (flag << 20);
}
export function moveFrom(m: number):     number { return m & 63; }
export function moveTo(m: number):       number { return (m >>> 6) & 63; }
export function movePiece(m: number):    number { return (m >>> 12) & 15; }
export function moveCaptured(m: number): number { return (m >>> 16) & 15; }
export function moveFlag(m: number):     number { return (m >>> 20) & 15; }
export function isPromotion(m: number):  boolean { const f = moveFlag(m); return f >= MF_PROMO_N && f <= MF_PROMO_Q; }
export function promoType(m: number): number {
  const f = moveFlag(m);
  return [0, 0, 0, 0, 0, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN][f] ?? 0;
}

// ── Square utilities ──────────────────────────────────────────────────────────

export function sqRank(sq: number): number { return 7 - (sq >> 3); }   // rank 1-8 (1=bottom)
export function sqFile(sq: number): number { return sq & 7; }           // 0=a ... 7=h
export function sqRow(sq: number):  number { return sq >> 3; }          // row 0=top(rank8)
export function sqCol(sq: number):  number { return sq & 7; }

export function sqFromAlg(alg: string): number {
  const col = alg.charCodeAt(0) - 97;          // 'a'=0
  const row = 8 - parseInt(alg[1]);            // '8'=0
  return row * 8 + col;
}
export function sqToAlg(sq: number): string {
  return String.fromCharCode(97 + sqCol(sq)) + String(8 - sqRow(sq));
}

// Material values (for ordering / eval bootstrap)
export const MATERIAL_VALUE = [0, 100, 320, 330, 500, 900, 20000,
                                  100, 320, 330, 500, 900, 20000];

// Piece chars for FEN
export const PIECE_CHAR = ' PNBRQKpnbrqk';
export const CHAR_TO_PIECE: Record<string, number> = {
  P: W_PAWN, N: W_KNIGHT, B: W_BISHOP, R: W_ROOK, Q: W_QUEEN, K: W_KING,
  p: B_PAWN, n: B_KNIGHT, b: B_BISHOP, r: B_ROOK, q: B_QUEEN, k: B_KING,
};
