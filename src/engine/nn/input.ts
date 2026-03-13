// ── Position → Neural Network Input Encoder ───────────────────────────────────
// Converts a chess position to a Float32Array[19×8×8] spatial tensor.
//
// Planes (19 total):
//   [0..5]   — White pieces: P N B R Q K
//   [6..11]  — Black pieces: P N B R Q K
//   [12]     — White kingside castling (all-1 or all-0 plane)
//   [13]     — White queenside castling
//   [14]     — Black kingside castling
//   [15]     — Black queenside castling
//   [16]     — En passant file (column of ones)
//   [17]     — Side to move (all-1 = white, all-0 = black)
//   [18]     — Bias plane (always all-1)
//
// Square convention: sq 0=a8 (top-left), sq 63=h1 (bottom-right)
// Input tensor shape: [1, 19, 8, 8] — matches Python train/input.py exactly

export const N_PLANES    = 19;
export const BOARD_CELLS = 64;
export const NN_INPUT_SIZE = N_PLANES * BOARD_CELLS; // 1216 (for flat compat)

// Plane indices: WHITE pieces 0-5, BLACK pieces 6-11
// P=0, N=1, B=2, R=3, Q=4, K=5  (×2 for color)
const PIECE_PLANE: Record<number, number> = {
  1: 0, // W_PAWN
  2: 1, // W_KNIGHT
  3: 2, // W_BISHOP
  4: 3, // W_ROOK
  5: 4, // W_QUEEN
  6: 5, // W_KING
  7: 6, // B_PAWN
  8: 7, // B_KNIGHT
  9: 8, // B_BISHOP
 10: 9, // B_ROOK
 11:10, // B_QUEEN
 12:11, // B_KING
};

/**
 * Encode a board position as Float32Array[19×8×8] for ResNet inference.
 * Output layout: plane-major — features[plane * 64 + sq]
 *
 * @param board    Uint8Array[64] piece placement (0=empty, 1-12=pieces)
 * @param turn     0=white, 1=black
 * @param castling CR_* bitmask: bit0=WK, bit1=WQ, bit2=BK, bit3=BQ
 * @param epSq     En passant target square index (-1 = none)
 */
export function encodePosition(
  board: Uint8Array,
  turn: number,
  castling: number,
  epSq: number,
): Float32Array {
  const input = new Float32Array(N_PLANES * BOARD_CELLS);

  // Piece planes 0-11
  for (let sq = 0; sq < 64; sq++) {
    const p = board[sq];
    if (p !== 0) {
      const plane = PIECE_PLANE[p];
      if (plane !== undefined) {
        input[plane * 64 + sq] = 1;
      }
    }
  }

  // Castling planes 12-15 (fill entire 8×8 plane with 1 or 0)
  if (castling & 1) input.fill(1, 12 * 64, 13 * 64); // WK
  if (castling & 2) input.fill(1, 13 * 64, 14 * 64); // WQ
  if (castling & 4) input.fill(1, 14 * 64, 15 * 64); // BK
  if (castling & 8) input.fill(1, 15 * 64, 16 * 64); // BQ

  // En passant file plane 16 (fill entire column)
  if (epSq >= 0) {
    const col = epSq & 7;
    for (let row = 0; row < 8; row++) {
      input[16 * 64 + row * 8 + col] = 1;
    }
  }

  // Side-to-move plane 17 (all-1 = white, all-0 = black)
  if (turn === 0) input.fill(1, 17 * 64, 18 * 64);

  // Bias plane 18 (always all-1)
  input.fill(1, 18 * 64, 19 * 64);

  return input;
}

/**
 * Encode a ChessEngine position into Float32Array[19×8×8].
 */
export function encodeChessEngine(chess: {
  getPosition(): { board: Uint8Array; turn: number; castling: number; epSq: number };
}): Float32Array {
  const pos = chess.getPosition();
  return encodePosition(pos.board, pos.turn, pos.castling, pos.epSq);
}

// ── Move index encoding ────────────────────────────────────────────────────────
// Policy space: from_sq * 73 + to_idx
// from_sq ∈ [0,63], to_idx ∈ [0,72] (0-63=target square, 64-67=promo N/B/R/Q)
// Max index: 63 * 73 + 72 = 4671 → POLICY_SIZE = 4672

export const POLICY_SIZE = 4672;

const PROMO_OFFSET: Record<string, number> = { n: 64, b: 65, r: 66, q: 67 };

function sqFromAlg(alg: string): number {
  const col = alg.charCodeAt(0) - 97;
  const row = 8 - parseInt(alg[1]);
  return row * 8 + col;
}

/**
 * Convert a LAN move string (e.g. "e2e4", "a7a8q") to a policy index (0..4671).
 * Returns -1 if the move format is unrecognised.
 */
export function moveToIndex(lan: string): number {
  if (lan.length < 4) return -1;
  const from  = sqFromAlg(lan.slice(0, 2));
  const to    = sqFromAlg(lan.slice(2, 4));
  const promo = lan[4];
  if (promo && PROMO_OFFSET[promo] !== undefined) {
    return from * 73 + PROMO_OFFSET[promo];
  }
  return from * 73 + to;
}

/**
 * Convert policy index back to a LAN prefix for matching with legal moves.
 */
export function indexToMove(idx: number): { from: number; to: number; promo?: string } {
  const from = Math.floor(idx / 73);
  const rest = idx % 73;
  if (rest >= 64) {
    const promos = ['n', 'b', 'r', 'q'];
    return { from, to: -1, promo: promos[rest - 64] };
  }
  return { from, to: rest };
}
