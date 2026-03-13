// ── Zobrist Hashing ───────────────────────────────────────────────────────────
// Each unique chess position → unique 53-bit JS number (fits in Number precision).
// Fast incremental update: XOR out old piece, XOR in new piece.
// Keys are pseudo-random but fixed (seeded LCG for reproducibility).

// LCG: low-quality but fast, enough for hash keys
let _seed = 0xDEADBEEF;
function _rand(): number {
  _seed = Math.imul(1664525, _seed) + 1013904223;
  // Combine with a second LCG step to improve distribution
  const hi = Math.imul(1664525, _seed ^ 0xBEEFCAFE) + 1013904223;
  // Keep 26 bits from each, combine into 52-bit float
  return ((_seed >>> 6) * 0x4000000 + (hi >>> 6)) / 0xFFFFFFFFFFFFF;
}

// 12 piece types × 64 squares
const _PIECE_SQ: Float64Array = new Float64Array(12 * 64);
// Side to move (XOR in when it's black's turn)
let _SIDE_KEY: number;
// Castling rights (16 combinations, use 4 independent bits)
const _CASTLE_KEYS: Float64Array = new Float64Array(4);
// En passant file (8 keys for files a-h)
const _EP_KEYS: Float64Array = new Float64Array(8);

(function initZobrist() {
  for (let i = 0; i < 12 * 64; i++) _PIECE_SQ[i] = _rand();
  _SIDE_KEY = _rand();
  for (let i = 0; i < 4; i++) _CASTLE_KEYS[i] = _rand();
  for (let i = 0; i < 8; i++) _EP_KEYS[i] = _rand();
})();

// Get Zobrist key for a piece (1-12 → index 0-11) on a square (0-63)
export function pieceSquareKey(piece: number, sq: number): number {
  return _PIECE_SQ[(piece - 1) * 64 + sq];
}

export function sideKey(): number { return _SIDE_KEY; }

export function castleKey(crBit: number): number {
  // crBit is a power of 2 (1,2,4,8)
  return _CASTLE_KEYS[Math.log2(crBit)];
}

export function epKey(col: number): number { return _EP_KEYS[col]; }

/**
 * Compute full Zobrist hash from scratch for a position.
 * Only call this once per position (e.g. at startup). Then update incrementally.
 */
export function computeHash(
  board: Uint8Array,
  turn: number,        // 0=white, 1=black
  castling: number,    // CR_* bit flags
  epSq: number,        // -1 or square 0-63
): number {
  let h = 0;
  for (let sq = 0; sq < 64; sq++) {
    const p = board[sq];
    if (p !== 0) h ^= pieceSquareKey(p, sq);
  }
  if (turn === 1) h ^= _SIDE_KEY;
  for (let bit = 1; bit <= 8; bit <<= 1) {
    if (castling & bit) h ^= castleKey(bit);
  }
  if (epSq >= 0) h ^= epKey(epSq & 7);
  return h;
}

// XOR is its own inverse — same function for adding/removing
export { pieceSquareKey as xorPiece };
