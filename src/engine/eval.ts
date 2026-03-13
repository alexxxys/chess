// ── Stage 3 Evaluation ───────────────────────────────────────────────────────
// Tapered evaluation: scores blend between middlegame (MG) and endgame (EG)
// based on remaining material (game phase). This technique is used in
// Stockfish, Ethereal, and most modern engines above 2500 ELO.

// ── Material values ──────────────────────────────────────────────────────────
// [middlegame, endgame]

export const PIECE_VALUE: Record<string, number> = {
  p: 100, n: 320, b: 330, r: 500, q: 900, k: 20000,
};

const MG_VALUE: Record<string, number> = { p: 82,  n: 337, b: 365, r: 477, q: 1025, k: 0 };
const EG_VALUE: Record<string, number> = { p: 94,  n: 281, b: 297, r: 512, q: 936,  k: 0 };

// Phase weights for game-phase calculation (no pawns or kings counted)
const PHASE_WEIGHT: Record<string, number> = { n: 1, b: 1, r: 2, q: 4 };
const TOTAL_PHASE = 24; // 2*(4*1 + 4*1 + 4*2 + 2*4) per side; standard value

// ── Piece-Square Tables (Middlegame) ─────────────────────────────────────────
// Index 0 = a8 (top-left), 63 = h1 (bottom-right) — chess.js board order
// From white's perspective; black mirrors (63 - sq)

const MG_PAWN_PST = [
   0,   0,   0,   0,   0,   0,   0,   0,
  98, 134,  61,  95,  68, 126,  34, -11,
  -6,   7,  26,  31,  65,  56,  25, -20,
 -14,  13,   6,  21,  23,  12,  17, -23,
 -27,  -2,  -5,  12,  17,   6,  10, -25,
 -26,  -4,  -4, -10,   3,   3,  33, -12,
 -35,  -1, -20, -23, -15,  24,  38, -22,
   0,   0,   0,   0,   0,   0,   0,   0,
];

const EG_PAWN_PST = [
   0,   0,   0,   0,   0,   0,   0,   0,
 178, 173, 158, 134, 147, 132, 165, 187,
  94, 100,  85,  67,  56,  53,  82,  84,
  32,  24,  13,   5,  -2,   4,  17,  17,
  13,   9,  -3,  -7,  -7,  -8,   3,  -1,
   4,   7,  -6,   1,   0,  -5,  -1,  -8,
  13,   8,   8,  10,  13,   0,   2,  -7,
   0,   0,   0,   0,   0,   0,   0,   0,
];

const MG_KNIGHT_PST = [
 -167, -89, -34, -49,  61, -97, -15, -107,
  -73, -41,  72,  36,  23,  62,   7,  -17,
  -47,  60,  37,  65,  84, 129,  73,   44,
   -9,  17,  19,  53,  37,  69,  18,   22,
  -13,   4,  16,  13,  28,  19,  21,   -8,
  -23,  -9,  12,  10,  19,  17,  25,  -16,
  -29, -53, -12,  -3,  -1,  18, -14,  -19,
 -105, -21, -58, -33, -17, -28, -19,  -23,
];

const EG_KNIGHT_PST = [
 -58, -38, -13, -28, -31, -27, -63, -99,
 -25,  -8, -25,  -2,  -9, -25, -24, -52,
 -24, -20,  10,   9,  -1,  -9, -19, -41,
 -17,   3,  22,  22,  22,  11,   8, -18,
 -18,  -6,  16,  25,  16,  17,   4, -18,
 -23,  -3,  -1,  15,  10,  -3, -20, -22,
 -42, -20, -10,  -5,  -2, -20, -23, -44,
 -29, -51, -23, -15, -22, -18, -50, -64,
];

const MG_BISHOP_PST = [
 -29,   4, -82, -37, -25, -42,   7,  -8,
 -26,  16, -18, -13,  30,  59,  18, -47,
 -16,  37,  43,  40,  35,  50,  37,  -2,
  -4,   5,  19,  50,  37,  37,   7,  -2,
  -6,  13,  13,  26,  34,  12,  10,   4,
   0,  15,  15,  15,  14,  27,  18,  10,
   4,  15,  16,   0,   7,  21,  33,   1,
 -33,  -3, -14, -21, -13, -12, -39, -21,
];

const EG_BISHOP_PST = [
 -14, -21, -11,  -8, -7,  -9, -17, -24,
  -8,  -4,   7, -12, -3, -13,  -4, -14,
   2,  -8,   0,  -1, -2,   6,   0,   4,
  -3,   9,  12,   9,  14,  10,   3,   2,
  -6,   3,  13,  19,   7,  10,  -3,  -9,
 -12,  -3,   8,  10,  13,   3,  -7, -15,
 -14, -18,  -7,  -1,   4,  -9, -15, -27,
 -23,  -9, -23,  -5,  -9, -16,  -5, -17,
];

const MG_ROOK_PST = [
  32,  42,  32,  51, 63,  9,  31,  43,
  27,  32,  58,  62, 80, 67,  26,  44,
  -5,  19,  26,  36, 17, 45,  61,  16,
 -24, -11,   7,  26, 24, 35,  -8, -20,
 -36, -26, -12,  -1,  9, -7,   6, -23,
 -45, -25, -16, -17,  3,  0,  -5, -33,
 -44, -16, -20,  -9, -1, 11,  -6, -71,
 -19, -13,   1,  17, 16,  7, -37, -26,
];

const EG_ROOK_PST = [
  13, 10, 18, 15, 12,  12,   8,   5,
  11, 13, 13, 11, -3,   3,   8,   3,
   7,  7,  7,  5,  4,  -3,  -5,  -3,
   4,  3, 13,  1,  2,   1,  -1,   2,
   3,  5,  8,  4, -5,  -6,  -8, -11,
  -4,  0, -5, -1, -7, -12,  -8, -16,
  -6, -6,  0,  2, -9,  -9, -11,  -3,
  -9,  2,  3, -1, -5, -13,   4, -20,
];

const MG_QUEEN_PST = [
 -28,   0,  29,  12,  59,  44,  43,  45,
 -24, -39,  -5,   1, -16,  57,  28,  54,
 -13, -17,   7,   8,  29,  56,  47,  57,
 -27, -27, -16, -16,  -1,  17,  -2,   1,
  -9, -26,  -9, -10,  -2,  -4,   3,  -3,
 -14,   2, -11,  -2,  -5,   2,  14,   5,
 -35,  -8,  11,   2,   8,  15,  -3,   1,
  -1, -18,  -9,  10, -15, -25, -31, -50,
];

const EG_QUEEN_PST = [
  -9,  22,  22,  27,  27,  19,  10,  20,
 -17,  20,  32,  41,  58,  25,  30,   0,
 -20,   6,   9,  49,  47,  35,  19,   9,
   3,  22,  24,  45,  57,  40,  57,  36,
 -18,  28,  19,  47,  31,  34,  39,  23,
 -16, -27,  15,   6,   9,  17,  10,   5,
 -22, -23, -30, -16, -16, -23, -36, -32,
 -33, -28, -22, -43,  -5, -32, -20, -41,
];

const MG_KING_PST = [
 -65,  23,  16, -15, -56, -34,   2,  13,
  29,  -1, -20,  -7,  -8,  -4, -38, -29,
  -9,  24,   2, -16, -20,   6,  22, -22,
 -17, -20, -12, -27, -30, -25, -14, -36,
 -49,  -1, -27, -39, -46, -44, -33, -51,
 -14, -14, -22, -46, -44, -30, -15, -27,
   1,   7,  -8, -64, -43, -16,   9,   8,
 -15,  36,  12, -54,   8, -28,  24,  14,
];

const EG_KING_PST = [
 -74, -35, -18, -18, -11,  15,   4, -17,
 -12,  17,  14,  17,  17,  38,  23,  11,
  10,  17,  23,  15,  20,  45,  44,  13,
  -8,  22,  24,  27,  26,  33,  26,   3,
 -18,  -4,  21,  24,  27,  23,   9, -11,
 -19,  -3,  11,  21,  23,  16,   7,  -9,
 -27, -11,   4,  13,  14,   4,  -5, -17,
 -53, -34, -21, -11, -28, -14, -24, -43,
];

// ── PST export for search (Stage 2 uses this) ─────────────────────────────────
// For backwards compat, export MG tables as PST (used in search.ts evaluate())
export const PST: Record<string, number[]> = {
  p: MG_PAWN_PST, n: MG_KNIGHT_PST, b: MG_BISHOP_PST,
  r: MG_ROOK_PST, q: MG_QUEEN_PST,  k: MG_KING_PST,
};

export function pstIndex(sq: number, color: 'w' | 'b'): number {
  return color === 'w' ? sq : 63 - sq;
}
export function squareToIndex(sq: string): number {
  const file = sq.charCodeAt(0) - 97;
  const rank = parseInt(sq[1]) - 1;
  return (7 - rank) * 8 + file;
}

// ── Main evaluation function ─────────────────────────────────────────────────
// Called from search.ts — returns score from side-to-move perspective

import { Chess } from 'chess.js';

export function evaluateFull(chess: Chess): number {
  const board = chess.board();

  // ── Game phase calculation ──────────────────────────────────────────────
  let phase = TOTAL_PHASE;
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const p = board[r][c];
      if (p) phase -= (PHASE_WEIGHT[p.type] ?? 0);
    }
  }
  phase = Math.max(0, Math.min(TOTAL_PHASE, phase));
  // phase=0 → full middlegame, phase=TOTAL_PHASE → full endgame
  const mgPhase = TOTAL_PHASE - phase;
  const egPhase = phase;

  // ── Per-side accumulators ───────────────────────────────────────────────
  let wMg = 0, wEg = 0, bMg = 0, bEg = 0;

  // Pawn bitboards (by file) for structure eval
  const wPawnsByFile = new Array(8).fill(0);
  const bPawnsByFile = new Array(8).fill(0);
  const wPawnsByRank: boolean[][] = Array.from({ length: 8 }, () => new Array(8).fill(false));
  const bPawnsByRank: boolean[][] = Array.from({ length: 8 }, () => new Array(8).fill(false));

  // King square (board row/col)
  let wKingRow = 7, wKingCol = 4;
  let bKingRow = 0, bKingCol = 4;

  // Bishop counts
  let wBishops = 0, bBishops = 0;

  // Rook data
  const wRookCols: number[] = [];
  const bRookCols: number[] = [];

  // ── First pass: material + PST ──────────────────────────────────────────
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const p = board[r][c];
      if (!p) continue;

      const sq = r * 8 + c;
      const idx = p.color === 'w' ? sq : 63 - sq;
      const type = p.type;

      let mgMat: number, egMat: number, mgPos: number, egPos: number;
      switch (type) {
        case 'p':
          mgMat = MG_VALUE.p; egMat = EG_VALUE.p;
          mgPos = MG_PAWN_PST[idx]; egPos = EG_PAWN_PST[idx];
          if (p.color === 'w') { wPawnsByFile[c]++; wPawnsByRank[r][c] = true; }
          else                  { bPawnsByFile[c]++; bPawnsByRank[r][c] = true; }
          break;
        case 'n':
          mgMat = MG_VALUE.n; egMat = EG_VALUE.n;
          mgPos = MG_KNIGHT_PST[idx]; egPos = EG_KNIGHT_PST[idx];
          break;
        case 'b':
          mgMat = MG_VALUE.b; egMat = EG_VALUE.b;
          mgPos = MG_BISHOP_PST[idx]; egPos = EG_BISHOP_PST[idx];
          if (p.color === 'w') wBishops++; else bBishops++;
          break;
        case 'r':
          mgMat = MG_VALUE.r; egMat = EG_VALUE.r;
          mgPos = MG_ROOK_PST[idx]; egPos = EG_ROOK_PST[idx];
          if (p.color === 'w') wRookCols.push(c); else bRookCols.push(c);
          break;
        case 'q':
          mgMat = MG_VALUE.q; egMat = EG_VALUE.q;
          mgPos = MG_QUEEN_PST[idx]; egPos = EG_QUEEN_PST[idx];
          break;
        case 'k':
          mgMat = 0; egMat = 0;
          mgPos = MG_KING_PST[idx]; egPos = EG_KING_PST[idx];
          if (p.color === 'w') { wKingRow = r; wKingCol = c; }
          else                  { bKingRow = r; bKingCol = c; }
          break;
        default:
          continue;
      }

      if (p.color === 'w') {
        wMg += mgMat + mgPos;
        wEg += egMat + egPos;
      } else {
        bMg += mgMat + mgPos;
        bEg += egMat + egPos;
      }
    }
  }

  // ── Pawn structure ────────────────────────────────────────────────────
  // Doubled pawns: penalty per extra pawn on same file
  for (let f = 0; f < 8; f++) {
    if (wPawnsByFile[f] > 1) { wMg -= 8 * (wPawnsByFile[f] - 1); wEg -= 15 * (wPawnsByFile[f] - 1); }
    if (bPawnsByFile[f] > 1) { bMg -= 8 * (bPawnsByFile[f] - 1); bEg -= 15 * (bPawnsByFile[f] - 1); }
  }

  // Isolated pawns: no friendly pawn on adjacent file
  for (let f = 0; f < 8; f++) {
    if (wPawnsByFile[f] > 0) {
      const isolated = (f === 0 || wPawnsByFile[f-1] === 0) && (f === 7 || wPawnsByFile[f+1] === 0);
      if (isolated) { wMg -= 10; wEg -= 20; }
    }
    if (bPawnsByFile[f] > 0) {
      const isolated = (f === 0 || bPawnsByFile[f-1] === 0) && (f === 7 || bPawnsByFile[f+1] === 0);
      if (isolated) { bMg -= 10; bEg -= 20; }
    }
  }

  // Passed pawns: no opponent pawn blocks or threatens on this or adjacent files
  // Bonus increases as pawn advances toward promotion
  const PASSED_BONUS_MG = [0, 5,  10,  20,  35,  60, 100, 0]; // by rank (0=start,7=promo)
  const PASSED_BONUS_EG = [0, 10, 20,  40,  70, 110, 170, 0];
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      if (wPawnsByRank[r][f]) {
        // White pawn at (r,f). row 0=rank8, so rank = 8-r. Pawn moves toward row 0.
        // Check if any black pawn on files f-1,f,f+1 at rows < r (ahead for white)
        let passed = true;
        for (let rr = 0; rr < r; rr++) {
          for (let ff = Math.max(0, f-1); ff <= Math.min(7, f+1); ff++) {
            if (bPawnsByRank[rr][ff]) { passed = false; break; }
          }
          if (!passed) break;
        }
        if (passed) {
          const rank = 8 - r; // 1-8
          wMg += PASSED_BONUS_MG[rank - 1] ?? 0;
          wEg += PASSED_BONUS_EG[rank - 1] ?? 0;
        }
      }
      if (bPawnsByRank[r][f]) {
        // Black pawn at (r,f). Moves toward row 7.
        let passed = true;
        for (let rr = r + 1; rr < 8; rr++) {
          for (let ff = Math.max(0, f-1); ff <= Math.min(7, f+1); ff++) {
            if (wPawnsByRank[rr][ff]) { passed = false; break; }
          }
          if (!passed) break;
        }
        if (passed) {
          const rank = r + 1; // 1-8, high rank = far advanced for black
          bMg += PASSED_BONUS_MG[rank - 1] ?? 0;
          bEg += PASSED_BONUS_EG[rank - 1] ?? 0;
        }
      }
    }
  }

  // ── Bishop pair ──────────────────────────────────────────────────────────
  if (wBishops >= 2) { wMg += 22; wEg += 30; }
  if (bBishops >= 2) { bMg += 22; bEg += 30; }

  // ── Rook bonuses ─────────────────────────────────────────────────────────
  for (const f of wRookCols) {
    const openFile = wPawnsByFile[f] === 0;
    const semiOpen = openFile && bPawnsByFile[f] > 0;
    if (openFile && !semiOpen) { wMg += 20; wEg += 15; }      // open file
    else if (semiOpen)         { wMg += 10; wEg += 8; }        // semi-open
    // Rook on 7th rank (row 1 in board coords)
    // wRookCols stores file, we need row — skip for now (would need board scan)
  }
  for (const f of bRookCols) {
    const openFile = bPawnsByFile[f] === 0;
    const semiOpen = openFile && wPawnsByFile[f] > 0;
    if (openFile && !semiOpen) { bMg += 20; bEg += 15; }
    else if (semiOpen)         { bMg += 10; bEg += 8; }
  }

  // ── Rook on 7th rank (absolute rank 7 = row index 1 for white, row 6 for black) ──
  for (let c = 0; c < 8; c++) {
    const wR = board[1][c];
    if (wR && wR.color === 'w' && wR.type === 'r') { wMg += 15; wEg += 25; }
    const bR = board[6][c];
    if (bR && bR.color === 'b' && bR.type === 'r') { bMg += 15; bEg += 25; }
  }

  // ── King safety ──────────────────────────────────────────────────────────
  // Only meaningful in middlegame; weight tapers to 0 in endgame
  const kingSafetyWeight = mgPhase / TOTAL_PHASE;

  // Pawn shield: pawns in the 3 squares directly in front of king
  // White king: pawns are at rows wKingRow-1, -2 (lower row numbers = higher ranks)
  const wShield = countPawnsNearKing(board, wKingRow, wKingCol, 'w');
  const bShield = countPawnsNearKing(board, bKingRow, bKingCol, 'b');
  wMg += Math.round(wShield * 8 * kingSafetyWeight);
  bMg += Math.round(bShield * 8 * kingSafetyWeight);

  // Penalty for open file near king
  for (let df = -1; df <= 1; df++) {
    const wf = wKingCol + df;
    const bf = bKingCol + df;
    if (wf >= 0 && wf < 8 && wPawnsByFile[wf] === 0) { wMg -= Math.round(20 * kingSafetyWeight); }
    if (bf >= 0 && bf < 8 && bPawnsByFile[bf] === 0) { bMg -= Math.round(20 * kingSafetyWeight); }
  }

  // ── Mobility bonus ────────────────────────────────────────────────────────
  // Expensive to compute legal moves here — use a lightweight approximation:
  // Count pieces capable of moving (not fully blocked). We'll do a light version.
  // Full mobility via chess.moves() is too slow here (called at every node).
  // Instead, grant a flat tempo bonus to the side-to-move:
  const TEMPO = 15;

  // ── Taper final score ─────────────────────────────────────────────────────
  const mgScore = wMg - bMg;
  const egScore = wEg - bEg;
  const tapered = Math.round((mgScore * mgPhase + egScore * egPhase) / TOTAL_PHASE);

  // Return from side-to-move perspective + tempo
  const sideFactor = chess.turn() === 'w' ? 1 : -1;
  return tapered * sideFactor + TEMPO;
}

// ── Helper: count pawn shield squares ────────────────────────────────────────
function countPawnsNearKing(
  board: ReturnType<Chess['board']>,
  kingRow: number, kingCol: number,
  color: 'w' | 'b',
): number {
  let count = 0;
  // Direction pawns advance (white moves up = smaller row index)
  const dir = color === 'w' ? -1 : 1;
  for (let dr = 1; dr <= 2; dr++) {
    const r = kingRow + dir * dr;
    if (r < 0 || r > 7) break;
    for (let df = -1; df <= 1; df++) {
      const f = kingCol + df;
      if (f < 0 || f > 7) continue;
      const p = board[r][f];
      if (p && p.type === 'p' && p.color === color) count++;
    }
  }
  return count;
}
