// ── Precomputed Attack Tables ─────────────────────────────────────────────────
// All tables are computed once at module load time (no runtime cost per search).
// Square 0 = a8 (top-left), 63 = h1. Row = sq>>3, Col = sq&7.

// KNIGHT_ATTACKS[sq]  = array of squares a knight on sq can jump to
// KING_ATTACKS[sq]    = array of squares a king on sq can move to
// PAWN_ATTACKS[0][sq] = squares attacked by a WHITE pawn on sq (diagonals forward)
// PAWN_ATTACKS[1][sq] = squares attacked by a BLACK pawn on sq
// ROOK_RAYS[sq]       = 4 rays (N,S,E,W): each is array of squares in that direction
// BISHOP_RAYS[sq]     = 4 rays (NW,NE,SW,SE)
// BETWEEN[a][b]       = squares strictly between a and b along a ray (for pin detection)

export const KNIGHT_ATTACKS: number[][] = [];
export const KING_ATTACKS:   number[][] = [];
export const PAWN_ATTACKS:   [number[][], number[][]] = [[], []];
export const ROOK_RAYS:      number[][][] = [];   // [sq][dir 0..3][0..7]=target sqs
export const BISHOP_RAYS:    number[][][] = [];

// Fast lookup: set of squares in each direction from each square
// Used in isAttacked for slider detection
export const BETWEEN: number[][][] = [];    // [a][b] = squares strictly between (empty arr if not on ray)
export const ROOK_BETWEEN: Set<number>[][] = [];  // for fast pin detection

const KNIGHT_DELTAS = [[-2,-1],[-2,1],[-1,-2],[-1,2],[1,-2],[1,2],[2,-1],[2,1]] as const;
const KING_DELTAS   = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]] as const;
const ROOK_DIRS     = [[-1,0],[1,0],[0,1],[0,-1]] as const;
const BISHOP_DIRS   = [[-1,-1],[-1,1],[1,-1],[1,1]] as const;

function inBounds(r: number, c: number) { return r >= 0 && r < 8 && c >= 0 && c < 8; }

// Initialize all tables
(function initTables() {
  for (let sq = 0; sq < 64; sq++) {
    const r = sq >> 3, c = sq & 7;

    // Knight
    KNIGHT_ATTACKS[sq] = [];
    for (const [dr, dc] of KNIGHT_DELTAS) {
      const nr = r + dr, nc = c + dc;
      if (inBounds(nr, nc)) KNIGHT_ATTACKS[sq].push(nr * 8 + nc);
    }

    // King
    KING_ATTACKS[sq] = [];
    for (const [dr, dc] of KING_DELTAS) {
      const nr = r + dr, nc = c + dc;
      if (inBounds(nr, nc)) KING_ATTACKS[sq].push(nr * 8 + nc);
    }

    // Pawn attacks (squares a pawn ON sq attacks, not squares that attack sq)
    PAWN_ATTACKS[0][sq] = []; // White pawn attacks (moves toward row 0)
    if (r > 0) {
      if (c > 0) PAWN_ATTACKS[0][sq].push((r-1)*8 + (c-1));
      if (c < 7) PAWN_ATTACKS[0][sq].push((r-1)*8 + (c+1));
    }
    PAWN_ATTACKS[1][sq] = []; // Black pawn attacks (moves toward row 7)
    if (r < 7) {
      if (c > 0) PAWN_ATTACKS[1][sq].push((r+1)*8 + (c-1));
      if (c < 7) PAWN_ATTACKS[1][sq].push((r+1)*8 + (c+1));
    }

    // Rook rays
    ROOK_RAYS[sq] = [];
    for (const [dr, dc] of ROOK_DIRS) {
      const ray: number[] = [];
      let nr = r + dr, nc = c + dc;
      while (inBounds(nr, nc)) { ray.push(nr * 8 + nc); nr += dr; nc += dc; }
      ROOK_RAYS[sq].push(ray);
    }

    // Bishop rays
    BISHOP_RAYS[sq] = [];
    for (const [dr, dc] of BISHOP_DIRS) {
      const ray: number[] = [];
      let nr = r + dr, nc = c + dc;
      while (inBounds(nr, nc)) { ray.push(nr * 8 + nc); nr += dr; nc += dc; }
      BISHOP_RAYS[sq].push(ray);
    }

    // BETWEEN table
    BETWEEN[sq] = [];
    for (let tgt = 0; tgt < 64; tgt++) BETWEEN[sq][tgt] = [];

    for (const dirs of [ROOK_DIRS, BISHOP_DIRS]) {
      for (const [dr, dc] of dirs) {
        const sqs: number[] = [];
        let nr = r + dr, nc = c + dc;
        while (inBounds(nr, nc)) {
          const tgt = nr * 8 + nc;
          // All squares so far are "between sq and tgt"
          // Actually, between sq and tgt = all squares in sqs (exclusive of sq, exclusive of tgt)
          BETWEEN[sq][tgt] = [...sqs]; // squares between sq and tgt on this ray
          sqs.push(tgt);
          nr += dr; nc += dc;
        }
      }
    }
  }
})();
