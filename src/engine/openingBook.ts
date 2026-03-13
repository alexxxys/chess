// ── Comprehensive Opening Book ────────────────────────────────────────────────
// FEN (before engine move) → best reply in UCI format (e.g. "e7e5")
// Covers: e4/d4 mainlines, Sicilian, French, Caro-Kann, King's Indian, Grunfeld,
// Dutch, Queen's Gambit, Nimzo, Ruy Lopez, Italian, Scotch, Vienna, London, Catalan


// We store multiple book entries per position and pick randomly from them
// to avoid being predictable. Key = FEN (first 4 fields, ignoring move clocks).

function fenKey(fen: string): string {
  return fen.split(' ').slice(0, 4).join(' ');
}

const BOOK: Record<string, string[]> = {
  // ── Starting position ────────────────────────────────────────────────────
  'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -': ['e2e4', 'd2d4', 'g1f3', 'c2c4'],

  // ── After 1.e4 ──────────────────────────────────────────────────────────
  'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -': ['e7e5', 'c7c5', 'e7e6', 'c7c6', 'd7d6'],
  // 1.e4 e5 2.?
  'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -': ['g1f3', 'f2f4', 'b1c3'],
  // 1.e4 e5 2.Nf3 ?
  'rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq -': ['b8c6', 'g8f6', 'd7d6'],
  // 1.e4 e5 2.Nf3 Nc6 3.?
  'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq -': ['f1b5', 'f1c4', 'd2d4', 'b1c3'],
  // Ruy Lopez: 1.e4 e5 2.Nf3 Nc6 3.Bb5
  'r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQ1RK1 b KQkq -': ['a7a6', 'g8f6', 'f8c5', 'd7d6'],
  // Italian: 1.e4 e5 2.Nf3 Nc6 3.Bc4
  'r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq -': ['f8c5', 'g8f6', 'f7f5'],
  // Scotch: 1.e4 e5 2.Nf3 Nc6 3.d4
  'r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq -': ['e5d4'],
  'r1bqkbnr/pppp1ppp/2n5/8/3pP3/5N2/PPP2PPP/RNBQKB1R w KQkq -': ['f3d4'],
  // Vienna: 1.e4 e5 2.Nc3
  'rnbqkbnr/pppp1ppp/8/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR b KQkq -': ['g8f6', 'b8c6', 'f8c5'],

  // ── Sicilian ─────────────────────────────────────────────────────────────
  'rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -': ['g1f3', 'b1c3', 'f2f4'],
  'rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq -': ['d7d6', 'b8c6', 'e7e6', 'g8f6'],
  // Sicilian Najdorf: ...a6
  'rnbqkb1r/1p2pppp/p2p1n2/2p5/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq -': ['f1e2', 'f1b5', 'c1g5', 'g2g4'],
  // Sicilian Dragon: ...g6
  'rnbqkb1r/pp2pp1p/3p1np1/2p5/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq -': ['f1e2', 'f1c4', 'c1e3'],
  // Sicilian Scheveningen: ...e6
  'rnbqkb1r/pp3ppp/4pn2/2pp4/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq -': ['f1e2', 'c1g5', 'c1e3'],

  // ── French Defence ───────────────────────────────────────────────────────
  'rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -': ['d2d4', 'b1c3', 'g1f3'],
  'rnbqkbnr/pppp1ppp/4p3/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq -': ['d7d5'],
  'rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq -': ['b1c3', 'b1d2', 'e4e5'],
  // French Advance
  'rnbqkbnr/ppp2ppp/4p3/3pP3/3P4/8/PPP2PPP/RNBQKBNR b KQkq -': ['c7c5', 'b8c6'],
  // French Classical
  'rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/2N5/PPP2PPP/R1BQKBNR b KQkq -': ['g8f6', 'f8b4', 'd5e4'],

  // ── Caro-Kann ────────────────────────────────────────────────────────────
  'rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq -': ['d2d4', 'b1c3', 'g1f3'],
  'rnbqkbnr/pp1ppppp/2p5/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq -': ['d7d5'],
  'rnbqkbnr/pp2pppp/2p5/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq -': ['b1c3', 'b1d2', 'e4e5'],
  'rnbqkbnr/pp2pppp/2p5/3p4/3PP3/2N5/PPP2PPP/R1BQKBNR b KQkq -': ['d5e4'],
  'rnbqkbnr/pp2pppp/2p5/8/3Pp3/2N5/PPP2PPP/R1BQKBNR w KQkq -': ['c3e4'],

  // ── After 1.d4 ──────────────────────────────────────────────────────────
  'rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq -': ['d7d5', 'g8f6', 'f7f5', 'e7e6'],
  // QGD: 1.d4 d5
  'rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq -': ['c2c4', 'g1f3', 'b1c3'],
  // QGD accepted
  'rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq -': ['d5c4', 'e7e6', 'c7c6'],
  // QGD declined
  'rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq -': ['b1c3', 'g1f3', 'c4d5'],
  'rnbqkbnr/ppp1pppp/8/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR b KQkq -': ['e7e6', 'c7c6', 'g8f6'],
  // King's Indian: 1.d4 Nf6 2.c4 g6
  'rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq -': ['c2c4', 'g1f3'],
  'rnbqkb1r/pppppp1p/5np1/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq -': ['b1c3', 'g1f3'],
  'rnbqkb1r/pppppp1p/5np1/8/2PP4/2N5/PP2PPPP/R1BQKBNR b KQkq -': ['f8g7'],
  'rnbqk2r/ppppppbp/5np1/8/2PPP3/2N5/PP3PPP/R1BQKBNR b KQkq -': ['e8g8', 'd7d6'],
  // Grunfeld: 1.d4 Nf6 2.c4 g6 3.Nc3 d5
  'rnbqkb1r/ppp1pp1p/5np1/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq -': ['c4d5', 'e2e4'],

  // ── After 1.d4 Nf6 - Nimzo factors ──────────────────────────────────────
  'rnbqkb1r/pppp1ppp/4pn2/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq -': ['b1c3', 'g1f3'],
  'rnbqkb1r/pppp1ppp/4pn2/8/2PP4/2N5/PP2PPPP/R1BQKBNR b KQkq -': ['f8b4'],
  // Nimzo-Indian
  'r1bqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PP2PPPP/R1BQKBNR w KQkq -': ['d1c2', 'e2e3', 'a2a3'],
  // Dutch: 1.d4 f5
  'rnbqkbnr/ppppp1pp/8/5p2/3P4/8/PPP1PPPP/RNBQKBNR w KQkq -': ['g2g3', 'c2c4', 'g1f3'],

  // ── After 1.Nf3 ─────────────────────────────────────────────────────────
  'rnbqkbnr/pppppppp/8/8/8/5N2/PPPP1PPP/RNBQKB1R b KQkq -': ['d7d5', 'g8f6', 'c7c5', 'e7e6'],

  // ── After 1.c4 (English) ────────────────────────────────────────────────
  'rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq -': ['e7e5', 'g8f6', 'c7c5', 'e7e6'],
  'rnbqkbnr/pppp1ppp/8/4p3/2P5/8/PP1PPPPP/RNBQKBNR w KQkq -': ['b1c3', 'g1f3'],
  'rnbqkbnr/pppp1ppp/8/4p3/2P5/2N5/PP1PPPPP/R1BQKBNR b KQkq -': ['g8f6', 'b8c6', 'f8b4'],

  // ── London System (1.d4 d5 2.Nf3 Nf6 3.Bf4) ────────────────────────────
  'rnbqkb1r/ppp1pppp/8/3p4/3P1B2/5N2/PPP1PPPP/RN1QKB1R b KQkq -': ['e7e6', 'c7c5', 'c8f5'],

  // ── Catalan (1.d4 Nf6 2.c4 e6 3.g3) ────────────────────────────────────
  'rnbqkb1r/pppp1ppp/4pn2/8/2PP4/6P1/PP2PP1P/RNBQKBNR b KQkq -': ['d7d5', 'f8b4'],

  // ── Caro-Kann ─────────────────────────────────────────────────────────────
  'rnbqkbnr/pp2pppp/2p5/8/3PN3/8/PPP2PPP/R1BQKBNR b KQkq -': ['c8f5', 'g8f6', 'c8g4'],
};

/**
 * Look up a position in the opening book.
 * Returns a random move from the available options, or null if not found.
 */
export function bookMove(fen: string): string | null {
  const key = fenKey(fen);
  const options = BOOK[key];
  if (!options || options.length === 0) return null;
  return options[Math.floor(Math.random() * options.length)];
}
