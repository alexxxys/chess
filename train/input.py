#!/usr/bin/env python3
"""
train/input.py — Position Encoder  
Converts chess positions to 8×8×19 spatial tensors for the ResNet.

Planes:
  [0..11]  - 12 piece planes (White P/N/B/R/Q/K, Black P/N/B/R/Q/K)
  [12..15] - 4 castling rights (WK, WQ, BK, BQ)
  [16]     - en passant file (plane of 1.0 on ep-file column)
  [17]     - side to move (all 1.0 = white, all 0.0 = black)
  [18]     - ones bias plane (always 1.0)

Move index: from_sq * 73 + to_idx
  to_idx: 0-63 = target square, 64-67 = promo (N/B/R/Q)
  Max index: 63 * 73 + 72 = 4671  →  POLICY_SIZE = 4672

Square convention (matching TypeScript engine):
  sq 0 = a8 (top-left), sq 63 = h1 (bottom-right)
  row = sq // 8, col = sq % 8
  chess.py uses sq 0 = a1 (bottom-left), so we flip rank:
    our_sq = (7 - chess.square_rank(chess_sq)) * 8 + chess.square_file(chess_sq)
"""

import chess
import numpy as np
import torch
from typing import Optional

# ── Constants ─────────────────────────────────────────────────────────────────

N_PLANES    = 19
POLICY_SIZE = 4672
BOARD_SIZE  = 8

# chess.py piece type → plane index (0-5 for white, 6-11 for black)
_PIECE_PLANE = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN:  4, chess.KING:   5,
}

_PROMO_OFFSET = {
    chess.KNIGHT: 64,
    chess.BISHOP: 65,
    chess.ROOK:   66,
    chess.QUEEN:  67,  # queen promotion = normal to target sq (no special offset)
}


# ── Square conversion ─────────────────────────────────────────────────────────

def chess_sq_to_our_sq(chess_sq: int) -> int:
    """Convert chess.py square (a1=0) to our convention (a8=0)."""
    return (7 - chess.square_rank(chess_sq)) * 8 + chess.square_file(chess_sq)


def our_sq_to_chess_sq(our_sq: int) -> int:
    """Convert our square (a8=0) to chess.py square (a1=0)."""
    row = our_sq // 8
    col = our_sq % 8
    return chess.square(col, 7 - row)


# ── Board encoding ────────────────────────────────────────────────────────────

def encode_board(board: chess.Board) -> np.ndarray:
    """
    Encode a chess.Board as float32[19, 8, 8] tensor.
    Always encodes from the perspective of the SIDE TO MOVE (board.turn).
    
    Returns numpy array [19, 8, 8].
    """
    planes = np.zeros((N_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    
    for chess_sq in range(64):
        piece = board.piece_at(chess_sq)
        if piece is None:
            continue
        
        plane_idx = _PIECE_PLANE[piece.piece_type]
        if piece.color == chess.BLACK:
            plane_idx += 6
        
        our_sq = chess_sq_to_our_sq(chess_sq)
        row, col = our_sq // 8, our_sq % 8
        planes[plane_idx, row, col] = 1.0
    
    # Castling rights (planes 12-15)
    if board.has_kingside_castling_rights(chess.WHITE):   planes[12] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):  planes[13] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):   planes[14] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):  planes[15] = 1.0
    
    # En passant file (plane 16)
    if board.ep_square is not None:
        ep_col = chess.square_file(board.ep_square)
        planes[16, :, ep_col] = 1.0
    
    # Side to move (plane 17): 1.0 = white to move
    if board.turn == chess.WHITE:
        planes[17] = 1.0
    
    # Bias plane (plane 18): always 1.0
    planes[18] = 1.0
    
    return planes


def encode_batch(boards: list[chess.Board]) -> torch.Tensor:
    """Encode multiple boards into a batched tensor [N, 19, 8, 8]."""
    batch = np.stack([encode_board(b) for b in boards], axis=0)
    return torch.from_numpy(batch)


# ── Move encoding ─────────────────────────────────────────────────────────────

def move_to_index(move: chess.Move) -> int:
    """
    Convert a chess.Move to a policy index in [0, 4671].
    Queen promotion uses the normal to-square (no promo offset).
    Under-promotions (N/B/R) use offset 64-66.
    """
    from_sq = chess_sq_to_our_sq(move.from_square)
    
    if move.promotion and move.promotion != chess.QUEEN:
        return from_sq * 73 + _PROMO_OFFSET[move.promotion]
    
    to_sq = chess_sq_to_our_sq(move.to_square)
    return from_sq * 73 + to_sq


def index_to_move_hint(idx: int) -> tuple[int, int, Optional[int]]:
    """
    Convert policy index to (from_our_sq, to_our_sq, promo_piece).
    promo_piece is a chess.* piece type constant or None.
    """
    from_sq = idx // 73
    rest    = idx % 73
    
    if rest == 64: return from_sq, -1, chess.KNIGHT
    if rest == 65: return from_sq, -1, chess.BISHOP
    if rest == 66: return from_sq, -1, chess.ROOK
    if rest == 67: return from_sq, -1, chess.QUEEN
    return from_sq, rest, None


def get_legal_move_indices(board: chess.Board) -> tuple[list[chess.Move], list[int]]:
    """Return (legal_moves, corresponding_policy_indices) for a board."""
    moves   = list(board.legal_moves)
    indices = [move_to_index(m) for m in moves]
    return moves, indices


# ── Game outcome ──────────────────────────────────────────────────────────────

def game_result_to_value(result: str, turn: chess.Color) -> float:
    """
    Convert PGN result string to value from the perspective of `turn`.
    '1-0' = white wins, '0-1' = black wins, '1/2-1/2' = draw
    """
    if result == '1-0':     outcome =  1.0  # white wins
    elif result == '0-1':   outcome = -1.0  # black wins
    else:                   outcome =  0.0  # draw
    
    return outcome if turn == chess.WHITE else -outcome


if __name__ == '__main__':
    # Sanity check
    board = chess.Board()
    planes = encode_board(board)
    assert planes.shape == (19, 8, 8), f"Shape mismatch: {planes.shape}"
    
    # Verify white pieces on starting position
    # White pawns on row 6 (rank 2), black pawns on row 1 (rank 7)
    assert planes[0, 6, :].sum() == 8, "Expected 8 white pawns on row 6"
    assert planes[6, 1, :].sum() == 8, "Expected 8 black pawns on row 1"
    
    # Move encoding
    e2e4 = chess.Move.from_uci('e2e4')
    idx  = move_to_index(e2e4)
    assert 0 <= idx < POLICY_SIZE, f"Index out of range: {idx}"
    
    # Batch encoding
    batch = encode_batch([board, board])
    assert batch.shape == (2, 19, 8, 8)
    
    print(f"✅ Input encoding OK — planes: {planes.shape}, dtype: {planes.dtype}")
    print(f"   e2e4 index: {idx}  (range: 0-{POLICY_SIZE-1})")
