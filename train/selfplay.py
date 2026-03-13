#!/usr/bin/env python3
"""
train/selfplay.py — Parallel Self-Play Game Generator

Generates training data by having the neural network play against itself.
Uses multiprocessing to run multiple games simultaneously and GPU batch
inference to evaluate positions efficiently.

Key AlphaZero insight: training targets come from MCTS visit distributions,
not just the move played. This means each position generates a full soft
policy target, providing much richer learning signal.
"""

import os
import random
import time
import multiprocessing as mp
from dataclasses import dataclass
from typing import Optional

import chess
import numpy as np
import torch

from .input import encode_board, move_to_index, get_legal_move_indices, N_PLANES
from .mcts_python import MCTS, C_PUCT


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class GameExample:
    """One training example extracted from a self-play game."""
    planes:         np.ndarray  # [19, 8, 8] float32
    policy_target:  np.ndarray  # [4672] float32 — MCTS visit distribution
    value_target:   float        # +1 / -1 / 0 from perspective of player to move


@dataclass
class GameResult:
    examples:   list[GameExample]
    n_moves:    int
    white_wins: bool
    draw:       bool
    duration_s: float


# ── Single game ───────────────────────────────────────────────────────────────

def play_game(
    model:         torch.nn.Module,
    device:        str = 'cuda',
    n_simulations: int = 200,
    max_moves:     int = 512,
    temp_threshold: int = 30,   # use T=1 for first N moves, then T→0
) -> GameResult:
    """Play one complete self-play game, return training examples."""
    mcts    = MCTS(model, device=device)
    board   = chess.Board()
    history: list[tuple[np.ndarray, dict[int, float], chess.Color]] = []
    t0      = time.time()

    move_num = 0
    while not board.is_game_over() and move_num < max_moves:
        # Temperature schedule: explore early, exploit later
        temp = 1.0 if move_num < temp_threshold else 0.0

        # MCTS search
        policy_dict, _ = mcts.search(
            board,
            n_simulations=n_simulations,
            temperature=temp,
            add_noise=(move_num < temp_threshold),
        )

        # Record position
        planes = encode_board(board)
        history.append((planes, policy_dict, chess.Color(board.turn)))

        # Sample / pick move
        if temp > 0:
            move = mcts.sample_move(board, policy_dict)
        else:
            move = mcts.best_move(board, policy_dict)

        board.push(move)
        move_num += 1

    # Game outcome
    result    = board.result()
    white_win = result == '1-0'
    draw      = result == '1/2-1/2'

    if result == '1-0':     outcome =  1.0
    elif result == '0-1':   outcome = -1.0
    else:                   outcome =  0.0

    # Build training examples
    POLICY_SIZE = 4672
    examples: list[GameExample] = []
    for planes, policy_dict, turn in history:
        # Policy target: sparse distribution over [0, 4672)
        policy_target = np.zeros(POLICY_SIZE, dtype=np.float32)
        for move_idx, prob in policy_dict.items():
            if 0 <= move_idx < POLICY_SIZE:
                policy_target[move_idx] = prob

        # Value target: from perspective of the player who moved
        value_target = outcome if turn == chess.WHITE else -outcome

        examples.append(GameExample(
            planes=planes,
            policy_target=policy_target,
            value_target=float(value_target),
        ))

    return GameResult(
        examples=examples,
        n_moves=move_num,
        white_wins=white_win,
        draw=draw,
        duration_s=time.time() - t0,
    )


# ── Parallel self-play ────────────────────────────────────────────────────────

def generate_games(
    model:         torch.nn.Module,
    n_games:       int = 100,
    n_simulations: int = 200,
    device:        str = 'cuda',
    n_workers:     int = 1,       # Note: GPU models can't share across processes easily;
                                   # use n_workers=1 and rely on speed of MCTS batch
    temp_threshold: int = 30,
    progress_cb    = None,
) -> list[GameExample]:
    """
    Generate n_games self-play games and return all training examples.
    
    For GPU training, n_workers=1 is recommended (single process uses GPU).
    For CPU-only, multiple workers may help (each uses CPU).
    """
    all_examples: list[GameExample] = []
    total_moves  = 0
    white_wins   = 0
    draws        = 0

    t0 = time.time()
    for game_i in range(n_games):
        result = play_game(
            model=model,
            device=device,
            n_simulations=n_simulations,
            temp_threshold=temp_threshold,
        )
        all_examples.extend(result.examples)
        total_moves += result.n_moves
        if result.white_wins: white_wins += 1
        if result.draw:       draws      += 1

        if progress_cb:
            progress_cb(game_i + 1, n_games, result)
        else:
            elapsed = time.time() - t0
            gps     = (game_i + 1) / elapsed
            remaining = (n_games - game_i - 1) / max(gps, 0.001)
            print(
                f"\r  Game {game_i+1}/{n_games} | "
                f"{result.n_moves} moves | "
                f"{result.duration_s:.1f}s | "
                f"{gps:.1f} g/s | "
                f"ETA: {remaining:.0f}s | "
                f"W:{white_wins} D:{draws}",
                end='', flush=True
            )

    print()  # newline after progress
    elapsed = time.time() - t0
    avg_moves = total_moves / max(n_games, 1)
    print(f"  Generated {len(all_examples)} examples from {n_games} games "
          f"({avg_moves:.0f} avg moves, {elapsed:.1f}s total)")

    return all_examples


# ── Replay buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Circular buffer of training examples across multiple generations."""

    def __init__(self, max_size: int = 500_000):
        self.max_size = max_size
        self.buffer:  list[GameExample] = []
        self._ptr     = 0

    def add(self, examples: list[GameExample]):
        for ex in examples:
            if len(self.buffer) < self.max_size:
                self.buffer.append(ex)
            else:
                self.buffer[self._ptr] = ex
            self._ptr = (self._ptr + 1) % self.max_size

    def sample(self, n: int) -> list[GameExample]:
        n = min(n, len(self.buffer))
        return random.sample(self.buffer, n)

    def __len__(self) -> int:
        return len(self.buffer)

    def to_tensors(self, examples: list[GameExample]):
        """Convert list of GameExample to batched tensors."""
        planes  = np.stack([ex.planes         for ex in examples])
        policy  = np.stack([ex.policy_target  for ex in examples])
        values  = np.array([ex.value_target   for ex in examples], dtype=np.float32)

        return (
            torch.from_numpy(planes),   # [N, 19, 8, 8]
            torch.from_numpy(policy),   # [N, 4672]
            torch.from_numpy(values),   # [N]
        )
