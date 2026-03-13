#!/usr/bin/env python3
"""
train/evaluate.py — ELO estimation and model comparison

Two uses:
  1. Head-to-head: new model vs old model (100 games) → promote if win_rate > 55%
  2. Vs Stockfish: measure absolute ELO using stockfish binary at fixed Elo setting

Usage:
  python -m train.evaluate --new checkpoints/gen_10.pt --old checkpoints/gen_9.pt
  python -m train.evaluate --new checkpoints/gen_10.pt --vs-stockfish --sf-elo 1500
"""

import argparse
import random
import time
import chess
import chess.engine
import torch
import numpy as np
from pathlib import Path

from .model import load_checkpoint, ChessResNet
from .mcts_python import MCTS
from .input import get_legal_move_indices, move_to_index


# ── Pure MCTS player ──────────────────────────────────────────────────────────

class MCTSPlayer:
    """Chess player backed by MCTS + neural network."""

    def __init__(self, model: ChessResNet, device: str, n_simulations: int = 200):
        self.mcts = MCTS(model, device=device, batch_size=8)
        self.n_simulations = n_simulations

    def get_move(self, board: chess.Board) -> chess.Move:
        policy, _ = self.mcts.search(
            board,
            n_simulations=self.n_simulations,
            temperature=0,     # greedy for evaluation
            add_noise=False,
        )
        return self.mcts.best_move(board, policy)


# ── Head-to-head match ────────────────────────────────────────────────────────

def play_match(
    player_white: MCTSPlayer,
    player_black:  MCTSPlayer,
    n_games:      int = 10,
    max_moves:    int = 300,
) -> dict:
    """
    Play n_games between two players, alternating colours.
    Returns {wins_white, wins_black, draws}.
    """
    wins_white = wins_black = draws = 0

    for game_i in range(n_games):
        board = chess.Board()
        moves = 0

        while not board.is_game_over() and moves < max_moves:
            if board.turn == chess.WHITE:
                move = player_white.get_move(board)
            else:
                move = player_black.get_move(board)
            board.push(move)
            moves += 1

        result = board.result()
        if result == '1-0':   wins_white += 1
        elif result == '0-1': wins_black += 1
        else:                 draws      += 1

        print(f"  Game {game_i+1}/{n_games}: {result} ({moves} moves)")

    return {'wins_white': wins_white, 'wins_black': wins_black, 'draws': draws}


def evaluate_models(
    new_model: ChessResNet,
    old_model: ChessResNet,
    device: str,
    n_games: int = 20,
    n_simulations: int = 100,
) -> float:
    """
    Compare two models, return win rate of new_model (0..1).
    Plays n_games//2 as white, n_games//2 as black.
    """
    new_player = MCTSPlayer(new_model, device, n_simulations)
    old_player = MCTSPlayer(old_model, device, n_simulations)

    half = n_games // 2
    print(f"  Round 1: New=White, Old=Black ({half} games)")
    r1 = play_match(new_player, old_player, n_games=half)
    print(f"  Round 2: New=Black, Old=White ({half} games)")
    r2 = play_match(old_player, new_player, n_games=half)

    new_wins  = r1['wins_white'] + r2['wins_black']
    old_wins  = r1['wins_black'] + r2['wins_white']
    all_draws = r1['draws'] + r2['draws']

    total     = n_games
    win_rate  = (new_wins + 0.5 * all_draws) / total

    print(f"\nNew: {new_wins}W / Old: {old_wins}W / {all_draws}D  →  win_rate={win_rate:.1%}")
    return win_rate


# ── vs Stockfish ──────────────────────────────────────────────────────────────

def evaluate_vs_stockfish(
    model: ChessResNet,
    device: str,
    sf_path: str,
    sf_elo: int = 1500,
    n_games: int = 10,
    n_simulations: int = 200,
    move_time_s: float = 0.1,
) -> float:
    """
    Play n_games against Stockfish limited to sf_elo, return our win rate.
    Requires stockfish binary installed at sf_path.
    """
    player = MCTSPlayer(model, device, n_simulations)

    try:
        engine = chess.engine.SimpleEngine.popen_uci(sf_path)
        engine.configure({'UCI_LimitStrength': True, 'UCI_Elo': sf_elo})
    except Exception as e:
        print(f"[Warning] Could not start Stockfish: {e}")
        return 0.0

    wins = draws = losses = 0

    for game_i in range(n_games):
        board    = chess.Board()
        we_white = (game_i % 2 == 0)
        moves    = 0

        while not board.is_game_over() and moves < 300:
            if board.turn == (chess.WHITE if we_white else chess.BLACK):
                move = player.get_move(board)
            else:
                result = engine.play(board, chess.engine.Limit(time=move_time_s))
                move   = result.move
            board.push(move)
            moves += 1

        res = board.result()
        our_col = chess.WHITE if we_white else chess.BLACK
        if (res == '1-0' and our_col == chess.WHITE) or \
           (res == '0-1' and our_col == chess.BLACK):
            wins += 1
        elif res == '1/2-1/2':
            draws += 1
        else:
            losses += 1

        print(f"  Game {game_i+1}: {'Win' if wins > game_i else ('Draw' if draws > game_i else 'Loss')}")

    engine.quit()
    win_rate = (wins + 0.5 * draws) / n_games
    print(f"\nvs SF{sf_elo}: {wins}W / {losses}L / {draws}D  →  win_rate={win_rate:.1%}")
    return win_rate


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--new',          required=True,  help='New model checkpoint')
    parser.add_argument('--old',          default=None,   help='Old model checkpoint (for head-to-head)')
    parser.add_argument('--vs-stockfish', action='store_true')
    parser.add_argument('--sf-path',      default='stockfish', help='Path to stockfish binary')
    parser.add_argument('--sf-elo',       type=int, default=1500)
    parser.add_argument('--games',        type=int, default=20)
    parser.add_argument('--sims',         type=int, default=100)
    parser.add_argument('--device',       default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = args.device
    new_model, gen, elo = load_checkpoint(args.new, device)
    print(f"Loaded new model: gen={gen}, elo={elo:.0f}")

    if args.vs_stockfish:
        evaluate_vs_stockfish(new_model, device, args.sf_path, args.sf_elo, args.games, args.sims)
    elif args.old:
        old_model, old_gen, old_elo = load_checkpoint(args.old, device)
        print(f"Loaded old model: gen={old_gen}, elo={old_elo:.0f}")
        evaluate_models(new_model, old_model, device, args.games, args.sims)
    else:
        print("Specify --old <checkpoint> or --vs-stockfish")
