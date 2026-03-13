"""
train/ — AlphaZero Chess Training Package

Modules:
  model       — ResNet-10 neural network (policy + value heads)
  input       — Position encoder: chess.Board → [19, 8, 8] tensor
  mcts_python — Python MCTS with GPU batch inference
  selfplay    — Self-play game generation + ReplayBuffer
  evaluate    — Model comparison and ELO measurement
  pipeline    — Full training loop orchestrator

Quick start:
  python -m train.pipeline --gens 200 --games 100 --sims 200 --device cuda

For CUDA check:
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
"""

from .model       import ChessResNet, make_model, save_checkpoint, load_checkpoint
from .input       import encode_board, encode_batch, move_to_index, N_PLANES, POLICY_SIZE
from .mcts_python import MCTS
from .selfplay    import generate_games, ReplayBuffer, GameExample
from .evaluate    import evaluate_models, evaluate_vs_stockfish

__all__ = [
    'ChessResNet', 'make_model', 'save_checkpoint', 'load_checkpoint',
    'encode_board', 'encode_batch', 'move_to_index', 'N_PLANES', 'POLICY_SIZE',
    'MCTS',
    'generate_games', 'ReplayBuffer', 'GameExample',
    'evaluate_models', 'evaluate_vs_stockfish',
]
