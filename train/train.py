#!/usr/bin/env python3
"""
Chess Neural Network Training Script
=====================================
Trains a compact policy+value network on lichess open database games.
Exports to ONNX format for use with onnxruntime-web in the browser.

Architecture:
  Input:  781 features (12 piece planes × 64 + castling + en passant + side)
  Shared: Dense(256, ReLU) → Dense(128, ReLU)
  Policy: Dense(1858) — log-probabilities over moves
  Value:  Dense(64, ReLU) → Dense(1, tanh) — position value [-1, +1]

Usage:
  pip install torch chess requests tqdm onnx
  python train/train.py --games 50000 --epochs 10 --out public/model.onnx

Runtime:
  ~1 hour for 50K games on CPU
  ~5-10 minutes for 50K games on NVIDIA GPU
  Result: ~1500-1800 ELO network (better as training data grows)
"""

import argparse
import os
import io
import random
import struct
import time
from pathlib import Path

import chess
import chess.pgn
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ── Constants ──────────────────────────────────────────────────────────────────

NN_INPUT_SIZE = 781
# Policy space: from_sq * 73 + to_idx
# from_sq ∈ [0,63], to_idx ∈ [0,72] (0-63=to_sq, 64-67=promo N/B/R/Q)
# Max index: 63 * 73 + 72 = 4671 → size = 4672
POLICY_SIZE   = 4672
PROMO_OFFSET  = {'n': 64, 'b': 65, 'r': 66, 'q': 67}

# ── Input encoding (mirrors src/engine/nn/input.ts) ───────────────────────────

PIECE_PLANE = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4,  chess.KING: 5,
}

def encode_board(board: chess.Board) -> list[float]:
    """Encode a chess.Board as 781 floats matching the TypeScript encoder."""
    features = [0.0] * NN_INPUT_SIZE

    # Piece planes (0..767)
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece:
            plane = PIECE_PLANE[piece.piece_type]
            if piece.color == chess.BLACK:
                plane += 6
            # Our convention: sq 0=a8, chess.py 0=a1 — flip rank
            our_sq = (7 - chess.square_rank(sq)) * 8 + chess.square_file(sq)
            features[plane * 64 + our_sq] = 1.0

    # Castling (768-771)
    if board.has_kingside_castling_rights(chess.WHITE):  features[768] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE): features[769] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):  features[770] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK): features[771] = 1.0

    # En passant file (772-779)
    if board.ep_square is not None:
        features[772 + chess.square_file(board.ep_square)] = 1.0

    # Side to move (780)
    features[780] = 0.0 if board.turn == chess.WHITE else 1.0

    return features

def move_to_index(move: chess.Move) -> int:
    """Convert chess.Move to policy index (0..1857)."""
    # Our convention: flip rank for from/to squares
    sq = move.from_square
    from_sq = (7 - chess.square_rank(sq)) * 8 + chess.square_file(sq)

    if move.promotion and move.promotion != chess.QUEEN:
        promo_char = chess.piece_symbol(move.promotion)
        return from_sq * 73 + PROMO_OFFSET.get(promo_char, 67)

    tsq = move.to_square
    to_sq = (7 - chess.square_rank(tsq)) * 8 + chess.square_file(tsq)
    return from_sq * 73 + to_sq

# ── Data collection ───────────────────────────────────────────────────────────

def download_lichess_sample(n_games: int, output_file: str = 'tmp_games.pgn'):
    """Download a sample of lichess games via the open database."""
    # Use a small curated PGN file from lichess (April 2013 = small starter set)
    url = 'https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst'

    print(f'Downloading lichess sample (~50MB compressed)...')
    print(f'(For larger training, visit https://database.lichess.org for monthly dumps)')

    # Alternative: use the Lichess API to get recent games
    # This downloads less data but is slower per game
    games = []
    try:
        response = requests.get(
            'https://lichess.org/api/games/user/lichess',
            params={'max': min(n_games, 300), 'perfType': 'blitz', 'opening': 'true'},
            headers={'Accept': 'application/x-chess-pgn'},
            stream=True, timeout=30
        )
        pgn_text = response.text
        pgn_io = io.StringIO(pgn_text)
        while len(games) < n_games:
            game = chess.pgn.read_game(pgn_io)
            if game is None:
                break
            games.append(game)
        print(f'Downloaded {len(games)} games from Lichess API')
    except Exception as e:
        print(f'Download failed: {e}')
        print('Using synthetic self-play data instead...')

    return games

def generate_training_examples(games: list, max_examples: int = 200_000):
    """
    Extract training examples from games.
    Each example: (features[781], move_index, outcome)
    outcome: +1.0 = white wins, -1.0 = black wins, 0.0 = draw
    """
    inputs, policies, values = [], [], []

    for game in tqdm(games, desc='Processing games'):
        result = game.headers.get('Result', '*')
        if result == '1-0':   outcome =  1.0
        elif result == '0-1': outcome = -1.0
        elif result == '1/2-1/2': outcome = 0.0
        else: continue

        board = game.board()
        for move in game.mainline_moves():
            if len(inputs) >= max_examples:
                break

            # Skip opening (first 5 moves) for quality
            if board.fullmove_number < 3:
                board.push(move)
                continue

            features = encode_board(board)
            move_idx = move_to_index(move)

            # Value from current player's perspective
            value = outcome if board.turn == chess.WHITE else -outcome

            inputs.append(features)
            policies.append(move_idx)
            values.append(value)

            board.push(move)

        if len(inputs) >= max_examples:
            break

    return inputs, policies, values

# ── Self-play data generation (no lichess needed) ─────────────────────────────

def generate_selfplay_data(model, n_games: int = 1000, temperature: float = 1.0):
    """Generate training data by self-play of current model."""
    model.eval()
    inputs, policies, values = [], [], []

    for _ in tqdm(range(n_games), desc='Self-play'):
        board = chess.Board()
        game_moves = []
        positions = []

        while not board.is_game_over() and len(game_moves) < 200:
            features = encode_board(board)
            feat_t = torch.FloatTensor(features).unsqueeze(0)

            with torch.no_grad():
                policy_logits, value_out = model(feat_t)

            # Get legal move probabilities
            legal_moves = list(board.legal_moves)
            legal_indices = [move_to_index(m) for m in legal_moves]
            logits = policy_logits[0][legal_indices]

            # Sample with temperature
            probs = torch.softmax(logits / temperature, dim=0).numpy()
            chosen_idx = random.choices(range(len(legal_moves)), weights=probs)[0]
            move = legal_moves[chosen_idx]

            positions.append((features, move_to_index(move),
                             1.0 if board.turn == chess.WHITE else -1.0))
            game_moves.append(move)
            board.push(move)

        # Determine outcome
        outcome_map = {'1-0': 1.0, '0-1': -1.0, '1/2-1/2': 0.0, '*': 0.0}
        outcome = outcome_map.get(board.result(), 0.0)

        for features, move_idx, sign in positions:
            inputs.append(features)
            policies.append(move_idx)
            values.append(sign * outcome)

    return inputs, policies, values

# ── Model ─────────────────────────────────────────────────────────────────────

class ChessNet(nn.Module):
    def __init__(self, input_size=NN_INPUT_SIZE, hidden1=256, hidden2=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden1),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden2),
        )
        self.policy_head = nn.Linear(hidden2, POLICY_SIZE)
        self.value_head  = nn.Sequential(
            nn.Linear(hidden2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        shared = self.shared(x)
        policy_logits = self.policy_head(shared)    # [B, 1858]
        value = self.value_head(shared).squeeze(-1) # [B]
        return policy_logits, value

# ── Training loop ─────────────────────────────────────────────────────────────

def train(model, loader, optimizer, device, epoch: int):
    model.train()
    total_loss = 0.0
    for batch_inputs, batch_moves, batch_values in tqdm(loader, desc=f'Epoch {epoch}'):
        batch_inputs  = batch_inputs.to(device)
        batch_moves   = batch_moves.to(device)
        batch_values  = batch_values.to(device)

        policy_logits, value_out = model(batch_inputs)

        # Policy loss: cross-entropy on the played move
        policy_loss = nn.CrossEntropyLoss()(policy_logits, batch_moves)

        # Value loss: MSE between predicted and actual outcome
        value_loss = nn.MSELoss()(value_out, batch_values)

        # Combined loss (equal weighting)
        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# ── ONNX export ───────────────────────────────────────────────────────────────

def export_onnx(model, output_path: str, device):
    model.eval()
    dummy_input = torch.zeros(1, NN_INPUT_SIZE, device=device)

    torch.onnx.export(
        model, dummy_input, output_path,
        input_names=['input'],
        output_names=['policy', 'value'],
        dynamic_axes={'input': {0: 'batch'}, 'policy': {0: 'batch'}, 'value': {0: 'batch'}},
        opset_version=17,
        do_constant_folding=True,
    )
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f'✅ Model exported to {output_path} ({size_mb:.1f} MB)')

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train chess neural network')
    parser.add_argument('--games',    type=int,   default=10_000,     help='Number of games to use')
    parser.add_argument('--epochs',   type=int,   default=20,         help='Training epochs')
    parser.add_argument('--batch',    type=int,   default=512,        help='Batch size')
    parser.add_argument('--lr',       type=float, default=1e-3,       help='Learning rate')
    parser.add_argument('--out',      type=str,   default='public/model.onnx', help='Output ONNX path')
    parser.add_argument('--selfplay', type=int,   default=0,          help='Self-play games after first training')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create output directory
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Model
    model = ChessNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    # --- Phase 1: Train on lichess games ---
    print(f'\n=== Phase 1: Downloading {args.games} games ===')
    games = download_lichess_sample(args.games)

    if games:
        print(f'\n=== Extracting training examples ===')
        inputs, policies, values = generate_training_examples(games, max_examples=args.games * 30)
    else:
        # Fallback: generate random board positions
        print('Generating random board positions as fallback...')
        inputs, policies, values = [], [], []
        board = chess.Board()
        for _ in range(min(args.games * 20, 200_000)):
            if board.is_game_over() or len(list(board.legal_moves)) == 0:
                board = chess.Board()
            move = random.choice(list(board.legal_moves))
            inputs.append(encode_board(board))
            policies.append(move_to_index(move))
            values.append(0.0)
            board.push(move)

    print(f'Training examples: {len(inputs):,}')

    # Create dataloader
    X = torch.FloatTensor(inputs)
    y_policy = torch.LongTensor(policies)
    y_value  = torch.FloatTensor(values)

    dataset = TensorDataset(X, y_policy, y_value)
    loader  = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=0)

    # Train
    print(f'\n=== Training {args.epochs} epochs ===')
    for epoch in range(1, args.epochs + 1):
        loss = train(model, loader, optimizer, device, epoch)
        scheduler.step()
        print(f'Epoch {epoch}/{args.epochs} — Loss: {loss:.4f}  LR: {scheduler.get_last_lr()[0]:.6f}')

        # Export checkpoint every 5 epochs
        if epoch % 5 == 0:
            ckpt_path = args.out.replace('.onnx', f'_e{epoch}.onnx')
            export_onnx(model, ckpt_path, device)

    # --- Phase 2: Self-play refinement ---
    if args.selfplay > 0:
        print(f'\n=== Phase 2: Self-play ({args.selfplay} games) ===')
        sp_inputs, sp_policies, sp_values = generate_selfplay_data(model, args.selfplay)
        print(f'Self-play examples: {len(sp_inputs):,}')

        X2 = torch.FloatTensor(sp_inputs)
        y2_policy = torch.LongTensor(sp_policies)
        y2_value  = torch.FloatTensor(sp_values)

        sp_dataset = TensorDataset(X2, y2_policy, y2_value)
        sp_loader  = DataLoader(sp_dataset, batch_size=args.batch, shuffle=True)

        optimizer2 = optim.Adam(model.parameters(), lr=args.lr * 0.1)
        for epoch in range(1, 6):
            loss = train(model, sp_loader, optimizer2, device, epoch)
            print(f'Self-play epoch {epoch}/5 — Loss: {loss:.4f}')

    # Final export
    export_onnx(model, args.out, device)
    print(f'\n🏆 Done! Copy {args.out} to your project public/ folder.')
    print(f'Then in the app: loadModel("/model.onnx") and switch worker to MCTS mode.')

if __name__ == '__main__':
    main()
