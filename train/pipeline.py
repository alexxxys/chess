#!/usr/bin/env python3
"""
train/pipeline.py — AlphaZero Training Pipeline

Full training loop:
  1. Generate self-play games (MCTS-guided)
  2. Train network on examples from replay buffer
  3. Evaluate: new vs old (promote if win_rate > 55%)
  4. Export ONNX for browser
  5. Repeat

Usage:
  python -m train.pipeline --gens 200 --games 100 --sims 200 --device cuda
"""

import argparse
import copy
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .model import make_model, save_checkpoint, load_checkpoint, ChessResNet
from .selfplay import generate_games, ReplayBuffer, GameExample
from .evaluate import evaluate_models


# ── Training config ───────────────────────────────────────────────────────────

PROMOTE_THRESHOLD = 0.55   # new model win rate to replace old
EVAL_GAMES        = 20     # games per evaluation round
EVAL_SIMS         = 100    # simulations per move during eval (faster than training)
POLICY_SIZE       = 4672


# ── Training step ─────────────────────────────────────────────────────────────

def train_epoch(
    model:     ChessResNet,
    examples:  list[GameExample],
    optimizer: torch.optim.Optimizer,
    device:    str,
    batch_size: int = 512,
    n_epochs:   int = 5,
) -> dict:
    """Train on a batch of examples for n_epochs. Returns loss stats."""
    buffer = ReplayBuffer()
    buffer.add(examples)
    sample = buffer.sample(min(len(examples), 100_000))
    X_planes, y_policy, y_value = buffer.to_tensors(sample)

    dataset = TensorDataset(X_planes, y_policy, y_value)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=(device == 'cuda'))

    policy_losses = []
    value_losses  = []

    model.train()
    for epoch in range(n_epochs):
        ep_policy = ep_value = 0.0
        for batch_X, batch_p, batch_v in loader:
            batch_X = batch_X.to(device)
            batch_p = batch_p.to(device)
            batch_v = batch_v.to(device)

            pred_policy, pred_value = model(batch_X)

            # Policy loss: cross-entropy with soft targets (MCTS distributions)
            # Use KL divergence: -Σ target * log(softmax(logits))
            log_probs    = torch.log_softmax(pred_policy, dim=-1)
            policy_loss  = -(batch_p * log_probs).sum(dim=-1).mean()

            # Value loss: MSE between predicted and actual game outcome
            value_loss = nn.MSELoss()(pred_value, batch_v)

            # Total loss (equal weighting, as in AlphaZero paper)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_policy += policy_loss.item()
            ep_value  += value_loss.item()

        n = len(loader)
        policy_losses.append(ep_policy / n)
        value_losses.append(ep_value / n)
        print(f"      Epoch {epoch+1}/{n_epochs}: "
              f"policy={ep_policy/n:.3f}, value={ep_value/n:.3f}")

    return {
        'policy_loss': sum(policy_losses) / len(policy_losses),
        'value_loss':  sum(value_losses)  / len(value_losses),
    }


# ── ONNX export ───────────────────────────────────────────────────────────────

def export_onnx(model: ChessResNet, output_path: str, device: str):
    """Export model to ONNX for the browser ONNX Runtime."""
    model.eval()
    import torch.onnx
    dummy = torch.zeros(1, 19, 8, 8, device=device)

    torch.onnx.export(
        model.cpu(), dummy.cpu(), output_path,
        input_names=['input'],
        output_names=['policy', 'value'],
        opset_version=18,
        do_constant_folding=True,
    )
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  ✅ Exported ONNX: {output_path} ({size_mb:.1f} MB)")
    model.to(device)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(args):
    device = args.device
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path('public').mkdir(exist_ok=True)

    # Load or create model
    current_ckpt = Path(args.checkpoint_dir) / 'current.pt'
    if current_ckpt.exists() and not args.fresh:
        model, start_gen, elo = load_checkpoint(str(current_ckpt), device)
        print(f"Resumed from checkpoint: gen={start_gen}, elo={elo:.0f}")
    else:
        model      = make_model(args.size).to(device)
        start_gen  = 0
        elo        = 0.0
        print(f"Starting fresh: {args.size} model, {model.parameter_count():,} params")

    optimizer  = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler  = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    replay_buf = ReplayBuffer(max_size=args.replay_size)

    # ── Generation loop ───────────────────────────────────────────────────────
    for gen in range(start_gen + 1, start_gen + args.gens + 1):
        print(f"\n{'='*60}")
        print(f"Generation {gen} / {start_gen + args.gens}")
        print(f"{'='*60}")

        # 1. Self-play
        print(f"\n[1/3] Self-Play ({args.games} games, {args.sims} sims/move)")
        t0 = time.time()
        examples = generate_games(
            model=model,
            n_games=args.games,
            n_simulations=args.sims,
            device=device,
        )
        replay_buf.add(examples)
        print(f"  Buffer size: {len(replay_buf):,} examples  ({time.time()-t0:.0f}s)")

        # 2. Train on replay buffer
        print(f"\n[2/3] Training ({args.train_epochs} epochs, batch={args.batch})")
        t0 = time.time()
        sample = replay_buf.sample(min(len(replay_buf), args.train_batch))
        losses = train_epoch(model, sample, optimizer, device,
                              batch_size=args.batch, n_epochs=args.train_epochs)
        scheduler.step()
        print(f"  Loss: policy={losses['policy_loss']:.4f}, "
              f"value={losses['value_loss']:.4f}  ({time.time()-t0:.0f}s)")

        # 3. Save checkpoint
        ckpt_path = Path(args.checkpoint_dir) / f'gen_{gen:04d}.pt'
        save_checkpoint(model, str(ckpt_path), gen, elo)
        save_checkpoint(model, str(current_ckpt), gen, elo)
        print(f"  Saved: {ckpt_path.name}")

        # 4. Export ONNX every N gens (for the browser)
        if gen % args.export_every == 0:
            export_onnx(model, 'public/model.onnx', device)

        # 5. Evaluate every N gens
        if gen % args.eval_every == 0 and gen > 1:
            prev_ckpt = Path(args.checkpoint_dir) / f'gen_{gen - args.eval_every:04d}.pt'
            if prev_ckpt.exists():
                print(f"\n[3/3] Evaluation: gen {gen} vs gen {gen - args.eval_every}")
                old_model, _, _ = load_checkpoint(str(prev_ckpt), device)
                win_rate = evaluate_models(
                    model, old_model, device,
                    n_games=EVAL_GAMES, n_simulations=EVAL_SIMS
                )
                if win_rate >= PROMOTE_THRESHOLD:
                    print(f"  ✅ Promoted (win_rate={win_rate:.1%})")
                else:
                    print(f"  ⚠️  Not promoted (win_rate={win_rate:.1%} < {PROMOTE_THRESHOLD:.0%})")
                    # Still keep training — not rolling back (one-player AlphaZero)

        print(f"\nGen {gen} complete.")

    # Final export
    export_onnx(model, 'public/model.onnx', device)
    print("\n🏆 Training complete! Model saved to public/model.onnx")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='AlphaZero Chess Training Pipeline')
    parser.add_argument('--gens',          type=int,   default=200,      help='Generations to train')
    parser.add_argument('--games',         type=int,   default=100,      help='Self-play games per gen')
    parser.add_argument('--sims',          type=int,   default=200,      help='MCTS simulations per move')
    parser.add_argument('--train-epochs',  type=int,   default=5,        help='Training epochs per gen')
    parser.add_argument('--train-batch',   type=int,   default=50_000,   help='Examples sampled from replay buffer per gen')
    parser.add_argument('--batch',         type=int,   default=512,      help='Mini-batch size')
    parser.add_argument('--lr',            type=float, default=1e-3,     help='Learning rate')
    parser.add_argument('--replay-size',   type=int,   default=500_000,  help='Max replay buffer size')
    parser.add_argument('--eval-every',    type=int,   default=10,       help='Evaluate every N gens')
    parser.add_argument('--export-every',  type=int,   default=5,        help='Export ONNX every N gens')
    parser.add_argument('--checkpoint-dir',type=str,   default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--size',          type=str,   default='medium', choices=['tiny','small','medium','large'])
    parser.add_argument('--device',        type=str,   default=('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--fresh',         action='store_true', help='Start fresh (ignore existing checkpoint)')
    args = parser.parse_args()

    print(f"Device: {args.device}")
    if args.device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    run_pipeline(args)


if __name__ == '__main__':
    main()
