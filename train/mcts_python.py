#!/usr/bin/env python3
"""
train/mcts_python.py — Python MCTS with GPU-accelerated neural network
AlphaZero-style PUCT search with Dirichlet noise at root.

Algorithm per move:
  1. Expand root with NN policy/value
  2. Add Dirichlet noise to root priors
  3. Run N simulations:  select → expand → evaluate → backpropagate
  4. Return visit distribution as training target policy
"""

import math
import random
import numpy as np
import torch
import chess
from dataclasses import dataclass, field

from .input import encode_batch, move_to_index, get_legal_move_indices


# ── Hyperparameters ───────────────────────────────────────────────────────────

C_PUCT          = 1.5
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPS   = 0.25


# ── MCTS Node ─────────────────────────────────────────────────────────────────

class Node:
    __slots__ = ('prior', 'visits', 'total_value', 'children', 'is_terminal', 'terminal_value')

    def __init__(self, prior: float = 0.0):
        self.prior         = prior
        self.visits        = 0
        self.total_value   = 0.0
        self.children: dict[int, 'Node'] = {}
        self.is_terminal   = False
        self.terminal_value = 0.0

    @property
    def q(self) -> float:
        return self.total_value / self.visits if self.visits > 0 else 0.0

    def puct(self, parent_visits: int) -> float:
        u = C_PUCT * self.prior * math.sqrt(parent_visits) / (1 + self.visits)
        return self.q + u

    @property
    def is_expanded(self) -> bool:
        return len(self.children) > 0 or self.is_terminal


# ── MCTS ─────────────────────────────────────────────────────────────────────

class MCTS:
    """PUCT MCTS backed by a neural network (policy + value heads)."""

    def __init__(self, model: torch.nn.Module, device: str = 'cuda'):
        self.model  = model
        self.device = torch.device(device)
        self.model.eval()

    # ── Public API ────────────────────────────────────────────────────────────

    def search(
        self,
        board: chess.Board,
        n_simulations: int = 200,
        temperature: float = 1.0,
        add_noise: bool = True,
    ) -> tuple[dict[int, float], 'Node']:
        """
        Run MCTS and return (policy_dict, root_node).
        policy_dict: {move_idx: probability}
        """
        root = Node()

        # Expand root
        root_policy, _ = self._nn_eval(board)
        self._expand_node(root, board, root_policy)

        if root.is_terminal or not root.children:
            return {}, root

        # Dirichlet noise at root
        if add_noise:
            self._add_noise(root)

        # Simulations
        for _ in range(n_simulations):
            sim_board = board.copy(stack=False)
            path      = self._select(root, sim_board)
            leaf      = path[-1]

            if leaf.is_terminal:
                value = leaf.terminal_value
            else:
                # Expand leaf if not yet expanded
                if not leaf.is_expanded:
                    policy, value = self._nn_eval(sim_board)
                    self._expand_node(leaf, sim_board, policy)
                    if leaf.is_terminal:
                        value = leaf.terminal_value
                else:
                    # Already expanded (shouldn't re-select unexpanded leaf normally)
                    value = 0.0

            self._backprop(path, value)

        # Build policy from visit counts
        return self._visit_policy(root, temperature), root

    def best_move(self, board: chess.Board, policy: dict[int, float]) -> chess.Move:
        legal_moves, legal_indices = get_legal_move_indices(board)
        best_prob, best_move = -1.0, legal_moves[0]
        for move, idx in zip(legal_moves, legal_indices):
            p = policy.get(idx, 0.0)
            if p > best_prob:
                best_prob, best_move = p, move
        return best_move

    def sample_move(self, board: chess.Board, policy: dict[int, float]) -> chess.Move:
        legal_moves, legal_indices = get_legal_move_indices(board)
        moves, probs = [], []
        for move, idx in zip(legal_moves, legal_indices):
            moves.append(move)
            probs.append(policy.get(idx, 0.0))
        total = sum(probs)
        if total < 1e-9:
            return random.choice(moves)
        return random.choices(moves, weights=[p / total for p in probs])[0]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _select(self, root: Node, board: chess.Board) -> list[Node]:
        """Traverse tree to a leaf, making moves on board along the way."""
        path = [root]
        node = root

        while node.is_expanded and not node.is_terminal:
            # PUCT selection
            best_idx  = max(node.children, key=lambda i: node.children[i].puct(node.visits))
            child     = node.children[best_idx]
            path.append(child)

            # Reconstruct move from index and push it
            for m in board.legal_moves:
                if move_to_index(m) == best_idx:
                    board.push(m)
                    break

            node = child

        return path

    def _expand_node(self, node: Node, board: chess.Board, policy_logits: np.ndarray):
        """Set node as terminal or create children with policy priors."""
        if board.is_game_over():
            node.is_terminal = True
            res = board.result()
            raw = 1.0 if res == '1-0' else (-1.0 if res == '0-1' else 0.0)
            # value from current side's perspective
            node.terminal_value = raw if board.turn == chess.WHITE else -raw
            return

        _, legal_indices = get_legal_move_indices(board)
        if not legal_indices:
            node.is_terminal    = True
            node.terminal_value = 0.0
            return

        # Masked softmax
        logits = policy_logits[legal_indices]
        logits = logits - logits.max()
        exps   = np.exp(logits)
        probs  = exps / (exps.sum() + 1e-9)

        for idx, prob in zip(legal_indices, probs):
            node.children[idx] = Node(prior=float(prob))

    def _backprop(self, path: list[Node], value: float):
        """
        Backpropagate value up the path.
        Value alternates sign at each level (AlphaZero style).
        """
        for i, node in enumerate(reversed(path)):
            v = value if i % 2 == 0 else -value
            node.visits      += 1
            node.total_value += v

    def _add_noise(self, root: Node):
        n     = len(root.children)
        noise = np.random.dirichlet([DIRICHLET_ALPHA] * n)
        for child, n_i in zip(root.children.values(), noise):
            child.prior = (1 - DIRICHLET_EPS) * child.prior + DIRICHLET_EPS * float(n_i)

    @torch.no_grad()
    def _nn_eval(self, board: chess.Board) -> tuple[np.ndarray, float]:
        x              = encode_batch([board]).to(self.device)
        policy_logits, value = self.model(x)
        return policy_logits[0].cpu().numpy(), float(value[0].item())

    def _visit_policy(self, root: Node, temperature: float) -> dict[int, float]:
        if not root.children:
            return {}

        if temperature == 0:
            best = max(root.children, key=lambda i: root.children[i].visits)
            return {best: 1.0}

        raw   = {i: c.visits ** (1.0 / temperature) for i, c in root.children.items()}
        total = sum(raw.values()) or 1.0
        return {i: v / total for i, v in raw.items()}
