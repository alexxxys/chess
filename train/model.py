#!/usr/bin/env python3
"""
train/model.py — ResNet-10 Chess Neural Network
AlphaZero-style architecture: convolutional residual network
with policy and value heads.

Input:  [B, 19, 8, 8] spatial tensor
Policy: [B, 4672] move logits
Value:  [B, 1]    tanh position value
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Input / Output sizes ──────────────────────────────────────────────────────

N_INPUT_PLANES = 19    # 12 pieces + 4 castling + 1 ep + 1 side + 1 ones
POLICY_SIZE    = 4672  # from_sq * 73 + to_idx
BOARD_SIZE     = 8


# ── Building Blocks ───────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Standard pre-activation residual block used in AlphaZero."""
    def __init__(self, channels: int):
        super().__init__()
        self.bn1  = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2  = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        return out + residual


# ── Main Network ──────────────────────────────────────────────────────────────

class ChessResNet(nn.Module):
    """
    AlphaZero-style residual network for chess.
    
    Architecture:
      Stem:   Conv(19→128, 3×3) → BN → ReLU
      Body:   N × ResidualBlock(128)
      Policy: Conv(128→2, 1×1) → BN → ReLU → Flatten → Linear(1024, 4672)
      Value:  Conv(128→1, 1×1) → BN → ReLU → Flatten → Linear(64, 256) → Linear(1, tanh)
    """
    
    def __init__(self, n_blocks: int = 10, channels: int = 128):
        super().__init__()
        self.n_blocks = n_blocks
        self.channels = channels
        
        # Stem
        self.stem_conv = nn.Conv2d(N_INPUT_PLANES, channels, 3, padding=1, bias=False)
        self.stem_bn   = nn.BatchNorm2d(channels)
        
        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(n_blocks)
        ])
        self.tower_bn = nn.BatchNorm2d(channels)
        
        # Policy head
        self.policy_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(2)
        self.policy_fc   = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, POLICY_SIZE)
        
        # Value head
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn   = nn.BatchNorm2d(1)
        self.value_fc1  = nn.Linear(BOARD_SIZE * BOARD_SIZE, 256)
        self.value_fc2  = nn.Linear(256, 1)
        
        # Weight initialisation (orthogonal for conv, zero bias)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=1.0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor):
        # x: [B, 19, 8, 8]
        
        # Stem
        out = F.relu(self.stem_bn(self.stem_conv(x)))
        
        # Residual tower
        for block in self.res_blocks:
            out = block(out)
        out = F.relu(self.tower_bn(out))
        
        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        policy = self.policy_fc(p)             # [B, 4672]
        
        # Value head
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))  # [B, 1]
        
        return policy, value.squeeze(-1)        # [B, 4672], [B]
    
    def predict(self, x: torch.Tensor):
        """Single-position inference — returns (policy_probs, value)."""
        self.eval()
        with torch.no_grad():
            policy_logits, value = self.forward(x)
            policy_probs = torch.softmax(policy_logits, dim=-1)
        return policy_probs, value
    
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── Factory ───────────────────────────────────────────────────────────────────

def make_model(size: str = 'medium') -> ChessResNet:
    """
    Presets:
      tiny   - 4 blocks × 64  channels  (~500K params)  fast iteration
      small  - 6 blocks × 96  channels  (~2M params)    2-3x stronger
      medium - 10 blocks × 128 channels (~8M params)    AlphaZero-lite ← default
      large  - 20 blocks × 256 channels (~40M params)   near AlphaZero
    """
    configs = {
        'tiny':   (4,  64),
        'small':  (6,  96),
        'medium': (10, 128),
        'large':  (20, 256),
    }
    blocks, channels = configs.get(size, configs['medium'])
    model = ChessResNet(n_blocks=blocks, channels=channels)
    print(f"[Model] {size}: {blocks} blocks × {channels} ch = {model.parameter_count():,} params")
    return model


# ── Checkpoint save/load ──────────────────────────────────────────────────────

def save_checkpoint(model: ChessResNet, path: str, generation: int, elo: float = 0.0):
    torch.save({
        'model_state': model.state_dict(),
        'n_blocks':    model.n_blocks,
        'channels':    model.channels,
        'generation':  generation,
        'elo':         elo,
    }, path)


def load_checkpoint(path: str, device: str = 'cpu') -> tuple[ChessResNet, int, float]:
    ckpt = torch.load(path, map_location=device)
    model = ChessResNet(n_blocks=ckpt['n_blocks'], channels=ckpt['channels'])
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    return model, ckpt.get('generation', 0), ckpt.get('elo', 0.0)


if __name__ == '__main__':
    # Quick sanity check
    model = make_model('medium')
    x = torch.randn(4, N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
    policy, value = model(x)
    assert policy.shape == (4, POLICY_SIZE), f"Policy shape: {policy.shape}"
    assert value.shape == (4,), f"Value shape: {value.shape}"
    print(f"✅ Forward pass OK — policy: {policy.shape}, value: {value.shape}")
