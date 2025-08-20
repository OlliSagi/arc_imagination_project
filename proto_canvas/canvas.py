from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .edits import apply_edits_to_state


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dw(x)
        x = F.gelu(self.norm(x))
        x = self.pw(x)
        return x + residual


class Sim2D(nn.Module):
    def __init__(self, H: int, W: int, C: int, dim: int = 256, layers: int = 4):
        super().__init__()
        self.H, self.W, self.C = H, W, C
        self.inp = nn.Conv2d(C, dim, kernel_size=1)
        self.blocks = nn.ModuleList([ResidualBlock(dim) for _ in range(layers)])
        self.out = nn.Conv2d(dim, C, kernel_size=1)
        self.decode_head = nn.Conv2d(C, 10, kernel_size=1)

    @torch.no_grad()
    def init_state(self, x_onehot: torch.Tensor) -> torch.Tensor:
        """Initialize state S with first 10 channels set to x_onehot; others zero.
        x_onehot: (B,10,H,W)
        """
        B = x_onehot.size(0)
        device = x_onehot.device
        S = torch.zeros(B, self.C, self.H, self.W, device=device)
        S[:, :10] = x_onehot
        return S

    def dynamics(self, S: torch.Tensor) -> torch.Tensor:
        x = self.inp(S)
        for blk in self.blocks:
            x = blk(x)
        dS = self.out(x)
        return S + dS

    def step(self, S: torch.Tensor, edits: Optional[torch.Tensor], brush_radius=(0.08, 0.35)) -> torch.Tensor:
        if edits is not None:
            target = apply_edits_to_state(S, edits, brush_radius)
        else:
            target = S
        S_pred = self.dynamics(S)
        # Pull slightly toward target to encourage edit-following
        S_pred = S_pred + 0.5 * (target - S)
        return S_pred

    def decode_logits(self, S: torch.Tensor) -> torch.Tensor:
        return self.decode_head(S)


