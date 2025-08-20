from __future__ import annotations

import math
from typing import Tuple

import torch


def gaussian_mask(cx: torch.Tensor, cy: torch.Tensor, radius: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Build a soft circular mask (B,1,H,W) given centers (normalized 0..1) and radius (fraction of min(H,W))."""
    device = cx.device
    yy = torch.linspace(0.0, 1.0, H, device=device).view(1, 1, H, 1)
    xx = torch.linspace(0.0, 1.0, W, device=device).view(1, 1, 1, W)
    cx = cx.view(-1, 1, 1, 1)
    cy = cy.view(-1, 1, 1, 1)
    r = radius.view(-1, 1, 1, 1)
    # Convert fractional radius to sigma; use smooth falloff
    sigma = (r.clamp(1e-4, 1.0) / 2.0)
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = torch.exp(-dist2 / (2.0 * sigma ** 2)).clamp(0.0, 1.0)
    return mask  # (B,1,H,W)


def apply_edits_to_state(S: torch.Tensor, edits: torch.Tensor, brush_radius: Tuple[float, float]) -> torch.Tensor:
    """Apply K edits to state S (B,C,H,W) arithmetically (target for next-step consistency).

    edits: (B,K,4+C) with fields [cx, cy, r, scale, delta[C]]. Centers/radius are 0..1 fractions.
    """
    B, C, H, W = S.shape
    K = edits.shape[1]
    out = S
    for k in range(K):
        token = edits[:, k, :]
        cx = token[:, 0]
        cy = token[:, 1]
        r_frac = token[:, 2].clamp(brush_radius[0], brush_radius[1])
        scale = token[:, 3].view(-1, 1, 1, 1)
        delta = token[:, 4:].view(B, C, 1, 1)
        mask = gaussian_mask(cx, cy, r_frac, H, W)  # (B,1,H,W)
        out = out + mask * scale * delta
    return out


def sample_random_edits(B: int, K: int, C: int, device: torch.device) -> torch.Tensor:
    """Sample K random edit tokens per batch element: (B,K,4+C)."""
    cx = torch.rand(B, K, 1, device=device)
    cy = torch.rand(B, K, 1, device=device)
    r = torch.rand(B, K, 1, device=device) * 0.3 + 0.05  # 0.05..0.35
    scale = (torch.rand(B, K, 1, device=device) - 0.5) * 1.0  # -0.5..0.5
    delta = (torch.randn(B, K, C, device=device) * 0.1)
    return torch.cat([cx, cy, r, scale, delta], dim=-1)


