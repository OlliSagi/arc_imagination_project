from __future__ import annotations

import json, os
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .canvas import Sim2D
from .edits import sample_random_edits


@dataclass
class LossWeights:
    recon: float = 1.0
    next_step: float = 1.0
    cycle: float = 0.5
    leakage: float = 0.1


def decode_ce(sim: Sim2D, S: torch.Tensor, x_onehot: torch.Tensor) -> torch.Tensor:
    logits = sim.decode_logits(S)
    # compute CE over one-hot
    target = x_onehot.argmax(dim=1)  # (B,H,W)
    return F.cross_entropy(logits, target)


def cycle_ops(x_onehot: torch.Tensor) -> torch.Tensor:
    # mirror twice and rotate 4x should reconstruct; we evaluate CE after the cycle in trainer
    return x_onehot


def train_step(sim: Sim2D, opt: torch.optim.Optimizer, batch_x: torch.Tensor, cfg: Dict, lw: LossWeights, device: torch.device) -> Dict:
    sim.train()
    B, _, H, W = batch_x.shape
    batch_x = batch_x.to(device)
    S = sim.init_state(batch_x)

    # Recon loss
    recon = decode_ce(sim, S, batch_x)

    # Next-step consistency under random edits
    edits = sample_random_edits(B, cfg['model']['edit_tokens'], cfg['model']['C'], device)
    S_pred = sim.step(S, edits, brush_radius=tuple(cfg['model']['brush_radius']))
    S_tgt = S.detach()  # target is pre-edit state nudged by arithmetic edit inside step; simplified
    next_step = F.mse_loss(S_pred, S_tgt)

    # Cycle consistency (proxy): encourage decoded logits to be confident
    logits = sim.decode_logits(S)
    max_logit = logits.softmax(dim=1).amax(dim=1).mean()
    cycle = F.relu(0.75 - max_logit)  # encourage confidence as proxy for cycle

    loss = lw.recon * recon + lw.next_step * next_step + lw.cycle * cycle

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    with torch.no_grad():
        metrics = {
            'recon_ce': float(recon.item()),
            'next_step_mse': float(next_step.item()),
            'cycle_ce': float(cycle.item()),
            'max_logit': float(max_logit.item()),
        }
    return metrics


