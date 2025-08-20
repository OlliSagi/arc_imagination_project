from __future__ import annotations

import torch
import torch.nn as nn


class SlotHeads(nn.Module):
    def __init__(self, input_dim: int, event_classes: int, axis_classes: int, dy_classes: int, dx_classes: int, feature_classes: int):
        super().__init__()
        hid = max(128, input_dim)
        self.event_head = nn.Sequential(nn.Linear(input_dim, hid), nn.GELU(), nn.Linear(hid, event_classes))
        self.axis_head = nn.Sequential(nn.Linear(input_dim, hid), nn.GELU(), nn.Linear(hid, axis_classes))
        self.dy_head   = nn.Sequential(nn.Linear(input_dim, hid), nn.GELU(), nn.Linear(hid, dy_classes))
        self.dx_head   = nn.Sequential(nn.Linear(input_dim, hid), nn.GELU(), nn.Linear(hid, dx_classes))
        self.feature_head = nn.Sequential(nn.Linear(input_dim, hid), nn.GELU(), nn.Linear(hid, feature_classes))

    def forward(self, r: torch.Tensor) -> dict:
        return {
            'event_logits': self.event_head(r),
            'axis_logits': self.axis_head(r),
            'dy_logits': self.dy_head(r),
            'dx_logits': self.dx_head(r),
            'feature_logits': self.feature_head(r),
        }


