from __future__ import annotations

import torch
import torch.nn as nn


class PredicateBank(nn.Module):
    def __init__(self, input_dim: int, num_predicates: int):
        super().__init__()
        hidden = max(128, input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.GELU(),
            nn.Linear(hidden, num_predicates)
        )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        # Returns raw logits; caller can sigmoid/softmax
        return self.net(r)


class StoryEncoder(nn.Module):
    def __init__(self, input_dim: int, z_dim: int):
        super().__init__()
        hidden = max(128, input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.GELU(),
            nn.Linear(hidden, z_dim)
        )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.net(r)


class StoryDecoder(nn.Module):
    def __init__(self, z_dim: int, event_classes: int, axis_classes: int, dy_classes: int, dx_classes: int, feature_classes: int):
        super().__init__()
        hidden = max(128, z_dim)
        self.shared = nn.Sequential(nn.Linear(z_dim, hidden), nn.GELU())
        self.event_head = nn.Linear(hidden, event_classes)
        self.axis_head = nn.Linear(hidden, axis_classes)
        self.dy_head = nn.Linear(hidden, dy_classes)
        self.dx_head = nn.Linear(hidden, dx_classes)
        self.feature_head = nn.Linear(hidden, feature_classes)

    def forward(self, z: torch.Tensor) -> dict:
        h = self.shared(z)
        return {
            'event_logits': self.event_head(h),
            'axis_logits': self.axis_head(h),
            'dy_logits': self.dy_head(h),
            'dx_logits': self.dx_head(h),
            'feature_logits': self.feature_head(h),
        }


class GeneralInterpreter(nn.Module):
    def __init__(self, input_dim: int, z_dim: int, num_predicates: int,
                 event_classes: int, axis_classes: int, dy_classes: int, dx_classes: int, feature_classes: int):
        super().__init__()
        self.predicates = PredicateBank(input_dim, num_predicates)
        self.encoder = StoryEncoder(input_dim, z_dim)
        self.decoder = StoryDecoder(z_dim, event_classes, axis_classes, dy_classes, dx_classes, feature_classes)

    def forward(self, r: torch.Tensor) -> dict:
        pred_logits = self.predicates(r)
        pred_gates = torch.sigmoid(pred_logits)
        z = self.encoder(r)
        out = self.decoder(z)
        out['predicate_gates'] = pred_gates
        return out


def execute_program(x_onehot: torch.Tensor, out: dict) -> torch.Tensor:
    """
    Minimal executor for diagnostics:
    - event_logits: {0: recolor_by_feature (noop here), 1: mirror, 2: translate}
    - axis_logits: {0: x, 1: y}
    - dy_logits/dx_logits: 7-way with center=3 -> shift in [-3,3]

    Args:
        x_onehot: (B,10,H,W) one-hot input grid
        out: dict from GeneralInterpreter.forward

    Returns:
        y_pred_logits: (B,10,H,W) logits after executing predicted op
    """
    B, K, H, W = x_onehot.shape
    device = x_onehot.device
    event = out['event_logits'].argmax(dim=1)  # (B,)
    axis = out['axis_logits'].argmax(dim=1) if 'axis_logits' in out else torch.zeros(B, dtype=torch.long, device=device)
    dy = (out['dy_logits'].argmax(dim=1) - 3) if 'dy_logits' in out else torch.zeros(B, dtype=torch.long, device=device)
    dx = (out['dx_logits'].argmax(dim=1) - 3) if 'dx_logits' in out else torch.zeros(B, dtype=torch.long, device=device)

    y = x_onehot.clone()
    for b in range(B):
        eb = int(event[b].item())
        if eb == 1:
            # mirror
            if int(axis[b].item()) == 0:
                y[b] = torch.flip(y[b], dims=[2])
            else:
                y[b] = torch.flip(y[b], dims=[3])
        elif eb == 2:
            # translate
            shy = int(dy[b].item())
            shx = int(dx[b].item())
            yb = torch.zeros_like(y[b])
            ys = slice(max(0, shy), H + min(0, shy))
            xs = slice(max(0, shx), W + min(0, shx))
            yd = slice(max(0, -shy), H + min(0, -shy))
            xd = slice(max(0, -shx), W + min(0, -shx))
            yb[:, yd, xd] = y[b][:, ys, xs]
            y[b] = yb
        else:
            # recolor_by_feature: noop placeholder
            pass
    # Return logits; simple log-softmax of one-hot
    eps = 1e-6
    y = y.clamp(min=eps, max=1.0)
    y_logits = (y / y.sum(dim=1, keepdim=True)).log()
    return y_logits


