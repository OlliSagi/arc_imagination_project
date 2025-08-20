from __future__ import annotations

import os
from typing import Tuple

import numpy as np
from PIL import Image
import torch

from .data import PALETTE


def save_grid_image(grid_logits: torch.Tensor, path: str) -> None:
    """grid_logits: (10,H,W) or (1,10,H,W) torch tensor. Saves a PNG."""
    if grid_logits.dim() == 4:
        grid_logits = grid_logits[0]
    H, W = grid_logits.shape[1], grid_logits.shape[2]
    pred = grid_logits.softmax(dim=0).argmax(dim=0).cpu().numpy().astype(np.uint8)
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for c, rgbv in PALETTE.items():
        rgb[pred == c] = np.array(rgbv, dtype=np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(rgb, mode='RGB').save(path)


