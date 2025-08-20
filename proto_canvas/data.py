from __future__ import annotations

import json, os, random
from typing import List, Tuple

import numpy as np
import torch


PALETTE = {
    0: (0, 0, 0),
    1: (0, 114, 217),
    2: (255, 65, 54),
    3: (46, 204, 64),
    4: (255, 220, 0),
    5: (170, 170, 170),
    6: (240, 18, 190),
    7: (255, 133, 27),
    8: (127, 219, 255),
    9: (135, 12, 37),
}


def read_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def center_pad(grid: List[List[int]], H: int, W: int) -> np.ndarray:
    h, w = len(grid), len(grid[0])
    out = np.zeros((H, W), dtype=np.int64)
    y0 = (H - h) // 2
    x0 = (W - w) // 2
    for i in range(h):
        for j in range(w):
            out[y0 + i, x0 + j] = int(grid[i][j])
    return out


def load_arc_grids(arc_dirs: List[str], H: int, W: int) -> List[np.ndarray]:
    grids: List[np.ndarray] = []
    for d in arc_dirs:
        if not d or not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if not f.endswith('.json'):
                continue
            task = read_json(os.path.join(d, f))
            for pair in task.get('train', []):
                grids.append(center_pad(pair['input'], H, W))
                grids.append(center_pad(pair['output'], H, W))
            for pair in task.get('test', []):
                if 'input' in pair:
                    grids.append(center_pad(pair['input'], H, W))
                if 'output' in pair:
                    grids.append(center_pad(pair['output'], H, W))
    # Deduplicate by bytes
    uniq = {}
    for g in grids:
        uniq[g.tobytes()] = g
    return list(uniq.values())


def augment_grids(grids: List[np.ndarray]) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for g in grids:
        # Rotations 0,90,180,270 and flips
        rots = [g, np.rot90(g, 1), np.rot90(g, 2), np.rot90(g, 3)]
        for r in rots:
            out.append(r.copy())
            out.append(np.flip(r, axis=0).copy())
            out.append(np.flip(r, axis=1).copy())
    return out


def batch_from_grids(grids: List[np.ndarray], B: int, H: int, W: int, num_colors: int = 10) -> torch.Tensor:
    idxs = np.random.choice(len(grids), size=B, replace=True)
    x = np.zeros((B, num_colors, H, W), dtype=np.float32)
    for bi, ix in enumerate(idxs):
        g = grids[ix]
        for i in range(H):
            for j in range(W):
                c = int(g[i, j])
                if 0 <= c < num_colors:
                    x[bi, c, i, j] = 1.0
    return torch.from_numpy(x)


