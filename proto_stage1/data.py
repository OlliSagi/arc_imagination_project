from __future__ import annotations

import json, os, random
from typing import Dict, Any, List, Tuple

import numpy as np
import torch


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


def episodes_in_dirs(dirs: List[str]) -> List[str]:
    out: List[str] = []
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.endswith('.json'):
                out.append(os.path.join(d, f))
    random.shuffle(out)
    return out


def collate_batch(paths: List[str], H: int, W: int, ignore_index: int) -> Dict[str, torch.Tensor]:
    # For simplicity, take the first train pair as supervision target
    xs: List[np.ndarray] = []
    event_labels: List[int] = []
    axis_labels: List[int] = []
    dy_labels: List[int] = []
    dx_labels: List[int] = []
    feature_labels: List[int] = []

    for p in paths:
        ep = read_json(p)
        pair = ep['train'][0]
        x = center_pad(pair['input'], H, W)
        xs.append(x)
        # Labels
        etype = ep['event']['type'][0]
        if etype == 'recolor_by_feature':
            event_labels.append(0)
            feature_labels.append(0)  # holes
            axis_labels.append(ignore_index)
            dy_labels.append(ignore_index)
            dx_labels.append(ignore_index)
        elif etype == 'mirror':
            event_labels.append(1)
            axis_labels.append(0 if ep['event']['params'].get('axis','x') == 'x' else 1)
            feature_labels.append(ignore_index)
            dy_labels.append(ignore_index)
            dx_labels.append(ignore_index)
        elif etype == 'translate':
            event_labels.append(2)
            dy = int(ep['event']['params']['dy'])
            dx = int(ep['event']['params']['dx'])
            dy_labels.append(dy + 3)
            dx_labels.append(dx + 3)
            axis_labels.append(ignore_index)
            feature_labels.append(ignore_index)
        else:
            event_labels.append(0)
            feature_labels.append(ignore_index)
            axis_labels.append(ignore_index)
            dy_labels.append(ignore_index)
            dx_labels.append(ignore_index)

    B = len(xs)
    x_onehot = np.zeros((B, 10, H, W), dtype=np.float32)
    for i, g in enumerate(xs):
        for y in range(H):
            for x in range(W):
                c = int(g[y, x])
                if 0 <= c < 10:
                    x_onehot[i, c, y, x] = 1.0
    return {
        'x_onehot': torch.from_numpy(x_onehot),
        'event': torch.tensor(event_labels, dtype=torch.long),
        'axis': torch.tensor(axis_labels, dtype=torch.long),
        'dy': torch.tensor(dy_labels, dtype=torch.long),
        'dx': torch.tensor(dx_labels, dtype=torch.long),
        'feature': torch.tensor(feature_labels, dtype=torch.long),
    }


