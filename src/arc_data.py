import json, os, glob, numpy as np

def load_task(path):
    with open(path, 'r') as f:
        return json.load(f)

def grid_to_tensor(grid, H=32, W=32, num_colors=10):
    # grid: list of lists ints 0..9, shape h x w
    h, w = len(grid), len(grid[0])
    pad_h, pad_w = H, W
    t = np.zeros((num_colors, H, W), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            c = grid[i][j]
            if 0 <= i < H and 0 <= j < W and 0 <= c < num_colors:
                t[c, i, j] = 1.0
    return t

def pair_to_tensors(pair, H=32, W=32):
    x = grid_to_tensor(pair['input'], H, W)   # (10,H,W)
    y = grid_to_tensor(pair['output'], H, W)  # (10,H,W)
    return x, y

def task_pairs(task_json):
    return task_json['train'], task_json['test']
