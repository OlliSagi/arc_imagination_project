from __future__ import annotations

import numpy as np
from typing import Dict, Any, List, Tuple

from .shapes import empty_grid, place_rect, carve_hole, extract_objects, recolor_by_feature, sample_object_with_holes


def compose_scene(h: int, w: int, rng: np.random.Generator, num_objs: int = 3) -> np.ndarray:
    grid = empty_grid(h, w, fill=0)
    for _ in range(num_objs):
        obj = sample_object_with_holes(h, w, rng)
        mask = obj != 0
        grid[mask] = obj[mask]
    return grid


def recolor_by_holes_episode(h: int, w: int, rng: np.random.Generator, n_train_pairs: int = 2, n_tests: int = 1) -> Dict[str, Any]:
    # Sample a mapping from hole_count to color
    # Map a few bins 0..3 (others default to a color or unchanged)
    bins = [0, 1, 2, 3]
    colors = rng.choice(np.arange(1, 10), size=len(bins), replace=False)
    lookup = {int(k): int(v) for k, v in zip(bins, colors)}

    def apply_mapping(grid: np.ndarray) -> np.ndarray:
        objects = extract_objects(grid)
        return recolor_by_feature(grid, objects, feature="holes", lookup=lookup)

    train_pairs: List[Dict[str, Any]] = []
    for _ in range(n_train_pairs):
        inp = compose_scene(h, w, rng, num_objs=int(rng.integers(2, 5)))
        out = apply_mapping(inp)
        train_pairs.append({"input": inp.tolist(), "output": out.tolist()})

    test_inputs: List[Any] = []
    for _ in range(n_tests):
        inp = compose_scene(h, w, rng, num_objs=int(rng.integers(2, 5)))
        # Remove colors to simulate uncolored shapes (set nonzero to 1 as placeholder or 0?)
        # Here we leave them as is; downstream can ignore colors and reassign per mapping
        test_inputs.append(inp.tolist())

    episode = {
        "episode_id": f"recolor_by_holes_{int(rng.integers(0, 1_000_000))}",
        "train_pairs": train_pairs,
        "test_inputs": test_inputs,
        "event": {
            "type": ["recolor_by_feature"],
            "params": {"feature": "holes", "lookup": lookup},
        },
        "program": [
            {"op": "recolor_by_feature", "feature": "holes", "lookup": lookup},
        ],
    }
    return episode


def mirror_x_episode(h: int, w: int, rng: np.random.Generator, n_train_pairs: int = 2, n_tests: int = 1) -> Dict[str, Any]:
    def apply_mirror_x(grid: np.ndarray) -> np.ndarray:
        return np.flip(grid, axis=1)

    train_pairs: List[Dict[str, Any]] = []
    for _ in range(n_train_pairs):
        inp = compose_scene(h, w, rng, num_objs=int(rng.integers(2, 5)))
        out = apply_mirror_x(inp)
        train_pairs.append({"input": inp.tolist(), "output": out.tolist()})

    test_inputs: List[Any] = []
    for _ in range(n_tests):
        inp = compose_scene(h, w, rng, num_objs=int(rng.integers(2, 5)))
        test_inputs.append(inp.tolist())

    return {
        "episode_id": f"mirror_x_{int(rng.integers(0, 1_000_000))}",
        "train_pairs": train_pairs,
        "test_inputs": test_inputs,
        "event": {"type": ["mirror"], "params": {"axis": "x"}},
        "program": [{"op": "mirror", "axis": "x"}],
    }


def translate_episode(h: int, w: int, rng: np.random.Generator, n_train_pairs: int = 2, n_tests: int = 1) -> Dict[str, Any]:
    dy = int(rng.integers(-3, 4))
    dx = int(rng.integers(-3, 4))

    def apply_translate(grid: np.ndarray) -> np.ndarray:
        out = empty_grid(h, w, fill=0)
        src_y0 = max(0, -dy)
        src_x0 = max(0, -dx)
        dst_y0 = max(0, dy)
        dst_x0 = max(0, dx)
        copy_h = min(h - src_y0, h - dst_y0)
        copy_w = min(w - src_x0, w - dst_x0)
        if copy_h > 0 and copy_w > 0:
            out[dst_y0:dst_y0+copy_h, dst_x0:dst_x0+copy_w] = grid[src_y0:src_y0+copy_h, src_x0:src_x0+copy_w]
        return out

    train_pairs: List[Dict[str, Any]] = []
    for _ in range(n_train_pairs):
        inp = compose_scene(h, w, rng, num_objs=int(rng.integers(2, 5)))
        out = apply_translate(inp)
        train_pairs.append({"input": inp.tolist(), "output": out.tolist()})

    test_inputs: List[Any] = []
    for _ in range(n_tests):
        inp = compose_scene(h, w, rng, num_objs=int(rng.integers(2, 5)))
        test_inputs.append(inp.tolist())

    return {
        "episode_id": f"translate_{dy}_{dx}_{int(rng.integers(0, 1_000_000))}",
        "train_pairs": train_pairs,
        "test_inputs": test_inputs,
        "event": {"type": ["translate"], "params": {"dy": dy, "dx": dx}},
        "program": [{"op": "translate", "dy": dy, "dx": dx}],
    }


