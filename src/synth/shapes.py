from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Any

Color = int  # 0-9


def empty_grid(h: int, w: int, fill: int = 0) -> np.ndarray:
    grid = np.full((h, w), fill, dtype=np.int8)
    return grid


def place_rect(grid: np.ndarray, top: int, left: int, height: int, width: int, color: Color) -> None:
    h, w = grid.shape
    y0, y1 = max(0, top), min(h, top + height)
    x0, x1 = max(0, left), min(w, left + width)
    if y0 < y1 and x0 < x1:
        grid[y0:y1, x0:x1] = color


def carve_hole(grid: np.ndarray, top: int, left: int, height: int, width: int, background: Color = 0) -> None:
    place_rect(grid, top, left, height, width, background)


def place_disk(grid: np.ndarray, cy: int, cx: int, radius: int, color: Color) -> None:
    h, w = grid.shape
    y0 = max(0, cy - radius)
    y1 = min(h, cy + radius + 1)
    x0 = max(0, cx - radius)
    x1 = min(w, cx + radius + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    grid[y0:y1, x0:x1][mask] = color


def connected_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """Label connected components in a binary mask (4-connectivity)."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current = 0
    for y in range(h):
        for x in range(w):
            if mask[y, x] and labels[y, x] == 0:
                current += 1
                stack = [(y, x)]
                labels[y, x] = current
                while stack:
                    cy, cx = stack.pop()
                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current
                            stack.append((ny, nx))
    return labels, current


def hole_count(shape_mask: np.ndarray, background: Color = 0) -> int:
    """Count holes as background components fully enclosed by the shape.

    Approach: flood-fill background from border; remaining background pixels are holes.
    """
    h, w = shape_mask.shape
    # Background mask: True where not shape
    bg = ~shape_mask
    # Flood-fill from border
    visited = np.zeros_like(bg, dtype=bool)
    stack: List[Tuple[int,int]] = []
    for y in range(h):
        for x in (0, w-1):
            if bg[y, x] and not visited[y, x]:
                visited[y, x] = True
                stack.append((y, x))
    for x in range(w):
        for y in (0, h-1):
            if bg[y, x] and not visited[y, x]:
                visited[y, x] = True
                stack.append((y, x))
    while stack:
        cy, cx = stack.pop()
        for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and bg[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                stack.append((ny, nx))
    # Holes are bg pixels not reached from the border
    hole_mask = bg & (~visited)
    _, holes = connected_components(hole_mask)
    return holes


def extract_objects(grid: np.ndarray) -> List[Dict[str, Any]]:
    """Extract connected same-color components as objects with simple features."""
    h, w = grid.shape
    objects: List[Dict[str, Any]] = []
    for color in range(10):
        mask = (grid == color)
        if not mask.any():
            continue
        labels, count = connected_components(mask)
        for idx in range(1, count + 1):
            comp = (labels == idx)
            ys, xs = np.where(comp)
            if ys.size == 0:
                continue
            y0, y1 = ys.min(), ys.max()
            x0, x1 = xs.min(), xs.max()
            bbox = (int(y0), int(x0), int(y1), int(x1))
            holes = hole_count(comp)
            area = int(comp.sum())
            objects.append({
                "color": int(color),
                "mask": comp,
                "bbox": bbox,
                "area": area,
                "holes": int(holes),
            })
    return objects


def recolor_by_feature(grid: np.ndarray, objects: List[Dict[str, Any]], feature: str, lookup: Dict[int, Color]) -> np.ndarray:
    out = grid.copy()
    for obj in objects:
        key = int(obj.get(feature, 0))
        if key in lookup:
            color = int(lookup[key])
            out[obj["mask"]] = color
    return out


def sample_rect_with_holes(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    grid = empty_grid(h, w, fill=0)
    H = int(rng.integers(4, max(5, min(12, h))))
    W = int(rng.integers(4, max(5, min(12, w))))
    top = int(rng.integers(0, max(1, h - H)))
    left = int(rng.integers(0, max(1, w - W)))
    color = int(rng.integers(1, 10))
    place_rect(grid, top, left, H, W, color)
    num_holes = int(rng.integers(0, 4))
    for _ in range(num_holes):
        hh = int(rng.integers(1, max(2, H // 2)))
        ww = int(rng.integers(1, max(2, W // 2)))
        ty = int(rng.integers(top + 1, min(h - 1, top + H - hh)))
        tx = int(rng.integers(left + 1, min(w - 1, left + W - ww)))
        carve_hole(grid, ty, tx, hh, ww, background=0)
    return grid


def sample_disk_with_holes(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    grid = empty_grid(h, w, fill=0)
    radius = int(rng.integers(3, max(4, min(h, w) // 3)))
    cy = int(rng.integers(radius + 1, h - radius - 1))
    cx = int(rng.integers(radius + 1, w - radius - 1))
    color = int(rng.integers(1, 10))
    place_disk(grid, cy, cx, radius, color)
    num_holes = int(rng.integers(0, 3))
    for _ in range(num_holes):
        r2 = int(max(1, radius // int(rng.integers(2, 4))))
        oy = int(rng.integers(-radius // 2, radius // 2))
        ox = int(rng.integers(-radius // 2, radius // 2))
        place_disk(grid, cy + oy, cx + ox, r2, 0)
    return grid


def sample_object_with_holes(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        return sample_rect_with_holes(h, w, rng)
    return sample_disk_with_holes(h, w, rng)


