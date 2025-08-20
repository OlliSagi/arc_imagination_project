import argparse, os, json, numpy as np
from typing import Dict, Any, List

from arc_data import load_task


def detect_recolor_by_feature(pair: Dict[str, Any]) -> Dict[str, Any] | None:
    """Very simple heuristic: if objects keep shape but colors change by a function of hole_count or area.
    Placeholder: returns None; extend with real detectors later.
    """
    return None


def detect_mirror(pair: Dict[str, Any]) -> Dict[str, Any] | None:
    x = np.array(pair['input'], dtype=np.int32)
    y = np.array(pair['output'], dtype=np.int32)
    if x.shape == y.shape and np.array_equal(np.flip(x, axis=1), y):
        return {"type": "mirror", "axis": "x"}
    if x.shape == y.shape and np.array_equal(np.flip(x, axis=0), y):
        return {"type": "mirror", "axis": "y"}
    return None


def detect_translate(pair: Dict[str, Any]) -> Dict[str, Any] | None:
    x = np.array(pair['input'], dtype=np.int32)
    y = np.array(pair['output'], dtype=np.int32)
    h, w = x.shape
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            out = np.zeros_like(x)
            sy0 = max(0, -dy); sx0 = max(0, -dx)
            dy0 = max(0, dy); dx0 = max(0, dx)
            hh = min(h - sy0, h - dy0); ww = min(w - sx0, w - dx0)
            if hh > 0 and ww > 0:
                out[dy0:dy0+hh, dx0:dx0+ww] = x[sy0:sy0+hh, sx0:sx0+ww]
            if np.array_equal(out, y):
                return {"type": "translate", "dy": dy, "dx": dx}
    return None


def mine_events(task_path: str) -> Dict[str, Any]:
    tj = load_task(task_path)
    findings: List[Dict[str, Any]] = []
    for pair in tj['train']:
        for detector in (detect_mirror, detect_translate, detect_recolor_by_feature):
            info = detector(pair)
            if info is not None:
                findings.append(info)
                break
    return {"task": os.path.basename(task_path), "findings": findings}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_dir", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    tasks = [os.path.join(args.task_dir, f) for f in os.listdir(args.task_dir) if f.endswith('.json')]
    results: List[Dict[str, Any]] = []
    for p in tasks:
        results.append(mine_events(p))
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Wrote mined events to {args.out}")


if __name__ == '__main__':
    main()


