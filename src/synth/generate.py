from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from .events import recolor_by_holes_episode, mirror_x_episode, translate_episode


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic ARC-style episodes")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for JSON episodes")
    parser.add_argument("--num", type=int, default=50, help="Number of episodes to generate")
    parser.add_argument("--events", type=str, default="recolor_by_holes,mirror_x,translate", help="Comma-separated list of event families")
    parser.add_argument("--height", type=int, default=20, help="Grid height")
    parser.add_argument("--width", type=int, default=20, help="Grid width")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    families = [s.strip() for s in args.events.split(',') if s.strip()]

    for i in range(args.num):
        family = families[i % len(families)]
        if family == "recolor_by_holes":
            episode = recolor_by_holes_episode(args.height, args.width, rng)
        elif family == "mirror_x":
            episode = mirror_x_episode(args.height, args.width, rng)
        elif family == "translate":
            episode = translate_episode(args.height, args.width, rng)
        else:
            episode = recolor_by_holes_episode(args.height, args.width, rng)
        episode_id = episode["episode_id"]
        path = out_dir / f"{episode_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(episode, f, ensure_ascii=False)

    print(f"Wrote {args.num} episodes to {str(out_dir)}")


if __name__ == "__main__":
    main()


