from __future__ import annotations

import argparse, json, os, random
import numpy as np
import torch

from src.utils import load_config
from proto_canvas.canvas import Sim2D
from proto_canvas.data import load_arc_grids, augment_grids, batch_from_grids
from proto_canvas.trainer import train_step, LossWeights
from proto_canvas.viz import save_grid_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='proto_canvas/config.yaml')
    ap.add_argument('--steps', type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.steps is not None:
        cfg['train']['steps'] = int(args.steps)

    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])

    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    H, W, C = cfg['model']['H'], cfg['model']['W'], cfg['model']['C']
    sim = Sim2D(H, W, C, dim=cfg['model']['dim'], layers=cfg['model']['layers']).to(device)

    # Data
    arc_dirs = [cfg['paths']['arc1_train_dir'], cfg['paths']['arc2_train_dir']]
    grids = load_arc_grids(arc_dirs, H, W)
    if len(grids) == 0:
        print('Warning: No ARC grids found. Stage 0 can still run with random shapes (not implemented here).')
    grids = augment_grids(grids)

    # Train
    opt = torch.optim.AdamW(sim.parameters(), lr=float(cfg['train']['lr']))
    lw = LossWeights()

    out_dir = cfg['paths']['out_dir']
    os.makedirs(os.path.join(out_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'samples'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'checkpoints'), exist_ok=True)
    metrics_path = os.path.join(out_dir, 'logs', 'metrics.jsonl')
    steps = int(cfg['train']['steps'])

    with open(metrics_path, 'w', encoding='utf-8') as mf:
        for step in range(1, steps + 1):
            batch_x = batch_from_grids(grids, cfg['train']['batch_size'], H, W)
            metrics = train_step(sim, opt, batch_x, cfg, lw, device)
            if step % cfg['train']['log_every'] == 0:
                mf.write(json.dumps({'step': step, **metrics}) + '\n')
                mf.flush()
                print({'step': step, **metrics})
            if step % cfg['train']['sample_every'] == 0:
                with torch.no_grad():
                    S = sim.init_state(batch_x.to(device))
                    logits = sim.decode_logits(S).cpu()
                save_grid_image(logits[0], os.path.join(out_dir, 'samples', f'step_{step:04d}.png'))

    torch.save(sim.state_dict(), os.path.join(out_dir, 'checkpoints', 'canvas_stage0.safetensors'))
    print(f"Saved {os.path.join(out_dir, 'checkpoints', 'canvas_stage0.safetensors')}")


if __name__ == '__main__':
    main()


