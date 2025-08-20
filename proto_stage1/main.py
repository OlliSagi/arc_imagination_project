from __future__ import annotations

import argparse, json, os, random
import numpy as np
import torch
import torch.nn.functional as F

from utils import load_config
from proto_canvas.canvas import Sim2D
from proto_stage1.heads import SlotHeads
from proto_stage1.data import episodes_in_dirs, collate_batch


def global_readout(S: torch.Tensor) -> torch.Tensor:
    # Simple readout: global average over H,W -> (B,C)
    return S.mean(dim=(2,3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='proto_stage1/config.yaml')
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
    if os.path.isfile(cfg['model']['load_stage0_checkpoint']):
        sd = torch.load(cfg['model']['load_stage0_checkpoint'], map_location=device)
        sim.load_state_dict(sd, strict=False)
    if cfg['model']['freeze_sim2d']:
        for p in sim.parameters():
            p.requires_grad_(False)

    heads = SlotHeads(
        input_dim=cfg['heads']['input_dim'],
        event_classes=cfg['heads']['event_classes'],
        axis_classes=cfg['heads']['axis_classes'],
        dy_classes=cfg['heads']['dy_classes'],
        dx_classes=cfg['heads']['dx_classes'],
        feature_classes=cfg['heads']['feature_classes'],
    ).to(device)

    opt = torch.optim.AdamW([p for p in heads.parameters() if p.requires_grad], lr=float(cfg['train']['lr_heads']))

    out_dir = cfg['paths']['out_dir']
    os.makedirs(os.path.join(out_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'checkpoints'), exist_ok=True)
    metrics_path = os.path.join(out_dir, 'logs', 'metrics_stage1.jsonl')

    episodes = episodes_in_dirs(cfg['data']['episode_dirs'])
    steps = int(cfg['train']['steps'])
    B = int(cfg['train']['batch_size'])

    def ce(logits, target):
        return F.cross_entropy(logits, target, ignore_index=cfg['train']['ignore_index'])

    with open(metrics_path, 'w', encoding='utf-8') as mf:
        for step in range(1, steps + 1):
            batch_paths = episodes[(step * B) % max(1, len(episodes)) : (step * B) % max(1, len(episodes)) + B]
            if len(batch_paths) < B:
                batch_paths = random.sample(episodes, B)
            batch = collate_batch(batch_paths, H, W, cfg['train']['ignore_index'])
            x = batch['x_onehot'].to(device)
            S = sim.init_state(x)
            r = global_readout(S)
            out = heads(r)
            loss = (
                ce(out['event_logits'], batch['event'].to(device)) +
                ce(out['axis_logits'], batch['axis'].to(device)) +
                ce(out['dy_logits'], batch['dy'].to(device)) +
                ce(out['dx_logits'], batch['dx'].to(device)) +
                ce(out['feature_logits'], batch['feature'].to(device))
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            with torch.no_grad():
                def acc(logits, target):
                    mask = target != cfg['train']['ignore_index']
                    if mask.sum() == 0:
                        return 1.0
                    pred = logits.argmax(dim=1)
                    return float(((pred == target) & mask).float().sum() / mask.float().sum())
                metrics = {
                    'step': step,
                    'loss': float(loss.item()),
                    'event_acc': acc(out['event_logits'], batch['event'].to(device)),
                    'axis_acc': acc(out['axis_logits'], batch['axis'].to(device)),
                    'dy_acc': acc(out['dy_logits'], batch['dy'].to(device)),
                    'dx_acc': acc(out['dx_logits'], batch['dx'].to(device)),
                    'feature_acc': acc(out['feature_logits'], batch['feature'].to(device)),
                }
                mf.write(json.dumps(metrics) + '\n')
                mf.flush()
                if step % cfg['train']['ckpt_every'] == 0:
                    torch.save(sim.state_dict(), os.path.join(out_dir, 'checkpoints', 'sim2d_stage1.safetensors'))
                    torch.save(heads.state_dict(), os.path.join(out_dir, 'checkpoints', 'readout_stage1.safetensors'))

    print('Stage 1 training scaffold complete (no run performed here).')


if __name__ == '__main__':
    main()


