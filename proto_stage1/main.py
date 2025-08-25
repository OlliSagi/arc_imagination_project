from __future__ import annotations

import argparse, json, os, random
import numpy as np
import torch
import torch.nn.functional as F

from src.utils import load_config
from proto_canvas.canvas import Sim2D
from proto_stage1.heads import SlotHeads
from proto_stage1.interpreter import GeneralInterpreter, execute_program
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
        sd = torch.load(cfg['model']['load_stage0_checkpoint'], map_location='cpu')
        sim.load_state_dict(sd, strict=False)
    if cfg['model']['freeze_sim2d']:
        for p in sim.parameters():
            p.requires_grad_(False)

    # Generalized interpreter (predicate bank + z_story) with diagnostic heads
    heads = GeneralInterpreter(
        input_dim=cfg['heads']['input_dim'],
        z_dim=cfg['heads']['z_dim'],
        num_predicates=cfg['heads']['num_predicates'],
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
            y = batch['y_onehot'].to(device)
            S = sim.init_state(x)
            r = global_readout(S)
            out = heads(r)
            # Diagnostic multi-head CE
            w_diag = float(cfg['train'].get('w_diag', 1.0))
            w_pred = float(cfg['train'].get('w_pred_usage', 0.001))
            w_recon = float(cfg['train'].get('w_recon', 1.0))
            w_probe = float(cfg['train'].get('w_probe', 0.0))

            loss_diag = w_diag * (
                ce(out['event_logits'], batch['event'].to(device)) +
                ce(out['axis_logits'], batch['axis'].to(device)) +
                ce(out['dy_logits'], batch['dy'].to(device)) +
                ce(out['dx_logits'], batch['dx'].to(device)) +
                ce(out['feature_logits'], batch['feature'].to(device))
            )
            # Predicate usage regularizer (encourage sparsity/selection)
            loss_pred = w_pred * out['predicate_gates'].mean()
            
            # Reconstruction
            use_exec = bool(cfg['train'].get('use_executor', True))
            if use_exec:
                # via minimal executor applying predicted ops to x
                y_logits = execute_program(x, out)
            else:
                # fallback: predict y directly from r via a small head or sim decode
                if not hasattr(sim, 'decode_logits'):
                    # small linear head from r
                    recon_head = getattr(main, '_recon_head', None)
                    if recon_head is None:
                        import torch.nn as nn
                        recon = nn.Sequential(nn.Linear(r.shape[1], 10 * H * W))
                        recon = recon.to(device)
                        main._recon_head = recon  # cache
                    recon_head = main._recon_head
                    y_logits_flat = recon_head(r)
                    y_logits = y_logits_flat.view(-1, 10, H, W)
                else:
                    y_logits = sim.decode_logits(S)
            loss_recon = w_recon * torch.nn.functional.cross_entropy(y_logits, y.argmax(dim=1))

            # Probe losses: mirror twice -> identity; translate + then - -> identity
            probe_loss = torch.tensor(0.0, device=device)
            if w_probe > 0.0 and use_exec:
                with torch.no_grad():
                    # derive opposite translation
                    dy_logits = out['dy_logits'] if 'dy_logits' in out else None
                    dx_logits = out['dx_logits'] if 'dx_logits' in out else None
                    event_logits = out['event_logits']
                # mirror twice
                out_mirror = {k: v for k, v in out.items()}
                out_mirror['event_logits'] = torch.nn.functional.one_hot(torch.full((x.shape[0],), 1, device=device), num_classes=cfg['heads']['event_classes']).float()
                y1 = execute_program(x, out_mirror)
                y2 = execute_program(torch.softmax(y1, dim=1), out_mirror)
                target_x = x.argmax(dim=1)
                probe_loss += torch.nn.functional.cross_entropy(y2, target_x)
                # translate inverse
                if (dy_logits is not None) and (dx_logits is not None):
                    dy_idx = dy_logits.argmax(dim=1)
                    dx_idx = dx_logits.argmax(dim=1)
                    inv = {k: v for k, v in out.items()}
                    inv['event_logits'] = torch.nn.functional.one_hot(torch.full((x.shape[0],), 2, device=device), num_classes=cfg['heads']['event_classes']).float()
                    inv['dy_logits'] = torch.nn.functional.one_hot(6 - dy_idx, num_classes=cfg['heads']['dy_classes']).float()
                    inv['dx_logits'] = torch.nn.functional.one_hot(6 - dx_idx, num_classes=cfg['heads']['dx_classes']).float()
                    y_forward = execute_program(x, out)
                    y_back = execute_program(torch.softmax(y_forward, dim=1), inv)
                    probe_loss += torch.nn.functional.cross_entropy(y_back, target_x)

            loss = loss_diag + loss_pred + loss_recon + w_probe * probe_loss
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
                    'loss_recon': float(loss_recon.item()),
                    'event_acc': acc(out['event_logits'], batch['event'].to(device)),
                    'axis_acc': acc(out['axis_logits'], batch['axis'].to(device)),
                    'dy_acc': acc(out['dy_logits'], batch['dy'].to(device)),
                    'dx_acc': acc(out['dx_logits'], batch['dx'].to(device)),
                    'feature_acc': acc(out['feature_logits'], batch['feature'].to(device)),
                }
                mf.write(json.dumps(metrics) + '\n')
                mf.flush()
                if step % int(cfg['train']['log_every']) == 0:
                    print(metrics)
                if step % cfg['train']['ckpt_every'] == 0:
                    torch.save(sim.state_dict(), os.path.join(out_dir, 'checkpoints', 'sim2d_stage1.safetensors'))
                    torch.save(heads.state_dict(), os.path.join(out_dir, 'checkpoints', 'readout_stage1.safetensors'))

    print('Stage 1 training scaffold complete (no run performed here).')


if __name__ == '__main__':
    main()


