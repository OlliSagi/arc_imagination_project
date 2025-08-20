import argparse, os, torch, torch.nn as nn, torch.optim as optim
from tqdm import trange
from utils import load_config, save_safetensors
from sim2d import Sim2D, init_field

def next_state_loss(pred, target):
    return ((pred - target)**2).mean()

def main(cfg):
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    H,W,C = cfg['sim2d']['H'], cfg['sim2d']['W'], cfg['sim2d']['C']
    model = Sim2D(H,W,C, dim=cfg['sim2d']['dim'], layers=cfg['sim2d']['layers'], heads=cfg['sim2d']['heads']).to(device)
    opt = optim.AdamW(model.parameters(), lr=float(cfg['train']['lr_sim']))
    B = cfg['train']['batch_size']
    steps = cfg['train']['steps_boot']
    S = init_field(B, C, H, W, device)

    pbar = trange(steps, desc="Booting Sim2D")
    for step in pbar:
        S_pred = model.step(S, edit_tokens=None)
        with torch.no_grad():
            S_tgt = model.step(S, edit_tokens=None).detach()
        loss = next_state_loss(S_pred, S_tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        S = S_pred.detach()
        if step % max(1, steps//10) == 0:
            pbar.set_postfix(loss=float(loss.item()))

    os.makedirs(cfg['paths']['ckpt_dir'], exist_ok=True)
    out = os.path.join(cfg['paths']['ckpt_dir'], "sim2d_boot.safetensors")
    save_safetensors(model.state_dict(), out)
    print(f"Saved {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/arc_mvp.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    torch.manual_seed(cfg['seed'])
    main(cfg)
