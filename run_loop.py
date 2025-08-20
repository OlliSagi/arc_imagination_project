import argparse, time, torch
from utils import load_config
from sim2d import Sim2D, init_field

def main(cfg):
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    H,W,C = cfg['sim2d']['H'], cfg['sim2d']['W'], cfg['sim2d']['C']
    sim = Sim2D(H,W,C, dim=cfg['sim2d']['dim'], layers=cfg['sim2d']['layers'], heads=cfg['sim2d']['heads']).to(device)
    S = init_field(1, C, H, W, device)
    print("Always-on Sim2D demo (dummy moving brush):")
    for t in range(60):
        # dummy edit that sweeps across
        cx = (t % H) / float(H-1); cy = (t % W) / float(W-1)
        edit = torch.zeros(1,1,4+C, device=device)
        edit[:,:,0]=cx; edit[:,:,1]=cy; edit[:,:,2]=0.18; edit[:,:,3]=0.6
        edit[:,:,4]=1.0  # nudge first channel
        S = sim.step(S, edit_tokens=edit)
        print(f"t={t:03d} | field-norm={S.norm().item():.4f}")
        time.sleep(0.05)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/arc_mvp.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)
