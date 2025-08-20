import argparse, os, glob, json, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
from tqdm import trange
from utils import load_config, save_safetensors
from sim2d import Sim2D, init_field
from readout import Readout
from arc_data import load_task, pair_to_tensors

"""
Simplified ARC-coupled training:
- For each training pair, we initialize S with the input grid (one-hot into color channels, zeros elsewhere).
- A tiny "planner" MLP (stand-in for an LLM edit head) emits K edit tokens per few steps.
- We optimize edit tokens (and Sim2D) to minimize output grid error.
Later: replace planner with an LLM+EditHead and add round-trip text supervision.
"""

class Planner(nn.Module):
    def __init__(self, readout_dim=256, C=96, K=3):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(readout_dim, 512), nn.GELU(), nn.Linear(512, K*(4+C)))
        self.C, self.K = C, K
    def forward(self, r):
        out = self.fc(r).view(r.size(0), self.K, 4+self.C)
        out[:,:,0:2] = out[:,:,0:2].sigmoid()
        out[:,:,2]   = out[:,:,2].sigmoid()*0.5 + 0.05
        out[:,:,3]   = torch.tanh(out[:,:,3])
        return out

def grid_loss(S_colors, Y_colors):
    # S_colors: (B,10,H,W) first 10 channels of S
    return ((S_colors - Y_colors)**2).mean()

def main(cfg):
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    H,W,C = cfg['sim2d']['H'], cfg['sim2d']['W'], cfg['sim2d']['C']
    sim = Sim2D(H,W,C, dim=cfg['sim2d']['dim'], layers=cfg['sim2d']['layers'], heads=cfg['sim2d']['heads']).to(device)
    readout = Readout(C=C, H=H, W=W, dim=cfg['sim2d']['dim']).to(device)
    planner = Planner(readout_dim=cfg['sim2d']['dim'], C=C, K=cfg['sim2d']['edit_tokens']).to(device)

    opt = optim.AdamW(list(sim.parameters())+list(readout.parameters())+list(planner.parameters()), lr=float(cfg['train']['lr_sim']))

    # find ARC training JSON (user supplies path; here we just expect env var or prompt path)
    arc_train_dir = os.environ.get("ARC_TRAIN_DIR", None)
    if arc_train_dir is None or not os.path.isdir(arc_train_dir):
        print("Set ARC_TRAIN_DIR to ARC-AGI-2/data/training to train on real tasks. Using synthetic placeholders only.")
        tasks = []
    else:
        tasks = [os.path.join(arc_train_dir, f) for f in os.listdir(arc_train_dir) if f.endswith(".json")]
        random.shuffle(tasks)

    steps = cfg['train']['steps_arc']
    pbar = trange(steps, desc="ARC-coupled training")
    for step in pbar:
        # random synthetic or ARC sample
        if tasks:
            tj = load_task(random.choice(tasks))
            pair = random.choice(tj['train'])
            x,y = pair_to_tensors(pair, H, W)          # (10,H,W)
        else:
            # synthetic: simple rectangle fill target
            x = np.zeros((10,H,W), dtype=np.float32)
            y = np.zeros((10,H,W), dtype=np.float32)
            y[1, 8:24, 10:22] = 1.0

        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(y).to(device)

        # initialize S with color channels set to x (rest zeros)
        S = torch.zeros(1, C, H, W, device=device)
        S[:, :10] = x.unsqueeze(0)

        # a few imagination/planning steps
        for _ in range(4):
            r = readout(S)
            edits = planner(r)
            S = sim.step(S, edit_tokens=edits)

        loss = grid_loss(S[:, :10], y.unsqueeze(0))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % max(1, steps//10) == 0:
            pbar.set_postfix(loss=float(loss.item()))

    os.makedirs(cfg['paths']['ckpt_dir'], exist_ok=True)
    save_safetensors(sim.state_dict(), os.path.join(cfg['paths']['ckpt_dir'], "sim2d_arc.safetensors"))
    save_safetensors(readout.state_dict(), os.path.join(cfg['paths']['ckpt_dir'], "readout_arc.safetensors"))
    save_safetensors(planner.state_dict(), os.path.join(cfg['paths']['ckpt_dir'], "planner_arc.safetensors"))
    print("Saved Sim2D + Readout + Planner ARC weights.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/arc_mvp.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    torch.manual_seed(cfg['seed'])
    main(cfg)
