import argparse, os, json, torch, numpy as np
from tqdm import tqdm
from utils import load_config
from sim2d import Sim2D, init_field
from readout import Readout
from arc_data import load_task, grid_to_tensor
from train_arc import Planner

"""
Evaluator that:
- loads tasks from ARC-AGI-2 JSON directory (evaluation/ or training/ for smoke tests)
- runs 2 attempts per test input by default (configurable)
- writes each task's prediction JSON in a "submissions/<model>/{task_id}.json" format
  compatible with the arc-agi-benchmarking scorer.
"""

def tensor_to_grid(t, H=32, W=32):
    # t: (10,H,W) -> argmax over channels, crop to tight bbox of nonzero if you want
    pred = t.argmax(0).cpu().numpy().tolist()
    # NOTE: ARC expects exact size; here we keep 32x32. A real solver should infer size.
    return pred

def infer_task(sim, readout, planner, task_json, attempts=2, H=32, W=32, device="cuda"):
    out_grids = []
    for test_pair in task_json['test']:
        x = grid_to_tensor(test_pair['input'], H, W)  # (10,H,W)
        x = torch.from_numpy(x).to(device)
        best = None
        for attempt in range(attempts):
            S = torch.zeros(1, sim.C, H, W, device=device)
            S[:, :10] = x.unsqueeze(0)
            # a few reasoning steps
            for _ in range(4 + attempt):  # small change for attempt 2
                r = readout(S)
                edits = planner(r)
                S = sim.step(S, edit_tokens=edits)
            grid = tensor_to_grid(S[0, :10])
            out_grids.append(grid)
    return out_grids

def main(cfg, task_dir, out_dir):
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    H,W,C = cfg['sim2d']['H'], cfg['sim2d']['W'], cfg['sim2d']['C']
    os.makedirs(out_dir, exist_ok=True)

    sim = Sim2D(H,W,C, dim=cfg['sim2d']['dim'], layers=cfg['sim2d']['layers'], heads=cfg['sim2d']['heads']).to(device)
    readout = Readout(C=C, H=H, W=W, dim=cfg['sim2d']['dim']).to(device)
    planner = Planner(readout_dim=cfg['sim2d']['dim'], C=C, K=cfg['sim2d']['edit_tokens']).to(device)

    # load weights if present
    ck = cfg['paths']['ckpt_dir']
    for m,fn in [(sim,"sim2d_arc.safetensors"),(readout,"readout_arc.safetensors"),(planner,"planner_arc.safetensors")]:
        p = os.path.join(ck, fn)
        if os.path.isfile(p):
            from safetensors.torch import load_file
            m.load_state_dict(load_file(p), strict=False)

    for fname in tqdm([f for f in os.listdir(task_dir) if f.endswith(".json")]):
        task_id = os.path.splitext(fname)[0]
        tj = load_task(os.path.join(task_dir, fname))
        preds = infer_task(sim, readout, planner, tj, attempts=cfg['eval']['attempts_per_test_input'], H=H, W=W, device=device)
        # Write submission format expected by arc-agi-benchmarking
        sub = {"task_id": task_id, "predictions": preds}
        with open(os.path.join(out_dir, f"{task_id}.json"), "w") as f:
            json.dump(sub, f)

    print(f"Wrote submissions to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/arc_mvp.yaml")
    parser.add_argument("--task_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="submissions/my_model")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg, args.task_dir, args.out_dir)
