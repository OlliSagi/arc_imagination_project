# GEUMA : Generalized Event Understanding via Mental Animation

This repo is a **practical scaffold** to test the "always-on imagination + text reasoning" idea directly on **ARC-AGI-2**.

- **Imagination (Sim2D):** a small always-on 2D latent field (32×32×C) updated by a transformer.
- **Reasoner (LLM):** reads a compact summary of Sim2D before emitting tokens, *and* emits **edit tokens** (soft brushes) that change Sim2D.
- **ARC alignment:** loader for ARC-AGI-2 JSON tasks, trainer that learns lawful edits on synthetic blob tasks + ARC training tasks, and an evaluator that writes submissions compatible with the `arc-agi-benchmarking` repo.
- **Runs on one 32GB GPU** with a 7B model in 4‑bit + tiny Sim2D. We recommend **DeepSeek‑R1‑Distill‑Qwen‑7B** or **Qwen2.5‑7B‑Instruct** as initial open LLMs.

> This is not a finished solver; it’s a clean and *runnable* starting point that keeps your imagination loop alive and shows how to couple it to ARC tasks.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 1) See the always-on Sim2D tick (no LLM yet; uses dummy edits)
python run_loop.py --config configs/arc_mvp.yaml

# 2) Train Sim2D boot (next-state pretrain + synthetic edit curriculum)
python src/train_boot.py --config configs/arc_mvp.yaml

# 3) (Optional) Wire an LLM (DeepSeek-R1-Distill-Qwen-7B / Qwen2.5-7B) and try a few ARC train tasks
python src/train_arc.py --config configs/arc_mvp.yaml

# 4) Evaluate on a task subset and dump predictions in benchmark format
python src/eval_arc.py --config configs/arc_mvp.yaml --task_dir /path/to/ARC-AGI-2/data/evaluation --out_dir submissions/my_model
```

To score against ARC-AGI-2 public eval with the official harness, use the `arc-agi-benchmarking` repo and point it at `submissions/my_model`.

## Repo Layout

```
configs/
  arc_mvp.yaml          # hyperparams & model choice (7B LLM, 4-bit, LoRA)
src/
  sim2d.py              # always-on 2D imagination + soft brush edits
  arc_data.py           # ARC JSON loader & utilities
  edit_head.py          # LLM hidden -> K edit tokens
  readout.py            # learned queries -> conditioning vector
  llm_wrapper.py        # HF loading (Qwen/DeepSeek) + PEFT LoRA
  train_boot.py         # next-state + synthetic edit curriculum (boot)
  train_arc.py          # ARC-coupled training (round-trip + grid loss)
  eval_arc.py           # run tasks, respect attempt budget, write submissions
  utils.py              # safetensors, config, logging, small helpers
run_loop.py             # always-on demo loop
requirements.txt
README.md
```

## What to commit (GitHub)
- Commit: `proto_canvas/`, `proto_stage1/`, `src/`, `configs/`, `docs/`, `requirements.txt`, `run_loop.py`, and the ARC JSONs if the repo stays < ~1 GB. If it gets too large, upload ARC data directly to your GPU pod instead.
- Don’t commit: `checkpoints/`, `logs/`, `samples/`, `submissions/`, `.venv/`, `__pycache__/`, `*.safetensors`.

Suggested `.gitignore`:
```
.venv/
__pycache__/
checkpoints/
logs/
samples/
submissions/
data/synthetic*/
*.safetensors
*.pth
*.pt
```

## Runpod quickstart (Linux)
```
git clone https://github.com/<your-user>/<your-repo>.git
cd <your-repo>
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Verify GPU
python - << 'PY'
import torch; print('cuda?', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
PY

# Stage 0 later
python proto_canvas/main.py --config proto_canvas/config.yaml --steps 3000
# Stage 1 later
python proto_stage1/main.py --config proto_stage1/config.yaml --steps 3000
```

## Notes
- The imagination field is **not pixels**. It’s a compact feature map that persists and is *controlled* by the LLM via differentiable brushes.
- For ARC, grids are ≤ 30×30; we pad to 32×32 internally. Colors are integers 0..9; we encode one-hot + extra channels.
- **Two attempts per test input** are supported in `eval_arc.py` (configurable).
