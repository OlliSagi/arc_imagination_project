# ARC-AGI-2 Imagination + Reasoning Plan

## What this project does (short)
- Always-on Sim2D: a small 32×32 latent Canvas with differentiable brush edits and a decode head to 10 colors.
- Reasoner: a planner (MLP now, optional 7B LLM later) that proposes probes/edits and small programs over typed slots.
- Episode solver (Stage 2): self‑play probes + belief update + explanation scoring (fit + why‑probes + stability + simplicity), then apply the chosen program to the test.
- ARC coupling: load ARC tasks, solve each episode via the above loop, and write JSON grids compatible with the scorer.

## Goal
Train Sim2D and the planner on the ARC training set you attached (`ARC-AGI-2/data/training`), then evaluate on the public evaluation set to produce a submission folder. Optionally swap the planner for a 7B LLM via LoRA after the Sim2D pipeline is working.

## System requirements
- Python 3.10+
- NVIDIA GPU recommended; 32 GB VRAM if you later enable a 7B LLM in 4-bit. CPU will run but slower.

## Data locations (your setup)
- ARC-AGI-1 (classic) training: `C:\Users\oy\Documents\arc_imagination_project\ARC-AGI\data\training`
- ARC-AGI-1 evaluation: `C:\Users\oy\Documents\arc_imagination_project\ARC-AGI\data\evaluation`
- ARC-AGI-2 training: `C:\Users\oy\Documents\arc_imagination_project\ARC-AGI-2\data\training`
- ARC-AGI-2 evaluation: `C:\Users\oy\Documents\arc_imagination_project\ARC-AGI-2\data\evaluation`

## Steps (Windows cmd)
1) Create venv and install deps
```
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2) Smoke test the always-on Sim2D loop (no LLM, dummy edits)
```
python run_loop.py --config configs\arc_mvp.yaml
```

3) Boot-train Sim2D dynamics (keeps the field coherent)
- Output: `checkpoints\sim2d_boot.safetensors`
```
python src\train_boot.py --config configs\arc_mvp.yaml
```

4) Train on ARC tasks (Sim2D + Readout + tiny Planner)
- Start with ARC-AGI-1 to get more training data:
```
set ARC_TRAIN_DIR=C:\Users\oy\Documents\arc_imagination_project\ARC-AGI\data\training
python src\train_arc.py --config configs\arc_mvp.yaml
```
- Then switch to ARC-AGI-2 by changing the path:
```
set ARC_TRAIN_DIR=C:\Users\oy\Documents\arc_imagination_project\ARC-AGI-2\data\training
python src\train_arc.py --config configs\arc_mvp.yaml
```
- Outputs after training: `checkpoints\sim2d_arc.safetensors`, `checkpoints\readout_arc.safetensors`, `checkpoints\planner_arc.safetensors`

5) Evaluate and write submissions (JSON grids, not text)
- ARC-AGI-1 uses 3 trials per test input. Temporarily set `eval.attempts_per_test_input: 3` in `configs\arc_mvp.yaml` before running:
```
python src\eval_arc.py --config configs\arc_mvp.yaml --task_dir C:\Users\oy\Documents\arc_imagination_project\ARC-AGI\data\evaluation --out_dir submissions\arc1_model
```
- ARC-AGI-2 uses 2 attempts (default in config):
```
python src\eval_arc.py --config configs\arc_mvp.yaml --task_dir C:\Users\oy\Documents\arc_imagination_project\ARC-AGI-2\data\evaluation --out_dir submissions\arc2_model
```

6) Score with the official harness (external repo)
- Use the `arc-agi-benchmarking` scorer against your `submissions\...` folder.

## How it works (plain language)
- Sim2D (we’ll refer to this as the "Canvas"): a persistent 32×32 latent field (not pixels) that updates every step with a small transformer; it’s the agent’s scratchpad/working memory.
- Brush edits: each edit token is like an airbrush stroke parameterized as `[center_x, center_y, radius, scale, delta_vector]`. It paints a smooth mask and adds `delta_vector` to all channels under the mask.
- Planner: looks at a compact summary of the Canvas (`readout.py`) and emits K brush edits each step. We start with a tiny MLP planner; later, we let a 7B LLM emit those edits for more human-like multi-step reasoning.
- Output: predictions are grids (list of lists of ints 0–9). The evaluator writes per-task JSONs for the official scorer.

## Closed-loop training track (sequence-first, meaning-first)
- Self‑play probes: per episode, the agent proposes small diagnostic edits (toggle hole, axis flip, undo‑shift, stripe period shift) to test invariances/counterfactuals.
- Belief update: maintain a belief over hypotheses/programs; update from probe outcomes and grid loss on all train pairs.
- Explanation scoring: Score = Fit (avg train loss) + (1−Why) (probe pass) + Stability (slot variance) + Simplicity (program length/edit budget).
- In‑episode adaptation: 1–5 tiny updates (LoRA if LLM; otherwise planner/slots) to refine the explanation within the episode; reset afterward.
- Attempts: multiple seeds/attempts; keep the explanation with best composite score; apply once to test.

## LLM integration (optional once Canvas+planner is stable)
- Model choice: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` or `Qwen/Qwen2.5-7B-Instruct`.
- Download: `transformers` fetches on first use; use 4‑bit + LoRA for VRAM efficiency.
- Enable in config: set `llm.model_name` (keep `load_in_4bit: true`). Use `edit_head.py` to turn LLM hidden states into probe/program tokens.
- Fine-tuning: LoRA‑tune on ARC episodes with the Stage 2 score (fit + why‑probes + stability + simplicity). The LLM proposes probes/programs; the Canvas and episode score keep it grounded.
- Scaling: larger models plug into the same loop; the Canvas and training remain unchanged.

## Notes & current limitations
- The evaluator currently emits fixed 32×32 predictions; a production solver should infer exact grid sizes before writing outputs.
- If `ARC_TRAIN_DIR` is not set, `train_arc.py` falls back to a simple synthetic rectangle task for smoke testing.
