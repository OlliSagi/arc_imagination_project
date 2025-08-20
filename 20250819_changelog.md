# 2025-08-19 — Changelog and Next Steps

## Decisions
- Start set: Train/evaluate first on ARC‑AGI‑1 (more tasks, 3 trials), then ARC‑AGI‑2 (2 attempts). Same JSON/grid format.
- Outputs: Always JSON grids (list of lists of 0–9), not text. Offline generation; submit files only.
- Imagination engine: Keep the 32×32×C latent board with iterative “strokes” (local soft edits). This is our core differentiator.
- Baselines to compare:
  - A: Imagination OFF (no strokes).
  - B: Imagination ON + small MLP planner.
  - C: Imagination ON + Qwen2.5‑7B planner (train with LoRA for practicality, then MERGE to a single native checkpoint). If GPUs allow, full FT is optional.
- LoRA policy: Train with LoRA first to validate; merge adapters → single native checkpoint with equivalent outputs (up to tiny FP drift). Rename under our brand.
- Naming: Call the latent board the Imagination engine.

## Imagination engine (how it works)
- Board: 32×32×C latent state. Channels 0–9 = color logits; remaining channels = working memory.
- Stroke (brush edit): one localized soft update parameterized by (x, y, radius, scale, delta_vector[C]). Effect: S ← S + mask(x,y,r) × (scale × delta).
- K strokes per step: a few parallel local edits each step (default K=3). Several steps compose a “story” of transformations.
- Decode: After N steps, take channels 0–9, argmax per cell → predicted grid.
- Closed loop: Per attempt, try multiple stroke sequences; score against train pairs; pick the best.

## Training and Evaluation (Windows cmd)
- Env & deps
```
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
- Smoke test (untrained loop)
```
python run_loop.py --config configs\arc_mvp.yaml
```
- Boot the imagination engine
```
python src\train_boot.py --config configs\arc_mvp.yaml
```
- Train on ARC‑AGI‑1
```
set ARC_TRAIN_DIR=C:\Users\oy\Documents\arc_imagination_project\ARC-AGI\data\training
python src\train_arc.py --config configs\arc_mvp.yaml
```
- Evaluate on ARC‑AGI‑1 (set attempts=3 in config before running)
```
python src\eval_arc.py --config configs\arc_mvp.yaml --task_dir C:\Users\oy\Documents\arc_imagination_project\ARC-AGI\data\evaluation --out_dir submissions\arc1_model
```
- Switch to ARC‑AGI‑2
```
set ARC_TRAIN_DIR=C:\Users\oy\Documents\arc_imagination_project\ARC-AGI-2\data\training
python src\train_arc.py --config configs\arc_mvp.yaml
python src\eval_arc.py --config configs\arc_mvp.yaml --task_dir C:\Users\oy\Documents\arc_imagination_project\ARC-AGI-2\data\evaluation --out_dir submissions\arc2_model
```

## Planned A/B/C Experiments
- A (No imagination): Disable strokes (edits) during train and eval; keep everything else identical.
- B (Imagination + MLP): Enable strokes; use small MLP planner; same budget.
- C (Imagination + Qwen2.5): Replace MLP planner with Qwen2.5‑7B + EditHead; train with LoRA; MERGE to single checkpoint; re‑eval to confirm parity.

## LLM Planner (when moving beyond MLP)
- Config: Set llm.model_name (e.g., Qwen/Qwen2.5-7B-Instruct), keep load_in_4bit: true.
- Training: Jointly train LoRA + Imagination engine with grid loss and closed‑loop search.
- Merge: merge_and_unload() adapters into base; save under our name (single native checkpoint).

## To‑Do (engineering)
- Toggle: Add use_edits: true|false to disable/enable strokes in both train and eval.
- Planner switch: Add planner: mlp|llm to select between tiny MLP and Qwen2.5 + EditHead.
- Search‑over‑edits: Implement R‑way candidate rollout per attempt (eval first; optionally in training with stepwise improvement bonus).
- ARC‑AGI‑1 attempts: Expose attempts override (3 for ARC‑AGI‑1, 2 for ARC‑AGI‑2) via CLI or config.
- Export: Script to MERGE LoRA and save a single checkpoint; offline inference script to write submission JSONs.
- (Optional) Qwen direct baseline: Add a simple Direct GridHead (no imagination) for a Qwen‑centric X vs Y comparison.

## Data Paths
- ARC‑AGI‑1 train: `C:\Users\oy\Documents\arc_imagination_project\ARC-AGI\data\training`
- ARC‑AGI‑1 eval: `C:\Users\oy\Documents\arc_imagination_project\ARC-AGI\data\evaluation`
- ARC‑AGI‑2 train: `C:\Users\oy\Documents\arc_imagination_project\ARC-AGI-2\data\training`
- ARC‑AGI‑2 eval: `C:\Users\oy\Documents\arc_imagination_project\ARC-AGI-2\data\evaluation`

## Notes
- 32×32 covers ARC’s ≤30×30; we can scale later if needed.
- No internet/tools needed for scoring; predictions are JSON grids.
