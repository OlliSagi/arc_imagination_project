# ARC-AGI-2 Imagination + Reasoning Plan

## What this project does (short)
- Always-on Sim2D: a tiny transformer that maintains a 32×32 latent field and accepts differentiable brush edits.
- Reasoner: a small planner (placeholder) now; later a 7B open LLM (4-bit + LoRA) that reads a summary of Sim2D and emits edit tokens.
- ARC coupling: load ARC-AGI-2 tasks, train Sim2D + planner to transform input grids into outputs, evaluate with two attempts per test input and write submissions compatible with the official scorer.

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

## Closed-loop training track (to reduce overfitting and encourage "imagination")
- Search-over-edits inside attempts: during training and eval, roll out R candidate edit sequences on the Canvas and pick the sequence that best reduces grid error on the train pairs (ARC-AGI-1/2 both provide train examples inside each task).
- Stepwise improvement objective: add a small reward if an edit step decreases grid loss vs. the previous step, encouraging iterative refinement rather than one-shot mapping.
- In-episode adaptation (few-shot): within each task episode, take 1–5 gradient steps (LoRA if using an LLM; otherwise planner+Sim2D) on the train pairs, then predict for the test input.
- Round-trip grounding (when LLM is enabled): add State→Text→State and Text→State→Text consistency losses so explanations and edits stay aligned.

## LLM integration (optional once Canvas+planner is stable)
- Model choice: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` or `Qwen/Qwen2.5-7B-Instruct`.
- Download: no manual download required—`transformers` will fetch on first use. GPU is recommended (you have one). 4-bit quantization + LoRA keeps VRAM low.
- Enable in config: set `llm.model_name` and keep `load_in_4bit: true`. Then replace the tiny `Planner` in code with an `EditHead` fed by the LLM’s hidden state (the hooks are in `llm_wrapper.py` and `edit_head.py`).
- Fine-tuning: train LoRA adapters jointly with Sim2D/Readout using the same grid-loss signal and the closed-loop objectives above. For ARC-AGI‑style tasks, outputs remain grids; the LLM’s role is to plan/edit, not to output text.
- Scaling: if a 7B is insufficient, the same setup works with larger open models given more VRAM; the Canvas and training loop remain unchanged.

## Notes & current limitations
- The evaluator currently emits fixed 32×32 predictions; a production solver should infer exact grid sizes before writing outputs.
- If `ARC_TRAIN_DIR` is not set, `train_arc.py` falls back to a simple synthetic rectangle task for smoke testing.
