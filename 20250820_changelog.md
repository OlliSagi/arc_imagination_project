# 2025-08-20 — Changelog and Current Status

## What we added today
- Documentation/specs
  - `docs/MENTAL_ANIMATION_SPEC.md`: mental animation → generalized event understanding (Canvas, stories, slots, search).
  - `docs/STAGE_0_PLAN.md`: Stage 0 goals, metrics, thresholds, and decision rules.
  - `docs/STAGE_1_PLAN.md`: Stage 1 goals (slots/motifs), metrics, thresholds.
  - `docs/index.html` + `docs/globals.css`: architecture overview page.
- Stage 0 scaffold (no external model needed)
  - `proto_canvas/` (new):
    - `config.yaml`, `README.md`, `main.py`
    - `canvas.py` (small conv Sim2D + decode head), `edits.py` (soft brush masks),
      `data.py` (pool ARC grids + augmentation), `trainer.py` (recon/next-step/cycle), `viz.py` (PNG frames).
  - Runs with unlabeled ARC grids; logs recon_ce, next_step_mse, cycle_ce, max_logit.
- Stage 1 scaffold (slots + motifs shaping)
  - `proto_stage1/` (new):
    - `config.yaml`, `README.md`, `main.py` (slot/event heads trainer)
    - `heads.py` (event, axis, dy/dx, feature heads), `data.py` (load synthetic episodes).
  - Trains typed heads on synthetic episodes; keeps Stage 0 losses active.
- Synthetic data generator (for Stage 1)
  - `src/synth/` (new): `shapes.py`, `events.py`, `generate.py`.
  - Event families: recolor_by_holes, mirror_x, translate (fixed translate bounds).
  - Shapes: rectangles and disks with optional holes.
  - Example:
    - `python -m src.synth.generate --out_dir data/synthetic --num 300 --height 20 --width 20 --seed 0 --events recolor_by_holes,mirror_x,translate`
- Utilities
  - `scripts/run_on_runpod.sh` (new launcher; adjust for preinstalled torch and set `PYTHONPATH=.` before Stage 0).
  - `src/arc_event_miner.py` (placeholder miner for simple events on ARC tasks).
- Repo hygiene
  - `requirements.txt`: added Pillow for image saving.
  - `.gitignore`: ignore venv, checkpoints, logs, samples, submissions, synthetic data, and weight files.
  - `README.md`: added “What to commit” and Runpod quickstart.

## Small code fixes
- `src/train_boot.py`, `src/train_arc.py`: cast LR to float to avoid string LR issues.
- `docs/index.html`: clarified components and loop.

## How to run (Linux pod)
- Stage 0 (Canvas grounding):
  - `python3 -m venv .venv && source .venv/bin/activate`
  - `pip install --upgrade pip && pip install -r requirements.txt`
  - `export PYTHONPATH=.`
  - `python proto_canvas/main.py --config proto_canvas/config.yaml --steps 3000`
- Stage 1 (after Stage 0):
  - Generate synthetic episodes (see above) if not present.
  - `export PYTHONPATH=.`
  - `python proto_stage1/main.py --config proto_stage1/config.yaml --steps 3000`

## Current status
- Stage 0/1 scaffolds are in place and runnable with only the ARC JSONs and Python deps.
- Synthetic generator produces recolor/mirror/translate episodes for Stage 1.
- Launcher script added (useful on clean pods; inside-repo usage may remove the extra clone and torch install lines).

## Next steps
- Run Stage 0 to validate metrics; adjust steps/data per thresholds if needed.
- Run Stage 1 to validate slot/event accuracies; add rotate, copy/move, tiling/stripes.
- Implement ARC episode loop with Layer‑1 slots and R‑way search-over-stories; produce dev submissions.
- Optional later: LLM planner (LoRA) after the above is stable.

## Data paths (relative)
- ARC‑AGI‑1: `ARC-AGI/data/training`, `ARC-AGI/data/evaluation`
- ARC‑AGI‑2: `ARC-AGI-2/data/training`, `ARC-AGI-2/data/evaluation`
- Synthetic: `data/synthetic`, `data/synthetic2`
