# 2025-08-22 — Progress Log (Stage 0 pass, Stage 1 baseline + generalized interpreter)

## Summary
- Stage 0 re-run (10k steps) met acceptance gates. Checkpoint frozen for Stage 1/2.
- Stage 1 baseline (3 ops: recolor-by-feature, mirror, translate) trained and logged; pipeline verified end-to-end on GPU.
- Upgraded Stage 1 to a generalized interpreter (predicate bank + z_story) and added output reconstruction loss.
- Updated spec with Concept Storyboard mode and WWII → Gen Z worked example to reflect long-term, domain-agnostic “2D storyteller” goal.

## Details

### Stage 0 — Canvas grounding (Sim2D)
- Action: Increased steps to 10k.
- Final metrics (approx.):
  - recon_ce ≈ 0.334 (< 0.8 gate)
  - cycle_ce ≈ 0.0 (< 0.05 gate)
  - max_logit ≈ 0.768 (> 0.75 gate)
  - next_step_mse ≈ 1.9e-5 (good)
- Decision: PASS. Freeze `proto_canvas/checkpoints/canvas_stage0.safetensors` for downstream.

### Stage 1 — Baseline “what” (3 ops) — verification run
- Fixes/infra:
  - `proto_stage1/main.py`: import from `src.utils`; load Stage 0 with `torch.load(..., map_location='cpu')` (file saved via torch.save, extension `.safetensors`). Training runs on GPU; only load was CPU-side.
  - `proto_stage1/data.py`: loader now supports `train_pairs` or `train` schemas.
- Data:
  - Generated `data/synthetic` (400 eps) and `data/synthetic2` (800 eps) with families: recolor_by_holes, mirror_x, translate (20×20 grids).
- Runs:
  - 1.2k-step and 5k-step baselines (heads-only, Sim2D frozen).
- Observed metrics (batch-level ranges):
  - event_acc ≈ 0.31–0.69 (often identifies which op)
  - axis_acc ≈ 1.0 (frequently N/A, ignored via mask)
  - dy/dx: inconsistent (data coverage + simplicity limit)
  - feature_acc ≈ 1.0 (often N/A)
- Interpretation: Smoke test passed — training/IO/checkpointing/logging behave; heads learn nontrivial “what,” translation parameters remain unstable without targeted data.

### Stage 1 — Generalized interpreter (predicate bank + z_story)
- Code additions:
  - `proto_stage1/interpreter.py`: `GeneralInterpreter` composed of `PredicateBank` (K=16), `StoryEncoder` (z_dim=32), and `StoryDecoder` (diagnostic heads only).
  - `proto_stage1/main.py`: switched heads → generalized interpreter; added predicate usage regularizer; added output reconstruction loss (temporary direct readout when `decode_logits` isn’t used in this loop).
  - `proto_stage1/config.yaml`: added `z_dim: 32`, `num_predicates: 16`.
- Run:
  - 2k steps on GPU; logged `loss_recon` along with diagnostic accuracies.
- Observed metrics (examples):
  - loss_recon ≈ 0.78–1.21 (learning signal present)
  - event_acc fluctuates (0.25–0.69); dy/dx still unstable (expected before adding an executor/probes)
- Interpretation: Model now learns from outputs, not only labels; to realize full gains, couple z_story/predicates to a simple executor and probe losses.

### Spec update
- `docs/MENTAL_ANIMATION_SPEC.md`:
  - Added “Concept Storyboard mode” (nodes/edges/timeline/flows) to unify the ARC pixel loop with abstract reasoning stories.
  - Added a worked example: WWII → Gen Z storyboard with probes and scoring.

## Next steps (actionable)
1) Stage 1 training signal
- Add a simple executor that applies predicted ops before reconstruction:
  - mirror (x/y), translate (dy,dx), recolor-by-soft predicate map (bins/colors)
- Add probe losses (invert/inverse checks):
  - mirror twice → identity; translate +(dy,dx) then −(dy,dx) → identity; recolor toggles under feature perturbation
- Rebalance data for translation (≥50% translate) and include more ops (mirror_y, rotate90/180; parity/stripe/checker; area/aspect; dilate/erode).

2) Training runs
- Generalized Stage 1, 5–10k steps, cosine LR on heads; Sim2D frozen.
- Metrics to watch: loss_recon downtrend; event_acc up; dy/dx stabilize; predicate_gates sparsity (selection emerges).

3) Stage 2 scaffold (begin in parallel once executor lands)
- Implement R-way search-over-stories + probe loop + composite scoring (fit + why + stability + simplicity).
- Log storyboard snapshots and episode reports.

## Commands (Windows cmd)
- Generate more synthetic data:
  - `set PYTHONPATH=.&& .\venv\Scripts\python.exe -m src.synth.generate --out_dir data\synthetic3 --num 1000 --events recolor_by_holes,mirror_x,translate --height 20 --width 20 --seed 2`
- Run generalized Stage 1 (current):
  - `set PYTHONPATH=.&& .\venv\Scripts\python.exe proto_stage1\main.py --config proto_stage1\config.yaml --steps 5000`
- Tail metrics (PowerShell):
  - `Get-Content proto_stage1\logs\metrics_stage1.jsonl -Tail 20 -Wait`

## Notes
- GPU verified: NVIDIA GeForce RTX 5090; training runs on CUDA. Stage 0 checkpoint load uses CPU map_location only for reading.
- Stage 0 file was saved via torch.save despite `.safetensors` suffix; loader adjusted accordingly.

