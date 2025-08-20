# Stage 0 — Canvas Grounding Plan & Validation

## Objective
Ground a 32x32 latent Canvas (Sim2D) that can apply small local edits stably and decode to valid ARC grids. No LLM, no planner, no labels required.

## Deliverables
- checkpoints/canvas_stage0.safetensors
- logs/metrics.jsonl (per-step metrics)
- samples/step_XXXX.png (decoded snapshots for quick visual sanity)

## Data (unlabeled)
- ARC grids: pool all `train.input` and `train.output` from ARC-AGI-1/2 as independent grids (both sides are valid world states).
- Augmentations: flips (X/Y), rotations (90/180/270), optional small translations within 32x32.
- Target size: 5k–20k unique grids (augmented to 20k–60k). Hold out 10% for validation.

## Data/steps sizing (when is it “enough”?)
- Unique grids: start with 5k–20k; augment to ~20k–60k via flip/rot90; keep 10% val.
- Steps: 3k–5k total; extend in +1k steps only if a specific threshold isn’t met.
- Add more data if: recon_ce_val > recon_ce_train by >10% after ~3k steps, or cycle_ce stuck > 0.10.
- Add more capacity if: recon_ce stalls (>1.2) and max_logit < 0.6 despite more data.
- Add stronger aug if: leakage OK but recon_ce_val lags (use flips/rot90/translate within bounds).

## Losses (what we optimize)
- recon_ce: cross-entropy between decoded grid (10-class per cell) and source grid.
- next_step_mse: MSE between predicted next latent and target next latent after a small random edit.
- cycle_ce: CE after mirrorx2 and rotatex4 (should reconstruct original grid).
- edit_budget: penalty on number/size/magnitude of random edits (encourage locality/sparsity).

## Metrics (what we log every N steps)
- recon_ce
- next_step_mse
- cycle_ce (mirrorx2, rotatex4)
- mask_leakage: fraction of pixels changed outside the edit mask
- max_logit: average max softmax over 10 colors (decode confidence)
- recon_ce_val: validation split cross-entropy

## Training procedure (steps we will follow)
1) Isolate implementation in `proto_canvas/` (no changes to existing `src/`).
   - Files to create: `canvas.py`, `edits.py`, `data.py`, `trainer.py`, `viz.py`, `config.yaml`, `main.py`, `README.md`.
   - `config.yaml` uses numeric types (no string LRs) and sets H=W=32, C>=32, dim≈256, layers≈4–6.
2) Warmup run: 1,000 steps with mixed losses; verify quick downward trend on recon_ce and next_step_mse.
3) Extend to 3,000–5,000 steps.
   - Stop early if all core metrics plateau (<1% change over last 200 steps) and thresholds are met.
4) Save checkpoints and sample frames every 500 steps. Keep metrics.jsonl for plots.

## Pass/Fail thresholds (tussentijdse test)
- recon_ce: >=50% drop from step 0 and absolute < 0.8 (10-class chance ≈ 2.30)
- next_step_mse: >=30% drop from step 0 and plateau (<1% change over last 200 steps)
- cycle_ce: < 0.05 for mirrorx2 and rotatex4
- mask_leakage: < 1% of cells changed outside mask on a standard random-edit suite
- max_logit: > 0.75 (decoded colors are confident)
- gen gap: recon_ce_val within 10% of recon_ce (train)

## Robustness checks
- Seeds: run 3 short seeds (1,000 steps). Variance on recon_ce and cycle_ce < 5%.
- Resolution: quick 24x24 trial (500 steps); curves should mirror 32x32 trends.

## Decision rules
- All thresholds met -> Stage 0 accepted; proceed to Stage 1.
- recon_ce OK, cycle_ce high -> increase model capacity or cycle loss weight; extend +1,000 steps.
- mask_leakage high -> reduce brush radius range and/or increase leakage penalty; extend +1,000 steps.
- gen gap high -> add augmentations or more grids; extend +1,000 steps.

## Compute budget (Stage 0)
- Model: 32x32 Canvas, C≈96, dim≈256, layers 4–6, heads 8; batch size 2–8.
- VRAM: ~4–8 GB typical; 32 GB VRAM is more than enough. CPU fallback works but ~3–5× slower.
- Throughput: ~1k–2k steps/hour on a single mid‑range GPU; 3k–5k steps ≈ 1–3 hours.
- Precision: bf16/fp16 recommended; fp32 is fine if bf16 not supported (slower, higher VRAM).
- Disk: dataset cache ~50–300 MB; metrics/samples negligible.

## Outputs to inspect
- Quantitative: metrics.jsonl plots for recon_ce, next_step_mse, cycle_ce (train/val), mask_leakage, max_logit.
- Qualitative: samples/step_XXXX.png — shapes should remain crisp; random edits should remain local; mirrorx2/rotatex4 should reconstruct.

## What happens after Stage 0 passes
- Stage 1 will introduce typed slots (holes, size, adjacency, symmetry, repetition) and a small synthetic event probe to shape slot usage before tackling ARC episodes.

---
Note: This plan is implementation-agnostic. We will place the code under `proto_canvas/` to avoid touching existing code until Stage 0 is validated.
