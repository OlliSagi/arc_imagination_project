# Stage 1 — Slots + Motifs Shaping Plan & Validation

## Objective
Learn typed “meaning” slots and reusable edit motifs on top of the grounded Canvas, so an episode can represent a generalized event (e.g., mirror, translate, recolor-by-feature) consistently across its train pairs.

## Deliverables
- checkpoints/sim2d_stage1.safetensors
- checkpoints/readout_stage1.safetensors (slot heads)
- checkpoints/motifs_stage1.safetensors (optional VQ codebook)
- logs/metrics_stage1.jsonl (slot accuracies, event accuracy, codebook stats, plus Stage 0 metrics)

## Data
- Synthetic episodes (light labels): 2k–5k episodes across 5–8 families
  - initial: recolor_by_feature, mirror, translate (already scaffolded)
  - next: rotate, copy/move, tiling/stripes, symmetry
- Keep ARC grids (unlabeled) mixed in to retain Stage 0 stability

## Losses (additions on top of Stage 0)
- event_type_ce: CE over event families
- mirror_axis_ce: CE over {x,y} when event=mirror
- translate_ce: CE over dy and dx bins (e.g., −3..3) when event=translate
- feature_selector_ce: CE over {holes, area} when event=recolor_by_feature (start with holes)
- (optional) VQ motif losses: codebook + commitment; motif usage entropy regularizer
- Keep Stage 0: recon_ce, next_step_mse, cycle_ce, leakage

## Metrics (per N steps)
- event_type_acc (overall and per-family)
- axis_acc (mirror), translate_acc (dy/dx), feature_acc (recolor)
- codebook: utilization %, entropy (optional)
- Stage 0 carry-overs: recon_ce, next_step_mse, cycle_ce, leakage, max_logit, recon_ce_val

## Procedure
1) Isolate implementation in `proto_stage1/` (no changes to `src/` or Stage 0); reuse Stage 0 Canvas.
2) Load synthetic episode JSONs from a directory; form batches of (input grid, labels).
3) Train for 3k–5k steps mixing:
   - Stage 0 losses on the input grid (stability)
   - Slot/head supervision from episode labels (meaning)
   - Optional: VQ motif losses (disabled by default)
4) Log metrics; save checkpoints every 500–1000 steps.

## Pass/Fail thresholds (tussentijdse tests)
- event_type_acc ≥ 0.80 on synthetic validation
- mirror axis_acc ≥ 0.90; translate_acc (dy/dx) ≥ 0.70 when present
- feature_acc (recolor) ≥ 0.90 (holes)
- codebook (if enabled): utilization ≥ 30%, entropy above target (no collapse)
- Stage 0 gates remain green (see Stage 0 plan)

## Data/steps sizing
- Episodes: 2k–5k total; ≥300 per family to avoid overfitting
- Steps: 3k–5k; extend +1k only if a specific metric lags
- If per-family accuracy < target: add 300–500 more episodes for that family

## Compute budget
- Similar to Stage 0 (32x32, small heads): 1–3 hours for 3k–5k steps on a single GPU; 32 GB VRAM is ample

## Outputs to inspect
- metrics_stage1.jsonl plots for event_type_acc, axis/translate/feature acc, codebook stats, and Stage 0 metrics
- Qualitative: quick storyboard from a few synthetic episodes (optional) and decoded samples

## What happens after Stage 1 passes
- Implement the ARC episode loop with Layer‑1 slots and R‑way search-over-stories; validate on a small ARC dev slice before enabling the LLM planner.

---
Note: Stage 1 relies on synthetic labels only; ARC remains unlabeled for meaning. Episode invariance will be enforced in the ARC loop (Stage 2).
