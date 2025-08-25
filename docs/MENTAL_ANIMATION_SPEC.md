# Mental Animation for Event Understanding (ARC)

## Intent (human-like generalization)
- Goal: generalized event understanding, not brittle per-task rules. Each episode (ARC task) induces an event latent that explains all train pairs and transfers to the test input.
- Loop: hypothesis -> simulate (a mental animation) -> maintain an event latent -> verify on train pairs -> apply to test.

## System overview
- Canvas (Sim2D): 32x32xC latent field, persistent within an episode.
  - Channels 0–9: color logits for ARC colors 0–9
  - Working memory channels: scene state
  - Episodic channels: typed "meaning" slots (event latent + attributes)
- Brush Edit (stroke): local, soft update parameterized by [x, y, radius, scale, delta_vector[C]].
- Planner: proposes K edits per step and optional updates to meaning slots.
  - mlp (fast baseline) or llm (Qwen2.5 / DeepSeek-R1-Distill-Qwen) emitting edit tokens, not text.
- Readout: summarizes Canvas + meaning slots for the planner; decodes final Canvas -> grid (argmax on channels 0–9).
- Closed-loop search: roll out R candidate stories (hypotheses), pick the one minimizing train-pair grid loss.

## What a step is (not just coloring)
A step is a small program over the latent scene and the event state:
- Perception: produce soft object masks and relations from the Canvas.
- Operations: transform/transport masks (mirror, rotate, translate, copy, paint, flood, reflect, tile) with differentiable masks.
- Meaning update: write to typed slots (e.g., color remap, symmetry, repetition period, object graph, counts/parity flags).
- Render: update Canvas logits + memory via Sim2D dynamics.

Recommended typed slot set (episodic channels ~ 8–16):
- color_map: 10x10 logits (sparse; encourage near-permutation)
- transform: logits over {id, mirrorX, mirrorY, rot90, rot180, rot270} + continuous translate (dx, dy)
- repetition: period logits (1–6) and phase
- objects[K]: small K (e.g., 4) slots with color, bbox, mask summary, intended move
- counts/flags: parity, adjacency, enclosure

## Stories and the event latent (z_story)
- Each candidate story builds an event latent z_story shared across all train pairs of the episode.
- Parameterization:
  - Discrete motifs: VQ codes from a learned motif codebook (edit primitives and short edit macros)
  - Continuous parameters: small vectors for magnitudes/positions/periods
  - Typed slots: the structured attributes listed above
- Invariance: for a given episode, the same z_story must explain all train pairs -> pushes generalization, not overfitting.
- Retrieval: store past z_story embeddings; initialize new episodes by nearest neighbors (reuse and faster convergence).

## Hypothesis testing (closed loop)
- Generate R candidate stories by sampling or planning.
- Simulate T steps (K edits per step), updating Canvas and meaning slots.
- Score by train-pair grid loss (plus auxiliaries below).
- Select best story -> apply its z_story/slots to the test input and render the output grid.

Pseudocode:
```python
for attempt in range(A):
    best = None
    for candidate in range(R):
        canvas = reset_canvas()
        slots = init_slots()
        z_story = init_event_latent(seed=candidate, retrieval=memory)
        for step in range(T):
            summary = readout(canvas, slots, z_story)
            edits, slot_delta, z_delta = planner(summary)
            canvas = apply_edits(canvas, edits)
            slots = update_slots(slots, slot_delta, canvas)
            z_story = update_event_latent(z_story, z_delta)
        score = train_pair_grid_loss(slots, z_story) + auxiliaries()
        if (best is None) or (score < best[0]):
            best = (score, slots, z_story)
    prediction = apply_story_to_test(best[1], best[2])
```

## Stage 2 — Self‑play probes, belief update, and explanation scoring
- Goal: move from "best fit" to "meaning via sequence": try → observe → update, then decide.

- Hypothesis DSL (minimal operators)
  - Geometry: MirrorX/Y, Rotate{90,180,270}, Translate(dy,dx)
  - Color: MapColors(feature∈{holes,area,aspect,compactness}, bins→colors)
  - Patterns: ExtendStripes(period, phase), Tile(period_x, period_y)
  - Relational: Copy/Move(selector∈{largest,enclosing,adjacent,majority}, to=target)
  - Control: If(cond on slots) Then(opA) Else(opB); Compose(op1→op2→…)

- Probe library (counterfactual checks)
  - Mirror probes: mirror twice → identity; mirror across other axis → mismatch
  - Rotate probes: rotate 4× → identity; wrong angle → mismatch
  - Translate probes: pre‑shift by −(dy,dx); undo/redo consistency
  - Feature recolor probes: toggle a hole / resize area; color must change as MapColors predicts
  - Pattern probes: shift by period; extend stripes consistency; wrong period fails
  - Relational probes: swap sizes (largest changes), break adjacency/enclosure; predicted selector should flip

- Sequence loop (per episode)
  1) Initialize belief b₀(h) over hypotheses and slots R₀
  2) Select next probe aₜ to maximize expected information gain between top hypotheses
  3) Apply probe (on copies), simulate edits for T steps, decode, compute probe losses
  4) Update belief and slots: bₜ₊₁(h) ∝ bₜ(h)·exp(−Loss_probe(h)); refine Rₜ→Rₜ₊₁ (small gradient/LoRA step)
  5) Repeat for K probes or until confidence/plateau criteria met
  6) Decide explanation h* and slots R*: highest posterior with stability and simplicity priors

- Scoring (what "best" means)
  - Fit: mean_train CE(ŷ, y)
  - Why: probe pass score (higher when expected invariances/counterfactuals hold)
  - Stability: low variance of slots (axis/period/color_map bins) across train pairs
  - Simplicity: MDL prior on program length, edit budget, mask leakage
  - Final: Score(h) = Fit(h) + λ₁·(1−Why(h)) + λ₂·Stability(h) + λ₃·Simplicity(h)

- Selection
  - h* = argmin Score(h); apply h* once to test inputs → JSON grids
  - Optional leave‑one‑out: fit on N−1 train pairs, check on held‑out; average to avoid one‑off hacks

- Episode report (for transparency)
  - Chosen program (DSL), slots (transform/period/color_map/selector)
  - Probe pass/fail table with brief reasoning
  - Per‑pair losses and stability stats

## Losses and regularizers
- Grid loss: cross-entropy on color logits vs. ground truth grid.
- Stepwise improvement bonus: reward negative deltas in grid loss across steps.
- Edit sparsity and locality: L1/L0-like budgets on stroke count, radius, and magnitude.
- Cycle/consistency: enforce invertibility where appropriate (e.g., mirror twice -> identity; translate forward/back).
- Slot priors:
  - color_map near-permutation (entropy regularization + doubly-stochastic sinkhorn prior)
  - transform one-hot-ish with small continuous offsets
  - repetition period low-entropy
- Motif (VQ) commitment: codebook and commitment losses for discrete motif codes.
- Contrastive retrieval: align z_story to nearby solved episodes; avoid collapse with InfoNCE.

## Training curriculum
- Stage 0: Sim2D bootstrapping
  - Train dynamics to keep the latent coherent under random small edits and to decode stable grids.
- Stage 1: Synthetic event library ("correct" data)
  - Generate tasks from a small program DSL with ground-truth event labels and edit programs.
  - Supervise typed slots and motif choices; mix in self-supervised objectives (next-frame prediction, inverse dynamics, cycle consistency).
- Stage 2: Meta-learning on synthetic episodes
  - Enable in-episode adaptation (LoRA on planner), search-over-stories, and retrieval init.
  - Optimize for fast inner-loop convergence and stable slot estimation.
- Stage 3: Transfer to ARC-AGI-1/2
  - Use ARC train pairs only (no labels). Keep all objectives that do not require labels; keep invariance (shared z_story across train pairs) and search-over-stories.
  - Optionally mix a small percentage of synthetic episodes to keep slot typing calibrated.
- Stage 4 (optional): LLM planner
  - Swap planner=llm (Qwen2.5-7B or DeepSeek-R1-Distill-Qwen), 4-bit + LoRA.
  - Train to emit edit motifs and slot deltas; grid loss + auxiliaries select good hypotheses. MERGE adapters for inference.
- Stage 5: Optional text alignment
  - Add lightweight State<->Text consistency to name events (e.g., "mirrorX then recolor red->blue"). Inference remains token-only.

## "Correct" data (meaningful synthetic episodes)
We do not require labels on ARC, but a compact synthetic suite helps shape generalizable slots and motifs.

Schema (per synthetic episode):
```json
{
  "episode_id": "mirrorx_recolor_small_0012",
  "train_pairs": [ {"input": [[...]], "output": [[...]]}, {"input": [[...]], "output": [[...]]} ],
  "test_inputs": [ [[...]], [[...]] ],
  "event": {
    "type": ["mirror_x", "recolor", "tile_periodic"],
    "motifs": [3, 17],
    "params": {"dx": 0, "dy": 0, "color_map": [[0,0],[1,2],...]},
    "objects": [ {"color": 2, "bbox": [x0,y0,x1,y1], "relation": "adjacent_to"} ]
  },
  "program": [
    {"op": "mirror", "axis": "x"},
    {"op": "map_colors", "map": {"1": 2}}
  ]
}
```

Generation principles:
- Factorized sampling: choose event types, then parameters and objects; compose 1–3 events per episode.
- Multi-example invariance: all train pairs share the same event description (type+params), input content varies.
- Difficulty ladder: start with single-event, extend to compositions.
- Class balance: colors, sizes, counts, and positions balanced to prevent shortcuts.
- Edge cases: empty sets, degenerate symmetries, overlapping objects.

## In-episode adaptation (smarter within an episode)
- LoRA inner-loop on the planner: 1–5 gradient steps on train pairs; reset after the episode.
- Latent refinement: allow small updates to z_story during the inner loop.
- Gradient-free complement: search-over-stories with R candidates; pick best by train loss.

## Across-episode learning (smarter over time)
- Meta-train the planner for rapid inner-loop gains.
- Learn a motif codebook (VQ) shared across episodes; reuse with the LLM or MLP.
- Retrieval memory: index z_story; initialize new episodes near similar solved ones.

## Implementation plan (hooks in current repo)
- Config (additions):
  - use_edits: true|false
  - planner: mlp|llm
  - search_over_stories: true|false, R: int, attempts: int, steps_per_attempt: int, edits_per_step: int
  - in_episode_adapt: true|false, adapt_modules: ["planner_lora"], adapt_steps, adapt_lr
  - episodic_memory_channels: int, slot_types: ["color_map","transform","repetition","objects","counts"]
- Code integration points:
  - sim2d.py: add episodic channels; ensure edit masks and dynamics support slot writes
  - readout.py: produce compact summary + slot readouts for planner; final decode
  - edit_head.py: planner heads for edits and slot deltas; optional VQ motif head
  - llm_wrapper.py: expose hidden states to EditHead when planner=llm
  - train_arc.py: implement episode loop, R-way search, in-episode adaptation
  - eval_arc.py: same loop without gradient updates; write JSON grids
- Logging & viz:
  - Storyboards: per-step Canvas snapshots + slot tracks
  - Scores: train-pair loss per candidate; chosen candidate index
  - Slots: distributions (e.g., color_map entropy), transform logits, period
  - Retrieval: nearest neighbors, hit rate

## Metrics
- Exact match on evaluation tasks (submission grids) and per-task success rate.
- Ablations: A (no imagination), B (MLP+search), C (LLM+LoRA+MERGE).
- Inner-loop delta: improvement after adaptation vs. before.
- Edit efficiency: steps to convergence, total edits used, average radius/magnitude.

## Risks and mitigations
- Overfitting to synthetic slots -> mix small synthetic ratio during ARC finetune; keep slot priors/regularizers.
- Planner collapse to painting-only -> enforce slot usage with losses and budgeted edits.
- Search cost too high -> prune with heuristic proposals, reuse retrieval inits, early stop by plateau.
- LLM hallucination -> grounded by grid loss/search; emit tokens only; no text outputs.

## Next steps (engineering)
- Add config toggles and slot scaffolding.
- Implement R-way search-over-stories and storyboard logging.
- Add VQ motif head and simple motif library.
- Implement in-episode LoRA adaptation for planner.
- Build a small synthetic event generator and data loader (schema above).
- Run A/B/C baselines, then enable slots + adaptation and compare.

## Worked example: holes → colors (episode walkthrough)
- Inputs: train pairs showing colored shapes with different numbers of holes; test input shows shapes without colors.
- Detect objects and features per train input: hole count (via flood-fill/Euler characteristic), size, bbox, adjacency.
- Propose R candidate stories with different meanings (e.g., color by hole_count; color by size; color by shape type).
- Simulate T steps per story:
  - Update slots: feature selector weights and a small color_map table (bins → colors).
  - Paint predicted outputs for each train input using soft masks under the current color_map.
- Score each story on all train pairs (grid loss + priors). Optionally take 1–2 tiny LoRA steps to tighten fit.
- Select best story; apply its slots (e.g., color_map keyed by hole_count) to the test input; decode to a JSON grid.

Why this generalizes: the same event latent/slots must explain multiple train pairs, pushing a compact rule like f(holes) → color instead of memorizing positions.

## Synthetic data via a 2D engine (pygame-like, optional)
Purpose: create "correct" event data to shape generalized slots and motifs for Stage 1 without hand-labeling.

What it does
- Renders ARC-like grids using a minimal 2D engine (pygame, numpy+PIL, or a custom rasterizer).
- Samples programmatic events (mirror, rotate, recolor-by-feature, translate, tile, copy/move, compose 1–3 ops).
- Emits both the rendered train/test grids and the event annotations matching the JSON schema in this spec (typed slots + program).

Why it helps
- Provides ground-truth event structure (meaning) and program steps to supervise slots and motif heads.
- Encourages reuse of event motifs and fast inner-loop adaptation before transferring to ARC-AGI-1/2.

Implementation sketch
- Renderer: draw shapes, compute holes (morphology/flood-fill), export H×W grids (0–9).
- Episode generator: sample event types and parameters, generate multiple train pairs per episode with the same event, plus held-out test inputs.
- Labels: fill `event.type`, `params`, `objects`, and `program` arrays; write per-episode JSON alongside grids.
- Loader: yields (train_pairs, test_inputs, labels) to Stage 1 training loops; later, switch to ARC where labels are unused.

Note: the 2D engine is only for data generation. Inference and the core solver continue to run on the latent Canvas (Sim2D), planner, and readout described above.

---
- ARC grids are <= 30x30; 32x32 Canvas safely covers them.
- Outputs are JSON grids written by eval scripts; no text outputs in submissions.

## Concept Storyboard mode (abstract stories beyond pixels)
- Purpose: use the same mental animation loop to reason about non‑pixel stories with nodes (concepts), edges (causal links), timelines, and magnitudes.
- Representation on the Canvas: dedicate episodic channels to a small concept graph: node slots, edge weights/signs, and a timeline axis; visualize as a 2D board for editability and probing.
- Same loop: propose z_story + slots → simulate updates over steps (propagate along edges over time) → probe counterfactuals → pick the simplest story that survives.
- Slots (concept mode examples):
  - graph: node embeddings, edge matrix (signed/weighted), community/group tags
  - timeline: key events, delays, periodicities
  - flows: resource/opinion/population flows between nodes
  - evidence: confidence per edge/node from retrieval

### Worked example: WWII → Gen Z (concept storyboard)
- Goal question: implications of WWII on Generation Z.
- Proposal: build a storyboard with nodes {WWII, postwar policy, baby boom, education expansion, tech adoption, Gen Z traits} and signed edges encoding influence.
- Simulate: step through decades, propagate effects along edges; accumulate impacts on Gen Z nodes.
- Probes (why):
  - remove edge WWII→postwar policy (does downstream collapse?)
  - flip sign on education→Gen Z tech adoption (contradiction with evidence?)
  - shift timeline of tech adoption earlier/later (misaligns with observed milestones?)
- Score: fit to retrieved facts (RAG), consistency across probes, simplicity (fewest explanatory edges) → pick the best storyboard and produce both the visual board and textual answer.

Note: The same z_story + slots abstraction powers both pixel ARC tasks and concept storyboards; only the slot types differ.
