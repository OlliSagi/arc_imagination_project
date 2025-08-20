# proto_canvas — Stage 0 grounding (scaffold only)

What this folder contains:
- A minimal 32x32 Canvas with small local edits and a decode head
- A data builder that pools ARC grids as unlabeled states
- A trainer that logs Stage 0 metrics and writes checkpoints
- A simple image saver for decoded frames

How to run later (Windows cmd):
- Ensure deps installed (includes Pillow for image saving): `pip install -r requirements.txt`
- Example: `python proto_canvas\main.py --config proto_canvas\config.yaml --steps 1000`

Artifacts:
- `proto_canvas/checkpoints/canvas_stage0.safetensors`
- `proto_canvas/logs/metrics.jsonl`
- `proto_canvas/samples/step_XXXX.png`

Validation gates and thresholds are in `docs/STAGE_0_PLAN.md`.
