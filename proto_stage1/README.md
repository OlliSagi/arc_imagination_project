# proto_stage1 — Slots + Motifs shaping (scaffold)

This folder trains typed slot heads and simple event heads on synthetic episodes, keeping Stage 0 stability losses.

How to run later (Windows cmd):
- Ensure dependencies are installed: `pip install -r requirements.txt`
- Optional: place Stage 0 checkpoint at `proto_canvas/checkpoints/canvas_stage0.safetensors`
- Run: `python proto_stage1\main.py --config proto_stage1\config.yaml --steps 1000`

Artifacts:
- `proto_stage1/checkpoints/sim2d_stage1.safetensors`
- `proto_stage1/checkpoints/readout_stage1.safetensors` (slot heads)
- `proto_stage1/logs/metrics_stage1.jsonl`

Validation gates are in `docs/STAGE_1_PLAN.md`.
