#!/usr/bin/env bash
set -Eeuo pipefail

# Usage:
#   bash scripts/run_on_runpod.sh [--run-stage1] [--stage0-steps N] [--stage1-steps N]
# Env overrides (optional):
#   REPO_URL=https://github.com/OlliSagi/arc_imagination_project.git
#   CLONE_DIR=arc_imagination_project

REPO_URL="${REPO_URL:-https://github.com/OlliSagi/arc_imagination_project.git}"
CLONE_DIR="${CLONE_DIR:-arc_imagination_project}"
RUN_STAGE1=false
STAGE0_STEPS=3000
STAGE1_STEPS=3000

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-stage1)
      RUN_STAGE1=true; shift ;;
    --stage0-steps)
      STAGE0_STEPS="$2"; shift 2 ;;
    --stage1-steps)
      STAGE1_STEPS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "[INFO] Repo URL: $REPO_URL"
echo "[INFO] Clone dir: $CLONE_DIR"
echo "[INFO] RUN_STAGE1: $RUN_STAGE1"
echo "[INFO] STAGE0_STEPS: $STAGE0_STEPS"
echo "[INFO] STAGE1_STEPS: $STAGE1_STEPS"

# 0) Clone if needed
if [[ ! -d "$CLONE_DIR" ]]; then
  echo "[INFO] Cloning repo..."
  git clone "$REPO_URL" "$CLONE_DIR"
fi
cd "$CLONE_DIR"

# 1) Python venv and deps
if [[ ! -d .venv ]]; then
  echo "[INFO] Creating venv..."
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip

echo "[INFO] Installing GPU Torch (CUDA 12.4 wheel) ..."
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio

echo "[INFO] Installing project dependencies..."
pip install -r requirements.txt

# 2) Verify GPU
echo "[INFO] Verifying GPU availability..."
python - << 'PY'
import torch
print("cuda?", torch.cuda.is_available(), "device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY

# 3) Check ARC data presence (optional but recommended)
if [[ ! -d ARC-AGI/data/training ]]; then
  echo "[WARN] ARC-AGI/data/training not found. Stage 0 can run but will have no real ARC grids."
fi

# 4) Stage 0 — Canvas grounding
echo "[INFO] Running Stage 0 (Canvas) for $STAGE0_STEPS steps..."
python proto_canvas/main.py --config proto_canvas/config.yaml --steps "$STAGE0_STEPS"
echo "[INFO] Stage 0 outputs: proto_canvas/checkpoints, proto_canvas/logs, proto_canvas/samples"

# 5) Optional Stage 1 — Slots + motifs
if [[ "$RUN_STAGE1" == "true" ]]; then
  echo "[INFO] Ensuring synthetic episodes exist for Stage 1..."
  if [[ ! -d data/synthetic || -z "$(ls -A data/synthetic 2>/dev/null || true)" ]]; then
    python -m src.synth.generate --out_dir data/synthetic --num 300 --height 20 --width 20 --seed 0 --events recolor_by_holes,mirror_x,translate
  fi
  if [[ ! -d data/synthetic2 || -z "$(ls -A data/synthetic2 2>/dev/null || true)" ]]; then
    python -m src.synth.generate --out_dir data/synthetic2 --num 300 --height 20 --width 20 --seed 42 --events recolor_by_holes,mirror_x,translate
  fi

  echo "[INFO] Running Stage 1 (Slots/Motifs) for $STAGE1_STEPS steps..."
  python proto_stage1/main.py --config proto_stage1/config.yaml --steps "$STAGE1_STEPS"
  echo "[INFO] Stage 1 outputs: proto_stage1/checkpoints, proto_stage1/logs"
fi

echo "[DONE] All requested stages completed."


