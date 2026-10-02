#!/usr/bin/env bash
# ==============================================================================
# Script: train_diffusion.sh
# Purpose: Launches ManiaDiffusion 1D Latent Beatmap Diffusion training.
# Optimized for multi-GPU setups (e.g. 3x RTX 5060) with batch size 64 per card.
# ==============================================================================

set -e

if command -v python3 &> /dev/null; then
    PYTHON_BIN="python3"
elif command -v python &> /dev/null; then
    PYTHON_BIN="python"
elif command -v py &> /dev/null; then
    PYTHON_BIN="py"
else
    echo "❌ [Error] Python not found."
    exit 1
fi

EPOCHS="${1:-100}"
BATCH_SIZE="${2:-64}"
LR="${3:-0.0001}"
RESUME_FLAG=""

if [ "$4" == "resume" ] || [ "${RESUME:-0}" == "1" ]; then
    RESUME_FLAG="--resume"
fi

echo "=================================================================="
echo "🌊 Starting ManiaDiffusion Training (1D Beatmap Latent Diffusion)"
echo "   Epochs:       $EPOCHS"
echo "   Batch/GPU:    $BATCH_SIZE"
echo "   LR:           $LR"
echo "   Resume:       ${RESUME_FLAG:-No}"
echo "=================================================================="

$PYTHON_BIN train_diffusion.py \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    $RESUME_FLAG
