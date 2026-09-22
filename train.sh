#!/usr/bin/env bash
# ==============================================================================
# Script: train.sh
# Purpose: Interactive or CLI-based training runner for osu!mania 4K AI models.
#
# Usage:
#   ./train.sh [model] [device] [epochs] [batch_size] [lr] [timing_source] [resume]
# Examples:
#   ./train.sh 3 gpu 20 16
#   ./train.sh 3 gpu 20 16 0.0002 map resume
#   RESUME=1 ./train.sh 3 gpu 20 16
# ==============================================================================

set -e

# Detect Python executable (python3, python, or py)
if command -v python3 &> /dev/null; then
    PYTHON_BIN="python3"
elif command -v python &> /dev/null; then
    PYTHON_BIN="python"
elif command -v py &> /dev/null; then
    PYTHON_BIN="py"
else
    echo "❌ [Error] Neither 'python3', 'python', nor 'py' was found in your PATH."
    echo "   Please make sure Python is installed and added to your system environment variables."
    exit 1
fi

MODEL="${1:-3}"
DEVICE="${2:-auto}"
EPOCHS="${3:-15}"
BATCH_SIZE="${4:-16}"
LR="${5:-0.0002}"
TIMING_SOURCE="${6:-map}"
RESUME_ARG="${7:-${RESUME:-0}}"
DATASET="dataset_processed/train_data.pt"

if [ ! -f "$DATASET" ]; then
    echo "⚠️  Dataset file '$DATASET' not found!"
    echo "    Running dataset preparation first..."
    ./prepare_dataset.sh
fi

EXTRA_FLAGS=()
if [ "$RESUME_ARG" = "1" ] || [ "$RESUME_ARG" = "true" ] || [ "$RESUME_ARG" = "resume" ] || [ "$RESUME_ARG" = "auto" ]; then
    EXTRA_FLAGS+=("--resume")
    RESUME_DISPLAY="YES (Auto-search in checkpoints/)"
elif [ -n "$RESUME_ARG" ] && [ "$RESUME_ARG" != "0" ] && [ "$RESUME_ARG" != "false" ]; then
    EXTRA_FLAGS+=("--resume" "$RESUME_ARG")
    RESUME_DISPLAY="YES ($RESUME_ARG)"
else
    RESUME_DISPLAY="NO (Fresh start)"
fi

echo "======================================================================"
echo " 🤖 OSU!MANIA 4K AI MODEL TRAINING"
echo "======================================================================"
echo " Python Binary:   $PYTHON_BIN"
echo " Target Model:    $MODEL (1=Sync, 2=Dynamics, 3=Transformer, 4=Evaluator, all)"
echo " Target Device:   $DEVICE (cpu / cuda / mps / auto)"
echo " Data Source:     $TIMING_SOURCE (map = Ground Truth .osu, model1 = Cascaded Sync)"
echo " Resume Mode:     $RESUME_DISPLAY"
echo " Epochs:          $EPOCHS"
echo " Batch Size:      $BATCH_SIZE"
echo " Learning Rate:   $LR"
echo " Checkpoints Dir: checkpoints/"
echo "======================================================================"

"$PYTHON_BIN" train.py \
    --model "$MODEL" \
    --device "$DEVICE" \
    --dataset "$DATASET" \
    --checkpoint_dir "checkpoints" \
    --timing_source "$TIMING_SOURCE" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    "${EXTRA_FLAGS[@]}"

echo ""
echo "✅ Finished training model $MODEL!"
echo "   Saved checkpoints can be found in 'checkpoints/'."
