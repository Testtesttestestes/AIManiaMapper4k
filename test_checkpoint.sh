#!/usr/bin/env bash
# ==============================================================================
# OSU!MANIA 4K AI MODEL CHECKPOINT TESTER (Bash)
# Usage:
#   ./test_checkpoint.sh 1 track.mp3
#   ./test_checkpoint.sh 1 dataset_processed/train_data.pt
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
    exit 1
fi

MODEL="${1:-1}"
TARGET="${2:-output_beatmaps/synthetic_test.wav}"
CHECKPOINT="$3"

EXTRA_ARGS=""
if [ -n "$CHECKPOINT" ]; then
    EXTRA_ARGS="--checkpoint $CHECKPOINT"
fi

if [[ "$TARGET" == *.pt ]]; then
    "$PYTHON_BIN" test_checkpoint.py --model "$MODEL" --dataset "$TARGET" $EXTRA_ARGS
else
    "$PYTHON_BIN" test_checkpoint.py --model "$MODEL" --audio "$TARGET" $EXTRA_ARGS
fi
