#!/usr/bin/env bash
# ==============================================================================
# Script: prepare_dataset.sh
# Purpose: Parse all .osz beatmaps in input directory, extract audio & notes,
#          and generate tokenized PyTorch training tensors.
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

INPUT_DIR="${1:-sample beatmaps}"
OUTPUT_DIR="${2:-dataset_processed}"
CHUNK_SIZE="${3:-256}"
HOP_SIZE="${4:-128}"
ALIGN_MODE="${5:-beat}"

echo "======================================================================"
echo " 📦 PREPARING OSU!MANIA 4K TRAINING DATASET"
echo "======================================================================"
echo " Python Binary:     $PYTHON_BIN"
echo " Input Directory:   $INPUT_DIR"
echo " Output Directory:  $OUTPUT_DIR"
echo " Alignment Mode:    $ALIGN_MODE"
echo " Chunk Size:        $CHUNK_SIZE frames (~4.1 sec)"
echo " Hop Size:          $HOP_SIZE frames (~2.0 sec)"
echo "======================================================================"

if ! command -v ffmpeg &> /dev/null; then
    echo "[Warning] ffmpeg was not found in PATH. Audio decoding might fail."
fi

"$PYTHON_BIN" prepare_dataset.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --chunk_size "$CHUNK_SIZE" \
    --hop_size "$HOP_SIZE" \
    --align_mode "$ALIGN_MODE"

echo ""
echo "✅ Dataset preparation finished. You can now train any of the 4 models:"
echo "   ./train.sh 3 gpu 20 16   # (Train Model 3 on GPU for 20 epochs)"
echo "   ./train.sh 1 cpu 10 32   # (Train Model 1 on CPU for 10 epochs)"
