#!/usr/bin/env bash
# ==============================================================================
# Script: prepare_dataset.sh
# Purpose: Parse all .osz beatmaps in input directory, extract audio & notes,
#          and generate tokenized PyTorch training tensors.
# ==============================================================================

set -e

INPUT_DIR="${1:-sample beatmaps}"
OUTPUT_DIR="${2:-dataset_processed}"
CHUNK_SIZE="${3:-256}"
HOP_SIZE="${4:-128}"

echo "======================================================================"
echo " 📦 PREPARING OSU!MANIA 4K TRAINING DATASET"
echo "======================================================================"
echo " Input Directory:   $INPUT_DIR"
echo " Output Directory:  $OUTPUT_DIR"
echo " Chunk Size:        $CHUNK_SIZE frames (~4.1 sec)"
echo " Hop Size:          $HOP_SIZE frames (~2.0 sec)"
echo "======================================================================"

if ! command -v ffmpeg &> /dev/null; then
    echo "[Warning] ffmpeg was not found in PATH. Audio decoding might fail."
fi

python3 prepare_dataset.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --chunk_size "$CHUNK_SIZE" \
    --hop_size "$HOP_SIZE"

echo ""
echo "✅ Dataset preparation finished. You can now train any of the 4 models:"
echo "   ./train.sh 3 gpu 20 16   # (Train Model 3 on GPU for 20 epochs)"
echo "   ./train.sh 1 cpu 10 32   # (Train Model 1 on CPU for 10 epochs)"
