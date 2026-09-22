#!/usr/bin/env bash
# ==============================================================================
# Script: benchmark.sh
# Purpose: Benchmark training speed between CPU and GPU with throughput comparison.
# ==============================================================================

set -e

MODEL="${1:-3}"
BATCH_SIZE="${2:-16}"
STEPS="${3:-15}"

python3 benchmark_train.py \
    --model "$MODEL" \
    --batch_size "$BATCH_SIZE" \
    --steps "$STEPS"
