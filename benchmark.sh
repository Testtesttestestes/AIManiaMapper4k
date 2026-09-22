#!/usr/bin/env bash
# ==============================================================================
# Script: benchmark.sh
# Purpose: Benchmark training speed between CPU and GPU with throughput comparison.
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
BATCH_SIZE="${2:-16}"
STEPS="${3:-15}"

"$PYTHON_BIN" benchmark_train.py \
    --model "$MODEL" \
    --batch_size "$BATCH_SIZE" \
    --steps "$STEPS"
