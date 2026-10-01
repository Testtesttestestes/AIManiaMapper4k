#!/usr/bin/env bash
# Quick runner for Dataset Compatibility, Grid Snapping, & Acoustic Critic Audit

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python3 test_dataset_critic_and_grid.py "$@"
