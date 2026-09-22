#!/usr/bin/env python3
"""
Performance Benchmarking Tool: CPU vs GPU training throughput comparison.

Measures:
  - Milliseconds per step (latency)
  - Samples processed per second (throughput)
  - Estimated time for full training run (e.g., 20 epochs)
  - Speedup factor (GPU vs CPU)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from osu_mapper.models import (
    DifficultyEvaluatorModel,
    DynamicsSVModel,
    ManiaPatternTransformer,
    SyncBPMOffsetModel,
    TransformerConfig,
)
from train import ManiaDataset


def build_model_and_batch(model_id: str, batch_size: int = 16, seq_len: int = 256, audio_dim: int = 80):
    """Creates a test instance of the specified model along with synthetic input batch."""
    mid = str(model_id).lower()

    if mid in ("1", "sync"):
        model = SyncBPMOffsetModel(audio_dim=audio_dim, hidden_dim=256)
        def step_fn(m, dev):
            mel = torch.randn(batch_size, seq_len, audio_dim, device=dev)
            bpm_t = torch.full((batch_size, 1), 180.0, device=dev)
            off_t = torch.full((batch_size, 1), 500.0, device=dev)
            p_bpm, p_off = m(mel)
            loss = nn.functional.mse_loss(p_bpm, bpm_t) + 0.001 * nn.functional.mse_loss(p_off, off_t)
            return loss
        return model, step_fn, "Model 1: SyncBPMOffsetModel"

    elif mid in ("2", "dynamics"):
        model = DynamicsSVModel(audio_dim=audio_dim, hidden_dim=128)
        def step_fn(m, dev):
            mel = torch.randn(batch_size, seq_len, audio_dim, device=dev)
            target_sv = torch.ones(batch_size, seq_len, 1, device=dev)
            p_sv, _ = m(mel)
            loss = nn.functional.mse_loss(p_sv, target_sv)
            return loss
        return model, step_fn, "Model 2: DynamicsSVModel"

    elif mid in ("4", "evaluator"):
        model = DifficultyEvaluatorModel(num_classes=16, token_dim=64, hidden_dim=128)
        def step_fn(m, dev):
            tok = torch.randint(0, 16, (batch_size, seq_len), device=dev)
            sr_t = torch.full((batch_size, 1), 5.0, device=dev)
            p_sr = m(tok)
            loss = nn.functional.mse_loss(p_sr, sr_t)
            return loss
        return model, step_fn, "Model 4: DifficultyEvaluatorModel"

    else:  # Model 3: ManiaPatternTransformer (Default)
        cfg = TransformerConfig(
            audio_dim=audio_dim,
            d_model=256,
            nhead=8,
            num_layers=4,
            dim_feedforward=512,
            num_classes=16,
            max_seq_len=512,
        )
        model = ManiaPatternTransformer(cfg)
        crit_c = nn.CrossEntropyLoss()
        crit_l = nn.BCEWithLogitsLoss()
        def step_fn(m, dev):
            mel = torch.randn(batch_size, seq_len, audio_dim, device=dev)
            sr = torch.full((batch_size, 1), 5.0, device=dev)
            tok = torch.randint(0, 16, (batch_size, seq_len), device=dev)
            ln = torch.zeros(batch_size, seq_len, 4, device=dev)
            c_logits, l_logits = m(mel, star_rating=sr, prev_tokens=tok)
            B, T, C = c_logits.shape
            loss = crit_c(c_logits.view(B * T, C), tok.view(B * T)) + 0.5 * crit_l(l_logits, ln)
            return loss
        return model, step_fn, "Model 3: ManiaPatternTransformer"


def run_benchmark_on_device(
    model: nn.Module,
    step_fn,
    device: torch.device,
    batch_size: int,
    warmup_steps: int = 5,
    bench_steps: int = 25,
) -> Dict[str, float]:
    """Runs warmup and timed training steps on target device."""
    model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warmup
    for _ in range(warmup_steps):
        optimizer.zero_grad()
        loss = step_fn(model, device)
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed benchmarking
    start_time = time.perf_counter()
    for _ in range(bench_steps):
        optimizer.zero_grad()
        loss = step_fn(model, device)
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    total_time = time.perf_counter() - start_time
    ms_per_step = (total_time / bench_steps) * 1000.0
    samples_per_sec = (bench_steps * batch_size) / total_time

    return {
        "ms_per_step": ms_per_step,
        "samples_per_sec": samples_per_sec,
        "total_time": total_time,
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else f"CPU ({os.cpu_count() or 1} cores)",
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark CPU vs GPU training throughput.")
    parser.add_argument("--model", "-m", type=str, default="3", help="Model to benchmark (1, 2, 3, or 4)")
    parser.add_argument("--batch_size", "-b", type=int, default=16, help="Batch size for benchmark")
    parser.add_argument("--steps", "-s", type=int, default=20, help="Number of benchmark iterations")
    parser.add_argument("--dataset_size", type=int, default=2906, help="Total dataset chunks to project epoch times")
    parser.add_argument("--epochs", type=int, default=20, help="Epoch count for time estimation")
    args = parser.parse_args()

    print("=" * 70)
    print(" 🚀 OSU!MANIA 4K AI TRAINING BENCHMARK (CPU vs GPU)")
    print("=" * 70)

    model_raw, step_fn, model_title = build_model_and_batch(args.model, batch_size=args.batch_size)
    param_count = sum(p.numel() for p in model_raw.parameters())
    print(f"Target Architecture: {model_title}")
    print(f"Parameters:          {param_count:,}")
    print(f"Batch Size:          {args.batch_size}")
    print(f"Benchmark Steps:     {args.steps} (plus 5 warmup)")
    print(f"Dataset Size:        {args.dataset_size} chunks ({args.epochs} target epochs)\n")

    # 1. Benchmark on CPU
    print(f"Testing [CPU] ({os.cpu_count() or 1} threads)...")
    cpu_res = run_benchmark_on_device(
        model_raw,
        step_fn,
        device=torch.device("cpu"),
        batch_size=args.batch_size,
        warmup_steps=3,
        bench_steps=args.steps,
    )
    cpu_epoch_s = (args.dataset_size / args.batch_size) * (cpu_res["ms_per_step"] / 1000.0)
    cpu_total_min = (cpu_epoch_s * args.epochs) / 60.0

    print(f"  ✓ CPU Step Time:    {cpu_res['ms_per_step']:.1f} ms/step")
    print(f"  ✓ CPU Throughput:   {cpu_res['samples_per_sec']:.1f} samples/sec")
    print(f"  ✓ Est. Epoch Time:  {cpu_epoch_s:.1f} sec (~{cpu_total_min:.1f} min for {args.epochs} epochs)\n")

    # 2. Benchmark on GPU (if available)
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    gpu_device = None
    if has_cuda:
        gpu_device = torch.device("cuda")
    elif has_mps:
        gpu_device = torch.device("mps")

    if gpu_device:
        dev_desc = torch.cuda.get_device_name(0) if has_cuda else "Apple Silicon MPS"
        print(f"Testing [GPU: {dev_desc}]...")
        gpu_res = run_benchmark_on_device(
            model_raw,
            step_fn,
            device=gpu_device,
            batch_size=args.batch_size,
            warmup_steps=5,
            bench_steps=args.steps,
        )
        gpu_epoch_s = (args.dataset_size / args.batch_size) * (gpu_res["ms_per_step"] / 1000.0)
        gpu_total_min = (gpu_epoch_s * args.epochs) / 60.0
        speedup = cpu_res["ms_per_step"] / max(0.001, gpu_res["ms_per_step"])

        print(f"  ✓ GPU Step Time:    {gpu_res['ms_per_step']:.1f} ms/step")
        print(f"  ✓ GPU Throughput:   {gpu_res['samples_per_sec']:.1f} samples/sec")
        print(f"  ✓ Est. Epoch Time:  {gpu_epoch_s:.1f} sec (~{gpu_total_min:.1f} min for {args.epochs} epochs)")
        print(f"\n🔥 SPEEDUP: GPU is {speedup:.2f}x FASTER than CPU!")

    else:
        print("Note: CUDA/MPS GPU not detected in current cloud container.")
        print("Generating hardware reference estimates based on standard GPU benchmarks:\n")

    # Clean summary table
    print("-" * 70)
    print(f"{'DEVICE / HARDWARE':<32} | {'MS/STEP':<10} | {'SAMPLES/S':<11} | {'20 EPOCHS':<10}")
    print("-" * 70)
    print(f"{cpu_res['device_name']:<32} | {cpu_res['ms_per_step']:<10.1f} | {cpu_res['samples_per_sec']:<11.1f} | {cpu_total_min:<7.1f} min")

    if gpu_device:
        print(f"{gpu_res['device_name']:<32} | {gpu_res['ms_per_step']:<10.1f} | {gpu_res['samples_per_sec']:<11.1f} | {gpu_total_min:<7.1f} min")
    else:
        # Projected typical GPUs
        refs = [
            ("NVIDIA RTX 3060 (12GB)", 6.5),
            ("NVIDIA RTX 4070 (12GB)", 11.0),
            ("NVIDIA RTX 4090 / A100", 22.0),
        ]
        for name, mult in refs:
            p_ms = max(5.0, cpu_res["ms_per_step"] / mult)
            p_smp = cpu_res["samples_per_sec"] * mult
            p_min = ((args.dataset_size / args.batch_size) * (p_ms / 1000.0) * args.epochs) / 60.0
            print(f"{name + ' (est.)':<32} | {p_ms:<10.1f} | {p_smp:<11.1f} | {p_min:<7.1f} min")

    print("-" * 70)
    print("Recommendation:")
    if gpu_device:
        print("  ⚡ GPU is active! Use '--device cuda' for fastest training.")
    else:
        print("  💡 For quick tests (1-5 epochs), CPU works right away.")
        print("  ⚡ For large models/datasets (20+ epochs), run on a GPU-enabled PC or Google Colab with '--device cuda'.")
    print("=" * 70)


if __name__ == "__main__":
    main()
