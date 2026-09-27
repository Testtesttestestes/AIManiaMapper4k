#!/usr/bin/env python3
"""
Unified PyTorch training script for osu!mania 4K AI models.

Supports training any of the 4 models defined in the specification:
  - Model 1 (sync):      SyncBPMOffsetModel (BPM & audio offset synchronization)
  - Model 2 (dynamics):  DynamicsSVModel (Slider Velocity & Timing Zones)
  - Model 3 (transformer): ManiaPatternTransformer (4K Hit Object pattern generator)
  - Model 4 (evaluator): DifficultyEvaluatorModel (Star Rating & density validation)
  - all:                 Train all models sequentially

Features:
  - Full seamless checkpoint resume (--resume / --resume_from): restores model weights,
    optimizer momentum states, scheduler, and epoch counters without any loss of training continuity.
  - 100% backward-compatible: gracefully loads legacy checkpoints that lack optimizer state.
  - Continuous checkpointing: automatically writes model_X_last.pt every epoch and preserves
    progress on Ctrl+C interruption.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from osu_mapper.models import (
    DifficultyEvaluatorModel,
    DynamicsSVModel,
    ManiaPatternTransformer,
    SyncBPMOffsetModel,
    TransformerConfig,
)
from osu_mapper.pipeline.onset_scaffold import compute_acoustic_onset_scaffold


class ManiaDataset(Dataset):
    """Loads preprocessed training chunks from single PyTorch tensor file or partitioned shard directory."""
    def __init__(self, data_path: str | Path):
        self.path = Path(data_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset path not found at '{self.path}'. Run 'python prepare_dataset.py' first.")

        self.chunks: List[Dict[str, torch.Tensor]] = []

        if self.path.is_file():
            print(f"[Dataset] Loading '{self.path}' into memory...")
            self.chunks = torch.load(self.path, weights_only=False)
        else:
            # Check for sharded parts: train_data_part_*.pt or train_data.pt
            shards = sorted(list(self.path.glob("train_data_part_*.pt")))
            if shards:
                print(f"[Dataset] Loading {len(shards)} dataset shards from '{self.path}'...")
                for s_i, shard in enumerate(shards):
                    shard_data = torch.load(shard, weights_only=False)
                    self.chunks.extend(shard_data)
                    print(f"  • Shard {s_i + 1}/{len(shards)}: {shard.name} ({len(shard_data)} samples, total: {len(self.chunks)})")
            elif (self.path / "train_data.pt").exists():
                main_file = self.path / "train_data.pt"
                print(f"[Dataset] Loading '{main_file}' into memory...")
                self.chunks = torch.load(main_file, weights_only=False)
            else:
                all_pts = sorted(list(self.path.glob("*.pt")))
                if all_pts:
                    for pt_file in all_pts:
                        if "song_sync" in pt_file.name:
                            continue
                        self.chunks.extend(torch.load(pt_file, weights_only=False))
                else:
                    raise FileNotFoundError(f"No .pt tensor dataset files or shards found in '{self.path}'.")

        print(f"[Dataset] Successfully loaded {len(self.chunks)} training samples.")

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.chunks[idx]


class SubsetDataset(Dataset):
    def __init__(self, full_dataset: ManiaDataset, indices: List[int]):
        self.full_dataset = full_dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.full_dataset[self.indices[idx]]


def stratified_difficulty_split(
    full_dataset: ManiaDataset,
    val_split: float = 0.10,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """
    Stratified Star Rating Difficulty Splitter:
    Groups chunks into discrete SR bins to guarantee that every difficulty tier
    (including 5.5* - 7.0*+ maps) is proportionally represented in the training set
    and NEVER completely starved into validation.
    """
    import random
    rng = random.Random(seed)

    bins: Dict[str, List[int]] = {
        "Easy (1.0 - 2.5*)": [],
        "Normal (2.5 - 3.75*)": [],
        "Hard (3.75 - 4.75*)": [],
        "Insane (4.75 - 5.75*)": [],
        "Extra (5.75 - 7.0*)": [],
        "Master (7.0*+)": [],
    }

    for idx, chunk in enumerate(full_dataset.chunks):
        sr = float(chunk["star_rating"].item()) if "star_rating" in chunk else 3.5
        if sr < 2.5:
            bins["Easy (1.0 - 2.5*)"].append(idx)
        elif sr < 3.75:
            bins["Normal (2.5 - 3.75*)"].append(idx)
        elif sr < 4.75:
            bins["Hard (3.75 - 4.75*)"].append(idx)
        elif sr < 5.75:
            bins["Insane (4.75 - 5.75*)"].append(idx)
        elif sr < 7.0:
            bins["Extra (5.75 - 7.0*)"].append(idx)
        else:
            bins["Master (7.0*+)"].append(idx)

    train_indices: List[int] = []
    val_indices: List[int] = []

    print("\n📊 [Dataset Stratified Difficulty Split]")
    for bin_name, idxs in bins.items():
        if not idxs:
            continue
        rng.shuffle(idxs)
        v_count = max(1, int(round(len(idxs) * val_split))) if len(idxs) > 1 else 0
        t_count = len(idxs) - v_count
        val_indices.extend(idxs[:v_count])
        train_indices.extend(idxs[v_count:])
        print(f"  • {bin_name:<22} -> Train: {t_count:>4} samples | Val: {v_count:>3} samples (Total: {len(idxs)})")

    if not train_indices:
        train_indices = list(range(len(full_dataset)))
        val_indices = []

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    print(f"  ✓ Balanced Stratified Split: {len(train_indices)} train / {len(val_indices)} val samples.\n")
    return SubsetDataset(full_dataset, train_indices), SubsetDataset(full_dataset, val_indices)


def get_device(
    requested: str = "auto",
    gpus: str = "auto",
    multi_gpu: bool = False,
) -> Tuple[torch.device, List[int]]:
    """Resolves target compute device and multi-GPU device IDs."""
    req = requested.lower()
    if req == "cpu" or not torch.cuda.is_available():
        if req == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("[Device] Using Apple Silicon MPS GPU")
            return torch.device("mps"), []
        print(f"[Device] Using CPU ({os.cpu_count() or 1} cores detected)")
        return torch.device("cpu"), []

    num_gpus = torch.cuda.device_count()
    print(f"\n🖥️  [CUDA Devices Detected: {num_gpus}]")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024 ** 3)
        print(f"  • GPU {i}: {props.name} ({vram_gb:.1f} GB VRAM)")

    device_ids: List[int] = []
    gpus_req = str(gpus).lower().strip()
    if gpus_req in ("all", "all_gpus") or multi_gpu or (gpus_req == "auto" and num_gpus > 1 and multi_gpu):
        device_ids = list(range(num_gpus))
    elif gpus_req not in ("auto", "none", "0"):
        try:
            device_ids = [int(x.strip()) for x in gpus_req.split(",") if x.strip()]
        except Exception:
            device_ids = [0]
    else:
        device_ids = [0]

    primary_device = torch.device(f"cuda:{device_ids[0]}")
    if len(device_ids) > 1:
        print(f"  🚀 Multi-GPU Acceleration Active across {len(device_ids)} GPUs: {device_ids} (PyTorch DataParallel)\n")
    else:
        print(f"  ✓ Single GPU mode: GPU {device_ids[0]} ({torch.cuda.get_device_name(device_ids[0])})\n")

    return primary_device, device_ids


def wrap_model_multi_gpu(model: nn.Module, device_ids: List[int]) -> nn.Module:
    """Wraps model in DataParallel if multiple GPU devices are targeted."""
    if len(device_ids) > 1 and torch.cuda.is_available():
        return nn.DataParallel(model, device_ids=device_ids)
    return model


def find_resume_checkpoint(checkpoint_dir: Path, model_prefix: str, resume_arg: Optional[str] = None) -> Optional[Path]:
    """
    Finds the appropriate checkpoint to resume from:
    1. If resume_arg is a valid file path, use it directly.
    2. If resume_arg is inside checkpoint_dir, use it.
    3. Auto search order: {model_prefix}_last.pt -> {model_prefix}_best.pt -> {model_prefix}.pt
    """
    if resume_arg and resume_arg.lower() not in ("auto", "true", "1", "yes", "default"):
        p = Path(resume_arg)
        if p.exists() and p.is_file():
            return p
        p2 = checkpoint_dir / resume_arg
        if p2.exists() and p2.is_file():
            return p2
        print(f"⚠️  [Resume Warning] Explicit checkpoint path '{resume_arg}' was not found.")

    candidates = [
        checkpoint_dir / f"{model_prefix}_last.pt",
        checkpoint_dir / f"{model_prefix}_best.pt",
        checkpoint_dir / f"{model_prefix}.pt",
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def load_checkpoint_state(
    ckpt_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
    override_lr: Optional[float] = None,
) -> Tuple[int, float, Dict[str, Any]]:
    """
    Loads checkpoint state with 100% backward compatibility:
    - Restores model weights regardless of key naming format (model_state, state_dict, raw dict).
    - Restores optimizer state (momentum, Adam second moments, step count) if present.
    - Gracefully initializes optimizer if checkpoint was saved without optimizer_state.
    - Applies custom override learning rate (--lr) upon resume if provided.
    - Restores LR scheduler state and starting epoch.
    """
    print(f"🔄 [Resume] Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device or "cpu", weights_only=False)

    # 1. Model weights
    state_dict = None
    if isinstance(ckpt, dict):
        for k in ("model_state", "model_state_dict", "state_dict"):
            if k in ckpt and isinstance(ckpt[k], dict):
                state_dict = ckpt[k]
                break
        if state_dict is None and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state_dict = ckpt
    else:
        raise ValueError(f"Unrecognized checkpoint format in {ckpt_path}")

    if state_dict is not None:
        # Clean prefix if saved with or without DataParallel / DDP
        target_keys = set(model.state_dict().keys())
        has_module_prefix_in_model = any(k.startswith("module.") for k in target_keys)
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module.") and not has_module_prefix_in_model:
                cleaned_state_dict[k[7:]] = v
            elif not k.startswith("module.") and has_module_prefix_in_model:
                cleaned_state_dict["module." + k] = v
            else:
                cleaned_state_dict[k] = v

        missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
        if missing_keys:
            print(f"  ⚡ Warm-start loaded backbone weights ({len(missing_keys)} new layer weight(s) initialized cleanly: {missing_keys[:3]}...).")
        else:
            print("  ✓ Restored all model weights successfully.")

    # 2. Starting epoch
    start_epoch = 0
    if isinstance(ckpt, dict):
        start_epoch = int(ckpt.get("epoch", 0))

    # 3. Best metric
    best_metric = float("inf")
    if isinstance(ckpt, dict):
        for k in ("best_val_loss", "val_loss", "best_val_mae", "val_bpm_mae", "best_metric"):
            if k in ckpt and ckpt[k] is not None:
                best_metric = float(ckpt[k])
                break

    # 4. Optimizer state (Exact continuation with optional LR override)
    if optimizer is not None and isinstance(ckpt, dict) and "optimizer_state" in ckpt and ckpt["optimizer_state"] is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            saved_lr = optimizer.param_groups[0].get("lr", None) if optimizer.param_groups else None

            if override_lr is not None:
                for group in optimizer.param_groups:
                    group["lr"] = override_lr
                if saved_lr is not None and abs(saved_lr - override_lr) > 1e-12:
                    print(f"  ⚡ Restored optimizer momentum states and updated LR: {saved_lr:.2e} -> {override_lr:.2e} (--lr applied)")
                else:
                    print(f"  ✓ Restored optimizer momentum, moving averages, and step states (LR: {override_lr:.2e}).")
            else:
                curr_lr_str = f"{saved_lr:.2e}" if saved_lr is not None else "default"
                print(f"  ✓ Restored optimizer momentum, moving averages, and step states (LR: {curr_lr_str}).")
        except Exception as opt_err:
            print(f"  ℹ️ Notice: Optimizer state format differed ({opt_err}); continuing with fresh optimizer parameters.")
            if override_lr is not None:
                for group in optimizer.param_groups:
                    group["lr"] = override_lr
    elif optimizer is not None:
        if override_lr is not None:
            for group in optimizer.param_groups:
                group["lr"] = override_lr
        print(f"  ℹ️ Legacy checkpoint detected: continuing with fresh optimizer parameters from Epoch {start_epoch}.")

    # 5. Scheduler state
    if scheduler is not None and isinstance(ckpt, dict) and "scheduler_state" in ckpt and ckpt["scheduler_state"] is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state"])
            if override_lr is not None and hasattr(scheduler, "base_lrs"):
                scheduler.base_lrs = [override_lr for _ in scheduler.base_lrs]
            print("  ✓ Restored LR scheduler state.")
        except Exception:
            pass

    return start_epoch, best_metric, ckpt if isinstance(ckpt, dict) else {}


def compute_multi_scenario_scaffold_and_strain(
    tokens: torch.Tensor,
    star_rating: torch.Tensor,
    grid_metrics: Optional[torch.Tensor] = None,
    audio_mel: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Synthesizes diverse realistic Multi-Scenario training inputs:
      Scenario A (35%): Algorithmic Heuristic Debris (Stage 1 downbeat/grid draft)
      Scenario B (35%): Extreme Neural/Human Debris (Spam bursts, wrong lanes, density drift)
      Scenario C (15%): Sparse / Blank Scaffold (forces model to reconstruct from pure audio)
      Scenario D (15%): Softly Perturbed Scaffold (cues with minor deviations)
    Ensures the model functions as a robust Garbage-to-Master-Map converter.
    """
    B, T = tokens.shape
    device = tokens.device

    # 1. Target Strain Trajectory Metrics [B, T, 2]: [running_density, target_ratio]
    is_note = tokens.gt(0).float()
    cum_notes = torch.cumsum(is_note, dim=1)
    time_steps = torch.arange(1, T + 1, device=device, dtype=torch.float32).unsqueeze(0).expand(B, T)
    running_density = cum_notes / time_steps
    target_density = (star_rating.view(B, 1) / 10.0).clamp(min=0.05).expand(B, T)
    density_ratio = (running_density / target_density).clamp(max=5.0)
    strain_metrics = torch.stack([running_density, density_ratio], dim=-1)

    # 2. Multi-Scenario Acoustic Scaffold Generator (Zero Ground-Truth Leakage)
    # Scenario A (35%): Stage 1 Acoustic Onset Scaffold
    # Scenario B (35%): Extreme Debris & Spam (Heavy noise 40-70%, uncoordinated minijack bursts, spam clusters)
    # Scenario C (15%): Blank Scaffold (Pure Audio-to-Pattern synthesis)
    # Scenario D (15%): Soft Draft (Acoustic scaffold with mild 15% noise)
    acoustic_scaffold = compute_acoustic_onset_scaffold(audio_mel, grid_metrics, star_rating)
    scaffold_tokens = torch.zeros(B, T, dtype=torch.int64, device=device)

    for b in range(B):
        scenario_choice = torch.rand(1).item()

        if scenario_choice < 0.35:
            # Scenario A: Standard Acoustic Spectral Onset Scaffold
            scaffold_tokens[b] = acoustic_scaffold[b]

        elif scenario_choice < 0.70:
            # Scenario B: Extreme Debris & Spam (Injected into acoustic scaffold, NOT ground truth)
            debris = acoustic_scaffold[b].clone()
            noise_rate = torch.empty(1).uniform_(0.40, 0.75).item()
            noise_mask = torch.rand(T, device=device) < noise_rate
            debris[noise_mask] = torch.randint(0, 16, (noise_mask.sum(),), device=device)

            # Inject uncoordinated minijacks (repeated identical tokens in a single column)
            if torch.rand(1).item() < 0.5:
                jack_span = torch.randint(4, 16, (1,)).item()
                start_j = torch.randint(0, max(1, T - jack_span), (1,)).item()
                debris[start_j : start_j + jack_span] = 1 << torch.randint(0, 4, (1,)).item()

            scaffold_tokens[b] = debris

        elif scenario_choice < 0.85:
            # Scenario C: Blank Scaffold (Forces model to synthesize 100% from audio)
            scaffold_tokens[b] = torch.zeros(T, dtype=torch.int64, device=device)

        else:
            # Scenario D: Soft Draft (15% jitter)
            soft = acoustic_scaffold[b].clone()
            soft_mask = torch.rand(T, device=device) < 0.15
            soft[soft_mask] = torch.randint(0, 16, (soft_mask.sum(),), device=device)
            scaffold_tokens[b] = soft

    return scaffold_tokens, strain_metrics


def train_model_3_transformer(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    checkpoint_dir: Path,
    timing_source: str = "map",
    resume_path: Optional[str] = None,
    context_dropout: float = 0.20,
    override_lr: Optional[float] = None,
    device_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Trains Model 3: ManiaPatternTransformer (Hit Object Synthesis).
    Supports resuming training with optional custom learning rate override.
    """
    print("\n" + "=" * 60)
    print("▶ TRAINING MODEL 3: ManiaPatternTransformer (Hit Object Synthesis)")
    source_desc = "ORIGINAL BEATMAP GROUND TRUTH (Direct from .osu — zero Model 1 drift)" if timing_source == "map" else "MODEL 1 CASCADED TIMING (AI Sync Inferred)"
    print(f"  Timing & Data Source: {source_desc}")
    print(f"  Context Dropout:      {context_dropout * 100:.0f}% (Anti-Cold-Start Regularization)")
    print("=" * 60)

    cfg = TransformerConfig(
        audio_dim=80,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        num_classes=16,
        max_seq_len=512,
        dropout=0.1,
        context_dropout=context_dropout,
    )
    raw_model = ManiaPatternTransformer(cfg).to(device)
    model = wrap_model_multi_gpu(raw_model, device_ids or [])
    total_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # 1. Analyze training data distribution for class balancing & LN penalty
    print("  • Calculating class frequency and Long Note sparsity on dataset...")
    chord_counts = torch.zeros(16)
    total_ln_active = 0
    total_ln_frames = 0
    for sample in train_loader.dataset:
        toks = sample["tokens"]
        for c in range(16):
            chord_counts[c] += (toks == c).sum().item()
        ln_m = sample["ln_mask"]
        total_ln_active += (ln_m > 0.5).sum().item()
        total_ln_frames += ln_m.numel()

    # Chord class weighting: penalize missing active notes (tokens 1..15) vs empty ticks (0)
    total_toks = chord_counts.sum().item()
    chord_weights = torch.ones(16, device=device)
    for c in range(16):
        if chord_counts[c] > 0:
            chord_weights[c] = (total_toks / (16.0 * chord_counts[c])) ** 0.5
        else:
            chord_weights[c] = 1.0
    chord_weights = chord_weights / chord_weights[0]  # Normalize relative to token 0
    chord_weights = torch.clamp(chord_weights, min=0.5, max=8.0).to(device)

    # LN positive penalty weight: heavily penalize missed hold notes/sliders
    if total_ln_active > 0:
        ln_neg = total_ln_frames - total_ln_active
        ln_pos_ratio = max(6.0, min(20.0, ln_neg / max(1.0, total_ln_active)))
    else:
        ln_pos_ratio = 12.0
    ln_pos_weight = torch.tensor([ln_pos_ratio], device=device)
    print(f"  • Active Note Class Multiplier: max {chord_weights.max():.2f}x | LN Penalty Weight (pos_weight): {ln_pos_ratio:.1f}x")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion_chord = nn.CrossEntropyLoss(weight=chord_weights)
    criterion_ln = nn.BCEWithLogitsLoss(pos_weight=ln_pos_weight)

    start_epoch = 0
    best_val_loss = float("inf")
    history: List[Dict[str, Any]] = []

    # Handle resume
    ckpt_file = find_resume_checkpoint(checkpoint_dir, "model_3_transformer", resume_path)
    if ckpt_file:
        start_epoch, best_metric, ckpt_dict = load_checkpoint_state(
            ckpt_file, model, optimizer=optimizer, device=device, override_lr=override_lr
        )
        if best_metric < float("inf"):
            best_val_loss = best_metric
        if "history" in ckpt_dict and isinstance(ckpt_dict["history"], list):
            history = list(ckpt_dict["history"])
        print(f"  ▶ Resumed at Epoch {start_epoch}. Best Val Loss so far: {best_val_loss:.4f}")

    target_total_epochs = epochs
    if target_total_epochs <= start_epoch:
        target_total_epochs = start_epoch + epochs
        print(f"  ℹ️ Target epochs ({epochs}) <= Loaded epoch ({start_epoch}). Extending total epochs to {target_total_epochs} (+{epochs} more epochs).")

    print(f"  Training from Epoch {start_epoch + 1} to {target_total_epochs} (Total: {target_total_epochs - start_epoch} epochs)...")

    current_epoch = start_epoch
    try:
        for epoch in range(start_epoch + 1, target_total_epochs + 1):
            current_epoch = epoch
            model.train()
            total_loss = 0.0
            total_correct, total_tokens = 0, 0
            true_note_total, note_detected_total, note_exact_total = 0, 0, 0
            ln_true_pos_total, ln_pred_pos_total, ln_hit_total = 0, 0, 0
            start_t = time.time()

            for batch_idx, batch in enumerate(train_loader):
                audio_mel = batch["audio_mel"].to(device)       # [B, T, 80]
                tokens = batch["tokens"].to(device)             # [B, T]
                B, T = tokens.shape
                ln_mask = batch["ln_mask"].to(device)           # [B, T, 4]
                star_rating = batch["star_rating"].to(device)   # [B, 1]
                multiscale_audio = batch["multiscale_audio"].to(device) if "multiscale_audio" in batch else None
                lane_states = batch["lane_states"].to(device) if "lane_states" in batch else None
                grid_metrics = batch["grid_metrics"].to(device) if "grid_metrics" in batch else None
                scaffold_tokens = batch["scaffold_tokens"].to(device) if "scaffold_tokens" in batch else None
                strain_metrics = batch["strain_metrics"].to(device) if "strain_metrics" in batch else None
                if scaffold_tokens is None or strain_metrics is None:
                    synth_scaffold, synth_strain = compute_multi_scenario_scaffold_and_strain(tokens, star_rating, grid_metrics, audio_mel)
                    if scaffold_tokens is None:
                        scaffold_tokens = synth_scaffold
                    if strain_metrics is None:
                        strain_metrics = synth_strain

                # Multi-Scenario History Injection (Scheduled Sampling with Model Error / Debris)
                # 50% Clean Gold History, 50% Noisy/Debris History (Trains model to recover from mistakes)
                prev_tok = torch.roll(tokens, shifts=1, dims=1)
                if "prev_tokens" in batch:
                    prev_tok[:, 0] = batch["prev_tokens"].to(device)[:, -1]
                else:
                    prev_tok[:, 0] = 0

                noise_history_mask = (torch.rand(B, T, device=device) < 0.35)
                # Roll history noise only on 60% of batches
                if torch.rand(1).item() < 0.60:
                    prev_tok = prev_tok.clone()
                    prev_tok[noise_history_mask] = torch.randint(0, 16, (noise_history_mask.sum(),), device=device)

                # Shifted lane states for autoregression
                if lane_states is not None:
                    prev_lane = torch.roll(lane_states, shifts=1, dims=1)
                    prev_lane[:, 0] = 0
                else:
                    prev_lane = None

                optimizer.zero_grad()
                chord_logits, ln_logits, pred_sr = model(
                    audio_mel,
                    star_rating=star_rating,
                    prev_tokens=prev_tok,
                    lane_states=prev_lane,
                    grid_metrics=grid_metrics,
                    scaffold_tokens=scaffold_tokens,
                    strain_metrics=strain_metrics,
                    multiscale_audio=multiscale_audio,
                    return_sr=True,
                )

                # 1. Standard Chord CrossEntropy with class balance
                B, T, C = chord_logits.shape
                loss_chord = criterion_chord(chord_logits.view(B * T, C), tokens.view(B * T))

                # 2. Strict 4-Lane Multi-Label Hamming Loss (heavily penalizes wrong lanes/columns)
                true_lanes = torch.stack([
                    (tokens & 1).gt(0).float(),
                    ((tokens >> 1) & 1).gt(0).float(),
                    ((tokens >> 2) & 1).gt(0).float(),
                    ((tokens >> 3) & 1).gt(0).float(),
                ], dim=-1)  # [B, T, 4]

                probs_16 = F.softmax(chord_logits, dim=-1)  # [B, T, 16]
                pred_lane_probs = torch.zeros_like(true_lanes)
                for bit in range(4):
                    bit_indices = [idx for idx in range(16) if (idx >> bit) & 1]
                    pred_lane_probs[..., bit] = probs_16[..., bit_indices].sum(dim=-1)

                loss_lane_bce = F.binary_cross_entropy(pred_lane_probs.clamp(1e-6, 1.0 - 1e-6), true_lanes)

                # 3. Weighted LN loss (missed or hallucinated long notes strictly penalized)
                loss_ln = criterion_ln(ln_logits, ln_mask)
                # 4. Difficulty consistency loss
                loss_sr = F.smooth_l1_loss(pred_sr, star_rating)

                loss = loss_chord + 2.0 * loss_lane_bce + 1.5 * loss_ln + 0.5 * loss_sr
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item() * B
                preds = chord_logits.argmax(dim=-1)
                total_correct += (preds == tokens).sum().item()
                total_tokens += (B * T)

                # Strict Honest Metrics (Active notes only, zero idle tick padding)
                is_true_note = (tokens > 0)
                true_note_cnt = is_true_note.sum().item()
                true_note_total += true_note_cnt
                if true_note_cnt > 0:
                    note_detected_total += (preds[is_true_note] > 0).sum().item()
                    note_exact_total += (preds[is_true_note] == tokens[is_true_note]).sum().item()

                # LN specific metrics
                is_ln_true = (ln_mask > 0.5)
                ln_pred_active = (torch.sigmoid(ln_logits) > 0.40)
                ln_true_pos_total += is_ln_true.sum().item()
                ln_pred_pos_total += ln_pred_active.sum().item()
                ln_hit_total += (ln_pred_active & is_ln_true).sum().item()

            train_loss = total_loss / len(train_loader.dataset)
            train_all_acc = (total_correct / max(1, total_tokens)) * 100.0
            train_note_recall = (note_detected_total / max(1, true_note_total)) * 100.0
            train_strict_exact = (note_exact_total / max(1, true_note_total)) * 100.0
            train_ln_recall = (ln_hit_total / max(1, ln_true_pos_total)) * 100.0
            train_ln_prec = (ln_hit_total / max(1, ln_pred_pos_total)) * 100.0
            train_ln_f1 = (2 * train_ln_prec * train_ln_recall) / max(1e-5, train_ln_prec + train_ln_recall)
            elapsed = time.time() - start_t
            samples_sec = len(train_loader.dataset) / max(0.001, elapsed)

            # Validation (Ruthless Honest Evaluation)
            model.eval()
            val_loss = 0.0
            val_correct, val_tokens = 0, 0
            val_true_note, val_note_det, val_note_exact = 0, 0, 0
            val_ln_true, val_ln_pred, val_ln_hit = 0, 0, 0
            val_sr_err = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    audio_mel = batch["audio_mel"].to(device)
                    tokens = batch["tokens"].to(device)
                    ln_mask = batch["ln_mask"].to(device)
                    star_rating = batch["star_rating"].to(device)
                    multiscale_audio = batch["multiscale_audio"].to(device) if "multiscale_audio" in batch else None
                    lane_states = batch["lane_states"].to(device) if "lane_states" in batch else None
                    grid_metrics = batch["grid_metrics"].to(device) if "grid_metrics" in batch else None
                    scaffold_tokens = batch["scaffold_tokens"].to(device) if "scaffold_tokens" in batch else None
                    strain_metrics = batch["strain_metrics"].to(device) if "strain_metrics" in batch else None
                    if scaffold_tokens is None or strain_metrics is None:
                        synth_scaffold, synth_strain = compute_multi_scenario_scaffold_and_strain(tokens, star_rating, grid_metrics, audio_mel)
                        if scaffold_tokens is None:
                            scaffold_tokens = synth_scaffold
                        if strain_metrics is None:
                            strain_metrics = synth_strain

                    prev_tok = torch.roll(tokens, shifts=1, dims=1)
                    if "prev_tokens" in batch:
                        prev_tok[:, 0] = batch["prev_tokens"].to(device)[:, -1]
                    else:
                        prev_tok[:, 0] = 0

                    if lane_states is not None:
                        prev_lane = torch.roll(lane_states, shifts=1, dims=1)
                        prev_lane[:, 0] = 0
                    else:
                        prev_lane = None

                    c_logits, l_logits, p_sr = model(
                        audio_mel,
                        star_rating=star_rating,
                        prev_tokens=prev_tok,
                        lane_states=prev_lane,
                        grid_metrics=grid_metrics,
                        scaffold_tokens=scaffold_tokens,
                        strain_metrics=strain_metrics,
                        multiscale_audio=multiscale_audio,
                        return_sr=True,
                    )
                    B, T, C = c_logits.shape
                    l_c = criterion_chord(c_logits.view(B * T, C), tokens.view(B * T))

                    v_true_lanes = torch.stack([
                        (tokens & 1).gt(0).float(),
                        ((tokens >> 1) & 1).gt(0).float(),
                        ((tokens >> 2) & 1).gt(0).float(),
                        ((tokens >> 3) & 1).gt(0).float(),
                    ], dim=-1)
                    v_probs_16 = F.softmax(c_logits, dim=-1)
                    v_pred_lane_probs = torch.zeros_like(v_true_lanes)
                    for bit in range(4):
                        bit_indices = [idx for idx in range(16) if (idx >> bit) & 1]
                        v_pred_lane_probs[..., bit] = v_probs_16[..., bit_indices].sum(dim=-1)

                    l_lane = F.binary_cross_entropy(v_pred_lane_probs.clamp(1e-6, 1.0 - 1e-6), v_true_lanes)
                    l_l = criterion_ln(l_logits, ln_mask)
                    l_sr = F.smooth_l1_loss(p_sr, star_rating)
                    val_loss += (l_c + 2.0 * l_lane + 1.5 * l_l + 0.5 * l_sr).item() * B
                    val_sr_err += torch.abs(p_sr - star_rating).sum().item()

                    preds = c_logits.argmax(dim=-1)
                    val_correct += (preds == tokens).sum().item()
                    val_tokens += (B * T)

                    is_true = (tokens > 0)
                    t_cnt = is_true.sum().item()
                    val_true_note += t_cnt
                    if t_cnt > 0:
                        val_note_det += (preds[is_true] > 0).sum().item()
                        val_note_exact += (preds[is_true] == tokens[is_true]).sum().item()

                    is_ln = (ln_mask > 0.5)
                    pred_ln = (torch.sigmoid(l_logits) > 0.40)
                    val_ln_true += is_ln.sum().item()
                    val_ln_pred += pred_ln.sum().item()
                    val_ln_hit += (pred_ln & is_ln).sum().item()

            val_loss /= len(val_loader.dataset)
            val_sr_mae = val_sr_err / len(val_loader.dataset)
            val_all_acc = (val_correct / max(1, val_tokens)) * 100.0
            val_note_recall = (val_note_det / max(1, val_true_note)) * 100.0
            val_strict_exact = (val_note_exact / max(1, val_true_note)) * 100.0
            val_ln_recall = (val_ln_hit / max(1, val_ln_true)) * 100.0
            val_ln_prec = (val_ln_hit / max(1, val_ln_pred)) * 100.0
            val_ln_f1 = (2 * val_ln_prec * val_ln_recall) / max(1e-5, val_ln_prec + val_ln_recall)

            print(
                f"Epoch {epoch:02d}/{target_total_epochs:02d} [{elapsed:.1f}s | {samples_sec:.1f} smp/s]\n"
                f"  Train: Loss={train_loss:.4f} | Note Rec={train_note_recall:.1f}% | Exact 4K={train_strict_exact:.1f}% | LN Rec={train_ln_recall:.1f}% (LN F1={train_ln_f1:.1f}%) | Raw Acc={train_all_acc:.1f}%\n"
                f"  Val:   Loss={val_loss:.4f} | Note Rec={val_note_recall:.1f}% | Exact 4K={val_strict_exact:.1f}% | LN Rec={val_ln_recall:.1f}% | SR MAE={val_sr_mae:.2f}* | Raw Acc={val_all_acc:.1f}%"
            )

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_note_recall": train_note_recall,
                "train_note_acc": train_strict_exact,
                "train_strict_exact": train_strict_exact,
                "train_ln_recall": train_ln_recall,
                "train_ln_f1": train_ln_f1,
                "val_loss": val_loss,
                "val_acc": val_all_acc,
                "val_note_recall": val_note_recall,
                "val_note_acc": val_strict_exact,
                "val_strict_exact": val_strict_exact,
                "val_ln_recall": val_ln_recall,
                "val_ln_f1": val_ln_f1,
            }
            history.append(record)

            # 1. Continuous checkpoint: Always save latest state to _last.pt
            model_to_save = model.module if hasattr(model, "module") else model
            last_ckpt_path = checkpoint_dir / "model_3_transformer_last.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model_to_save.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": cfg,
                "best_val_loss": best_val_loss,
                "val_loss": val_loss,
                "history": history,
            }, last_ckpt_path)

            # 2. If improved, save to _best.pt
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt_path = checkpoint_dir / "model_3_transformer_best.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state": model_to_save.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": cfg,
                    "best_val_loss": best_val_loss,
                    "val_loss": val_loss,
                    "history": history,
                }, best_ckpt_path)
                print(f"  ⭐ Saved new best checkpoint to '{best_ckpt_path}' (Val Loss: {val_loss:.4f})")

    except KeyboardInterrupt:
        print(f"\n🛑 [Paused] Training safely paused by user at Epoch {current_epoch}.")
        model_to_save = model.module if hasattr(model, "module") else model
        pause_path = checkpoint_dir / "model_3_transformer_last.pt"
        torch.save({
            "epoch": current_epoch,
            "model_state": model_to_save.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": cfg,
            "best_val_loss": best_val_loss,
            "history": history,
        }, pause_path)
        print(f"  💾 Current weights and optimizer state preserved in '{pause_path}'.")
        print(f"  ▶ To resume later without losing progress, run: python train.py --model 3 --resume")

    return {"model": "model_3_transformer", "best_val_loss": best_val_loss, "history": history}


def train_model_1_sync(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    checkpoint_dir: Path,
    is_song_level: bool = False,
    augment: bool = True,
    resume_path: Optional[str] = None,
    override_lr: Optional[float] = None,
    device_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Trains Model 1: SyncBPMOffsetModel (Audio tempo and beat synchronization).
    Features high-efficiency small-dataset training:
    - Pitch-invariant Speed/Tempo perturbation (alpha in [0.75, 1.35])
    - SpecAugment frequency band masking
    - Soft-Argmax Gaussian smoothed CrossEntropy over 301 BPM bins + Smooth L1
    - Cyclic Phase loss for downbeat offset (modulo beat length)
    - Full resume support with optional LR override.
    """
    print("\n" + "=" * 60)
    print("▶ TRAINING MODEL 1: SyncBPMOffsetModel (BPM & Offset Sync)")
    print(f"  Mode: {'Song-Level (First 16s audio)' if is_song_level else 'Musical Chunks'}")
    print(f"  Augmentation: {'Speed Perturbation [0.75..1.35x] + SpecAugment' if augment else 'Disabled'}")
    print("=" * 60)

    raw_model = SyncBPMOffsetModel(audio_dim=80, hidden_dim=256).to(device)
    model = wrap_model_multi_gpu(raw_model, device_ids or [])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    start_epoch = 0
    best_val_loss = float("inf")
    best_val_mae = float("inf")
    history: List[Dict[str, Any]] = []

    # Handle resume
    ckpt_file = find_resume_checkpoint(checkpoint_dir, "model_1_sync", resume_path)
    if ckpt_file:
        start_epoch, best_metric, ckpt_dict = load_checkpoint_state(
            ckpt_file, model, optimizer=optimizer, scheduler=scheduler, device=device, override_lr=override_lr
        )
        if "val_bpm_mae" in ckpt_dict:
            best_val_mae = float(ckpt_dict["val_bpm_mae"])
        elif best_metric < float("inf"):
            best_val_mae = best_metric
        if "history" in ckpt_dict and isinstance(ckpt_dict["history"], list):
            history = list(ckpt_dict["history"])
        print(f"  ▶ Resumed at Epoch {start_epoch}. Best BPM MAE so far: {best_val_mae:.2f}")

    target_total_epochs = epochs
    if target_total_epochs <= start_epoch:
        target_total_epochs = start_epoch + epochs
        print(f"  ℹ️ Target epochs ({epochs}) <= Loaded epoch ({start_epoch}). Extending total epochs to {target_total_epochs} (+{epochs} more epochs).")

    current_epoch = start_epoch
    try:
        for epoch in range(start_epoch + 1, target_total_epochs + 1):
            current_epoch = epoch
            model.train()
            total_loss, total_bpm_err, total_phase_err = 0.0, 0.0, 0.0
            start_t = time.time()

            for batch in train_loader:
                audio_mel = batch["audio_mel"].to(device)
                target_bpm = batch["bpm"].to(device).squeeze(-1)
                target_offset = batch["offset_ms"].to(device).squeeze(-1)

                if augment:
                    # Dynamic Pitch-Invariant Speed Perturbation: alpha in [0.80, 1.25]
                    alpha = float(torch.empty(1).uniform_(0.80, 1.25).item())
                    mel_t = audio_mel.transpose(1, 2)
                    orig_T = mel_t.shape[-1]
                    new_T = max(32, int(round(orig_T * alpha)))
                    mel_aug = F.interpolate(mel_t, size=new_T, mode="linear", align_corners=False).transpose(1, 2)
                    if new_T >= orig_T:
                        audio_mel = mel_aug[:, :orig_T, :]
                    else:
                        pad = torch.zeros(audio_mel.shape[0], orig_T - new_T, audio_mel.shape[2], device=device)
                        audio_mel = torch.cat([mel_aug, pad], dim=1)

                    target_bpm = torch.clamp(target_bpm / alpha, min=60.0, max=360.0)
                    target_offset = target_offset * alpha

                    # SpecAugment frequency masking
                    if torch.rand(1).item() > 0.35 and audio_mel.shape[-1] > 16:
                        f_start = int(torch.randint(0, audio_mel.shape[-1] - 8, (1,)).item())
                        audio_mel[:, :, f_start:f_start + 8] = 0.0

                optimizer.zero_grad()
                pred_bpm, pred_offset, bpm_logits = model(audio_mel, return_logits=True)
                pred_bpm = pred_bpm.squeeze(-1)
                pred_offset = pred_offset.squeeze(-1)

                # Soft Gaussian smoothed Cross-Entropy loss over 301 BPM bins
                bins = model.bpm_bins.unsqueeze(0)  # [1, 301]
                sigma = 1.0
                target_dist = torch.exp(- (bins - target_bpm.unsqueeze(1)) ** 2 / (2.0 * sigma ** 2))
                target_dist = target_dist / (target_dist.sum(dim=-1, keepdim=True) + 1e-8)
                loss_ce = - (target_dist * F.log_softmax(bpm_logits, dim=-1)).sum(dim=-1).mean()

                # Smooth L1 regression loss on continuous BPM
                loss_reg = F.smooth_l1_loss(pred_bpm, target_bpm)

                # Beat-Periodic Phase offset loss (modulo beat length)
                beat_len_true = 60000.0 / (target_bpm + 1e-5)
                phase_diff = (pred_offset - target_offset).abs() % beat_len_true
                phase_dist = torch.min(phase_diff, beat_len_true - phase_diff)
                loss_phase = (phase_dist / beat_len_true).mean()
                loss_offset = loss_phase + 0.0005 * F.smooth_l1_loss(pred_offset, target_offset)

                loss = loss_ce + 0.5 * loss_reg + 1.0 * loss_offset
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

                B = audio_mel.shape[0]
                total_loss += loss.item() * B
                total_bpm_err += torch.abs(pred_bpm - target_bpm).sum().item()
                total_phase_err += phase_dist.sum().item()

            scheduler.step()
            elapsed = time.time() - start_t
            train_loss = total_loss / len(train_loader.dataset)
            bpm_mae = total_bpm_err / len(train_loader.dataset)
            phase_mae = total_phase_err / len(train_loader.dataset)

            # Validation
            model.eval()
            val_loss, val_bpm_err, val_phase_err = 0.0, 0.0, 0.0
            with torch.no_grad():
                for batch in val_loader:
                    audio_mel = batch["audio_mel"].to(device)
                    target_bpm = batch["bpm"].to(device).squeeze(-1)
                    target_offset = batch["offset_ms"].to(device).squeeze(-1)
                    pred_bpm, pred_offset, bpm_logits = model(audio_mel, return_logits=True)
                    pred_bpm = pred_bpm.squeeze(-1)
                    pred_offset = pred_offset.squeeze(-1)

                    bins = model.bpm_bins.unsqueeze(0)
                    sigma = 1.0
                    target_dist = torch.exp(- (bins - target_bpm.unsqueeze(1)) ** 2 / (2.0 * sigma ** 2))
                    target_dist = target_dist / (target_dist.sum(dim=-1, keepdim=True) + 1e-8)
                    loss_ce = - (target_dist * F.log_softmax(bpm_logits, dim=-1)).sum(dim=-1).mean()
                    loss_reg = F.smooth_l1_loss(pred_bpm, target_bpm)

                    beat_len_true = 60000.0 / (target_bpm + 1e-5)
                    phase_diff = (pred_offset - target_offset).abs() % beat_len_true
                    phase_dist = torch.min(phase_diff, beat_len_true - phase_diff)
                    loss_phase = (phase_dist / beat_len_true).mean()
                    loss_offset = loss_phase + 0.0005 * F.smooth_l1_loss(pred_offset, target_offset)

                    B = audio_mel.shape[0]
                    val_loss += (loss_ce + 0.5 * loss_reg + 1.0 * loss_offset).item() * B
                    val_bpm_err += torch.abs(pred_bpm - target_bpm).sum().item()
                    val_phase_err += phase_dist.sum().item()

            val_loss /= len(val_loader.dataset)
            val_mae = val_bpm_err / len(val_loader.dataset)
            val_phase_mae = val_phase_err / len(val_loader.dataset)

            print(
                f"Epoch {epoch:02d}/{target_total_epochs:02d} [{elapsed:.1f}s] "
                f"Train Loss: {train_loss:.2f} (BPM MAE: {bpm_mae:.2f}, Phase: {phase_mae:.1f}ms) | "
                f"Val Loss: {val_loss:.2f} (BPM MAE: {val_mae:.2f}, Phase: {val_phase_mae:.1f}ms)"
            )

            history.append({"epoch": epoch, "val_loss": val_loss, "val_bpm_mae": val_mae, "val_phase_mae": val_phase_mae})

            # Save continuous latest checkpoint
            model_to_save = model.module if hasattr(model, "module") else model
            last_path = checkpoint_dir / "model_1_sync_last.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model_to_save.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_bpm_mae": val_mae,
                "val_phase_mae": val_phase_mae,
                "history": history,
            }, last_path)

            if val_mae < best_val_mae or val_loss < best_val_loss:
                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                best_val_loss = val_loss
                best_path = checkpoint_dir / "model_1_sync_best.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state": model_to_save.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "val_bpm_mae": val_mae,
                    "val_phase_mae": val_phase_mae,
                    "history": history,
                }, best_path)
                print(f"  ⭐ Saved best checkpoint to '{best_path}' (BPM MAE: {val_mae:.2f}, Phase MAE: {val_phase_mae:.1f}ms)")

    except KeyboardInterrupt:
        print(f"\n🛑 [Paused] Training paused by user at Epoch {current_epoch}.")
        model_to_save = model.module if hasattr(model, "module") else model
        pause_path = checkpoint_dir / "model_1_sync_last.pt"
        torch.save({
            "epoch": current_epoch,
            "model_state": model_to_save.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "val_bpm_mae": best_val_mae,
            "history": history,
        }, pause_path)
        print(f"  💾 Saved last progress state to '{pause_path}'.")
        print(f"  ▶ To resume later without losing progress: python train.py --model 1 --resume")

    return {"model": "model_1_sync", "best_val_loss": best_val_loss, "best_val_mae": best_val_mae, "history": history}


def train_model_2_dynamics(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    checkpoint_dir: Path,
    timing_source: str = "map",
    resume_path: Optional[str] = None,
    override_lr: Optional[float] = None,
    device_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Trains Model 2: DynamicsSVModel (Slider Velocity & Timing changes).
    - timing_source="map" (Default): Takes reference SV curves and timing points directly from .osu beatmap,
      isolating dynamic modeling from any Model 1 errors.
    - resume_path: Resumes training exactly from previous checkpoint.
    """
    print("\n" + "=" * 60)
    print("▶ TRAINING MODEL 2: DynamicsSVModel (Slider Velocity & Zones)")
    source_desc = "ORIGINAL BEATMAP GROUND TRUTH (Direct from .osu — exact green lines)" if timing_source == "map" else "MODEL 1 CASCADED TIMING (AI Sync Inferred)"
    print(f"  Timing & Data Source: {source_desc}")
    print("=" * 60)

    raw_model = DynamicsSVModel(audio_dim=80, hidden_dim=128).to(device)
    model = wrap_model_multi_gpu(raw_model, device_ids or [])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_epoch = 0
    best_val_loss = float("inf")
    history: List[Dict[str, Any]] = []

    ckpt_file = find_resume_checkpoint(checkpoint_dir, "model_2_dynamics", resume_path)
    if ckpt_file:
        start_epoch, best_metric, ckpt_dict = load_checkpoint_state(
            ckpt_file, model, optimizer=optimizer, device=device, override_lr=override_lr
        )
        if best_metric < float("inf"):
            best_val_loss = best_metric
        if "history" in ckpt_dict and isinstance(ckpt_dict["history"], list):
            history = list(ckpt_dict["history"])
        print(f"  ▶ Resumed at Epoch {start_epoch}. Best SV MSE so far: {best_val_loss:.4f}")

    target_total_epochs = epochs
    if target_total_epochs <= start_epoch:
        target_total_epochs = start_epoch + epochs
        print(f"  ℹ️ Target epochs ({epochs}) <= Loaded epoch ({start_epoch}). Extending total epochs to {target_total_epochs} (+{epochs} more epochs).")

    current_epoch = start_epoch
    try:
        for epoch in range(start_epoch + 1, target_total_epochs + 1):
            current_epoch = epoch
            model.train()
            total_loss = 0.0
            start_t = time.time()

            for batch in train_loader:
                audio_mel = batch["audio_mel"].to(device)
                target_sv = batch["sv_curve"].to(device).unsqueeze(-1)  # [B, T, 1]

                optimizer.zero_grad()
                pred_sv, _ = model(audio_mel)
                loss = F.mse_loss(pred_sv, target_sv)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * audio_mel.shape[0]

            train_loss = total_loss / len(train_loader.dataset)
            elapsed = time.time() - start_t

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    audio_mel = batch["audio_mel"].to(device)
                    target_sv = batch["sv_curve"].to(device).unsqueeze(-1)
                    pred_sv, _ = model(audio_mel)
                    val_loss += F.mse_loss(pred_sv, target_sv).item() * audio_mel.shape[0]

            val_loss /= len(val_loader.dataset)
            print(f"Epoch {epoch:02d}/{target_total_epochs:02d} [{elapsed:.1f}s] Train SV MSE: {train_loss:.4f} | Val SV MSE: {val_loss:.4f}")
            history.append({"epoch": epoch, "val_loss": val_loss})

            model_to_save = model.module if hasattr(model, "module") else model
            last_path = checkpoint_dir / "model_2_dynamics_last.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model_to_save.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "history": history,
            }, last_path)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = checkpoint_dir / "model_2_dynamics_best.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state": model_to_save.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "history": history,
                }, best_path)
                print(f"  ⭐ Saved best checkpoint to '{best_path}'")

    except KeyboardInterrupt:
        print(f"\n🛑 [Paused] Training paused by user at Epoch {current_epoch}.")
        model_to_save = model.module if hasattr(model, "module") else model
        pause_path = checkpoint_dir / "model_2_dynamics_last.pt"
        torch.save({
            "epoch": current_epoch,
            "model_state": model_to_save.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "history": history,
        }, pause_path)
        print(f"  💾 Saved last progress state to '{pause_path}'.")
        print(f"  ▶ To resume later: python train.py --model 2 --resume")

    return {"model": "model_2_dynamics", "best_val_loss": best_val_loss, "history": history}


def train_model_4_evaluator(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    checkpoint_dir: Path,
    resume_path: Optional[str] = None,
    override_lr: Optional[float] = None,
    device_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Trains Model 4: DifficultyEvaluatorModel (Star Rating compliance validation)."""
    print("\n" + "=" * 60)
    print("▶ TRAINING MODEL 4: DifficultyEvaluatorModel (Star Rating Validator)")
    print("=" * 60)

    raw_model = DifficultyEvaluatorModel(num_classes=16, token_dim=64, hidden_dim=128).to(device)
    model = wrap_model_multi_gpu(raw_model, device_ids or [])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_epoch = 0
    best_val_loss = float("inf")
    history: List[Dict[str, Any]] = []

    ckpt_file = find_resume_checkpoint(checkpoint_dir, "model_4_evaluator", resume_path)
    if ckpt_file:
        start_epoch, best_metric, ckpt_dict = load_checkpoint_state(
            ckpt_file, model, optimizer=optimizer, device=device, override_lr=override_lr
        )
        if best_metric < float("inf"):
            best_val_loss = best_metric
        if "history" in ckpt_dict and isinstance(ckpt_dict["history"], list):
            history = list(ckpt_dict["history"])
        print(f"  ▶ Resumed at Epoch {start_epoch}. Best Val Loss so far: {best_val_loss:.4f}")

    target_total_epochs = epochs
    if target_total_epochs <= start_epoch:
        target_total_epochs = start_epoch + epochs
        print(f"  ℹ️ Target epochs ({epochs}) <= Loaded epoch ({start_epoch}). Extending total epochs to {target_total_epochs} (+{epochs} more epochs).")

    current_epoch = start_epoch
    try:
        for epoch in range(start_epoch + 1, target_total_epochs + 1):
            current_epoch = epoch
            model.train()
            total_loss, total_err = 0.0, 0.0
            start_t = time.time()

            for batch in train_loader:
                tokens = batch["tokens"].to(device)
                target_sr = batch["star_rating"].to(device)

                optimizer.zero_grad()
                pred_sr = model(tokens)
                loss = F.mse_loss(pred_sr, target_sr)
                loss.backward()
                optimizer.step()

                B = tokens.shape[0]
                total_loss += loss.item() * B
                total_err += torch.abs(pred_sr - target_sr).sum().item()

            elapsed = time.time() - start_t
            train_loss = total_loss / len(train_loader.dataset)
            train_mae = total_err / len(train_loader.dataset)

            model.eval()
            val_loss, val_err = 0.0, 0.0
            with torch.no_grad():
                for batch in val_loader:
                    tokens = batch["tokens"].to(device)
                    target_sr = batch["star_rating"].to(device)
                    pred_sr = model(tokens)
                    B = tokens.shape[0]
                    val_loss += F.mse_loss(pred_sr, target_sr).item() * B
                    val_err += torch.abs(pred_sr - target_sr).sum().item()

            val_loss /= len(val_loader.dataset)
            val_mae = val_err / len(val_loader.dataset)

            print(
                f"Epoch {epoch:02d}/{target_total_epochs:02d} [{elapsed:.1f}s] "
                f"Train SR MSE: {train_loss:.4f} (MAE: {train_mae:.2f}) | "
                f"Val SR MSE: {val_loss:.4f} (MAE: {val_mae:.2f})"
            )
            history.append({"epoch": epoch, "val_sr_mae": val_mae})

            model_to_save = model.module if hasattr(model, "module") else model
            last_path = checkpoint_dir / "model_4_evaluator_last.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model_to_save.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "history": history,
            }, last_path)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = checkpoint_dir / "model_4_evaluator_best.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state": model_to_save.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "history": history,
                }, best_path)
                print(f"  ⭐ Saved best checkpoint to '{best_path}'")

    except KeyboardInterrupt:
        print(f"\n🛑 [Paused] Training paused by user at Epoch {current_epoch}.")
        model_to_save = model.module if hasattr(model, "module") else model
        pause_path = checkpoint_dir / "model_4_evaluator_last.pt"
        torch.save({
            "epoch": current_epoch,
            "model_state": model_to_save.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "history": history,
        }, pause_path)
        print(f"  💾 Saved last progress state to '{pause_path}'.")
        print(f"  ▶ To resume later: python train.py --model 4 --resume")

    return {"model": "model_4_evaluator", "best_val_loss": best_val_loss, "history": history}


def main():
    parser = argparse.ArgumentParser(description="Train osu!mania 4K AI models on CPU, Single GPU, or Multi-GPU with seamless checkpoint resuming.")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="3",
        help="Model choice: 1 (sync), 2 (dynamics), 3 (transformer), 4 (evaluator), or all",
    )
    parser.add_argument("--device", "-d", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--gpus", type=str, default="auto", help="GPUs to target: 'all', 'auto', '0,1,2', or '0'")
    parser.add_argument("--multi_gpu", action="store_true", help="Enable Multi-GPU Parallel Acceleration across all detected GPUs (PyTorch DataParallel)")
    parser.add_argument("--dataset", type=str, default="dataset_processed/train_data.pt")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--timing_source", "-t", type=str, choices=["map", "model1"], default="map", help="Data/Timing source for Model 2 & 3: 'map' (Original beatmap ground truth) or 'model1' (Cascaded AI Sync)")
    parser.add_argument("--ground_truth", "--use_map_data", action="store_true", help="Force using pure ground truth data directly from beatmap (.osu)")
    parser.add_argument("--sync_checkpoint", type=str, default=None, help="Optional Model 1 checkpoint path when timing_source=model1")
    parser.add_argument("--epochs", "-e", type=int, default=15, help="Number of epochs to train (or additional epochs if resuming)")
    parser.add_argument("--batch_size", "-b", type=int, default=16)
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: 2e-4 for fresh training, or custom override on resume)")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument(
        "--context_dropout",
        type=float,
        default=0.20,
        help="Probability of dropping/zeroing history tokens during Model 3 training to mitigate cold-start (default 0.20).",
    )
    parser.add_argument(
        "--resume",
        "-r",
        nargs="?",
        const="auto",
        default=None,
        help="Resume training from previous checkpoint without losing momentum/progress. Optionally pass specific checkpoint path.",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Explicit path to a checkpoint (.pt) to resume training from.",
    )
    args = parser.parse_args()

    timing_source = "map" if args.ground_truth else args.timing_source
    resume_target = args.resume_from if args.resume_from is not None else args.resume

    # Resolve learning rate
    user_lr = args.lr
    effective_lr = user_lr if user_lr is not None else 2e-4
    override_lr_on_resume = user_lr  # None preserves checkpoint LR, float overrides it

    # Create checkpoints dir
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Resolve device & Multi-GPU
    device, device_ids = get_device(args.device, gpus=args.gpus, multi_gpu=args.multi_gpu)

    # Load dataset with Stratified Difficulty Splitting
    full_dataset = ManiaDataset(args.dataset)
    train_set, val_set = stratified_difficulty_split(full_dataset, val_split=args.val_split)
    train_size = len(train_set)
    val_size = len(val_set)

    # Dataloaders (num_workers=0 for instant in-memory access and cross-platform compatibility)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    lr_info = f"{effective_lr} (override on resume)" if (resume_target and user_lr is not None) else (f"{effective_lr} (or from checkpoint)" if resume_target else f"{effective_lr}")
    print(f"\n[Training Configuration]")
    print(f"  Model target:      {args.model}")
    print(f"  Primary Device:    {device}")
    if len(device_ids) > 1:
        print(f"  Multi-GPU Engines: {device_ids} (Parallel Training active)")
    print(f"  Data Source (M2/3):{timing_source.upper()} ({'Original Map Ground Truth' if timing_source == 'map' else 'Model 1 AI Sync'})")
    print(f"  Resume Mode:       {'ENABLED (' + str(resume_target) + ')' if resume_target else 'Disabled (Fresh start)'}")
    print(f"  Train samples:     {train_size}")
    print(f"  Validation samples:{val_size}")
    print(f"  Batch size:        {args.batch_size}")
    print(f"  Epochs:            {args.epochs}")
    print(f"  Learning rate:     {lr_info}")
    print(f"  Checkpoints dir:   {ckpt_dir.resolve()}")

    target = str(args.model).lower()
    if target in ("1", "sync", "all"):
        is_song = "song_sync" in str(args.dataset).lower()
        train_model_1_sync(train_loader, val_loader, device, args.epochs, effective_lr, ckpt_dir, is_song_level=is_song, resume_path=resume_target, override_lr=override_lr_on_resume, device_ids=device_ids)
    if target in ("2", "dynamics", "all"):
        train_model_2_dynamics(train_loader, val_loader, device, args.epochs, effective_lr, ckpt_dir, timing_source=timing_source, resume_path=resume_target, override_lr=override_lr_on_resume, device_ids=device_ids)
    if target in ("3", "transformer", "pattern", "all"):
        train_model_3_transformer(train_loader, val_loader, device, args.epochs, effective_lr, ckpt_dir, timing_source=timing_source, resume_path=resume_target, context_dropout=args.context_dropout, override_lr=override_lr_on_resume, device_ids=device_ids)
    if target in ("4", "evaluator", "sr", "all"):
        train_model_4_evaluator(train_loader, val_loader, device, args.epochs, effective_lr, ckpt_dir, resume_path=resume_target, override_lr=override_lr_on_resume, device_ids=device_ids)

    print("\n✅ Training procedure completed successfully!")


if __name__ == "__main__":
    main()
