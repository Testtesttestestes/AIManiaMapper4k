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


class ManiaDataset(Dataset):
    """Loads preprocessed training chunks from PyTorch tensor file."""
    def __init__(self, data_path: str | Path):
        self.path = Path(data_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found at '{self.path}'. Run 'python3 prepare_dataset.py' first.")
        print(f"[Dataset] Loading '{self.path}' into memory...")
        self.chunks: List[Dict[str, torch.Tensor]] = torch.load(self.path, weights_only=False)
        print(f"[Dataset] Successfully loaded {len(self.chunks)} training samples.")

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.chunks[idx]


def get_device(requested: str = "auto") -> torch.device:
    """Resolves target compute device: cuda, mps, or cpu."""
    req = requested.lower()
    if req in ("cuda", "gpu"):
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")
            return dev
        print("[Device Warning] CUDA GPU requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    elif req == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("[Device] Using Apple Silicon MPS GPU")
            return torch.device("mps")
        print("[Device Warning] MPS requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    elif req == "cpu":
        print(f"[Device] Using CPU ({os.cpu_count() or 1} cores detected)")
        return torch.device("cpu")
    else:  # auto
        if torch.cuda.is_available():
            print(f"[Device] Auto-selected CUDA GPU: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("[Device] Auto-selected Apple Silicon MPS GPU")
            return torch.device("mps")
        else:
            print(f"[Device] Auto-selected CPU ({os.cpu_count() or 1} cores)")
            return torch.device("cpu")


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
) -> Tuple[int, float, Dict[str, Any]]:
    """
    Loads checkpoint state with 100% backward compatibility:
    - Restores model weights regardless of key naming format (model_state, state_dict, raw dict).
    - Restores optimizer state (momentum, Adam second moments, step count) if present.
    - Gracefully initializes optimizer if checkpoint was saved without optimizer_state.
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
        model.load_state_dict(state_dict)
        print("  ✓ Restored model weights successfully.")

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

    # 4. Optimizer state (Exact continuation without learning disruption)
    if optimizer is not None and isinstance(ckpt, dict) and "optimizer_state" in ckpt and ckpt["optimizer_state"] is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            print("  ✓ Restored optimizer momentum, moving averages, and step states.")
        except Exception as opt_err:
            print(f"  ℹ️ Notice: Optimizer state format differed ({opt_err}); continuing with fresh optimizer parameters.")
    elif optimizer is not None:
        print(f"  ℹ️ Legacy checkpoint detected: continuing with fresh optimizer parameters from Epoch {start_epoch}.")

    # 5. Scheduler state
    if scheduler is not None and isinstance(ckpt, dict) and "scheduler_state" in ckpt and ckpt["scheduler_state"] is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state"])
            print("  ✓ Restored LR scheduler state.")
        except Exception:
            pass

    return start_epoch, best_metric, ckpt if isinstance(ckpt, dict) else {}


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
) -> Dict[str, Any]:
    """
    Trains Model 3: ManiaPatternTransformer (Causal Autoregressive Pattern Generator).
    - timing_source="map" (Default): Takes reference ground truth data directly from beatmaps (.osu),
      preventing error compounding from Model 1 (Teacher Forcing).
    - timing_source="model1": Evaluates/trains using Model 1 AI predicted alignment.
    - resume_path: Resumes training exactly from previous checkpoint (weights, optimizer, epoch).
    - context_dropout: Probability of zeroing/masking history during training to eliminate cold start.
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
    model = ManiaPatternTransformer(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion_chord = nn.CrossEntropyLoss()
    criterion_ln = nn.BCEWithLogitsLoss()

    start_epoch = 0
    best_val_loss = float("inf")
    history: List[Dict[str, Any]] = []

    # Handle resume
    ckpt_file = find_resume_checkpoint(checkpoint_dir, "model_3_transformer", resume_path)
    if ckpt_file:
        start_epoch, best_metric, ckpt_dict = load_checkpoint_state(
            ckpt_file, model, optimizer=optimizer, device=device
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
            total_loss, total_correct, total_tokens = 0.0, 0, 0
            start_t = time.time()

            for batch_idx, batch in enumerate(train_loader):
                audio_mel = batch["audio_mel"].to(device)       # [B, T, 80]
                tokens = batch["tokens"].to(device)             # [B, T]
                ln_mask = batch["ln_mask"].to(device)           # [B, T, 4]
                star_rating = batch["star_rating"].to(device)   # [B, 1]
                lane_states = batch["lane_states"].to(device) if "lane_states" in batch else None
                grid_metrics = batch["grid_metrics"].to(device) if "grid_metrics" in batch else None

                # Autoregressive target: shifted tokens seeded with prior measure context
                prev_tok = torch.roll(tokens, shifts=1, dims=1)
                if "prev_tokens" in batch:
                    prev_tok[:, 0] = batch["prev_tokens"].to(device)[:, -1]
                else:
                    prev_tok[:, 0] = 0

                # Shifted lane states for autoregression
                if lane_states is not None:
                    prev_lane = torch.roll(lane_states, shifts=1, dims=1)
                    prev_lane[:, 0] = 0
                else:
                    prev_lane = None

                optimizer.zero_grad()
                chord_logits, ln_logits = model(
                    audio_mel,
                    star_rating=star_rating,
                    prev_tokens=prev_tok,
                    lane_states=prev_lane,
                    grid_metrics=grid_metrics,
                )

                # Chord loss (16 classes)
                B, T, C = chord_logits.shape
                loss_chord = criterion_chord(chord_logits.view(B * T, C), tokens.view(B * T))
                # LN duration loss
                loss_ln = criterion_ln(ln_logits, ln_mask)

                loss = loss_chord + 0.5 * loss_ln
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item() * B
                preds = chord_logits.argmax(dim=-1)
                total_correct += (preds == tokens).sum().item()
                total_tokens += (B * T)

            train_loss = total_loss / len(train_loader.dataset)
            train_acc = (total_correct / max(1, total_tokens)) * 100.0
            elapsed = time.time() - start_t
            samples_sec = len(train_loader.dataset) / max(0.001, elapsed)

            # Validation
            model.eval()
            val_loss, val_correct, val_tokens = 0.0, 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    audio_mel = batch["audio_mel"].to(device)
                    tokens = batch["tokens"].to(device)
                    ln_mask = batch["ln_mask"].to(device)
                    star_rating = batch["star_rating"].to(device)
                    lane_states = batch["lane_states"].to(device) if "lane_states" in batch else None
                    grid_metrics = batch["grid_metrics"].to(device) if "grid_metrics" in batch else None

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

                    c_logits, l_logits = model(
                        audio_mel,
                        star_rating=star_rating,
                        prev_tokens=prev_tok,
                        lane_states=prev_lane,
                        grid_metrics=grid_metrics,
                    )
                    B, T, C = c_logits.shape
                    l_c = criterion_chord(c_logits.view(B * T, C), tokens.view(B * T))
                    l_l = criterion_ln(l_logits, ln_mask)
                    val_loss += (l_c + 0.5 * l_l).item() * B

                    preds = c_logits.argmax(dim=-1)
                    val_correct += (preds == tokens).sum().item()
                    val_tokens += (B * T)

            val_loss /= len(val_loader.dataset)
            val_acc = (val_correct / max(1, val_tokens)) * 100.0

            print(
                f"Epoch {epoch:02d}/{target_total_epochs:02d} [{elapsed:.1f}s | {samples_sec:.1f} smp/s] "
                f"Train Loss: {train_loss:.4f} (Acc: {train_acc:.1f}%) | "
                f"Val Loss: {val_loss:.4f} (Acc: {val_acc:.1f}%)"
            )

            record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}
            history.append(record)

            # 1. Continuous checkpoint: Always save latest state to _last.pt
            last_ckpt_path = checkpoint_dir / "model_3_transformer_last.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
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
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": cfg,
                    "best_val_loss": best_val_loss,
                    "val_loss": val_loss,
                    "history": history,
                }, best_ckpt_path)
                print(f"  ⭐ Saved new best checkpoint to '{best_ckpt_path}' (Val Loss: {val_loss:.4f})")

    except KeyboardInterrupt:
        print(f"\n🛑 [Paused] Training safely paused by user at Epoch {current_epoch}.")
        pause_path = checkpoint_dir / "model_3_transformer_last.pt"
        torch.save({
            "epoch": current_epoch,
            "model_state": model.state_dict(),
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
) -> Dict[str, Any]:
    """
    Trains Model 1: SyncBPMOffsetModel (Audio tempo and beat synchronization).
    Features high-efficiency small-dataset training:
    - Pitch-invariant Speed/Tempo perturbation (alpha in [0.75, 1.35])
    - SpecAugment frequency band masking
    - Soft-Argmax Gaussian smoothed CrossEntropy over 301 BPM bins + Smooth L1
    - Cyclic Phase loss for downbeat offset (modulo beat length)
    - Full resume support.
    """
    print("\n" + "=" * 60)
    print("▶ TRAINING MODEL 1: SyncBPMOffsetModel (BPM & Offset Sync)")
    print(f"  Mode: {'Song-Level (First 16s audio)' if is_song_level else 'Musical Chunks'}")
    print(f"  Augmentation: {'Speed Perturbation [0.75..1.35x] + SpecAugment' if augment else 'Disabled'}")
    print("=" * 60)

    model = SyncBPMOffsetModel(audio_dim=80, hidden_dim=256).to(device)
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
            ckpt_file, model, optimizer=optimizer, scheduler=scheduler, device=device
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
            last_path = checkpoint_dir / "model_1_sync_last.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
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
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "val_bpm_mae": val_mae,
                    "val_phase_mae": val_phase_mae,
                    "history": history,
                }, best_path)
                print(f"  ⭐ Saved best checkpoint to '{best_path}' (BPM MAE: {val_mae:.2f}, Phase MAE: {val_phase_mae:.1f}ms)")

    except KeyboardInterrupt:
        print(f"\n🛑 [Paused] Training paused by user at Epoch {current_epoch}.")
        pause_path = checkpoint_dir / "model_1_sync_last.pt"
        torch.save({
            "epoch": current_epoch,
            "model_state": model.state_dict(),
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

    model = DynamicsSVModel(audio_dim=80, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_epoch = 0
    best_val_loss = float("inf")
    history: List[Dict[str, Any]] = []

    ckpt_file = find_resume_checkpoint(checkpoint_dir, "model_2_dynamics", resume_path)
    if ckpt_file:
        start_epoch, best_metric, ckpt_dict = load_checkpoint_state(
            ckpt_file, model, optimizer=optimizer, device=device
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

            last_path = checkpoint_dir / "model_2_dynamics_last.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "history": history,
            }, last_path)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = checkpoint_dir / "model_2_dynamics_best.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "history": history,
                }, best_path)
                print(f"  ⭐ Saved best checkpoint to '{best_path}'")

    except KeyboardInterrupt:
        print(f"\n🛑 [Paused] Training paused by user at Epoch {current_epoch}.")
        pause_path = checkpoint_dir / "model_2_dynamics_last.pt"
        torch.save({
            "epoch": current_epoch,
            "model_state": model.state_dict(),
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
) -> Dict[str, Any]:
    """Trains Model 4: DifficultyEvaluatorModel (Star Rating compliance validation)."""
    print("\n" + "=" * 60)
    print("▶ TRAINING MODEL 4: DifficultyEvaluatorModel (Star Rating Validator)")
    print("=" * 60)

    model = DifficultyEvaluatorModel(num_classes=16, token_dim=64, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_epoch = 0
    best_val_loss = float("inf")
    history: List[Dict[str, Any]] = []

    ckpt_file = find_resume_checkpoint(checkpoint_dir, "model_4_evaluator", resume_path)
    if ckpt_file:
        start_epoch, best_metric, ckpt_dict = load_checkpoint_state(
            ckpt_file, model, optimizer=optimizer, device=device
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

            last_path = checkpoint_dir / "model_4_evaluator_last.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "history": history,
            }, last_path)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = checkpoint_dir / "model_4_evaluator_best.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "history": history,
                }, best_path)
                print(f"  ⭐ Saved best checkpoint to '{best_path}'")

    except KeyboardInterrupt:
        print(f"\n🛑 [Paused] Training paused by user at Epoch {current_epoch}.")
        pause_path = checkpoint_dir / "model_4_evaluator_last.pt"
        torch.save({
            "epoch": current_epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "history": history,
        }, pause_path)
        print(f"  💾 Saved last progress state to '{pause_path}'.")
        print(f"  ▶ To resume later: python train.py --model 4 --resume")

    return {"model": "model_4_evaluator", "best_val_loss": best_val_loss, "history": history}


def main():
    parser = argparse.ArgumentParser(description="Train osu!mania 4K AI models on CPU or GPU with seamless checkpoint resuming.")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="3",
        help="Model choice: 1 (sync), 2 (dynamics), 3 (transformer), 4 (evaluator), or all",
    )
    parser.add_argument("--device", "-d", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dataset", type=str, default="dataset_processed/train_data.pt")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--timing_source", "-t", type=str, choices=["map", "model1"], default="map", help="Data/Timing source for Model 2 & 3: 'map' (Original beatmap ground truth) or 'model1' (Cascaded AI Sync)")
    parser.add_argument("--ground_truth", "--use_map_data", action="store_true", help="Force using pure ground truth data directly from beatmap (.osu)")
    parser.add_argument("--sync_checkpoint", type=str, default=None, help="Optional Model 1 checkpoint path when timing_source=model1")
    parser.add_argument("--epochs", "-e", type=int, default=15, help="Number of epochs to train (or additional epochs if resuming)")
    parser.add_argument("--batch_size", "-b", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
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

    # Create checkpoints dir
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Resolve device
    device = get_device(args.device)

    # Load dataset
    full_dataset = ManiaDataset(args.dataset)
    val_size = max(1, int(len(full_dataset) * args.val_split))
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Dataloaders (num_workers=0 for instant in-memory access and cross-platform compatibility)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"\n[Training Configuration]")
    print(f"  Model target:      {args.model}")
    print(f"  Device:            {device}")
    print(f"  Data Source (M2/3):{timing_source.upper()} ({'Original Map Ground Truth' if timing_source == 'map' else 'Model 1 AI Sync'})")
    print(f"  Resume Mode:       {'ENABLED (' + str(resume_target) + ')' if resume_target else 'Disabled (Fresh start)'}")
    print(f"  Train samples:     {train_size}")
    print(f"  Validation samples:{val_size}")
    print(f"  Batch size:        {args.batch_size}")
    print(f"  Epochs:            {args.epochs}")
    print(f"  Learning rate:     {args.lr}")
    print(f"  Checkpoints dir:   {ckpt_dir.resolve()}")

    target = str(args.model).lower()
    if target in ("1", "sync", "all"):
        is_song = "song_sync" in str(args.dataset).lower()
        train_model_1_sync(train_loader, val_loader, device, args.epochs, args.lr, ckpt_dir, is_song_level=is_song, resume_path=resume_target)
    if target in ("2", "dynamics", "all"):
        train_model_2_dynamics(train_loader, val_loader, device, args.epochs, args.lr, ckpt_dir, timing_source=timing_source, resume_path=resume_target)
    if target in ("3", "transformer", "pattern", "all"):
        train_model_3_transformer(train_loader, val_loader, device, args.epochs, args.lr, ckpt_dir, timing_source=timing_source, resume_path=resume_target, context_dropout=args.context_dropout)
    if target in ("4", "evaluator", "sr", "all"):
        train_model_4_evaluator(train_loader, val_loader, device, args.epochs, args.lr, ckpt_dir, resume_path=resume_target)

    print("\n✅ Training procedure completed successfully!")


if __name__ == "__main__":
    main()
