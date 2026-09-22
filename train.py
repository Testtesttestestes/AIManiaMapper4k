#!/usr/bin/env python3
"""
Unified PyTorch training script for osu!mania 4K AI models.

Supports training any of the 4 models defined in the specification:
  - Model 1 (sync):      SyncBPMOffsetModel (BPM & audio offset synchronization)
  - Model 2 (dynamics):  DynamicsSVModel (Slider Velocity & Timing Zones)
  - Model 3 (transformer): ManiaPatternTransformer (4K Hit Object pattern generator)
  - Model 4 (evaluator): DifficultyEvaluatorModel (Star Rating & density validation)
  - all:                 Train all models sequentially

Execution devices:
  - CPU: Multi-threaded CPU execution (torch.set_num_threads)
  - GPU: NVIDIA CUDA (with automatic mixed precision fp16/bf16) or Apple Silicon MPS
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    snap_offset_to_audio_transients,
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


def train_model_3_transformer(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    checkpoint_dir: Path,
) -> Dict[str, Any]:
    """Trains Model 3: ManiaPatternTransformer (Causal Autoregressive Pattern Generator)."""
    print("\n" + "=" * 60)
    print("▶ TRAINING MODEL 3: ManiaPatternTransformer (Hit Object Synthesis)")
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
    )
    model = ManiaPatternTransformer(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion_chord = nn.CrossEntropyLoss()
    criterion_ln = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_correct, total_tokens = 0.0, 0, 0
        start_t = time.time()

        for batch_idx, batch in enumerate(train_loader):
            audio_mel = batch["audio_mel"].to(device)       # [B, T, 80]
            tokens = batch["tokens"].to(device)             # [B, T]
            ln_mask = batch["ln_mask"].to(device)           # [B, T, 4]
            star_rating = batch["star_rating"].to(device)   # [B, 1]

            # Autoregressive target: input is shifted tokens seeded with prior square/measure context
            prev_tok = torch.roll(tokens, shifts=1, dims=1)
            if "prev_tokens" in batch:
                prev_tok[:, 0] = batch["prev_tokens"].to(device)[:, -1]
            else:
                prev_tok[:, 0] = 0

            optimizer.zero_grad()
            chord_logits, ln_logits = model(audio_mel, star_rating=star_rating, prev_tokens=prev_tok)

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

                prev_tok = torch.roll(tokens, shifts=1, dims=1)
                if "prev_tokens" in batch:
                    prev_tok[:, 0] = batch["prev_tokens"].to(device)[:, -1]
                else:
                    prev_tok[:, 0] = 0

                c_logits, l_logits = model(audio_mel, star_rating=star_rating, prev_tokens=prev_tok)
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
            f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s | {samples_sec:.1f} smp/s] "
            f"Train Loss: {train_loss:.4f} (Acc: {train_acc:.1f}%) | "
            f"Val Loss: {val_loss:.4f} (Acc: {val_acc:.1f}%)"
        )

        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}
        history.append(record)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = checkpoint_dir / "model_3_transformer_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "config": cfg,
                "best_val_loss": best_val_loss,
            }, ckpt_path)
            print(f"  ⭐ Saved new best checkpoint to '{ckpt_path}' (Val Loss: {val_loss:.4f})")

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
) -> Dict[str, Any]:
    """
    Trains Model 1: SyncBPMOffsetModel (Audio tempo and beat synchronization).
    Features high-efficiency small-dataset training:
    - Pitch-invariant Speed/Tempo perturbation (alpha in [0.75, 1.35])
    - SpecAugment frequency band masking
    - Soft-Argmax Gaussian smoothed CrossEntropy over 301 BPM bins + Smooth L1
    - Cyclic Phase loss for downbeat offset (modulo beat length)
    """
    print("\n" + "=" * 60)
    print("▶ TRAINING MODEL 1: SyncBPMOffsetModel (BPM & Offset Sync)")
    print(f"  Mode: {'Song-Level (First 16s audio)' if is_song_level else 'Musical Chunks'}")
    print(f"  Augmentation: {'Speed Perturbation [0.75..1.35x] + SpecAugment' if augment else 'Disabled'}")
    print("=" * 60)

    model = SyncBPMOffsetModel(audio_dim=80, hidden_dim=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    best_val_mae = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_bpm_err, total_phase_err = 0.0, 0.0, 0.0
        start_t = time.time()

        for batch in train_loader:
            audio_mel = batch["audio_mel"].to(device)
            target_bpm = batch["bpm"].to(device).squeeze(-1)
            target_offset = batch["offset_ms"].to(device).squeeze(-1)

            if augment:
                # 1. Dynamic Pitch-Invariant Speed Perturbation: alpha in [0.80, 1.25]
                # Stretching audio in time (alpha > 1) makes music SLOWER, so BPM decreases (BPM / alpha).
                # Compressing audio in time (alpha < 1) makes music FASTER, so BPM increases (BPM / alpha).
                alpha = float(torch.empty(1).uniform_(0.80, 1.25).item())
                # Resample along time axis: [B, T, 80] -> [B, 80, T]
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

                # 2. SpecAugment frequency masking
                if torch.rand(1).item() > 0.35 and audio_mel.shape[-1] > 16:
                    f_start = int(torch.randint(0, audio_mel.shape[-1] - 8, (1,)).item())
                    audio_mel[:, :, f_start:f_start + 8] = 0.0

            optimizer.zero_grad()
            pred_bpm, pred_offset, bpm_logits = model(audio_mel, return_logits=True)
            pred_bpm = pred_bpm.squeeze(-1)
            pred_offset = pred_offset.squeeze(-1)

            # 3. Soft Gaussian smoothed Cross-Entropy loss over 301 BPM bins
            bins = model.bpm_bins.unsqueeze(0)  # [1, 301]
            sigma = 1.0
            target_dist = torch.exp(- (bins - target_bpm.unsqueeze(1)) ** 2 / (2.0 * sigma ** 2))
            target_dist = target_dist / (target_dist.sum(dim=-1, keepdim=True) + 1e-8)
            loss_ce = - (target_dist * F.log_softmax(bpm_logits, dim=-1)).sum(dim=-1).mean()

            # 4. Smooth L1 regression loss on continuous BPM
            loss_reg = F.smooth_l1_loss(pred_bpm, target_bpm)

            # 5. Pure High-Precision Cyclic Phase Loss (Modulo beat length, zero whole-beat penalty)
            beat_len_true = 60000.0 / (target_bpm + 1e-5)
            pred_phase = pred_offset % beat_len_true
            true_phase = target_offset % beat_len_true
            phase_diff = (pred_phase - true_phase).abs() % beat_len_true
            phase_dist = torch.min(phase_diff, beat_len_true - phase_diff)  # in milliseconds

            # SmoothL1 on milliseconds with beta=2.0 ms
            loss_phase_ms = F.smooth_l1_loss(phase_dist, torch.zeros_like(phase_dist), beta=2.0)
            # Circular Cosine phase loss for global phase angle gradient
            phase_rad = (phase_dist / beat_len_true) * (2.0 * math.pi)
            loss_phase_cos = (1.0 - torch.cos(phase_rad)).mean()

            loss_offset = 0.15 * loss_phase_ms + 2.0 * loss_phase_cos

            loss = loss_ce + 0.5 * loss_reg + 1.0 * loss_offset
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            B = audio_mel.shape[0]
            total_loss += loss.item() * B
            total_bpm_err += torch.abs(pred_bpm - target_bpm).sum().item()
            total_phase_err += phase_dist.sum().item()

        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start_t
        train_loss = total_loss / len(train_loader.dataset)
        bpm_mae = total_bpm_err / len(train_loader.dataset)
        phase_mae = total_phase_err / len(train_loader.dataset)

        # Validation (Evaluated on clean real data without speed perturbation)
        model.eval()
        val_loss, val_bpm_err, val_phase_err, val_snapped_err = 0.0, 0.0, 0.0, 0.0
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
                pred_phase = pred_offset % beat_len_true
                true_phase = target_offset % beat_len_true
                phase_diff = (pred_phase - true_phase).abs() % beat_len_true
                phase_dist = torch.min(phase_diff, beat_len_true - phase_diff)

                loss_phase_ms = F.smooth_l1_loss(phase_dist, torch.zeros_like(phase_dist), beta=2.0)
                phase_rad = (phase_dist / beat_len_true) * (2.0 * math.pi)
                loss_phase_cos = (1.0 - torch.cos(phase_rad)).mean()
                loss_offset = 0.15 * loss_phase_ms + 2.0 * loss_phase_cos

                # Micro-transient snapped phase error
                snapped_offset = snap_offset_to_audio_transients(
                    audio_mel, pred_offset.unsqueeze(-1), pred_bpm.unsqueeze(-1)
                ).squeeze(-1)
                snapped_phase = snapped_offset % beat_len_true
                snapped_diff = (snapped_phase - true_phase).abs() % beat_len_true
                snapped_dist = torch.min(snapped_diff, beat_len_true - snapped_diff)

                B = audio_mel.shape[0]
                val_loss += (loss_ce + 0.5 * loss_reg + 1.0 * loss_offset).item() * B
                val_bpm_err += torch.abs(pred_bpm - target_bpm).sum().item()
                val_phase_err += phase_dist.sum().item()
                val_snapped_err += snapped_dist.sum().item()

        val_loss /= len(val_loader.dataset)
        val_mae = val_bpm_err / len(val_loader.dataset)
        val_phase_mae = val_phase_err / len(val_loader.dataset)
        val_snapped_mae = val_snapped_err / len(val_loader.dataset)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s | lr: {cur_lr:.6f}] "
            f"Train: BPM MAE {bpm_mae:.2f}, Phase {phase_mae:.1f}ms | "
            f"Val: BPM MAE {val_mae:.2f}, Phase {val_phase_mae:.1f}ms (Snapped: {val_snapped_mae:.1f}ms)"
        )

        history.append({"epoch": epoch, "val_loss": val_loss, "val_bpm_mae": val_mae, "val_phase_mae": val_phase_mae})
        if val_mae < best_val_mae or val_loss < best_val_loss:
            if val_mae < best_val_mae:
                best_val_mae = val_mae
            best_val_loss = val_loss
            ckpt_path = checkpoint_dir / "model_1_sync_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_bpm_mae": val_mae,
                "val_phase_mae": val_phase_mae,
            }, ckpt_path)
            print(f"  ⭐ Saved best checkpoint to '{ckpt_path}' (BPM MAE: {val_mae:.2f}, Phase MAE: {val_phase_mae:.1f}ms)")

    return {"model": "model_1_sync", "best_val_loss": best_val_loss, "best_val_mae": best_val_mae, "history": history}


def train_model_2_dynamics(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    checkpoint_dir: Path,
) -> Dict[str, Any]:
    """Trains Model 2: DynamicsSVModel (Slider Velocity & Timing changes)."""
    print("\n" + "=" * 60)
    print("▶ TRAINING MODEL 2: DynamicsSVModel (Slider Velocity & Zones)")
    print("=" * 60)

    model = DynamicsSVModel(audio_dim=80, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
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
        print(f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] Train SV MSE: {train_loss:.4f} | Val SV MSE: {val_loss:.4f}")
        history.append({"epoch": epoch, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = checkpoint_dir / "model_2_dynamics_best.pt"
            torch.save({"epoch": epoch, "model_state": model.state_dict()}, ckpt_path)
            print(f"  ⭐ Saved best checkpoint to '{ckpt_path}'")

    return {"model": "model_2_dynamics", "best_val_loss": best_val_loss, "history": history}


def train_model_4_evaluator(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    checkpoint_dir: Path,
) -> Dict[str, Any]:
    """Trains Model 4: DifficultyEvaluatorModel (Star Rating compliance validation)."""
    print("\n" + "=" * 60)
    print("▶ TRAINING MODEL 4: DifficultyEvaluatorModel (Star Rating Validator)")
    print("=" * 60)

    model = DifficultyEvaluatorModel(num_classes=16, token_dim=64, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
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
            f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] "
            f"Train SR MSE: {train_loss:.4f} (MAE: {train_mae:.2f}) | "
            f"Val SR MSE: {val_loss:.4f} (MAE: {val_mae:.2f})"
        )
        history.append({"epoch": epoch, "val_sr_mae": val_mae})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = checkpoint_dir / "model_4_evaluator_best.pt"
            torch.save({"epoch": epoch, "model_state": model.state_dict()}, ckpt_path)
            print(f"  ⭐ Saved best checkpoint to '{ckpt_path}'")

    return {"model": "model_4_evaluator", "best_val_loss": best_val_loss, "history": history}


def main():
    parser = argparse.ArgumentParser(description="Train osu!mania 4K AI models on CPU or GPU.")
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
    parser.add_argument("--epochs", "-e", type=int, default=50)
    parser.add_argument("--batch_size", "-b", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--val_split", type=float, default=0.1)
    args = parser.parse_args()

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

    print(f"\n[Training Config]")
    print(f"  Model target:      {args.model}")
    print(f"  Device:            {device}")
    print(f"  Train samples:     {train_size}")
    print(f"  Validation samples:{val_size}")
    print(f"  Batch size:        {args.batch_size}")
    print(f"  Epochs:            {args.epochs}")
    print(f"  Learning rate:     {args.lr}")
    print(f"  Checkpoints dir:   {ckpt_dir.resolve()}")

    target = str(args.model).lower()
    if target in ("1", "sync", "all"):
        is_song = "song_sync" in str(args.dataset).lower()
        train_model_1_sync(train_loader, val_loader, device, args.epochs, args.lr, ckpt_dir, is_song_level=is_song)
    if target in ("2", "dynamics", "all"):
        train_model_2_dynamics(train_loader, val_loader, device, args.epochs, args.lr, ckpt_dir)
    if target in ("3", "transformer", "pattern", "all"):
        train_model_3_transformer(train_loader, val_loader, device, args.epochs, args.lr, ckpt_dir)
    if target in ("4", "evaluator", "sr", "all"):
        train_model_4_evaluator(train_loader, val_loader, device, args.epochs, args.lr, ckpt_dir)

    print("\n✅ Training procedure completed successfully!")


if __name__ == "__main__":
    main()
