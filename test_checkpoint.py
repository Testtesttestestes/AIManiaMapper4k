#!/usr/bin/env python3
"""
Diagnostic and inference test tool for saved osu!mania 4K AI model checkpoints.

Usage examples:
    # 1. Test Model 1 (BPM/Offset) on your own audio file (.wav, .mp3, .ogg):
    python test_checkpoint.py --model 1 --audio path/to/my_song.mp3

    # 2. Test Model 1 on samples from the training dataset:
    python test_checkpoint.py --model 1 --dataset dataset_processed/train_data.pt --num_samples 5

    # 3. Test with explicit checkpoint path:
    python test_checkpoint.py --model 1 --checkpoint checkpoints/model_1_sync_best.pt --audio track.wav

    # 4. Test other models:
    python test_checkpoint.py --model 2 --audio track.wav        # Test SV dynamics
    python test_checkpoint.py --model 3 --audio track.wav --sr 4.5  # Test Transformer pattern generation
    python test_checkpoint.py --model 4 --audio track.wav        # Test Difficulty evaluator
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


def check_torch_installed():
    if not TORCH_AVAILABLE:
        print("\n" + "=" * 70)
        print("❌ [Error] PyTorch (torch) is not installed in the current Python environment.")
        print("=" * 70)
        print("Please install PyTorch to run neural network checkpoint inference:")
        print("  pip install torch numpy\n")
        sys.exit(1)


def resolve_device(requested_device: str) -> torch.device:
    req = requested_device.lower()
    if req in ["cuda", "gpu"]:
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[Device Warning] CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    elif req == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    elif req == "cpu":
        return torch.device("cpu")
    else:  # auto
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


def load_model_checkpoint(checkpoint_path: Path, model: torch.nn.Module, device: torch.device) -> Dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
        meta = {k: v for k, v in checkpoint.items() if k not in ["model_state_dict", "model_state", "state_dict"]}
    else:
        model.load_state_dict(checkpoint)
        meta = {}

    model.to(device)
    model.eval()
    return meta


# ==============================================================================
# Model 1: Sync BPM / Offset Tester
# ==============================================================================
def test_model_1_audio(
    audio_path: str,
    checkpoint_path: Path,
    device: torch.device,
    compare_heuristic: bool = True,
):
    from osu_mapper.models.sync_model import SyncBPMOffsetModel, snap_offset_to_audio_transients
    from osu_mapper.dataset.feature_extractor import compute_mel_spectrogram, extract_audio_waveform
    from osu_mapper.pipeline.audio import analyze_audio_rhythm

    print(f"\n[Model 1 Tester] Loading checkpoint: {checkpoint_path}")
    model = SyncBPMOffsetModel(audio_dim=80, hidden_dim=256)
    meta = load_model_checkpoint(checkpoint_path, model, device)

    epoch_info = f" (Epoch {meta['epoch']})" if "epoch" in meta else ""
    print(f"[Model 1 Tester] Checkpoint loaded successfully{epoch_info} on {device}!")

    print(f"\n[Audio Processing] Loading audio: {audio_path}")
    audio_wav = extract_audio_waveform(audio_path, target_sr=16000)
    mel = compute_mel_spectrogram(audio_wav, sr=16000, n_mels=80)
    print(f"[Audio Processing] Mel spectrogram shape: {mel.shape} ({mel.shape[0] * 16 / 1000:.2f}s duration)")

    # Run inference on audio
    seq_len = min(1024, mel.shape[0])
    inp = mel[:seq_len].unsqueeze(0).to(device)

    with torch.no_grad():
        bpm_out, offset_out = model(inp)
        pred_bpm = float(bpm_out.squeeze().item())
        pred_offset = int(round(float(offset_out.squeeze().item())))
        snapped_offset = int(round(float(snap_offset_to_audio_transients(inp, offset_out, bpm_out).squeeze().item())))

    pred_bpm = max(45.0, min(360.0, pred_bpm))
    pred_offset = max(0, pred_offset)
    snapped_offset = max(0, snapped_offset)

    beat_len_ms = 60000.0 / pred_bpm
    bar_len_ms = 4.0 * beat_len_ms

    # First 8 beat timestamps
    first_beats = [int(round(snapped_offset + i * beat_len_ms)) for i in range(8)]

    print("\n" + "=" * 65)
    print(" 🎯 MODEL 1 (SYNC BPM & OFFSET) INFERENCE RESULTS")
    print("=" * 65)
    print(f"  • Predicted BPM:          {pred_bpm:.2f} BPM")
    print(f"  • Model Raw Offset:       {pred_offset} ms")
    print(f"  • Transient-Snapped Off:  {snapped_offset} ms (Micro-transient locked)")
    print(f"  • Beat Duration (1/1):    {beat_len_ms:.2f} ms")
    print(f"  • Measure Length (4/4):   {bar_len_ms:.2f} ms")
    print(f"  • Half-beat Snap (1/2):   {beat_len_ms / 2.0:.2f} ms")
    print(f"  • Quarter-beat (1/4):     {beat_len_ms / 4.0:.2f} ms")
    print(f"  • First 8 Downbeats:      {first_beats} ms")

    if compare_heuristic and audio_path.lower().endswith(".wav"):
        try:
            print("\n[Comparison] Running heuristic autocorrelation audio analysis...")
            heuristic = analyze_audio_rhythm(audio_path)
            print(f"  • Heuristic BPM:          {heuristic.bpm:.2f} BPM  (Diff: {abs(pred_bpm - heuristic.bpm):.2f})")
            print(f"  • Heuristic Offset:       {heuristic.offset_ms} ms  (Diff: {abs(pred_offset - heuristic.offset_ms)} ms)")
        except Exception as e:
            print(f"  [Notice] Heuristic analysis skipped: {e}")
    print("=" * 65 + "\n")


def test_model_1_dataset(
    dataset_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    num_samples: int = 5,
):
    from osu_mapper.models.sync_model import SyncBPMOffsetModel, snap_offset_to_audio_transients

    print(f"\n[Model 1 Tester] Loading checkpoint: {checkpoint_path}")
    model = SyncBPMOffsetModel(audio_dim=80, hidden_dim=256)
    meta = load_model_checkpoint(checkpoint_path, model, device)

    # For Model 1, prefer the dedicated whole-song sync dataset if available
    dataset_p = Path(dataset_path)
    sync_cand = dataset_p.parent / "song_sync_data.pt"
    if "song_sync" not in str(dataset_path).lower() and sync_cand.exists():
        print(f"💡 [Model 1 Auto-Select] Dedicated whole-song sync dataset detected: '{sync_cand}'")
        print("   Testing Model 1 against whole-song 16s audio tracks.")
        dataset_path = sync_cand

    print(f"[Dataset] Loading dataset: {dataset_path}")
    try:
        dataset: List[Dict[str, Any]] = torch.load(dataset_path, map_location="cpu", weights_only=False)
    except TypeError:
        dataset = torch.load(dataset_path, map_location="cpu")
    total_chunks = len(dataset)
    print(f"[Dataset] Total samples in dataset: {total_chunks}")

    num_eval = min(num_samples, total_chunks)
    step = max(1, total_chunks // num_eval)
    sample_indices = [i * step for i in range(num_eval)]

    print("\n" + "=" * 98)
    print(f" 📊 TESTING MODEL 1 ON {num_eval} DATASET SAMPLES")
    print("=" * 98)
    print(f"{'CHUNK':<6} | {'TRUE BPM':<8} | {'PRED BPM':<8} | {'BPM ERR':<7} | {'TRUE OFF':<8} | {'PRED OFF':<8} | {'PHASE ERR':<9} | {'MICRO (SNAP)':<12}")
    print("-" * 98)

    total_bpm_err = 0.0
    total_off_err = 0.0
    total_phase_err = 0.0
    total_snapped_err = 0.0

    for idx in sample_indices:
        item = dataset[idx]
        mel = item["audio_mel"].unsqueeze(0).to(device)
        true_bpm = float(item["bpm"].squeeze().item())
        true_off = float(item["offset_ms"].squeeze().item())

        # Downbeat phase modulo beat duration
        beat_len = 60000.0 / (true_bpm + 1e-5)
        # If evaluating a mid-song chunk (from train_data.pt), calibrate target offset to chunk start
        if "start_ms" in item:
            chunk_start = float(item["start_ms"])
            true_off = (true_off - chunk_start) % beat_len

        with torch.no_grad():
            bpm_out, off_out = model(mel)
            pred_bpm = float(bpm_out.squeeze().item())
            pred_off = float(off_out.squeeze().item())
            snapped_off = float(snap_offset_to_audio_transients(mel, off_out, bpm_out).squeeze().item())

        bpm_err = abs(pred_bpm - true_bpm)
        off_err = abs(pred_off - true_off)

        phase_diff = abs(pred_off - true_off) % beat_len
        phase_err = min(phase_diff, beat_len - phase_diff)

        snapped_diff = abs(snapped_off - true_off) % beat_len
        snapped_err = min(snapped_diff, beat_len - snapped_diff)

        total_bpm_err += bpm_err
        total_off_err += off_err
        total_phase_err += phase_err
        total_snapped_err += snapped_err

        print(f"#{idx:<5} | {true_bpm:<8.1f} | {pred_bpm:<8.1f} | {bpm_err:<7.2f} | {true_off:<8.0f} | {pred_off:<8.0f} | {phase_err:<9.1f}ms | {snapped_err:<12.1f}ms")

    print("-" * 98)
    print(f"Average BPM Absolute Error (MAE):           {total_bpm_err / num_eval:.2f} BPM")
    print(f"Average Model Phase Error (MAE):            {total_phase_err / num_eval:.1f} ms")
    print(f"Average Transient-Snapped Phase Error (MAE): {total_snapped_err / num_eval:.1f} ms  (Target: +/- 2 ms)")
    print(f"Raw Scalar Offset Error (MAE):              {total_off_err / num_eval:.1f} ms  (Whole-beat integer offsets)")
    print("=" * 98 + "\n")


# ==============================================================================
# Model 2: SV Dynamics Tester
# ==============================================================================
def test_model_2_audio(audio_path: str, checkpoint_path: Path, device: torch.device):
    from osu_mapper.models.dynamics_model import DynamicsSVModel
    from osu_mapper.dataset.feature_extractor import compute_mel_spectrogram, extract_audio_waveform

    print(f"\n[Model 2 Tester] Loading checkpoint: {checkpoint_path}")
    model = DynamicsSVModel(audio_dim=80, hidden_dim=128)
    load_model_checkpoint(checkpoint_path, model, device)

    audio_wav = extract_audio_waveform(audio_path, target_sr=16000)
    mel = compute_mel_spectrogram(audio_wav, sr=16000, n_mels=80)
    inp = mel[:512].unsqueeze(0).to(device)

    with torch.no_grad():
        sv_pred = model(inp)  # [1, T, 1]
        sv_vals = sv_pred.squeeze().cpu().numpy()

    print("\n" + "=" * 65)
    print(" 🚀 MODEL 2 (DYNAMICS & SV) INFERENCE RESULTS")
    print("=" * 65)
    print(f"  • SV Minimum:             {sv_vals.min():.2f}x")
    print(f"  • SV Maximum:             {sv_vals.max():.2f}x")
    print(f"  • SV Mean:                {sv_vals.mean():.2f}x")
    print(f"  • First 10 SV values:     {[round(float(x), 2) for x in sv_vals[:10]]}")
    print("=" * 65 + "\n")


# ==============================================================================
# Model 3: Pattern Transformer Tester
# ==============================================================================
def test_model_3_audio(
    audio_path: str,
    checkpoint_path: Path,
    device: torch.device,
    star_rating: float = 4.0,
):
    from osu_mapper.models.transformer import ManiaPatternTransformer, TransformerConfig
    from osu_mapper.dataset.feature_extractor import compute_mel_spectrogram, extract_audio_waveform, token_to_keys

    print(f"\n[Model 3 Tester] Loading checkpoint: {checkpoint_path}")
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
    model = ManiaPatternTransformer(cfg)
    load_model_checkpoint(checkpoint_path, model, device)

    audio_wav = extract_audio_waveform(audio_path, target_sr=16000)
    mel = compute_mel_spectrogram(audio_wav, sr=16000, n_mels=80)

    # Autoregressively synthesize first 128 frames (~2 seconds)
    chunk_len = min(128, mel.shape[0])
    chunk_mel = mel[:chunk_len].unsqueeze(0).to(device)
    sr_tensor = torch.tensor([[star_rating]], dtype=torch.float32).to(device)

    prev_tokens = torch.zeros(1, chunk_len, dtype=torch.int64).to(device)

    with torch.no_grad():
        chord_logits, ln_logits = model(chunk_mel, star_rating=sr_tensor, prev_tokens=prev_tokens)
        pred_tokens = chord_logits.argmax(dim=-1).squeeze().cpu().tolist()

    non_empty = [t for t in pred_tokens if t > 0]
    note_count = len(non_empty)

    print("\n" + "=" * 65)
    print(" 🎹 MODEL 3 (MANIA PATTERN TRANSFORMER) INFERENCE RESULTS")
    print("=" * 65)
    print(f"  • Conditioned Star Rating: {star_rating:.1f}*")
    print(f"  • Evaluated Sequence:     {chunk_len} frames (~{chunk_len * 16 / 1000:.2f}s)")
    print(f"  • Total Chords Generated: {note_count} (out of {chunk_len} frames)")
    print(f"  • Estimated NPS:          {note_count / (chunk_len * 0.016):.1f} notes/sec")
    print(f"  • Sample Generated Tokens: {pred_tokens[:30]}")
    print("=" * 65 + "\n")


# ==============================================================================
# CLI Entry Point
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Test trained osu!mania 4K AI checkpoints on audio or dataset.")
    parser.add_argument("--model", "-m", type=int, choices=[1, 2, 3, 4], default=1, help="Model to test (1=Sync, 2=Dynamics, 3=Transformer, 4=Evaluator)")
    parser.add_argument("--checkpoint", "-c", type=str, default=None, help="Path to .pt checkpoint file (defaults to checkpoints/model_X_..._best.pt)")
    parser.add_argument("--audio", "-a", type=str, default=None, help="Path to audio file (.wav, .mp3, .ogg, .flac)")
    parser.add_argument("--dataset", "-d", type=str, default=None, help="Path to train_data.pt dataset for evaluation")
    parser.add_argument("--num_samples", "-n", type=int, default=5, help="Number of dataset samples to evaluate")
    parser.add_argument("--sr", "--star_rating", type=float, default=4.0, help="Star rating conditioning for Model 3")
    parser.add_argument("--device", type=str, default="auto", help="Execution device: auto, cpu, cuda, gpu, mps")

    args = parser.parse_args()
    check_torch_installed()

    device = resolve_device(args.device)

    # Determine default checkpoint path if not provided
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        defaults = {
            1: "checkpoints/model_1_sync_best.pt",
            2: "checkpoints/model_2_dynamics_best.pt",
            3: "checkpoints/model_3_transformer_best.pt",
            4: "checkpoints/model_4_evaluator_best.pt",
        }
        ckpt_path = defaults.get(args.model)

    resolved_ckpt = Path(ckpt_path)
    if not resolved_ckpt.exists():
        print("\n" + "=" * 70)
        print(f"❌ [Error] Checkpoint not found at '{resolved_ckpt}'")
        print("=" * 70)
        print(f"You haven't trained Model {args.model} yet, or the checkpoint is in a different path.")
        print(f"To train Model {args.model}, run:")
        print(f"  python train.py --model {args.model} --epochs 10")
        print("Or specify the path manually with: --checkpoint path/to/weights.pt\n")
        sys.exit(1)

    # Run mode
    if args.audio:
        if not os.path.exists(args.audio):
            print(f"❌ [Error] Audio file not found: {args.audio}")
            sys.exit(1)

        if args.model == 1:
            test_model_1_audio(args.audio, resolved_ckpt, device)
        elif args.model == 2:
            test_model_2_audio(args.audio, resolved_ckpt, device)
        elif args.model == 3:
            test_model_3_audio(args.audio, resolved_ckpt, device, star_rating=args.sr)
        elif args.model == 4:
            print("Model 4 audio evaluation: Please supply a beatmap or run full pipeline.")

    elif args.dataset:
        ds_path = Path(args.dataset)
        if not ds_path.exists():
            print(f"❌ [Error] Dataset file not found: {ds_path}")
            sys.exit(1)

        if args.model == 1:
            test_model_1_dataset(ds_path, resolved_ckpt, device, num_samples=args.num_samples)
        else:
            print(f"[Notice] Dataset evaluation is currently optimized for Model 1. Testing Model {args.model}...")

    else:
        # Check if default dataset or sample audio exists
        sample_audio = "output_beatmaps/synthetic_test.wav"
        default_ds = Path("dataset_processed/train_data.pt")

        if default_ds.exists():
            print(f"[Auto-Detection] No audio file specified. Found dataset at '{default_ds}'. Running dataset test...")
            test_model_1_dataset(default_ds, resolved_ckpt, device, num_samples=args.num_samples)
        elif os.path.exists(sample_audio):
            print(f"[Auto-Detection] Found sample audio '{sample_audio}'. Running audio test...")
            test_model_1_audio(sample_audio, resolved_ckpt, device)
        else:
            print("\n" + "=" * 70)
            print("ℹ️  HOW TO RUN CHECKPOINT TESTS:")
            print("=" * 70)
            print("To test on an audio file (wav / mp3 / ogg):")
            print(f"  python test_checkpoint.py --model {args.model} --audio my_song.mp3")
            print("")
            print("To test on the training dataset:")
            print(f"  python test_checkpoint.py --model {args.model} --dataset dataset_processed/train_data.pt")
            print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
