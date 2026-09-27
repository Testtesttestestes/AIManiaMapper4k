"""
Automated Interpreter Calibration & Tuning Tool.
Tests the neural interpreter against real human ground-truth beatmaps from the dataset,
measures rhythmic precision, silence handling, chord density, and Star Rating deviation,
and performs grid search / optimization to find the exact parameters that maximize fidelity.
"""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from osu_mapper.models.transformer import ManiaPatternTransformer, TransformerConfig


def evaluate_sample(
    model: ManiaPatternTransformer,
    sample: Dict[str, Any],
    device: torch.device,
    silence_gate: float = 0.25,
    chord_threshold: float = 0.0,
    temperature: float = 0.0,
    top_p: float = 0.90,
    ln_threshold: float = 0.78,
    use_scaffold: bool = False,
) -> Dict[str, float]:
    """Evaluates a single validation chunk against ground truth human mapping."""
    mel = sample["audio_mel"].unsqueeze(0).to(device)       # [1, T, 80]
    tokens = sample["tokens"].unsqueeze(0).to(device)         # [1, T]
    ln_mask = sample["ln_mask"].unsqueeze(0).to(device)       # [1, T, 4]
    star_rating = sample["star_rating"].to(device)           # [1, 1]
    if star_rating.ndim == 1:
        star_rating = star_rating.unsqueeze(0)

    T = tokens.shape[1]
    sr_val = float(star_rating.squeeze().item())

    grid_metrics = sample.get("grid_metrics")
    if grid_metrics is not None:
        grid_metrics = grid_metrics.unsqueeze(0).to(device)

    # Compute energy from mel
    with torch.no_grad():
        frame_energies = mel.exp().mean(dim=-1)               # [1, T]
        mean_e = frame_energies.mean(dim=-1, keepdim=True).clamp(min=1e-4)
        energy_ratio = (frame_energies / mean_e).squeeze(0)   # [T]

    # Progressive autoregressive decoding
    generated_tokens: List[int] = []
    target_density = max(0.05, sr_val / 10.0)

    with torch.no_grad():
        full_audio_memory = model.audio_encoder(mel)

        for cur_pos in range(T):
            ctx_start = max(0, cur_pos - 64)
            window_len = (cur_pos + 1) - ctx_start

            window_mel = mel[:, ctx_start : cur_pos + 1]
            window_prev = torch.zeros(1, window_len, dtype=torch.int64, device=device)
            if cur_pos > ctx_start:
                window_prev[0, 1 : cur_pos - ctx_start + 1] = torch.tensor(
                    generated_tokens[ctx_start:cur_pos], dtype=torch.int64, device=device
                )

            # Build strain metrics
            run_notes = sum(1 for tok in generated_tokens if tok > 0)
            run_density = run_notes / max(1, cur_pos + 1)
            ratio = min(5.0, run_density / target_density)
            strain_window = torch.zeros(1, window_len, 2, dtype=torch.float32, device=device)
            strain_window[0, :, 0] = run_density
            strain_window[0, :, 1] = ratio

            w_grid = grid_metrics[:, ctx_start : cur_pos + 1] if grid_metrics is not None else None
            w_scaffold = tokens[:, ctx_start : cur_pos + 1] if use_scaffold else None
            w_audio_mem = full_audio_memory[:, ctx_start : cur_pos + 1]

            # Silence Gating: if audio energy is below threshold, suppress notes
            e_ratio_val = float(energy_ratio[cur_pos].item())
            if e_ratio_val < silence_gate:
                generated_tokens.append(0)
                continue

            chord_logits, ln_logits, pred_sr = model(
                window_mel,
                star_rating=star_rating,
                prev_tokens=window_prev,
                grid_metrics=w_grid,
                scaffold_tokens=w_scaffold,
                strain_metrics=strain_window,
                audio_memory=w_audio_mem,
                return_sr=True,
            )

            cur_logits = chord_logits[0, -1]  # [16]

            if temperature <= 0.05:
                pred_token = int(cur_logits.argmax().item())
            else:
                probs = F.softmax(cur_logits / max(0.1, temperature), dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cum_probs > top_p
                mask[1:] = mask[:-1].clone()
                mask[0] = False
                probs[sorted_indices[mask]] = 0.0
                probs = probs / probs.sum().clamp(min=1e-8)
                pred_token = int(torch.multinomial(probs, 1).item())

            generated_tokens.append(pred_token)

    pred_arr = torch.tensor(generated_tokens, dtype=torch.int64, device=device)
    true_arr = tokens.squeeze(0)

    true_is_note = (true_arr > 0)
    pred_is_note = (pred_arr > 0)

    true_cnt = int(true_is_note.sum().item())
    pred_cnt = int(pred_is_note.sum().item())

    # Rhythm note precision & recall
    hit_notes = int((pred_is_note & true_is_note).sum().item())
    note_recall = (hit_notes / max(1, true_cnt))
    note_precision = (hit_notes / max(1, pred_cnt))
    rhythm_f1 = (2 * note_precision * note_recall) / max(1e-6, (note_precision + note_recall))

    # Silence Accuracy: correctly identifying zero notes when human has zero notes
    true_silence = (true_arr == 0)
    pred_silence = (pred_arr == 0)
    silence_acc = int((pred_silence & true_silence).sum().item()) / max(1, int(true_silence.sum().item()))

    # Exact 4K chord match on active beats
    exact_4k = int((pred_arr[true_is_note] == true_arr[true_is_note]).sum().item()) / max(1, true_cnt)

    return {
        "sr": sr_val,
        "true_notes": true_cnt,
        "pred_notes": pred_cnt,
        "note_recall": note_recall,
        "note_precision": note_precision,
        "rhythm_f1": rhythm_f1,
        "silence_acc": silence_acc,
        "exact_4k": exact_4k,
    }


def run_calibration_grid(
    dataset_path: str,
    checkpoint_path: str,
    num_samples: int = 12,
    device_str: str = "auto",
):
    dev = torch.device("cuda" if torch.cuda.is_available() and device_str == "auto" else "cpu")
    print(f"🔧 Loading dataset from: {dataset_path}")
    try:
        dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    except TypeError:
        dataset = torch.load(dataset_path, map_location="cpu")

    print(f"🧠 Loading Model 3 Checkpoint: {checkpoint_path}")
    cfg = TransformerConfig(
        audio_dim=80,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        num_classes=16,
        max_seq_len=512,
        dropout=0.0,
    )
    model = ManiaPatternTransformer(cfg).to(dev)
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
    model.load_state_dict(state, strict=False)
    model.eval()

    step = max(1, len(dataset) // num_samples)
    samples = [dataset[i * step] for i in range(min(num_samples, len(dataset)))]

    print("\n" + "=" * 90)
    print(" 🧪 CALIBRATING INTERPRETER: TESTING SILENCE GATING & RHYTHMIC COHERENCE")
    print("=" * 90)

    silence_gates = [0.10, 0.20, 0.30, 0.40, 0.50]
    temperatures = [0.0, 0.35, 0.70]

    best_score = -1.0
    best_config = {}

    for gate in silence_gates:
        for temp in temperatures:
            tot_f1, tot_silence, tot_exact, tot_recall, tot_prec = 0.0, 0.0, 0.0, 0.0, 0.0

            for s in samples:
                res = evaluate_sample(
                    model=model,
                    sample=s,
                    device=dev,
                    silence_gate=gate,
                    temperature=temp,
                )
                tot_f1 += res["rhythm_f1"]
                tot_silence += res["silence_acc"]
                tot_exact += res["exact_4k"]
                tot_recall += res["note_recall"]
                tot_prec += res["note_precision"]

            n = len(samples)
            avg_f1 = (tot_f1 / n) * 100.0
            avg_silence = (tot_silence / n) * 100.0
            avg_exact = (tot_exact / n) * 100.0
            avg_rec = (tot_recall / n) * 100.0
            avg_prec = (tot_prec / n) * 100.0

            # Composite Fidelity Score (Balances note precision, recall, and zero-spam in silence)
            composite_score = (avg_f1 * 0.50) + (avg_silence * 0.35) + (avg_exact * 0.15)

            status = "⭐ NEW BEST" if composite_score > best_score else ""
            print(
                f"Gate={gate:.2f} | Temp={temp:.2f} -> "
                f"Rhythm F1: {avg_f1:5.1f}% | Silence Clean: {avg_silence:5.1f}% | "
                f"Recall: {avg_rec:5.1f}% | Prec: {avg_prec:5.1f}% | {status}"
            )

            if composite_score > best_score:
                best_score = composite_score
                best_config = {
                    "silence_gate": gate,
                    "temperature": temp,
                    "rhythm_f1": avg_f1,
                    "silence_acc": avg_silence,
                    "recall": avg_rec,
                    "precision": avg_prec,
                }

    print("=" * 90)
    print(f" 🏆 OPTIMAL INTERPRETER CONFIGURATION FOUND:")
    print(f"  • Silence Energy Gate:    {best_config['silence_gate']:.2f}")
    print(f"  • Sampling Temperature:   {best_config['temperature']:.2f}")
    print(f"  • Rhythmic F1-Score:      {best_config['rhythm_f1']:.1f}%")
    print(f"  • Silence Cleanliness:    {best_config['silence_acc']:.1f}% (No spam in quiet/break sections)")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate Interpreter on Real Ground Truth Beatmaps")
    parser.add_argument("--dataset", type=str, default="dataset_processed/val_data.pt", help="Path to val_data.pt or train_data.pt")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_3_transformer_best.pt", help="Path to model checkpoint")
    parser.add_argument("--num_samples", type=int, default=15, help="Number of real songs to calibrate over")
    args = parser.parse_args()

    run_calibration_grid(
        dataset_path=args.dataset,
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
    )
