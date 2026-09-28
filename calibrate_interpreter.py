"""
Automated Multi-Parameter Interpreter Calibration Tool.
Tests the neural interpreter against real human ground-truth beatmaps from the dataset.
Accurately replicates the production Two-Stage (Scaffold + Neural Refiner) pipeline.
"""

import argparse
import itertools
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from osu_mapper.models.transformer import ManiaPatternTransformer, TransformerConfig
from osu_mapper.pipeline.onset_scaffold import compute_acoustic_onset_scaffold


def build_algorithmic_scaffold_from_grid(
    grid_metrics: torch.Tensor,
    star_rating: float,
    energy_ratio: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Builds the deterministic rhythmic scaffold matching Stage 1 of the generation pipeline.
    Places structured downbeats and periodic beats scaled by difficulty.
    """
    T = grid_metrics.shape[1]
    scaffold = torch.zeros(1, T, dtype=torch.int64, device=device)

    is_downbeat = (grid_metrics[0, :, 2] > 0.5)
    divisor_frac = grid_metrics[0, :, 0]
    is_1_1 = (divisor_frac < 0.05) | (divisor_frac > 0.95)
    is_1_2 = (divisor_frac - 0.5).abs() < 0.05
    is_1_4 = ((divisor_frac - 0.25).abs() < 0.05) | ((divisor_frac - 0.75).abs() < 0.05)

    chord_size = 1 if star_rating < 1.8 else (2 if star_rating < 3.8 else (3 if star_rating < 5.5 else 4))

    for t in range(T):
        e = float(energy_ratio[t].item())
        if e < 0.20:
            continue

        if is_downbeat[t]:
            scaffold[0, t] = 3 if chord_size >= 2 else 1  # [0, 1] or [0]
        elif is_1_1[t]:
            scaffold[0, t] = 1 << (t % 4)
        elif is_1_2[t] and (star_rating >= 2.5):
            scaffold[0, t] = 1 << ((t + 1) % 4)
        elif is_1_4[t] and (star_rating >= 4.0):
            scaffold[0, t] = 1 << ((t + 2) % 4)

    return scaffold


def evaluate_sample_with_thresholds(
    model: ManiaPatternTransformer,
    sample: Dict[str, Any],
    device: torch.device,
    silence_gate: float = 0.20,
    note_conf_thresh: float = 0.0,
    ln_thresh: float = 0.78,
    temperature: float = 0.0,
    top_p: float = 0.90,
    use_scaffold: bool = True,
    step_mode: str = "neural_step",
) -> Dict[str, float]:
    """
    Evaluates a single validation chunk against ground truth human mapping.
    Supports:
    - step_mode='neural_step': Model predicts next-note forward jump \Delta t in ms and advances dynamically.
    - step_mode='dense_grid': Fixed tick-by-tick dense evaluation across every subdivision.
    """
    mel = sample["audio_mel"].unsqueeze(0).to(device)       # [1, T, 80]
    tokens = sample["tokens"].unsqueeze(0).to(device)         # [1, T]
    ln_mask = sample["ln_mask"].unsqueeze(0).to(device)       # [1, T, 4]
    star_rating = sample["star_rating"].to(device)           # [1, 1]
    if star_rating.ndim == 1:
        star_rating = star_rating.unsqueeze(0)

    T = tokens.shape[1]
    sr_val = float(star_rating.squeeze().item())

    # Multi-scale audio if available
    multiscale_audio = sample.get("multiscale_audio")
    if multiscale_audio is not None:
        multiscale_audio = multiscale_audio.unsqueeze(0).to(device)

    grid_metrics = sample.get("grid_metrics")
    if grid_metrics is not None:
        grid_metrics = grid_metrics.unsqueeze(0).to(device)
    else:
        grid_metrics = torch.zeros(1, T, 8, device=device)

    # Estimate ms duration per tick
    bpm_val = float(sample["bpm"].item()) if "bpm" in sample else 150.0
    beat_len_ms = float(sample.get("beat_len_ms", 60000.0 / max(30.0, bpm_val)))
    if "start_ms" in sample and "end_ms" in sample:
        tick_len_ms = max(5.0, (float(sample["end_ms"]) - float(sample["start_ms"])) / max(1, T - 1))
    else:
        tick_len_ms = max(5.0, beat_len_ms / 4.0)

    # Compute energy from mel
    with torch.no_grad():
        frame_energies = mel.exp().mean(dim=-1)               # [1, T]
        mean_e = frame_energies.mean(dim=-1, keepdim=True).clamp(min=1e-4)
        energy_ratio = (frame_energies / mean_e).squeeze(0)   # [T]

    # Build Stage 1 Acoustic Onset Scaffold if enabled
    if use_scaffold:
        scaffold_tokens = compute_acoustic_onset_scaffold(
            audio_mel=mel,
            grid_metrics=grid_metrics,
            star_rating=star_rating,
            silence_threshold=silence_gate,
        )
    else:
        scaffold_tokens = torch.zeros(1, T, dtype=torch.int64, device=device)

    generated_tokens = [0] * T
    generated_lns = [[0.0, 0.0, 0.0, 0.0] for _ in range(T)]
    target_density = max(0.05, sr_val / 10.0)

    true_arr = tokens.squeeze(0)
    true_note_indices = [i for i, tok in enumerate(true_arr.cpu().tolist()) if tok > 0]
    step_errors: List[float] = []

    with torch.no_grad():
        full_audio_memory = model.audio_encoder(mel)

        cur_pos = 0
        while cur_pos < T:
            ctx_start = max(0, cur_pos - 64)
            window_len = (cur_pos + 1) - ctx_start

            window_mel = mel[:, ctx_start : cur_pos + 1]
            window_prev = torch.zeros(1, window_len, dtype=torch.int64, device=device)
            if cur_pos > ctx_start:
                window_prev[0, 1 : cur_pos - ctx_start + 1] = torch.tensor(
                    generated_tokens[ctx_start:cur_pos], dtype=torch.int64, device=device
                )

            # Build progressive strain metrics
            run_notes = sum(1 for tok in generated_tokens[:cur_pos] if tok > 0)
            run_density = run_notes / max(1, cur_pos + 1)
            ratio = min(5.0, run_density / target_density)
            strain_window = torch.zeros(1, window_len, 2, dtype=torch.float32, device=device)
            strain_window[0, :, 0] = run_density
            strain_window[0, :, 1] = ratio

            w_grid = grid_metrics[:, ctx_start : cur_pos + 1]
            w_scaffold = scaffold_tokens[:, ctx_start : cur_pos + 1]
            w_audio_mem = full_audio_memory[:, ctx_start : cur_pos + 1]
            w_multi = multiscale_audio[:, ctx_start : cur_pos + 1] if multiscale_audio is not None else None

            # 1. Acoustic Silence Gate
            e_ratio_val = float(energy_ratio[cur_pos].item())
            if e_ratio_val < silence_gate:
                generated_tokens[cur_pos] = 0
                generated_lns[cur_pos] = [0.0, 0.0, 0.0, 0.0]
                cur_pos += 1
                continue

            try:
                chord_logits, ln_logits, _, pred_time_deltas = model(
                    window_mel,
                    star_rating=star_rating,
                    prev_tokens=window_prev,
                    grid_metrics=w_grid,
                    scaffold_tokens=w_scaffold,
                    strain_metrics=strain_window,
                    multiscale_audio=w_multi,
                    audio_memory=w_audio_mem,
                    return_sr=True,
                    return_time_delta=True,
                )
                pred_dt_ms = float(pred_time_deltas[0, -1, 0].item())
            except Exception:
                chord_logits, ln_logits, _ = model(
                    window_mel,
                    star_rating=star_rating,
                    prev_tokens=window_prev,
                    grid_metrics=w_grid,
                    scaffold_tokens=w_scaffold,
                    strain_metrics=strain_window,
                    multiscale_audio=w_multi,
                    audio_memory=w_audio_mem,
                    return_sr=True,
                )
                pred_dt_ms = tick_len_ms

            cur_logits = chord_logits[0, -1]       # [16]
            cur_ln_raw = ln_logits[0, -1]          # [4]
            cur_ln_probs = torch.sigmoid(cur_ln_raw).cpu().tolist()

            # Softmax over all 16 chord options
            probs_16 = F.softmax(cur_logits, dim=-1)
            p_idle = float(probs_16[0].item())
            p_active = 1.0 - p_idle

            # 2. Active Note Confidence Thresholding
            if (note_conf_thresh > 0.0) and (p_active < note_conf_thresh):
                pred_token = 0
            else:
                active_logits = cur_logits.clone()
                if note_conf_thresh > 0.0:
                    active_logits[0] = -1e9  # Active condition met

                # Difficulty chord mask
                max_k = 1 if sr_val < 1.8 else (2 if sr_val < 3.5 else (3 if sr_val < 5.0 else 4))
                for t_i in range(1, 16):
                    k_cnt = bin(t_i).count("1")
                    if k_cnt > max_k:
                        active_logits[t_i] = -1e9

                if temperature <= 0.05:
                    pred_token = int(active_logits.argmax().item())
                else:
                    scaled = active_logits / max(0.1, temperature)
                    p_act = F.softmax(scaled, dim=-1)
                    s_probs, s_idx = torch.sort(p_act, descending=True)
                    c_probs = torch.cumsum(s_probs, dim=-1)
                    mask = c_probs > top_p
                    mask[1:] = mask[:-1].clone()
                    mask[0] = False
                    p_act[s_idx[mask]] = 0.0
                    p_act = p_act / p_act.sum().clamp(min=1e-8)
                    pred_token = int(torch.multinomial(p_act, 1).item())

            generated_tokens[cur_pos] = pred_token
            generated_lns[cur_pos] = cur_ln_probs

            # Record step prediction error if at a real human note
            next_human_notes = [idx for idx in true_note_indices if idx > cur_pos]
            if next_human_notes:
                true_dt_ms = (next_human_notes[0] - cur_pos) * tick_len_ms
                step_errors.append(abs(pred_dt_ms - true_dt_ms))

            # Step advance logic
            if step_mode == "neural_step" and pred_token > 0:
                jump_ticks = max(1, int(round(pred_dt_ms / tick_len_ms)))
                cur_pos += jump_ticks
            else:
                cur_pos += 1

    pred_arr = torch.tensor(generated_tokens, dtype=torch.int64, device=device)
    true_arr = tokens.squeeze(0)

    true_is_note = (true_arr > 0)
    pred_is_note = (pred_arr > 0)

    true_cnt = int(true_is_note.sum().item())
    pred_cnt = int(pred_is_note.sum().item())

    # Rhythm precision, recall and F1-score
    hit_notes = int((pred_is_note & true_is_note).sum().item())
    note_recall = (hit_notes / max(1, true_cnt))
    note_precision = (hit_notes / max(1, pred_cnt))
    rhythm_f1 = (2 * note_precision * note_recall) / max(1e-6, (note_precision + note_recall))

    # Silence Cleanliness (zero false-positive spam during human pauses)
    true_silence = (true_arr == 0)
    pred_silence = (pred_arr == 0)
    silence_acc = int((pred_silence & true_silence).sum().item()) / max(1, int(true_silence.sum().item()))

    # Exact 4K chord matching
    exact_4k = int((pred_arr[true_is_note] == true_arr[true_is_note]).sum().item()) / max(1, true_cnt)

    # LN Recall & Precision
    pred_ln_tensor = torch.tensor(generated_lns, dtype=torch.float32, device=device) > ln_thresh
    true_ln_tensor = ln_mask.squeeze(0) > 0.5
    ln_true_cnt = int(true_ln_tensor.sum().item())
    ln_pred_cnt = int(pred_ln_tensor.sum().item())
    ln_hits = int((pred_ln_tensor & true_ln_tensor).sum().item())
    ln_recall = (ln_hits / max(1, ln_true_cnt)) if ln_true_cnt > 0 else 1.0
    ln_prec = (ln_hits / max(1, ln_pred_cnt)) if ln_pred_cnt > 0 else 1.0
    ln_f1 = (2 * ln_prec * ln_recall) / max(1e-6, (ln_prec + ln_recall))
    avg_step_err = float(sum(step_errors) / max(1, len(step_errors)))

    return {
        "sr": sr_val,
        "true_notes": true_cnt,
        "pred_notes": pred_cnt,
        "note_recall": note_recall,
        "note_precision": note_precision,
        "rhythm_f1": rhythm_f1,
        "silence_acc": silence_acc,
        "exact_4k": exact_4k,
        "ln_f1": ln_f1,
        "step_mae_ms": avg_step_err,
    }


def run_full_calibration_search(
    dataset_path: str,
    checkpoint_path: str,
    num_samples: int = 12,
    device_str: str = "auto",
):
    dev = torch.device("cuda" if torch.cuda.is_available() and device_str == "auto" else "cpu")

    ds_p = Path(dataset_path)
    if not ds_p.exists():
        candidates = [
            Path("dataset_processed/train_data.pt"),
            Path("dataset_processed/val_data.pt"),
            Path("train_data.pt"),
            Path("val_data.pt"),
            Path("dataset/train_data.pt"),
            Path("dataset/val_data.pt"),
            Path("data/train_data.pt"),
        ]
        found = False
        for c in candidates:
            if c.exists():
                ds_p = c
                found = True
                print(f"💡 Auto-detected existing dataset: {ds_p}")
                break
        if not found:
            raise FileNotFoundError(f"Could not find dataset file at '{dataset_path}'. Please pass --dataset path/to/train_data.pt")

    print(f"🔧 Loading dataset from: {ds_p}")
    try:
        dataset = torch.load(ds_p, map_location="cpu", weights_only=False)
    except TypeError:
        dataset = torch.load(ds_p, map_location="cpu")

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

    print("\n" + "=" * 125)
    print(" 🧪 COMPREHENSIVE MULTI-DIMENSIONAL CALIBRATION ON REAL HUMAN BEATMAPS (WITH NEURAL TIME STEP PACING)")
    print("=" * 125)
    print(f"{'STEP MODE':<12} | {'SCAFFOLD':<9} | {'CONF':<5} | {'GATE':<5} | {'TEMP':<5} | {'NOTE REC':<9} | {'RHYTHM F1':<10} | {'STEP MAE':<10} | {'SILENCE':<9} | {'SCORE':<6}")
    print("-" * 125)

    step_modes = ["neural_step", "dense_grid"]
    scaffold_options = [True, False]
    conf_thresholds = [0.0, 0.25]
    silence_gates = [0.15, 0.25]
    temperatures = [0.0, 0.30]

    best_score = -1.0
    best_config = {}

    for mode, use_scaff, conf, gate, temp in itertools.product(step_modes, scaffold_options, conf_thresholds, silence_gates, temperatures):
        tot_f1, tot_silence, tot_exact, tot_recall, tot_prec, tot_ln, tot_step_err = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        for s in samples:
            res = evaluate_sample_with_thresholds(
                model=model,
                sample=s,
                device=dev,
                silence_gate=gate,
                note_conf_thresh=conf,
                ln_thresh=0.78,
                temperature=temp,
                use_scaffold=use_scaff,
                step_mode=mode,
            )
            tot_f1 += res["rhythm_f1"]
            tot_silence += res["silence_acc"]
            tot_exact += res["exact_4k"]
            tot_recall += res["note_recall"]
            tot_prec += res["note_precision"]
            tot_ln += res["ln_f1"]
            tot_step_err += res["step_mae_ms"]

        n = len(samples)
        avg_f1 = (tot_f1 / n) * 100.0
        avg_silence = (tot_silence / n) * 100.0
        avg_exact = (tot_exact / n) * 100.0
        avg_rec = (tot_recall / n) * 100.0
        avg_prec = (tot_prec / n) * 100.0
        avg_ln = (tot_ln / n) * 100.0
        avg_step_mae = tot_step_err / n

        # Balanced Composite Optimization Score
        step_score = max(0.0, 100.0 - min(100.0, avg_step_mae))
        composite_score = (avg_rec * 0.35) + (avg_silence * 0.30) + (avg_f1 * 0.15) + (avg_exact * 0.10) + (step_score * 0.10)

        marker = "⭐ BEST" if composite_score > best_score else ""
        scaff_str = "Two-Stage" if use_scaff else "Pure-AI"
        mode_str = "Neural \u0394t" if mode == "neural_step" else "Dense Grid"
        print(
            f"{mode_str:<12} | {scaff_str:<9} | {conf:<5.2f} | {gate:<5.2f} | {temp:<5.2f} | "
            f"{avg_rec:<8.1f}% | {avg_f1:<9.1f}% | ±{avg_step_mae:<7.1f}ms | {avg_silence:<8.1f}% | {composite_score:<6.1f} {marker}"
        )

        if composite_score > best_score:
            best_score = composite_score
            best_config = {
                "step_mode": mode_str,
                "scaffold": scaff_str,
                "conf": conf,
                "silence": gate,
                "temp": temp,
                "recall": avg_rec,
                "f1": avg_f1,
                "silence_clean": avg_silence,
                "exact_4k": avg_exact,
                "precision": avg_prec,
                "step_mae": avg_step_mae,
                "score": composite_score,
            }

    print("=" * 125)
    print(f" 🏆 OPTIMAL MULTI-PARAM INTERPRETER CONFIGURATION:")
    print(f"  • Pacing / Step Advance Mode:            {best_config['step_mode']}")
    print(f"  • Architecture Mode:                     {best_config['scaffold']}")
    print(f"  • Note Confidence Threshold (P_active):  {best_config['conf']:.2f}")
    print(f"  • Silence Energy Gate (Noise Floor):     {best_config['silence']:.2f}")
    print(f"  • Sampling Temperature:                  {best_config['temp']:.2f}")
    print(f"  • Note Recall (Human Beat Hits):         {best_config['recall']:.1f}%")
    print(f"  • Silence Cleanliness:                   {best_config['silence_clean']:.1f}% (Zero phantom notes)")
    print(f"  • Exact 4K Chord Match:                  {best_config['exact_4k']:.1f}%")
    print(f"  • Neural Time Step MAE (Pacing Precision): ±{best_config['step_mae']:.1f} ms")
    print("=" * 125 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Parameter Interpreter Calibration")
    parser.add_argument("--dataset", type=str, default="dataset_processed/train_data.pt", help="Path to train_data.pt")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_3_transformer_best.pt", help="Path to checkpoint")
    parser.add_argument("--num_samples", type=int, default=12, help="Number of real ground-truth songs to calibrate over")
    args = parser.parse_args()

    run_full_calibration_search(
        dataset_path=args.dataset,
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
    )
