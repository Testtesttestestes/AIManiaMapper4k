#!/usr/bin/env python3
"""
Pre-launch check #2: very long tracks (ranked maps are mostly <= 6 min, what about 60 min?).

Failure modes this test looks for:
  * only the first minutes get mapped / mapping stops early,
  * the whole track is mapped but with big holes where music is playing,
  * density drifts with position (the network "knowing" where in the song it is),
  * memory/time growing faster than linearly.

Modes:
    --repeat N        the same song N times in a row: every copy must get (almost) the same map
    --concat A B ...  different songs back to back (a DJ-mix style 1 h file)
    --synthetic_minutes M   synthetic music (no files needed)

    python hybrid_long_track_test.py --checkpoint checkpoints/hybrid/hybrid_best.pt --audio song.mp3 --repeat 15 --sr 4
    python hybrid_long_track_test.py --checkpoint checkpoints/hybrid/hybrid_best.pt --synthetic_minutes 60 --sr 4
    python hybrid_long_track_test.py --untrained --synthetic_minutes 60     # machinery only, no checkpoint

Exit code 1 if a check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from osu_mapper.hybrid.audio_features import FeatureConfig, load_audio
from osu_mapper.hybrid.inference import HybridPredictor, prepare_song_inputs, resolve_device
from osu_mapper.hybrid.metrics import match_times
from osu_mapper.hybrid.model import HybridParamNet, build_config
from osu_mapper.hybrid.renderer import DecodeKnobs, decode_outputs
from osu_mapper.hybrid.synthetic import synth_song


def peak_rss_mb() -> float:
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 1024.0 if sys.platform != "darwin" else r / (1024.0 * 1024.0)
    except Exception:
        try:
            import psutil
            return psutil.Process().memory_info().peak_wset / (1024.0 * 1024.0)  # type: ignore[attr-defined]
        except Exception:
            return float("nan")


def active_audio_mask(wave: np.ndarray, sr: int, block_ms: int = 100) -> np.ndarray:
    n = int(sr * block_ms / 1000)
    m = wave.size // n
    rms = np.sqrt((wave[: m * n].reshape(m, n) ** 2).mean(axis=1) + 1e-12)
    ref = np.median(rms[rms > 1e-4]) if (rms > 1e-4).any() else 1.0
    return rms > max(1e-4, 0.05 * ref)


def find_holes(event_times_ms: np.ndarray, active: np.ndarray, duration_ms: int, hole_sec: float, block_ms: int = 100) -> List[Tuple[float, float, float]]:
    """Gaps between consecutive events (and track start/end) longer than hole_sec that are mostly music."""
    bounds = np.concatenate([[0], event_times_ms, [duration_ms]]).astype(np.float64)
    holes = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if (b - a) / 1000.0 < hole_sec:
            continue
        i0, i1 = int(a // block_ms), int(b // block_ms)
        seg = active[i0:i1]
        frac = float(seg.mean()) if seg.size else 0.0
        if frac >= 0.8:
            holes.append((a / 1000.0, b / 1000.0, frac))
    return holes


def main() -> None:
    ap = argparse.ArgumentParser(description="Long-track behaviour test of the hybrid mapper")
    ap.add_argument("--checkpoint", "-c", default="checkpoints/hybrid/hybrid_best.pt")
    ap.add_argument("--untrained", action="store_true", help="random tiny network: tests windowing/rendering/memory only")
    ap.add_argument("--audio", default=None)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--concat", nargs="*", default=[])
    ap.add_argument("--synthetic_minutes", type=float, default=0.0)
    ap.add_argument("--sr", type=float, default=4.0)
    ap.add_argument("--hole_sec", type=float, default=4.0, help="a gap with music longer than this is a hole")
    ap.add_argument("--min_repeat_f1", type=float, default=0.95)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="hybrid_long_track_report.json")
    args = ap.parse_args()

    dev = resolve_device(args.device)
    if args.untrained:
        fcfg = FeatureConfig()
        torch.manual_seed(0)
        model = HybridParamNet(build_config(fcfg.dims(), fcfg.total_dim(), "tiny"))
        predictor = HybridPredictor(model, fcfg, 2048, dev)
        print("[warn] --untrained: random network, densities are meaningless; checks windowing, determinism, memory, coverage")
    else:
        if not Path(args.checkpoint).exists():
            sys.exit(f"checkpoint '{args.checkpoint}' not found (or use --untrained)")
        predictor = HybridPredictor.from_checkpoint(args.checkpoint, device=args.device)
        fcfg = predictor.feature_cfg
    sr_audio = fcfg.sample_rate

    pieces: List[np.ndarray] = []
    piece_names: List[str] = []
    if args.synthetic_minutes > 0:
        total, seed = 0.0, 100
        while total < args.synthetic_minutes * 60.0:
            dur = min(180.0, args.synthetic_minutes * 60.0 - total)
            w, _, _ = synth_song(seed, max(20.0, dur), sr_audio)
            pieces.append(w)
            piece_names.append(f"synthetic#{seed}")
            total += w.size / sr_audio
            seed += 1
    elif args.concat:
        for p in args.concat:
            pieces.append(load_audio(p, sr_audio))
            piece_names.append(Path(p).name)
    elif args.audio:
        base = load_audio(args.audio, sr_audio)
        pieces = [base] * max(1, args.repeat)
        piece_names = [Path(args.audio).name] * len(pieces)
    else:
        sys.exit("choose --audio (+--repeat), --concat or --synthetic_minutes")

    repeat_mode = bool(args.audio) and args.repeat > 1 and not args.concat and args.synthetic_minutes <= 0
    if repeat_mode:  # keep copies sample-aligned to whole frames so every copy sees identical features
        hop = fcfg.hop
        L = (pieces[0].size // hop) * hop
        pieces = [pieces[0][:L]] * len(pieces)
    wave = np.concatenate(pieces).astype(np.float32)
    duration_ms = int(wave.size * 1000 / sr_audio)
    print(f"[track] {len(pieces)} piece(s), {duration_ms / 60000:.1f} min")

    t0 = time.time()
    inputs = prepare_song_inputs(wave, fcfg)
    t_feat = time.time() - t0
    t0 = time.time()
    out = predictor.predict(inputs, args.sr)
    t_pred = time.time() - t0
    threshold = predictor.decode_threshold
    if args.untrained:
        threshold = float(np.quantile(1.0 - out.time_probs[:, 0], 0.97))
    t0 = time.time()
    notes = decode_outputs(out, DecodeKnobs(onset_threshold=threshold, ln_threshold=predictor.ln_threshold))
    t_dec = time.time() - t0
    ev = np.array(sorted({n.start for n in notes}), dtype=np.int64)
    print(f"[speed] features {t_feat:.1f}s, network {t_pred:.1f}s, algorithm {t_dec:.2f}s, peak RSS {peak_rss_mb():.0f} MB")

    active = active_audio_mask(wave, sr_audio)
    holes = find_holes(ev, active, duration_ms, args.hole_sec)
    minutes = int(np.ceil(duration_ms / 60000))
    per_min = np.bincount(np.minimum(ev // 60000, minutes - 1), minlength=minutes) if ev.size else np.zeros(minutes, dtype=np.int64)
    act_min = np.array([active[i * 600:(i + 1) * 600].mean() if active[i * 600:(i + 1) * 600].size else 0 for i in range(minutes)])
    first_s = ev[0] / 1000 if ev.size else float("nan")
    last_s = ev[-1] / 1000 if ev.size else float("nan")
    active_idx = np.flatnonzero(active)
    music_start, music_end = (active_idx[0] * 0.1, active_idx[-1] * 0.1) if active_idx.size else (0.0, 0.0)

    report: Dict[str, object] = {
        "duration_min": duration_ms / 60000,
        "notes": len(notes),
        "events": int(ev.size),
        "first_note_s": first_s,
        "last_note_s": last_s,
        "music_start_s": music_start,
        "music_end_s": music_end,
        "holes": holes,
        "events_per_minute": per_min.tolist(),
        "active_fraction_per_minute": [round(float(a), 3) for a in act_min],
        "seconds_features": t_feat,
        "seconds_network": t_pred,
        "seconds_algorithm": t_dec,
        "peak_rss_mb": peak_rss_mb(),
        "threshold": threshold,
    }
    checks: List[Tuple[str, bool, str]] = []
    checks.append(("mapping reaches the end", bool(ev.size) and last_s >= music_end - args.hole_sec, f"last note {last_s:.1f}s, music ends {music_end:.1f}s"))
    checks.append(("mapping starts with the music", bool(ev.size) and first_s <= music_start + args.hole_sec, f"first note {first_s:.1f}s, music starts {music_start:.1f}s"))
    checks.append(("no holes in music", len(holes) == 0, f"{len(holes)} hole(s) > {args.hole_sec}s" + (f", longest {max(b - a for a, b, _ in holes):.1f}s" if holes else "")))

    if repeat_mode:
        L_ms = pieces[0].size * 1000.0 / sr_audio
        copies = []
        for k in range(len(pieces)):
            sel = ev[(ev >= k * L_ms) & (ev < (k + 1) * L_ms)] - int(round(k * L_ms))
            copies.append(sel)
        f1s, ratios = [], []
        ref_mid = copies[min(1, len(copies) - 1)]
        for k, c in enumerate(copies[1:-1] if len(copies) > 2 else copies[1:], start=1):
            tp = len(match_times(c, ref_mid, 2))
            f1s.append(2 * tp / max(1, len(c) + len(ref_mid)))
            ratios.append(len(c) / max(1, len(ref_mid)))
        report["repeat_event_f1@2ms"] = f1s
        report["repeat_density_ratio"] = ratios
        min_f1 = min(f1s) if f1s else 1.0
        cv = float(np.std([len(c) for c in copies[1:-1]]) / max(1e-9, np.mean([len(c) for c in copies[1:-1]]))) if len(copies) > 3 else 0.0
        report["repeat_density_cv"] = cv
        checks.append(("every repetition gets the same map", min_f1 >= args.min_repeat_f1, f"min event F1@2ms between copies {min_f1:.3f}"))
        checks.append(("density does not drift with position", cv <= 0.05, f"coefficient of variation of notes per copy {cv:.3f}"))

    print("\n events per minute: " + " ".join(str(int(x)) for x in per_min))
    print(" music fraction   : " + " ".join(f"{a:.1f}" for a in act_min))
    ok_all = True
    for name, ok, info in checks:
        ok = bool(ok)
        ok_all = ok_all and ok
        print(f" [{'PASS' if ok else 'FAIL'}] {name}: {info}")
    report["checks"] = [{"name": n, "ok": bool(o), "info": i} for n, o, i in checks]
    Path(args.out).write_text(json.dumps(report, indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v)), encoding="utf-8")
    print(f" report: {args.out}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
