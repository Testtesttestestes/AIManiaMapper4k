#!/usr/bin/env python3
"""
Compares decoder settings of a trained hybrid checkpoint on the validation songs (no retraining).

    python hybrid_decode_ab.py --checkpoint checkpoints/hybrid/hybrid_best.pt --data dataset_hybrid

The network runs once per map; the same probabilities are then decoded with several settings:
  old       - decoder before duplicate suppression and LN release gaps (merge 3 ms, no hedge rules, 1 ms gap)
  default   - current defaults (merge 12 ms, hedge 60 ms / sum 1.5, duplicate-jack 50 ms)
  + variants of the hedge parameters.
Prints event F1, density and where surplus notes come from (duplicates next to a real note, isolated extras).
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from osu_mapper.hybrid.audio_features import beat_features
from osu_mapper.hybrid.dataset import HybridIndex
from osu_mapper.hybrid.inference import HybridPredictor, SongInputs
from osu_mapper.hybrid.metrics import per_ms_report, summarize_reports
from osu_mapper.hybrid.renderer import DecodeKnobs, decode_outputs, render_track

VARIANTS = {
    "old": dict(merge_ms=3, hedge_ms=0, jack_hedge_ms=0, ln_release_gap_ms=1, ln_release_gap_beats=0),
    "default": dict(),
    "jack only": dict(hedge_ms=0),
    "hedge only": dict(jack_hedge_ms=0),
    "hedge 80/1.6": dict(hedge_ms=80, hedge_sum=1.6),
    "hedge 45/1.4": dict(hedge_ms=45, hedge_sum=1.4),
    "LN gap 1/8": dict(ln_release_gap_beats=0.125),
}
COLS = [("event_f1@16", "F1@16", 1), ("event_f1@5", "F1@5", 1), ("density_ratio", "density", 1),
        ("surplus_duplicate_events_pct", "dup%", 0), ("surplus_isolated_events_pct", "isol%", 0),
        ("missed_events_pct", "miss%", 0), ("chord_size_acc", "chord", 1), ("ln_f1", "LN F1", 1),
        ("ln_length_mae_ms", "LNerr", 0), ("min_release_gap_ms", "minGap", 0)]


def min_release_gap(notes) -> float:
    """Shortest pause between an LN release and the next note on the same lane (nan if none)."""
    best = float("nan")
    by_lane = {}
    for n in sorted(notes):
        by_lane.setdefault(n.col, []).append(n)
    for lane in by_lane.values():
        for a, b in zip(lane, lane[1:]):
            if a.ln:
                g = float(b.start - a.end)
                best = g if not best <= g else best
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="checkpoints/hybrid/hybrid_best.pt")
    ap.add_argument("--data", default="dataset_hybrid")
    ap.add_argument("--maps", type=int, default=40, help="validation maps to use")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    targs = ck.get("args", {})
    predictor = HybridPredictor.from_checkpoint(args.checkpoint, device=args.device)
    index = HybridIndex(args.data)
    _, val_idx = index.split_by_song(float(targs.get("val_fraction", 0.1)), int(targs.get("seed", 42)))
    if not val_idx:
        raise SystemExit("no validation songs in this dataset")
    chosen = sorted(random.Random(0).sample(val_idx, min(args.maps, len(val_idx))))
    base = dict(onset_threshold=predictor.decode_threshold, ln_threshold=predictor.ln_threshold,
                onset_shift_ms=predictor.onset_shift_ms, ln_length_scale=predictor.ln_length_scale)
    reports = {name: [] for name in VARIANTS}
    for i, mi in enumerate(chosen, 1):
        entry, track = index.maps[mi], index.track(mi)
        feats = np.asarray(index.features(entry["song_id"], 0), dtype=np.float32)
        F = feats.shape[0]
        inputs = SongInputs(feats, index.coarse(entry["song_id"]), beat_features(track.timing_points, 0, F, track.frame_ms), F * track.frame_ms)
        out = predictor.predict(inputs, float(entry.get("sr", 0.0)))
        ref = render_track(track)
        for name, kw in VARIANTS.items():
            notes = decode_outputs(out, DecodeKnobs(**base, **kw), track.timing_points)
            rep = per_ms_report(notes, ref)
            rep["min_release_gap_ms"] = min_release_gap(notes)
            reports[name].append(rep)
        print(f"  [{i}/{len(chosen)}] {entry.get('version', '')}", flush=True)

    print(f"\nthreshold {base['onset_threshold']:.2f}, {len(chosen)} validation maps\n")
    print(f"{'variant':<14}" + "".join(f"{label:>9}" for _, label, _ in COLS))
    for name in VARIANTS:
        s = summarize_reports(reports[name])
        print(f"{name:<14}" + "".join(f"{s.get(key, float('nan')):>9.3f}" if frac else f"{s.get(key, float('nan')):>9.2f}" for key, _, frac in COLS))


if __name__ == "__main__":
    main()
