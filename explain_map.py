#!/usr/bin/env python3
"""
Why does the v5 network put notes where a map has them? Integrated-gradients attribution of every note's timing
decision to the sound sources (osu_mapper/v5/explain.py), for any map - a human one or a generated one.

    python explain_map.py "song.osz" --diff "Insane"
    python explain_map.py "song.osz" --diff "Insane" --checkpoint checkpoints/v5/v5_best.pt --max_events 120

Prints, for the bursts (runs of 1/4 and faster, "rice") and for the other notes separately, and for every part of
the song (intro / verse / chorus / drop ...): which sources decided where the notes go - vocals, drums, bass, other
(each demucs stem with its attack curve), the mix (the other audio representations: mel, MERT, ...), the part of the
song, the beat grid - and the input groups behind them. If the network had learned that the rice of a phrase follows
the singing, the vocals share of the burst notes would be clearly higher than of the other notes.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from generate_hybrid import load_reference, read_template
from osu_mapper.hybrid.audio_features import FeatureConfig, beat_features, load_audio
from osu_mapper.hybrid.inference import prepare_song_inputs, resolve_device
from osu_mapper.hybrid.metrics import bursts
from osu_mapper.v5.data import song_inputs
from osu_mapper.v5.explain import explain_notes, format_shares, summarize
from osu_mapper.v5.model import V5Config, V5Net
from osu_mapper.v5.sections import describe, find_sections


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map", help=".osz or .osu (with its audio next to it)")
    ap.add_argument("--diff", default="", help="difficulty name (substring) inside the .osz")
    ap.add_argument("--checkpoint", "-c", default="checkpoints/v5/v5_best.pt")
    ap.add_argument("--max_events", type=int, default=60, help="notes explained per group (bursts / others)")
    ap.add_argument("--steps", type=int, default=8, help="integration steps (more = more exact, slower)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    dev = resolve_device(args.device)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = V5Config.from_dict(ck["config"])
    model = V5Net(cfg)
    model.load_state_dict(ck.get("ema") or ck["model"])
    model.to(dev).eval()
    fcfg = FeatureConfig.from_dict(ck["feature_config"])
    work = Path(tempfile.mkdtemp(prefix="v5_explain_"))
    try:
        audio_path, timing, _meta = read_template(Path(args.map), work)
        ref = load_reference(args.map, args.diff)
        notes = sorted(ref.notes)
        if not notes:
            sys.exit("no notes in that difficulty")
        timing = list(ref.timing_points) or timing
        t0 = time.time()
        wave = load_audio(audio_path, fcfg.sample_rate)
        inputs = prepare_song_inputs(wave, fcfg, timing, device="cuda" if dev.type == "cuda" else "cpu")
        feats_all = song_inputs(inputs.feats, fcfg.slices(), timing, highlight=cfg.highlight)
        beat_all = beat_features(timing, 0, inputs.feats.shape[0], fcfg.frame_ms)
        secs = find_sections(inputs.feats, fcfg.slices(), timing, highlight=cfg.highlight)
        print(f"[map] {ref.metadata.get('Version', '')}: {len(notes)} notes; features in {time.time() - t0:.0f}s")
        print(f"[sections] {describe(secs)}")
        t = np.array(sorted({n.start for n in notes}))
        in_burst = np.zeros(t.size, dtype=bool)
        for a, b in bursts(t, timing):
            in_burst |= (t >= a) & (t <= b)
        sr = float(ref.metadata.get("_sr", 0.0)) or 4.0
        try:
            from osu_mapper.hybrid.inference import star_rating_of
            sr = float(star_rating_of(notes, timing, ref.overall_difficulty))
        except Exception:
            pass
        t1 = time.time()
        rb = explain_notes(model, feats_all, beat_all, notes, timing, sr, times=t[in_burst].tolist(),
                           max_events=args.max_events, steps=args.steps, device=dev) if in_burst.any() else []
        ro = explain_notes(model, feats_all, beat_all, notes, timing, sr, times=t[~in_burst].tolist(),
                           max_events=args.max_events, steps=args.steps, device=dev)
        print(f"[why] {len(rb) + len(ro)} notes explained in {time.time() - t1:.0f}s (star rating {sr:.2f})")
        if rb:
            print(f"  burst notes ({int(in_burst.sum())} in the map): {format_shares(summarize(rb), 7)}")
            print(f"      input groups: {format_shares(summarize(rb, groups=True), 6)}")
        print(f"  other notes ({int((~in_burst).sum())}): {format_shares(summarize(ro), 7)}")
        print(f"      input groups: {format_shares(summarize(ro, groups=True), 6)}")
        rows = rb + ro
        print("  by part of the song:")
        for s in secs:
            r = [x for x in rows if s.start * 10 <= x["time_ms"] < s.end * 10]
            if r:
                print(f"    {s.start / 100:6.1f}s {s.label:<9} {len(r):3d} notes: {format_shares(summarize(r), 4)}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
