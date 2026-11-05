#!/usr/bin/env python3
"""
Composer: planned osu!mania 4K maps that hit the requested star rating exactly (osu_mapper/hybrid/composer.py).

    python generate_composer.py --template "song.osz" --sr 2,3,4,5,6
    python generate_composer.py song.mp3 --sr 4.5 --bpm 174 --offset 120
    python generate_composer.py --template "song.osz" --sr 5 --creativity 0.8 --variants 6
    python generate_composer.py --template "song.osz" --sr 3.5 --no_model          # audio + learned patterns only

What decides what:
  * the network (checkpoint, optional) - WHERE musically meaningful notes are and which lanes it likes;
  * the learned pattern model (learn_patterns.py, checkpoints/hybrid/patterns.npz) - what real maps of this
    star rating do: lane transitions, chords, divisors, jacks, holds, bursts, density contrast;
  * the composer - sections and repeats of the song, density plan, pattern families (stairs, jumpstream,
    chordjack, trills, melodic, LN...), speed-ups, and the star-rating search on the final map.
"""

from __future__ import annotations

import argparse
import dataclasses
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from generate_hybrid import load_reference, parse_sr_list, parse_style, read_template
from osu_mapper.hybrid.audio_features import FeatureConfig, beat_features, compute_stacked_features, load_audio
from osu_mapper.hybrid.beatmap import ManiaMap, RawTimingPoint, map_to_osu_text
from osu_mapper.hybrid.composer import FAMILIES, ComposerConfig, NetHints, analyze_song, compose, segment_sections
from osu_mapper.hybrid.patterns import density_correlation, density_graph, load_default


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", nargs="?", default=None, help="audio file or template .osz/.osu")
    ap.add_argument("--template", default=None, help="template .osz / .osu / folder: its audio and timing points are used")
    ap.add_argument("--sr", nargs="*", default=["3.5"], help="target star rating(s), e.g. --sr 2,3,4,5")
    ap.add_argument("--checkpoint", "-c", default="checkpoints/hybrid/hybrid_best.pt", help="hybrid network (optional)")
    ap.add_argument("--no_model", action="store_true", help="do not use the network even if the checkpoint exists")
    ap.add_argument("--patterns", default=None, help="learned pattern model (default checkpoints/hybrid/patterns.npz)")
    ap.add_argument("--output", "-o", default="output_beatmaps/composer_map.osz")
    ap.add_argument("--title", default=None)
    ap.add_argument("--artist", default=None)
    ap.add_argument("--creator", default="AIManiaMapper Composer")
    ap.add_argument("--version", default=None, help="difficulty name prefix")
    ap.add_argument("--bpm", type=float, default=None)
    ap.add_argument("--offset", type=float, default=None)
    ap.add_argument("--creativity", type=float, default=0.5, help="0 = conservative, 1 = adventurous (pattern choice, speed-ups, "
                                                                     "how far it may go beyond what is typical for the star rating)")
    ap.add_argument("--contrast", type=float, default=1.0, help="density contrast between sections x what real maps do (0 = flat, 2 = extreme)")
    ap.add_argument("--variants", type=int, default=4, help="maps composed per difficulty; the critic keeps the best")
    ap.add_argument("--sr_tolerance", type=float, default=0.05)
    ap.add_argument("--families", nargs="*", default=None, choices=list(FAMILIES), help="allowed pattern families (default: all)")
    ap.add_argument("--ln_ratio", type=float, default=None, help="share of long notes (default: learned for the star rating x sustained sounds)")
    ap.add_argument("--no_ln", action="store_true", help="taps only")
    ap.add_argument("--max_chord", type=int, default=4)
    ap.add_argument("--no_repeat_memory", action="store_true", help="map repeated sections independently")
    ap.add_argument("--net_weight", type=float, default=1.0, help="weight of the network hints (0 = audio + patterns only)")
    ap.add_argument("--sr_guidance", type=float, default=1.0, help="network classifier-free guidance (see generate_hybrid.py)")
    ap.add_argument("--style", nargs="*", default=[], help="network style controls key=value (see generate_hybrid.py)")
    ap.add_argument("--disable_reps", nargs="*", default=[], metavar="NAME")
    ap.add_argument("--reference", default=None, help="reference map of the same audio: event F1, density graph correlation")
    ap.add_argument("--reference_diff", default="")
    ap.add_argument("--hitsounds", action="store_true")
    ap.add_argument("--od", type=float, default=8.0)
    ap.add_argument("--hp", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="composer_"))
    try:
        template = args.template
        if template is None and args.audio and Path(args.audio).suffix.lower() in (".osz", ".osu"):
            template = args.audio
        meta: Dict[str, str] = {}
        timing: List[RawTimingPoint] = []
        timing_from_user = False
        if template:
            audio_path, timing, meta = read_template(Path(template), work)
            timing_from_user = any(tp.uninherited for tp in timing)
        elif args.audio:
            audio_path = Path(args.audio)
        else:
            sys.exit("give an audio file or --template")
        if args.bpm is not None:
            timing = [tp for tp in timing if not tp.uninherited] + [RawTimingPoint(float(args.offset or 0.0), 60000.0 / args.bpm, 4, 0, 0, 70, True, 0)]
            timing_from_user = True

        predictor = None
        ckpt = Path(args.checkpoint)
        if not args.no_model and ckpt.exists() and args.net_weight > 0:
            from osu_mapper.hybrid.inference import HybridPredictor
            predictor = HybridPredictor.from_checkpoint(ckpt, device=args.device, disable_reps=args.disable_reps)
            print(f"[model] {ckpt}: network hints on (weight {args.net_weight:g})")
        else:
            print("[model] no network: notes come from the audio analysis and the learned patterns")
        patterns = load_default(args.patterns)
        if patterns is None:
            print("[patterns] no pattern model (python learn_patterns.py --data dataset_hybrid): built-in defaults are used, "
                  "chords / jacks / density will not be calibrated to real maps")
        fcfg = predictor.feature_cfg if predictor is not None else FeatureConfig()

        t0 = time.time()
        wave = load_audio(audio_path, fcfg.sample_rate)
        duration_ms = int(round(wave.size * 1000.0 / fcfg.sample_rate))
        dev = "cuda" if (predictor is not None and predictor.device.type == "cuda") else "cpu"
        if predictor is not None:
            from osu_mapper.hybrid.inference import prepare_song_inputs
            inputs = prepare_song_inputs(wave, fcfg, timing if timing_from_user else None, keep_reps=True, device=dev,
                                         skip_external=args.disable_reps)
            feats = inputs.feats
        else:
            feats = compute_stacked_features(wave, fcfg, device=dev)
        slices = fcfg.slices()
        if not any(tp.uninherited for tp in timing):
            from osu_mapper.hybrid.audio_features import estimate_tempo
            reps = {k: feats[:, a:b] for k, (a, b) in slices.items() if k in ("onset", "fine_env")}
            bpm, off = estimate_tempo(reps, fcfg.frame_ms)
            timing = [RawTimingPoint(float(off), 60000.0 / bpm, 4, 0, 0, 70, True, 0)] + timing
            print(f"[timing] estimated red line: {bpm:.2f} BPM, offset {off:.0f} ms (use --bpm/--offset or a template for exact timing)")
        an = analyze_song(feats, slices, timing, duration_ms, fcfg.frame_ms)
        sections = segment_sections(an)
        n_rep = sum(1 for s in sections if s.repeat_of is not None)
        print(f"[audio] {audio_path.name}: {duration_ms / 1000:.1f}s, {len(sections)} sections ({n_rep} repeats), analysis in {time.time() - t0:.1f}s")

        ref_map = load_reference(args.reference, args.reference_diff) if args.reference else None
        user_style = parse_style(args.style)
        title = args.title or meta.get("Title") or audio_path.stem
        artist = args.artist or meta.get("Artist") or "Unknown Artist"
        from osu_mapper.pipeline.packager import BeatmapMetadata, BeatmapPackager

        cfg = ComposerConfig(
            creativity=float(np.clip(args.creativity, 0.0, 1.0)), contrast=args.contrast,
            ln_ratio=0.0 if args.no_ln else args.ln_ratio, max_chord=args.max_chord, variants=args.variants,
            sr_tolerance=args.sr_tolerance, families=args.families, net_weight=args.net_weight,
            repeat_memory=not args.no_repeat_memory, seed=args.seed, od=args.od,
        )
        diff_items = []
        for target in parse_sr_list(args.sr):
            t1 = time.time()
            hints = None
            if predictor is not None:
                from osu_mapper.hybrid.inference import style_from_dict
                style, smask = style_from_dict(user_style)
                out = predictor.predict(inputs, target, style, smask, sr_guidance=args.sr_guidance)
                hints = NetHints.from_outputs(out)
            print(f"[{target:.2f}*] composing {cfg.variants} variant(s)")
            res = compose(an, target, cfg, hints=hints, sections=[dataclasses.replace(s) for s in sections],
                          log=print, patterns=patterns)
            notes = res.notes
            version = f"{args.version + ' ' if args.version else ''}Composer {target:.2f}*"
            print(res.plan_text())
            print(f"  density |{density_graph(notes, duration_ms)}|")
            print(f"[{version}] {len(notes)} notes ({sum(n.ln for n in notes)} LN), star rating {res.star_rating:.2f} "
                  f"(target {target:.2f}), level {res.level:.3f}, {time.time() - t1:.1f}s")
            if ref_map is not None:
                from osu_mapper.hybrid.metrics import per_ms_report
                r = per_ms_report(notes, ref_map.notes)
                print(f"  reference |{density_graph(ref_map.notes, duration_ms)}|")
                print(f"    vs reference: density graph correlation {density_correlation(notes, ref_map.notes, duration_ms):.2f} | "
                      f"event F1 @16ms {r['event_f1@16']:.3f} | density x{r['density_ratio']:.2f}")
            m = ManiaMap(notes=notes, timing_points=timing, overall_difficulty=args.od, hp_drain=args.hp,
                         metadata={"Title": title, "Artist": artist, "Creator": args.creator, "Version": version,
                                   "AudioFilename": audio_path.name})
            text = map_to_osu_text(m)
            if args.hitsounds:
                from osu_mapper.hybrid.hitsounds import apply_hitsounds_to_text, compute_hitsounds
                text = apply_hitsounds_to_text(text, compute_hitsounds(notes, feats, slices, timing, fcfg.frame_ms))
            diff_items.append((BeatmapMetadata(title=title, artist=artist, creator=args.creator, version=version,
                                               audio_filename=audio_path.name), text))
        out_path = BeatmapPackager.package_multi_diff_osz(diff_items, str(audio_path), args.output)
        print(f"[done] {out_path}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
