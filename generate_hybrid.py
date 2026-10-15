#!/usr/bin/env python3
"""
Generate osu!mania 4K maps with the hybrid mapper (network -> parameters -> universal algorithm).

    python generate_hybrid.py song.mp3 --sr 3.5
    python generate_hybrid.py --template "my_template.osz" --sr 2.5,3.5,4.5,5.5
    python generate_hybrid.py song.mp3 --sr 4 --bpm 174 --offset 120 --snap_ms 8
    python generate_hybrid.py song.mp3 --sr 4 --style ln_ratio=0.35 jack_ratio=0.02
    python generate_hybrid.py song.mp3 --sr 4.8 --style_from "Grimes - We Appreciate Power.osz" --no_ln

Works for tracks of any length (1 hour is fine): the network runs window by window and the
algorithm renders the whole track in one pass. Density follows the requested star rating through the
network's own conditioning; the rosu-pp star rating of the result is printed for reference.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

from osu_mapper.hybrid.audio_features import load_audio
from osu_mapper.hybrid.beatmap import ManiaMap, RawTimingPoint, _kv, _split_sections, map_to_osu_text, parse_timing_point
from osu_mapper.hybrid.inference import (
    HybridPredictor,
    estimated_timing_points,
    prepare_song_inputs,
    star_rating_of,
    style_from_dict,
)
from osu_mapper.hybrid.renderer import DecodeKnobs, decode_outputs
from osu_mapper.hybrid.audio_features import beat_features

AUDIO_EXTS = (".mp3", ".ogg", ".wav", ".flac", ".m4a", ".opus")


def read_template(path: Path, work: Path) -> Tuple[Path, List[RawTimingPoint], Dict[str, str]]:
    """Audio file, timing points and metadata of a template .osz / .osu / song folder (any game mode)."""
    if path.suffix.lower() == ".osz":
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(work / "template")
        folder = work / "template"
        osu_files = sorted(folder.rglob("*.osu"))
    elif path.suffix.lower() == ".osu":
        folder = path.parent
        osu_files = [path]
    else:
        folder = path
        osu_files = sorted(folder.glob("*.osu"))
    if not osu_files:
        raise FileNotFoundError(f"no .osu file in template '{path}'")
    text = osu_files[0].read_text(encoding="utf-8", errors="ignore")
    sections = _split_sections(text)
    general = _kv(sections.get("General", []))
    meta = _kv(sections.get("Metadata", []))
    tps = [tp for tp in (parse_timing_point(line) for line in sections.get("TimingPoints", [])) if tp is not None]
    audio_name = general.get("AudioFilename", "")
    audio = folder / audio_name if audio_name else None
    if audio is None or not audio.exists():
        cands = [p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_EXTS]
        if not cands:
            raise FileNotFoundError(f"no audio file in template '{path}'")
        audio = max(cands, key=lambda p: p.stat().st_size)
    return audio, tps, meta


def parse_sr_list(items: List[str]) -> List[float]:
    out: List[float] = []
    for item in items:
        for tok in str(item).replace(",", " ").split():
            out.append(round(float(tok), 2))
    return out or [3.5]


def reference_style(path: str, diff: str = ""):
    """Style vector of an existing map (.osz/.osu): "map like this one". The star rating is left to --sr."""
    import numpy as np

    from osu_mapper.hybrid.beatmap import parse_osu_mania
    from osu_mapper.hybrid.solver import STYLE_KEYS, compute_style, solve_map

    p = Path(path)
    texts: List[Tuple[str, str]] = []
    if p.suffix.lower() == ".osz":
        with zipfile.ZipFile(p) as z:
            texts = [(n, z.read(n).decode("utf-8", errors="replace")) for n in z.namelist() if n.lower().endswith(".osu")]
    else:
        texts = [(p.name, p.read_text(encoding="utf-8", errors="replace"))]
    maps = []
    for name, text in texts:
        try:
            m = parse_osu_mania(text)
        except Exception:
            continue
        if m.notes and (not diff or diff.lower() in name.lower() or diff.lower() in m.metadata.get("Version", "").lower()):
            maps.append((name, m))
    if not maps:
        raise SystemExit(f"--style_from: no 4K mania difficulty{' matching ' + repr(diff) if diff else ''} in {path}")
    name, m = maps[0]
    style, known = compute_style(solve_map(m), m.notes, None, m.overall_difficulty)
    i = STYLE_KEYS.index("sr")
    known[i] = 0.0
    shown = ", ".join(f"{k}={style[j]:.3f}" for j, k in enumerate(STYLE_KEYS) if known[j])
    print(f"[style] from {Path(path).name} [{m.metadata.get('Version', name)}]: {shown} (normalized)")
    return np.asarray(style, np.float32), np.asarray(known, np.float32)


def parse_style(items: List[str]) -> Dict[str, float]:
    res: Dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--style expects key=value, got '{item}'")
        k, v = item.split("=", 1)
        res[k.strip()] = float(v)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Hybrid osu!mania 4K map generator")
    ap.add_argument("audio", nargs="?", default=None, help="audio file (.mp3/.ogg/.wav/.flac) or template .osz/.osu")
    ap.add_argument("--template", default=None, help="template .osz / .osu / folder: its audio and timing points are used")
    ap.add_argument("--sr", nargs="*", default=["3.5"], help="target star rating(s), e.g. --sr 2.5,3.5,4.5")
    ap.add_argument("--checkpoint", "-c", default="checkpoints/hybrid/hybrid_best.pt")
    ap.add_argument("--output", "-o", default="output_beatmaps/hybrid_map.osz")
    ap.add_argument("--title", default=None)
    ap.add_argument("--artist", default=None)
    ap.add_argument("--creator", default="AIManiaMapper Hybrid")
    ap.add_argument("--version", default=None, help="difficulty name prefix")
    ap.add_argument("--bpm", type=float, default=None, help="manual BPM (with --offset): writes a red line and enables beat features")
    ap.add_argument("--offset", type=float, default=None)
    ap.add_argument("--beat_features", default="auto", choices=["auto", "on", "off"], help="feed red-line phase to the network (auto: only for template/manual timing)")
    ap.add_argument("--style", nargs="*", default=[], help="style controls key=value: " + "sr nps ln_ratio chord_mean jack_ratio alt_ratio roll_ratio symmetric_ratio explicit_ratio offgrid_ratio od")
    ap.add_argument("--style_from", default=None, help="copy the style of an existing map (.osz/.osu): pattern mix, LN share, density...; --style keys override it")
    ap.add_argument("--style_diff", default="", help="difficulty name (substring) inside --style_from .osz")
    ap.add_argument("--no_ln", action="store_true", help="taps only: no long notes at all (style ln_ratio=0 + every LN becomes a tap)")
    ap.add_argument("--threshold", type=float, default=None, help="event threshold (default: calibrated value stored in the checkpoint)")
    ap.add_argument("--ln_threshold", type=float, default=None, help="tap-vs-LN threshold (default: calibrated value from the checkpoint)")
    ap.add_argument("--jack_min_ms", type=int, default=0, help="forbid lane repeats faster than this many ms")
    ap.add_argument("--max_chord", type=int, default=4)
    ap.add_argument("--hedge_ms", type=int, default=60, help="two unsure events closer than this = one note, keep the stronger (0 = off)")
    ap.add_argument("--hedge_sum", type=float, default=1.5, help="...unsure means their probabilities add up to less than this")
    ap.add_argument("--jack_hedge_ms", type=int, default=50, help="drop an event that repeats a lane hit less than this ago (0 = off)")
    ap.add_argument("--merge_ms", type=int, default=12, help="events closer than this are always merged")
    ap.add_argument("--ln_gap_ms", type=int, default=40, help="minimum pause between an LN release and the next note on its lane")
    ap.add_argument("--ln_gap_beats", type=float, default=0.25, help="...and at least this fraction of a beat (0.125 = 1/8, 0 = off)")
    ap.add_argument("--snap_ms", type=float, default=0.0, help="snap note starts to the nearest red-line tick within this tolerance")
    ap.add_argument("--ln_snap_ms", type=float, default=None, help="snap LN releases to the PREVIOUS tick within this tolerance (default: = --snap_ms)")
    ap.add_argument("--snap_divisors", default="1,2,4,3,6,8,12,16", help="allowed beat divisors for snapping, e.g. 1,2,4 for a straight 4/4 song")
    ap.add_argument("--onset_shift_ms", type=int, default=None, help="constant note-time correction (default: calibrated value from the checkpoint)")
    ap.add_argument("--ln_scale", type=float, default=None, help="LN length multiplier (default: calibrated value from the checkpoint)")
    ap.add_argument("--lane_temperature", type=float, default=0.0)
    ap.add_argument("--no_stagger", action="store_true")
    ap.add_argument("--od", type=float, default=8.0)
    ap.add_argument("--hp", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        sys.exit(f"checkpoint '{ckpt}' not found. Train first: python train_hybrid.py --data dataset_hybrid")

    work = Path(tempfile.mkdtemp(prefix="hybrid_gen_"))
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

        predictor = HybridPredictor.from_checkpoint(ckpt, device=args.device)
        fcfg = predictor.feature_cfg
        t0 = time.time()
        wave = load_audio(audio_path, fcfg.sample_rate)
        use_beat = args.beat_features == "on" or (args.beat_features == "auto" and timing_from_user)
        inputs = prepare_song_inputs(wave, fcfg, timing if use_beat else None, keep_reps=True, device="cuda" if predictor.device.type == "cuda" else "cpu")
        if not any(tp.uninherited for tp in timing):
            timing = estimated_timing_points(inputs.reps or {}, fcfg.frame_ms) + timing
            print(f"[timing] estimated red line: {60000.0 / timing[0].beat_len:.2f} BPM, offset {timing[0].time:.0f} ms (use --bpm/--offset or a template for exact timing)")
        if use_beat and inputs.beat[:, -1].max() == 0:
            inputs.beat = beat_features(timing, 0, inputs.num_frames, fcfg.frame_ms)
        print(f"[audio] {audio_path.name}: {inputs.duration_ms / 1000:.1f}s, {inputs.num_frames} frames, features in {time.time() - t0:.1f}s")

        user_style = parse_style(args.style)
        if args.no_ln:
            user_style.setdefault("ln_ratio", 0.0)
        style, smask = style_from_dict(user_style)
        if args.style_from:
            ref_style, ref_mask = reference_style(args.style_from, args.style_diff)
            take = (ref_mask > 0) & (smask == 0)
            style[take], smask[take] = ref_style[take], 1.0
        knobs = DecodeKnobs(
            onset_threshold=args.threshold if args.threshold is not None else predictor.decode_threshold,
            ln_threshold=2.0 if args.no_ln else (args.ln_threshold if args.ln_threshold is not None else predictor.ln_threshold),
            jack_min_ms=args.jack_min_ms,
            max_chord=args.max_chord,
            merge_ms=args.merge_ms,
            hedge_ms=args.hedge_ms,
            hedge_sum=args.hedge_sum,
            jack_hedge_ms=args.jack_hedge_ms,
            ln_release_gap_ms=args.ln_gap_ms,
            ln_release_gap_beats=args.ln_gap_beats,
            snap_ms=args.snap_ms,
            ln_snap_ms=args.ln_snap_ms,
            snap_divisors=tuple(int(x) for x in args.snap_divisors.replace(" ", "").split(",") if x),
            onset_shift_ms=args.onset_shift_ms if args.onset_shift_ms is not None else predictor.onset_shift_ms,
            ln_length_scale=args.ln_scale if args.ln_scale is not None else predictor.ln_length_scale,
            lane_temperature=args.lane_temperature,
            use_stagger=not args.no_stagger,
            seed=args.seed,
        )

        print(f"[decode] event threshold {knobs.onset_threshold:.2f}, LN threshold {knobs.ln_threshold:.2f}, "
              f"note shift {knobs.onset_shift_ms:+d} ms, LN scale {knobs.ln_length_scale:.2f}, snap {knobs.snap_ms:g} ms "
              f"(LN release snap {knobs.ln_snap_ms if knobs.ln_snap_ms is not None else knobs.snap_ms:g} ms back), divisors {knobs.snap_divisors}, "
              f"hedge {knobs.hedge_ms} ms / sum {knobs.hedge_sum:g}, duplicate-jack {knobs.jack_hedge_ms} ms, "
              f"LN release gap >= {knobs.ln_release_gap_ms} ms and {knobs.ln_release_gap_beats:g} beat")
        title = args.title or meta.get("Title") or audio_path.stem
        artist = args.artist or meta.get("Artist") or "Unknown Artist"
        audio_name = audio_path.name
        diff_items = []
        from osu_mapper.pipeline.packager import BeatmapMetadata, BeatmapPackager

        for target in parse_sr_list(args.sr):
            t1 = time.time()
            out = predictor.predict(inputs, target, style, smask, batch_size=args.batch_size)
            notes = decode_outputs(out, knobs, timing)
            sr_val, th = star_rating_of(notes, timing, args.od), knobs.onset_threshold
            version = f"{args.version + ' ' if args.version else ''}Hybrid {target:.2f}*"
            m = ManiaMap(
                notes=notes,
                timing_points=timing,
                overall_difficulty=args.od,
                hp_drain=args.hp,
                metadata={"Title": title, "Artist": artist, "Creator": args.creator, "Version": version, "AudioFilename": audio_name},
            )
            text = map_to_osu_text(m)
            last = notes[-1].start / 1000.0 if notes else 0.0
            print(f"[{version}] {len(notes)} notes ({sum(n.ln for n in notes)} LN), last note {last:.1f}s / {inputs.duration_ms / 1000:.1f}s, "
                  f"threshold {th:.3f}, star rating {sr_val:.2f} (target {target:.2f}) in {time.time() - t1:.1f}s")
            diff_items.append((BeatmapMetadata(title=title, artist=artist, creator=args.creator, version=version, audio_filename=audio_name), text))

        out_path = BeatmapPackager.package_multi_diff_osz(diff_items, str(audio_path), args.output)
        print(f"[done] {out_path}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
