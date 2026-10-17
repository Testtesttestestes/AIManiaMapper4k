#!/usr/bin/env python3
"""
Algorithmic lane polishing of an existing 4K map (any .osu / .osz, generated or not).

    python polish_map.py output_beatmaps/hybrid_map.osz --strength 0.5
    python polish_map.py my_map.osu --strength 0.7 -o my_map_polished.osu
    python polish_map.py map.osz --strength 0.5 --hitsounds          # + volume / sample set / additions from the audio
    python polish_map.py map.osz --strength 0 --hitsounds            # hitsounds only, lanes untouched

Note times, chord sizes and long-note lengths stay exactly; only lanes are re-assigned against fast jacks,
one-hand passages and anchors (osu_mapper/hybrid/ergonomics.py). For an .osz every 4K difficulty is
polished and written into a new .osz next to the input (suffix _polished) with " (polished)" in its name.
"""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

from osu_mapper.hybrid.beatmap import parse_osu_mania
from osu_mapper.hybrid.ergonomics import ergonomics_report, format_report, improve_ergonomics


def polish_text(text: str, strength: float, tag: str, audio=None, hs_args=None) -> str:
    m = parse_osu_mania(text)
    before = ergonomics_report(m.notes)
    notes = improve_ergonomics(m.notes, m.timing_points, strength) if strength > 0 else sorted(m.notes)
    print(f"  [{m.metadata.get('Version', '')}] {format_report(before)}  ->  {format_report(ergonomics_report(notes))}")
    head, _, _ = text.partition("[HitObjects]")
    head = re.sub(r"(?m)^(Version:.*?)\s*$", lambda g: g.group(1) + tag, head, count=1)
    head = re.sub(r"(?m)^BeatmapID:.*$", "BeatmapID:0", head, count=1)
    lines = []
    for n in sorted(notes):
        x = int(n.col * 512 / 4 + 256 / 4)
        lines.append(f"{x},192,{n.start},128,0,{n.end}:0:0:0:0:" if n.ln else f"{x},192,{n.start},1,0,0:0:0:0:")
    out = head + "[HitObjects]\n" + "\n".join(lines) + "\n"
    if audio is not None:
        from osu_mapper.hybrid.hitsounds import apply_hitsounds_to_text, compute_hitsounds, hitsound_summary
        feats, slices = audio
        hs = compute_hitsounds(notes, feats, slices, m.timing_points, 10, hs_args["volume"], not hs_args["no_additions"], hs_args["set"])
        out = apply_hitsounds_to_text(out, hs)
        print(f"      hitsounds: {hitsound_summary(hs)}")
    return out


def audio_features(raw: bytes, ext: str):
    from osu_mapper.hybrid.audio_features import FeatureConfig, compute_stacked_features, load_audio

    cfg = FeatureConfig()
    wave = load_audio(raw, cfg.sample_rate, ext=ext)
    return compute_stacked_features(wave, cfg, device="auto"), cfg.slices()


def audio_name(text: str) -> str:
    m = re.search(r"(?m)^AudioFilename:\s*(.+?)\s*$", text)
    return m.group(1) if m else ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--hitsounds", action="store_true", help="also set per-note volume, sample set and additions from the audio")
    ap.add_argument("--hitsound_volume", default="40,90")
    ap.add_argument("--hitsound_set", default="auto", choices=["auto", "normal", "soft", "drum"])
    ap.add_argument("--no_additions", action="store_true")
    args = ap.parse_args()
    hs_args = {"volume": tuple(int(v) for v in args.hitsound_volume.split(",")), "set": args.hitsound_set, "no_additions": args.no_additions}
    feats_cache = {}
    src = Path(args.input)
    tag = (f" (polished {args.strength:g})" if args.strength > 0 else "") + (" (hitsounded)" if args.hitsounds else "")
    if src.suffix.lower() == ".osz":
        dst = Path(args.output) if args.output else src.with_name(src.stem + "_polished.osz")
        with zipfile.ZipFile(src) as zin:
            members = [(info, zin.read(info.filename)) for info in zin.infolist()]
        contents = {info.filename: data for info, data in members}
        with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
            for info, data in members:
                name = info.filename
                if name.lower().endswith(".osu"):
                    try:
                        text = data.decode("utf-8", errors="replace")
                        audio = None
                        if args.hitsounds:
                            aname = audio_name(text)
                            if aname not in feats_cache:
                                feats_cache[aname] = audio_features(contents[aname], Path(aname).suffix)
                            audio = feats_cache[aname]
                        data = polish_text(text, args.strength, tag, audio, hs_args).encode("utf-8")
                        name = name[:-4] + " polished.osu"
                    except Exception as err:
                        print(f"  [skip] {name}: {err}")
                zout.writestr(name, data)
    else:
        dst = Path(args.output) if args.output else src.with_name(src.stem + " polished.osu")
        text = src.read_text(encoding="utf-8", errors="replace")
        audio = audio_features((src.parent / audio_name(text)).read_bytes(), Path(audio_name(text)).suffix) if args.hitsounds else None
        dst.write_text(polish_text(text, args.strength, tag, audio, hs_args), encoding="utf-8")
    print(f"[done] {dst}")


if __name__ == "__main__":
    main()
