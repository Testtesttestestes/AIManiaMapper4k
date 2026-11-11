#!/usr/bin/env python3
"""
DEPRECATED together with the composer (see generate_composer.py).

How close the composer gets to the human maps of the same songs: one table, one number per measure.

    python eval_composer.py --songs originals/ --per_song 2 --workers 3
    python eval_composer.py --songs originals/ --save_maps eval_out/      # also keeps the composed .osu files

For every song (.osz with audio) the audio analysis is cached (--cache), the composer maps the star rating of
up to --per_song human difficulties (same audio file as the first .osu, rate edits skipped) and both maps are
measured:

* ``dens_r``     correlation of notes per 2 bars (does the map get dense where the human's does, on time?)
* ``lag_bars``   shift (in bars) that maximises that correlation: > 0 = the composer reacts late
* ``nps_ratio``  composer notes / human notes
* ``rhythm_js``  Jensen-Shannon distance of the rhythm (gap in beats) distributions; ``q4``: share of 1/4 gaps
* ``on_human`` / ``human_hit``  composer chords on a human chord time (+-10 ms) / human chords the composer hits
* ``jack``, ``ln``, ``chords``  shares (human / composer)
* ``copies``     chords that copy an earlier passage (exact + mirrored)
* ``rest``       composer notes where every human difficulty rests (>= 3 s gaps, after the last human note)
* ``lane4`` / ``lane6``  share of the human's 4- / 6-chord lane patterns (mirror-free) the composer also uses
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import tempfile
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from compare_zones import Diff, read_diffs, rest_spans
from osu_mapper.hybrid.beatmap import ManiaMap, Note, map_to_osu_text
from osu_mapper.hybrid.motifs import MIRROR_MASK, N_R, chord_seq, self_repeats, tokenize

MEASURES = ("dens_r", "lag_bars", "nps_ratio", "rhythm_js", "q4_h", "q4_c", "on_human", "human_hit", "jack_h", "jack_c",
            "ln_h", "ln_c", "chords_h", "chords_c", "copies_h", "copies_c", "rest", "lane4", "lane6", "sr_err")


def song_analysis_net(osz: Path, checkpoint: str, cache: str = "eval_cache"):
    """(SongAnalysis, sections, predictor, inputs) with the network, the way generate_composer.py does it."""
    from generate_hybrid import read_template
    from osu_mapper.hybrid.audio_features import load_audio
    from osu_mapper.hybrid.composer import analyze_song, segment_sections
    from osu_mapper.hybrid.inference import HybridPredictor, prepare_song_inputs

    predictor = HybridPredictor.from_checkpoint(Path(checkpoint), device="cpu")
    fcfg = predictor.feature_cfg
    with tempfile.TemporaryDirectory() as work:
        audio, timing, _ = read_template(osz, Path(work))
        wave = load_audio(audio, fcfg.sample_rate)
    dur = int(round(wave.size * 1000.0 / fcfg.sample_rate))
    import torch

    cf = Path(cache) / f"{osz.stem}_net.pt"
    if cf.exists():
        inputs = torch.load(cf, map_location="cpu", weights_only=False)
    else:
        inputs = prepare_song_inputs(wave, fcfg, timing if any(t.uninherited for t in timing) else None, keep_reps=True,
                                     device="cpu")
        Path(cache).mkdir(parents=True, exist_ok=True)
        torch.save(inputs, cf)
    an = analyze_song(inputs.feats, fcfg.slices(), timing, dur, fcfg.frame_ms)
    return an, segment_sections(an), predictor, inputs


def song_analysis(osz: Path, cache: Path):
    """(SongAnalysis, sections) of a template .osz; the audio features are cached as .npz."""
    from generate_hybrid import read_template
    from osu_mapper.hybrid.audio_features import FeatureConfig, compute_stacked_features, load_audio
    from osu_mapper.hybrid.composer import analyze_song, segment_sections

    cache.mkdir(parents=True, exist_ok=True)
    fcfg = FeatureConfig()
    with tempfile.TemporaryDirectory() as work:
        audio, timing, _ = read_template(osz, Path(work))
        cf = cache / f"{osz.stem}.npz"
        if cf.exists():
            z = np.load(cf)
            feats, dur = z["feats"], int(z["duration_ms"])
        else:
            wave = load_audio(audio, fcfg.sample_rate)
            dur = int(round(wave.size * 1000.0 / fcfg.sample_rate))
            feats = compute_stacked_features(wave, fcfg, device="cpu")
            np.savez_compressed(cf, feats=feats.astype(np.float32), duration_ms=dur)
    an = analyze_song(feats, fcfg.slices(), timing, dur, fcfg.frame_ms)
    return an, segment_sections(an)


def _js(p: np.ndarray, q: np.ndarray) -> float:
    p = p / max(1e-9, p.sum())
    q = q / max(1e-9, q.sum())
    m = 0.5 * (p + q)
    kl = lambda a, b: float((a[a > 0] * np.log2(a[a > 0] / b[a > 0])).sum())
    return math.sqrt(max(0.0, 0.5 * kl(p, m) + 0.5 * kl(q, m)))


def _jack_share(notes) -> float:
    last = {}
    j = 0
    for n in sorted(notes, key=lambda x: x.start):
        if n.start - last.get(n.col, -10 ** 9) < 160:
            j += 1
        last[n.col] = n.end if n.ln else n.start
    return j / max(1, len(notes))


def _grams(masks: np.ndarray, n: int) -> set:
    out = set()
    for i in range(masks.size - n + 1):
        a = tuple(int(x) for x in masks[i:i + n])
        out.add(min(a, tuple(int(MIRROR_MASK[x]) for x in a)))
    return out


def measure(human: Diff, notes: List[Note], an, rests, song_end: int, sr: float, lib=None) -> Dict[str, float]:
    hm, tp = human.m, human.m.timing_points
    H, C = chord_seq(hm.notes, tp), chord_seq(notes, tp)
    if C.size < 8:
        return {}
    d: Dict[str, float] = {}
    # density per 2 bars
    beats = an.beats
    bar = max(1, an.bar_beats) * 2
    edges = np.append(beats[::bar], an.duration_ms)
    hc = np.histogram([n.start for n in hm.notes], edges)[0].astype(float)
    cc = np.histogram([n.start for n in notes], edges)[0].astype(float)
    best, lag = -2.0, 0
    for s in range(-3, 4):
        a, b = (hc[:hc.size - s], cc[s:]) if s >= 0 else (hc[-s:], cc[:cc.size + s])
        if a.size > 4 and a.std() > 0 and b.std() > 0:
            r = float(np.corrcoef(a, b)[0, 1])
            if s == 0:
                d["dens_r"] = r
            if r > best + 1e-9:
                best, lag = r, s
    d.setdefault("dens_r", 0.0)
    d["lag_bars"] = 2.0 * lag
    d["nps_ratio"] = len(notes) / max(1, len(hm.notes))
    th, tc = tokenize(H), tokenize(C)
    rh = np.bincount(th["exact"][0] // 256, minlength=N_R).astype(float)
    rc = np.bincount(tc["exact"][0] // 256, minlength=N_R).astype(float)
    d["rhythm_js"] = _js(rh, rc)
    d["q4_h"], d["q4_c"] = rh[5] / rh.sum(), rc[5] / rc.sum()

    def near(x, ref):
        pos = np.clip(np.searchsorted(ref, x), 1, ref.size - 1)
        return float(np.mean(np.minimum(np.abs(ref[pos] - x), np.abs(ref[pos - 1] - x)) <= 10))

    d["on_human"], d["human_hit"] = near(C.times, H.times), near(H.times, C.times)
    d["jack_h"], d["jack_c"] = _jack_share(hm.notes), _jack_share(notes)
    d["ln_h"], d["ln_c"] = float(np.mean([n.ln for n in hm.notes])), float(np.mean([n.ln for n in notes]))
    d["chords_h"], d["chords_c"] = float(np.mean(H.masks & (H.masks - 1) > 0)), float(np.mean(C.masks & (C.masks - 1) > 0))
    for key, cs, tk in (("copies_h", H, th), ("copies_c", C, tc)):
        d[key] = sum(c.length for c in self_repeats(cs, tk) if c.kind in ("exact", "mirror")) / cs.size
    d["rest"] = sum(1 for n in notes if n.start > song_end + 500 or any(a <= n.start <= b for a, b in rests)) / max(1, len(notes))
    for n in (4, 6):
        A, B = _grams(H.masks, n), _grams(C.masks, n)
        d[f"lane{n}"] = len(A & B) / max(1, len(A))
    d["sr_err"] = abs(sr - human.sr)
    if lib is not None and "lanes" in lib.tables:
        from osu_mapper.hybrid.motifs import covered_share
        d["cov8_h"] = covered_share(th["lanes"][0], th["lanes"][1], lib.tables["lanes"], 8)
        d["cov8_c"] = covered_share(tc["lanes"][0], tc["lanes"][1], lib.tables["lanes"], 8)
    return d


def run_song(job) -> List[dict]:
    osz, cache, per_song, cfg_kw, patterns_path, motifs_path, save, use_motifs, checkpoint = job
    import os

    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    from osu_mapper.hybrid.composer import ComposerConfig, compose
    from osu_mapper.hybrid.patterns import load_default

    osz = Path(osz)
    hum = read_diffs(osz)
    if not hum:
        return []
    audio = min(hum, key=lambda d: d.order).audio
    hum = [d for d in hum if d.audio == audio]
    if per_song and len(hum) > per_song:  # spread over the star ratings, always including the hardest
        idx = np.unique(np.round(np.linspace(len(hum) - 1, 0, per_song)).astype(int))
        hum = [hum[i] for i in sorted(idx)]
    predictor = inputs = None
    if checkpoint:
        an, sections, predictor, inputs = song_analysis_net(osz, checkpoint, cache)
    else:
        an, sections = song_analysis(osz, Path(cache))
    pat = load_default(patterns_path, log=lambda *_: None)
    lib = None
    if motifs_path and Path(motifs_path).exists():
        from osu_mapper.hybrid.motifs import MotifLibrary
        lib = MotifLibrary.load(motifs_path, levels=["lanes"])
    rests, song_end = rest_spans(read_diffs(osz))
    out = []
    for h in hum:
        t0 = time.time()
        kw = dict(cfg_kw)
        if kw.pop("match_ln", True):  # the LN style is the mapper's choice, not the music's: give the human's LN share
            kw.setdefault("ln_ratio", float(np.mean([n.ln for n in h.m.notes])))
        cfg = ComposerConfig(**kw)
        hints = None
        if predictor is not None:
            from osu_mapper.hybrid.composer import NetHints
            from osu_mapper.hybrid.inference import style_from_dict
            style, smask = style_from_dict({})
            hints = NetHints.from_outputs(predictor.predict(inputs, h.sr, style, smask))
        res = compose(an, h.sr, cfg, hints=hints, sections=[dataclasses.replace(s) for s in sections], log=lambda *_: None,
                      patterns=pat, motifs=lib if use_motifs else None)
        d = measure(h, res.notes, an, sorted(rests + h.breaks), song_end, res.star_rating, lib)
        d.update(song=osz.stem, diff=h.name, sr=h.sr, secs=time.time() - t0)
        out.append(d)
        if save:
            Path(save).mkdir(parents=True, exist_ok=True)
            mm = ManiaMap(notes=res.notes, timing_points=h.m.timing_points, overall_difficulty=8.0,
                          metadata={"Title": osz.stem, "Artist": "", "Creator": "eval", "Version": f"{h.name} composer"})
            text = map_to_osu_text(mm)
            if getattr(res, "breaks", None):
                from osu_mapper.hybrid.composer import with_break_periods
                text = with_break_periods(text, res.breaks)
            (Path(save) / f"{osz.stem} [{h.name[:40]}].osu".replace("/", "_")).write_text(text, encoding="utf-8")
    return out


def table(rows: List[dict]) -> str:
    cols = ["dens_r", "lag_bars", "nps_ratio", "rhythm_js", "q4_h", "q4_c", "on_human", "human_hit", "jack_h", "jack_c",
            "ln_h", "ln_c", "copies_h", "copies_c", "rest", "lane4", "lane6", "cov8_h", "cov8_c", "sr_err"]
    out = ["| song | diff | sr | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 3)]
    for r in rows:
        out.append(f"| {r['song'][:14]} | {r['diff'][:18]} | {r['sr']:.2f} | " + " | ".join(f"{r.get(c, float('nan')):.2f}" for c in cols) + " |")
    out.append("| **mean** | | | " + " | ".join(f"**{np.nanmean([r.get(c, np.nan) for r in rows]):.2f}**" for c in cols) + " |")
    return "\n".join(out)


def main() -> None:
    print("[deprecated] eval_composer.py evaluates the deprecated composer", file=sys.stderr)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--songs", required=True, help="folder of .osz files with audio (or one .osz)")
    ap.add_argument("--cache", default="eval_cache")
    ap.add_argument("--per_song", type=int, default=2)
    ap.add_argument("--variants", type=int, default=1)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--patterns", default="test_NN/patterns.npz")
    ap.add_argument("--motifs", default="test_NN/motifs.npz")
    ap.add_argument("--no_motifs", action="store_true", help="compose without the motif library (still used to measure)")
    ap.add_argument("--save_maps", default=None)
    ap.add_argument("--checkpoint", default=None, help="hybrid network: compose with its hints, as the console does")
    ap.add_argument("--out", default=None, help="write the rows as JSON")
    ap.add_argument("--label", default="")
    ap.add_argument("--no_match_ln", action="store_true", help="let the composer pick the LN style itself")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="ComposerConfig fields, e.g. bar_follow=0")
    args = ap.parse_args()
    p = Path(args.songs)
    songs = [p] if p.suffix.lower() == ".osz" else sorted(p.glob("*.osz"))
    cfg_kw = dict(variants=args.variants, match_ln=not args.no_match_ln)
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg_kw[k] = type(getattr(__import__("osu_mapper.hybrid.composer", fromlist=["ComposerConfig"]).ComposerConfig(), k))(v)
    jobs = [(str(s), args.cache, args.per_song, cfg_kw, args.patterns, args.motifs, args.save_maps, not args.no_motifs,
             args.checkpoint)
            for s in songs]
    rows = []
    t0 = time.time()
    if args.workers <= 1:
        for j in jobs:
            rows += run_song(j)
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for r in pool.map(run_song, jobs):
                rows += r
    print(f"## eval {args.label} ({len(rows)} maps, {time.time() - t0:.0f}s)\n")
    print(table(rows))
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
