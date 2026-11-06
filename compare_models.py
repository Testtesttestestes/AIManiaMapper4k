#!/usr/bin/env python3
"""
Compares hybrid models (or settings) on songs with human maps: every run generates every difficulty of every song with
exactly the pipeline of generate_hybrid.py (the maps you play), and each generated difficulty is compared with the
human map of the same star rating - not only "does it match" but how it plays.

    python compare_models.py --maps "new maps" --data "dataset hybrid" \
        --run "v2|test_NN/hybrid_best.pt" --run "new|checkpoints/hybrid/hybrid_best.pt" \
        --run "new 1 pass|checkpoints/hybrid/hybrid_best.pt|--refine_steps 1"

--run "NAME|CHECKPOINT|EXTRA generate_hybrid ARGS" (extra args optional). --maps: a folder of .osz (4K mania).
Songs that are in the training dataset (--data) are skipped: a model would be judged on maps it learnt.

Per difficulty, generated vs human (all numbers already used elsewhere in the code):
  star rating     |SR - human SR|, and whether it is within 0.1*
  match           note-time F1 at 16 ms and 2 ms, LN F1 (how close to what a human did)
  density graph   correlation of the 2 s density curves (notes rise and fall with the song like the human map)
  jacks           share of same-lane repeats faster than 150 ms (generated and human)
  one-hand run    longest run of events on one hand (< 200 ms apart) - long runs are uncomfortable
  hand balance    imbalance of left/right hand in 2 s windows (0 = equal)
  chords          share of single notes / jumps / hands / quads
  pattern entropy variety of lane patterns (too low = monotone, too high = random)
  LN              share of long notes, and LN shape similarity (lengths, releases, what is pressed during a hold; 1 = same)
  duplicates      notes <= 30 ms from another hit (unplayable doubles)
The summary shows the mean of each and, for the "like the human map" numbers, the mean distance to the human value.
Results: <out>/results.json and one line per difficulty in the log.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from osu_mapper.hybrid.beatmap import ManiaMap, parse_osu_mania

ROOT = Path(__file__).resolve().parent


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def read_osz(path: Path) -> List[ManiaMap]:
    maps = []
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if name.lower().endswith(".osu"):
                m = parse_osu_mania(z.read(name).decode("utf-8", "replace"))
                if m is not None and m.notes:
                    maps.append(m)
    return maps


def dataset_keys(data: Optional[str]) -> Tuple[set, set]:
    """(source file names, normalized 'artist title') of the songs in a prepared dataset."""
    if not data:
        return set(), set()
    idx = json.loads((Path(data) / "index.json").read_text(encoding="utf-8"))
    files, titles = set(), set()
    for s in idx.get("songs", {}).values():
        files.add(Path(str(s.get("source", ""))).name.lower())
    for m in idx.get("maps", []):
        files.add(Path(str(m.get("source", ""))).name.lower())
        titles.add(norm(str(m.get("piece", ""))))
    return files, titles


def chord_shares(notes) -> np.ndarray:
    g: Dict[int, int] = {}
    for n in notes:
        g[n.start] = g.get(n.start, 0) + 1
    c = np.bincount(np.clip(np.array(list(g.values())), 1, 4) - 1, minlength=4).astype(float)
    return c / max(1.0, c.sum())


def quality(notes, ref, timing, duration_ms: int) -> Dict[str, float]:
    from osu_mapper.hybrid.composer import pattern_entropy
    from osu_mapper.hybrid.ergonomics import ergonomics_report
    from osu_mapper.hybrid.metrics import per_ms_report
    from osu_mapper.hybrid.patterns import density_correlation, ln_quality, ln_similarity

    r = per_ms_report(notes, ref)
    e, eh = ergonomics_report(notes), ergonomics_report(ref)
    lq = ln_similarity(ln_quality(notes, timing), ln_quality(ref, timing))
    ch, chh = chord_shares(notes), chord_shares(ref)
    return {
        "f1_16": r["event_f1@16"], "f1_2": r["event_f1@2"], "ln_f1": r["ln_f1"],
        "density_corr": density_correlation(notes, ref, duration_ms),
        "jacks": e.get("fast_jack_pct", 0.0), "jacks_human": eh.get("fast_jack_pct", 0.0),
        "one_hand_run": e.get("max_one_hand_run", 0.0), "one_hand_run_human": eh.get("max_one_hand_run", 0.0),
        "hand_imbalance": e.get("hand_imbalance", 0.0), "hand_imbalance_human": eh.get("hand_imbalance", 0.0),
        "entropy": pattern_entropy(notes), "entropy_human": pattern_entropy(ref),
        "ln_share": float(np.mean([n.ln for n in notes])) if notes else 0.0,
        "ln_share_human": float(np.mean([n.ln for n in ref])),
        "ln_shape": float(np.mean(list(lq.values()))) if lq else float("nan"),
        "chords_l1": float(np.abs(ch - chh).sum()),
        "duplicates_pct": r["surplus_duplicate_events_pct"],
        "missed_pct": r["missed_events_pct"],
    }


SUMMARY = (
    ("SR within 0.1*", "sr_ok", "share"), ("|SR error|", "sr_err", "mean"),
    ("note F1 @16 ms", "f1_16", "mean"), ("note F1 @2 ms", "f1_2", "mean"), ("LN F1", "ln_f1", "mean"),
    ("density graph corr.", "density_corr", "mean"),
    ("fast jacks % (|vs human|)", "jacks", "dist"), ("one-hand run (|vs human|)", "one_hand_run", "dist"),
    ("hand imbalance (|vs human|)", "hand_imbalance", "dist"), ("pattern entropy (|vs human|)", "entropy", "dist"),
    ("LN share (|vs human|)", "ln_share", "dist"), ("LN shape similarity", "ln_shape", "mean"),
    ("chord mix distance", "chords_l1", "mean"), ("duplicates %", "duplicates_pct", "mean"), ("missed %", "missed_pct", "mean"),
)


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    out = {}
    for label, key, how in SUMMARY:
        if how == "dist":
            v = [abs(r[key] - r[key + "_human"]) for r in rows]
            out[label] = float(np.mean(v))
            out[label.split(" (")[0] + " (generated / human)"] = (float(np.mean([r[key] for r in rows])),
                                                                 float(np.mean([r[key + "_human"] for r in rows])))
        elif how == "share":
            out[label] = float(np.mean([r[key] for r in rows]))
        else:
            v = [r[key] for r in rows if not (isinstance(r[key], float) and np.isnan(r[key]))]
            out[label] = float(np.mean(v)) if v else float("nan")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", required=True, help="folder with .osz of 4K mania maps by humans")
    ap.add_argument("--run", action="append", required=True, help='"NAME|CHECKPOINT|EXTRA ARGS" (repeat for every model / setting)')
    ap.add_argument("--data", default=None, help="training dataset: its songs are skipped")
    ap.add_argument("--allow_seen", action="store_true", help="also use songs that are in --data")
    ap.add_argument("--out", default="output_beatmaps/compare", help="generated maps and results.json")
    ap.add_argument("--max_songs", type=int, default=0)
    args = ap.parse_args(argv)

    import shlex
    import generate_hybrid
    from osu_mapper.hybrid.network_fit import star_rating

    runs = []
    for spec in args.run:
        parts = spec.split("|")
        if len(parts) < 2:
            sys.exit(f"--run '{spec}': expected NAME|CHECKPOINT[|EXTRA ARGS]")
        runs.append((parts[0].strip(), parts[1].strip(), shlex.split(parts[2]) if len(parts) > 2 else []))
    files, titles = dataset_keys(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    songs = sorted(Path(args.maps).glob("*.osz"))
    results: Dict[str, List[Dict[str, float]]] = {name: [] for name, _, _ in runs}
    used = 0
    for osz in songs:
        maps = read_osz(osz)
        if not maps:
            continue
        piece = norm(f"{maps[0].metadata.get('Artist', '')} {maps[0].metadata.get('Title', '')}")
        seen = osz.name.lower() in files or piece in titles or norm(maps[0].metadata.get("Title", "")) in titles
        if seen and not args.allow_seen:
            print(f"[skip] {osz.name}: in the training dataset")
            continue
        if args.max_songs and used >= args.max_songs:
            break
        used += 1
        refs = []
        for m in maps:
            refs.append((round(star_rating(m.notes, m.timing_points, m.overall_difficulty), 2), m))
        refs.sort(key=lambda x: x[0])
        duration = max(n.end for n in maps[0].notes) + 2000
        print(f"\n[song] {osz.name}: " + ", ".join(f"{m.metadata.get('Version', '?')} {sr:.2f}*" for sr, m in refs), flush=True)
        for name, ckpt, extra in runs:
            gen = out_dir / re.sub(r"[^\w.-]+", "_", f"{osz.stem}__{name}")
            gen_osz = gen.with_suffix(".osz")
            argv_g = ["--template", str(osz), "--sr", ",".join(f"{sr:.2f}" for sr, _ in refs), "-c", ckpt,
                      "-o", str(gen_osz), "--creator", f"eval {name}"] + extra
            print(f"  [{name}] generate_hybrid {' '.join(shlex.quote(a) for a in argv_g)}", flush=True)
            try:
                generate_hybrid.main(argv_g)
            except SystemExit as e:
                if e.code not in (None, 0):
                    print(f"  [{name}] generation failed ({e.code})")
                    continue
            got = {}
            for m in read_osz(gen_osz):
                mt = re.search(r"Hybrid ([0-9.]+)\*", m.metadata.get("Version", ""))
                if mt:
                    got[round(float(mt.group(1)), 2)] = m
            for sr, ref in refs:
                g = got.get(sr)
                if g is None:
                    continue
                gsr = star_rating(g.notes, ref.timing_points, g.overall_difficulty)
                q = quality(g.notes, ref.notes, ref.timing_points, duration)
                q.update(sr=sr, sr_got=gsr, sr_err=abs(gsr - sr), sr_ok=float(abs(gsr - sr) <= 0.1), song=osz.name,
                         version=ref.metadata.get("Version", ""))
                results[name].append(q)
                print(f"  [{name}] {ref.metadata.get('Version', '')[:20]:20s} {sr:5.2f}* -> {gsr:5.2f}* | F1 {q['f1_16']:.2f}/{q['f1_2']:.2f} "
                      f"| density corr {q['density_corr']:.2f} | jacks {q['jacks']:.0f}% (human {q['jacks_human']:.0f}%) "
                      f"| one-hand run {q['one_hand_run']:.0f} ({q['one_hand_run_human']:.0f}) | entropy {q['entropy']:.2f} ({q['entropy_human']:.2f}) "
                      f"| LN {100 * q['ln_share']:.0f}% ({100 * q['ln_share_human']:.0f}%), shape {q['ln_shape']:.2f} | dup {q['duplicates_pct']:.1f}%", flush=True)
    if not any(results.values()):
        sys.exit("no difficulty was compared (all songs skipped as seen in training, or generation failed)")
    summary = {name: summarize(rows) for name, rows in results.items() if rows}
    n = {name: len(rows) for name, rows in results.items()}
    print(f"\n==== {used} songs | difficulties per run: {n}")
    names = list(summary)
    w = max(len(x) for x in names) + 2
    print(f"{'':32s}" + "".join(f"{x:>{max(w, 16)}s}" for x in names))
    for label in summary[names[0]]:
        vals = []
        for x in names:
            v = summary[x][label]
            vals.append(f"{v[0]:.2f} / {v[1]:.2f}" if isinstance(v, tuple) else f"{v:.3f}")
        print(f"{label:32s}" + "".join(f"{v:>{max(w, 16)}s}" for v in vals))
    print("(|vs human| and distances: lower is closer to the human maps; F1, correlation, similarity: higher is better)")
    (out_dir / "results.json").write_text(json.dumps({"summary": summary, "rows": results}, indent=1, default=float), encoding="utf-8")
    print(f"[done] {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
