#!/usr/bin/env python3
"""
Zone-by-zone comparison of composed maps with the human maps of the same song, down to the hit-object lines.

    python compare_zones.py --original song.osz --generated output_beatmaps/composer_map.osz
    python compare_zones.py --pairs_dir originals/ --generated_dir generated/ --report zone_report.md

Zones are the composer's own sections (same audio analysis as generate_composer.py), or 8-bar blocks with
--bar_zones / when the audio cannot be read. Every generated difficulty is paired with the human difficulty of
the nearest star rating. Per zone it measures, human vs composer:

* notes per second, chord sizes, rhythm (gaps in beats: 1/1 .. 1/16, triplets), jacks, long notes, hold lengths;
* **placement**: share of the composer's chords that sit on a millisecond where the human put a chord (+-10 ms)
  and share of the human chords the composer also hit;
* **patterns**: the lane / shape / exact motifs of the motif library (mine_patterns.py) the zone contains - what
  the human used there that the composer never does, and the other way round; the pattern variety (distinct
  4-chord shapes per 100 chords);
* how much of the zone copies an earlier passage (exact, mirrored, variation).

Zones are ranked by a combined distance: the report lists, over all songs, the worst and the best zones with both
maps' raw [HitObjects] lines side by side, and the statistics that separate bad zones from good ones.
"""

from __future__ import annotations

import argparse
import math
import re
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from osu_mapper.hybrid.beatmap import ManiaMap, Note, RawTimingPoint, parse_osu_mania
from osu_mapper.hybrid.motifs import (N_R, POP, RHYTHM_NAMES, MotifLibrary, canonical_hash, chord_seq, draw,
                                      name_motif, self_repeats, tokenize)


# ------------------------------------------------------------------------------------------------------------
# reading maps
# ------------------------------------------------------------------------------------------------------------

@dataclass
class Diff:
    name: str
    text: str
    m: ManiaMap
    sr: float
    lines: List[Tuple[int, str]] = field(default_factory=list)   # (time, raw hit-object line)
    breaks: List[Tuple[int, int]] = field(default_factory=list)  # [Events] break periods "2,start,end"


def _sr(text: str, m: ManiaMap) -> float:
    try:
        import rosu_pp_py

        return float(rosu_pp_py.Difficulty().calculate(rosu_pp_py.Beatmap(content=text)).stars)
    except Exception:
        from osu_mapper.hybrid.composer import star_rating

        return star_rating(m.notes, m.timing_points, m.overall_difficulty)


def hit_lines(text: str) -> List[Tuple[int, str]]:
    out, on = [], False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("["):
            on = s == "[HitObjects]"
            continue
        if on and s:
            p = s.split(",")
            if len(p) >= 3 and p[2].lstrip("-").isdigit():
                out.append((int(p[2]), s))
    return out


def break_periods(text: str) -> List[Tuple[int, int]]:
    """Break periods of the [Events] section ("2,start,end" or "Break,start,end"): the game shows a rest there."""
    out, on = [], False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("["):
            on = s == "[Events]"
            continue
        if on:
            p = s.split(",")
            if len(p) >= 3 and p[0] in ("2", "Break"):
                try:
                    out.append((int(float(p[1])), int(float(p[2]))))
                except ValueError:
                    pass
    return out


def read_diffs(path: Path) -> List[Diff]:
    texts = []
    if path.suffix.lower() == ".osz":
        with zipfile.ZipFile(path) as z:
            texts = [(n, z.read(n).decode("utf-8", errors="replace")) for n in z.namelist() if n.lower().endswith(".osu")]
    elif path.suffix.lower() == ".osu":
        texts = [(path.name, path.read_text(encoding="utf-8", errors="replace"))]
    else:
        texts = [(p.name, p.read_text(encoding="utf-8", errors="replace")) for p in sorted(path.glob("*.osu"))]
    out = []
    for name, text in texts:
        m = parse_osu_mania(text)
        if m is None or len(m.notes) < 20:
            continue
        out.append(Diff(m.metadata.get("Version", name), text, m, _sr(text, m), hit_lines(text), break_periods(text)))
    return sorted(out, key=lambda d: d.sr)


# ------------------------------------------------------------------------------------------------------------
# zones
# ------------------------------------------------------------------------------------------------------------

def composer_zones(osz: Path, log=print):
    """(start, end, label) of the composer's sections for the song of a template .osz."""
    from generate_hybrid import read_template
    from osu_mapper.hybrid.audio_features import FeatureConfig, compute_stacked_features, load_audio
    from osu_mapper.hybrid.composer import analyze_song, segment_sections

    with tempfile.TemporaryDirectory() as work:
        audio, timing, _ = read_template(osz, Path(work))
        fcfg = FeatureConfig()
        wave = load_audio(audio, fcfg.sample_rate)
        dur = int(round(wave.size * 1000.0 / fcfg.sample_rate))
        feats = compute_stacked_features(wave, fcfg, device="cpu")
        an = analyze_song(feats, fcfg.slices(), timing, dur, fcfg.frame_ms)
        secs = segment_sections(an)
    out = []
    for i, s in enumerate(secs):
        rep = f" repeat of {s.repeat_src_ms / 1000:.1f}s" if s.repeat_src_ms is not None else ""
        out.append((s.start_ms, s.end_ms, f"#{i} intensity {s.intensity:.2f}{rep}"))
    return out


def bar_zones(m: ManiaMap, duration_ms: int, bars: int = 8):
    reds = sorted((tp for tp in m.timing_points if tp.uninherited and tp.beat_len > 1), key=lambda r: r.time)
    if not reds:
        return [(a, min(duration_ms, a + 15000), f"{a / 1000:.0f}s") for a in range(0, duration_ms, 15000)]
    out, t, k = [], 0.0, 0
    edges = [0.0]
    r = reds[0]
    t = r.time
    while t < duration_ms:
        nxt = [x for x in reds if x.time > t + 1]
        bl = max([x for x in reds if x.time <= t + 1] or [r], key=lambda x: x.time).beat_len
        step = bl * 4 * bars
        t2 = t + step
        if nxt and nxt[0].time < t2:
            t2 = nxt[0].time
        edges.append(t2)
        t = t2
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a > 500:
            out.append((int(a), int(b), f"{a / 1000:.0f}s"))
    return out


# ------------------------------------------------------------------------------------------------------------
# zone statistics
# ------------------------------------------------------------------------------------------------------------

def rest_spans(hum: List[Diff], min_gap_ms: int = 3000) -> Tuple[List[Tuple[int, int]], int]:
    """Where the human maps rest: their break periods plus every gap of >= min_gap_ms that NO human difficulty fills
    (silence / intro / outro), and the end of the mapped song (last human note of any difficulty)."""
    spans = [b for d in hum for b in d.breaks]
    times = np.unique(np.concatenate([[n.start for n in d.m.notes] + [n.end for n in d.m.notes] for d in hum]))
    g = np.diff(times)
    for i in np.flatnonzero(g >= min_gap_ms):
        spans.append((int(times[i]) + 250, int(times[i + 1]) - 250))
    return sorted(spans), int(times[-1])


STAT_NAMES = ("nps", "chord", "jack", "ln", "rhythm", "lanes", "shape", "motifs", "variety", "placement")


@dataclass
class ZoneStats:
    n: int
    nps: float
    chord_hist: np.ndarray
    rhythm_hist: np.ndarray
    jack: float
    ln: float
    lane_hist: np.ndarray       # lane-pair transitions 16x16 flattened
    shape4: Counter
    motifs: Dict[str, Counter]
    copies: np.ndarray          # chords inside exact / mirror / variation / rhythm copies
    times: np.ndarray
    in_rest: int = 0            # notes inside a human rest span (break period / silence no human maps)
    after_end: int = 0          # notes after the last human note of the song


def _js(p: np.ndarray, q: np.ndarray) -> float:
    p = p / max(1e-9, p.sum())
    q = q / max(1e-9, q.sum())
    m = 0.5 * (p + q)

    def kl(a, b):
        k = a > 0
        return float((a[k] * np.log2(a[k] / b[k])).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def map_tokens(m: ManiaMap):
    cs = chord_seq(m.notes, m.timing_points)
    return cs, tokenize(cs)


def motif_hits(tokens: np.ndarray, mirrored: Optional[np.ndarray], lib: Optional[MotifLibrary], level: str,
               lengths=(4, 6, 8)) -> List[Tuple[int, int, int]]:
    """(start index, length, motif id) of every library motif occurrence in a token stream."""
    if lib is None or level not in lib.tables:
        return []
    t = lib.tables[level]
    out = []
    for L in lengths:
        for i in range(0, tokens.size - L + 1):
            k = t.find(tokens[i:i + L].tolist())
            if k >= 0:
                out.append((i, L, k))
    return out


def zone_stats(m: ManiaMap, cs, tok, hits: Dict[str, list], copies, a: int, b: int, rests=None, song_end: int = 10 ** 9) -> ZoneStats:
    sel = np.flatnonzero((cs.times >= a) & (cs.times < b))
    dur = max(0.001, (b - a) / 1000.0)
    notes = [n for n in m.notes if a <= n.start < b]
    chord_hist = np.bincount(np.clip(POP[cs.masks[sel]], 1, 4) - 1, minlength=4).astype(float)
    r = tok["exact"][0][sel] // 256
    rhythm_hist = np.bincount(r, minlength=N_R).astype(float)
    # jacks: same lane again within 160 ms
    lane_last = {}
    jacks = 0
    for n in sorted(m.notes, key=lambda x: x.start):
        if a <= n.start < b and n.start - lane_last.get(n.col, -10 ** 9) < 160:
            jacks += 1
        lane_last[n.col] = n.end if n.ln else n.start
    lanes = cs.masks[sel]
    lt = np.zeros(256)
    if lanes.size > 1:
        np.add.at(lt, lanes[:-1] * 16 + lanes[1:], 1)
    sh = tok["shape"][0][sel]
    shape4 = Counter(tuple(sh[i:i + 4]) for i in range(max(0, sh.size - 3)))
    motifs = {}
    for lv, hl in hits.items():
        motifs[lv] = Counter(k for (i, L, k) in hl if sel.size and sel[0] <= i <= sel[-1])
    cp = np.zeros(4)
    kinds = ("exact", "mirror", "variation", "rhythm")
    for c in copies:
        lo, hi = max(c.start, sel[0] if sel.size else 0), min(c.start + c.length, (sel[-1] + 1) if sel.size else 0)
        if hi > lo:
            cp[kinds.index(c.kind)] += hi - lo
    in_rest = sum(1 for n in notes if any(x <= n.start <= y for x, y in (rests or [])))
    after = sum(1 for n in notes if n.start > song_end + 500)
    return ZoneStats(len(notes), len(notes) / dur, chord_hist, rhythm_hist, jacks / max(1, len(notes)),
                     float(np.mean([n.ln for n in notes])) if notes else 0.0, lt, shape4, motifs, cp, cs.times[sel], in_rest, after)


def placement(h: ZoneStats, g: ZoneStats, tol: int = 10) -> Tuple[float, float]:
    """(share of composer chords on a human chord time, share of human chords the composer hit)."""
    if not h.times.size or not g.times.size:
        return (0.0, 0.0)

    def hit(x, ref):
        pos = np.clip(np.searchsorted(ref, x), 1, ref.size - 1) if ref.size > 1 else np.zeros(x.size, int)
        d = np.minimum(np.abs(ref[pos] - x), np.abs(ref[np.maximum(pos - 1, 0)] - x))
        return float(np.mean(d <= tol))

    return hit(g.times, h.times), hit(h.times, g.times)


def compare(h: ZoneStats, g: ZoneStats) -> Dict[str, float]:
    d = {}
    d["nps"] = abs(math.log((g.nps + 0.3) / (h.nps + 0.3)))
    d["chord"] = _js(h.chord_hist + 1e-3, g.chord_hist + 1e-3)
    d["jack"] = abs(g.jack - h.jack)
    d["ln"] = abs(g.ln - h.ln)
    d["rhythm"] = _js(h.rhythm_hist + 1e-3, g.rhythm_hist + 1e-3)
    d["lanes"] = _js(h.lane_hist + 1e-3, g.lane_hist + 1e-3)
    keys = set(h.shape4) | set(g.shape4)
    if keys:
        ks = sorted(keys)
        d["shape"] = _js(np.array([h.shape4[k] for k in ks], float) + 1e-3, np.array([g.shape4[k] for k in ks], float) + 1e-3)
    else:
        d["shape"] = 0.0
    js = []
    for lv in h.motifs:
        a, b = set(h.motifs[lv]), set(g.motifs.get(lv, {}))
        if a or b:
            js.append(1.0 - len(a & b) / max(1, len(a | b)))
    d["motifs"] = float(np.mean(js)) if js else 0.0
    vh = len(h.shape4) / max(1, sum(h.shape4.values())) if h.shape4 else 0.0
    vg = len(g.shape4) / max(1, sum(g.shape4.values())) if g.shape4 else 0.0
    d["variety"] = vh - vg            # > 0: the human zone has more different shapes
    pg, ph = placement(h, g)
    d["placement"] = 1.0 - 0.5 * (pg + ph)
    d["on_human"], d["human_hit"] = pg, ph
    d["rest"] = (g.in_rest + g.after_end) / max(1, g.n)   # composer notes where every human map rests
    return d


WEIGHTS = {"nps": 1.0, "chord": 1.5, "jack": 1.5, "ln": 1.0, "rhythm": 1.5, "lanes": 1.0, "shape": 1.0, "motifs": 0.7,
           "variety": 1.0, "placement": 1.0, "rest": 3.0}


def zone_score(d: Dict[str, float]) -> float:
    return float(sum(WEIGHTS[k] * (max(0.0, d[k]) if k == "variety" else d[k]) for k in WEIGHTS))


# ------------------------------------------------------------------------------------------------------------
# one song
# ------------------------------------------------------------------------------------------------------------

@dataclass
class ZoneRow:
    song: str
    human: Diff
    gen: Diff
    a: int
    b: int
    label: str
    d: Dict[str, float]
    hs: ZoneStats
    gs: ZoneStats
    score: float


def compare_song(original: Path, generated: Path, lib: Optional[MotifLibrary], use_audio: bool = True, log=print) -> List[ZoneRow]:
    hum, gen = read_diffs(original), read_diffs(generated)
    if not hum or not gen:
        log(f"  [skip] {original.name}: no 4K maps on one side")
        return []
    duration = max(max(n.end for n in d.m.notes) for d in hum + gen) + 1000
    zones = None
    if use_audio:
        try:
            zones = composer_zones(original, log)
        except Exception as err:  # no audio / ffmpeg: fall back to bars
            log(f"  [zones] audio analysis failed ({err}); 8-bar zones")
    if zones is None:
        zones = bar_zones(hum[0].m, duration)
    rows = []
    prep = {}

    def prepare(dd: Diff):
        if id(dd) not in prep:
            cs, tok = map_tokens(dd.m)
            hits = {lv: motif_hits(tok[lv][0], tok[lv][1], lib, lv) for lv in ("lanes", "shape", "exact")}
            prep[id(dd)] = (cs, tok, hits, self_repeats(cs, tok))
        return prep[id(dd)]

    for g in gen:
        h = min(hum, key=lambda x: abs(x.sr - g.sr))
        rests, song_end = rest_spans(hum)
        hc, ht, hh, hcp = prepare(h)
        gc, gt, gh, gcp = prepare(g)
        for a, b, label in zones:
            hs = zone_stats(h.m, hc, ht, hh, hcp, a, b, rests, song_end)
            gs = zone_stats(g.m, gc, gt, gh, gcp, a, b, rests, song_end)
            if hs.n < 4 and gs.n < 4:
                continue
            d = compare(hs, gs)
            rows.append(ZoneRow(original.stem, h, g, a, b, label, d, hs, gs, zone_score(d)))
    return rows


# ------------------------------------------------------------------------------------------------------------
# report
# ------------------------------------------------------------------------------------------------------------

def side_by_side(row: ZoneRow, max_lines: int = 24) -> str:
    """Both maps' raw [HitObjects] lines of the first seconds of the zone, plus a 4-column drawing."""
    def pick(d: Diff):
        return [s for t, s in d.lines if row.a <= t < row.b][:max_lines]

    def col(s: str) -> int:
        try:
            return min(3, int(int(s.split(",")[0]) * 4 // 512))
        except ValueError:
            return 0

    def picture(lines: List[str]) -> List[str]:
        by_t: Dict[int, List[str]] = {}
        for s in lines:
            p = s.split(",")
            by_t.setdefault(int(p[2]), []).append(s)
        out = []
        for t, ls in sorted(by_t.items()):
            cells = ["."] * 4
            for s in ls:
                p = s.split(",")
                cells[col(s)] = "O" if (int(p[3]) & 128) else "#"
            out.append(f"{t:>7d} |{''.join(cells)}|")
        return out

    hl, gl = pick(row.human), pick(row.gen)
    hp, gp = picture(hl), picture(gl)
    w = max([len(x) for x in hl] + [20])
    out = [f"{'human: ' + row.human.name[:w - 7]:<{w}}   composer: {row.gen.name}"]
    for i in range(max(len(hl), len(gl))):
        out.append(f"{hl[i] if i < len(hl) else '':<{w}}   {gl[i] if i < len(gl) else ''}")
    out.append("")
    w2 = max([len(x) for x in hp] + [16])
    for i in range(max(len(hp), len(gp))):
        out.append(f"{hp[i] if i < len(hp) else '':<{w2}}   {gp[i] if i < len(gp) else ''}")
    return "\n".join(out)


def explain(row: ZoneRow, lib: Optional[MotifLibrary]) -> List[str]:
    h, g, d = row.hs, row.gs, row.d
    out = [f"- notes/s: human {h.nps:.1f}, composer {g.nps:.1f}; chords 1/2/3/4: human "
           f"{'/'.join(f'{x:.0f}' for x in h.chord_hist)}, composer {'/'.join(f'{x:.0f}' for x in g.chord_hist)}",
           f"- jacks: human {h.jack:.0%}, composer {g.jack:.0%}; long notes: human {h.ln:.0%}, composer {g.ln:.0%}",
           f"- notes where all human maps rest (break / silence / after the end): composer {g.in_rest + g.after_end}"
           f" of {g.n}; human break periods: {len(row.human.breaks)}, composer: {len(row.gen.breaks)}",
           f"- placement: {d['on_human']:.0%} of composer chords on a human chord time, composer hits {d['human_hit']:.0%} of human chords",
           f"- distinct 4-chord shapes per chord: human {len(h.shape4) / max(1, sum(h.shape4.values())):.2f}, "
           f"composer {len(g.shape4) / max(1, sum(g.shape4.values())):.2f}"]
    hr = ", ".join(f"{RHYTHM_NAMES[i]} {h.rhythm_hist[i] / max(1, h.rhythm_hist.sum()):.0%}" for i in np.argsort(-h.rhythm_hist)[:4] if h.rhythm_hist[i])
    gr = ", ".join(f"{RHYTHM_NAMES[i]} {g.rhythm_hist[i] / max(1, g.rhythm_hist.sum()):.0%}" for i in np.argsort(-g.rhythm_hist)[:4] if g.rhythm_hist[i])
    out.append(f"- rhythm (gap to previous chord): human {hr}; composer {gr}")
    cpk = ("exact", "mirror", "variation", "rhythm")
    out.append("- copies of earlier passages: human " + ", ".join(f"{k} {v / max(1, h.n):.0%}" for k, v in zip(cpk, h.copies) if v)
               + "; composer " + ", ".join(f"{k} {v / max(1, g.n):.0%}" for k, v in zip(cpk, g.copies) if v))
    if lib is not None and "lanes" in lib.tables:
        t = lib.tables["lanes"]
        only_h = [k for k, _ in h.motifs.get("lanes", Counter()).most_common() if k not in g.motifs.get("lanes", {})][:4]
        only_g = [k for k, _ in g.motifs.get("lanes", Counter()).most_common() if k not in h.motifs.get("lanes", {})][:4]
        if only_h:
            out.append("- lane motifs only the human uses here: " + "; ".join(
                f"{name_motif('lanes', t.tokens(k)) or 'motif'} [{' '.join(str(int(x)) for x in t.tokens(k))}]" for k in only_h))
        if only_g:
            out.append("- lane motifs only the composer uses here: " + "; ".join(
                f"{name_motif('lanes', t.tokens(k)) or 'motif'} [{' '.join(str(int(x)) for x in t.tokens(k))}]" for k in only_g))
    return out


def report(rows: List[ZoneRow], lib: Optional[MotifLibrary], n_show: int = 12) -> str:
    out = ["# Composer vs human, zone by zone", ""]
    if not rows:
        return "\n".join(out + ["no zones compared"])
    keys = [k for k in WEIGHTS]
    pairs_seen = {}
    for r in rows:
        pairs_seen[(r.song, r.gen.name)] = (len(r.human.breaks), len(r.gen.breaks))
    out += [f"Break periods ([Events] 2,start,end): human difficulties paired {sum(v[0] for v in pairs_seen.values())}, "
            f"composer {sum(v[1] for v in pairs_seen.values())}.", ""]
    out += [f"{len(rows)} zones of {len({r.song for r in rows})} songs, {len({(r.song, r.gen.name) for r in rows})} composed difficulties.", "",
            "Distance per statistic (0 = identical; mean over zones, and the mean of the worst / best fifth):", "",
            "| statistic | all | worst 20 % | best 20 % |", "|---|---|---|---|"]
    srt = sorted(rows, key=lambda r: r.score)
    q = max(1, len(srt) // 5)
    for k in keys + ["on_human", "human_hit"]:
        out.append(f"| {k} | {np.mean([r.d[k] for r in srt]):.3f} | {np.mean([r.d[k] for r in srt[-q:]]):.3f} | "
                   f"{np.mean([r.d[k] for r in srt[:q]]):.3f} |")
    # what drives bad zones: weighted contribution
    contrib = {k: np.mean([WEIGHTS[k] * max(0.0, r.d[k]) for r in srt[-q:]]) for k in keys}
    out += ["", "Largest contributions to the worst zones: " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(contrib.items(), key=lambda x: -x[1])[:5]), ""]
    # systematic differences over all zones
    agg = lambda f: (np.mean([f(r.hs) for r in rows]), np.mean([f(r.gs) for r in rows]))
    out += ["## Systematic differences (mean over zones, human vs composer)", "", "| measure | human | composer |", "|---|---|---|"]
    for name, f in (("notes per second", lambda s: s.nps), ("jack share", lambda s: s.jack), ("LN share", lambda s: s.ln),
                    ("chord share (2+ notes)", lambda s: s.chord_hist[1:].sum() / max(1, s.chord_hist.sum())),
                    ("distinct 4-chord shapes / chord", lambda s: len(s.shape4) / max(1, sum(s.shape4.values()))),
                    ("chords copied from earlier (exact+mirror)", lambda s: s.copies[:2].sum() / max(1, s.n)),
                    ("chords in a variation copy", lambda s: s.copies[2] / max(1, s.n)),
                    ("off-grid chords", lambda s: s.rhythm_hist[-1] / max(1, s.rhythm_hist.sum())),
                    ("library lane motifs per zone", lambda s: len(s.motifs.get("lanes", {}))),
                    ("notes in human rests (break / silence)", lambda s: s.in_rest),
                    ("notes after the song's last human note", lambda s: s.after_end)):
        a, b = agg(f)
        out.append(f"| {name} | {a:.3f} | {b:.3f} |")
    hr = sum((r.hs.rhythm_hist for r in rows), np.zeros(N_R))
    gr = sum((r.gs.rhythm_hist for r in rows), np.zeros(N_R))
    out += ["", "Rhythm classes over all zones (share of chords):", "", "| gap | human | composer |", "|---|---|---|"]
    for i in np.argsort(-(hr + gr)):
        if hr[i] + gr[i] > 0:
            out.append(f"| {RHYTHM_NAMES[i]} | {hr[i] / hr.sum():.1%} | {gr[i] / gr.sum():.1%} |")
    # per song
    out += ["", "## Per song and difficulty", "", "| song | human diff | composer diff | zones | mean distance | worst zone |", "|---|---|---|---|---|---|"]
    by = {}
    for r in rows:
        by.setdefault((r.song, r.human.name, r.gen.name), []).append(r)
    for (s, hn, gn), rs in sorted(by.items(), key=lambda x: -np.mean([r.score for r in x[1]])):
        w = max(rs, key=lambda r: r.score)
        out.append(f"| {s} | {hn} ({rs[0].human.sr:.2f}*) | {gn} ({rs[0].gen.sr:.2f}*) | {len(rs)} | {np.mean([r.score for r in rs]):.2f} | "
                   f"{w.a / 1000:.1f}-{w.b / 1000:.1f}s ({w.score:.2f}) |")
    for title, sel in (("Worst zones", srt[::-1][:n_show]), ("Best zones", srt[:n_show])):
        out += ["", f"## {title}", ""]
        for r in sel:
            out += [f"### {r.song}: {r.a / 1000:.1f}-{r.b / 1000:.1f}s ({r.label}), human {r.human.name} {r.human.sr:.2f}* vs composer {r.gen.sr:.2f}*, distance {r.score:.2f}", ""]
            out += explain(r, lib)
            out += ["", "```", side_by_side(r), "```", ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original", default=None, help="human .osz / .osu / folder")
    ap.add_argument("--generated", default=None, help="composed .osz / .osu / folder of the same song")
    ap.add_argument("--pairs_dir", default=None, help="folder of human .osz files ...")
    ap.add_argument("--generated_dir", default=None, help="... and a folder with composed .osz files of the same names")
    ap.add_argument("--motifs", default="checkpoints/hybrid/motifs.npz", help="motif library (mine_patterns.py)")
    ap.add_argument("--bar_zones", action="store_true", help="8-bar zones instead of the composer's sections (no audio needed)")
    ap.add_argument("--report", default="zone_report.md")
    ap.add_argument("--show", type=int, default=12, help="worst / best zones shown with their hit-object lines")
    args = ap.parse_args()
    lib = MotifLibrary.load(args.motifs) if Path(args.motifs).exists() else None
    if lib is None:
        print(f"[zones] no motif library at {args.motifs}: motif statistics are skipped (python mine_patterns.py)")
    pairs = []
    if args.original and args.generated:
        pairs.append((Path(args.original), Path(args.generated)))
    if args.pairs_dir and args.generated_dir:
        for p in sorted(Path(args.pairs_dir).glob("*.osz")):
            g = Path(args.generated_dir) / p.name
            if g.exists():
                pairs.append((p, g))
    if not pairs:
        raise SystemExit("give --original + --generated, or --pairs_dir + --generated_dir")
    rows = []
    for o, g in pairs:
        print(f"[zones] {o.name}", flush=True)
        rows += compare_song(o, g, lib, use_audio=not args.bar_zones)
    Path(args.report).write_text(report(rows, lib, args.show), encoding="utf-8")
    print(f"[zones] {len(rows)} zones -> {args.report}")


if __name__ == "__main__":
    main()
