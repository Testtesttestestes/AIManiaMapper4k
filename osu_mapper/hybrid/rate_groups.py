"""
Rate versions (sped up / slowed down copies of a chart, with their own resampled audio) found by CONTENT, not by name.

A rate edit is the same chart with every time multiplied by one factor: the sequence of chords (lanes and long-note
members) is identical and the note times of one map are a straight line of the other's (t_b = t_a / rate + offset). Names
are unreliable ("1.1x", "x1.1", "130%", "Sped Up", "DT", nothing at all...), so the train / validation split groups
songs by this: two songs with a rate version of each other are one piece of music, and must not end up on both
sides (validation would contain a song the network trained on at another tempo).

    from osu_mapper.hybrid.rate_groups import find_rate_versions
    pairs = find_rate_versions({"a": notes_a, "b": notes_b, ...})   # [(key_a, key_b, rate), ...]
"""

from __future__ import annotations

import hashlib
from typing import Dict, Hashable, List, Sequence, Tuple

import numpy as np

from .beatmap import Note


def chord_sequence(notes: Sequence[Note]) -> Tuple[np.ndarray, List[int]]:
    """(chord start times, chord codes): code = lanes + 16 x long-note lanes of the chord."""
    groups: Dict[int, int] = {}
    for n in notes:
        code = (1 << int(n.col)) | ((16 << int(n.col)) if n.ln else 0)
        groups[int(n.start)] = groups.get(int(n.start), 0) | code
    times = sorted(groups)
    return np.array(times, dtype=np.float64), [groups[t] for t in times]


def _sig(codes: Sequence[int]) -> str:
    return hashlib.sha1(bytes(int(c) & 255 for c in codes)).hexdigest()


def fit_rate(ta: np.ndarray, tb: np.ndarray, max_resid_ms: float = 3.0) -> float:
    """Rate r with tb ~ ta / r + c (r > 1: b is the faster version), or 0.0 when the times are not a straight line."""
    if ta.size < 8 or ta.size != tb.size:
        return 0.0
    va = ta - ta.mean()
    s = float((va * (tb - tb.mean())).sum() / max(1e-9, (va * va).sum()))
    if s <= 0:
        return 0.0
    resid = tb - (tb.mean() + s * va)
    if float(np.median(np.abs(resid))) > max_resid_ms or float(np.percentile(np.abs(resid), 95)) > 4 * max_resid_ms:
        return 0.0
    return 1.0 / s


def find_rate_versions(charts: Dict[Hashable, Sequence[Note]], min_chords: int = 50, prefix: int = 200,
                       min_rate: float = 0.5, max_rate: float = 2.0) -> List[Tuple[Hashable, Hashable, float]]:
    """Pairs of charts that are rate versions of each other (or the same chart at the same rate: rate ~1).

    Candidates share the chord sequence (whole, or its first ``prefix`` chords - a rate pack may trim the end); the
    times of the shared part must lie on one straight line with a slope between ``min_rate`` and ``max_rate``."""
    seqs: Dict[Hashable, Tuple[np.ndarray, List[int]]] = {}
    buckets: Dict[str, List[Hashable]] = {}
    for key, notes in charts.items():
        t, c = chord_sequence(notes)
        if len(c) < min_chords:
            continue
        seqs[key] = (t, c)
        for sig in {"full:" + _sig(c), "head:" + _sig(c[:prefix])}:
            buckets.setdefault(sig, []).append(key)
    out: List[Tuple[Hashable, Hashable, float]] = []
    seen = set()
    for keys in buckets.values():
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                (ta, ca), (tb, cb) = seqs[a], seqs[b]
                n = min(len(ca), len(cb))
                if ca[:n] != cb[:n]:
                    continue
                r = fit_rate(ta[:n], tb[:n])
                if min_rate <= r <= max_rate:
                    out.append((a, b, round(r, 4)))
    return out


def regroup_index(data: Dict, root, log=print) -> int:
    """Gives every song that has a rate version of another song (any name) the same split piece as that song.
    Changes data['maps'][*]['piece'] in place; returns the number of maps whose piece changed."""
    from pathlib import Path

    from .params import ParamTrack
    from .renderer import render_track

    root = Path(root)
    real = [m for m in data["maps"] if not m.get("derived") and m.get("roundtrip_exact", True)]
    charts = {}
    for m in real:
        try:
            charts[m["map_id"]] = render_track(ParamTrack.load_npz(root / m["track"]))
        except (OSError, ValueError, KeyError):
            continue
    pairs = find_rate_versions(charts)
    song_of = {m["map_id"]: m["song_id"] for m in real}
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_id = {m["map_id"]: m for m in real}
    for a, b, r in pairs:
        sa, sb = song_of[a], song_of[b]
        if sa != sb:
            parent[find(sa)] = find(sb)
            log(f"  [rate] x{r:g}: {by_id[a].get('osu_file', a)}  <->  {by_id[b].get('osu_file', b)}")
    if not parent:
        return 0
    piece_of_song: Dict[str, str] = {}
    for m in data["maps"]:
        piece_of_song.setdefault(m["song_id"], str(m.get("piece") or m["song_id"]))
    group_piece: Dict[str, str] = {}
    for sid, pc in piece_of_song.items():
        root_ = find(sid)
        group_piece[root_] = min(group_piece.get(root_, pc), pc)
    changed = 0
    for m in data["maps"]:
        new = group_piece[find(m["song_id"])]
        if m.get("piece") != new:
            m["piece"] = new
            changed += 1
    return changed
