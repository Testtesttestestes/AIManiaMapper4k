"""
Parts of a song ("regimes", as market regimes in trading: the same signal means different things in a trend and in a
panic): intro, verse, chorus, build-up, drop, bridge, solo, arpeggio, break, outro.

Mappers treat them differently - rice on the vocal phrases of a chorus, streams in a drop, a lighter bridge, the melody
followed note by note in a solo or an arpeggio. The v5 network gets the part of every frame as an input group of its
source selector (``section_features``), so it can learn that difference from the maps.

How the parts are found (from the audio and the red lines only):

1. every bar (4 beats of the red lines; 2 s blocks without them) gets a description: the mean of the mid mel bands,
   chroma, onset strength, and - with demucs - the loudness and attack rate of every stem; standardized per song;
2. novelty at every bar line = how different the 4 bars after it are from the 4 bars before it (cosine distance);
   the strongest peaks at least ``min_bars`` apart are the boundaries (a part is 4+ bars);
3. the climaxes come from the maps themselves: mappers switch kiai time on for the high points of a song (the
   chorus, the drop) - a human label in most ranked maps. A small logistic model learns from the training maps'
   kiai which parts are climaxes from their description (``fit_highlight``; its weights go into the checkpoint),
   so at generation, without a map, every part gets its climax probability ``p_high``;
4. names: a climax (p_high >= 0.5) is a chorus when someone sings in it (vocals stem), otherwise a drop; the rest by
   rules on loudness, stems and change relative to the rest of the song:

   intro / outro  the first / last part, quieter than the song's middle
   break     quiet (loudness z < -1)
   build-up  loudness rising through the part and a louder part next
   verse     singing, not a climax
   solo      no singing, the "other" stem (lead instrument) leads with many attacks
   arpeggio  no singing, "other" attacks at 1/4 or faster, the harmony (chroma) moving
   bridge    the rest

There is no human label for solo / arpeggio / bridge / build-up: those names are a heuristic (and without demucs there
is no "singing" at all). The network does not depend on the names: it gets the numbers behind them (climax
probability, loudness, stems, attack rates, novelty, position) and learns from the maps how mappers treat parts
that sound like that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

LABELS: Tuple[str, ...] = ("intro", "verse", "chorus", "buildup", "drop", "bridge", "solo", "arpeggio", "break", "outro")
N_NUM = 11                                   # numeric features per frame (see section_features)
DESC_KEYS = ("loud", "drums", "bass", "other", "vocals", "drums_rate", "bass_rate", "other_rate", "vocals_rate",
             "rise", "novelty", "pos", "length")
SECTION_DIM = len(LABELS) + N_NUM


@dataclass
class Section:
    start: int                               # frame
    end: int                                 # frame (exclusive)
    label: str
    stats: Dict[str, float] = field(default_factory=dict)


def _bars(F: int, timing_points, frame_ms: int) -> np.ndarray:
    """Bar index of every frame (2 s blocks without red lines)."""
    if timing_points and any(getattr(tp, "uninherited", False) for tp in timing_points):
        from ..hybrid.stitch import bar_index
        b = bar_index(np.arange(F, dtype=np.float64) * frame_ms, timing_points)
        return b - b.min()
    return np.arange(F) // max(1, 2000 // frame_ms)


def _z(x: np.ndarray) -> np.ndarray:
    s = x.std(axis=0)
    return (x - x.mean(axis=0)) / np.where(s > 1e-6, s, 1.0)


def find_sections(feats: np.ndarray, slices: Dict[str, Tuple[int, int]], timing_points, frame_ms: int = 10,
                  min_bars: int = 4, context_bars: int = 4, highlight: Optional[Sequence[float]] = None) -> List[Section]:
    """Parts of the song; ``highlight`` = weights of the climax model (fit_highlight), else climaxes by loudness."""
    from ..hybrid.causes import stem_attacks

    F = feats.shape[0]
    if F == 0:
        return []
    bars = _bars(F, timing_points, frame_ms)
    nb = int(bars.max()) + 1
    cols = []
    for name in ("mel_mid", "chroma", "onset"):
        if name in slices:
            a, b = slices[name]
            cols.append(np.asarray(feats[:, a:b], dtype=np.float32))
    att = stem_attacks(feats, slices)
    stems = None
    if "demucs_stems" in slices and slices["demucs_stems"][1] - slices["demucs_stems"][0] == 128:
        a, _ = slices["demucs_stems"]
        stems = np.asarray(feats[:, a:a + 128], dtype=np.float32).reshape(F, 4, 32).mean(axis=2)     # [F, 4]
        cols.append(stems)
    if not cols:
        return [Section(0, F, "verse")]
    X = np.concatenate(cols, axis=1)
    counts = np.bincount(bars, minlength=nb).astype(np.float64)
    D = np.zeros((nb, X.shape[1]))
    np.add.at(D, bars, X)
    D /= np.maximum(counts, 1)[:, None]
    D = _z(D)
    # novelty at every bar line
    nov = np.zeros(nb)
    K = context_bars
    for i in range(1, nb):
        a = D[max(0, i - K):i].mean(axis=0)
        b = D[i:min(nb, i + K)].mean(axis=0)
        na, nb_ = np.linalg.norm(a), np.linalg.norm(b)
        nov[i] = 1.0 - float(a @ b / (na * nb_)) if na > 1e-6 and nb_ > 1e-6 else 0.0
    # strongest peaks, at least min_bars apart, above the song's typical novelty
    cand = [i for i in range(min_bars, nb - min_bars + 1) if nov[i] >= nov[max(0, i - 1)] and nov[i] >= nov[min(nb - 1, i + 1)]]
    thr = float(np.mean(nov[1:]) + 0.5 * np.std(nov[1:])) if nb > 2 else 1e9
    bounds: List[int] = []
    for i in sorted(cand, key=lambda j: -nov[j]):
        if nov[i] < thr:
            break
        if all(abs(i - j) >= min_bars for j in bounds):
            bounds.append(i)
    edges = [0] + sorted(bounds) + [nb]
    first_frame = np.searchsorted(bars, np.arange(nb + 1), side="left")
    # per-part statistics
    spec = slices.get("spectral")
    loud = np.asarray(feats[:, spec[0]], dtype=np.float32) if spec else X.mean(axis=1)
    secs: List[Section] = []
    for a_bar, b_bar in zip(edges[:-1], edges[1:]):
        fa, fb = int(first_frame[a_bar]), int(first_frame[b_bar]) if b_bar < nb else F
        if fb <= fa:
            continue
        seg = slice(fa, fb)
        st: Dict[str, float] = {"loud": float(loud[seg].mean())}
        third = max(1, (fb - fa) // 3)
        st["rise"] = float(loud[fb - third:fb].mean() - loud[fa:fa + third].mean())
        if stems is not None:
            for k, nm in enumerate(("drums", "bass", "other", "vocals")):
                st[nm] = float(stems[seg, k].mean())
                st[nm + "_rate"] = float((att[seg, k] > 3.0).mean() * 1000.0 / frame_ms) if att is not None else 0.0
        if "chroma" in slices:
            a, b = slices["chroma"]
            ch = np.asarray(feats[seg, a:b], dtype=np.float32)
            st["chroma_move"] = float(np.abs(np.diff(ch, axis=0)).mean()) if ch.shape[0] > 1 else 0.0
        st["novelty"] = float(nov[a_bar]) if a_bar < nb else 0.0
        st["pos"] = (fa + fb) / 2.0 / F
        st["length"] = (fb - fa) / max(1.0, F)
        secs.append(Section(fa, fb, "", st))
    _standardize(secs)
    for sec in secs:
        sec.stats["p_high"] = highlight_prob(sec, highlight)
    _name(secs, timing_points, frame_ms)
    return secs


def _standardize(secs: List[Section]) -> None:
    """z of every statistic against the rest of the song (weighted by length), stored as stats['z_<key>']."""
    if not secs:
        return
    keys = sorted({k for s in secs for k in s.stats})
    M = np.array([[s.stats.get(k, 0.0) for k in keys] for s in secs], dtype=np.float64)
    w = np.array([s.end - s.start for s in secs], dtype=np.float64)
    mu = (M * w[:, None]).sum(0) / w.sum()
    sd = np.sqrt(((M - mu) ** 2 * w[:, None]).sum(0) / w.sum())
    Z = (M - mu) / np.where(sd > 1e-6, sd, 1.0)
    for i, sec in enumerate(secs):
        for j, k in enumerate(keys):
            sec.stats["z_" + k] = float(Z[i, j])


def descriptor(sec: Section) -> np.ndarray:
    z = [sec.stats.get("z_" + k, 0.0) for k in DESC_KEYS if k not in ("pos", "length")]
    return np.array(z + [sec.stats.get("pos", 0.5), sec.stats.get("length", 0.0) * 10.0, 1.0], dtype=np.float64)


def highlight_prob(sec: Section, w: Optional[Sequence[float]]) -> float:
    if not w:
        return float(1.0 / (1.0 + np.exp(-2.0 * (sec.stats.get("z_loud", 0.0) - 0.5))))   # by loudness alone
    x = descriptor(sec)
    return float(1.0 / (1.0 + np.exp(-float(np.dot(np.asarray(w, dtype=np.float64)[: x.size], x)))))


def kiai_frames(timing_points, F: int, frame_ms: int = 10) -> Optional[np.ndarray]:
    """Frames in kiai time (bit 0 of the effects of the timing points), or None when the map has no kiai at all."""
    pts = sorted(timing_points or [], key=lambda tp: tp.time)
    if not any(int(getattr(tp, "effects", 0)) & 1 for tp in pts):
        return None
    out = np.zeros(F, dtype=bool)
    for i, tp in enumerate(pts):
        if int(getattr(tp, "effects", 0)) & 1:
            a = max(0, int(tp.time // frame_ms))
            b = int(pts[i + 1].time // frame_ms) if i + 1 < len(pts) else F
            out[a:max(a, min(F, b))] = True
    return out


def fit_highlight(samples: Sequence[Tuple[np.ndarray, float]], l2: float = 1e-2, steps: int = 3000) -> List[float]:
    """Logistic regression: part descriptor -> share of the part in kiai time (the mappers' climax label)."""
    X = np.stack([x for x, _ in samples])
    y = np.array([t for _, t in samples], dtype=np.float64)
    w = np.zeros(X.shape[1])
    lr = 0.5
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-X @ w))
        g = X.T @ (p - y) / len(y) + l2 * w
        w -= lr * g
    return [float(v) for v in w]


def highlight_samples(feats: np.ndarray, slices, timing_points, frame_ms: int = 10) -> List[Tuple[np.ndarray, float]]:
    """(descriptor, share in kiai) of every part of one map with kiai (empty without kiai)."""
    F = feats.shape[0]
    k = kiai_frames(timing_points, F, frame_ms)
    if k is None:
        return []
    return [(descriptor(s), float(k[s.start:s.end].mean())) for s in find_sections(feats, slices, timing_points, frame_ms)]


def _name(secs: List[Section], timing_points, frame_ms: int) -> None:
    if not secs:
        return
    z = lambda i, k: float(secs[i].stats.get("z_" + k, 0.0)) if 0 <= i < len(secs) else 0.0
    has_stems = "vocals" in secs[0].stats
    beat_ms = 500.0
    if timing_points:
        reds = [tp for tp in timing_points if getattr(tp, "uninherited", False) and tp.beat_len > 1]
        if reds:
            beat_ms = float(reds[0].beat_len)
    n = len(secs)
    for i, s in enumerate(secs):
        loud = z(i, "loud")
        sing = z(i, "vocals") > 0.0 if has_stems else None
        nxt = z(i + 1, "loud") if i + 1 < n else loud
        other_fast = has_stems and s.stats.get("other_rate", 0.0) * beat_ms / 1000.0 >= 3.0   # ~1/4 and faster
        if s.stats.get("p_high", 0.0) >= 0.5:
            lab = "chorus" if (sing or not has_stems) else "drop"
        elif i == 0 and loud < 0 and n > 2:
            lab = "intro"
        elif i == n - 1 and loud < 0 and n > 2:
            lab = "outro"
        elif loud < -1.0:
            lab = "break"
        elif s.stats.get("rise", 0.0) > 0 and z(i, "rise") > 1.0 and nxt > loud + 0.5:
            lab = "buildup"
        elif sing:
            lab = "verse"
        elif has_stems and z(i, "other") > 0.5 and z(i, "other_rate") > 0.3:
            lab = "arpeggio" if other_fast and z(i, "chroma_move") > 0.0 else "solo"
        elif not has_stems:
            lab = "verse" if loud > -0.3 else "bridge"
        else:
            lab = "bridge"
        s.label = lab
        s.stats["loud_z"] = loud


def section_features(secs: Sequence[Section], F: int) -> np.ndarray:
    """[F, SECTION_DIM]: one-hot part name + loudness z, drums / bass / other / vocals, attack rate of the leading
    stem, rise, novelty at the start, position in the song, position in the part, climax probability."""
    out = np.zeros((F, SECTION_DIM), dtype=np.float32)
    if not secs:
        return out
    L = len(LABELS)
    keys = ("loud_z", "drums", "bass", "other", "vocals")
    for s in secs:
        a, b = max(0, s.start), min(F, s.end)
        if b <= a:
            continue
        out[a:b, LABELS.index(s.label) if s.label in LABELS else 1] = 1.0
        for j, k in enumerate(keys):
            out[a:b, L + j] = float(np.clip(s.stats.get(k, 0.0), -5, 5))
        out[a:b, L + 5] = float(np.clip(max((s.stats.get(f"{k}_rate", 0.0) for k in ("drums", "bass", "other", "vocals")),
                                            default=0.0) / 10.0, 0, 5))
        out[a:b, L + 6] = float(np.clip(s.stats.get("rise", 0.0), -5, 5))
        out[a:b, L + 7] = float(np.clip(s.stats.get("novelty", 0.0) * 5.0, 0, 5))
        out[a:b, L + 9] = np.linspace(0.0, 1.0, b - a, endpoint=False)
        out[a:b, L + 10] = float(s.stats.get("p_high", 0.0))
    out[:, L + 8] = np.linspace(0.0, 1.0, F, endpoint=False)
    return out


def describe(secs: Sequence[Section], frame_ms: int = 10) -> str:
    return " | ".join(f"{s.start * frame_ms / 1000:5.1f}s {s.label}" + (" *" if s.stats.get("p_high", 0) >= 0.5 else "")
                      for s in secs)


def kiai_timing_points(secs: Sequence[Section], timing_points, frame_ms: int = 10) -> list:
    """The map's timing points + green lines that switch kiai time on at the start of every climax part
    (p_high >= 0.5) and off at its end; scroll speed stays 1.0. Existing kiai is kept as it is."""
    from ..hybrid.beatmap import RawTimingPoint

    pts = list(timing_points or [])
    if any(int(getattr(tp, "effects", 0)) & 1 for tp in pts):
        return pts
    vol = next((tp.volume for tp in pts if getattr(tp, "uninherited", False)), 70)
    add = []
    for s in secs:
        if s.stats.get("p_high", 0.0) >= 0.5:
            add.append(RawTimingPoint(float(s.start * frame_ms), -100.0, 4, 0, 0, vol, False, 1))
            add.append(RawTimingPoint(float(s.end * frame_ms), -100.0, 4, 0, 0, vol, False, 0))
    # an "off" line at the very start of the next climax would cancel it: drop those
    on = {p.time for p in add if p.effects & 1}
    add = [p for p in add if p.effects & 1 or p.time not in on]
    return sorted(pts + add, key=lambda tp: (tp.time, not tp.uninherited))
