"""
Learned mapping patterns: what real mappers place where, at every difficulty.

A count model (n-gram with back-off) over the whole dataset, conditioned on the star rating. It learns

* **lanes** - P(chord mask | previous two chord masks, time since the previous chord, star rating):
  which rolls, trills, jumps, jacks, splits, hand alternations real maps use after each pattern and how
  that changes with speed and difficulty (a 1/4 jack at 4* is rare, a jumptrill after a [12] chord is
  common...). Every map is also counted mirrored (left-right), so the model is symmetric.
* **chords** - P(chord size | star rating, metric position, gap): how many 2-, 3-, 4-note chords a
  difficulty has and where (downbeats vs. fills).
* **rhythm** - P(a tick of beat divisor d carries a note | star rating): at 2* mappers use 1/1 and 1/2,
  at 5* most 1/4 ticks, 1/3 only in swing parts...
* **long notes** - P(long note | star rating, gap to the next chord) and the LN share per difficulty.

Unlike the network it cannot overfit a few songs: it stores frequencies of local patterns over all maps
(millions of events) and smooths sparse contexts by backing off to coarser ones. The composer uses it as
its main "is this appropriate and nice at this difficulty" score, the network adds what is specific to
the audio.

    python learn_patterns.py --data dataset_hybrid          # from the prepared dataset (recommended)
    python learn_patterns.py --input_dir "sample beatmaps"  # straight from .osz archives (no audio needed)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .beatmap import Note, RawTimingPoint

PATTERN_VERSION = "patterns-v3"
SR_BINS = 12                                               # 0-1*, 1-2*, ..., 10-11*, 11+*
GAP_EDGES_MS = np.array([45, 70, 95, 125, 160, 210, 280, 380, 550, 800, 1500], dtype=np.float64)
N_GAP = GAP_EDGES_MS.size + 1
DIV_CLASSES: Tuple[str, ...] = ("1/1", "1/2", "1/4", "1/3-1/6", "1/8", "1/12-1/16", "off-grid")
N_DIV = len(DIV_CLASSES)
DIV_TICKS_PER_BEAT = np.array([1, 1, 2, 4, 4, 12, 1], dtype=np.float64)  # ticks of each class per beat
POP = np.array([bin(m).count("1") for m in range(16)])
MIN_BIN_MAPS = 10        # a star-rating bin with fewer maps is mixed with its nearest well-filled neighbour
STAGGER_MS = 30          # chord-to-chord gaps below this are one chord with offset notes (or a grace note), not rhythm
REPEAT_BIN_MS = 5
REPEAT_BINS = 400                                          # 0..2 s
LN_SPAN_BINS = 33                                          # 0..31 chords during a hold, 32 = more
# long notes: the LN style of a map (its LN share) is the mapper's choice, so every LN statistic is kept per style
LN_STYLE_EDGES = np.array([0.05, 0.15, 0.30, 0.45])        # rice < 5% | light | hybrid | LN | full LN > 45%
N_LN_STYLE = LN_STYLE_EDGES.size + 1
LN_STYLE_NAMES = ("rice", "light", "hybrid", "ln", "full")
LN_STYLE_SHARE = (0.02, 0.10, 0.22, 0.38, 0.55)            # typical LN share of each style (for presets)
BEAT_GAP_EDGES = np.array([0.13, 0.2, 0.3, 0.4, 0.6, 0.9, 1.4, 2.5])  # gaps in beats: 1/8 .. 2+ beats
N_BGAP = BEAT_GAP_EDGES.size + 1
LN_LEN_BEATS = np.array([0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])
N_LN_LEN = LN_LEN_BEATS.size


def ln_style_bin(share: float) -> int:
    return int(np.searchsorted(LN_STYLE_EDGES, float(share), side="right"))


def beat_gap_bin(gap_beats: float) -> int:
    return int(np.searchsorted(BEAT_GAP_EDGES, float(gap_beats), side="right"))


def ln_len_class(beats: float) -> int:
    return int(np.argmin(np.abs(np.log(LN_LEN_BEATS) - math.log(max(1e-3, beats)))))


def gap_bin(gap_ms: float) -> int:
    return int(np.searchsorted(GAP_EDGES_MS, float(gap_ms), side="right"))


def sr_bin(sr: float) -> int:
    return int(np.clip(math.floor(float(sr)), 0, SR_BINS - 1))


def _mirror_mask(m: int) -> int:
    return ((m & 1) << 3) | ((m & 2) << 1) | ((m & 4) >> 1) | ((m & 8) >> 3)


MIRROR = np.array([_mirror_mask(m) for m in range(16)], dtype=np.int64)


def div_class(t: float, reds: Sequence[RawTimingPoint], tol_ms: float = 3.0) -> int:
    if not reds:
        return N_DIV - 1
    times = [r.time for r in reds]
    i = max(0, int(np.searchsorted(times, t, side="right")) - 1)
    bl = reds[i].beat_len
    beats = (t - reds[i].time) / bl
    for d, cls in ((1, 0), (2, 1), (4, 2), (3, 3), (6, 3), (8, 4), (12, 5), (16, 5)):
        if abs(round(beats * d) / d - beats) * bl <= tol_ms:
            return cls
    return N_DIV - 1


def metric_bin(cls: int) -> int:
    return 0 if cls == 0 else (1 if cls in (1, 2) else 2)


def chord_events(notes: Sequence[Note]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(start times, chord masks, LN-member masks) of the map's chords, sorted by time."""
    groups: Dict[int, List[Note]] = {}
    for n in notes:
        groups.setdefault(int(n.start), []).append(n)
    times = np.array(sorted(groups), dtype=np.int64)
    masks = np.array([sum(1 << n.col for n in groups[t]) & 15 for t in times], dtype=np.int64)
    lns = np.array([sum(1 << n.col for n in groups[t] if n.ln) & 15 for t in times], dtype=np.int64)
    return times, masks, lns


@dataclass
class PatternModel:
    lane: np.ndarray        # [SR, GAP, 16 prev, 16 prev2, 16 mask] counts
    chord: np.ndarray       # [SR, 3 metric, GAP, 4 sizes]
    rhythm_hits: np.ndarray # [SR, DIV] events on ticks of the class
    rhythm_beats: np.ndarray  # [SR] beats covered by maps of the bin (denominator)
    ln: np.ndarray          # [SR, GAP(next), 2] tap / LN member counts
    maps: np.ndarray        # [SR] maps per bin
    repeat_gap: Optional[np.ndarray] = None  # [SR, REPEAT_BINS] same-lane gaps (previous press or LN release -> next note), 5 ms bins
    ln_span: Optional[np.ndarray] = None     # [SR, LN_SPAN_BINS] chords that start while a long note is held
    density_cv: Optional[np.ndarray] = None  # [SR, 2] sum / count of the density variation (std/mean of notes per 4 s)
    event_gap: Optional[np.ndarray] = None   # [SR, REPEAT_BINS] chord-to-chord gaps (any lanes), 5 ms bins
    intensity_fit: Optional[np.ndarray] = None
    ln_ctx: Optional[np.ndarray] = None      # [SR, STYLE, gap next, gap prev (beat bins), size, metric, 2] chords without / with LN
    ln_members: Optional[np.ndarray] = None  # [SR, STYLE, size, 5] how many members of a chord with LN are long
    ln_len: Optional[np.ndarray] = None      # [SR, STYLE, gap next (beat bin), N_LN_LEN] LN length in beats
    ln_style_maps: Optional[np.ndarray] = None  # [SR, STYLE] maps of each LN style
    ln_span_style: Optional[np.ndarray] = None  # [SR, STYLE, LN_SPAN_BINS] chords during a hold
    ln_release: Optional[np.ndarray] = None  # [SR, STYLE, 3] LN releases: on another chord's start / on a free tick / off grid  # [SR, 3] least squares of log section density vs log audio intensity: sum xy, sum xx, n
    version: str = PATTERN_VERSION

    # ------------------------------------------------------------------------------------------
    @classmethod
    def empty(cls) -> "PatternModel":
        return cls(
            lane=np.zeros((SR_BINS, N_GAP, 16, 16, 16), np.float32),
            chord=np.zeros((SR_BINS, 3, N_GAP, 4), np.float32),
            rhythm_hits=np.zeros((SR_BINS, N_DIV), np.float64),
            rhythm_beats=np.zeros(SR_BINS, np.float64),
            ln=np.zeros((SR_BINS, N_GAP, 2), np.float64),
            maps=np.zeros(SR_BINS, np.int64),
            repeat_gap=np.zeros((SR_BINS, REPEAT_BINS), np.float64),
            ln_span=np.zeros((SR_BINS, LN_SPAN_BINS), np.float64),
            density_cv=np.zeros((SR_BINS, 2), np.float64),
            intensity_fit=np.zeros((SR_BINS, 3), np.float64),
            event_gap=np.zeros((SR_BINS, REPEAT_BINS), np.float64),
        )

    def add_map(self, notes: Sequence[Note], timing_points: Sequence[RawTimingPoint], star_rating: float) -> None:
        times, masks, lns = chord_events(notes)
        if times.size < 8:
            return
        s = sr_bin(star_rating)
        self.maps[s] += 1
        reds = sorted((tp for tp in timing_points if tp.uninherited and tp.beat_len > 0), key=lambda r: r.time)
        gaps = np.diff(times, prepend=times[0] - 10 ** 6)
        gnext = np.append(np.diff(times), 10 ** 6)
        gb = np.searchsorted(GAP_EDGES_MS, gaps, side="right")
        gnb = np.searchsorted(GAP_EDGES_MS, gnext, side="right")
        cls = np.array([div_class(float(t), reds) for t in times], dtype=np.int64)
        for mirror in (False, True):
            mk = MIRROR[masks] if mirror else masks
            prev = np.concatenate([[0], mk[:-1]])
            prev2 = np.concatenate([[0, 0], mk[:-2]])
            np.add.at(self.lane, (s, gb, prev, prev2, mk), 0.5)
        sizes = np.clip(POP[masks] - 1, 0, 3)
        np.add.at(self.chord, (s, np.array([metric_bin(c) for c in cls]), gb, sizes), 1.0)
        np.add.at(self.rhythm_hits, (s, cls), 1.0)
        if reds:
            span = float(times[-1] - times[0])
            beat = float(np.median([r.beat_len for r in reds]))
            self.rhythm_beats[s] += span / max(1.0, beat)
        # how fast real maps of this difficulty go at all (bursts of 1/8, 1/12, 1/16)
        if self.event_gap is None:
            self.event_gap = np.zeros((SR_BINS, REPEAT_BINS), np.float64)
        np.add.at(self.event_gap[s], np.clip(np.diff(times) // REPEAT_BIN_MS, 0, REPEAT_BINS - 1), 1.0)
        # how fast real maps of this difficulty repeat a lane (jacks), measured from the press / LN release
        by_lane: Dict[int, List[Note]] = {}
        for n in notes:
            by_lane.setdefault(int(n.col), []).append(n)
        for lane_notes in by_lane.values():
            lane_notes.sort(key=lambda n: n.start)
            for a, b in zip(lane_notes[:-1], lane_notes[1:]):
                g = b.start - (a.end if a.ln else a.start)
                if g >= 0:
                    self.repeat_gap[s, min(REPEAT_BINS - 1, g // REPEAT_BIN_MS)] += 1
        # how unevenly real maps of this difficulty distribute their notes (quiet parts vs. drops)
        win = 4000
        n_win = int((times[-1] - times[0]) // win)
        if n_win >= 4:
            cnt = np.bincount(((times - times[0]) // win).clip(0, n_win - 1), minlength=n_win).astype(np.float64)
            self.density_cv[s] += (float(cnt.std() / max(1e-6, cnt.mean())), 1.0)
        # how many chords real mappers start while a long note is held (dense holds vs. isolated holds)
        for n in notes:
            if n.ln and n.end > n.start:
                inside = int(np.searchsorted(times, n.end, side="left") - np.searchsorted(times, n.start, side="right"))
                self.ln_span[s, min(LN_SPAN_BINS - 1, inside)] += 1
        n_ln = POP[lns]
        np.add.at(self.ln, (s, gnb, 1), n_ln)
        np.add.at(self.ln, (s, gnb, 0), POP[masks] - n_ln)
        self._add_ln_style(notes, times, masks, lns, reds, cls, s)

    def _ensure_ln(self) -> None:
        if self.ln_ctx is None:
            self.ln_ctx = np.zeros((SR_BINS, N_LN_STYLE, N_BGAP, N_BGAP, 4, 3, 2), np.float64)
            self.ln_members = np.zeros((SR_BINS, N_LN_STYLE, 4, 5), np.float64)
            self.ln_len = np.zeros((SR_BINS, N_LN_STYLE, N_BGAP, N_LN_LEN), np.float64)
            self.ln_style_maps = np.zeros((SR_BINS, N_LN_STYLE), np.float64)
            self.ln_span_style = np.zeros((SR_BINS, N_LN_STYLE, LN_SPAN_BINS), np.float64)
        if self.ln_release is None:
            self.ln_release = np.zeros((SR_BINS, N_LN_STYLE, 3), np.float64)

    def _add_ln_style(self, notes, times, masks, lns, reds, cls, s: int) -> None:
        """Where real mappers of a given LN style put long notes (rhythm context), how many chord members and how long."""
        self._ensure_ln()
        share = float(np.mean([n.ln for n in notes])) if notes else 0.0
        st = ln_style_bin(share)
        self.ln_style_maps[s, st] += 1
        if reds:
            rt = np.array([r.time for r in reds])
            bl = np.array([r.beat_len for r in reds])
            beat = bl[np.clip(np.searchsorted(rt, times, side="right") - 1, 0, len(reds) - 1)]
        else:
            beat = np.full(times.size, 500.0)
        gn = np.append(np.diff(times), 10 ** 6) / beat
        gp = np.diff(times, prepend=times[0] - 10 ** 6) / beat
        gnb = np.searchsorted(BEAT_GAP_EDGES, gn, side="right")
        gpb = np.searchsorted(BEAT_GAP_EDGES, gp, side="right")
        size = np.clip(POP[masks] - 1, 0, 3)
        met = np.array([metric_bin(int(c)) for c in cls])
        has = (lns > 0).astype(np.int64)
        np.add.at(self.ln_ctx, (s, st, gnb, gpb, size, met, has), 1.0)
        np.add.at(self.ln_members, (s, st, size[has > 0], POP[lns[has > 0]]), 1.0)
        index = {int(t): i for i, t in enumerate(times)}
        for n in notes:
            if n.ln and n.end > n.start:
                i = index.get(int(n.start))
                if i is None:
                    continue
                self.ln_len[s, st, gnb[i], ln_len_class((n.end - n.start) / beat[i])] += 1
                inside = int(np.searchsorted(times, n.end, side="left") - np.searchsorted(times, n.start, side="right"))
                self.ln_span_style[s, st, min(LN_SPAN_BINS - 1, inside)] += 1
                j = int(np.searchsorted(times, n.end))
                near = min(abs(int(times[min(j, times.size - 1)]) - n.end), abs(int(times[max(j - 1, 0)]) - n.end))
                kind = 0 if near <= 5 else (1 if div_class(float(n.end), reds) != N_DIV - 1 else 2)
                self.ln_release[s, st, kind] += 1

    # --- long-note queries (backed off over neighbouring styles when a cell is thin) -----------
    def _style_weights(self, style: int, sr_bin_: Optional[int] = None) -> List[Tuple[int, float]]:
        """The requested LN style (or the nearest style that has maps in this star-rating bin) + its neighbours."""
        if sr_bin_ is not None and self.ln_style_maps is not None:
            filled = np.flatnonzero(self.ln_style_maps[sr_bin_] > 0)
            if filled.size and style not in filled:
                style = int(filled[np.argmin(np.abs(filled - style))])
        return [(style, 1.0)] + [(x, 0.15) for x in (style - 1, style + 1) if 0 <= x < N_LN_STYLE]

    def has_ln_model(self) -> bool:
        return self.ln_ctx is not None and self.ln_ctx.sum() > 0

    def typical_ln_style(self, sr: float) -> int:
        """Most common LN style of real maps at this star rating."""
        if not self.has_ln_model():
            return 1
        c = np.zeros(N_LN_STYLE)
        for b, wt in self._sr_weights(sr):
            c += wt * self.ln_style_maps[b]
        return int(np.argmax(c)) if c.sum() > 0 else 1

    def ln_chord_prob(self, sr: float, style: int, gap_next_beats: float, gap_prev_beats: float, size: int, metric: int) -> float:
        """P(this chord has a long note | rhythm context) for maps of this star rating and LN style."""
        gn, gp, k = beat_gap_bin(gap_next_beats), beat_gap_bin(gap_prev_beats), int(np.clip(size - 1, 0, 3))
        p = 0.0
        wsum = 0.0
        for b, wt in self._sr_weights(sr):
            for st, ws in self._style_weights(style, b):
                T = self.ln_ctx[b, st]
                prior = T.sum(axis=(0, 1, 2, 3))
                if prior.sum() == 0:
                    continue
                q = (prior[1] + 0.5) / (prior.sum() + 1.0)
                for c in (T[gn].sum(axis=(0, 1, 2)), T[gn, gp].sum(axis=(0, 1)), T[gn, gp, k].sum(axis=0), T[gn, gp, k, metric]):
                    q = (c[1] + 4.0 * q) / (c.sum() + 4.0)
                p += wt * ws * q
                wsum += wt * ws
        return float(p / wsum) if wsum > 0 else 0.0

    def ln_member_count(self, sr: float, style: int, size: int) -> np.ndarray:
        """P(number of long members = 1..size | the chord has a long note)."""
        c = np.zeros(5)
        for b, wt in self._sr_weights(sr):
            for st, ws in self._style_weights(style, b):
                c += wt * ws * self.ln_members[b, st, int(np.clip(size - 1, 0, 3))]
        c = c[1:size + 1] + 0.1
        return c / c.sum()

    def ln_length_dist(self, sr: float, style: int, gap_next_beats: float) -> np.ndarray:
        """P(LN length class) over LN_LEN_BEATS for this context."""
        g = beat_gap_bin(gap_next_beats)
        c = np.zeros(N_LN_LEN)
        for b, wt in self._sr_weights(sr):
            for st, ws in self._style_weights(style, b):
                T = self.ln_len[b, st]
                c += wt * ws * (T[g] + 0.2 * T.sum(axis=0) / max(1.0, T.sum()) * 5.0)
        c = c + 1e-3
        return c / c.sum()

    def ln_release_on_chord(self, sr: float, style: int) -> float:
        """Share of LN releases of real maps (this SR and style) that coincide with another chord's press."""
        if self.ln_release is None:
            return 0.0
        c = np.zeros(3)
        for b, wt in self._sr_weights(sr):
            for st, ws in self._style_weights(style, b):
                c += wt * ws * self.ln_release[b, st]
        return float((c[0] + 0.5) / (c.sum() + 1.0))

    def ln_span_limit_style(self, sr: float, style: int, quantile: float = 0.9) -> Optional[int]:
        h = np.zeros(LN_SPAN_BINS)
        for b, wt in self._sr_weights(sr):
            for st, ws in self._style_weights(style, b):
                hb = self.ln_span_style[b, st]
                if hb.sum() > 0:
                    h += wt * ws * hb / hb.sum()
        if h.sum() == 0:
            return None
        return int(np.searchsorted(np.cumsum(h) / h.sum(), quantile))

    def merge(self, other: "PatternModel") -> None:
        """Adds the counts of another model (models counted in parallel on separate processes)."""
        for k in ("lane", "chord", "rhythm_hits", "rhythm_beats", "ln", "maps") + OPTIONAL_ARRAYS:
            b = getattr(other, k, None)
            if b is None:
                continue
            a = getattr(self, k, None)
            setattr(self, k, b.copy() if a is None else a + b)
        for cache in ("_m_uni", "_m_gap", "_m_prev"):
            self.__dict__.pop(cache, None)

    # ------------------------------------------------------------------------------------------
    def _sr_weights(self, sr: float) -> List[Tuple[int, float]]:
        x = float(np.clip(sr - 0.5, 0.0, SR_BINS - 1.0))
        a = int(math.floor(x))
        w = x - a
        out = [(a, 1.0 - w)]
        if w > 0 and a + 1 < SR_BINS:
            out.append((a + 1, w))
        # empty bins borrow from the nearest filled one; thin bins (a handful of maps, e.g. 10* in most datasets) are
        # mixed with the nearest well-filled bin in proportion to their map count, so the statistics of a star rating
        # are not the style of one or two maps
        res: List[Tuple[int, float]] = []
        filled = np.flatnonzero(self.maps > 0)
        solid = np.flatnonzero(self.maps >= MIN_BIN_MAPS)
        for b, wt in out:
            if self.maps[b] == 0 and filled.size:
                b = int(filled[np.argmin(np.abs(filled - b))])
            n = float(self.maps[b])
            if 0 < n < MIN_BIN_MAPS and solid.size:
                nb = int(solid[np.argmin(np.abs(solid - b))])
                own = n / MIN_BIN_MAPS
                res.append((b, wt * own))
                res.append((nb, wt * (1.0 - own)))
            else:
                res.append((b, wt))
        return res

    def lane_logprob(self, sr: float, gap_ms: float, prev: int, prev2: int, size: int, alpha: float = 4.0) -> np.ndarray:
        """log P(mask | context) over all 16 masks, restricted to masks with ``size`` lanes (-inf elsewhere)."""
        g = gap_bin(gap_ms)
        valid = POP == size
        out = np.zeros(16)
        if not hasattr(self, "_m_uni"):
            self._m_uni = self.lane.sum(axis=(1, 2, 3)) + self.lane.sum(axis=(0, 1, 2, 3))[None] * 0.05  # [SR, 16]
            self._m_gap = self.lane.sum(axis=(2, 3))                                                   # [SR, GAP, 16]
            self._m_prev = self.lane.sum(axis=3)                                                       # [SR, GAP, 16, 16]
        for b, wt in self._sr_weights(sr):
            c_uni = self._m_uni[b]
            c_gap = self._m_gap[b, g]
            c_prev = self._m_prev[b, g, prev]
            c_full = self.lane[b, g, prev, prev2]
            p = (c_uni + 1e-3) * valid
            p = p / max(1e-12, p.sum())
            for c in (c_gap, c_prev, c_full):
                c = c * valid
                n = c.sum()
                p = (c + alpha * p) / (n + alpha)
            out += wt * p
        out = out * valid
        with np.errstate(divide="ignore"):
            return np.where(valid, np.log(np.maximum(out / max(1e-12, out.sum()), 1e-9)), -np.inf)

    def chord_distribution(self, sr: float, metric: int = 1, gap_ms: Optional[float] = None) -> np.ndarray:
        """P(chord size 1..4 | star rating [, metric position, gap])."""
        p = np.zeros(4)
        for b, wt in self._sr_weights(sr):
            c = self.chord[b, metric] if metric is not None else self.chord[b].sum(axis=0)
            c = c[gap_bin(gap_ms)] * 1.0 + 0.2 * c.sum(axis=0) if gap_ms is not None else c.sum(axis=0)
            c = c + 0.5
            p += wt * c / c.sum()
        return p / p.sum()

    def chord_marginal(self, sr: float) -> np.ndarray:
        p = np.zeros(4)
        for b, wt in self._sr_weights(sr):
            c = self.chord[b].sum(axis=(0, 1)) + 0.5
            p += wt * c / c.sum()
        return p / p.sum()

    def tick_usage(self, sr: float) -> np.ndarray:
        """P(a tick of each divisor class carries a note) at this star rating."""
        u = np.zeros(N_DIV)
        for b, wt in self._sr_weights(sr):
            beats = max(1.0, self.rhythm_beats[b])
            u += wt * np.clip(self.rhythm_hits[b] / (beats * DIV_TICKS_PER_BEAT), 1e-4, 0.999)
        return u

    def repeat_floor(self, sr: float, quantile: float = 0.01, after_ln: bool = False) -> Optional[float]:
        """Lane-repeat gap (ms) that only ``quantile`` of the repeats in real maps of this difficulty are faster than."""
        if self.repeat_gap is None or self.repeat_gap.sum() == 0:
            return None
        h = np.zeros(REPEAT_BINS)
        for b, wt in self._sr_weights(sr):
            hb = self.repeat_gap[b]
            if hb.sum() > 0:
                h += wt * hb / hb.sum()
        if h.sum() == 0:
            return None
        c = np.cumsum(h) / h.sum()
        return float(np.searchsorted(c, quantile) * REPEAT_BIN_MS)

    def _hist(self, table: Optional[np.ndarray], sr: float) -> Optional[np.ndarray]:
        if table is None or table.sum() == 0:
            return None
        h = np.zeros(table.shape[1])
        for b, wt in self._sr_weights(sr):
            if table[b].sum() > 0:
                h += wt * table[b] / table[b].sum()
        return h if h.sum() > 0 else None

    def _rhythm_gaps(self, sr: float) -> Optional[np.ndarray]:
        """Chord-to-chord gap histogram without the < 30 ms gaps: those are chords whose notes are a few ms apart
        (or grace notes), counted as two events by the chord grouping, not a rhythm anyone plays."""
        h = self._hist(self.event_gap, sr)
        if h is None:
            return None
        h = h.copy()
        h[: STAGGER_MS // REPEAT_BIN_MS] = 0.0
        return h if h.sum() > 0 else None

    def gap_share_below(self, sr: float, gap_ms: float) -> Optional[float]:
        """Share of chord-to-chord gaps (rhythm, >= 30 ms) in real maps of this difficulty shorter than ``gap_ms``."""
        h = self._rhythm_gaps(sr)
        if h is None:
            return None
        return float(h[: max(0, int(gap_ms) // REPEAT_BIN_MS)].sum() / h.sum())

    def event_gap_floor(self, sr: float, quantile: float = 0.005) -> Optional[float]:
        """Chord-to-chord gap (ms, rhythm only) that only ``quantile`` of real gaps at this difficulty are shorter than."""
        h = self._rhythm_gaps(sr)
        if h is None:
            return None
        return float(np.searchsorted(np.cumsum(h) / h.sum(), quantile) * REPEAT_BIN_MS)

    def repeat_cdf(self, sr: float) -> Optional[np.ndarray]:
        """[REPEAT_BINS] share of lane repeats in real maps of this difficulty that are faster than each 5 ms bin."""
        if self.repeat_gap is None or self.repeat_gap.sum() == 0:
            return None
        h = np.zeros(REPEAT_BINS)
        for b, wt in self._sr_weights(sr):
            hb = self.repeat_gap[b]
            if hb.sum() > 0:
                h += wt * hb / hb.sum()
        return np.cumsum(h) / max(1e-12, h.sum())

    def ln_span_limit(self, sr: float, quantile: float = 0.9) -> Optional[int]:
        """Number of chords that ``quantile`` of the long notes of real maps of this difficulty hold over at most."""
        if self.ln_span is None or self.ln_span.sum() == 0:
            return None
        h = np.zeros(LN_SPAN_BINS)
        for b, wt in self._sr_weights(sr):
            hb = self.ln_span[b]
            if hb.sum() > 0:
                h += wt * hb / hb.sum()
        if h.sum() == 0:
            return None
        return int(np.searchsorted(np.cumsum(h) / h.sum(), quantile))

    def density_variation(self, sr: float) -> Optional[float]:
        """Typical std/mean of the note count in 4 s windows for real maps of this difficulty."""
        if self.density_cv is None or self.density_cv[:, 1].sum() == 0:
            return None
        tot = wsum = 0.0
        for b, wt in self._sr_weights(sr):
            if self.density_cv[b, 1] > 0:
                tot += wt * self.density_cv[b, 0] / self.density_cv[b, 1]
                wsum += wt
        return tot / wsum if wsum > 0 else None

    def add_sections(self, notes: Sequence[Note], sections, star_rating: float) -> None:
        """Audio intensity of every section (composer analysis) vs the note density real mappers gave it."""
        if self.intensity_fit is None:
            self.intensity_fit = np.zeros((SR_BINS, 3), np.float64)
        secs = [x for x in sections if x.intensity > 0 and x.dur_s >= 2.0]
        if len(secs) < 3 or not notes:
            return
        starts = np.array(sorted(n.start for n in notes), dtype=np.int64)
        dens = np.array([(np.searchsorted(starts, x.end_ms) - np.searchsorted(starts, x.start_ms)) / x.dur_s for x in secs])
        base = np.array([0.3 + 0.7 * x.intensity for x in secs])
        w = np.array([x.dur_s for x in secs])
        ok = dens > 0
        if ok.sum() < 3:
            return
        y = np.log(dens[ok]) - np.average(np.log(dens[ok]), weights=w[ok])
        x = np.log(base[ok]) - np.average(np.log(base[ok]), weights=w[ok])
        b = sr_bin(star_rating)
        self.intensity_fit[b] += (float((w[ok] * x * y).sum()), float((w[ok] * x * x).sum()), 1.0)

    def intensity_exponent(self, sr: float, min_maps: int = 3) -> Optional[float]:
        """How strongly real maps of this difficulty follow the audio intensity: section density ~ (0.3 + 0.7 i)^e."""
        if self.intensity_fit is None:
            return None
        xy = xx = n = 0.0
        for b, wt in self._sr_weights(sr):
            xy += wt * self.intensity_fit[b, 0]
            xx += wt * self.intensity_fit[b, 1]
            n += wt * self.intensity_fit[b, 2]
        if n < min_maps * 0.5 or xx <= 1e-9:
            return None
        return float(np.clip(xy / xx, 0.0, 6.0))

    def max_chord(self, sr: float, min_share: float = 0.003) -> int:
        """Largest chord size that real maps of this difficulty actually use (share >= min_share)."""
        p = self.chord_marginal(sr)
        ok = [k + 1 for k in range(4) if p[k] >= min_share]
        return max(ok) if ok else 1

    def ln_share(self, sr: float) -> float:
        tot = 0.0
        for b, wt in self._sr_weights(sr):
            c = self.ln[b].sum(axis=0)
            tot += wt * (c[1] + 0.5) / (c.sum() + 1.0)
        return float(tot)

    def ln_prob(self, sr: float, gap_next_ms: float) -> float:
        tot = 0.0
        g = gap_bin(gap_next_ms)
        for b, wt in self._sr_weights(sr):
            c = self.ln[b, g]
            tot += wt * (c[1] + 0.5) / (c.sum() + 1.0)
        return float(tot)

    # ------------------------------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        extra = {k: getattr(self, k) for k in OPTIONAL_ARRAYS if getattr(self, k) is not None}
        np.savez_compressed(path, lane=self.lane, chord=self.chord, rhythm_hits=self.rhythm_hits,
                            rhythm_beats=self.rhythm_beats, ln=self.ln, maps=self.maps,
                            version=np.array([self.version]), **extra)

    @classmethod
    def load(cls, path: str | Path) -> "PatternModel":
        d = np.load(path, allow_pickle=False)
        v = str(d["version"][0])
        if v != PATTERN_VERSION:
            raise ValueError(f"pattern model '{path}' has version {v}, code expects {PATTERN_VERSION}: re-run learn_patterns.py")
        extra = {k: d[k] for k in OPTIONAL_ARRAYS if k in d}
        return cls(lane=d["lane"], chord=d["chord"], rhythm_hits=d["rhythm_hits"], rhythm_beats=d["rhythm_beats"],
                   ln=d["ln"], maps=d["maps"], version=v, **extra)

    def summary(self) -> str:
        rows = [f"pattern model: {int(self.maps.sum())} maps, {int(self.lane.sum())} chord transitions"]
        for b in range(SR_BINS):
            if self.maps[b] == 0:
                continue
            ch = self.chord[b].sum(axis=(0, 1))
            ch = ch / max(1.0, ch.sum())
            u = self.tick_usage(b + 0.5)
            L = self.lane[b]
            tot = max(1.0, L.sum())
            jack = sum(L[:, p, :, p].sum() for p in range(1, 16)) / tot
            fast_jack = sum(L[:4, p, :, p].sum() for p in range(1, 16)) / tot
            rows.append(f"  {b}-{b + 1}*: {int(self.maps[b]):4d} maps | chords 1/2/3/4 = {ch[0]:.2f}/{ch[1]:.2f}/{ch[2]:.2f}/{ch[3]:.2f} "
                        f"| tick usage 1/1 {u[0]:.2f} 1/2 {u[1]:.2f} 1/4 {u[2]:.2f} 1/3 {u[3]:.2f} 1/8 {u[4]:.2f} "
                        f"| fastest 0.5% of gaps < {self.event_gap_floor(b + 0.5) or 0:.0f} ms "
                        f"| repeats {100 * jack:.1f}% (<125ms {100 * fast_jack:.1f}%) | lane repeat 1%/5% faster than "
                        f"{self.repeat_floor(b + 0.5, 0.01) or 0:.0f}/{self.repeat_floor(b + 0.5, 0.05) or 0:.0f} ms "
                        f"| LN {100 * self.ln_share(b + 0.5):.0f}%, 90% of holds span <= {self.ln_span_limit(b + 0.5) or 0} chords"
                        f" | density variation {self.density_variation(b + 0.5) or 0:.2f}"
                        + (f", follows audio intensity ^{e:.2f}" if (e := self.intensity_exponent(b + 0.5)) is not None else ""))
        return "\n".join(rows)


OPTIONAL_ARRAYS = ("repeat_gap", "ln_span", "density_cv", "intensity_fit", "event_gap",
                   "ln_ctx", "ln_members", "ln_len", "ln_style_maps", "ln_span_style", "ln_release")


def default_pattern_path() -> Path:
    return Path("checkpoints") / "hybrid" / "patterns.npz"


def load_default(path: Optional[str] = None, log=print) -> Optional[PatternModel]:
    p = Path(path) if path else default_pattern_path()
    if not p.exists():
        if path:
            raise FileNotFoundError(f"pattern model '{p}' not found (python learn_patterns.py --data dataset_hybrid)")
        return None
    m = PatternModel.load(p)
    log(f"[patterns] {p}: {int(m.maps.sum())} maps")
    return m


# ----------------------------------------------------------------------------------------------
# density graph (the stepped "intensity" bars of the osu! editor timeline)
# ----------------------------------------------------------------------------------------------

def density_profile(notes: Sequence[Note], duration_ms: int, window_ms: int = 2000) -> np.ndarray:
    """Notes per second in consecutive windows over the whole track."""
    n_win = max(1, int(math.ceil(duration_ms / window_ms)))
    starts = np.array([n.start for n in notes], dtype=np.int64)
    idx = np.clip(starts // window_ms, 0, n_win - 1)
    return np.bincount(idx, minlength=n_win).astype(np.float64) / (window_ms / 1000.0)


def density_correlation(a: Sequence[Note], b: Sequence[Note], duration_ms: int, window_ms: int = 2000) -> float:
    """Pearson correlation of two density graphs (1 = notes rise and fall at the same places)."""
    pa, pb = density_profile(a, duration_ms, window_ms), density_profile(b, duration_ms, window_ms)
    if pa.std() < 1e-9 or pb.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(pa, pb)[0, 1])


def density_graph(notes: Sequence[Note], duration_ms: int, width: int = 72, scale: Optional[float] = None) -> str:
    """One-line bar graph of the note density, like the editor timeline (scale = notes/s of a full bar)."""
    if duration_ms <= 0:
        return ""
    prof = density_profile(notes, duration_ms, max(250, int(math.ceil(duration_ms / width))))
    top = float(scale or max(1e-6, prof.max()))
    bars = " ▁▂▃▄▅▆▇█"
    return "".join(bars[int(round(np.clip(v / top, 0.0, 1.0) * (len(bars) - 1)))] for v in prof)


# ----------------------------------------------------------------------------------------------
# long-note quality: how the holds of a map look, compared distribution by distribution
# ----------------------------------------------------------------------------------------------

LN_SPAN_GROUPS = ((0, 0), (1, 1), (2, 2), (3, 3), (4, 7), (8, 10 ** 9))


def ln_quality(notes: Sequence[Note], timing_points: Sequence[RawTimingPoint]) -> Dict[str, np.ndarray]:
    """Distributions that describe the long notes of a map (shares, each sums to 1 when there are LNs):

    length       length in beats over LN_LEN_BEATS (1/8 .. 8 beats)
    release      release on another chord's press / on a free grid tick / off the grid
    after        pause from the release to the next press on the same lane, beat bins (BEAT_GAP_EDGES)
    span         chords pressed while the LN is held: 0, 1, 2, 3, 4-7, 8+
    hand         presses during the hold on the same hand / on the other hand (mean per LN)
    """
    reds = sorted((tp for tp in timing_points if tp.uninherited and tp.beat_len > 0), key=lambda r: r.time)
    times, masks, _ = chord_events(notes)
    lns = [n for n in notes if n.ln and n.end > n.start]
    out: Dict[str, np.ndarray] = {"share": np.array([len(lns) / max(1, len(notes))])}
    if not lns or times.size == 0:
        return out
    rt = np.array([r.time for r in reds]) if reds else np.array([0.0])
    bl = np.array([r.beat_len for r in reds]) if reds else np.array([500.0])

    def beat_at(t: float) -> float:
        return float(bl[max(0, int(np.searchsorted(rt, t, side="right")) - 1)])

    by_lane: Dict[int, List[int]] = {}
    for n in notes:
        by_lane.setdefault(n.col, []).append(n.start)
    for v in by_lane.values():
        v.sort()
    length = np.zeros(N_LN_LEN)
    release = np.zeros(3)
    after = np.zeros(N_BGAP + 1)  # last bin: no further note on the lane
    span = np.zeros(len(LN_SPAN_GROUPS))
    same = other = 0.0
    for n in lns:
        b = beat_at(n.start)
        length[ln_len_class((n.end - n.start) / b)] += 1
        j = int(np.searchsorted(times, n.end))
        near = min(abs(int(times[min(j, times.size - 1)]) - n.end), abs(int(times[max(j - 1, 0)]) - n.end))
        release[0 if near <= 5 else (1 if div_class(float(n.end), reds) != N_DIV - 1 else 2)] += 1
        lane = by_lane[n.col]
        k = int(np.searchsorted(lane, n.end, side="right"))
        after[beat_gap_bin((lane[k] - n.end) / b) if k < len(lane) else N_BGAP] += 1
        a0, a1 = int(np.searchsorted(times, n.start, side="right")), int(np.searchsorted(times, n.end, side="left"))
        inside = a1 - a0
        span[[g for g, (lo, hi) in enumerate(LN_SPAN_GROUPS) if lo <= inside <= hi][0]] += 1
        hand = (0b0011 if n.col < 2 else 0b1100) & ~(1 << n.col)
        for m in masks[a0:a1]:
            same += POP[m & hand]
            other += POP[m & ~(0b0011 if n.col < 2 else 0b1100) & 15]
    out.update(length=length / length.sum(), release=release / release.sum(), after=after / after.sum(),
               span=span / span.sum(), hand=np.array([same / len(lns), other / len(lns)]))
    return out


def ln_similarity(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> Dict[str, float]:
    """1 - Jensen-Shannon divergence (base 2) of every LN distribution: 1 = identical, 0 = nothing in common."""
    def js(p: np.ndarray, q: np.ndarray) -> float:
        p, q = p + 1e-12, q + 1e-12
        p, q = p / p.sum(), q / q.sum()
        m = 0.5 * (p + q)
        return float(0.5 * (p * np.log2(p / m)).sum() + 0.5 * (q * np.log2(q / m)).sum())
    out = {}
    for k in ("length", "release", "after", "span"):
        if k in a and k in b:
            out[k] = 1.0 - js(a[k], b[k])
    return out


def format_ln_quality(q: Dict[str, np.ndarray]) -> str:
    if "length" not in q:
        return f"LN {100 * float(q['share'][0]):.0f}% (no holds)"
    L = q["length"]
    short = L[LN_LEN_BEATS <= 0.375].sum()
    mid = L[(LN_LEN_BEATS > 0.375) & (LN_LEN_BEATS <= 1.0)].sum()
    long_ = L[LN_LEN_BEATS > 1.0].sum()
    r, s = q["release"], q["span"]
    return (f"LN {100 * float(q['share'][0]):.0f}% | length <=3/8 beat {100 * short:.0f}%, 1/2-1 {100 * mid:.0f}%, >1 beat {100 * long_:.0f}% "
            f"| release on a press {100 * r[0]:.0f}%, free tick {100 * r[1]:.0f}%, off grid {100 * r[2]:.0f}% "
            f"| holds over 0/1/2/3/4-7/8+ chords {'/'.join(f'{100 * x:.0f}' for x in s)}% "
            f"| presses during a hold: same hand {q['hand'][0]:.1f}, other hand {q['hand'][1]:.1f}")
